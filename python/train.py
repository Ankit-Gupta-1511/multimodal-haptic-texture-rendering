from __future__ import annotations

import os
import sys
import math
import time
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from .dataset_textures import VisuoTactileDataset
from .model_texturenet import TextureNet
from .losses import time_mse, stft_loss, mel_loss_audio, envelope_sync_loss
from .metrics import lsd_torch, rmse_torch
from .utils import (
    seed_all,
    device_report,
    get_tb_writer,
    memory_info,
    ensure_dir,
    amp_autocast,
    to_device,
    numpyify,
    WarmupCosine,
    save_json,
)


def load_params(path: Path) -> Dict[str, Any]:
    import yaml
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    # fallback to python/params.yaml if running root script
    alt = Path("python/params.yaml")
    if alt.exists():
        with open(alt, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(f"params.yaml not found at {path} or {alt}")


def build_loaders(data_path: str, mode: str, params: Dict[str, Any], workers: int):
    use_prev = mode == "low_delay_vibro" or (mode == "multitask_av" and params.get("model", {}).get("use_prev", False))
    use_audio = mode == "multitask_av"
    common = dict(
        parquet_or_h5_path=data_path,
        fs_vib=int(params.get("fs_vib", 1000)),
        fs_aud=int(params.get("fs_aud", 8000)),
        win_ms=int(params.get("win_ms", 100)),
        hop_ratio=float(params.get("hop_ratio", 0.5)),
        patch=int(params.get("patch", 96)),
        band_vib=tuple(params.get("band_vib", [20, 400])),
        use_audio=use_audio,
        use_prev=use_prev,
    )
    ds_train = VisuoTactileDataset(split="train", **common)
    ds_val = VisuoTactileDataset(split="val", **common)
    dl_train = DataLoader(ds_train, batch_size=int(params["train"]["batch"]), shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=int(params["train"]["batch"]), shuffle=False, num_workers=workers, pin_memory=True)
    return ds_train, ds_val, dl_train, dl_val


def build_model(mode: str, params: Dict[str, Any]) -> TextureNet:
    mcfg = params.get("model", {})
    net = TextureNet(
        mode=mode,
        latent=int(mcfg.get("latent", 128)),
        tcn_blocks=int(mcfg.get("tcn_blocks", 4)),
        tcn_growth=int(mcfg.get("tcn_growth", 64)),
        use_prev=bool(mcfg.get("use_prev", False) or mode == "low_delay_vibro"),
    )
    return net


def compute_losses(mode: str, batch: Dict[str, torch.Tensor], out, params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    lw = params.get("loss", {})
    w_stft_vib = float(lw.get("w_stft_vib", 0.5))
    w_time_aud = float(lw.get("w_time_aud", 1.0))
    w_mel_aud = float(lw.get("w_mel_aud", 0.5))
    w_sync = float(lw.get("w_sync", 0.2))

    losses: Dict[str, torch.Tensor] = {}
    if mode in {"baseline_vibro", "low_delay_vibro"}:
        vib_hat = out
        vib = batch["vib"]
        losses["mse_vib"] = time_mse(vib_hat, vib)
        losses["stft_vib"] = stft_loss(vib_hat, vib)
        losses["total"] = losses["mse_vib"] + w_stft_vib * losses["stft_vib"]
    else:
        vib_hat, aud_hat = out
        vib = batch["vib"]
        aud = batch["audio"]
        losses["mse_vib"] = time_mse(vib_hat, vib)
        losses["stft_vib"] = stft_loss(vib_hat, vib)
        losses["mse_aud"] = time_mse(aud_hat, aud)
        losses["mel_aud"] = mel_loss_audio(aud_hat, aud)
        losses["sync"] = envelope_sync_loss(vib_hat, aud_hat)
        losses["total"] = (
            losses["mse_vib"] + w_stft_vib * losses["stft_vib"]
            + w_time_aud * losses["mse_aud"] + w_mel_aud * losses["mel_aud"] + w_sync * losses["sync"]
        )
    return losses


def eval_metrics(mode: str, batch: Dict[str, torch.Tensor], out) -> Dict[str, float]:
    mets: Dict[str, float] = {}
    vib_hat = out[0] if isinstance(out, (tuple, list)) else out
    vib = batch["vib"]
    mets["rmse_vib"] = float(rmse_torch(vib_hat, vib).detach().cpu())
    mets["lsd_vib"] = float(lsd_torch(vib_hat, vib).detach().cpu())
    if mode == "multitask_av":
        aud_hat = out[1]
        aud = batch["audio"]
        mets["rmse_aud"] = float(rmse_torch(aud_hat, aud).detach().cpu())
        mets["lsd_aud"] = float(lsd_torch(aud_hat, aud, n_fft=512, hop=128).detach().cpu())
    return mets


def train_loop(args) -> int:
    console = Console()
    params = load_params(Path(args.params))
    mode = args.mode or params.get("mode", "baseline_vibro")
    seed_all(42)
    torch.backends.cudnn.deterministic = True

    # data
    ds_train, ds_val, dl_train, dl_val = build_loaders(args.data, mode, params, workers=int(params["train"]["workers"]))

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_model(mode, params).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(params["train"]["lr"]), weight_decay=float(params["train"]["weight_decay"]))
    for g in opt.param_groups:
        g.setdefault("initial_lr", g["lr"])  # scheduler support
    total_steps = int(params["train"]["epochs"]) * max(1, len(dl_train))
    sched = WarmupCosine(opt, warmup_steps=max(10, total_steps // 50), total_steps=total_steps, min_lr_scale=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(params["train"]["amp"]) and torch.cuda.is_available())

    # logging
    tb_dir = Path(params["logs"]["tb_dir"]).joinpath(time.strftime("%Y%m%d-%H%M%S"))
    writer = get_tb_writer(tb_dir)
    debug_dir = Path(params["logs"]["debug_dir"]).joinpath("train")
    ensure_dir(debug_dir)

    console.log(f"Start training: {device_report()}")
    table = Table("key", "value")
    table.add_row("mode", mode)
    table.add_row("train_size", str(len(ds_train)))
    table.add_row("val_size", str(len(ds_val)))
    table.add_row("tb_dir", str(tb_dir))
    console.print(table)

    best_val = math.inf
    best_path = Path("assets/ckpt/best.pt")
    ensure_dir(best_path.parent)

    overfit_batches = int(params["train"].get("overfit_batches", 0))
    grad_clip = float(params["train"]["grad_clip"]) if params["train"].get("grad_clip", 0) else 0.0
    early_stop_patience = int(params["train"]["early_stop_patience"]) or 10
    epochs = int(args.epochs or params["train"]["epochs"])

    step = 0
    no_improve = 0

    with Progress() as progress:
        task = progress.add_task("train", total=epochs * max(1, len(dl_train)))
        for ep in range(epochs):
            net.train()
            for i, batch in enumerate(dl_train):
                if overfit_batches and i >= overfit_batches:
                    break
                batch = to_device(batch, device)
                opt.zero_grad(set_to_none=True)
                with amp_autocast(bool(params["train"]["amp"])):
                    if mode == "low_delay_vibro":
                        out = net(batch["patch"], batch["state"], batch.get("prev_vib"))
                    else:
                        out = net(batch["patch"], batch["state"], batch.get("prev_vib"))
                    losses = compute_losses(mode, batch, out, params)
                    loss = losses["total"]
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
                sched.step()

                # logging
                if step % 10 == 0:
                    for k, v in losses.items():
                        writer.add_scalar(f"train/{k}", float(v.detach().cpu()), step)
                    mi = memory_info()
                    for k, v in mi.items():
                        writer.add_scalar(f"sys/{k}", v, step)

                step += 1
                progress.advance(task)

            # validation
            net.eval()
            val_losses = []
            val_mets = []
            with torch.no_grad():
                for j, vb in enumerate(dl_val):
                    vb = to_device(vb, device)
                    if mode == "low_delay_vibro":
                        vout = net(vb["patch"], vb["state"], vb.get("prev_vib"))
                    else:
                        vout = net(vb["patch"], vb["state"], vb.get("prev_vib"))
                    vlosses = compute_losses(mode, vb, vout, params)
                    mets = eval_metrics(mode, vb, vout)
                    val_losses.append(float(vlosses["total"].detach().cpu()))
                    val_mets.append(mets["lsd_vib"])  # early stop on spectral loss

            val_loss = float(np.mean(val_losses)) if len(val_losses) else float("inf")
            val_spec = float(np.mean(val_mets)) if len(val_mets) else float("inf")
            writer.add_scalar("val/loss", val_loss, ep)
            writer.add_scalar("val/lsd_vib", val_spec, ep)

            console.log({"epoch": ep, "val_loss": val_loss, "val_lsd_vib": val_spec})

            improved = val_spec < best_val
            if improved:
                best_val = val_spec
                no_improve = 0
                ckpt = {
                    "model": net.state_dict(),
                    "mode": mode,
                    "params_path": args.params,
                    "val_spec": best_val,
                }
                torch.save(ckpt, best_path)
                console.log(f"Saved best checkpoint to {best_path}")
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    console.log("Early stopping: no improvement")
                    break

    writer.close()
    console.log(f"Best val spectral loss: {best_val}")
    return 0


def main():
    import argparse
    p = argparse.ArgumentParser(description="Train TextureNet")
    p.add_argument("--data", type=str, required=True, help="Path to parquet/h5 dataset table")
    p.add_argument("--mode", type=str, default=None, choices=["baseline_vibro", "low_delay_vibro", "multitask_av"])
    p.add_argument("--params", type=str, default="params.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--sanity_check", type=int, default=0, help="If >0, run short train/val to emit artifacts")
    args = p.parse_args()
    sys.exit(train_loop(args))


if __name__ == "__main__":
    main()

