from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort

from .model_texturenet import TextureNet
from .train import load_params
from .utils import ensure_dir, save_json


def _load_weights_robust(net: torch.nn.Module, ckpt_path: Path, device: torch.device):
    raw = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict) and "model" in raw:
        sd = raw["model"]
    else:
        sd = raw
    model_sd = net.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
    net.load_state_dict(filtered, strict=False)


def export_model(ckpt_path: Path, mode: str, params_path: Path, out_path: Path, opset: int = 17, check_runtime: bool = True) -> int:
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    params = load_params(params_path)
    mcfg = params.get("model", {})
    use_prev = bool(mcfg.get("use_prev", False) or mode == "low_delay_vibro")
    fs_aud = int(params.get("fs_aud", 8000))
    net = TextureNet(
        mode=mode,
        latent=int(mcfg.get("latent", 128)),
        tcn_blocks=int(mcfg.get("tcn_blocks", 4)),
        tcn_growth=int(mcfg.get("tcn_growth", 64)),
        use_prev=use_prev,
        fs_aud=fs_aud,
    ).eval()

    _load_weights_robust(net, ckpt_path, device=torch.device("cpu"))

    # Dummy inputs
    B = 4
    patch = torch.randn(B, 1, 96, 96)
    state_in = torch.rand(B, 2)
    inputs = [patch, state_in]
    input_names = ["patch", "state"]
    dynamic_axes = {"patch": {0: "B"}, "state": {0: "B"}}
    if use_prev:
        prev_vib = torch.randn(B, 100)
        inputs.append(prev_vib)
        input_names.append("prev_vib")
        dynamic_axes["prev_vib"] = {0: "B"}

    if mode == "multitask_av":
        output_names = ["vib", "audio"]
    else:
        output_names = ["vib"]

    ensure_dir(out_path.parent)
    torch.onnx.export(
        net,
        tuple(inputs),
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        dynamic_axes=dynamic_axes,
    )
    meta = {
        "mode": mode,
        "use_prev": use_prev,
        "fs_vib": int(params.get("fs_vib", 1000)),
        "fs_aud": fs_aud,
        "inputs": input_names,
        "outputs": output_names,
        "opset": opset,
    }
    save_json(out_path.with_suffix(".json"), meta)

    if check_runtime:
        # Validate with onnxruntime
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])  # portable
        feeds = {"patch": patch.numpy(), "state": state_in.numpy()}
        if use_prev:
            feeds["prev_vib"] = inputs[-1].numpy()
        ort_outs = sess.run(None, feeds)

        with torch.no_grad():
            pt_out = net(*inputs)
            if mode == "multitask_av":
                pt_out = [pt_out[0].numpy(), pt_out[1].numpy()]
            else:
                pt_out = [pt_out.numpy()]
        max_diffs = [float(np.max(np.abs(a - b))) for a, b in zip(ort_outs, pt_out)]
        print({"onnx_outputs": [o.shape for o in ort_outs], "max_abs_diff": max_diffs})
        assert all(d <= 1e-4 for d in max_diffs), f"ONNX validation failed: diffs={max_diffs}"
    return 0


def main():
    import argparse
    p = argparse.ArgumentParser(description="Export TextureNet to ONNX")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--mode", type=str, required=True, choices=["baseline_vibro", "low_delay_vibro", "multitask_av"])
    p.add_argument("--params", type=str, default="params.yaml")
    p.add_argument("--out", type=str, default="assets/model.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no_check", action="store_true", help="Skip ONNX Runtime parity check")
    args = p.parse_args()
    sys.exit(export_model(Path(args.ckpt), args.mode, Path(args.params), Path(args.out), opset=int(args.opset), check_runtime=not args.no_check))


if __name__ == "__main__":
    main()
