"""VisuoTactileDataset and a CLI sanity inspection tool.

Expected preprocessed Parquet/HDF5 with columns similar to:
  - texture_id (str/int), material_id (str/int)
  - image_path (str), u (float in [0,1]), v (float in [0,1]) [optional]
  - accel or accel_path (1D vibration waveform at native fs)
  - audio or audio_path (optional, 1D waveform at native fs)
  - speed (1D tangential speed over time) and force (1D normal force over time)
  - t (1D time axis, seconds) or implicit sampling rate columns

Resampling, band-pass, windowing to target 100 ms windows happens here.
"""

from __future__ import annotations

import os
import sys
import math
import json
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

from .utils import ensure_dir, safe_mean_std_from_yaml, plot_psd, plot_hist, make_grid_save, rms


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return pd.read_hdf(path)
    raise ValueError(f"Unsupported dataset table format: {path}")


def _load_array_maybe(path_or_arr, key: str) -> np.ndarray:
    """Load array from possible sources: inline numpy-like, .npy, .wav, or .json lists."""
    if path_or_arr is None:
        raise FileNotFoundError(f"Missing array for {key}")
    if isinstance(path_or_arr, (list, tuple)):
        return np.asarray(path_or_arr, dtype=np.float32)
    if isinstance(path_or_arr, np.ndarray):
        return path_or_arr.astype(np.float32)
    s = str(path_or_arr)
    p = Path(s)
    if p.suffix.lower() == ".npy" and p.exists():
        return np.load(p).astype(np.float32)
    if p.suffix.lower() in {".wav", ".wave"} and p.exists():
        import soundfile as sf
        arr, _sr = sf.read(p)
        return np.asarray(arr, dtype=np.float32)
    if p.suffix.lower() in {".json"} and p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return np.asarray(json.load(f), dtype=np.float32)
    # fallback: if it looks like a Python list in string
    try:
        if s.startswith("[") and s.endswith("]"):
            return np.asarray(json.loads(s), dtype=np.float32)
    except Exception:
        pass
    raise FileNotFoundError(f"Cannot load array for {key} from {path_or_arr}")


def _butter_bandpass_sos(lowcut: float, highcut: float, fs: float, order: int = 4):
    from scipy.signal import butter
    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    return sos


def _sosfiltfilt(sos, x: np.ndarray) -> np.ndarray:
    from scipy.signal import sosfiltfilt
    return sosfiltfilt(sos, x).astype(np.float32)


def _resample_poly(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g
    y = resample_poly(x, up, down)
    return y.astype(np.float32)


def _rms_normalize(x: np.ndarray, target_rms: float) -> np.ndarray:
    cur = np.sqrt(np.mean(np.square(x.astype(np.float64)))) + 1e-12
    scale = float(target_rms / cur)
    return (x * scale).astype(np.float32)


def _window_indices(n: int, win: int, hop: int) -> List[Tuple[int, int]]:
    idx = []
    s = 0
    while s + win <= n:
        idx.append((s, s + win))
        s += hop
    return idx


def _to_uv_or_deterministic(u: Optional[float], v: Optional[float], seed_key: str) -> Tuple[float, float]:
    if u is not None and v is not None and 0 <= u <= 1 and 0 <= v <= 1:
        return float(u), float(v)
    # deterministic pseudo-UV from hash
    h = abs(hash(seed_key))
    rng = np.random.RandomState(h % (2**32))
    return float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8))


def _crop_uv_patch_gray(img_path: Path, u: float, v: float, patch: int) -> np.ndarray:
    with Image.open(img_path) as im:
        im = im.convert("L")  # grayscale
        w, h = im.size
        cx = int(u * (w - 1))
        cy = int(v * (h - 1))
        half = patch // 2
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(w, x0 + patch)
        y1 = min(h, y0 + patch)
        im_c = im.crop((x0, y0, x1, y1))
        if im_c.size != (patch, patch):
            im_c = im_c.resize((patch, patch), Image.BICUBIC)
        arr = np.asarray(im_c, dtype=np.float32) / 255.0
        return arr


@dataclass
class DatasetConfig:
    fs_vib: int = 1000
    fs_aud: int = 8000
    win_ms: int = 100
    hop_ratio: float = 0.5
    band_vib: Tuple[float, float] = (20.0, 400.0)
    patch: int = 96
    use_audio: bool = False
    use_prev: bool = False
    norm_yaml: str = "assets/norm.yaml"
    deterministic_patches: bool = True


class VisuoTactileDataset(Dataset):
    def __init__(
        self,
        parquet_or_h5_path: str,
        split: str,
        fs_vib: int = 1000,
        fs_aud: int = 8000,
        win_ms: int = 100,
        hop_ratio: float = 0.5,
        patch: int = 96,
        band_vib: Tuple[float, float] = (20, 400),
        use_audio: bool = False,
        use_prev: bool = False,
        norm_yaml: str = "assets/norm.yaml",
        deterministic_patches: bool = True,
    ) -> None:
        super().__init__()
        self.path = Path(parquet_or_h5_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset table not found: {self.path}")
        self.split = str(split)
        self.cfg = DatasetConfig(
            fs_vib=fs_vib,
            fs_aud=fs_aud,
            win_ms=win_ms,
            hop_ratio=hop_ratio,
            patch=patch,
            band_vib=(float(band_vib[0]), float(band_vib[1])),
            use_audio=use_audio,
            use_prev=use_prev,
            norm_yaml=norm_yaml,
            deterministic_patches=deterministic_patches,
        )
        self._mean, self._std = safe_mean_std_from_yaml(Path(norm_yaml))
        self._items: List[Dict[str, Any]] = []

        df = _read_table(self.path)
        # Optional split column; otherwise stratify by modulo
        if "split" in df.columns:
            df = df[df["split"].astype(str) == self.split].reset_index(drop=True)
        else:
            # simple 80/10/10 split by hash of texture_id
            def _split_of(row):
                key = str(row.get("texture_id", row.get("image_path", "")))
                h = abs(hash(key)) % 10
                return "train" if h < 8 else ("val" if h < 9 else "test")
            df["split"] = df.apply(_split_of, axis=1)
            df = df[df["split"].astype(str) == self.split].reset_index(drop=True)

        # Process rows into 100 ms windows
        win = int(round(self.cfg.fs_vib * (self.cfg.win_ms / 1000.0)))
        hop = int(max(1, round(win * self.cfg.hop_ratio)))
        sos = _butter_bandpass_sos(self.cfg.band_vib[0], self.cfg.band_vib[1], self.cfg.fs_vib)

        for i, row in df.iterrows():
            tex_id = row.get("texture_id", f"row{i}")
            mat_id = row.get("material_id", f"row{i}")
            img_path = Path(str(row.get("image_path", "")))
            u = row.get("u", None)
            v = row.get("v", None)

            # Load timeseries
            # Vibration
            vib_arr = None
            vib_sr = int(row.get("accel_sr", self.cfg.fs_vib))
            if "accel" in row and isinstance(row["accel"], (list, np.ndarray)):
                vib_arr = _load_array_maybe(row["accel"], "accel")
            elif "accel_path" in row and isinstance(row["accel_path"], (str, os.PathLike)):
                vib_arr = _load_array_maybe(row["accel_path"], "accel")
            if vib_arr is None:
                warnings.warn(f"Missing accel for texture_id={tex_id} image={img_path}")
                continue
            vib_res = _resample_poly(vib_arr, vib_sr, self.cfg.fs_vib)
            vib_bp = _sosfiltfilt(sos, vib_res).astype(np.float32)

            # Audio (optional)
            aud_bp = None
            if self.cfg.use_audio:
                aud_arr = None
                aud_sr = int(row.get("audio_sr", self.cfg.fs_aud))
                if "audio" in row and isinstance(row["audio"], (list, np.ndarray)):
                    aud_arr = _load_array_maybe(row["audio"], "audio")
                elif "audio_path" in row and isinstance(row["audio_path"], (str, os.PathLike)):
                    aud_arr = _load_array_maybe(row["audio_path"], "audio")
                if aud_arr is not None:
                    aud_res = _resample_poly(aud_arr, aud_sr, self.cfg.fs_aud)
                    # normalize to -12 dBFS RMS
                    aud_bp = _rms_normalize(aud_res, target_rms=10 ** (-12 / 20)).astype(np.float32)

            # Speed/force
            sp = None
            if "speed" in row and isinstance(row["speed"], (list, np.ndarray)):
                sp = _load_array_maybe(row["speed"], "speed")
            elif "speed_path" in row:
                sp = _load_array_maybe(row["speed_path"], "speed")
            fo = None
            if "force" in row and isinstance(row["force"], (list, np.ndarray)):
                fo = _load_array_maybe(row["force"], "force")
            elif "force_path" in row:
                fo = _load_array_maybe(row["force_path"], "force")
            # resample speed/force to fs_vib grid for aggregation
            if sp is None or fo is None:
                # Allow constant placeholders 0.5 if missing
                sp_res = np.full_like(vib_bp, 0.5)
                fo_res = np.full_like(vib_bp, 0.5)
            else:
                sp_res = _resample_poly(sp, int(row.get("speed_sr", self.cfg.fs_vib)), self.cfg.fs_vib)
                fo_res = _resample_poly(fo, int(row.get("force_sr", self.cfg.fs_vib)), self.cfg.fs_vib)
                # scale to [0,1] per-row robustly
                def _to_01(x):
                    lo, hi = np.percentile(x, [1, 99])
                    return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)
                sp_res = _to_01(sp_res)
                fo_res = _to_01(fo_res)

            # windows
            for s, e in _window_indices(len(vib_bp), win, hop):
                vib_win = vib_bp[s:e].astype(np.float32)
                if len(vib_win) != win:
                    continue
                # zero-mean vib
                vib_win = (vib_win - np.mean(vib_win)).astype(np.float32)
                aud_win = None
                if aud_bp is not None:
                    # 100 ms of audio at fs_aud
                    n_aud = int(round(self.cfg.fs_aud * (self.cfg.win_ms / 1000.0)))
                    # align via proportional index
                    s_a = int(round(s * (self.cfg.fs_aud / self.cfg.fs_vib)))
                    e_a = s_a + n_aud
                    if e_a <= len(aud_bp):
                        aud_win = aud_bp[s_a:e_a].astype(np.float32)
                        aud_win = np.clip(aud_win, -1.0, 1.0)

                # prev vib 100 ms
                prev_vib = None
                if self.cfg.use_prev:
                    ps = s - win
                    pe = s
                    if ps >= 0:
                        prev_vib = (vib_bp[ps:pe] - np.mean(vib_bp[ps:pe])).astype(np.float32)

                # patch crop
                uu, vv = _to_uv_or_deterministic(u, v, seed_key=str(tex_id))
                try:
                    patch_arr = _crop_uv_patch_gray(img_path, uu, vv, self.cfg.patch)
                except Exception as ex:
                    raise RuntimeError(
                        f"Failed cropping patch for texture_id={tex_id} material_id={mat_id} image={img_path}: {ex}"
                    )
                # z-score normalize
                patch_arr = (patch_arr - self._mean) / (self._std + 1e-6)

                state_speed = float(np.mean(sp_res[s:e]))
                state_force = float(np.mean(fo_res[s:e]))

                item = {
                    "texture_id": tex_id,
                    "material_id": mat_id,
                    "image_path": str(img_path),
                    "patch": patch_arr[None, ...].astype(np.float32),
                    "state": np.array([state_speed, state_force], dtype=np.float32),
                    "vib": vib_win.astype(np.float32),
                }
                if aud_win is not None and self.cfg.use_audio:
                    item["audio"] = aud_win.astype(np.float32)
                if prev_vib is not None and self.cfg.use_prev:
                    item["prev_vib"] = prev_vib.astype(np.float32)

                self._items.append(item)

        if len(self._items) == 0:
            raise RuntimeError(
                f"No usable windows extracted from {self.path} for split={self.split}. "
                f"Ensure columns accel/audio exist and image paths are valid."
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        it = self._items[idx]
        patch = torch.from_numpy(it["patch"]).to(torch.float32)  # (1,96,96)
        state = torch.from_numpy(it["state"]).to(torch.float32)  # (2,)
        vib = torch.from_numpy(it["vib"]).to(torch.float32)      # (100,)
        out: Dict[str, torch.Tensor] = {
            "patch": patch,
            "state": state,
            "vib": vib,
        }
        if "audio" in it:
            out["audio"] = torch.from_numpy(it["audio"]).to(torch.float32)  # (800,)
        if "prev_vib" in it:
            out["prev_vib"] = torch.from_numpy(it["prev_vib"]).to(torch.float32)  # (100,)

        # Assertions with helpful messages
        tex = it.get("texture_id", "?")
        img = it.get("image_path", "?")
        assert out["patch"].shape == (1, 96, 96), f"patch shape wrong for texture_id={tex} image={img}: {out['patch'].shape}"
        assert out["patch"].dtype == torch.float32
        assert out["state"].shape == (2,), f"state shape wrong for texture_id={tex}: {out['state'].shape}"
        assert out["vib"].shape == (100,), f"vib shape wrong for texture_id={tex}: {out['vib'].shape}"
        if "audio" in out:
            assert out["audio"].shape == (800,), f"audio shape wrong for texture_id={tex}: {out['audio'].shape}"
        if "prev_vib" in out:
            assert out["prev_vib"].shape == (100,), f"prev_vib shape wrong for texture_id={tex}: {out['prev_vib'].shape}"
        return out


def _cli_inspect(args) -> int:
    path = Path(args.data)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    ds = VisuoTactileDataset(
        parquet_or_h5_path=str(path),
        split=args.split,
        fs_vib=1000,
        fs_aud=8000,
        use_audio=True,
        use_prev=True,
    )
    n = min(int(args.inspect), len(ds))
    patches = []
    speeds = []
    forces = []
    vib_psd_y = []
    aud_psd_y = []
    for i in range(n):
        s = ds[i]
        patches.append(s["patch"])
        speeds.append(float(s["state"][0]))
        forces.append(float(s["state"][1]))
        vib_psd_y.append(s["vib"].numpy())
        if "audio" in s:
            aud_psd_y.append(s["audio"].numpy())

    patches_t = torch.stack(patches, dim=0)
    make_grid_save(out_dir / "patch_grid.png", patches_t, nrow=8, normalize=True)
    plot_hist(out_dir / "speeds_hist.png", np.array(speeds), title="speed")
    plot_hist(out_dir / "forces_hist.png", np.array(forces), title="force")
    plot_psd(out_dir / "vib_psd.png", np.concatenate(vib_psd_y), sr=1000, title="vibration PSD")
    if len(aud_psd_y) > 0:
        plot_psd(out_dir / "audio_psd.png", np.concatenate(aud_psd_y), sr=8000, title="audio PSD")

    # print stats
    print(json.dumps({
        "num_samples": len(ds),
        "inspect": n,
        "speed_mean": float(np.mean(speeds)),
        "speed_std": float(np.std(speeds)),
        "force_mean": float(np.mean(forces)),
        "force_std": float(np.std(forces)),
        "vib_rms": float(np.mean([rms(v) for v in vib_psd_y])),
        "audio_rms": float(np.mean([rms(a) for a in aud_psd_y])) if len(aud_psd_y) > 0 else None,
    }, indent=2))

    return 0


def main():
    import argparse

    p = argparse.ArgumentParser(description="VisuoTactileDataset sanity inspection")
    p.add_argument("--data", type=str, required=True, help="Path to parquet/h5 table")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--inspect", type=int, default=64)
    p.add_argument("--out", type=str, default="assets/debug_samples")
    args = p.parse_args()
    sys.exit(_cli_inspect(args))


if __name__ == "__main__":
    main()

