"""VisuoTactileDataset and a CLI sanity inspection tool.

This dataset expects preprocessed windows serialized by `preprocess.py` into a
single Parquet (preferred) or HDF5 file. Each row represents a fixed window
with metadata and compact arrays:

Required columns (per-window):
- texture_id: str/int identifier used for splitting
- image_path: path to RGB image of the texture
- window_index: stable per-row index within texture/recording
- u, v: center UV in [0,1] used for cropping (fallback deterministic if missing)
- speed_mean, force_mean: per-window means prior to normalization
- vib: list[float] length=win_vib (e.g., 100)
- split: train/val/test

Optional columns:
- audio: list[float] length=win_aud (e.g., 800)
- prev_vib: list[float] length=win_vib (e.g., 100)
- fs_vib, fs_aud: stored for sanity (the dataset will check and trust)

Normalization parameters are read from a YAML at `norm_yaml` containing:
- patch_mean, patch_std (for grayscale image patches)
- speed_min, speed_max, force_min, force_max (for [0,1] scaling)
- vib_rms (dataset-wide RMS used to unit-scale vib during training)

CLI sanity command:
    python -m dataset_textures --data assets/dataset.parquet --split train \
        --inspect 32 --out assets/debug_samples
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

try:
    from .utils import ensure_dir, plot_psd, plot_hist, make_grid_save
except Exception:  # pragma: no cover - fallback when run as top-level module
    from utils import ensure_dir, plot_psd, plot_hist, make_grid_save
import yaml


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return pd.read_hdf(path)
    raise ValueError(f"Unsupported dataset table format: {path}")


def _to_numpy_array(x, key: str) -> np.ndarray:
    """Convert a column value to a float32 numpy array.

    Supports: Python list/tuple, numpy array, JSON stringified list (for Parquet
    compatibility). Raises with a helpful error if conversion fails.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        raise FileNotFoundError(f"Missing array for {key}")
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if isinstance(x, (bytes, bytearray)):
        # Arrow may store as bytes for large lists; try JSON decode
        try:
            j = json.loads(x.decode("utf-8"))
            return np.asarray(j, dtype=np.float32)
        except Exception as ex:
            raise TypeError(f"Unsupported bytes format for {key}: {ex}")
    if isinstance(x, str):
        xs = x.strip()
        if xs.startswith("[") and xs.endswith("]"):
            try:
                return np.asarray(json.loads(xs), dtype=np.float32)
            except Exception as ex:
                raise TypeError(f"JSON parse failed for {key}: {ex}")
        # Fallthrough: maybe path; dataset should not inline big files here
    raise TypeError(f"Unsupported type for {key}: {type(x)}")


def _to_uv_or_deterministic(u: Optional[float], v: Optional[float], texture_id: Any, window_index: int) -> Tuple[float, float]:
    """Return (u,v) in [0,1]. If inputs are invalid, create deterministic pseudo-UV
    using a hash of (texture_id, window_index) to ensure stable sampling.
    """
    if u is not None and v is not None and 0.0 <= float(u) <= 1.0 and 0.0 <= float(v) <= 1.0:
        return float(u), float(v)
    h = abs(hash((str(texture_id), int(window_index)))) % (2**32)
    rng = np.random.RandomState(h)
    return float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8))


def _crop_wrap_gray(img_path: Path, u: float, v: float, patch: int) -> np.ndarray:
    """Grayscale UV-centered crop with wrapping at edges.

    - Converts to grayscale
    - Centers crop at (u,v) in normalized [0,1]
    - Wraps indices to avoid edge shrinkage
    - Output shape: (patch, patch), float32 in [0,1]
    """
    with Image.open(img_path) as im:
        im = im.convert("L")
        W, H = im.size
        cx = int(round(u * (W - 1)))
        cy = int(round(v * (H - 1)))
        half = patch // 2
        xs = np.arange(cx - half, cx - half + patch)
        ys = np.arange(cy - half, cy - half + patch)
        xs = np.mod(xs, W)
        ys = np.mod(ys, H)
        arr = np.asarray(im, dtype=np.float32)
        patch_arr = arr[np.ix_(ys, xs)] / 255.0
        return patch_arr.astype(np.float32)


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
    """Dataset that loads pre-windowed samples and assembles tensors.

    It crops image patches on-the-fly using stored UV centers, normalizes patch
    via z-score (dataset mean/std), scales state to [0,1] using dataset min/max,
    and unit-scales vibration using dataset RMS.
    """

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
            fs_vib=int(fs_vib),
            fs_aud=int(fs_aud),
            win_ms=int(win_ms),
            hop_ratio=float(hop_ratio),
            patch=int(patch),
            band_vib=(float(band_vib[0]), float(band_vib[1])),
            use_audio=bool(use_audio),
            use_prev=bool(use_prev),
            norm_yaml=str(norm_yaml),
            deterministic_patches=bool(deterministic_patches),
        )

        # Load normalization parameters
        ny = Path(norm_yaml)
        if not ny.exists():
            raise FileNotFoundError(f"Normalization YAML not found: {ny}")
        with open(ny, "r", encoding="utf-8") as f:
            norm = yaml.safe_load(f) or {}
        self.patch_mean = float(norm.get("patch_mean", 0.5))
        self.patch_std = float(norm.get("patch_std", 0.25))
        self.speed_min = float(norm.get("speed_min", 0.0))
        self.speed_max = float(norm.get("speed_max", 1.0))
        self.force_min = float(norm.get("force_min", 0.0))
        self.force_max = float(norm.get("force_max", 1.0))
        self.vib_rms = float(norm.get("vib_rms", 1.0))

        df = _read_table(self.path)
        if "split" not in df.columns:
            warnings.warn("Dataset has no 'split' column; defaulting to hash-based 80/10/10 split")
            def _split_of(row):
                key = str(row.get("texture_id", row.get("image_path", "")))
                h = abs(hash(key)) % 10
                return "train" if h < 8 else ("val" if h < 9 else "test")
            df["split"] = df.apply(_split_of, axis=1)
        df = df[df["split"].astype(str) == self.split].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(f"No rows for split={self.split} in {self.path}")
        self.df = df
        self._fallback_uv_count = 0
        self._total_uv = len(df)

    def __len__(self) -> int:
        return len(self.df)

    def _scale01(self, x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return float(np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[int(idx)]
        tex_id = row.get("texture_id", f"row{idx}")
        img_path = Path(str(row.get("image_path", "")))
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found for texture_id={tex_id}: {img_path}")
        u = row.get("u", None)
        v = row.get("v", None)
        widx = int(row.get("window_index", idx))
        uu, vv = _to_uv_or_deterministic(u if pd.notna(u) else None, v if pd.notna(v) else None, tex_id, widx)
        if (u is None or v is None or not (0.0 <= float(u) <= 1.0 and 0.0 <= float(v) <= 1.0)) and self.cfg.deterministic_patches:
            self._fallback_uv_count += 1
        try:
            patch_arr = _crop_wrap_gray(img_path, uu, vv, self.cfg.patch)
        except Exception as ex:
            raise RuntimeError(f"Patch crop failed texture_id={tex_id} image={img_path} (u={uu:.3f},v={vv:.3f}): {ex}")
        patch_arr = (patch_arr - self.patch_mean) / (self.patch_std + 1e-6)

        sp = float(row.get("speed_mean", 0.0))
        fo = float(row.get("force_mean", 0.0))
        sp01 = self._scale01(sp, self.speed_min, self.speed_max)
        fo01 = self._scale01(fo, self.force_min, self.force_max)

        vib = _to_numpy_array(row.get("vib", None), "vib")
        if vib.ndim != 1:
            raise ValueError(f"vib must be 1D for texture_id={tex_id} window={widx}: got shape {vib.shape}")
        if len(vib) != int(round(self.cfg.fs_vib * (self.cfg.win_ms / 1000.0))):
            raise ValueError(
                f"vib length mismatch for texture_id={tex_id} window={widx}: got {len(vib)},"
                f" expected {int(round(self.cfg.fs_vib * (self.cfg.win_ms / 1000.0)))}"
            )
        vib = (vib.astype(np.float32)) / max(self.vib_rms, 1e-6)

        out: Dict[str, torch.Tensor] = {
            "patch": torch.from_numpy(patch_arr[None, ...]).to(torch.float32),
            "state": torch.tensor([sp01, fo01], dtype=torch.float32),
            "vib": torch.from_numpy(vib.astype(np.float32)),
        }

        aud_val = row.get("audio", None)
        if self.cfg.use_audio and aud_val is not None and not (isinstance(aud_val, float) and np.isnan(aud_val)):
            aud = _to_numpy_array(aud_val, "audio").astype(np.float32)
            out["audio"] = torch.from_numpy(aud)

        pv_val = row.get("prev_vib", None)
        if self.cfg.use_prev and pv_val is not None and not (isinstance(pv_val, float) and np.isnan(pv_val)):
            pv = _to_numpy_array(pv_val, "prev_vib").astype(np.float32)
            out["prev_vib"] = torch.from_numpy(pv)

        # Assertions with helpful messages
        assert out["patch"].shape == (1, self.cfg.patch, self.cfg.patch), (
            f"patch shape wrong for texture_id={tex_id} image={img_path}: {tuple(out['patch'].shape)}")
        assert out["state"].shape == (2,), f"state shape wrong for texture_id={tex_id}: {tuple(out['state'].shape)}"
        assert out["vib"].shape[0] == int(round(self.cfg.fs_vib * (self.cfg.win_ms / 1000.0))), (
            f"vib length wrong for texture_id={tex_id}: {tuple(out['vib'].shape)}")
        if self.cfg.use_audio and "audio" in out:
            assert out["audio"].shape[0] == int(round(self.cfg.fs_aud * (self.cfg.win_ms / 1000.0))), (
                f"audio length wrong for texture_id={tex_id}: {tuple(out['audio'].shape)}")
        if self.cfg.use_prev and "prev_vib" in out:
            assert out["prev_vib"].shape[0] == int(round(self.cfg.fs_vib * (self.cfg.win_ms / 1000.0))), (
                f"prev_vib length wrong for texture_id={tex_id}: {tuple(out['prev_vib'].shape)}")

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
    vib_cat = []
    aud_cat = []
    for i in range(n):
        s = ds[i]
        patches.append(s["patch"])  # (1,H,W)
        speeds.append(float(s["state"][0]))
        forces.append(float(s["state"][1]))
        vib_cat.append(s["vib"].numpy())
        if "audio" in s:
            aud_cat.append(s["audio"].numpy())

    patches_t = torch.stack(patches, dim=0)
    make_grid_save(out_dir / "patch_grid.png", patches_t, nrow=8, normalize=True)
    plot_hist(out_dir / "speeds_hist.png", np.array(speeds), title="speed [0..1]")
    plot_hist(out_dir / "forces_hist.png", np.array(forces), title="force [0..1]")
    plot_psd(out_dir / "vib_psd.png", np.concatenate(vib_cat), sr=1000, title="vibration PSD")
    if len(aud_cat) > 0:
        plot_psd(out_dir / "audio_psd.png", np.concatenate(aud_cat), sr=8000, title="audio PSD")

    # Print dataset stats
    stats = {
        "num_samples": len(ds),
        "inspected": n,
        "speed_mean": float(np.mean(speeds)) if speeds else None,
        "force_mean": float(np.mean(forces)) if forces else None,
        "fallback_uv_frac": float(ds._fallback_uv_count / max(1, ds._total_uv)),
    }
    print(json.dumps(stats, indent=2))
    if ds._fallback_uv_count / max(1, ds._total_uv) > 0.3:
        warnings.warn("High fraction of fallback UVs; check UV mapping or set deterministic_patches=True")
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
