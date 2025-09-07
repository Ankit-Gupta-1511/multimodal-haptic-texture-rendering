"""Preprocess raw visuo-haptic recordings into fixed training windows.

CLI example:

python preprocess.py --raw <RAW_DIR_OR_TABLE> --out assets/dataset.parquet \
    --win_ms 100 --hop_ratio 0.5 --fs_vib 1000 --fs_aud 8000 \
    --band_vib 20 400 --speed_thresh 8 --deterministic_patches 1 \
    --val_textures 2 --test_textures 2

Outputs:
- assets/dataset.parquet: per-window samples (minimal duplication, images by path)
- assets/norm.yaml: normalization parameters (patch mean/std, state scaling, vib RMS)
- assets/debug_samples/: sanity plots (patch grid, histograms, PSDs)
"""

from __future__ import annotations

import os
import sys
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from scipy.signal import butter, sosfiltfilt, resample_poly, welch

try:
    from .utils import ensure_dir, make_grid_save, plot_hist, plot_psd
except Exception:  # pragma: no cover - fallback when run as top-level script
    from utils import ensure_dir, make_grid_save, plot_hist, plot_psd


# ----------------------------- I/O helpers -----------------------------

def _find_table_in_dir(root: Path) -> Optional[Path]:
    for ext in (".parquet", ".pq", ".csv", ".h5", ".hdf5"):
        cand = list(root.rglob(f"*{ext}"))
        if cand:
            return cand[0]
    return None


def _read_raw_table(path: Path) -> pd.DataFrame:
    if path.is_dir():
        inner = _find_table_in_dir(path)
        if inner is None:
            raise FileNotFoundError(f"No raw table found under directory: {path}")
        path = inner
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return pd.read_hdf(path)
    raise ValueError(f"Unsupported raw table format: {path}")


def _load_numeric_array(x: Any, key: str) -> Optional[np.ndarray]:
    """Return float32 numpy array or None. Accepts:
    - list/tuple/np.ndarray
    - JSON stringified list
    - path to .npy or .csv (single column or comma-separated)
    - path to .wav for audio
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    s = str(x)
    if s.strip().startswith("[") and s.strip().endswith("]"):
        try:
            return np.asarray(json.loads(s), dtype=np.float32)
        except Exception:
            pass
    p = Path(s)
    if p.suffix.lower() == ".npy" and p.exists():
        return np.load(p).astype(np.float32)
    if p.suffix.lower() == ".csv" and p.exists():
        try:
            arr = np.loadtxt(p, delimiter=",")
        except Exception:
            arr = np.loadtxt(p)
        return np.asarray(arr, dtype=np.float32)
    if p.suffix.lower() in {".wav", ".wave"} and p.exists():
        import soundfile as sf
        y, _sr = sf.read(p)
        return np.asarray(y, dtype=np.float32)
    return None


def _mel(value: float, hz: bool = True) -> float:
    return 2595.0 * math.log10(1.0 + value / 700.0) if hz else 700.0 * (10 ** (value / 2595.0) - 1.0)


# ---------------------------- Signal helpers ---------------------------

def _butter_bandpass_sos(low: float, high: float, fs: int, order: int = 4):
    return butter(order, [low, high], btype="band", fs=fs, output="sos")


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))) + 1e-12)


def _window_indices(n: int, win: int, hop: int) -> List[Tuple[int, int]]:
    idx = []
    s = 0
    while s + win <= n:
        idx.append((s, s + win))
        s += hop
    return idx


def _crop_wrap_gray(img_path: Path, u: float, v: float, patch: int) -> np.ndarray:
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


def _pseudo_uv(texture_id: Any, window_index: int) -> Tuple[float, float]:
    h = abs(hash((str(texture_id), int(window_index)))) % (2**32)
    rng = np.random.RandomState(h)
    return float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8))


# ---------------------------- Core pipeline ----------------------------

@dataclass
class PreCfg:
    raw_path: Path
    out_path: Path
    win_ms: int
    hop_ratio: float
    fs_vib: int
    fs_aud: int
    band_vib: Tuple[float, float]
    speed_thresh: float
    deterministic_patches: bool
    val_textures: int
    test_textures: int
    patch: int
    debug_dir: Path


def _split_by_texture(textures: List[Any], val_n: int, test_n: int, seed: int = 1337) -> Dict[Any, str]:
    rng = np.random.RandomState(seed)
    uniq = list(sorted(set(map(str, textures))))
    rng.shuffle(uniq)
    test = set(uniq[: test_n])
    val = set(uniq[test_n : test_n + val_n])
    split = {}
    for t in textures:
        st = str(t)
        if st in test:
            split[t] = "test"
        elif st in val:
            split[t] = "val"
        else:
            split[t] = "train"
    return split


def preprocess(cfg: PreCfg) -> int:
    ensure_dir(cfg.out_path.parent)
    ensure_dir(cfg.debug_dir)

    df_raw = _read_raw_table(cfg.raw_path)
    if "texture_id" not in df_raw.columns or "image_path" not in df_raw.columns:
        raise ValueError("Raw table must contain at least 'texture_id' and 'image_path' columns")

    # Stats accumulators
    patch_sum = 0.0
    patch_sq_sum = 0.0
    patch_count = 0
    speeds_all = []
    forces_all = []
    vib_sq_acc = 0.0
    vib_count = 0

    fs_v = int(cfg.fs_vib)
    fs_a = int(cfg.fs_aud)
    win_v = int(round(fs_v * (cfg.win_ms / 1000.0)))
    win_a = int(round(fs_a * (cfg.win_ms / 1000.0)))
    hop_v = int(max(1, round(win_v * cfg.hop_ratio)))
    sos = _butter_bandpass_sos(cfg.band_vib[0], cfg.band_vib[1], fs_v)

    rows: List[Dict[str, Any]] = []
    uv_fallbacks = 0
    total_windows = 0

    for ridx, row in df_raw.iterrows():
        tex_id = row.get("texture_id")
        img_path = Path(str(row.get("image_path")))
        if not img_path.exists():
            warnings.warn(f"Image not found for texture_id={tex_id}: {img_path}")
            continue

        # Load raw arrays (any reasonable naming)
        vib = _load_numeric_array(row.get("accel"), "accel") or _load_numeric_array(row.get("accel_signal"), "accel_signal") or _load_numeric_array(row.get("accel_signal_path"), "accel_signal_path") or _load_numeric_array(row.get("accel_path"), "accel_path")
        if vib is None:
            warnings.warn(f"Missing vibration for texture_id={tex_id}")
            continue
        vib_sr = int(row.get("accel_sr", fs_v))
        vib = resample_poly(vib, fs_v, vib_sr) if vib_sr != fs_v else vib
        vib = vib.astype(np.float32)
        vib = vib - float(np.mean(vib))
        try:
            vib = sosfiltfilt(sos, vib).astype(np.float32)
        except Exception:
            # If SciPy stability issues on very short input, skip filtering
            pass

        # Audio (optional)
        audio = _load_numeric_array(row.get("audio"), "audio") or _load_numeric_array(row.get("audio_path"), "audio_path")
        aud_sr = int(row.get("audio_sr", fs_a))
        if audio is not None and len(audio) >= 8:
            audio = resample_poly(audio, fs_a, aud_sr) if aud_sr != fs_a else audio.astype(np.float32)
            # Target -12 dBFS RMS
            target_rms = 10 ** (-12 / 20)
            cur = _rms(audio)
            if cur > 1e-9:
                audio = (audio * (target_rms / cur)).astype(np.float32)
            audio = np.clip(audio, -1.0, 1.0)
        else:
            audio = None

        # Speed/force: accept array or scalar; if missing use zeros
        speed_arr = _load_numeric_array(row.get("speed"), "speed") or _load_numeric_array(row.get("speed_path"), "speed_path")
        force_arr = _load_numeric_array(row.get("normal_force"), "normal_force") or _load_numeric_array(row.get("force"), "force") or _load_numeric_array(row.get("force_path"), "force_path")
        sp_sr = int(row.get("speed_sr", fs_v))
        fo_sr = int(row.get("force_sr", fs_v))
        if speed_arr is None:
            speed_arr = np.full_like(vib, fill_value=float(row.get("speed", 0.0)), dtype=np.float32)
        else:
            speed_arr = resample_poly(speed_arr, fs_v, sp_sr) if sp_sr != fs_v else speed_arr.astype(np.float32)
        if force_arr is None:
            force_arr = np.full_like(vib, fill_value=float(row.get("normal_force", 0.0)), dtype=np.float32)
        else:
            force_arr = resample_poly(force_arr, fs_v, fo_sr) if fo_sr != fs_v else force_arr.astype(np.float32)

        # UV per-sample optional
        uv = _load_numeric_array(row.get("uv"), "uv") or _load_numeric_array(row.get("uv_path"), "uv_path")
        if uv is not None:
            uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
            if len(uv) != len(vib):
                # resample UV by indexing proportionally
                idx = np.linspace(0, len(uv) - 1, num=len(vib))
                uv = np.stack([
                    np.interp(idx, np.arange(len(uv)), uv[:, 0]),
                    np.interp(idx, np.arange(len(uv)), uv[:, 1]),
                ], axis=1).astype(np.float32)

        # Windowing & thresholds
        win_idx = _window_indices(len(vib), win_v, hop_v)
        for wj, (s, e) in enumerate(win_idx):
            total_windows += 1
            sp_m = float(np.mean(speed_arr[s:e])) if e <= len(speed_arr) else 0.0
            fo_m = float(np.mean(force_arr[s:e])) if e <= len(force_arr) else 0.0
            # Sliding detection by threshold
            if sp_m <= float(cfg.speed_thresh):
                continue

            # Vibration window
            v_win = vib[s:e].astype(np.float32)
            if len(v_win) != win_v:
                continue
            v_win = (v_win - float(np.mean(v_win))).astype(np.float32)
            vib_sq_acc += float(np.sum(np.square(v_win)))
            vib_count += len(v_win)

            # Previous window (optional, always stored for convenience)
            pv_win = None
            ps, pe = s - win_v, s
            if ps >= 0:
                pv_win = vib[ps:pe].astype(np.float32)
                pv_win = (pv_win - float(np.mean(pv_win))).astype(np.float32)

            # Audio window
            a_win = None
            if audio is not None:
                s_a = int(round(s * (fs_a / fs_v)))
                e_a = s_a + win_a
                if e_a <= len(audio):
                    a_win = audio[s_a:e_a].astype(np.float32)

            # UV center per-window
            if uv is not None and e <= len(uv):
                u_c = float(np.clip(np.mean(uv[s:e, 0]), 0.0, 1.0))
                v_c = float(np.clip(np.mean(uv[s:e, 1]), 0.0, 1.0))
            else:
                u_c, v_c = _pseudo_uv(tex_id, wj)
                uv_fallbacks += 1

            # Accumulate patch stats (streamed)
            try:
                patch_arr = _crop_wrap_gray(img_path, u_c, v_c, cfg.patch)
                patch_sum += float(np.sum(patch_arr))
                patch_sq_sum += float(np.sum(patch_arr ** 2))
                patch_count += patch_arr.size
            except Exception:
                # keep going; dataset will crop again at load time
                pass

            speeds_all.append(sp_m)
            forces_all.append(fo_m)

            rows.append({
                "texture_id": tex_id,
                "image_path": str(img_path),
                "window_index": int(wj),
                "u": u_c,
                "v": v_c,
                "speed_mean": sp_m,
                "force_mean": fo_m,
                "vib": v_win.tolist(),
                "prev_vib": pv_win.tolist() if pv_win is not None else None,
                "audio": a_win.tolist() if a_win is not None else None,
                "fs_vib": fs_v,
                "fs_aud": fs_a,
            })

    if not rows:
        raise RuntimeError("No windows produced. Check speed threshold, raw signals, and timing alignment.")

    # Splits by texture_id
    textures = [r["texture_id"] for r in rows]
    split_map = _split_by_texture(textures, cfg.val_textures, cfg.test_textures)
    for r in rows:
        r["split"] = split_map[r["texture_id"]]

    # Save parquet
    df_out = pd.DataFrame(rows)
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df_out.to_parquet(cfg.out_path)
    except Exception:
        # Fallback to HDF5 if pyarrow/fastparquet not available
        alt = cfg.out_path.with_suffix(".h5")
        warnings.warn(f"Parquet write failed; writing HDF5 to {alt} instead")
        df_out.to_hdf(alt, key="data", mode="w")

    # Compute normalization
    patch_mean = float(patch_sum / max(1, patch_count))
    patch_var = max(0.0, float(patch_sq_sum / max(1, patch_count) - patch_mean ** 2))
    patch_std = float(np.sqrt(patch_var + 1e-12))
    speed_min, speed_max = (float(np.min(speeds_all)), float(np.max(speeds_all))) if speeds_all else (0.0, 1.0)
    force_min, force_max = (float(np.min(forces_all)), float(np.max(forces_all))) if forces_all else (0.0, 1.0)
    vib_rms = float(np.sqrt(vib_sq_acc / max(1, vib_count))) if vib_count else 1.0

    norm = {
        "patch_mean": patch_mean,
        "patch_std": patch_std,
        "speed_min": speed_min,
        "speed_max": speed_max,
        "force_min": force_min,
        "force_max": force_max,
        "vib_rms": vib_rms,
        "fs_vib": fs_v,
        "fs_aud": fs_a,
        "win_ms": cfg.win_ms,
        "hop_ratio": cfg.hop_ratio,
        "band_vib": list(cfg.band_vib),
        "patch": cfg.patch,
    }
    norm_path = cfg.out_path.parent.joinpath("norm.yaml")
    import yaml
    with open(norm_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(norm, f)

    # Sanity plots
    dbg = cfg.debug_dir
    # Patch grid: take first 32 (or fewer) and crop now for quick visual
    n_inspect = min(32, len(rows))
    patch_imgs = []
    for i in range(n_inspect):
        r = rows[i]
        try:
            patch_arr = _crop_wrap_gray(Path(r["image_path"]), r["u"], r["v"], cfg.patch)
            patch_imgs.append(torchify(patch_arr[None, ...]))
        except Exception:
            continue
    if patch_imgs:
        import torch
        grid = torch.stack(patch_imgs, dim=0)
        make_grid_save(dbg / "patch_grid.png", grid, nrow=8, normalize=True)
    plot_hist(dbg / "speed_hist.png", np.asarray(speeds_all, dtype=np.float32), title="speed (raw mean)")
    plot_hist(dbg / "force_hist.png", np.asarray(forces_all, dtype=np.float32), title="force (raw mean)")
    # PSDs: concatenate a subset for readability
    vib_cat = np.concatenate([np.asarray(r["vib"], dtype=np.float32) for r in rows[:512]], axis=0)
    plot_psd(dbg / "vib_psd.png", vib_cat, sr=fs_v, title="vib PSD (band-pass check)")
    aud_cat = np.concatenate([np.asarray(r["audio"], dtype=np.float32) for r in rows[:512] if r.get("audio") is not None], axis=0)
    if len(aud_cat) > 0:
        plot_psd(dbg / "audio_psd.png", aud_cat, sr=fs_a, title="audio PSD")

    # Print dataset stats to stdout
    print(json.dumps({
        "rows": len(rows),
        "split_counts": {k: int(v) for k, v in dict(df_out["split"].value_counts()).items()},
        "textures": int(len(set(df_out["texture_id"]))),
        "speed_range": [float(speed_min), float(speed_max)],
        "force_range": [float(force_min), float(force_max)],
        "vib_rms": float(vib_rms),
        "uv_fallback_ratio": float(uv_fallbacks / max(1, total_windows)),
        "parquet": str(cfg.out_path),
        "norm_yaml": str(norm_path),
        "debug_dir": str(cfg.debug_dir),
    }, indent=2))

    return 0


# tiny utility to avoid importing torch unless needed
def torchify(arr: np.ndarray):
    import torch
    return torch.from_numpy(arr.astype(np.float32))


def main():
    import argparse
    p = argparse.ArgumentParser(description="Preprocess visuo-haptic dataset into fixed windows")
    p.add_argument("--raw", type=str, required=True, help="Raw directory or table (csv/parquet/hdf5)")
    p.add_argument("--out", type=str, required=True, help="Output parquet path (e.g., assets/dataset.parquet)")
    p.add_argument("--win_ms", type=int, default=100)
    p.add_argument("--hop_ratio", type=float, default=0.5)
    p.add_argument("--fs_vib", type=int, default=1000)
    p.add_argument("--fs_aud", type=int, default=8000)
    p.add_argument("--band_vib", type=float, nargs=2, default=[20.0, 400.0])
    p.add_argument("--speed_thresh", type=float, default=8.0)
    p.add_argument("--deterministic_patches", type=int, default=1)
    p.add_argument("--val_textures", type=int, default=2)
    p.add_argument("--test_textures", type=int, default=2)
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--debug_dir", type=str, default="assets/debug_samples")
    args = p.parse_args()

    cfg = PreCfg(
        raw_path=Path(args.raw),
        out_path=Path(args.out),
        win_ms=int(args.win_ms),
        hop_ratio=float(args.hop_ratio),
        fs_vib=int(args.fs_vib),
        fs_aud=int(args.fs_aud),
        band_vib=(float(args.band_vib[0]), float(args.band_vib[1])),
        speed_thresh=float(args.speed_thresh),
        deterministic_patches=bool(int(args.deterministic_patches)),
        val_textures=int(args.val_textures),
        test_textures=int(args.test_textures),
        patch=int(args.patch),
        debug_dir=Path(args.debug_dir),
    )
    sys.exit(preprocess(cfg))


if __name__ == "__main__":
    main()
