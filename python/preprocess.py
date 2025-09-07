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
import warnings
from dataclasses import dataclass
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import h5py
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt, resample_poly
try:
    from tqdm import tqdm
except Exception:  # fallback if tqdm not available
    def tqdm(x, **kwargs):
        return x

try:
    from .utils import ensure_dir, make_grid_save, plot_hist, plot_psd
except Exception:  # pragma: no cover - fallback when run as top-level script
    from utils import ensure_dir, make_grid_save, plot_hist, plot_psd


# ----------------------------- I/O helpers -----------------------------


# ------------------------- HAVdb integration (embedded) -----------------

def hav_get_data(path: str, user: str, texture: str, trial: str, time_offset: int = -1691745000):
    """Embedded reader for HAVdb .h5 files.

    Returns: ([time_kistler, kistler_data], [time_pos, pos_data], [time_ft_sensor, ft_sensor_data])
    """
    filename = f"{user}_{texture}_{trial}.h5"
    with h5py.File(Path(path) / user / texture / filename, "r") as f:
        time_kistler = f["time_kistler"][:] + time_offset
        kistler_data = f["kistler_data"][:]
        time_pos = f["time_pos"][:] + time_offset
        pos_data = f["pos_data"][:]
        time_ft_sensor = f["time_ft_sensor"][:] + time_offset
        ft_sensor_data = f["ft_sensor_data"][:]
    return ([time_kistler, kistler_data], [time_pos, pos_data], [time_ft_sensor, ft_sensor_data])


def hav_get_position_features(time: np.ndarray, values: np.ndarray):
    """Resample position to 50 Hz then 240 Hz, lowpass, compute speed and direction."""
    # 50 Hz uniform
    new_time = np.arange(time[0, 0], time[-1, 0], 1 / 50)
    new_values = np.zeros((new_time.shape[0], values.shape[1]), dtype=np.float64)
    for k in range(values.shape[1]):
        new_values[:, k] = interp1d(time[:, 0], values[:, k], axis=0, bounds_error=False, fill_value=0, kind="linear")(new_time)
    time = np.reshape(new_time, (new_time.shape[0], 1))
    values = new_values
    # 240 Hz uniform
    Fs = 240
    new_time = np.arange(time[0, 0], time[-1, 0], 1 / Fs)
    new_values = np.zeros((new_time.shape[0], values.shape[1]), dtype=np.float64)
    for k in range(values.shape[1]):
        new_values[:, k] = interp1d(time[:, 0], values[:, k], axis=0, bounds_error=False, fill_value=0, kind="cubic")(new_time)
    time = np.reshape(new_time, (new_time.shape[0], 1))
    values = new_values
    # lowpass 15 Hz
    fc = 15
    a, b = signal.butter(10, fc / (Fs / 2), "low")
    for k in range(values.shape[1]):
        values[:, k] = signal.filtfilt(a, b, values[:, k])
    # compute features
    posX = values[:, 0]
    posY = values[:, 1]
    posX2 = values[:, 2]
    posY2 = values[:, 3]
    step_distance = np.sqrt(np.diff(posX) ** 2 + np.diff(posY) ** 2)
    speed = step_distance / (np.diff(time[:, 0]))
    speed = np.append(speed, speed[-1])
    position = np.concatenate((posX.reshape(-1, 1), posY.reshape(-1, 1)), axis=1)
    direction = np.zeros((position.shape[0],), dtype=np.float64)
    finger_direction = np.zeros((position.shape[0],), dtype=np.float64)
    nb_turn = 0
    for i in range(1, position.shape[0] - 1):
        direction[i] = np.arctan2(posY[i + 1] - posY[i - 1], posX[i + 1] - posX[i - 1]) * 180 / np.pi
        if direction[i] - (direction[i - 1] - nb_turn * 360) < -320:
            nb_turn = nb_turn + 1
        elif direction[i] - (direction[i - 1] - nb_turn * 360) > 320:
            nb_turn = nb_turn - 1
        direction[i] = direction[i] + nb_turn * 360
        dist = np.sqrt((posX2[i] - posX[i]) ** 2 + (posY2[i] - posY[i]) ** 2)
        if abs(0.150 - dist) < 0.04:
            finger_direction[i] = np.arctan2(posY2[i] - posY[i], posX2[i] - posX[i]) * 180 / np.pi + 90
        else:
            finger_direction[i] = finger_direction[i - 1]
    direction = direction - finger_direction
    return [time, position, speed, direction]


def hav_get_kistler_features(time: np.ndarray, values: np.ndarray):
    audio = values[:, 0:2]
    time_audio = time
    vibration = values[:, 2:4]
    time_vibration = time
    return [time_audio, audio], [time_vibration, vibration]


def hav_get_ft_sensor_features(time: np.ndarray, values: np.ndarray):
    force = values[:, 0:3]
    time_force = time
    torque = values[:, 3:6]
    time_torque = time
    return [time_force, force], [time_torque, torque]


def hav_align_all(list_of_times, list_of_data, sampling_rate: int):
    time_1d = [t.squeeze() for t in list_of_times]
    start_time = max([time[0] for time in list_of_times])[0]
    end_time = min([time[-1] for time in list_of_times])[0]
    common_time = np.arange(start_time, end_time, 1 / sampling_rate)
    aligned_data = [interp1d(time, data, axis=0)(common_time) for time, data in zip(time_1d, list_of_data)]
    return common_time, aligned_data


def hav_load_data(path: str, user: str, texture: str, trial: str, sampling_rate: int, time_offset: int = -1691745000):
    ([time_kistler, kistler_data], [time_pos, pos_data], [time_ft_sensor, ft_sensor_data]) = hav_get_data(path, user, texture, trial, time_offset)
    [time_pos, pos, spd, dir] = hav_get_position_features(time_pos, pos_data)
    [time_force, force], [time_torque, torque] = hav_get_ft_sensor_features(time_ft_sensor, ft_sensor_data)
    [time_audio, mic], [time_vibration, vib] = hav_get_kistler_features(time_kistler, kistler_data)
    times = [time_pos, time_pos, time_pos, time_vibration, time_audio, time_force, time_torque]
    datas = [pos, spd, dir, vib, mic, force, torque]
    time_min = min([time[0] for time in times])
    times = [time - time_min for time in times]
    common_time, aligned_data = hav_align_all(times, datas, sampling_rate)
    time_pos = time_pos[:, 0] - time_pos[0, 0]
    common_time = common_time[:] - common_time[0]
    position = aligned_data[0]
    speed = aligned_data[1]
    direction = aligned_data[2]
    vibration = aligned_data[3]
    audio = aligned_data[4]
    force = aligned_data[5]
    torque = aligned_data[6]
    return [common_time, position, speed, direction, vibration, audio, force, torque], [time_pos, pos, spd, dir]


def _scan_havdb_h5(root: Path) -> List[Dict[str, Any]]:
    """Scan a HAVdb-style directory for h5 recordings.

    Pattern: <root>/**/subject_X/<texture>/<user>_<texture>_<trial>.h5
    Returns list of dicts: {user, texture, trial, h5_path}
    """
    out: List[Dict[str, Any]] = []
    for p in root.rglob("*.h5"):
        stem = p.stem
        # Prefer parent folders for user/texture
        texture = p.parent.name if p.parent else None
        user = p.parent.parent.name if p.parent and p.parent.parent else None
        if not user or not texture:
            parts = stem.split("_")
            if len(parts) >= 3:
                user = parts[0]
                texture = parts[1]
        trial = stem
        if user and texture:
            pref = f"{user}_{texture}_"
            if stem.startswith(pref):
                trial = stem[len(pref):]
        out.append({"user": user, "texture": texture, "trial": trial, "h5_path": p})
    return out


def _choose_texture_image(img_root: Path, texture: str) -> Optional[Path]:
    """Pick a representative image for a texture from common subfolders."""
    # common candidates first
    for cand in [
        img_root / "img" / texture / f"{texture}_l_0.jpg",
        img_root / "img" / texture / f"{texture}_r_0.jpg",
        img_root / "img" / texture / f"{texture}_0.jpg",
        img_root / texture / f"{texture}_l_0.jpg",
        img_root / texture / f"{texture}_r_0.jpg",
    ]:
        if cand.is_file():
            return cand
    # any image inside the texture folder
    for folder in [img_root / "img" / texture, img_root / texture, img_root / "img", img_root]:
        if folder.is_dir():
            imgs = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
            if imgs:
                # prefer files starting with texture id
                for im in imgs:
                    if im.name.startswith(texture):
                        return im
                return imgs[0]
    # search broadly but limit to first match
    for sub in [img_root, img_root / "img", img_root / "images", img_root / "textures"]:
        if not sub.exists():
            continue
        for cand in sub.rglob("*.jpg"):
            if cand.parent.name == texture or cand.name.startswith(texture):
                return cand
        for cand in sub.rglob("*.png"):
            if cand.parent.name == texture or cand.name.startswith(texture):
                return cand
    return None


# (no generic array loaders necessary for fixed HAVdb structure)


# (no mel utilities required in preprocessing)


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


def _load_image_gray_arr(img_path: Path) -> Optional[np.ndarray]:
    try:
        with Image.open(img_path) as im:
            im = im.convert("L")
            return (np.asarray(im, dtype=np.float32) / 255.0)
    except Exception:
        return None


def _crop_from_arr_wrap(arr: np.ndarray, u: float, v: float, patch: int) -> np.ndarray:
    H, W = arr.shape
    cx = int(round(u * (W - 1)))
    cy = int(round(v * (H - 1)))
    half = patch // 2
    xs = np.arange(cx - half, cx - half + patch)
    ys = np.arange(cy - half, cy - half + patch)
    xs = np.mod(xs, W)
    ys = np.mod(ys, H)
    return arr[np.ix_(ys, xs)].astype(np.float32)


def _process_record_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    """Worker-safe per-file processing for parallel mode."""
    try:
        user = args["user"]; tex_id = args["texture"]; trial = args["trial"]
        h5_path = Path(args["h5_path"])  # not used directly; local_root holds base
        local_root = Path(args["local_root"])  # root containing user folders
        fs_v = int(args["fs_v"]); fs_a = int(args["fs_a"]) 
        win_v = int(args["win_v"]); win_a = int(args["win_a"]); hop_v = int(args["hop_v"]) 
        lo, hi = float(args["band_lo"]), float(args["band_hi"]) 
        speed_thresh = float(args["speed_thresh"]) 
        patch = int(args["patch"]) 
        hop_prune = max(1, int(args.get("hop_prune", 1)))
        img_roots = [Path(p) for p in args["img_roots"]]

        data, _ = hav_load_data(str(local_root), user, tex_id, trial, sampling_rate=max(fs_v, fs_a))
        _, position, speed, direction, vibration, audio, force, torque = data
        vib_ch = vibration[:, 0].astype(np.float32)
        aud_ch = audio[:, 0].astype(np.float32) if isinstance(audio, np.ndarray) and audio.size else None
        vib_rs = resample_poly(vib_ch, fs_v, max(fs_v, fs_a)) if max(fs_v, fs_a) != fs_v else vib_ch
        if aud_ch is not None:
            aud_rs = resample_poly(aud_ch, fs_a, max(fs_v, fs_a)) if max(fs_v, fs_a) != fs_a else aud_ch
        else:
            aud_rs = None
        vib_rs = vib_rs - float(np.mean(vib_rs))
        try:
            sos = _butter_bandpass_sos(lo, hi, fs_v)
            vib_rs = sosfiltfilt(sos, vib_rs).astype(np.float32)
        except Exception:
            pass

        sp = speed.squeeze().astype(np.float32)
        if sp.ndim > 1:
            sp = sp[:, 0]
        sp_rs = resample_poly(sp, fs_v, max(fs_v, fs_a)) if max(fs_v, fs_a) != fs_v else sp
        if isinstance(force, np.ndarray) and force.ndim == 2 and force.shape[1] >= 3:
            fo_mag = np.linalg.norm(force[:, :3], axis=1).astype(np.float32)
        else:
            fo_mag = np.asarray(force).squeeze().astype(np.float32)
        fo_rs = resample_poly(fo_mag, fs_v, max(fs_v, fs_a)) if max(fs_v, fs_a) != fs_v else fo_mag

        # choose image
        img_path = None
        for base in img_roots:
            cand = _choose_texture_image(base, tex_id)
            if cand is not None:
                img_path = cand
                break
        if img_path is None:
            return {"failed": False, "rows": [], "images_missing": 1, "uv_fallbacks": 0, "total_windows": 0,
                    "patch_sum": 0.0, "patch_sq_sum": 0.0, "patch_count": 0, "speeds": [], "forces": [], "vib_sq_acc": 0.0, "vib_count": 0}
        img_arr = _load_image_gray_arr(img_path)
        if img_arr is None:
            return {"failed": False, "rows": [], "images_missing": 1, "uv_fallbacks": 0, "total_windows": 0,
                    "patch_sum": 0.0, "patch_sq_sum": 0.0, "patch_count": 0, "speeds": [], "forces": [], "vib_sq_acc": 0.0, "vib_count": 0}

        rows: List[Dict[str, Any]] = []
        patch_sum = 0.0; patch_sq_sum = 0.0; patch_count = 0
        speeds_all: List[float] = []; forces_all: List[float] = []
        vib_sq_acc = 0.0; vib_count = 0
        uv_fallbacks = 0; total_windows = 0
        kept = 0
        for wj, (s, e) in enumerate(_window_indices(len(vib_rs), win_v, hop_v)):
            total_windows += 1
            sp_m = float(np.mean(sp_rs[s:e])) if e <= len(sp_rs) else 0.0
            fo_m = float(np.mean(fo_rs[s:e])) if e <= len(fo_rs) else 0.0
            if sp_m <= speed_thresh:
                continue
            if hop_prune > 1:
                if (kept % hop_prune) != 0:
                    kept += 1
                    continue
                kept += 1
            v_win = vib_rs[s:e].astype(np.float32)
            if len(v_win) != win_v:
                continue
            v_win = (v_win - float(np.mean(v_win))).astype(np.float32)
            vib_sq_acc += float(np.sum(np.square(v_win)))
            vib_count += len(v_win)
            pv_win = None
            ps, pe = s - win_v, s
            if ps >= 0:
                pv_win = vib_rs[ps:pe].astype(np.float32)
                pv_win = (pv_win - float(np.mean(pv_win))).astype(np.float32)
            a_win = None
            if aud_rs is not None:
                s_a = int(round(s * (fs_a / fs_v)))
                e_a = s_a + win_a
                if e_a <= len(aud_rs):
                    a_win = aud_rs[s_a:e_a].astype(np.float32)
                    target_rms = 10 ** (-12 / 20)
                    cur = float(np.sqrt(np.mean(np.square(a_win))) + 1e-12)
                    if cur > 1e-9:
                        a_win = (a_win * (target_rms / cur)).astype(np.float32)
                    a_win = np.clip(a_win, -1.0, 1.0)
            u_c, v_c = _pseudo_uv(tex_id, wj); uv_fallbacks += 1
            try:
                p_arr = _crop_from_arr_wrap(img_arr, u_c, v_c, patch)
                patch_sum += float(np.sum(p_arr)); patch_sq_sum += float(np.sum(np.square(p_arr))); patch_count += p_arr.size
            except Exception:
                pass
            speeds_all.append(sp_m); forces_all.append(fo_m)
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
        return {
            "failed": False,
            "rows": rows,
            "patch_sum": patch_sum,
            "patch_sq_sum": patch_sq_sum,
            "patch_count": patch_count,
            "speeds": speeds_all,
            "forces": forces_all,
            "vib_sq_acc": vib_sq_acc,
            "vib_count": vib_count,
            "uv_fallbacks": uv_fallbacks,
            "total_windows": total_windows,
            "images_missing": 0,
        }
    except Exception:
        return {"failed": True}


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
    max_files: int = 0
    max_windows: int = 0
    verbose: int = 1
    progress: int = 1
    workers: int = 0
    hop_prune: int = 1


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

    # Prepare scan roots aligned to your dataset layout
    rows: List[Dict[str, Any]] = []
    uv_fallbacks = 0
    total_windows = 0
    raw = cfg.raw_path
    h5_scan_root = raw / "datah5" if (raw / "datah5").exists() else raw
    h5_list = _scan_havdb_h5(h5_scan_root) if h5_scan_root.exists() else []
    if not h5_list:
        raise FileNotFoundError(f"No .h5 recordings found under {h5_scan_root}")
    # Sort deterministically and optionally limit for debugging
    h5_list = sorted(h5_list, key=lambda r: (str(r.get("user")), str(r.get("texture")), str(r.get("trial"))))
    if cfg.max_files and cfg.max_files > 0:
        h5_list = h5_list[: cfg.max_files]

    t0 = time.time()
    print(f"[preprocess] raw={raw} scan_root={h5_scan_root} files_found={len(h5_list)}", flush=True)
    if cfg.verbose and len(h5_list):
        for ex in h5_list[:3]:
            print(f"  sample file: user={ex['user']} texture={ex['texture']} trial={ex['trial']} path={ex['h5_path']}", flush=True)

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

    # Image roots to search
    candidate_img_roots = [raw / "img" / "img", raw / "img", raw.parent / "img" / "img", raw.parent / "img"]
    common_sr = max(fs_v, fs_a)
    files_processed = 0
    files_failed = 0
    images_missing = 0
    if int(cfg.workers) > 0:
        # Parallel path
        futures = []
        with ProcessPoolExecutor(max_workers=int(cfg.workers)) as pool:
            for rec in h5_list:
                h5_path = rec["h5_path"]
                local_root = h5_path.parents[2] if len(h5_path.parents) >= 2 else raw
                args_map = {
                    "user": rec["user"],
                    "texture": rec["texture"],
                    "trial": rec["trial"],
                    "h5_path": str(rec["h5_path"]),
                    "local_root": str(local_root),
                    "fs_v": fs_v,
                    "fs_a": fs_a,
                    "win_v": win_v,
                    "win_a": win_a,
                    "hop_v": hop_v,
                    "band_lo": float(cfg.band_vib[0]),
                    "band_hi": float(cfg.band_vib[1]),
                    "speed_thresh": float(cfg.speed_thresh),
                    "patch": int(cfg.patch),
                    "hop_prune": int(cfg.hop_prune),
                    "img_roots": [str(p) for p in candidate_img_roots],
                }
                futures.append(pool.submit(_process_record_worker, args_map))
            prog = tqdm(total=len(futures), desc="[preprocess] files", disable=not bool(cfg.progress))
            for fut in as_completed(futures):
                prog.update(1)
                res = fut.result()
                if res.get("failed"):
                    files_failed += 1
                    continue
                files_processed += 1
                images_missing += int(res.get("images_missing", 0))
                uv_fallbacks += int(res.get("uv_fallbacks", 0))
                total_windows += int(res.get("total_windows", 0))
                patch_sum += float(res.get("patch_sum", 0.0))
                patch_sq_sum += float(res.get("patch_sq_sum", 0.0))
                patch_count += int(res.get("patch_count", 0))
                speeds_all.extend(res.get("speeds", []))
                forces_all.extend(res.get("forces", []))
                rws = res.get("rows", [])
                if cfg.max_windows and cfg.max_windows > 0:
                    remaining = cfg.max_windows - len(rows)
                    if remaining > 0:
                        rows.extend(rws[:remaining])
                else:
                    rows.extend(rws)
                if cfg.max_windows and len(rows) >= cfg.max_windows:
                    break
            prog.close()
        # fall-through to writing
    else:
        iterator = tqdm(h5_list, total=len(h5_list), desc="[preprocess] files", disable=not bool(cfg.progress))
        for rec in iterator:
            file_t0 = time.time()
            user = rec["user"]; tex_id = rec["texture"]; trial = rec["trial"]
            h5_path = rec["h5_path"]
            # base path containing user folders
            local_root = h5_path.parents[2] if len(h5_path.parents) >= 2 else raw
            try:
                data, _ = hav_load_data(str(local_root), user, tex_id, trial, sampling_rate=common_sr)
            except Exception as ex:
                warnings.warn(f"HAVdb load failed for {h5_path}: {ex}")
                files_failed += 1
                continue
            common_time, position, speed, direction, vibration, audio, force, torque = data
            vib_ch = vibration[:, 0].astype(np.float32)
            aud_ch = audio[:, 0].astype(np.float32) if isinstance(audio, np.ndarray) and audio.size else None
            # resample to target rates
            vib_rs = resample_poly(vib_ch, fs_v, common_sr) if common_sr != fs_v else vib_ch
            if aud_ch is not None:
                aud_rs = resample_poly(aud_ch, fs_a, common_sr) if common_sr != fs_a else aud_ch
            else:
                aud_rs = None
            # vib preprocessing
            vib_rs = vib_rs - float(np.mean(vib_rs))
            try:
                vib_rs = sosfiltfilt(sos, vib_rs).astype(np.float32)
            except Exception:
                pass
            # states
            sp = speed.squeeze().astype(np.float32)
            if sp.ndim > 1:
                sp = sp[:, 0]
            sp_rs = resample_poly(sp, fs_v, common_sr) if common_sr != fs_v else sp
            if isinstance(force, np.ndarray) and force.ndim == 2 and force.shape[1] >= 3:
                fo_mag = np.linalg.norm(force[:, :3], axis=1).astype(np.float32)
            else:
                fo_mag = np.asarray(force).squeeze().astype(np.float32)
            fo_rs = resample_poly(fo_mag, fs_v, common_sr) if common_sr != fs_v else fo_mag
            # select texture image
            img_path = None
            for base in candidate_img_roots:
                img_path = _choose_texture_image(base, tex_id)
                if img_path is not None:
                    break
            if img_path is None:
                warnings.warn(f"No texture image found for {tex_id} near {raw}")
                images_missing += 1
                continue
            # cache image array once per file
            img_arr = _load_image_gray_arr(img_path)
            if img_arr is None:
                warnings.warn(f"Failed to load image {img_path} for texture {tex_id}")
                images_missing += 1
                continue
            # windows over vib
            win_idx = _window_indices(len(vib_rs), win_v, hop_v)
            kept_in_file = 0
            for wj, (s, e) in enumerate(win_idx):
                total_windows += 1
                sp_m = float(np.mean(sp_rs[s:e])) if e <= len(sp_rs) else 0.0
                fo_m = float(np.mean(fo_rs[s:e])) if e <= len(fo_rs) else 0.0
                if sp_m <= float(cfg.speed_thresh):
                    continue
                # optional density pruning: keep every k-th kept window
                if int(cfg.hop_prune) > 1:
                    if (kept_in_file % int(cfg.hop_prune)) != 0:
                        kept_in_file += 1
                        continue
                    kept_in_file += 1
                v_win = vib_rs[s:e].astype(np.float32)
                if len(v_win) != win_v:
                    continue
                v_win = (v_win - float(np.mean(v_win))).astype(np.float32)
                vib_sq_acc += float(np.sum(np.square(v_win)))
                vib_count += len(v_win)
                # previous
                pv_win = None
                ps, pe = s - win_v, s
                if ps >= 0:
                    pv_win = vib_rs[ps:pe].astype(np.float32)
                    pv_win = (pv_win - float(np.mean(pv_win))).astype(np.float32)
                # audio align
                a_win = None
                if aud_rs is not None:
                    s_a = int(round(s * (fs_a / fs_v)))
                    e_a = s_a + win_a
                    if e_a <= len(aud_rs):
                        a_win = aud_rs[s_a:e_a].astype(np.float32)
                        # target RMS -12 dBFS
                        target_rms = 10 ** (-12 / 20)
                        cur = _rms(a_win)
                        if cur > 1e-9:
                            a_win = (a_win * (target_rms / cur)).astype(np.float32)
                        a_win = np.clip(a_win, -1.0, 1.0)
                # pseudo UV
                u_c, v_c = _pseudo_uv(tex_id, wj)
                uv_fallbacks += 1
                # patch stats using cached image array
                try:
                    patch_arr = _crop_from_arr_wrap(img_arr, u_c, v_c, cfg.patch)
                    patch_sum += float(np.sum(patch_arr))
                    patch_sq_sum += float(np.sum(np.square(patch_arr)))
                    patch_count += patch_arr.size
                except Exception:
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
                kept_in_file += 1
                if cfg.max_windows and len(rows) >= cfg.max_windows:
                    print(f"[preprocess] Reached max_windows={cfg.max_windows}; early stopping.", flush=True)
                    break
            files_processed += 1
            if cfg.verbose:
                dt = time.time() - file_t0
                print(f"[preprocess] done user={user} texture={tex_id} trial={trial} kept={kept_in_file} dt={dt:.2f}s", flush=True)
            if cfg.max_windows and len(rows) >= cfg.max_windows:
                break

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
    summary = {
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
        "files_processed": files_processed,
        "files_failed": files_failed,
        "images_missing": images_missing,
        "scan_root": str(h5_scan_root),
        "elapsed_sec": float(time.time() - t0),
    }
    print(json.dumps(summary, indent=2), flush=True)
    try:
        with open(Path(cfg.debug_dir) / "preprocess_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass

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
    p.add_argument("--max_files", type=int, default=0, help="Limit number of H5 files for debugging")
    p.add_argument("--max_windows", type=int, default=0, help="Limit number of windows for debugging")
    p.add_argument("--verbose", type=int, default=1, help="Verbosity level (0=silent,1=info)")
    p.add_argument("--progress", type=int, default=1, help="Show progress bar (1=yes,0=no)")
    p.add_argument("--workers", type=int, default=0, help="[experimental] parallelize per-file (0=off)")
    p.add_argument("--hop_prune", type=int, default=1, help="Keep every k-th kept window to cap density")
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
        max_files=int(args.max_files),
        max_windows=int(args.max_windows),
        verbose=int(args.verbose),
        progress=int(args.progress),
        workers=int(args.workers),
        hop_prune=int(args.hop_prune),
    )
    sys.exit(preprocess(cfg))


if __name__ == "__main__":
    main()
