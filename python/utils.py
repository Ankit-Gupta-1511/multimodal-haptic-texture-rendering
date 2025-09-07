import os
import io
import json
import math
import time
import atexit
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

import psutil
import soundfile as sf
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def seed_all(seed: int = 42) -> None:
    """Set seeds across random, numpy, torch (CPU/CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def device_report() -> str:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lines = [f"device={dev}"]
    if dev == "cuda":
        lines.append(f"cuda_version={torch.version.cuda}")
        lines.append(f"cudnn={torch.backends.cudnn.version()}")
        lines.append(f"cudnn_enabled={torch.backends.cudnn.enabled}")
        lines.append(f"cudnn_deterministic={torch.backends.cudnn.deterministic}")
    lines.append(f"conda_env={os.environ.get('CONDA_DEFAULT_ENV','n/a')}")
    lines.append(f"torch={torch.__version__}")
    return ", ".join(lines)


def memory_info() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    gpu = {}
    if torch.cuda.is_available():
        gpu = {
            "gpu_mem_alloc": float(torch.cuda.memory_allocated() / (1024**2)),
            "gpu_mem_reserved": float(torch.cuda.memory_reserved() / (1024**2)),
        }
    return {
        "rss_mb": float(psutil.Process().memory_info().rss / (1024**2)),
        "vmem_percent": vm.percent,
        **gpu,
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_tb_writer(log_dir: Path) -> SummaryWriter:
    ensure_dir(log_dir)
    return SummaryWriter(log_dir=str(log_dir))


def write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    ensure_dir(path.parent)
    y = np.asarray(y, dtype=np.float32)
    y = np.clip(y, -1.0, 1.0)
    sf.write(str(path), y, sr, subtype="PCM_16")


def plot_time_series(path: Path, ys: Dict[str, np.ndarray], sr: int, title: str = "") -> None:
    ensure_dir(path.parent)
    t = np.arange(len(next(iter(ys.values())))) / sr
    plt.figure(figsize=(8, 3))
    for k, v in ys.items():
        plt.plot(t, v, label=k)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(path))
    plt.close()


def plot_hist(path: Path, arr: np.ndarray, bins: int = 50, title: str = "") -> None:
    ensure_dir(path.parent)
    plt.figure(figsize=(4, 3))
    plt.hist(arr, bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(str(path))
    plt.close()


def plot_psd(path: Path, y: np.ndarray, sr: int, title: str = "") -> None:
    from scipy.signal import welch

    ensure_dir(path.parent)
    f, Pxx = welch(y, fs=sr, nperseg=min(512, len(y)))
    plt.figure(figsize=(6, 3))
    plt.semilogy(f, Pxx + 1e-12)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(str(path))
    plt.close()


def log_nan_inf_tensors(batch: Dict[str, torch.Tensor]) -> Dict[str, bool]:
    flags = {}
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            continue
        flags[k] = bool(torch.isnan(v).any() or torch.isinf(v).any())
    return flags


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))


@dataclass
class WarmupCosine:
    optimizer: torch.optim.Optimizer
    warmup_steps: int
    total_steps: int
    min_lr_scale: float = 0.0
    _last_step: int = 0

    def step(self):
        self._last_step += 1
        t = self._last_step
        for group in self.optimizer.param_groups:
            base_lr = group.get("initial_lr", group["lr"])  # support set before
            if t < self.warmup_steps:
                lr = base_lr * float(t) / max(1, self.warmup_steps)
            else:
                progress = (t - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                cos_val = 0.5 * (1 + math.cos(math.pi * progress))
                lr = base_lr * (self.min_lr_scale + (1 - self.min_lr_scale) * cos_val)
            group["lr"] = lr


def kaiming_init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def safe_mean_std_from_yaml(path: Path, default_mean: float = 0.5, default_std: float = 0.25) -> Tuple[float, float]:
    if path.exists():
        try:
            dat = load_yaml(path)
            return float(dat.get("mean", default_mean)), float(dat.get("std", default_std))
        except Exception:
            pass
    return default_mean, default_std


def amp_autocast(enabled: bool):
    return torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"), enabled=enabled)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def numpyify(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def set_deterministic_algorithms(flag: bool = True) -> None:
    torch.use_deterministic_algorithms(flag, warn_only=True)


def make_grid_save(path: Path, images: torch.Tensor, nrow: int = 8, normalize: bool = True):
    from torchvision.utils import make_grid
    ensure_dir(path.parent)
    grid = make_grid(images, nrow=nrow, normalize=normalize, scale_each=True)
    nd = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(nd.squeeze(), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(path))
    plt.close()


