from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------ Torch ops ------------------------------

def rmse_torch(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(y_hat, y))


def lsd_torch(y_hat: torch.Tensor, y: torch.Tensor, n_fft: int = 256, hop: int = 64, eps: float = 1e-8) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=y.device)
    def spec_db(x):
        X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        mag = torch.abs(X) + eps
        db = 20.0 * torch.log10(mag)
        return db.transpose(1, 2)
    A = spec_db(y_hat)
    B = spec_db(y)
    return torch.sqrt(torch.mean((A - B) ** 2))


def spectral_centroid_torch(y: torch.Tensor, sr: int = 1000, n_fft: int = 256, hop: int = 64, eps: float = 1e-8) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=y.device)
    X = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
    mag = torch.abs(X) + eps
    freqs = torch.linspace(0, sr / 2, steps=mag.shape[1], device=y.device).view(1, -1, 1)
    num = torch.sum(freqs * mag, dim=1)
    den = torch.sum(mag, dim=1) + eps
    centroid = num / den
    return torch.mean(centroid, dim=1)


def spectral_centroid_error_torch(y_hat: torch.Tensor, y: torch.Tensor, sr: int = 1000) -> torch.Tensor:
    c1 = spectral_centroid_torch(y_hat, sr=sr)
    c2 = spectral_centroid_torch(y, sr=sr)
    return torch.mean(torch.abs(c1 - c2))


def envelope_corr_torch(y_hat: torch.Tensor, y: torch.Tensor, sr: int = 1000, win_ms: float = 25.0) -> torch.Tensor:
    from .losses import _env
    e1 = _env(y_hat, sr)
    e2 = _env(y, sr)
    def _norm(z):
        z = z - z.mean(dim=1, keepdim=True)
        z = z / (z.std(dim=1, keepdim=True) + 1e-6)
        return z
    e1 = _norm(e1)
    e2 = _norm(e2)
    return torch.mean((e1 * e2).mean(dim=1))


# ------------------------------ NumPy ops ------------------------------

def rmse_np(y_hat: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_hat.astype(np.float64) - y.astype(np.float64)) ** 2)))


def lsd_np(y_hat: np.ndarray, y: np.ndarray, n_fft: int = 256, hop: int = 64, eps: float = 1e-8) -> float:
    import numpy.fft as nfft
    def frames(a, win, hop):
        return np.stack([a[i:i+win] for i in range(0, len(a) - win + 1, hop)], axis=0)
    win = n_fft
    Xh = np.abs(np.fft.rfft(frames(y_hat, win, hop), n=win)) + eps
    X = np.abs(np.fft.rfft(frames(y, win, hop), n=win)) + eps
    Ah = 20.0 * np.log10(Xh)
    A = 20.0 * np.log10(X)
    return float(np.sqrt(np.mean((Ah - A) ** 2)))


def envelope_corr_np(y_hat: np.ndarray, y: np.ndarray, sr: int = 1000, win_ms: float = 25.0) -> float:
    import scipy.signal
    win = int(round(sr * (win_ms / 1000.0)))
    win = max(3, win | 1)
    w = np.hanning(win)
    w /= np.sum(w)
    def env(a):
        a = np.abs(a)
        a = np.pad(a, (win // 2, win // 2), mode="reflect")
        return scipy.signal.convolve(a, w, mode="valid")
    e1 = env(y_hat)
    e2 = env(y)
    if np.std(e1) < 1e-6 or np.std(e2) < 1e-6:
        return 0.0
    return float(np.corrcoef(e1, e2)[0, 1])

