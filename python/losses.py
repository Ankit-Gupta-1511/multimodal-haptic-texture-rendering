from __future__ import annotations

from typing import Tuple
import math
import torch
import torch.nn.functional as F


def time_mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(y_hat, y)


def _stft_mag(y: torch.Tensor, n_fft: int = 256, hop: int = 64, win_length: int = 256) -> torch.Tensor:
    """Adaptive STFT magnitude to handle short signals.

    Clamps FFT and window sizes to fit the time length to avoid padding errors
    on very short inputs (e.g., vib length=100 samples).
    """
    T = y.shape[-1]
    # choose effective sizes compatible with T
    n_fft_eff = min(n_fft, max(4, 2 ** int(math.floor(math.log2(max(4, T))))))
    win_eff = min(win_length, n_fft_eff, T)
    hop_eff = max(1, min(hop, max(1, win_eff // 2)))
    center = T >= win_eff
    window = torch.hann_window(win_eff, device=y.device)
    Y = torch.stft(y, n_fft=n_fft_eff, hop_length=hop_eff, win_length=win_eff, window=window, return_complex=True, center=center)
    mag = torch.abs(Y) + 1e-8
    return mag


def stft_loss(y_hat: torch.Tensor, y: torch.Tensor, n_fft: int = 256, hop: int = 64, win_length: int = 256, log_mag: bool = True) -> torch.Tensor:
    """STFT magnitude loss; log magnitude by default for robustness."""
    m_hat = _stft_mag(y_hat, n_fft=n_fft, hop=hop, win_length=win_length)
    m = _stft_mag(y, n_fft=n_fft, hop=hop, win_length=win_length)
    if log_mag:
        m_hat = torch.log(m_hat)
        m = torch.log(m)
    return F.l1_loss(m_hat, m)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: Optional[float] = None) -> torch.Tensor:
    """Create a mel filterbank matrix (n_mels, n_fft//2+1) without librosa.
    Returns torch float32 on CPU; caller moves to device.
    """
    if fmax is None:
        fmax = sr / 2
    def hz_to_mel(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)
    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)
    m_min = hz_to_mel(fmin)
    m_max = hz_to_mel(fmax)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    bins = torch.floor((n_fft + 1) * f_pts / sr).long()
    fb = torch.zeros(n_mels, n_fft // 2 + 1, dtype=torch.float32)
    for i in range(n_mels):
        l, c, r = bins[i].item(), bins[i+1].item(), bins[i+2].item()
        if c == l: c += 1
        if r == c: r += 1
        fb[i, l:c] = torch.linspace(0, 1, c - l, dtype=torch.float32)
        fb[i, c:r] = torch.linspace(1, 0, r - c, dtype=torch.float32)
    return fb


def mel_loss_audio(y_hat: torch.Tensor, y: torch.Tensor, sr: int = 8000, n_mels: int = 64, n_fft: int = 512, hop: int = 128) -> torch.Tensor:
    """Mel-spectrogram L1 on log-magnitude mels (no external deps)."""
    window = torch.hann_window(n_fft, device=y.device)
    def spec(x):
        X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        return torch.abs(X).transpose(1, 2)  # (B, T, F)
    S_hat = spec(y_hat)
    S = spec(y)
    M = _mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels).to(S.device)
    mel_hat = S_hat @ M.T
    mel = S @ M.T
    mel_hat = torch.log(mel_hat + 1e-8)
    mel = torch.log(mel + 1e-8)
    return F.l1_loss(mel_hat, mel)


def _env(x: torch.Tensor, sr: int, win_ms: float = 25.0) -> torch.Tensor:
    """Rectify + Hann lowpass envelope."""
    x = torch.abs(x)
    win = int(round(sr * (win_ms / 1000.0)))
    win = max(3, win | 1)
    w = torch.hann_window(win, device=x.device)
    w = w / w.sum()
    pad = win // 2
    x = F.pad(x.unsqueeze(1), (pad, pad), mode="reflect")
    y = F.conv1d(x, w.view(1, 1, -1))
    return y.squeeze(1)


def envelope_sync_loss(vib_hat: torch.Tensor, aud_hat: torch.Tensor, vib_sr: int = 1000, aud_sr: int = 8000) -> torch.Tensor:
    """1 - Pearson correlation between vib and audio envelopes.
    Audio envelope is interpolated to vib length.
    """
    ev = _env(vib_hat, vib_sr)
    ea = _env(aud_hat, aud_sr)
    T = ev.shape[1]
    ea = F.interpolate(ea.unsqueeze(1), size=T, mode="linear", align_corners=False).squeeze(1)
    # normalize
    def _norm(z):
        z = z - z.mean(dim=1, keepdim=True)
        z = z / (z.std(dim=1, keepdim=True) + 1e-6)
        return z
    ev = _norm(ev)
    ea = _norm(ea)
    corr = torch.mean((ev * ea).mean(dim=1))
    return 1.0 - corr
