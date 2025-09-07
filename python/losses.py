from __future__ import annotations

from typing import Tuple
import torch
import torch.nn.functional as F


def time_mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(y_hat, y)


def _stft_mag(y: torch.Tensor, n_fft: int = 256, hop: int = 64, win_length: int = 256) -> torch.Tensor:
    window = torch.hann_window(win_length, device=y.device)
    Y = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=win_length, window=window, return_complex=True)
    mag = torch.abs(Y) + 1e-8
    return mag


def stft_loss(y_hat: torch.Tensor, y: torch.Tensor, n_fft: int = 256, hop: int = 64, win_length: int = 256, log_mag: bool = True) -> torch.Tensor:
    """Multi-resolution-style basic STFT loss on magnitude spectra."""
    m_hat = _stft_mag(y_hat, n_fft=n_fft, hop=hop, win_length=win_length)
    m = _stft_mag(y, n_fft=n_fft, hop=hop, win_length=win_length)
    if log_mag:
        m_hat = torch.log(m_hat)
        m = torch.log(m)
    return F.l1_loss(m_hat, m)


def mel_loss_audio(y_hat: torch.Tensor, y: torch.Tensor, sr: int = 8000, n_mels: int = 64, n_fft: int = 512, hop: int = 128) -> torch.Tensor:
    """Mel-spectrogram L1 loss on log-magnitude mels."""
    # Build linear spectrograms
    window = torch.hann_window(n_fft, device=y.device)
    def spec(x):
        X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        return torch.abs(X).transpose(1, 2)  # (B, T, F)

    S_hat = spec(y_hat)
    S = spec(y)

    # Build mel filter using librosa and cache on CPU, then move
    import librosa
    import numpy as np
    M = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels).astype("float32")  # (n_mels, n_freq)
    M_t = torch.from_numpy(M).to(S.device)

    mel_hat = torch.matmul(S_hat, M_t.T)
    mel = torch.matmul(S, M_t.T)
    mel_hat = torch.log(mel_hat + 1e-8)
    mel = torch.log(mel + 1e-8)
    return F.l1_loss(mel_hat, mel)


def _env(x: torch.Tensor, sr: int, win_ms: float = 25.0) -> torch.Tensor:
    """Approximate amplitude envelope via rectification + lowpass Hann smoothing."""
    x = torch.abs(x)
    win = int(round(sr * (win_ms / 1000.0)))
    win = max(3, win | 1)  # odd
    w = torch.hann_window(win, device=x.device)
    w = w / w.sum()
    pad = win // 2
    x = F.pad(x.unsqueeze(1), (pad, pad), mode="reflect")
    y = F.conv1d(x, w.view(1, 1, -1))
    return y.squeeze(1)


def envelope_sync_loss(vib_hat: torch.Tensor, aud_hat: torch.Tensor, vib_sr: int = 1000, aud_sr: int = 8000) -> torch.Tensor:
    """Synchronize envelopes between vib and audio by aligning to vib length and using L1.

    Downsample audio envelope to vib resolution using linear interpolation.
    """
    B = vib_hat.shape[0]
    env_v = _env(vib_hat, vib_sr)  # (B, T_v)
    env_a = _env(aud_hat, aud_sr)  # (B, T_a)
    # resample env_a to T_v by interpolation
    T_v = env_v.shape[1]
    T_a = env_a.shape[1]
    env_a_rs = F.interpolate(env_a.unsqueeze(1), size=T_v, mode="linear", align_corners=False).squeeze(1)
    # normalize per-sample to unit energy to avoid trivial scaling issues
    def _norm(z):
        return z / (z.norm(p=2, dim=1, keepdim=True) + 1e-6)
    env_v = _norm(env_v)
    env_a_rs = _norm(env_a_rs)
    return F.l1_loss(env_v, env_a_rs)


