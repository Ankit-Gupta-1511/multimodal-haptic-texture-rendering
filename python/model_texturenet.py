from __future__ import annotations

from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from .utils import kaiming_init


class CNNEncoder(nn.Module):
    """3 conv blocks with stride-2, BN+GELU, then Linear -> 256-D."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        )
        # 96x96 -> 12x12 with 3 stride-2 blocks; flatten
        self.fc = nn.Linear(128 * 12 * 12, out_dim)
        kaiming_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h)


class StateMLP(nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, out_dim), nn.GELU()
        )
        kaiming_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrevVibEncoder(nn.Module):
    """Encode prev_vib (B, 100) into temporal features (B, 64, T)."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, out_dim, kernel_size=5, padding=2), nn.BatchNorm1d(out_dim), nn.GELU(),
        )
        kaiming_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(1))  # (B, C, T)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.pad = (dilation * 2, 0)  # causal for k=3
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)
        kaiming_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x, self.pad)
        y = self.conv1(y)
        y = self.norm1(F.gelu(y))
        y = self.drop(y)
        y = F.pad(y, self.pad)
        y = self.conv2(y)
        y = self.norm2(F.gelu(y))
        y = self.drop(y)
        return x + y


class VibDecoderTCN(nn.Module):
    def __init__(self, cond_dim: int, channels: int = 64, blocks: int = 4, out_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.out_len = out_len
        self.in_proj = nn.Conv1d(cond_dim, channels, kernel_size=1)
        self.tcn = nn.Sequential(*[TCNBlock(channels, dilation=2 ** i, dropout=dropout) for i in range(blocks)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Conv1d(channels, 1, kernel_size=1)
        kaiming_init(self)

    def forward(self, cond_seq: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(cond_seq)
        h = self.tcn(h)
        h = self.drop(h)
        y = self.head(h)
        y = y[..., : self.out_len]
        return y.squeeze(1)


class TextureNet(nn.Module):
    def __init__(self, mode: str = "baseline_vibro", latent: int = 128, tcn_blocks: int = 4, tcn_growth: int = 64, use_prev: bool = False, fs_aud: int = 8000):
        super().__init__()
        assert mode in {"baseline_vibro", "low_delay_vibro", "multitask_av"}
        self.mode = mode
        self.use_prev = use_prev
        self.enc_img = CNNEncoder(out_dim=256)
        self.enc_state = StateMLP(in_dim=2, out_dim=32)
        self.enc_prev = PrevVibEncoder(out_dim=64) if use_prev else None
        fuse_in = 256 + 32 + (64 if use_prev else 0)
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, latent), nn.GELU(),
        )

        vib_cond_dim = latent + (64 if use_prev else 0)
        self.dec_vib = VibDecoderTCN(cond_dim=vib_cond_dim, channels=tcn_growth, blocks=tcn_blocks, out_len=100, dropout=0.1)
        if self.mode == "multitask_av":
            self.dec_aud = VibDecoderTCN(cond_dim=vib_cond_dim, channels=tcn_growth, blocks=tcn_blocks, out_len=800 if fs_aud == 8000 else 1600, dropout=0.1)
        else:
            self.dec_aud = None
        kaiming_init(self)

    def _fuse_latent(self, patch: torch.Tensor, state: torch.Tensor, prev_vib: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        img_h = self.enc_img(patch)
        st_h = self.enc_state(state)
        if self.use_prev and prev_vib is not None:
            pv_h = self.enc_prev(prev_vib)
            pv_pool = torch.mean(pv_h, dim=2)
            fused = torch.cat([img_h, st_h, pv_pool], dim=1)
        else:
            pv_h = None
            fused = torch.cat([img_h, st_h], dim=1)
        z = self.fuse(fused)
        return z, pv_h

    @torch.no_grad()
    def inference_chunk(self, patch: torch.Tensor, state: torch.Tensor, prev_vib: Optional[torch.Tensor] = None):
        self.eval()
        return self.forward(patch, state, prev_vib)

    def forward(self, patch: torch.Tensor, state: torch.Tensor, prev_vib: Optional[torch.Tensor] = None):
        z, pv_h = self._fuse_latent(patch, state, prev_vib)
        z_vib = z.unsqueeze(-1).repeat(1, 1, 100)
        if self.use_prev and pv_h is not None:
            cond_vib = torch.cat([z_vib, pv_h], dim=1)
        else:
            cond_vib = z_vib
        vib = self.dec_vib(cond_vib)

        if self.mode == "multitask_av":
            z_aud = z.unsqueeze(-1).repeat(1, 1, 800)
            if self.use_prev and pv_h is not None:
                pv_up = F.interpolate(pv_h, size=800, mode="linear", align_corners=False)
                cond_aud = torch.cat([z_aud, pv_up], dim=1)
            else:
                cond_aud = z_aud
            aud = self.dec_aud(cond_aud)
            return vib, aud
        return vib


if __name__ == "__main__":
    # Smoke test: create models in each mode, run dummy data, print shapes
    B = 4
    patch = torch.randn(B, 1, 96, 96)
    state = torch.rand(B, 2)
    prev = torch.randn(B, 100)

    for mode in ["baseline_vibro", "low_delay_vibro", "multitask_av"]:
        use_prev = mode == "low_delay_vibro"
        net = TextureNet(mode=mode, latent=128, tcn_blocks=4, tcn_growth=64, use_prev=use_prev).eval()
        with torch.no_grad():
            if use_prev:
                out = net(patch, state, prev)
            else:
                out = net(patch, state, prev)
        if mode == "multitask_av":
            v, a = out
            print({"mode": mode, "vib": tuple(v.shape), "audio": tuple(a.shape)})
        else:
            print({"mode": mode, "vib": tuple(out.shape)})

