"""Spatio-Temporal Discriminators for NWP-NowcastNet on the BJ dataset.

Includes:
- SpatialDiscriminator: 2D spectral-normalized multi-scale PatchGAN conditioned on radar/evo/NWP.
- TemporalDiscriminator: 3D spectral-normalized Conv discriminator checking temporal evolution continuity.
- SpatioTemporalDiscriminator: unified wrapper returning (spatial_logits, temporal_logits).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _sn_conv2d(in_c: int, out_c: int, k: int = 4, s: int = 2, p: int = 1) -> nn.Module:
    return spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p))


def _sn_conv3d(in_c: int, out_c: int, k=(3, 4, 4), s=(1, 2, 2), p=(1, 1, 1)) -> nn.Module:
    return spectral_norm(nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=p))


class SpatialDiscriminator(nn.Module):
    """2D Spectral-Normalized Patch Discriminator operating on per-frame predictions + conditions."""

    def __init__(self, in_channels: int, base_c: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            _sn_conv2d(in_channels, base_c, k=4, s=2, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv2d(base_c, base_c * 2, k=4, s=2, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv2d(base_c * 2, base_c * 4, k=4, s=2, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv2d(base_c * 4, base_c * 8, k=4, s=2, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_c * 8, 1, kernel_size=3, stride=1, padding=1)),
        )

    def forward(self, frames: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            frames: (B, T, 1, H, W) precipitation field (raw or floored)
            cond:   (B, T, C_cond, H, W) optional conditioning frames (e.g., evo_rain + radar/NWP summary)
        """
        b, t, c, h, w = frames.shape
        x = torch.log1p(torch.clamp(frames, min=0.0))
        if cond is not None:
            x = torch.cat([x, cond], dim=2)
        x = x.reshape(b * t, -1, h, w)
        return self.net(x)


class TemporalDiscriminator(nn.Module):
    """3D Spectral-Normalized Discriminator checking spatiotemporal consistency across lead times."""

    def __init__(self, in_channels: int, base_c: int = 32):
        super().__init__()
        # Downsample spatially aggressively first, then jointly in time & space
        self.net = nn.Sequential(
            _sn_conv3d(in_channels, base_c, k=(3, 4, 4), s=(1, 2, 2), p=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv3d(base_c, base_c * 2, k=(3, 4, 4), s=(2, 2, 2), p=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv3d(base_c * 2, base_c * 4, k=(3, 4, 4), s=(2, 2, 2), p=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv3d(base_c * 4, base_c * 8, k=(3, 4, 4), s=(1, 2, 2), p=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(base_c * 8, 1, kernel_size=3, stride=1, padding=1)),
        )

    def forward(self, frames: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            frames: (B, T, 1, H, W)
            cond:   (B, T, C_cond, H, W) optional
        """
        x = torch.log1p(torch.clamp(frames, min=0.0))
        if cond is not None:
            x = torch.cat([x, cond], dim=2)
        # (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.net(x)


class SpatioTemporalDiscriminator(nn.Module):
    """Combined 2D Spatial + 3D Temporal conditional discriminator."""

    def __init__(self, cond_channels: int = 2, base_c: int = 32):
        super().__init__()
        total_in = 1 + cond_channels
        self.spatial_d = SpatialDiscriminator(in_channels=total_in, base_c=base_c)
        self.temporal_d = TemporalDiscriminator(in_channels=total_in, base_c=base_c)

    @staticmethod
    def build_condition(
        evo_rain: torch.Tensor,
        x_radar: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build lightweight per-step spatial condition: [log1p(evo_rain), last_radar_CR]."""
        cond_list = [torch.log1p(torch.clamp(evo_rain.detach(), min=0.0))]
        if x_radar is not None:
            # Index 6 in RADAR_PRODS is CR; repeat across T_out
            t_out = evo_rain.shape[1]
            last_cr = x_radar[:, -1:, 6:7].expand(-1, t_out, -1, -1, -1).detach()
            cond_list.append(last_cr)
        return torch.cat(cond_list, dim=2)

    def forward(
        self,
        frames: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.spatial_d(frames, cond), self.temporal_d(frames, cond)
