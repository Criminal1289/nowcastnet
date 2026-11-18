import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from nowcasting.data_provider.bj_transforms import NWP_MOTION_INDICES
from nowcasting.layers.evolution.module import DoubleConv, Down, OutConv, Up
from nowcasting.layers.generation.generative_network import Generative_Decoder, Generative_Encoder
from nowcasting.layers.generation.noise_projector import Noise_Projector
from nowcasting.layers.utils import warp


def make_coord_grid(b: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create pixel coordinate grid (B, 2, H, W) on target device without .cuda() hardcoding."""
    xx = torch.arange(0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    yy = torch.arange(0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    return torch.cat((xx, yy), dim=1)


class SpatioTemporalChannelBias(nn.Module):
    """Factorized Spatio-Temporal & Channel Bias using canonical BJ coordinates + lead-time embeddings.

    Works identically across training patches and full 301x301 (320x320 padded) inference.
    """

    def __init__(self, num_channels: int, hidden_dim: int = 16):
        super().__init__()
        self.num_channels = num_channels
        # 6 spatial features: y, x, sin(pi*y), cos(pi*y), sin(pi*x), cos(pi*x)
        self.spatial_mlp = nn.Sequential(
            nn.Conv2d(6, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, num_channels, kernel_size=1),
        )
        # Zero-init final projection so baseline behavior starts unperturbed
        nn.init.zeros_(self.spatial_mlp[-1].weight)
        nn.init.zeros_(self.spatial_mlp[-1].bias)
        self.channel_time_bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor, coord_yx: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, h, w = x.shape
        if coord_yx is None:
            gy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
            gx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
            yy, xx = torch.meshgrid(gy, gx, indexing="ij")
            coord = torch.stack([yy, xx], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        else:
            if coord_yx.shape[-2:] != (h, w):
                coord = F.interpolate(coord_yx.to(dtype=x.dtype), size=(h, w), mode="bilinear", align_corners=True)
            else:
                coord = coord_yx.to(dtype=x.dtype)

        y, x_c = coord[:, 0:1], coord[:, 1:2]
        feats = torch.cat([
            y, x_c,
            torch.sin(math.pi * y), torch.cos(math.pi * y),
            torch.sin(math.pi * x_c), torch.cos(math.pi * x_c),
        ], dim=1)
        return x + self.spatial_mlp(feats) + self.channel_time_bias


class ConvGRUCell(nn.Module):
    """Lightweight 2D ConvGRU cell for radar temporal sequence modeling."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv_gates = nn.Conv2d(in_channels + hidden_channels, 2 * hidden_channels, kernel_size, padding=pad)
        self.conv_cand = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        combined = torch.cat([x, h], dim=1)
        z, r = torch.chunk(torch.sigmoid(self.conv_gates(combined)), 2, dim=1)
        cand = torch.tanh(self.conv_cand(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * cand


class RadarTemporalEncoder(nn.Module):
    """Encodes multi-product radar history (B, T_in, 10, H, W) preserving temporal dynamics."""

    def __init__(self, radar_ch: int = 10, hidden_ch: int = 24, out_frames_ch: int = 22):
        super().__init__()
        self.frame_embed = nn.Sequential(
            nn.Conv2d(radar_ch, hidden_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_ch),
            nn.GELU(),
        )
        self.gru = ConvGRUCell(hidden_ch, hidden_ch, kernel_size=3)
        # Fuse final hidden state + temporal difference trend + per-step CR history
        self.proj_out = nn.Sequential(
            nn.Conv2d(hidden_ch * 2, out_frames_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x_radar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (evolution_input_feats (B, C_out, H, W), temporal_summary (B, H_c, H, W))."""
        b, t, c, h, w = x_radar.shape
        embeds = self.frame_embed(x_radar.reshape(b * t, c, h, w)).reshape(b, t, -1, h, w)
        state = None
        for step in range(t):
            state = self.gru(embeds[:, step], state)
        # Recent trend: difference between last step and mean of earlier steps
        trend = embeds[:, -1] - embeds[:, :-1].mean(dim=1)
        fused = self.proj_out(torch.cat([state, trend], dim=1))
        return fused, state


class RadarTemporalRefiner(nn.Module):
    """Lightweight (2+1)D temporal smoothing & evolution refiner on the T_out sequence."""

    def __init__(self, channels: int, pred_length: int):
        super().__init__()
        self.pred_length = pred_length
        # Depthwise temporal conv over T_out frames (treats space as batch, channels=1 per lead channel? No: 3D conv over (1, T, H, W))
        self.temporal_conv = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        nn.init.zeros_(self.temporal_conv.weight)
        nn.init.zeros_(self.temporal_conv.bias)

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """pred: (B, T_out, 1, H, W)"""
        # (B, 1, T_out, H, W)
        x = pred.permute(0, 2, 1, 3, 4)
        return pred + self.temporal_conv(x).permute(0, 2, 1, 3, 4)


class NWPMotionGuidance(nn.Module):
    """Lightweight NWP dynamic field (WS + DVG) motion guidance without heavy cross-attention."""

    def __init__(self, n_dyn_vars: int = 6, pred_length: int = 10, hidden_c: int = 24):
        super().__init__()
        self.pred_length = pred_length
        # Takes analysis + forecast dynamic vars (2 * 6 = 12 channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(n_dyn_vars * 2, hidden_c, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_c),
            nn.GELU(),
            nn.Conv2d(hidden_c, hidden_c, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # Residual motion delta (u, v) and blending gate alpha per lead step
        self.motion_head = nn.Conv2d(hidden_c, pred_length * 2, kernel_size=3, padding=1)
        self.gate_head = nn.Conv2d(hidden_c, pred_length, kernel_size=1)
        nn.init.zeros_(self.motion_head.weight)
        nn.init.zeros_(self.motion_head.bias)
        nn.init.constant_(self.gate_head.bias, 1.5)  # Start trusting radar motion ~82%

    def forward(
        self,
        x_nwp: torch.Tensor,
        radar_motion: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_nwp:        (B, 2, 27, H, W)
            radar_motion: (B, T_out, 2, H, W)
        Returns:
            guided_motion: (B, T_out, 2, H, W)
            nwp_dyn_feat:  (B, hidden_c, H, W)
        """
        b, _, _, h, w = x_nwp.shape
        dyn = x_nwp[:, :, NWP_MOTION_INDICES].reshape(b, -1, h, w)  # (B, 12, H, W)
        feat = self.encoder(dyn)
        delta_v = self.motion_head(feat).reshape(b, self.pred_length, 2, h, w)
        alpha = torch.sigmoid(self.gate_head(feat)).unsqueeze(2)  # (B, T_out, 1, H, W)
        guided = alpha * radar_motion + (1.0 - alpha) * (radar_motion + delta_v)
        return guided, feat


class NWPEnvNoiseModulator(nn.Module):
    """Conditions GAN latent noise z on NWP thermodynamic/moisture environment (27 vars x 2 slices)."""

    def __init__(self, nwp_channels: int = 54, ngf: int = 32):
        super().__init__()
        self.ngf = ngf
        # Downsample H, W -> H//32, W//32 via 5 stride-2 convs
        self.down_net = nn.Sequential(
            nn.Conv2d(nwp_channels, ngf, kernel_size=4, stride=2, padding=1),      # /2
            nn.GELU(),
            nn.Conv2d(ngf, ngf, kernel_size=4, stride=2, padding=1),               # /4
            nn.GELU(),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),           # /8
            nn.GELU(),
            nn.Conv2d(ngf * 2, ngf * 2, kernel_size=4, stride=2, padding=1),       # /16
            nn.GELU(),
            nn.Conv2d(ngf * 2, ngf * 2, kernel_size=4, stride=2, padding=1),       # /32
            nn.GELU(),
        )
        self.mu_head = nn.Conv2d(ngf * 2, ngf, kernel_size=3, padding=1)
        self.sigma_head = nn.Conv2d(ngf * 2, ngf, kernel_size=3, padding=1)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.constant_(self.sigma_head.bias, 0.5413)  # softplus(0.5413) ~= 1.0

    def forward(self, x_nwp: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x_nwp.shape
        feat = self.down_net(x_nwp.reshape(b, t * c, h, w))
        mu = self.mu_head(feat)
        sigma = F.softplus(self.sigma_head(feat)) + 0.05
        return mu + sigma * eps


class EvolutionBranchBJ(nn.Module):
    """Physics evolution network adapted to 10-channel BJ radar -> RA precipitation."""

    def __init__(
        self,
        in_channels: int,
        pred_length: int,
        base_c: int = 32,
        use_st_bias: bool = True,
        use_nwp_motion: bool = True,
    ):
        super().__init__()
        self.pred_length = pred_length
        self.use_st_bias = use_st_bias
        self.use_nwp_motion = use_nwp_motion

        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 8)

        # Intensity residual branch
        self.up1 = Up(base_c * 16, base_c * 4, bilinear=True)
        self.up2 = Up(base_c * 8, base_c * 2, bilinear=True)
        self.up3 = Up(base_c * 4, base_c, bilinear=True)
        self.up4 = Up(base_c * 2, base_c, bilinear=True)
        self.outc = OutConv(base_c, pred_length)
        self.gamma = nn.Parameter(torch.ones(1, pred_length, 1, 1) * 0.1)

        # Motion (optical flow) branch
        self.up1_v = Up(base_c * 16, base_c * 4, bilinear=True)
        self.up2_v = Up(base_c * 8, base_c * 2, bilinear=True)
        self.up3_v = Up(base_c * 4, base_c, bilinear=True)
        self.up4_v = Up(base_c * 2, base_c, bilinear=True)
        self.outc_v = OutConv(base_c, pred_length * 2)

        if use_st_bias:
            self.intensity_bias = SpatioTemporalChannelBias(pred_length)
            self.motion_bias = SpatioTemporalChannelBias(pred_length * 2)
        if use_nwp_motion:
            self.nwp_motion = NWPMotionGuidance(n_dyn_vars=len(NWP_MOTION_INDICES), pred_length=pred_length)

    def forward(
        self,
        x_feat: torch.Tensor,
        x_nwp: Optional[torch.Tensor] = None,
        coord_yx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.inc(x_feat)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        xi = self.up1(x5, x4)
        xi = self.up2(xi, x3)
        xi = self.up3(xi, x2)
        xi = self.up4(xi, x1)
        intensity = self.outc(xi) * self.gamma
        if self.use_st_bias:
            intensity = self.intensity_bias(intensity, coord_yx)

        xv = self.up1_v(x5, x4)
        xv = self.up2_v(xv, x3)
        xv = self.up3_v(xv, x2)
        xv = self.up4_v(xv, x1)
        motion = self.outc_v(xv)
        if self.use_st_bias:
            motion = self.motion_bias(motion, coord_yx)

        b, _, h, w = motion.shape
        motion = motion.reshape(b, self.pred_length, 2, h, w)
        if self.use_nwp_motion and x_nwp is not None:
            motion, _ = self.nwp_motion(x_nwp, motion)

        intensity = intensity.reshape(b, self.pred_length, 1, h, w)
        return intensity, motion


class NWPNowcastNet(nn.Module):
    """Full NWP-ST-NowcastNet model with toggleable improvements for ablation."""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.input_length = getattr(configs, "input_length", 11)
        self.pred_length = getattr(configs, "pred_length", 10)
        self.radar_ch = getattr(configs, "radar_ch", 10)
        self.nwp_ch = getattr(configs, "nwp_ch", 27)
        self.ngf = getattr(configs, "ngf", 32)

        self.use_st_bias = bool(getattr(configs, "use_st_bias", True))
        self.use_nwp_motion = bool(getattr(configs, "use_nwp_motion", True))
        self.use_nwp_noise = bool(getattr(configs, "use_nwp_noise", True))
        self.use_temporal_radar = bool(getattr(configs, "use_temporal_radar", True))
        self.use_nwp = not bool(getattr(configs, "no_nwp", False))

        if not self.use_nwp:
            self.use_nwp_motion = False
            self.use_nwp_noise = False

        # 1. Radar-to-Rain initial estimator from recent radar products (used as advection seed R0)
        self.radar2rain_init = nn.Sequential(
            nn.Conv2d(self.radar_ch * 3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),  # linear head, no hard ReLU
        )

        # 2. Radar history encoder (temporal ConvGRU vs plain channel stack)
        if self.use_temporal_radar:
            evo_in_ch = 24
            self.radar_temp_enc = RadarTemporalEncoder(
                radar_ch=self.radar_ch,
                hidden_ch=24,
                out_frames_ch=evo_in_ch,
            )
            gen_radar_ch = evo_in_ch
        else:
            # Plain baseline uses CR channel across all input frames + last frame's 10 products
            evo_in_ch = self.input_length + self.radar_ch
            gen_radar_ch = evo_in_ch

        # 3. Physics Evolution Branch
        self.evo_net = EvolutionBranchBJ(
            in_channels=evo_in_ch,
            pred_length=self.pred_length,
            base_c=32,
            use_st_bias=self.use_st_bias,
            use_nwp_motion=self.use_nwp_motion,
        )

        # 4. Generative Branch (Encoder + NWP-Conditioned Noise Projector + SPADE Decoder)
        # Also feeds NWP summary if use_nwp is enabled (following MASTER.md M1 concat finding)
        nwp_enc_ch = self.nwp_ch * 2 if self.use_nwp else 0
        gen_enc_in = gen_radar_ch + self.pred_length + nwp_enc_ch
        self.gen_enc = Generative_Encoder(gen_enc_in, base_c=self.ngf)
        self.proj = Noise_Projector(self.ngf, configs)
        # PixelShuffle(4) maps (B, ngf*32, H//32, W//32) -> (B, 2*ngf, H//8, W//8) for ANY H,W % 32 == 0
        self.noise_unshuffle = nn.PixelShuffle(4)

        if self.use_nwp_noise:
            self.nwp_noise_mod = NWPEnvNoiseModulator(nwp_channels=self.nwp_ch * 2, ngf=self.ngf)

        # Ensure Decoder receives exact ic_feature = 10 * ngf and evo_ic = pred_length
        configs.ic_feature = self.ngf * 10
        configs.evo_ic = self.pred_length
        configs.gen_oc = self.pred_length
        self.gen_dec = Generative_Decoder(configs)

        if self.use_st_bias:
            self.gen_st_bias = SpatioTemporalChannelBias(self.pred_length)
        if self.use_temporal_radar:
            self.temporal_refiner = RadarTemporalRefiner(channels=1, pred_length=self.pred_length)

    def forward_evolution(
        self,
        x_radar: torch.Tensor,
        x_nwp: Optional[torch.Tensor] = None,
        coord_yx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run Stage-1 physics evolution branch only."""
        b, t_in, c_r, h, w = x_radar.shape

        # Initial rain seed R0 from last 3 radar frames (B, 3*10, H, W)
        recent_3 = x_radar[:, -3:].reshape(b, 3 * c_r, h, w)
        init_rain = self.radar2rain_init(recent_3)  # (B, 1, H, W)

        if self.use_temporal_radar:
            radar_feat, _ = self.radar_temp_enc(x_radar)
        else:
            cr_seq = x_radar[:, :, 6]  # (B, T_in, H, W) CR channel
            last_all = x_radar[:, -1]  # (B, 10, H, W)
            radar_feat = torch.cat([cr_seq, last_all], dim=1)

        intensity, motion = self.evo_net(radar_feat, x_nwp=x_nwp if self.use_nwp else None, coord_yx=coord_yx)

        # Differentiable bilinear advection rollout over T_out steps
        grid = make_coord_grid(b, h, w, x_radar.device, x_radar.dtype)
        last_frame = init_rain
        series = []
        for i in range(self.pred_length):
            last_frame = warp(last_frame, motion[:, i], grid, mode="bilinear", padding_mode="border")
            last_frame = last_frame + intensity[:, i]
            series.append(last_frame)

        evo_rain = torch.stack(series, dim=1)  # (B, T_out, 1, H, W)
        return {
            "evo_rain": evo_rain,
            "init_rain": init_rain,
            "motion": motion,
            "intensity": intensity,
            "radar_feat": radar_feat,
        }

    def forward(
        self,
        x_radar: torch.Tensor,
        x_nwp: Optional[torch.Tensor] = None,
        coord_yx: Optional[torch.Tensor] = None,
        run_gen: bool = True,
        noise_eps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward_evolution(x_radar, x_nwp=x_nwp, coord_yx=coord_yx)
        if not run_gen:
            out["pred_rain"] = out["evo_rain"]
            return out

        b, _, _, h, w = x_radar.shape
        evo_cond = torch.log1p(torch.clamp(out["evo_rain"].squeeze(2), min=0.0))  # (B, T_out, H, W)

        enc_parts = [out["radar_feat"], evo_cond]
        if self.use_nwp:
            if x_nwp is not None:
                nwp_flat = x_nwp.reshape(b, self.nwp_ch * 2, h, w)
            else:
                nwp_flat = torch.zeros(b, self.nwp_ch * 2, h, w, device=x_radar.device, dtype=x_radar.dtype)
            enc_parts.append(nwp_flat)

        evo_feature = self.gen_enc(torch.cat(enc_parts, dim=1))  # (B, 8*ngf, H//8, W//8)

        if noise_eps is None:
            noise_eps = torch.randn(b, self.ngf, h // 32, w // 32, device=x_radar.device, dtype=x_radar.dtype)

        if self.use_nwp_noise and x_nwp is not None:
            z = self.nwp_noise_mod(x_nwp, noise_eps)
        else:
            z = noise_eps

        noise_feature = self.noise_unshuffle(self.proj(z))  # (B, 2*ngf, H//8, W//8)
        feature = torch.cat([evo_feature, noise_feature], dim=1)  # (B, 10*ngf, H//8, W//8)

        gen_raw = self.gen_dec(feature, evo_cond)  # (B, T_out, H, W)
        if self.use_st_bias:
            gen_raw = self.gen_st_bias(gen_raw, coord_yx)

        pred_rain = gen_raw.unsqueeze(2)  # (B, T_out, 1, H, W)
        if self.use_temporal_radar:
            pred_rain = self.temporal_refiner(pred_rain)

        out["pred_rain"] = pred_rain
        return out
