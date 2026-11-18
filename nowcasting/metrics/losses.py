"""Score-aware, physics-consistent, and GAN losses for NWP-NowcastNet on BJ dataset.

Follows MASTER.md §3.9 & §4.5:
- Never rely solely on unweighted log-space regression (causes severe underestimation at >=2.0 mm/6min).
- Smooth C1 extension for log1p below zero so linear output heads (without hard ReLU) maintain healthy gradients.
- Combines BalancedExtremePrecipLoss (weighted Huber + differentiable multi-threshold Soft-CSI)
  with NowcastNet's MultiScalePoolLoss, optical flow smoothness, and Hinge GAN + feature matching losses.
"""

from typing import Dict, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _smooth_log1p(x: torch.Tensor) -> torch.Tensor:
    """C1-smooth log1p extension for negative values: log1p(x) for x>=0, x for x<0."""
    return torch.where(x >= 0.0, torch.log1p(torch.clamp(x, min=0.0)), x)


class BalancedExtremePrecipLoss(nn.Module):
    """Score-aware precipitation loss: intensity-weighted log-Huber + multi-threshold Soft-CSI."""

    def __init__(
        self,
        thresholds: Sequence[float] = (0.1, 0.5, 2.0, 4.0),
        thr_weights: Sequence[float] = (0.15, 0.25, 0.35, 0.25),
        alpha_csi: float = 0.35,
        huber_delta: float = 0.5,
        temp: float = 0.25,
    ):
        super().__init__()
        self.thresholds = tuple(float(t) for t in thresholds)
        w = torch.tensor(thr_weights, dtype=torch.float32)
        self.register_buffer("thr_weights", w / w.sum())
        self.alpha_csi = float(alpha_csi)
        self.huber_delta = float(huber_delta)
        self.temp = float(temp)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred:   (B, T, 1, H, W) raw model output (mm/6min, linear head before floor_at_zero)
            target: (B, T, 1, H, W) ground truth RA in mm/6min
            valid_mask: (B, 1, H, W) or None
        """
        pred_f = pred.float()
        tgt_f = torch.clamp(target.float(), min=0.0)

        if valid_mask is None:
            mask = torch.ones_like(pred_f)
        else:
            m = valid_mask.unsqueeze(1) if valid_mask.ndim == 4 else valid_mask[:, None, None, ...]
            mask = m.expand_as(pred_f).float()

        # 1. Intensity-weighted log-space Huber loss (keeps dry pixels from dominating 97% of gradient)
        log_p = _smooth_log1p(pred_f)
        log_y = torch.log1p(tgt_f)
        pixel_weight = 1.0 + 4.0 * torch.clamp(tgt_f, max=8.0) / 8.0
        # Penalize strongly negative predictions lightly so output stays near >=0 without dead ReLU
        neg_pen = torch.where(pred_f < 0.0, 0.5 * pred_f.pow(2), torch.zeros_like(pred_f))

        huber = F.smooth_l1_loss(log_p, log_y, reduction="none", beta=self.huber_delta)
        loss_reg = ((huber * pixel_weight + neg_pen) * mask).sum() / torch.clamp(mask.sum(), min=1.0)

        # 2. Differentiable multi-threshold Soft-CSI
        # Decision boundary sits strictly on the threshold: sigmoid((pred - thr)/temp)
        loss_csi = pred_f.new_zeros(())
        for i, thr in enumerate(self.thresholds):
            scale = max(self.temp * (1.0 + 0.25 * thr), 0.1)
            p_prob = torch.sigmoid((pred_f - thr) / scale) * mask
            y_bin = (tgt_f >= thr).float() * mask

            tp = (p_prob * y_bin).sum()
            fp = (p_prob * (1.0 - y_bin)).sum()
            fn = ((1.0 - p_prob) * y_bin).sum()
            # Skip thresholds absent in both pred & target with small smooth eps
            soft_csi = (tp + 1e-4) / (tp + fp + fn + 1e-4)
            loss_csi = loss_csi + self.thr_weights[i] * (1.0 - soft_csi)

        total = (1.0 - self.alpha_csi) * loss_reg + self.alpha_csi * loss_csi
        return total, {
            "loss_reg": float(loss_reg.detach().item()),
            "loss_soft_csi": float(loss_csi.detach().item()),
        }


class MultiScalePoolLoss(nn.Module):
    """NowcastNet pooling loss preserving convective peaks while allowing small spatial jitter."""

    def __init__(self, kernels: Sequence[int] = (4, 8)):
        super().__init__()
        self.kernels = tuple(kernels)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        b, t, c, h, w = pred.shape
        pred_pos = F.relu(pred.float()).reshape(b * t, c, h, w)
        tgt = torch.clamp(target.float(), min=0.0).reshape(b * t, c, h, w)

        if valid_mask is not None:
            m = valid_mask.unsqueeze(1).expand(b, t, 1, h, w).reshape(b * t, 1, h, w)
            pred_pos = pred_pos * m
            tgt = tgt * m

        loss = pred.new_zeros(())
        for k in self.kernels:
            p_max = F.max_pool2d(pred_pos, kernel_size=k, stride=k)
            y_max = F.max_pool2d(tgt, kernel_size=k, stride=k)
            p_avg = F.avg_pool2d(pred_pos, kernel_size=k, stride=k)
            y_avg = F.avg_pool2d(tgt, kernel_size=k, stride=k)
            loss = loss + F.l1_loss(torch.log1p(p_max), torch.log1p(y_max)) + F.l1_loss(torch.log1p(p_avg), torch.log1p(y_avg))
        return loss / max(len(self.kernels), 1)


def motion_smoothness_loss(motion: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    """Spatial gradient smoothness penalty on predicted optical flow fields (B, T, 2, H, W)."""
    dy = motion[:, :, :, 1:, :] - motion[:, :, :, :-1, :]
    dx = motion[:, :, :, :, 1:] - motion[:, :, :, :, :-1]
    if valid_mask is not None:
        my = valid_mask[:, None, :, 1:, :] * valid_mask[:, None, :, :-1, :]
        mx = valid_mask[:, None, :, :, 1:] * valid_mask[:, None, :, :, :-1]
        loss_y = (dy.pow(2) * my).sum() / torch.clamp(my.sum() * 2.0, min=1.0)
        loss_x = (dx.pow(2) * mx).sum() / torch.clamp(mx.sum() * 2.0, min=1.0)
        return loss_y + loss_x
    return dy.pow(2).mean() + dx.pow(2).mean()


class EvolutionLoss(nn.Module):
    """Stage-1 physics evolution training loss."""

    def __init__(self, alpha_csi: float = 0.35, motion_smooth_weight: float = 0.02, init_weight: float = 0.2):
        super().__init__()
        self.precip_loss = BalancedExtremePrecipLoss(alpha_csi=alpha_csi)
        self.motion_smooth_weight = motion_smooth_weight
        self.init_weight = init_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        evo_rain = outputs["evo_rain"]
        loss_main, stats = self.precip_loss(evo_rain, target, valid_mask)
        loss_smooth = motion_smoothness_loss(outputs["motion"], valid_mask)
        total = loss_main + self.motion_smooth_weight * loss_smooth
        stats["loss_motion_smooth"] = float(loss_smooth.detach().item())

        # Optional supervision on initial t+6 frame estimate R0 vs target[:, 0]
        if "init_rain" in outputs and self.init_weight > 0:
            loss_init, _ = self.precip_loss(outputs["init_rain"].unsqueeze(1), target[:, :1], valid_mask)
            total = total + self.init_weight * loss_init
            stats["loss_init"] = float(loss_init.detach().item())

        stats["loss_total"] = float(total.detach().item())
        return total, stats


class GenerationLoss(nn.Module):
    """Stage-2 generator loss: score-aware precip + multi-scale pool + Hinge GAN + feature matching."""

    def __init__(
        self,
        alpha_csi: float = 0.35,
        pool_weight: float = 0.5,
        adv_spatial_weight: float = 0.05,
        adv_temporal_weight: float = 0.05,
        evo_anchor_weight: float = 0.15,
    ):
        super().__init__()
        self.precip_loss = BalancedExtremePrecipLoss(alpha_csi=alpha_csi)
        self.pool_loss = MultiScalePoolLoss()
        self.pool_weight = pool_weight
        self.adv_spatial_weight = adv_spatial_weight
        self.adv_temporal_weight = adv_temporal_weight
        self.evo_anchor_weight = evo_anchor_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        d_fake_spatial: torch.Tensor,
        d_fake_temporal: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred = outputs["pred_rain"]
        loss_precip, stats = self.precip_loss(pred, target, valid_mask)
        loss_pool = self.pool_loss(pred, target, valid_mask)
        loss_adv_s = -d_fake_spatial.mean()
        loss_adv_t = -d_fake_temporal.mean()

        total = (
            loss_precip
            + self.pool_weight * loss_pool
            + self.adv_spatial_weight * loss_adv_s
            + self.adv_temporal_weight * loss_adv_t
        )

        # Keep evolution branch anchored if unfrozen in Stage 2
        if "evo_rain" in outputs and outputs["evo_rain"].requires_grad and self.evo_anchor_weight > 0:
            loss_evo, _ = self.precip_loss(outputs["evo_rain"], target, valid_mask)
            total = total + self.evo_anchor_weight * loss_evo
            stats["loss_evo_anchor"] = float(loss_evo.detach().item())

        stats.update({
            "loss_pool": float(loss_pool.detach().item()),
            "loss_adv_s": float(loss_adv_s.detach().item()),
            "loss_adv_t": float(loss_adv_t.detach().item()),
            "loss_total": float(total.detach().item()),
        })
        return total, stats


class DiscriminatorLoss(nn.Module):
    """Hinge adversarial loss for Spatial + Temporal Discriminators."""

    def forward(
        self,
        d_real_s: torch.Tensor,
        d_fake_s: torch.Tensor,
        d_real_t: torch.Tensor,
        d_fake_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_s = 0.5 * (F.relu(1.0 - d_real_s).mean() + F.relu(1.0 + d_fake_s).mean())
        loss_t = 0.5 * (F.relu(1.0 - d_real_t).mean() + F.relu(1.0 + d_fake_t).mean())
        total = loss_s + loss_t
        return total, {
            "d_loss_s": float(loss_s.detach().item()),
            "d_loss_t": float(loss_t.detach().item()),
            "d_loss_total": float(total.detach().item()),
        }
