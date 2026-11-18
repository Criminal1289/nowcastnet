"""Two-stage Trainer & Evaluator for NWP-ST-NowcastNet on the BJ dataset.

Supports:
- Stage 1: Evolution branch pretraining (EvolutionLoss: score-aware precip + motion smoothness)
- Stage 2: Joint Generative + Spatio-Temporal Discriminator training (Hinge GAN + Pool + score-aware)
- Mixed precision (AMP), gradient accumulation, gradient clipping, and checkpoint config.json persistence.
- Full-field 301x301 validation scoring via PrecipitationVerifier, saving best ckpts by Val CSI@2.0.
"""

import json
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn

from nowcasting.metrics.losses import DiscriminatorLoss, EvolutionLoss, GenerationLoss
from nowcasting.metrics.plotting import plot_training_curves
from nowcasting.metrics.precipitation import MAIN_THRESHOLDS, PrecipitationVerifier
from nowcasting.models.discriminator import SpatioTemporalDiscriminator


def save_config_json(save_dir: str, configs) -> None:
    cfg_dict = {k: v for k, v in vars(configs).items() if isinstance(v, (str, int, float, bool, list, tuple)) or v is None}
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, ensure_ascii=False)


def _format_scores(metrics: Dict[str, object]) -> str:
    ov = metrics["overall"]
    parts = [f"MAE={metrics['MAE']:.4f}"]
    for thr in MAIN_THRESHOLDS:
        k = f"{thr:g}"
        m = ov[k]
        parts.append(f"@{k}:CSI={m['CSI']:.4f}/FAR={m['FAR']:.3f}/BIAS={m['BIAS']:.3f}")
    return " | ".join(parts)


class BJTrainer:
    """Manages Stage-1 & Stage-2 training and full-field validation."""

    def __init__(self, model: nn.Module, discriminator: SpatioTemporalDiscriminator, configs):
        self.model = model
        self.disc = discriminator
        self.configs = configs
        self.device = torch.device(configs.device)
        self.save_dir = configs.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        save_config_json(self.save_dir, configs)
        self.run_ts = getattr(configs, "run_timestamp", "") or time.strftime("%Y%m%d_%H%M%S")
        self.history_path = os.path.join(self.save_dir, f"history_{self.run_ts}.json")
        self.curves_path = os.path.join(self.save_dir, f"train_curves_{self.run_ts}.png")
        if os.path.isfile(self.history_path):
            with open(self.history_path, "r", encoding="utf-8") as f:
                self.history = json.load(f)
        else:
            self.history = {"stage1": [], "stage2": []}

        self.use_amp = bool(getattr(configs, "amp", True)) and self.device.type == "cuda"
        self.grad_accum = max(1, int(getattr(configs, "grad_accum", 1)))
        self.clip_grad = float(getattr(configs, "clip_grad", 1.0))

        self.evo_loss_fn = EvolutionLoss(
            alpha_csi=getattr(configs, "alpha_csi", 0.35),
            motion_smooth_weight=getattr(configs, "motion_smooth_weight", 0.02),
        ).to(self.device)
        self.gen_loss_fn = GenerationLoss(
            alpha_csi=getattr(configs, "alpha_csi", 0.35),
            pool_weight=getattr(configs, "pool_weight", 0.5),
            adv_spatial_weight=getattr(configs, "adv_spatial_weight", 0.05),
            adv_temporal_weight=getattr(configs, "adv_temporal_weight", 0.05),
        ).to(self.device)
        self.disc_loss_fn = DiscriminatorLoss().to(self.device)

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    @torch.no_grad()
    def evaluate(self, val_loader, run_gen: bool = True) -> Dict[str, object]:
        self.model.eval()
        verifier = PrecipitationVerifier(num_leads=getattr(self.configs, "pred_length", 10))
        for batch in val_loader:
            b = self._move_batch(batch)
            x_nwp = None if getattr(self.configs, "no_nwp", False) else b["x_nwp"]
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                out = self.model(b["x_radar"], x_nwp=x_nwp, coord_yx=b["coord_yx"], run_gen=run_gen)
            pred = out["pred_rain"] if run_gen else out["evo_rain"]
            verifier.update(pred, b["y_rain"], b["valid_mask"])
        return verifier.compute()

    def _record_history(self, stage_key: str, entry: Dict[str, float]) -> None:
        self.history.setdefault(stage_key, []).append(entry)
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        plot_training_curves(self.history, self.curves_path)

    def _save_checkpoint(self, filename: str, epoch: int, stage: int, metrics: Dict[str, object]) -> None:
        ckpt = {
            "epoch": epoch,
            "stage": stage,
            "generator": self.model.state_dict(),
            "discriminator": self.disc.state_dict(),
            "metrics": metrics["overall"],
        }
        torch.save(ckpt, os.path.join(self.save_dir, filename))

    def train_stage1(self, train_loader, val_loader, epochs: int) -> None:
        print(f"=== Stage 1 (Evolution Pretraining) for {epochs} epochs ===")
        # Collect evolution parameters
        evo_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and any(k in n for k in ("radar2rain_init", "radar_temp_enc", "evo_net"))
        ]
        optimizer = torch.optim.AdamW(evo_params, lr=getattr(self.configs, "lr_evo", 3e-4), weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        best_score = -1.0
        for epoch in range(epochs):
            self.model.train()
            t0 = time.time()
            running_loss = 0.0
            n_steps = 0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader):
                b = self._move_batch(batch)
                x_nwp = None if getattr(self.configs, "no_nwp", False) else b["x_nwp"]
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    out = self.model.forward_evolution(b["x_radar"], x_nwp=x_nwp, coord_yx=b["coord_yx"])
                    loss, _ = self.evo_loss_fn(out, b["y_rain"], b["valid_mask"])
                    loss_scaled = loss / self.grad_accum

                scaler.scale(loss_scaled).backward()
                if (step + 1) % self.grad_accum == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(evo_params, self.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                running_loss += float(loss.detach().item())
                n_steps += 1

            scheduler.step()
            val_metrics = self.evaluate(val_loader, run_gen=False)
            ov = val_metrics["overall"]
            sel_score = 0.6 * ov["2"]["CSI"] + 0.4 * ov["0.5"]["CSI"]
            dt = time.time() - t0
            print(f"[Stage1 Ep {epoch:02d}] loss={running_loss/max(n_steps,1):.4f} ({dt:.1f}s) | Val: {_format_scores(val_metrics)}")

            self._save_checkpoint("last_stage1.pt", epoch, stage=1, metrics=val_metrics)
            if sel_score >= best_score:
                best_score = sel_score
                self._save_checkpoint("best_stage1.pt", epoch, stage=1, metrics=val_metrics)
            self._record_history("stage1", {
                "epoch": epoch,
                "loss": running_loss / max(n_steps, 1),
                "val_mae": float(val_metrics["MAE"]),
                "val_csi_0.1": float(ov["0.1"]["CSI"]),
                "val_csi_0.5": float(ov["0.5"]["CSI"]),
                "val_csi_2": float(ov["2"]["CSI"]),
            })

    def train_stage2(self, train_loader, val_loader, epochs: int) -> None:
        print(f"=== Stage 2 (GAN + Spatio-Temporal Refinement) for {epochs} epochs ===")
        freeze_evo = bool(getattr(self.configs, "freeze_evo_stage2", False))
        g_params = []
        for n, p in self.model.named_parameters():
            # Never turn non-trainable buffers (e.g., spectral_norm weight_u/weight_v) into trainable params
            if not p.requires_grad:
                continue
            is_evo = any(k in n for k in ("radar2rain_init", "radar_temp_enc", "evo_net"))
            if is_evo and freeze_evo:
                p.requires_grad = False
            else:
                lr_scale = 0.2 if is_evo else 1.0
                g_params.append({"params": [p], "lr": getattr(self.configs, "lr_gen", 2e-4) * lr_scale})

        disc_trainable = [p for p in self.disc.parameters() if p.requires_grad]
        opt_g = torch.optim.AdamW(g_params, weight_decay=1e-4, betas=(0.5, 0.999))
        opt_d = torch.optim.AdamW(
            disc_trainable,
            lr=getattr(self.configs, "lr_disc", 2e-4),
            weight_decay=1e-4,
            betas=(0.5, 0.999),
        )
        scaler_g = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        scaler_d = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        best_score = -1.0
        for epoch in range(epochs):
            self.model.train()
            self.disc.train()
            t0 = time.time()
            g_sum, d_sum, n_steps = 0.0, 0.0, 0
            opt_g.zero_grad(set_to_none=True)
            opt_d.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader):
                b = self._move_batch(batch)
                x_nwp = None if getattr(self.configs, "no_nwp", False) else b["x_nwp"]

                # --- 1. Update Discriminator ---
                for p in disc_trainable:
                    p.requires_grad = True
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    with torch.no_grad():
                        out_d = self.model(b["x_radar"], x_nwp=x_nwp, coord_yx=b["coord_yx"], run_gen=True)
                        cond = self.disc.build_condition(out_d["evo_rain"], b["x_radar"])
                    d_real_s, d_real_t = self.disc(b["y_rain"], cond)
                    d_fake_s, d_fake_t = self.disc(out_d["pred_rain"].detach(), cond)
                    d_loss, _ = self.disc_loss_fn(d_real_s, d_fake_s, d_real_t, d_fake_t)
                    d_loss_scaled = d_loss / self.grad_accum

                scaler_d.scale(d_loss_scaled).backward()
                if (step + 1) % self.grad_accum == 0 or (step + 1) == len(train_loader):
                    scaler_d.unscale_(opt_d)
                    nn.utils.clip_grad_norm_(disc_trainable, self.clip_grad)
                    scaler_d.step(opt_d)
                    scaler_d.update()
                    opt_d.zero_grad(set_to_none=True)

                # --- 2. Update Generator ---
                for p in disc_trainable:
                    p.requires_grad = False
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    out_g = self.model(b["x_radar"], x_nwp=x_nwp, coord_yx=b["coord_yx"], run_gen=True)
                    cond_g = self.disc.build_condition(out_g["evo_rain"], b["x_radar"])
                    g_fake_s, g_fake_t = self.disc(out_g["pred_rain"], cond_g)
                    g_loss, _ = self.gen_loss_fn(out_g, b["y_rain"], g_fake_s, g_fake_t, b["valid_mask"])
                    g_loss_scaled = g_loss / self.grad_accum

                scaler_g.scale(g_loss_scaled).backward()
                if (step + 1) % self.grad_accum == 0 or (step + 1) == len(train_loader):
                    scaler_g.unscale_(opt_g)
                    nn.utils.clip_grad_norm_([p for grp in g_params for p in grp["params"]], self.clip_grad)
                    scaler_g.step(opt_g)
                    scaler_g.update()
                    opt_g.zero_grad(set_to_none=True)

                for p in disc_trainable:
                    p.requires_grad = True

                g_sum += float(g_loss.detach().item())
                d_sum += float(d_loss.detach().item())
                n_steps += 1

            val_metrics = self.evaluate(val_loader, run_gen=True)
            ov = val_metrics["overall"]
            sel_score = 0.6 * ov["2"]["CSI"] + 0.4 * ov["0.5"]["CSI"]
            dt = time.time() - t0
            print(
                f"[Stage2 Ep {epoch:02d}] g_loss={g_sum/max(n_steps,1):.4f} "
                f"d_loss={d_sum/max(n_steps,1):.4f} ({dt:.1f}s) | Val: {_format_scores(val_metrics)}"
            )

            self._save_checkpoint("last_stage2.pt", epoch, stage=2, metrics=val_metrics)
            if sel_score >= best_score:
                best_score = sel_score
                self._save_checkpoint("best_stage2.pt", epoch, stage=2, metrics=val_metrics)
            self._record_history("stage2", {
                "epoch": epoch,
                "g_loss": g_sum / max(n_steps, 1),
                "d_loss": d_sum / max(n_steps, 1),
                "val_mae": float(val_metrics["MAE"]),
                "val_csi_0.1": float(ov["0.1"]["CSI"]),
                "val_csi_0.5": float(ov["0.5"]["CSI"]),
                "val_csi_2": float(ov["2"]["CSI"]),
            })
