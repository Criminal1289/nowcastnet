"""Full-field precipitation verification metrics and Hann-window patch stitching.

Implements MASTER.md §1.2 and §3.8:
- Thresholds in mm/6min: 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0 (main table: 0.1, 0.5, 2.0, 4.0)
- CSI, HSS, POD, FAR, Precision, BIAS, ETS, MAE, RMSE + per-lead-time scores + FSS
- Floor at zero before scoring; restricts evaluation strictly to valid_mask (301x301).
"""

from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_THRESHOLDS = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
MAIN_THRESHOLDS = (0.1, 0.5, 2.0, 4.0)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _compute_contingency_metrics(tp: float, fp: float, fn: float, tn: float) -> Dict[str, float]:
    total = tp + fp + fn + tn
    csi = _safe_div(tp, tp + fp + fn)
    pod = _safe_div(tp, tp + fn)
    far = _safe_div(fp, tp + fp) if (tp + fp) > 0 else float("nan")
    precision = _safe_div(tp, tp + fp)
    bias = _safe_div(tp + fp, tp + fn)

    # Equitable Threat Score (ETS) and Heidke Skill Score (HSS)
    hits_random = ((tp + fn) * (tp + fp)) / total if total > 0 else 0.0
    ets = _safe_div(tp - hits_random, tp + fp + fn - hits_random)
    hss_num = 2.0 * (tp * tn - fp * fn)
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = _safe_div(hss_num, hss_den)

    return {
        "CSI": csi,
        "HSS": hss,
        "ETS": ets,
        "POD": pod,
        "FAR": far,
        "Precision": precision,
        "BIAS": bias,
        "hits": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


class PrecipitationVerifier:
    """Streaming accumulator for full-field precipitation scores across thresholds and lead times."""

    def __init__(
        self,
        thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
        num_leads: int = 10,
        fss_window: int = 9,
    ):
        self.thresholds = tuple(float(t) for t in thresholds)
        self.num_leads = num_leads
        self.fss_window = fss_window
        self.reset()

    def reset(self) -> None:
        n_thr = len(self.thresholds)
        self.tp = np.zeros((self.num_leads, n_thr), dtype=np.float64)
        self.fp = np.zeros((self.num_leads, n_thr), dtype=np.float64)
        self.fn = np.zeros((self.num_leads, n_thr), dtype=np.float64)
        self.tn = np.zeros((self.num_leads, n_thr), dtype=np.float64)
        self.fss_num = np.zeros((self.num_leads, n_thr), dtype=np.float64)
        self.fss_den = np.zeros((self.num_leads, n_thr), dtype=np.float64)
        self.sum_abs_err = np.zeros((self.num_leads,), dtype=np.float64)
        self.sum_sq_err = np.zeros((self.num_leads,), dtype=np.float64)
        self.count = np.zeros((self.num_leads,), dtype=np.float64)

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> None:
        """Accumulate batch metrics.

        Args:
            pred:   (B, T, 1, H, W) in mm/6min
            target: (B, T, 1, H, W) in mm/6min
            valid_mask: (B, 1, H, W) or (B, H, W), 1 for valid 301x301 pixels, 0 for padding.
        """
        pred_pos = torch.clamp(pred.detach().float(), min=0.0)
        tgt = torch.clamp(target.detach().float(), min=0.0)
        b, t, _, h, w = pred_pos.shape

        if valid_mask is None:
            mask = torch.ones((b, 1, h, w), device=pred.device, dtype=torch.bool)
        else:
            if valid_mask.ndim == 3:
                mask = valid_mask.unsqueeze(1) > 0.5
            else:
                mask = valid_mask > 0.5

        diff = (pred_pos - tgt) * mask.unsqueeze(1)
        n_valid_per_item = mask.sum().item()

        for lead in range(min(t, self.num_leads)):
            p_l = pred_pos[:, lead, 0]  # (B, H, W)
            y_l = tgt[:, lead, 0]
            m_l = mask[:, 0]

            err = (p_l - y_l)[m_l]
            self.sum_abs_err[lead] += float(err.abs().sum().item())
            self.sum_sq_err[lead] += float((err * err).sum().item())
            self.count[lead] += float(n_valid_per_item)

            for ti, thr in enumerate(self.thresholds):
                p_bin = (p_l >= thr) & m_l
                y_bin = (y_l >= thr) & m_l
                tp = float((p_bin & y_bin).sum().item())
                fp = float((p_bin & (~y_bin) & m_l).sum().item())
                fn = float(((~p_bin) & y_bin & m_l).sum().item())
                tn = float(((~p_bin) & (~y_bin) & m_l).sum().item())

                self.tp[lead, ti] += tp
                self.fp[lead, ti] += fp
                self.fn[lead, ti] += fn
                self.tn[lead, ti] += tn

                # Neighborhood FSS on valid region
                if self.fss_window > 1:
                    k = self.fss_window
                    pad = k // 2
                    p_f = F.avg_pool2d(p_bin.float().unsqueeze(1), k, stride=1, padding=pad)
                    y_f = F.avg_pool2d(y_bin.float().unsqueeze(1), k, stride=1, padding=pad)
                    m_f = m_l.unsqueeze(1)
                    num = float((((p_f - y_f) ** 2) * m_f).sum().item())
                    den = float((((p_f ** 2) + (y_f ** 2)) * m_f).sum().item())
                    self.fss_num[lead, ti] += num
                    self.fss_den[lead, ti] += den

    def compute(self) -> Dict[str, object]:
        overall: Dict[str, Dict[str, float]] = {}
        lead_scores: Dict[int, Dict[str, Dict[str, float]]] = {}

        for ti, thr in enumerate(self.thresholds):
            key = f"{thr:g}"
            m = _compute_contingency_metrics(
                self.tp[:, ti].sum(),
                self.fp[:, ti].sum(),
                self.fn[:, ti].sum(),
                self.tn[:, ti].sum(),
            )
            fss_den = self.fss_den[:, ti].sum()
            m["FSS"] = float(1.0 - self.fss_num[:, ti].sum() / fss_den) if fss_den > 0 else 0.0
            overall[key] = m

        for lead in range(self.num_leads):
            l_dict: Dict[str, Dict[str, float]] = {}
            for ti, thr in enumerate(self.thresholds):
                key = f"{thr:g}"
                m = _compute_contingency_metrics(
                    self.tp[lead, ti],
                    self.fp[lead, ti],
                    self.fn[lead, ti],
                    self.tn[lead, ti],
                )
                fss_den = self.fss_den[lead, ti]
                m["FSS"] = float(1.0 - self.fss_num[lead, ti] / fss_den) if fss_den > 0 else 0.0
                l_dict[key] = m
            lead_cnt = max(self.count[lead], 1.0)
            l_dict["MAE"] = float(self.sum_abs_err[lead] / lead_cnt)
            l_dict["RMSE"] = float(np.sqrt(self.sum_sq_err[lead] / lead_cnt))
            lead_scores[lead] = l_dict

        tot_cnt = max(self.count.sum(), 1.0)
        mae = float(self.sum_abs_err.sum() / tot_cnt)
        rmse = float(np.sqrt(self.sum_sq_err.sum() / tot_cnt))

        return {
            "overall": overall,
            "lead_scores": lead_scores,
            "MAE": mae,
            "RMSE": rmse,
        }


class PatchStitcher:
    """Hann-window patch stitcher reconstructing full 301x301 predictions from overlapping patches."""

    def __init__(self, height: int = 301, width: int = 301, num_leads: int = 10, patch_size: int = 128):
        self.height = height
        self.width = width
        self.num_leads = num_leads
        self.patch_size = patch_size
        wy = np.hanning(patch_size + 2)[1:-1].astype(np.float32)
        wx = np.hanning(patch_size + 2)[1:-1].astype(np.float32)
        self.window = np.outer(wy, wx)[None, None, :, :]  # (1, 1, ps, ps)
        self.acc = np.zeros((num_leads, 1, height, width), dtype=np.float64)
        self.w_acc = np.zeros((1, 1, height, width), dtype=np.float64)

    def add_patch(self, patch: np.ndarray, y0: int, x0: int) -> None:
        ps = self.patch_size
        self.acc[:, :, y0:y0 + ps, x0:x0 + ps] += patch * self.window
        self.w_acc[:, :, y0:y0 + ps, x0:x0 + ps] += self.window

    def result(self) -> np.ndarray:
        return (self.acc / np.maximum(self.w_acc, 1e-8)).astype(np.float32)
