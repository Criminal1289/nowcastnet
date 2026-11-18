"""Error visualization and diagnostic plotting for BJ precipitation nowcasting."""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def _crop_valid(arr: np.ndarray, valid_h: int = 301, valid_w: int = 301) -> np.ndarray:
    """Crop padded (..., H, W) tensor/array back to valid 301x301 region."""
    return arr[..., :valid_h, :valid_w]


class ErrorDistributionAccumulator:
    """Accumulates pixel-level error distributions and intensity-binned bias/RMSE over 301x301."""

    BIN_EDGES = (0.0, 0.1, 0.5, 2.0, 4.0, float("inf"))
    BIN_LABELS = ("Dry\n(<0.1)", "Light\n(0.1-0.5)", "Moderate\n(0.5-2.0)", "Heavy\n(2.0-4.0)", "Extreme\n(>=4.0)")

    def __init__(self, hist_range: Tuple[float, float] = (-5.0, 5.0), n_bins: int = 100):
        self.hist_range = hist_range
        self.hist_edges = np.linspace(hist_range[0], hist_range[1], n_bins + 1)
        self.hist_counts_wet = np.zeros(n_bins, dtype=np.int64)
        self.hist_counts_all = np.zeros(n_bins, dtype=np.int64)

        n_int = len(self.BIN_LABELS)
        self.bin_count = np.zeros(n_int, dtype=np.int64)
        self.bin_sum_err = np.zeros(n_int, dtype=np.float64)
        self.bin_sum_sq = np.zeros(n_int, dtype=np.float64)
        self.bin_sum_abs = np.zeros(n_int, dtype=np.float64)

    def update(self, pred: np.ndarray, target: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> None:
        pred_pos = np.maximum(pred, 0.0)
        tgt = np.maximum(target, 0.0)
        if valid_mask is not None:
            m = np.broadcast_to(valid_mask[:, None, ...] > 0.5, pred_pos.shape)
            p_flat = pred_pos[m]
            y_flat = tgt[m]
        else:
            p_flat = pred_pos.ravel()
            y_flat = tgt.ravel()

        err = p_flat - y_flat
        c_all, _ = np.histogram(np.clip(err, self.hist_range[0], self.hist_range[1]), bins=self.hist_edges)
        self.hist_counts_all += c_all

        wet = (y_flat >= 0.1) | (p_flat >= 0.1)
        if np.any(wet):
            c_wet, _ = np.histogram(np.clip(err[wet], self.hist_range[0], self.hist_range[1]), bins=self.hist_edges)
            self.hist_counts_wet += c_wet

        for i in range(len(self.BIN_LABELS)):
            lo, hi = self.BIN_EDGES[i], self.BIN_EDGES[i + 1]
            if i == 0:
                sel = y_flat < hi
            elif np.isinf(hi):
                sel = y_flat >= lo
            else:
                sel = (y_flat >= lo) & (y_flat < hi)
            if np.any(sel):
                e_sel = err[sel]
                self.bin_count[i] += int(e_sel.size)
                self.bin_sum_err[i] += float(e_sel.sum())
                self.bin_sum_abs[i] += float(np.abs(e_sel).sum())
                self.bin_sum_sq[i] += float(np.square(e_sel).sum())


def plot_lead_time_curves(
    named_results: Sequence[Tuple[str, Dict[str, object]]],
    save_path: str,
    thresholds: Sequence[str] = ("0.1", "0.5", "2", "4"),
) -> None:
    """Plot lead-time (T+6..T+60 min) CSI, HSS, FAR, BIAS, and MAE/RMSE curves."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=150)

    # Panels 0..3: CSI at main thresholds; Panel 4: FAR & BIAS @0.5; Panel 5: MAE & RMSE vs lead time
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for m_idx, (label, res) in enumerate(named_results):
        ls = res["lead_scores"]
        leads = sorted(int(k) for k in ls.keys())
        mins = [(l + 1) * 6 for l in leads]
        col = colors[m_idx % len(colors)]

        for ax_idx, thr in enumerate(thresholds[:4]):
            ax = axes[ax_idx // 3, ax_idx % 3]
            vals = [ls[str(l) if str(l) in ls else l][thr]["CSI"] for l in leads]
            ax.plot(mins, vals, marker="o", linewidth=2, markersize=4, color=col, label=label)
            ax.set_title(f"CSI @ {thr} mm/6min", fontsize=11, fontweight="bold")
            ax.set_xlabel("Lead Time (min)")
            ax.set_ylabel("CSI")
            ax.set_ylim(0.0, max(0.05, ax.get_ylim()[1]))
            ax.grid(True, linestyle="--", alpha=0.35)

        # Panel (1, 1): FAR & BIAS at 0.5 mm/6min
        ax_fb = axes[1, 1]
        far_vals = [ls[str(l) if str(l) in ls else l]["0.5"]["FAR"] for l in leads]
        bias_vals = [ls[str(l) if str(l) in ls else l]["0.5"]["BIAS"] for l in leads]
        ax_fb.plot(mins, far_vals, marker="s", linestyle="-", linewidth=2, markersize=4, color=col, label=f"{label} FAR@0.5")
        ax_fb.plot(mins, bias_vals, marker="^", linestyle="--", linewidth=1.8, markersize=4, color=col, alpha=0.8, label=f"{label} BIAS@0.5")
        ax_fb.axhline(1.0, color="gray", linestyle=":", linewidth=1)
        ax_fb.set_title("FAR & BIAS @ 0.5 mm/6min", fontsize=11, fontweight="bold")
        ax_fb.set_xlabel("Lead Time (min)")
        ax_fb.set_ylabel("Score")
        ax_fb.grid(True, linestyle="--", alpha=0.35)

        # Panel (1, 2): MAE & RMSE vs Lead Time
        ax_err = axes[1, 2]
        mae_vals = [ls[str(l) if str(l) in ls else l].get("MAE", np.nan) for l in leads]
        rmse_vals = [ls[str(l) if str(l) in ls else l].get("RMSE", np.nan) for l in leads]
        ax_err.plot(mins, mae_vals, marker="o", linestyle="-", linewidth=2, markersize=4, color=col, label=f"{label} MAE")
        ax_err.plot(mins, rmse_vals, marker="x", linestyle="--", linewidth=1.8, markersize=5, color=col, alpha=0.8, label=f"{label} RMSE")
        ax_err.set_title("Lead-Time MAE & RMSE (mm/6min)", fontsize=11, fontweight="bold")
        ax_err.set_xlabel("Lead Time (min)")
        ax_err.set_ylabel("Error (mm/6min)")
        ax_err.grid(True, linestyle="--", alpha=0.35)

    for ax in axes.ravel():
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("BJ Precipitation Nowcasting: Lead-Time Error & Skill Curves", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_error_case(
    gt: np.ndarray,
    evo: np.ndarray,
    pred: np.ndarray,
    save_path: str,
    case_id: str,
    issue_stamp: str,
    lead_indices: Sequence[int] = (1, 3, 5, 7, 9),
    hit_thr: float = 0.5,
) -> None:
    """Plot 5-row spatial error map across selected lead times on the 301x301 BJ domain.

    Rows:
      1. Ground Truth RA (mm/6min)
      2. Stage-1 Evolution Prediction (evo_rain)
      3. Final Prediction (pred_rain)
      4. Signed Error (Pred - GT): Red = Overestimation, Blue = Underestimation
      5. Contingency Map @ hit_thr mm/6min: Hit (Green), False Alarm (Red), Miss (Blue)
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    gt_c = np.maximum(_crop_valid(gt), 0.0)
    evo_c = np.maximum(_crop_valid(evo), 0.0)
    pred_c = np.maximum(_crop_valid(pred), 0.0)

    n_cols = len(lead_indices)
    fig, axes = plt.subplots(5, n_cols, figsize=(3.2 * n_cols + 1.2, 15.0), dpi=150)

    vmax_rain = max(4.0, float(np.percentile(np.concatenate([gt_c.ravel(), pred_c.ravel()]), 99.7)))
    err_all = pred_c - gt_c
    vmax_err = max(1.5, float(np.percentile(np.abs(err_all), 99.5)))

    # 0=Correct Neg (light gray), 1=Hit (green), 2=False Alarm (red), 3=Miss (blue)
    cmap_cat = ListedColormap(["#f3f4f6", "#16a34a", "#dc2626", "#2563eb"])

    row_titles = [
        "Ground Truth\n(mm/6min)",
        "Evolution Branch\n(mm/6min)",
        "Final Prediction\n(mm/6min)",
        "Signed Error\n(Pred - GT)",
        f"Hit / FA / Miss\n(@ {hit_thr:g} mm/6min)",
    ]

    im_rain = im_err = im_cat = None
    for c_idx, l_idx in enumerate(lead_indices):
        lead_min = (l_idx + 1) * 6
        g_l = gt_c[l_idx, 0]
        e_l = evo_c[l_idx, 0]
        p_l = pred_c[l_idx, 0]
        diff = p_l - g_l

        mae_l = float(np.abs(diff).mean())
        rmse_l = float(np.sqrt(np.mean(diff ** 2)))

        g_bin = g_l >= hit_thr
        p_bin = p_l >= hit_thr
        cat = np.zeros_like(g_l, dtype=np.int32)
        cat[g_bin & p_bin] = 1     # Hit
        cat[(~g_bin) & p_bin] = 2  # False Alarm
        cat[g_bin & (~p_bin)] = 3  # Miss

        im_rain = axes[0, c_idx].imshow(g_l, origin="lower", cmap="turbo", vmin=0.0, vmax=vmax_rain)
        axes[0, c_idx].set_title(f"T+{lead_min} min\nmax={g_l.max():.2f}", fontsize=10, fontweight="bold")

        axes[1, c_idx].imshow(e_l, origin="lower", cmap="turbo", vmin=0.0, vmax=vmax_rain)
        axes[1, c_idx].set_title(f"MAE={np.abs(e_l - g_l).mean():.3f}", fontsize=9)

        axes[2, c_idx].imshow(p_l, origin="lower", cmap="turbo", vmin=0.0, vmax=vmax_rain)
        axes[2, c_idx].set_title(f"MAE={mae_l:.3f} | max={p_l.max():.2f}", fontsize=9)

        im_err = axes[3, c_idx].imshow(diff, origin="lower", cmap="RdBu_r", vmin=-vmax_err, vmax=vmax_err)
        axes[3, c_idx].set_title(f"RMSE={rmse_l:.3f} | bias={diff.mean():+.3f}", fontsize=9)

        im_cat = axes[4, c_idx].imshow(cat, origin="lower", cmap=cmap_cat, vmin=0, vmax=3, interpolation="nearest")
        tp = int((g_bin & p_bin).sum())
        fp = int(((~g_bin) & p_bin).sum())
        fn = int((g_bin & (~p_bin)).sum())
        csi = tp / max(tp + fp + fn, 1)
        axes[4, c_idx].set_title(f"CSI={csi:.3f} (H:{tp} FA:{fp} M:{fn})", fontsize=8)

        for r in range(5):
            axes[r, c_idx].set_xticks([])
            axes[r, c_idx].set_yticks([])

    for r, rt in enumerate(row_titles):
        axes[r, 0].set_ylabel(rt, fontsize=10, fontweight="bold")

    fig.subplots_adjust(right=0.91, hspace=0.28, wspace=0.08, top=0.92)
    cax1 = fig.add_axes([0.925, 0.43, 0.015, 0.47])
    fig.colorbar(im_rain, cax=cax1, label="Rain (mm/6min)")

    cax2 = fig.add_axes([0.925, 0.24, 0.015, 0.16])
    fig.colorbar(im_err, cax=cax2, label="Pred - GT")

    cax3 = fig.add_axes([0.925, 0.06, 0.015, 0.15])
    cbar3 = fig.colorbar(im_cat, cax=cax3, ticks=[0.375, 1.125, 1.875, 2.625])
    cbar3.ax.set_yticklabels(["Neg", "Hit", "FA", "Miss"], fontsize=9)

    fig.suptitle(f"Spatial Error Diagnostic | Case: {case_id} | Issue: {issue_stamp} (301x301)", fontsize=13, fontweight="bold")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_error_distribution(
    acc: ErrorDistributionAccumulator,
    results: Dict[str, object],
    save_path: str,
) -> None:
    """Plot residual histogram (Pred - GT) and intensity-binned bias/RMSE breakdown."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)

    centers = 0.5 * (acc.hist_edges[:-1] + acc.hist_edges[1:])
    width = acc.hist_edges[1] - acc.hist_edges[0]

    # Panel 0: Wet-pixel error distribution
    ax0 = axes[0]
    wet_tot = max(int(acc.hist_counts_wet.sum()), 1)
    ax0.bar(centers, acc.hist_counts_wet / wet_tot, width=width * 0.9, color="#2563eb", alpha=0.8)
    ax0.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax0.set_title("Wet-Area Residual Distribution (Pred - GT)", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Pred - GT (mm/6min)")
    ax0.set_ylabel("Fraction of Wet Pixels")
    ax0.grid(True, linestyle="--", alpha=0.35)

    # Panel 1: Mean Bias & RMSE across true rainfall intensity bins
    ax1 = axes[1]
    cnt = np.maximum(acc.bin_count, 1)
    mean_bias = acc.bin_sum_err / cnt
    bin_rmse = np.sqrt(acc.bin_sum_sq / cnt)
    x = np.arange(len(acc.BIN_LABELS))
    w = 0.36
    ax1.bar(x - w / 2, mean_bias, width=w, color="#dc2626", alpha=0.85, label="Mean Bias (Pred - GT)")
    ax1.bar(x + w / 2, bin_rmse, width=w, color="#0284c7", alpha=0.85, label="RMSE")
    ax1.axhline(0.0, color="black", linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(acc.BIN_LABELS, fontsize=9)
    ax1.set_title("Error by Ground-Truth Rain Intensity Bin", fontsize=11, fontweight="bold")
    ax1.set_ylabel("mm/6min")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.35)

    # Panel 2: Lead-time MAE/RMSE
    ax2 = axes[2]
    ls = results["lead_scores"]
    leads = sorted(int(k) for k in ls.keys())
    mins = [(l + 1) * 6 for l in leads]
    maes = [ls[str(l) if str(l) in ls else l].get("MAE", np.nan) for l in leads]
    rmses = [ls[str(l) if str(l) in ls else l].get("RMSE", np.nan) for l in leads]
    ax2.plot(mins, maes, marker="o", linewidth=2, color="#16a34a", label=f"MAE (avg={results['MAE']:.4f})")
    ax2.plot(mins, rmses, marker="s", linewidth=2, color="#ea580c", label=f"RMSE (avg={results['RMSE']:.4f})")
    ax2.set_title("Lead-Time Overall MAE & RMSE", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Lead Time (min)")
    ax2.set_ylabel("mm/6min")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("BJ Full-Field Precipitation Error Analysis (301x301)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: Dict[str, List[dict]], save_path: str) -> None:
    """Plot Stage-1 and Stage-2 training loss and validation MAE/CSI curves."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    s1 = history.get("stage1", [])
    s2 = history.get("stage2", [])
    if not s1 and not s2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    # Build continuous epoch index
    ep_s1 = list(range(1, len(s1) + 1))
    ep_s2 = list(range(len(s1) + 1, len(s1) + len(s2) + 1))

    ax0 = axes[0]
    if s1:
        ax0.plot(ep_s1, [x["loss"] for x in s1], marker="o", label="Stage1 Evo Loss")
    if s2:
        ax0.plot(ep_s2, [x["g_loss"] for x in s2], marker="s", label="Stage2 G Loss")
        ax0.plot(ep_s2, [x["d_loss"] for x in s2], marker="^", label="Stage2 D Loss")
    ax0.set_title("Training Loss")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.legend(fontsize=8)
    ax0.grid(True, linestyle="--", alpha=0.35)

    all_records = s1 + s2
    all_eps = ep_s1 + ep_s2
    ax1 = axes[1]
    ax1.plot(all_eps, [x["val_mae"] for x in all_records], marker="o", color="#ea580c", label="Val MAE")
    if s1 and s2:
        ax1.axvline(len(s1) + 0.5, color="gray", linestyle="--", label="Stage1->Stage2")
    ax1.set_title("Validation MAE")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MAE (mm/6min)")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = axes[2]
    ax2.plot(all_eps, [x["val_csi_0.1"] for x in all_records], marker="o", label="CSI@0.1")
    ax2.plot(all_eps, [x["val_csi_0.5"] for x in all_records], marker="s", label="CSI@0.5")
    ax2.plot(all_eps, [x["val_csi_2"] for x in all_records], marker="^", label="CSI@2.0")
    if s1 and s2:
        ax2.axvline(len(s1) + 0.5, color="gray", linestyle="--")
    ax2.set_title("Validation CSI")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("CSI")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("Training & Validation Error / Skill Curves", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
