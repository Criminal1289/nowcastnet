"""Full-field or single-sample 301x301 evaluation & error plotting script for BJ dataset.

Reads config.json alongside the checkpoint to prevent architecture/flag mismatch (MASTER.md §5).
Writes timestamped logs and figures (YYYYMMDD_HHMMSS) inside the checkpoint's run directory.
"""

import argparse
from datetime import datetime
import glob
import json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from types import SimpleNamespace
import numpy as np
import torch

from nowcasting.data_provider import datasets_factory
from nowcasting.metrics.plotting import (
    ErrorDistributionAccumulator,
    plot_error_distribution,
    plot_lead_time_curves,
    plot_spatial_error_case,
)
from nowcasting.metrics.precipitation import DEFAULT_THRESHOLDS, PrecipitationVerifier
from nowcasting.models.nwp_nowcastnet import NWPNowcastNet


class _TeeStream:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def find_latest_ckpt(ckpt_arg: str = "") -> str:
    """Resolve explicit checkpoint path, or pick the newest best_stage2.pt under runs/."""
    if ckpt_arg:
        if os.path.isfile(ckpt_arg):
            return ckpt_arg
        code_rel = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt_arg)
        if os.path.isfile(code_rel):
            return code_rel

    search_roots = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"),
        "runs",
    ]
    candidates = []
    for r in search_roots:
        if os.path.isdir(r):
            candidates.extend(glob.glob(os.path.join(r, "**", "best_stage2.pt"), recursive=True))
    if not candidates:
        for r in search_roots:
            if os.path.isdir(r):
                candidates.extend(glob.glob(os.path.join(r, "**", "*.pt"), recursive=True))
    if not candidates:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_arg!r}")
    return max(candidates, key=os.path.getmtime)


def main():
    p = argparse.ArgumentParser(description="Evaluate & Plot Errors for NWP-ST-NowcastNet on BJ dataset")
    p.add_argument("--ckpt", type=str, default="", help="Checkpoint path (defaults to latest under runs/)")
    p.add_argument("--dataset_path", type=str, default="")
    p.add_argument("--split", type=str, default="test", choices=["val", "test", "robustness_missing"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--cpu_worker", type=int, default=0)
    p.add_argument("--eval_stride_steps", type=int, default=1)
    p.add_argument("--sample_idx", type=int, default=-1, help="If >=0, evaluate/plot ONLY this single sample index")
    p.add_argument("--max_samples", type=int, default=0, help="If >0, stop after this many samples")
    p.add_argument("--evo_only", action="store_true", help="Evaluate Stage-1 evolution output only")
    p.add_argument("--save_plots", action="store_true", default=True)
    p.add_argument("--no_plots", action="store_true", help="Disable saving PNG error figures")
    p.add_argument("--num_plot_samples", type=int, default=1, help="Number of distinct samples to plot spatial error maps")
    p.add_argument("--hit_thr", type=float, default=0.5, help="Threshold (mm/6min) for Hit/FA/Miss map")
    p.add_argument("--out_json", type=str, default="")
    p.add_argument("--plot_dir", type=str, default="")
    args = p.parse_args()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = find_latest_ckpt(args.ckpt)
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
    else:
        cfg_dict = {}

    cfg_dict.setdefault("dataset_name", "bj")
    cfg_dict.setdefault("input_length", 11)
    cfg_dict.setdefault("pred_length", 10)
    cfg_dict.setdefault("radar_ch", 10)
    cfg_dict.setdefault("nwp_ch", 27)
    cfg_dict.setdefault("pad_to", 320)
    cfg_dict.setdefault("ngf", 32)
    cfg_dict["cpu_worker"] = args.cpu_worker
    cfg_dict["eval_batch_size"] = 1
    cfg_dict["eval_stride_steps"] = args.eval_stride_steps
    if args.dataset_path:
        cfg_dict["dataset_path"] = args.dataset_path
    cfg_dict["device"] = args.device if torch.cuda.is_available() else "cpu"

    log_path = os.path.join(ckpt_dir, f"eval_{args.split}_{run_ts}.log")
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    with open(log_path, "w", encoding="utf-8") as log_f:
        sys.stdout = _TeeStream(orig_stdout, log_f)
        sys.stderr = _TeeStream(orig_stderr, log_f)
        try:
            configs = SimpleNamespace(**cfg_dict)
            loader = datasets_factory.data_provider(configs, split=args.split)

            model = NWPNowcastNet(configs).to(configs.device)
            ckpt = torch.load(ckpt_path, map_location=configs.device)
            state = ckpt["generator"] if isinstance(ckpt, dict) and "generator" in ckpt else ckpt
            model.load_state_dict(state)
            model.eval()

            run_gen = not args.evo_only
            do_plots = args.save_plots and (not args.no_plots)
            verifier = PrecipitationVerifier(thresholds=DEFAULT_THRESHOLDS, num_leads=configs.pred_length)
            err_acc = ErrorDistributionAccumulator()
            best_per_case = {}
            processed = 0

            with torch.no_grad():
                if args.sample_idx >= 0:
                    sample = loader.dataset[args.sample_idx]
                    batches = [{
                        k: (v.unsqueeze(0) if torch.is_tensor(v) else [v])
                        for k, v in sample.items()
                    }]
                else:
                    batches = loader

                for batch in batches:
                    x_r = batch["x_radar"].to(configs.device)
                    x_n = None if getattr(configs, "no_nwp", False) else batch["x_nwp"].to(configs.device)
                    coord = batch["coord_yx"].to(configs.device)
                    y = batch["y_rain"].to(configs.device)
                    mask = batch["valid_mask"].to(configs.device)
                    out = model(x_r, x_nwp=x_n, coord_yx=coord, run_gen=run_gen)
                    pred = out["pred_rain"] if run_gen else out["evo_rain"]
                    verifier.update(pred, y, mask)
                    processed += 1

                    if do_plots:
                        pred_np = pred.detach().cpu().numpy()
                        evo_np = out["evo_rain"].detach().cpu().numpy()
                        y_np = y.detach().cpu().numpy()
                        mask_np = mask.detach().cpu().numpy()
                        err_acc.update(pred_np, y_np, mask_np)

                        if args.num_plot_samples > 0:
                            cid = batch["case_id"][0]
                            stamp = batch["issue_stamp"][0]
                            y_valid = y_np[0, :, :, :301, :301]
                            score = float(np.sum(y_valid >= 0.5) + 10.0 * np.sum(y_valid >= 2.0))
                            prev = best_per_case.get(cid)
                            if prev is None or score > prev[0]:
                                best_per_case[cid] = (score, cid, stamp, y_np[0], evo_np[0], pred_np[0])

                    if args.max_samples > 0 and processed >= args.max_samples:
                        break

            results = verifier.compute()
            print(f"=== Eval split={args.split} | ts={run_ts} | ckpt={ckpt_path} | n_samples={processed} ===")
            print(f"MAE={results['MAE']:.4f} | RMSE={results['RMSE']:.4f}")
            for thr in DEFAULT_THRESHOLDS:
                m = results["overall"][f"{thr:g}"]
                print(
                    f"  thr={thr:>4g} mm/6min | CSI={m['CSI']:.4f} | HSS={m['HSS']:.4f} | "
                    f"POD={m['POD']:.4f} | FAR={m['FAR']:.4f} | BIAS={m['BIAS']:.4f} | FSS={m['FSS']:.4f}"
                )

            out_path = args.out_json or os.path.join(ckpt_dir, f"eval_{args.split}_{run_ts}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Saved metrics -> {out_path}")
            print(f"Saved log     -> {log_path}")

            if do_plots:
                plot_dir = args.plot_dir or os.path.join(ckpt_dir, f"plots_{args.split}_{run_ts}")
                os.makedirs(plot_dir, exist_ok=True)
                model_label = os.path.basename(ckpt_dir) + (" (Evo)" if args.evo_only else " (Full)")
                curves_path = os.path.join(plot_dir, f"lead_time_curves_{run_ts}.png")
                dist_path = os.path.join(plot_dir, f"error_distribution_{run_ts}.png")
                plot_lead_time_curves([(model_label, results)], curves_path)
                plot_error_distribution(err_acc, results, dist_path)
                print(f"Saved error curves -> {curves_path}")
                print(f"Saved error dist   -> {dist_path}")

                top_cases = sorted(best_per_case.values(), key=lambda x: x[0], reverse=True)[:args.num_plot_samples]
                for k, (_, cid, stamp, y_c, evo_c, pred_c) in enumerate(top_cases):
                    sp_path = os.path.join(plot_dir, f"spatial_error_{run_ts}_{k:02d}_{cid}_{stamp}.png")
                    plot_spatial_error_case(y_c, evo_c, pred_c, sp_path, case_id=cid, issue_stamp=stamp, hit_thr=args.hit_thr)
                    print(f"Saved spatial map  -> {sp_path}")
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr


if __name__ == "__main__":
    main()
