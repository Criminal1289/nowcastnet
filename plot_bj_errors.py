"""Quick single-sample spatial error plotter (or multi-run curve comparator) for BJ dataset.

Default behavior (no arguments, direct PyCharm Run):
  - Finds the latest trained checkpoint under runs/
  - Runs inference on ONE sample (takes ~2s)
  - Saves spatial error map & log named with YYYYMMDD_HHMMSS inside that run's folder.

Multi-run comparison mode:
  python code/plot_bj_errors.py --compare_runs runs/20261118_150000 runs/20261118_160000
"""

import argparse
from datetime import datetime
import glob
import json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from types import SimpleNamespace
import torch

from eval_bj import find_latest_ckpt
from nowcasting.data_provider import datasets_factory
from nowcasting.metrics.plotting import plot_lead_time_curves, plot_spatial_error_case, plot_training_curves
from nowcasting.models.nwp_nowcastnet import NWPNowcastNet


def _find_latest_eval_json(run_dir: str, split: str) -> str:
    matches = glob.glob(os.path.join(run_dir, f"eval_{split}_*.json"))
    legacy = os.path.join(run_dir, f"eval_{split}.json")
    if os.path.isfile(legacy):
        matches.append(legacy)
    if not matches:
        raise FileNotFoundError(f"No eval_{split}*.json found in {run_dir}")
    return max(matches, key=os.path.getmtime)


def main():
    p = argparse.ArgumentParser(description="Plot single-sample spatial error map or compare multi-run curves")
    p.add_argument("--ckpt", type=str, default="", help="Checkpoint path (defaults to latest under runs/)")
    p.add_argument("--split", type=str, default="test", choices=["val", "test", "robustness_missing"])
    p.add_argument("--sample_idx", type=int, default=0, help="Dataset sample index to plot (1 sample only)")
    p.add_argument("--hit_thr", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--compare_runs", nargs="*", default=[], help="Compare eval_<split>*.json across run dirs")
    p.add_argument("--labels", nargs="*", default=[])
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.compare_runs:
        named_results = []
        for idx, run_dir in enumerate(args.compare_runs):
            if not os.path.isdir(run_dir):
                alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), run_dir)
                if os.path.isdir(alt):
                    run_dir = alt
            eval_file = _find_latest_eval_json(run_dir, args.split)
            with open(eval_file, "r", encoding="utf-8") as f:
                res = json.load(f)
            label = args.labels[idx] if idx < len(args.labels) else os.path.basename(os.path.normpath(run_dir))
            named_results.append((label, res))

        out_path = args.out or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "runs",
            f"compare_{args.split}_{run_ts}.png",
        )
        plot_lead_time_curves(named_results, out_path)
        print(f"Saved comparison error curves -> {out_path}")
        return

    # Single-sample fast spatial error plot
    ckpt_path = find_latest_ckpt(args.ckpt)
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    cfg_path = os.path.join(ckpt_dir, "config.json")
    cfg_dict = json.load(open(cfg_path, "r", encoding="utf-8")) if os.path.isfile(cfg_path) else {}
    cfg_dict.setdefault("dataset_name", "bj")
    cfg_dict.setdefault("input_length", 11)
    cfg_dict.setdefault("pred_length", 10)
    cfg_dict.setdefault("radar_ch", 10)
    cfg_dict.setdefault("nwp_ch", 27)
    cfg_dict.setdefault("pad_to", 320)
    cfg_dict.setdefault("ngf", 32)
    cfg_dict["cpu_worker"] = 0
    cfg_dict["eval_batch_size"] = 1
    cfg_dict["eval_stride_steps"] = 1
    cfg_dict["device"] = args.device if torch.cuda.is_available() else "cpu"

    configs = SimpleNamespace(**cfg_dict)
    loader = datasets_factory.data_provider(configs, split=args.split)
    sample = loader.dataset[args.sample_idx]

    model = NWPNowcastNet(configs).to(configs.device)
    ckpt = torch.load(ckpt_path, map_location=configs.device)
    state = ckpt["generator"] if isinstance(ckpt, dict) and "generator" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        x_r = sample["x_radar"].unsqueeze(0).to(configs.device)
        x_n = None if getattr(configs, "no_nwp", False) else sample["x_nwp"].unsqueeze(0).to(configs.device)
        coord = sample["coord_yx"].unsqueeze(0).to(configs.device)
        out = model(x_r, x_nwp=x_n, coord_yx=coord, run_gen=True)

    cid = sample["case_id"]
    stamp = sample["issue_stamp"]
    y_np = sample["y_rain"].numpy()
    evo_np = out["evo_rain"][0].detach().cpu().numpy()
    pred_np = out["pred_rain"][0].detach().cpu().numpy()

    out_png = args.out or os.path.join(ckpt_dir, f"spatial_error_{run_ts}_{cid}_{stamp}.png")
    plot_spatial_error_case(y_np, evo_np, pred_np, out_png, case_id=cid, issue_stamp=stamp, hit_thr=args.hit_thr)

    log_path = os.path.join(ckpt_dir, f"plot_{run_ts}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"timestamp={run_ts}\nckpt={ckpt_path}\nsplit={args.split}\nsample_idx={args.sample_idx}\ncase_id={cid}\nissue_stamp={stamp}\noutput_png={out_png}\n")

    print(f"Saved single-sample spatial error map -> {out_png}")
    print(f"Saved log -> {log_path}")


if __name__ == "__main__":
    main()
