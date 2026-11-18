"""Training entrypoint for NWP-ST-NowcastNet on the BJ dataset."""

import argparse
from datetime import datetime
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import random
import sys
import numpy as np
import torch

from nowcasting.data_provider import datasets_factory
from nowcasting.models.discriminator import SpatioTemporalDiscriminator
from nowcasting.models.nwp_nowcastnet import NWPNowcastNet
from nowcasting.trainer import BJTrainer


class _TeeStream:
    """Mirrors stdout/stderr to a timestamped log file."""

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train NWP-ST-NowcastNet on BJ dataset")
    p.add_argument("--dataset_name", type=str, default="bj")
    p.add_argument("--dataset_path", type=str, default=r"C:\Users\17878\Desktop\gestdgn\data\BJ")
    p.add_argument("--save_dir", type=str, default="runs")
    p.add_argument("--exp_name", type=str, default="", help="Optional experiment tag prefix before YYYYMMDD_HHMMSS")
    p.add_argument("--timestamp_run", type=str2bool, default=True, help="Append YYYYMMDD_HHMMSS folder & log naming")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu_worker", type=int, default=2)

    # Data / IO contract (MASTER.md)
    p.add_argument("--input_length", type=int, default=11)
    p.add_argument("--pred_length", type=int, default=10)
    p.add_argument("--radar_ch", type=int, default=10)
    p.add_argument("--nwp_ch", type=int, default=27)
    p.add_argument("--patch_size", type=int, default=160, help="Train patch size (multiple of 32; 0 for full 320x320)")
    p.add_argument("--pad_to", type=int, default=320)
    p.add_argument("--stride_steps", type=int, default=1)
    p.add_argument("--eval_stride_steps", type=int, default=1)
    p.add_argument("--filter_dry", type=str2bool, default=True)
    p.add_argument("--dry_keep_prob", type=float, default=0.2)
    p.add_argument("--rain_threshold", type=float, default=0.1)

    # Model & 4 ablation toggles
    p.add_argument("--ngf", type=int, default=32)
    p.add_argument("--use_st_bias", type=str2bool, default=True)
    p.add_argument("--use_nwp_motion", type=str2bool, default=True)
    p.add_argument("--use_nwp_noise", type=str2bool, default=True)
    p.add_argument("--use_temporal_radar", type=str2bool, default=True)
    p.add_argument("--no_nwp", type=str2bool, default=False)

    # Training schedule
    p.add_argument("--stage", type=str, default="both", choices=["1", "2", "both"])
    p.add_argument("--resume_ckpt", type=str, default="")
    p.add_argument("--epochs_stage1", type=int, default=10)
    p.add_argument("--epochs_stage2", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr_evo", type=float, default=3e-4)
    p.add_argument("--lr_gen", type=float, default=2e-4)
    p.add_argument("--lr_disc", type=float, default=2e-4)
    p.add_argument("--freeze_evo_stage2", type=str2bool, default=False)
    p.add_argument("--amp", type=str2bool, default=True)
    p.add_argument("--clip_grad", type=float, default=1.0)

    # Loss hyperparameters
    p.add_argument("--alpha_csi", type=float, default=0.35)
    p.add_argument("--pool_weight", type=float, default=0.5)
    p.add_argument("--adv_spatial_weight", type=float, default=0.05)
    p.add_argument("--adv_temporal_weight", type=float, default=0.05)
    p.add_argument("--motion_smooth_weight", type=float, default=0.02)
    return p


def main():
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    set_seed(args.seed)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_timestamp = run_ts
    if args.timestamp_run:
        tag = f"{args.exp_name}_{run_ts}" if args.exp_name else run_ts
        args.save_dir = os.path.join(args.save_dir, tag)
    os.makedirs(args.save_dir, exist_ok=True)

    log_path = os.path.join(args.save_dir, f"{run_ts}.log")
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    with open(log_path, "w", encoding="utf-8") as log_f:
        sys.stdout = _TeeStream(orig_stdout, log_f)
        sys.stderr = _TeeStream(orig_stderr, log_f)
        try:
            print(f"Run timestamp: {run_ts} | Save dir: {os.path.abspath(args.save_dir)}")
            train_loader = datasets_factory.data_provider(args, split="train")
            val_loader = datasets_factory.data_provider(args, split="val")
            print(f"Loaded BJ dataset -> train batches: {len(train_loader)}, val batches: {len(val_loader)}")

            model = NWPNowcastNet(args).to(args.device)
            disc = SpatioTemporalDiscriminator(cond_channels=2, base_c=args.ngf).to(args.device)

            if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
                ckpt = torch.load(args.resume_ckpt, map_location=args.device)
                model.load_state_dict(ckpt["generator"], strict=False)
                if "discriminator" in ckpt:
                    disc.load_state_dict(ckpt["discriminator"], strict=False)
                print(f"Resumed checkpoint from {args.resume_ckpt}")

            trainer = BJTrainer(model, disc, args)

            if args.stage in ("1", "both"):
                trainer.train_stage1(train_loader, val_loader, epochs=args.epochs_stage1)
                best1 = os.path.join(args.save_dir, "best_stage1.pt")
                if os.path.isfile(best1) and args.stage == "both":
                    ckpt = torch.load(best1, map_location=args.device)
                    model.load_state_dict(ckpt["generator"])
                    print(f"Loaded best Stage-1 weights from {best1} for Stage-2 training")

            if args.stage in ("2", "both"):
                trainer.train_stage2(train_loader, val_loader, epochs=args.epochs_stage2)
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr


if __name__ == "__main__":
    main()
