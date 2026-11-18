"""Timestamp indexing, irregular-case filtering, and window construction for BJ dataset.

Follows MASTER.md §3.1 and §3.3:
- Uses proper calendar arithmetic on MMDD-HHMM timestamps (never naive integer math).
- Separates 23 IRREGULAR_CASES from 39 clean cases (23 train / 8 val / 8 test).
- Builds 11-frame radar history [t-60, ..., t] and 10-frame label [t+6, ..., t+60] by timestamp.
- Aligns NWP via FORECAST strategy: (analysis_hour, forecast_hour covering t+60).
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from nowcasting.data_provider.bj_transforms import RADAR_PRODS, NWP_VARS

# Reference non-leap year for MMDD-HHMM calendar arithmetic (BJ data is May-Sept)
_REF_YEAR = 2021

IRREGULAR_CASES = frozenset({
    "CP_01_05301413_00001",
    "CP_01_06220410_00001",
    "CP_01_06230359_00001",
    "CP_01_06251506_00001",
    "CP_01_06252006_00001",
    "CP_01_06280022_00001",
    "CP_01_06300430_00001",
    "CP_01_07011914_00001",
    "CP_01_07061806_00001",
    "CP_01_07120834_00001",
    "CP_01_07161413_00001",
    "CP_01_07291341_00001",
    "CP_01_08020529_00001",
    "CP_01_08051911_00001",
    "CP_01_08061235_00001",
    "CP_01_08092015_00001",
    "CP_01_08111925_00001",
    "CP_01_08120710_00001",
    "CP_01_08140134_00001",
    "CP_01_08161336_00001",
    "CP_01_09042035_00001",
    "CP_01_09071830_00001",
    "CP_01_09260116_00001",
})


def parse_stamp_from_filename(fname: str) -> Optional[str]:
    m = re.search(r"(\d{4}-\d{4})", fname)
    return m.group(1) if m else None


def stamp_to_dt(stamp: str) -> datetime:
    mmdd, hhmm = stamp.split("-")
    return datetime(_REF_YEAR, int(mmdd[:2]), int(mmdd[2:]), int(hhmm[:2]), int(hhmm[2:]))


def dt_to_stamp(dt: datetime) -> str:
    return dt.strftime("%m%d-%H%M")


def add_minutes(stamp: str, minutes: int) -> str:
    return dt_to_stamp(stamp_to_dt(stamp) + timedelta(minutes=minutes))


def minutes_between(a: str, b: str) -> int:
    return int((stamp_to_dt(b) - stamp_to_dt(a)).total_seconds() // 60)


def floor_hour_stamp(stamp: str) -> str:
    dt = stamp_to_dt(stamp)
    return dt_to_stamp(dt.replace(minute=0))


def ceil_or_cover_hour_stamp(stamp: str) -> str:
    """Return the hourly NWP stamp covering `stamp` (e.g., 16:00 covers 15:06..16:00)."""
    dt = stamp_to_dt(stamp)
    if dt.minute == 0:
        return dt_to_stamp(dt)
    return dt_to_stamp((dt.replace(minute=0) + timedelta(hours=1)))


@dataclass
class SampleSpec:
    case_id: str
    issue_stamp: str
    radar_stamps: List[Optional[str]]          # length T_in (11); None means missing slot to interpolate
    radar_interp_pairs: List[Optional[Tuple[str, str]]]  # (prev_stamp, next_stamp) for missing slot
    label_stamps: List[str]                    # length T_out (10)
    nwp_stamps: Tuple[str, str]                # (analysis_stamp, forecast_stamp)
    missing_radar_steps: List[int]             # indices in [0, T_in) that were interpolated


def scan_cases(data_root: str) -> List[str]:
    """Return sorted list of valid BJ case directories."""
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"BJ dataset root not found: {data_root}")
    return sorted(
        c for c in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, c))
    )


def split_cases(data_root: str, split: str) -> List[str]:
    """Deterministic case-level split matching MASTER.md §3.1 (23 train / 8 val / 8 test)."""
    all_cases = scan_cases(data_root)
    clean_cases = [c for c in all_cases if c not in IRREGULAR_CASES]
    irregular_cases = [c for c in all_cases if c in IRREGULAR_CASES]

    if len(clean_cases) >= 39:
        train_cases = clean_cases[:23]
        val_cases = clean_cases[23:31]
        test_cases = clean_cases[31:39]
    else:
        n = len(clean_cases)
        n_train = max(1, int(round(n * 23 / 39)))
        n_val = max(1, int(round(n * 8 / 39)))
        train_cases = clean_cases[:n_train]
        val_cases = clean_cases[n_train:n_train + n_val]
        test_cases = clean_cases[n_train + n_val:]

    if split == "train":
        return train_cases
    if split == "val":
        return val_cases
    if split == "test":
        return test_cases
    if split == "robustness_missing":
        return irregular_cases
    if split == "all_clean":
        return clean_cases
    if split == "all":
        return all_cases
    raise ValueError(f"Unknown split: {split}")


def build_case_file_maps(case_dir: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, str]]]:
    """Index all available files by timestamp for RADAR, LABEL/RA, and NWP."""
    radar_map: Dict[str, Dict[str, str]] = {}
    for prod in RADAR_PRODS:
        prod_dir = os.path.join(case_dir, "RADAR", prod)
        pmap: Dict[str, str] = {}
        if os.path.isdir(prod_dir):
            for f in os.listdir(prod_dir):
                if f.endswith(".npy"):
                    ts = parse_stamp_from_filename(f)
                    if ts is not None:
                        pmap[ts] = os.path.join(prod_dir, f)
        radar_map[prod] = pmap

    label_map: Dict[str, str] = {}
    ra_dir = os.path.join(case_dir, "LABEL", "RA")
    if os.path.isdir(ra_dir):
        for f in os.listdir(ra_dir):
            if f.endswith(".npy"):
                ts = parse_stamp_from_filename(f)
                if ts is not None:
                    label_map[ts] = os.path.join(ra_dir, f)

    nwp_map: Dict[str, Dict[str, str]] = {}
    for var in NWP_VARS:
        var_dir = os.path.join(case_dir, "NWP", var)
        vmap: Dict[str, str] = {}
        if os.path.isdir(var_dir):
            for f in os.listdir(var_dir):
                if f.endswith(".npy"):
                    ts = parse_stamp_from_filename(f)
                    if ts is not None:
                        vmap[ts] = os.path.join(var_dir, f)
        nwp_map[var] = vmap

    return radar_map, label_map, nwp_map


def build_case_windows(
    case_id: str,
    case_dir: str,
    input_length: int = 11,
    pred_length: int = 10,
    stride_steps: int = 1,
    allow_interpolate: bool = False,
) -> List[SampleSpec]:
    """Construct timestamp-verified sliding windows for a single BJ case."""
    radar_map, label_map, nwp_map = build_case_file_maps(case_dir)

    # Timestamp is considered complete for radar iff all 10 products exist
    common_radar_stamps = set.intersection(*(set(radar_map[p].keys()) for p in RADAR_PRODS))
    common_nwp_stamps = set.intersection(*(set(nwp_map[v].keys()) for v in NWP_VARS))
    valid_label_stamps = set(label_map.keys())

    if not common_radar_stamps or not valid_label_stamps or not common_nwp_stamps:
        return []

    candidate_issues = sorted(common_radar_stamps, key=stamp_to_dt)
    specs: List[SampleSpec] = []

    for idx in range(0, len(candidate_issues), stride_steps):
        issue = candidate_issues[idx]
        # History: issue - (input_length-1)*6 ... issue
        hist_slots = [add_minutes(issue, -6 * (input_length - 1 - k)) for k in range(input_length)]
        # Future labels: issue + 6 ... issue + pred_length*6
        fut_slots = [add_minutes(issue, 6 * (k + 1)) for k in range(pred_length)]

        # All future label frames must exist
        if not all(s in valid_label_stamps for s in fut_slots):
            continue

        # NWP FORECAST alignment: analysis hour + forecast hour covering last target
        nwp_t0 = floor_hour_stamp(issue)
        nwp_t1 = ceil_or_cover_hour_stamp(fut_slots[-1])
        if nwp_t0 not in common_nwp_stamps or nwp_t1 not in common_nwp_stamps:
            continue

        # Check radar slots; optionally allow single-frame 12-min gaps interpolated from neighbors
        radar_stamps: List[Optional[str]] = []
        radar_interp_pairs: List[Optional[Tuple[str, str]]] = []
        missing_steps: List[int] = []
        window_ok = True

        for k, s in enumerate(hist_slots):
            if s in common_radar_stamps:
                radar_stamps.append(s)
                radar_interp_pairs.append(None)
            elif allow_interpolate:
                prev_s = add_minutes(s, -6)
                next_s = add_minutes(s, 6)
                if prev_s in common_radar_stamps and next_s in common_radar_stamps:
                    radar_stamps.append(None)
                    radar_interp_pairs.append((prev_s, next_s))
                    missing_steps.append(k)
                else:
                    window_ok = False
                    break
            else:
                window_ok = False
                break

        if not window_ok:
            continue

        specs.append(
            SampleSpec(
                case_id=case_id,
                issue_stamp=issue,
                radar_stamps=radar_stamps,
                radar_interp_pairs=radar_interp_pairs,
                label_stamps=fut_slots,
                nwp_stamps=(nwp_t0, nwp_t1),
                missing_radar_steps=missing_steps,
            )
        )

    return specs
