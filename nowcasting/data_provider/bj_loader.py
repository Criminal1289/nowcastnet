"""PyTorch Dataset for the BJ Radar+NWP precipitation nowcasting benchmark.

Implements MASTER.md §1.2 and §3:
- RADAR: (11, 10, H, W)
- NWP:   (2, 27, H, W) resampled via 3x bilinear upsample + 301x301 center crop
- LABEL: (10, 1, H, W) in mm/6min
- Supports full-field padded (320x320) for val/test and optional random patch crop (e.g., 160x160 or 320x320) for training.
- Always provides canonical [-1, 1] spatial coordinate grid `coord_yx` (2, H, W) and `valid_mask` (1, H, W) for exact 301x301 evaluation.
"""

from collections import OrderedDict
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from nowcasting.data_provider.bj_index import (
    SampleSpec,
    build_case_file_maps,
    build_case_windows,
    split_cases,
)
from nowcasting.data_provider.bj_transforms import (
    NWP_VARS,
    RADAR_PRODS,
    RADAR_SIZE,
    bilinear_upsample_center_crop,
    clean_label_rain,
    clean_nwp_field,
    clean_radar_field,
)


class _LRUFrameCache:
    """Bounded per-worker LRU cache for cleaned 2D arrays to accelerate sliding windows."""

    def __init__(self, max_items: int = 4096):
        self.max_items = max_items
        self._cache: OrderedDict[Tuple[str, str], np.ndarray] = OrderedDict()

    def get(self, key: Tuple[str, str]) -> Optional[np.ndarray]:
        val = self._cache.get(key)
        if val is not None:
            self._cache.move_to_end(key)
        return val

    def put(self, key: Tuple[str, str], val: np.ndarray) -> None:
        if self.max_items <= 0:
            return
        self._cache[key] = val
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)


class BJDataset(Dataset):
    """BJ Radar + NWP dataset returning aligned (radar, nwp, label) windows."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        input_length: int = 11,
        pred_length: int = 10,
        stride_steps: int = 1,
        allow_interpolate: Optional[bool] = None,
        patch_size: int = 0,
        pad_to: int = 320,
        filter_dry: bool = False,
        dry_keep_prob: float = 0.2,
        rain_threshold: float = 0.1,
        seed: int = 42,
        cache_frames: int = 4096,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.input_length = input_length
        self.pred_length = pred_length
        self.patch_size = patch_size
        self.pad_to = pad_to
        self.filter_dry = filter_dry
        self.dry_keep_prob = dry_keep_prob
        self.rain_threshold = rain_threshold
        self.rng = np.random.RandomState(seed)

        if allow_interpolate is None:
            allow_interpolate = (split == "robustness_missing")
        self.allow_interpolate = allow_interpolate

        self.cases = split_cases(data_root, split)
        self.case_maps: Dict[str, tuple] = {}
        self.specs: List[SampleSpec] = []

        for cid in self.cases:
            cdir = os.path.join(data_root, cid)
            self.case_maps[cid] = build_case_file_maps(cdir)
            case_specs = build_case_windows(
                cid,
                cdir,
                input_length=input_length,
                pred_length=pred_length,
                stride_steps=stride_steps,
                allow_interpolate=allow_interpolate,
            )
            self.specs.extend(case_specs)

        if filter_dry and split == "train" and len(self.specs) > 0:
            self.specs = self._apply_dry_filter(self.specs)

        self._cache = _LRUFrameCache(max_items=cache_frames)

        # Canonical [-1, 1] coordinate grid on the 301x301 domain
        gy = np.linspace(-1.0, 1.0, RADAR_SIZE, dtype=np.float32)
        gx = np.linspace(-1.0, 1.0, RADAR_SIZE, dtype=np.float32)
        yy, xx = np.meshgrid(gy, gx, indexing="ij")
        self.full_coord_yx = np.stack([yy, xx], axis=0)  # (2, 301, 301)

    def _apply_dry_filter(self, specs: List[SampleSpec]) -> List[SampleSpec]:
        kept: List[SampleSpec] = []
        for sp in specs:
            _, label_map, _ = self.case_maps[sp.case_id]
            # Quick check on middle & last target frames to decide if window is wet
            is_wet = False
            for idx in (0, len(sp.label_stamps) // 2, len(sp.label_stamps) - 1):
                arr = clean_label_rain(np.load(label_map[sp.label_stamps[idx]]))
                if float(arr.max()) >= self.rain_threshold:
                    is_wet = True
                    break
            if is_wet or self.rng.rand() < self.dry_keep_prob:
                kept.append(sp)
        return kept

    def _load_radar_frame(self, case_id: str, prod: str, stamp: str) -> np.ndarray:
        key = (f"{case_id}:R:{prod}", stamp)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        radar_map, _, _ = self.case_maps[case_id]
        arr = clean_radar_field(np.load(radar_map[prod][stamp]), prod)
        self._cache.put(key, arr)
        return arr

    def _load_label_frame(self, case_id: str, stamp: str) -> np.ndarray:
        key = (f"{case_id}:L", stamp)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        _, label_map, _ = self.case_maps[case_id]
        arr = clean_label_rain(np.load(label_map[stamp]))
        self._cache.put(key, arr)
        return arr

    def _load_nwp_slice(self, case_id: str, stamp: str) -> np.ndarray:
        key = (f"{case_id}:N", stamp)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        _, _, nwp_map = self.case_maps[case_id]
        raw_stack = np.stack(
            [clean_nwp_field(np.load(nwp_map[var][stamp]), var) for var in NWP_VARS],
            axis=0,
        )  # (27, 167, 167)
        aligned = bilinear_upsample_center_crop(raw_stack)  # (27, 301, 301)
        self._cache.put(key, aligned)
        return aligned

    def __len__(self) -> int:
        return len(self.specs)

    def _pad_or_crop(
        self,
        radar: np.ndarray,
        nwp: np.ndarray,
        label: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Either crop a training patch (must be multiple of 32) or pad 301->pad_to (320)."""
        h, w = RADAR_SIZE, RADAR_SIZE
        valid_mask = np.ones((1, h, w), dtype=np.float32)

        if self.patch_size > 0 and self.split == "train":
            ps = self.patch_size
            # Bias crop toward rainy regions 75% of the time if rain exists
            rain_sum = label.sum(axis=(0, 1))
            if rain_sum.max() > self.rain_threshold and self.rng.rand() < 0.75:
                ys, xs = np.where(rain_sum >= self.rain_threshold)
                pick = self.rng.randint(0, len(ys))
                cy, cx = int(ys[pick]), int(xs[pick])
                y0 = int(np.clip(cy - ps // 2 + self.rng.randint(-ps // 4, ps // 4 + 1), 0, h - ps))
                x0 = int(np.clip(cx - ps // 2 + self.rng.randint(-ps // 4, ps // 4 + 1), 0, w - ps))
            else:
                y0 = int(self.rng.randint(0, h - ps + 1))
                x0 = int(self.rng.randint(0, w - ps + 1))

            return (
                radar[:, :, y0:y0 + ps, x0:x0 + ps],
                nwp[:, :, y0:y0 + ps, x0:x0 + ps],
                label[:, :, y0:y0 + ps, x0:x0 + ps],
                valid_mask[:, y0:y0 + ps, x0:x0 + ps],
                self.full_coord_yx[:, y0:y0 + ps, x0:x0 + ps],
            )

        # Full-field symmetric/bottom-right pad 301 -> pad_to (e.g. 320)
        if self.pad_to > h:
            pad_h = self.pad_to - h
            pad_w = self.pad_to - w
            radar = np.pad(radar, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
            nwp = np.pad(nwp, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
            label = np.pad(label, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
            valid_mask = np.pad(valid_mask, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
            coord = np.pad(self.full_coord_yx, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
            return radar, nwp, label, valid_mask, coord

        return radar, nwp, label, valid_mask, self.full_coord_yx

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sp = self.specs[index]
        cid = sp.case_id

        # Radar: (T_in, 10, 301, 301)
        radar_frames = []
        for t_idx in range(self.input_length):
            st = sp.radar_stamps[t_idx]
            if st is not None:
                chans = [self._load_radar_frame(cid, p, st) for p in RADAR_PRODS]
            else:
                prev_s, next_s = sp.radar_interp_pairs[t_idx]
                chans = [
                    0.5 * (self._load_radar_frame(cid, p, prev_s) + self._load_radar_frame(cid, p, next_s))
                    for p in RADAR_PRODS
                ]
            radar_frames.append(np.stack(chans, axis=0))
        radar = np.stack(radar_frames, axis=0)

        # Missing radar mask: (T_in,) 1.0 if interpolated, 0.0 if observed
        radar_missing_mask = np.zeros((self.input_length,), dtype=np.float32)
        for m_idx in sp.missing_radar_steps:
            radar_missing_mask[m_idx] = 1.0

        # NWP: (2, 27, 301, 301)
        nwp = np.stack(
            [self._load_nwp_slice(cid, sp.nwp_stamps[0]), self._load_nwp_slice(cid, sp.nwp_stamps[1])],
            axis=0,
        )

        # Label: (T_out, 1, 301, 301)
        label = np.stack(
            [self._load_label_frame(cid, st)[None, ...] for st in sp.label_stamps],
            axis=0,
        )

        radar, nwp, label, valid_mask, coord_yx = self._pad_or_crop(radar, nwp, label)

        return {
            "x_radar": torch.from_numpy(np.ascontiguousarray(radar)),
            "x_nwp": torch.from_numpy(np.ascontiguousarray(nwp)),
            "y_rain": torch.from_numpy(np.ascontiguousarray(label)),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid_mask)),
            "coord_yx": torch.from_numpy(np.ascontiguousarray(coord_yx)),
            "radar_missing_mask": torch.from_numpy(radar_missing_mask),
            "case_id": cid,
            "issue_stamp": sp.issue_stamp,
        }
