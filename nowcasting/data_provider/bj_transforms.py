"""Data decoding, coordinate resampling, and normalization for the BJ radar+NWP dataset.

Implements the data contracts specified in MASTER.md:
- 10 radar products (301x301 @ 0.01 deg), sentinel cleaning, HBR decoding, log1p(dBZ)/6 transform
- 27 NWP variables (167x167 @ 0.03 deg -> 3x bilinear upsample -> 301x301 center crop), norm-v2
- RA precipitation label (301x301), sentinel -9 -> 0, scale * 0.1 to mm/6min
"""

import numpy as np
import torch
import torch.nn.functional as F

RADAR_PRODS = [
    "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70",
    "CR", "ET", "HBR", "VIL",
]
DBZ_PRODS = {"CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70", "CR", "HBR"}

NWP_VARS = [
    "CAPE", "DVG200", "DVG850", "DVG925",
    "HT0", "HT10", "HTw0", "KI",
    "LCL", "LFC", "LI300", "LI500",
    "PE", "PWAT", "Q1000", "Q700", "Q850",
    "RH1000", "RH500", "RH700",
    "TTdMax74", "TTdMean74", "TdSfc850",
    "WS500", "WS700", "WS925", "muLCL",
]

# Indices within NWP_VARS for dynamic (motion) vs thermodynamic/moisture subsets
NWP_MOTION_VARS = ["WS500", "WS700", "WS925", "DVG200", "DVG850", "DVG925"]
NWP_MOTION_INDICES = [NWP_VARS.index(v) for v in NWP_MOTION_VARS]

# norm-v2 statistics computed strictly on the 23 clean train cases (valid pixels > -9000)
NWP_NORM = {
    "CAPE": (345.061404, 660.440439),
    "DVG200": (3e-06, 0.000122),
    "DVG850": (0.0, 0.000157),
    "DVG925": (-5e-06, 0.000181),
    "HT0": (4252.912825, 467.209113),
    "HT10": (5958.083962, 502.594638),
    "HTw0": (3820.888331, 451.716014),
    "KI": (31.768477, 6.018868),
    "LCL": (1584.974108, 957.664890),
    "LFC": (3494.130648, 1391.512007),
    "LI300": (2.182488, 4.705154),
    "LI500": (0.035586, 3.084679),
    "PE": (28.114827, 12.407349),
    "PWAT": (41.598928, 10.441490),
    "Q1000": (12.212021, 3.267742),
    "Q700": (6.307302, 1.810299),
    "Q850": (10.417154, 2.877373),
    "RH1000": (59.498297, 19.855322),
    "RH500": (58.841428, 25.092190),
    "RH700": (66.600874, 19.120818),
    "TTdMax74": (12.469318, 7.937609),
    "TTdMean74": (7.410776, 4.963097),
    "TdSfc850": (13.205939, 4.531317),
    "WS500": (9.216064, 5.450697),
    "WS700": (5.987866, 3.532827),
    "WS925": (3.732024, 2.681390),
    "muLCL": (2008.508435, 887.366381),
}

NWP_MEANS = np.array([NWP_NORM[v][0] for v in NWP_VARS], dtype=np.float32)
NWP_STDS = np.array([NWP_NORM[v][1] for v in NWP_VARS], dtype=np.float32)

NWP_TO_RADAR_SCALE = 3
RADAR_SIZE = 301
NWP_SIZE = 167


def clean_radar_field(arr: np.ndarray, prod: str) -> np.ndarray:
    """Decode sentinel/encoded values and apply log1p normalization to a single radar product."""
    x = arr.astype(np.float32, copy=True)
    # Standard missing/no-echo sentinels
    invalid = (x <= -32767.0) | (x == -1280.0)

    if prod == "HBR":
        # Decode low-bit offset bands observed in BJ HBR archives
        low_band = x < -20000.0
        mid_band = (x > -6000.0) & (x < -4000.0)
        x[low_band] = x[low_band] + 28655.0
        x[mid_band] = x[mid_band] + 5000.0

    x[invalid] = 0.0

    if prod in DBZ_PRODS:
        # Stored in 0.1 dBZ units; clip negative clutter/noise to 0 dBZ
        dbz = np.clip(x * 0.1, 0.0, 85.0)
        return np.log1p(dbz) / 6.0
    if prod == "ET":
        # Echo top in 0.1 km units (~0..20 km)
        et_km = np.clip(x * 0.1, 0.0, 25.0)
        return np.log1p(et_km) / 4.0
    if prod == "VIL":
        # Vertically integrated liquid (0.01 kg/m^2 scale -> ~0..80)
        vil = np.clip(x * 0.01, 0.0, 100.0)
        return np.log1p(vil) / 5.0
    raise ValueError(f"Unknown radar product: {prod}")


def clean_label_rain(arr: np.ndarray) -> np.ndarray:
    """Decode RA precipitation label (0.1 mm/6min raw -> mm/6min float32)."""
    x = arr.astype(np.float32, copy=False)
    x = np.where(x < 0.0, 0.0, x * 0.1)
    return x


def clean_nwp_field(arr: np.ndarray, var: str) -> np.ndarray:
    """Mask NWP -9999/-10000 fill values and apply norm-v2 z-score normalization."""
    x = arr.astype(np.float32, copy=False)
    valid = x > -9000.0
    mean, std = NWP_NORM[var]
    normed = np.where(valid, (x - mean) / max(std, 1e-8), 0.0)
    return np.clip(normed, -10.0, 10.0).astype(np.float32)


def bilinear_upsample_center_crop(nwp_hw: np.ndarray, scale: int = NWP_TO_RADAR_SCALE, target_size: int = RADAR_SIZE) -> np.ndarray:
    """Resample NWP (..., 167, 167) @ 0.03 deg -> x3 bilinear (501x501) -> center crop 301x301 @ 0.01 deg.

    Uses half-pixel alignment src = (dst + 0.5) / scale - 0.5 with reflection boundary indexing.
    Supports shapes (H, W) or (..., H, W).
    """
    arr = np.asarray(nwp_hw, dtype=np.float32)
    orig_ndim = arr.ndim
    if orig_ndim == 2:
        arr = arr[None, ...]
    lead_shape = arr.shape[:-2]
    src_h, src_w = arr.shape[-2:]
    flat = arr.reshape(-1, src_h, src_w)

    up_h = src_h * scale
    up_w = src_w * scale
    start_y = (up_h - target_size) // 2
    start_x = (up_w - target_size) // 2

    dst_y = np.arange(start_y, start_y + target_size, dtype=np.float32)
    dst_x = np.arange(start_x, start_x + target_size, dtype=np.float32)
    src_y = (dst_y + 0.5) / scale - 0.5
    src_x = (dst_x + 0.5) / scale - 0.5

    y0 = np.floor(src_y).astype(np.int64)
    x0 = np.floor(src_x).astype(np.int64)
    wy = (src_y - y0)[:, None]
    wx = (src_x - x0)[None, :]

    def reflect_idx(idx, limit):
        idx = np.where(idx < 0, -idx, idx)
        idx = np.where(idx >= limit, 2 * limit - 2 - idx, idx)
        return np.clip(idx, 0, limit - 1)

    y0r = reflect_idx(y0, src_h)
    y1r = reflect_idx(y0 + 1, src_h)
    x0r = reflect_idx(x0, src_w)
    x1r = reflect_idx(x0 + 1, src_w)

    v00 = flat[:, y0r[:, None], x0r[None, :]]
    v01 = flat[:, y0r[:, None], x1r[None, :]]
    v10 = flat[:, y1r[:, None], x0r[None, :]]
    v11 = flat[:, y1r[:, None], x1r[None, :]]

    one = np.float32(1.0)
    out = (
        (one - wy)[None, :, :] * (one - wx)[None, :, :] * v00
        + (one - wy)[None, :, :] * wx[None, :, :] * v01
        + wy[None, :, :] * (one - wx)[None, :, :] * v10
        + wy[None, :, :] * wx[None, :, :] * v11
    ).astype(np.float32, copy=False)

    if orig_ndim == 2:
        return out[0]
    return out.reshape(*lead_shape, target_size, target_size)


def resample_nwp_to_radar_torch(nwp_tensor: torch.Tensor, scale: int = NWP_TO_RADAR_SCALE, target_size: int = RADAR_SIZE) -> torch.Tensor:
    """PyTorch equivalent of bilinear_upsample_center_crop for (..., H, W) tensors."""
    orig_ndim = nwp_tensor.ndim
    if orig_ndim == 2:
        t = nwp_tensor.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        t = nwp_tensor.unsqueeze(0)
    else:
        lead = nwp_tensor.shape[:-2]
        t = nwp_tensor.reshape(1, -1, nwp_tensor.shape[-2], nwp_tensor.shape[-1])

    _, c, src_h, src_w = t.shape
    up_h, up_w = src_h * scale, src_w * scale
    # align_corners=False in PyTorch uses exact half-pixel mapping (dst + 0.5)/scale - 0.5
    padded = F.pad(t, (1, 1, 1, 1), mode="reflect")
    # Map center crop coordinates directly via grid_sample for exact reflection boundary match
    start_y = (up_h - target_size) // 2
    start_x = (up_w - target_size) // 2
    dst_y = torch.arange(start_y, start_y + target_size, device=t.device, dtype=t.dtype)
    dst_x = torch.arange(start_x, start_x + target_size, device=t.device, dtype=t.dtype)
    src_y = (dst_y + 0.5) / scale - 0.5 + 1.0  # +1 for 1-px reflect pad
    src_x = (dst_x + 0.5) / scale - 0.5 + 1.0

    # Normalize to [-1, 1] with align_corners=True on the (src_h+2, src_w+2) padded grid
    gy = 2.0 * src_y / (src_h + 1) - 1.0
    gx = 2.0 * src_x / (src_w + 1) - 1.0
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    out = F.grid_sample(padded, grid, mode="bilinear", padding_mode="border", align_corners=True)

    if orig_ndim == 2:
        return out[0, 0]
    if orig_ndim == 3:
        return out[0]
    return out.reshape(*nwp_tensor.shape[:-2], target_size, target_size)
