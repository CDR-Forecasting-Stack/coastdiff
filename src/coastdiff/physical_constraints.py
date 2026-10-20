"""Physical-consistency constraints applied to the high-resolution output
(`unet` and `corrdiff`) in physical units, just before writing:

- `ssrd` (shortwave) and `strd` (longwave) and `tp` (rain): no negative values.
- `ssrd` is zero wherever the sun is at or below the horizon (solar zenith
  angle >= 90 deg, i.e. cos(SZA) <= 0) -- no daylight, no shortwave.

NaN (outside the requested region / missing data) is left as NaN.
"""
from __future__ import annotations

import numpy as np

from .data.channels import OUT_CH
from .data.transforms import compute_cos_sza

SSRD_IDX = OUT_CH.index("ssrd")
STRD_IDX = OUT_CH.index("strd")
TP_IDX = OUT_CH.index("tp")


def apply_physical_constraints(field: np.ndarray, cos_sza: np.ndarray) -> np.ndarray:
    """field: (..., C, H, W) in physical units (tp in mm), channels in OUT_CH
    order. cos_sza: (H, W), cosine of the solar zenith angle per pixel.
    Returns a new array; the input is not modified."""
    out = field.copy()
    for idx in (SSRD_IDX, STRD_IDX, TP_IDX):
        ch = out[..., idx, :, :]
        out[..., idx, :, :] = np.where(ch < 0.0, 0.0, ch)
    night = cos_sza <= 0.0
    ssrd = out[..., SSRD_IDX, :, :]
    out[..., SSRD_IDX, :, :] = np.where(night & ~np.isnan(ssrd), 0.0, ssrd)
    return out


def solar_cos_zenith(lat: np.ndarray, lon: np.ndarray, dt) -> np.ndarray:
    """cos(SZA) on the output grid (>= 0; exactly 0 at/after sunset). Same
    formula the model's own `cos_sza` input channel uses."""
    return compute_cos_sza(lat, lon, dt)
