"""Thresholded UNet+CorrDiff blend for precipitation (tp).

CorrDiff's diffusion residual restores realistic small-scale rain texture,
but it perturbs the field everywhere, including light/no-rain areas where
that texture is just noise. Chosen via a threshold sweep (0.005/0.01/0.02/
0.03mm tested against the 40M-sample checkpoint): 0.02mm gave the best
balance -- it improves both correlation and rmse over plain UNet while only
replacing ~0.4% of pixels, vs. e.g. 0.01mm's broader (and noisier) swap-in.

    blended = where(unet_tp_mm > threshold_mm, corrdiff, unet)

applied per ensemble member, using UNet's OWN predicted rain to decide
where diffusion detail is trusted.
"""
import numpy as np

from .data.transforms import precip_inv

DEFAULT_RAIN_FILTER_MM = 0.02


def apply_rain_filter(corrdiff_tp_log: np.ndarray, unet_tp_log: np.ndarray,
                       threshold_mm: float = DEFAULT_RAIN_FILTER_MM) -> np.ndarray:
    """corrdiff_tp_log: (H,W) or (E,H,W), unet_tp_log: (H,W) -- both still in
    the internal log10(1+mm) representation (call this BEFORE converting to
    mm for output). Returns an array shaped like corrdiff_tp_log."""
    mask = precip_inv(unet_tp_log) > threshold_mm  # (H,W) bool
    return np.where(mask, corrdiff_tp_log, unet_tp_log)
