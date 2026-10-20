"""Pure physical transforms shared by input construction and output writing."""
from datetime import datetime

import numpy as np


def precip_log(x: np.ndarray) -> np.ndarray:
    """mm -> log10(1+mm). Clips negatives. Internal model representation."""
    return np.log10(1.0 + np.maximum(0.0, x)).astype(np.float32)


def precip_inv(x: np.ndarray) -> np.ndarray:
    """log10(1+mm) -> mm. Inverse of precip_log; used at the output boundary
    so every written field is in plain physical units."""
    return np.maximum(0.0, 10.0 ** x - 1.0).astype(np.float32)


def derive_q_surface(d2m: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """2m specific humidity from dewpoint + surface pressure (Magnus formula).
    d2m in K, sp in Pa. Returns kg/kg."""
    e_s = 611.2 * np.exp(17.67 * (d2m - 273.15) / (d2m - 29.65))
    return np.maximum(0.0, 0.622 * e_s / (sp - 0.378 * e_s)).astype(np.float32)


def compute_cos_sza(lat_2d: np.ndarray, lon_2d: np.ndarray, dt: datetime) -> np.ndarray:
    """Cosine solar zenith angle, in [0,1]."""
    doy = dt.timetuple().tm_yday
    hr = dt.hour + dt.minute / 60.0
    decl = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (doy - 81))))
    ha = np.radians(15.0 * (hr + lon_2d / 15.0 - 12.0))
    latr = np.radians(lat_2d)
    return np.maximum(
        0.0, np.sin(latr) * np.sin(decl) + np.cos(latr) * np.cos(decl) * np.cos(ha)
    ).astype(np.float32)


def zoom_extrapolate(x: np.ndarray, factor: int) -> np.ndarray:
    """Half-pixel-center-aligned bilinear zoom with edge extrapolation, the
    same upsampling the model's own conditioning uses to go from the coarse
    (32x32) grid to the native (256x256) grid. Pure numpy (no numba) --
    verified bit-for-bit identical against the numba original on random
    input. x: (C, Hs, Ws) -> (C, Hs*factor, Ws*factor)."""
    C, Hs, Ws = x.shape
    Ht, Wt = Hs * factor, Ws * factor
    iy = np.arange(Ht)
    ix = (iy + 0.5) / factor - 0.5
    ix0 = np.clip(np.floor(ix).astype(int), 0, Hs - 2)
    ix1 = ix0 + 1
    fy = ix - ix0
    jy = np.arange(Wt)
    jx = (jy + 0.5) / factor - 0.5
    jx0 = np.clip(np.floor(jx).astype(int), 0, Ws - 2)
    jx1 = jx0 + 1
    fx = jx - jx0
    x00 = x[:, ix0][:, :, jx0]
    x01 = x[:, ix0][:, :, jx1]
    x10 = x[:, ix1][:, :, jx0]
    x11 = x[:, ix1][:, :, jx1]
    x0 = x00 + fx[None, None, :] * (x01 - x00)
    x1 = x10 + fx[None, None, :] * (x11 - x10)
    return (x0 + fy[None, :, None] * (x1 - x0)).astype(np.float32)


def downsample_256_to_32(stack: np.ndarray) -> np.ndarray:
    """(C,256,256) -> (C,32,32) via 8x8 area average. Used only to derive a
    patch's own local low-res lat/lon grid from the native reference grid
    (inputs to this package always come from ERA5)."""
    C = stack.shape[0]
    return stack.astype(np.float32).reshape(C, 32, 8, 32, 8).mean(axis=(2, 4))
