"""Output grid + tiling/stitching geometry.

The output grid is a regular lat/lon grid built on the fly for whatever
region is requested (build_generic_grid) -- ~3km spacing, anywhere ERA5 has
data. Terrain and land-sea conditioning also come from ERA5, so nothing
region-specific is bundled (see docs/ARCHITECTURE.md "Geographic coverage").
"""
from __future__ import annotations

import warnings

import numpy as np

HR_SIZE = 256   # native patch size the model was trained on
LR_SIZE = 32    # coarse input patch size
UPSAMPLE_FACTOR = HR_SIZE // LR_SIZE  # 8

KM_PER_DEG_LAT = 111.32  # ~constant everywhere; longitude spacing is latitude-adjusted below


def build_generic_grid(lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                        target_km: float = 3.0, min_size: int = HR_SIZE) -> dict:
    """The output grid for a requested region: a regular lat/lon grid,
    anywhere ERA5 has data (i.e. anywhere on Earth) -- see
    docs/ARCHITECTURE.md "Geographic coverage".

    Grid spacing approximates the model's native ~3km training resolution:
    latitude spacing is fixed (a degree of latitude is ~111km everywhere),
    longitude spacing is adjusted for the region's central latitude (a
    degree of longitude shrinks by cos(latitude)) so each cell is close to
    target_km x target_km in real distance. This is still only an
    approximation: a plain regular lat/lon grid, not the Lambert conformal
    projection of the model's training data.

    The returned grid has NO `elev_mean`/`lsm_mean` keys -- the caller
    (inference.py) fills those in from ERA5's own global orography/land-sea
    mask fields (there is no bundled high-resolution terrain asset).

    Padded to at least `min_size` pixels in each dimension (needed for one
    model patch) and roughly centered on the requested box.
    """
    center_lat = (lat_min + lat_max) / 2.0
    lat_step = target_km / KM_PER_DEG_LAT
    lon_step = target_km / (KM_PER_DEG_LAT * max(np.cos(np.radians(center_lat)), 0.05))

    n_lat = max(min_size, int(np.ceil((lat_max - lat_min) / lat_step)) + 1)
    n_lon = max(min_size, int(np.ceil((lon_max - lon_min) / lon_step)) + 1)

    lat_pad = (n_lat - 1) * lat_step - (lat_max - lat_min)
    lon_pad = (n_lon - 1) * lon_step - (lon_max - lon_min)
    lat0 = lat_min - lat_pad / 2
    lon0 = lon_min - lon_pad / 2

    lats = (lat0 + np.arange(n_lat) * lat_step).astype(np.float32)
    lons = (lon0 + np.arange(n_lon) * lon_step).astype(np.float32)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return {"latitude": lat2d, "longitude": lon2d}


def bbox_to_rowcol(grid: dict, lat_min, lat_max, lon_min, lon_max):
    """Returns the (row,col) bounding rectangle of `grid` that contains the
    requested lat/lon box, plus the exact per-pixel boolean mask (used to
    NaN out any pixels of that rectangle outside the true box). `grid` is
    built with margin around the requested box (build_generic_grid), so
    this always succeeds for a grid built for that same box. Written without
    assuming a regular grid, so it also works for curvilinear grids where a
    lat/lon box is not a rectangle in (row,col) space."""
    lat, lon = grid["latitude"], grid["longitude"]
    mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    if not mask.any():
        raise ValueError(
            "Requested lat/lon bbox does not intersect the grid "
            f"(grid covers roughly lat [{lat.min():.2f},{lat.max():.2f}], "
            f"lon [{lon.min():.2f},{lon.max():.2f}])"
        )
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1, mask


def tile_starts(a0: int, a1: int, size: int, stride: int) -> list[int]:
    """Start indices of `size`-length tiles covering [a0,a1), spaced by
    `stride`, with a final tile snapped to end exactly at a1."""
    if a1 - a0 <= size:
        return [a0]
    starts = list(range(a0, a1 - size, stride))
    if starts[-1] != a1 - size:
        starts.append(a1 - size)
    return starts


def clamp_start(s: int, size: int, n: int) -> int:
    return max(0, min(s, n - size))


def edge_weight_1d(size: int, overlap: int) -> np.ndarray:
    """Trapezoidal feather window: ramps up over the first overlap+1 pixels,
    flat in the middle, ramps down at the end. overlap=0 -> uniform weight 1."""
    o = np.arange(size)
    w = np.minimum(np.minimum(o + 1, overlap + 1), size - o)
    return w.astype(np.float32)


def patch_weight_2d(size: int, overlap: int) -> np.ndarray:
    w1 = edge_weight_1d(size, overlap)
    return w1[:, None] * w1[None, :]


def block_nanmean(x: np.ndarray, factor: int) -> np.ndarray:
    """(C,H,W) -> (C, ceil(H/factor), ceil(W/factor)) NaN-aware block
    average, padded from the (0,0) origin."""
    C, H, W = x.shape
    Hp, Wp = int(np.ceil(H / factor)) * factor, int(np.ceil(W / factor)) * factor
    xp = np.full((C, Hp, Wp), np.nan, dtype=np.float32)
    xp[:, :H, :W] = x
    xp = xp.reshape(C, Hp // factor, factor, Wp // factor, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(xp, axis=(2, 4)).astype(np.float32)
