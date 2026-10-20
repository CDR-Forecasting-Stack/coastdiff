"""Core ERA5-driven inference: region + one-or-more timestamps -> high-res
NetCDF, one file per timestamp. Single-device by design (as CorrDiff itself
is): uses one GPU if available, else CPU. Scale out by running several
independent jobs over different time ranges or sub-regions -- see the
README section "Parallelizing".
"""
from __future__ import annotations

import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .data import era5
from .data.channels import IN_CH, INVARIANT_CH, OUT_CH
from .data.grid import (
    HR_SIZE, UPSAMPLE_FACTOR, bbox_to_rowcol, block_nanmean, build_generic_grid,
    clamp_start, patch_weight_2d, tile_starts,
)
from .data.normalization import Normalizer
from .data.transforms import downsample_256_to_32, precip_inv, zoom_extrapolate
from .models.checkpoint_loader import (
    init_single_process_distributed, load_models, run_patch,
)
from .output import write_output
from .physical_constraints import apply_physical_constraints, solar_cos_zenith
from .rain_filter import DEFAULT_RAIN_FILTER_MM, apply_rain_filter

log = logging.getLogger("coastdiff.inference")

TP_IDX = OUT_CH.index("tp")


def _build_patch_input(r0, c0, grid, era5_slice, inv2, normalizer):
    lat_patch = grid["latitude"][r0:r0 + HR_SIZE, c0:c0 + HR_SIZE]
    lon_patch = grid["longitude"][r0:r0 + HR_SIZE, c0:c0 + HR_SIZE]
    latlon_lr = downsample_256_to_32(np.stack([lat_patch, lon_patch], axis=0))
    lr = era5.build_era5_input_stack(era5_slice, latlon_lr[0], latlon_lr[1])  # (34,32,32)
    if not np.isfinite(lr).all():
        return None
    x = zoom_extrapolate(lr, UPSAMPLE_FACTOR)  # (34,256,256)
    inv_patch = inv2[:, r0:r0 + HR_SIZE, c0:c0 + HR_SIZE]
    x = np.concatenate([x, inv_patch], axis=0)  # (36,256,256)
    return normalizer.normalize_input(x).astype(np.float32)


def run_inference(
    *, lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    timestamps: list[datetime], output_dir: str | Path, config: dict | None = None,
    config_path: str | Path | None = None, device: torch.device | str | None = None,
    num_ensembles: int = 4, overlap: int = 3, mode: str = "all", seed: int = 0,
    seed_batch_size: int = 1, sampler: str = "stochastic", sampler_steps: int | None = None,
    solver: str = "euler", rain_filter_mm: float | None = DEFAULT_RAIN_FILTER_MM,
    era5_pad: int = 3, skip_missing: bool = False,
) -> list[Path]:
    """Runs inference for every timestamp in `timestamps` over one region,
    writing one NetCDF per timestamp into `output_dir`. Returns the list of
    written file paths. `rain_filter_mm=None` disables the rain filter.
    Hours without ERA5 data raise an error unless `skip_missing=True`, which
    runs only the hours that exist."""
    if not (0 <= overlap < HR_SIZE):
        raise ValueError("overlap must be in [0, 256)")
    if not timestamps:
        raise ValueError("timestamps must be non-empty")

    np.random.seed(seed)
    torch.manual_seed(seed)
    init_single_process_distributed()

    cfg = config if config is not None else load_config(config_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    log.info(f"Device: {device}, {len(timestamps)} timestamp(s)")

    # Output grid + tiling geometry: a regular lat/lon grid built for this
    # exact request, anywhere ERA5 has data -- see
    # docs/ARCHITECTURE.md "Geographic coverage". Terrain/land-sea
    # invariants come from ERA5's own global static fields (fetched below),
    # not from any bundled region-specific asset.
    grid = build_generic_grid(lat_min, lat_max, lon_min, lon_max)
    row0, row1, col0, col1, bbox_mask = bbox_to_rowcol(grid, lat_min, lat_max, lon_min, lon_max)
    canvas_h, canvas_w = row1 - row0, col1 - col0
    log.info(f"Canvas: rows[{row0}:{row1}) cols[{col0}:{col1}) -> {canvas_h}x{canvas_w}px")

    stride = HR_SIZE - overlap
    row_starts = sorted({clamp_start(s, HR_SIZE, grid["latitude"].shape[0])
                          for s in tile_starts(row0, row1, HR_SIZE, stride)})
    col_starts = sorted({clamp_start(s, HR_SIZE, grid["latitude"].shape[1])
                          for s in tile_starts(col0, col1, HR_SIZE, stride)})
    n_patches = len(row_starts) * len(col_starts)
    log.info(f"Tiling into {len(row_starts)}x{len(col_starts)}={n_patches} patches, "
             f"overlap={overlap}px, stride={stride}px")

    # ERA5 fetched ONCE for every requested timestamp, over the exact pixel
    # rectangle every patch can touch -- see data/era5.py for the batching.
    all_r0, all_r1 = min(row_starts), max(row_starts) + HR_SIZE
    all_c0, all_c1 = min(col_starts), max(col_starts) + HR_SIZE
    lat_box = grid["latitude"][all_r0:all_r1, all_c0:all_c1]
    lon_box = grid["longitude"][all_r0:all_r1, all_c0:all_c1]
    log.info(f"Fetching ERA5 for {len(timestamps)} timestamp(s) over lat "
             f"[{lat_box.min():.2f},{lat_box.max():.2f}] lon [{lon_box.min():.2f},{lon_box.max():.2f}] ...")
    store = era5.open_arco_store()
    available = era5.check_availability(store, timestamps, skip_missing=skip_missing)
    if len(available) < len(timestamps):
        log.warning(f"Skipping {len(timestamps) - len(available)} hour(s) without ERA5 data; "
                    f"running {len(available)}.")
        timestamps = available
    region = era5.Era5RawRegion(timestamps, store, float(lat_box.min()), float(lat_box.max()),
                                 float(lon_box.min()), float(lon_box.max()), pad=era5_pad)

    # No bundled terrain asset: regrid ERA5's own global elevation/land-sea
    # mask onto the output grid, once, as the model's invariant conditioning
    # channels (elev_mean, lsm_mean) -- see docs/ARCHITECTURE.md.
    elev_raw, lsm_raw, e_lat, e_lon = region.global_invariants()
    grid["elev_mean"] = era5.regrid_linear(e_lat, e_lon, elev_raw, grid["latitude"], grid["longitude"])
    grid["lsm_mean"] = era5.regrid_linear(e_lat, e_lon, lsm_raw, grid["latitude"], grid["longitude"])

    models = load_models(cfg["regression_checkpoint_dir"], cfg["diffusion_checkpoint_dir"],
                          device, mode=mode, sampler=sampler, sampler_steps=sampler_steps,
                          solver=solver)
    normalizer = Normalizer(cfg["stats_path"])
    inv2 = np.stack([grid[k] for k in INVARIANT_CH], axis=0)
    weight_patch = patch_weight_2d(HR_SIZE, overlap)

    lat_c = grid["latitude"][row0:row1, col0:col1]
    lon_c = grid["longitude"][row0:row1, col0:col1]
    lat_lr = block_nanmean(lat_c[None], UPSAMPLE_FACTOR)[0]
    lon_lr = block_nanmean(lon_c[None], UPSAMPLE_FACTOR)[0]
    outside = ~bbox_mask[row0:row1, col0:col1]
    outside_lr = ~((lat_lr >= lat_min) & (lat_lr <= lat_max)
                   & (lon_lr >= lon_min) & (lon_lr <= lon_max))

    output_dir = Path(output_dir)
    out_paths: list[Path] = []

    for ts_idx, dt in enumerate(timestamps):
        t0 = time.time()
        era5_slice = region.slice_for(dt)

        accum_w = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        accum_unet = np.zeros((len(OUT_CH), canvas_h, canvas_w), dtype=np.float32)
        accum_ens = (np.zeros((num_ensembles, len(OUT_CH), canvas_h, canvas_w), dtype=np.float32)
                     if mode == "all" else None)

        n_valid = n_skipped = 0
        for r0 in row_starts:
            for c0 in col_starts:
                x_norm = _build_patch_input(r0, c0, grid, era5_slice, inv2, normalizer)
                wr0, wr1 = max(r0, row0), min(r0 + HR_SIZE, row1)
                wc0, wc1 = max(c0, col0), min(c0 + HR_SIZE, col1)
                pr0, pc0 = wr0 - r0, wc0 - c0
                pr1, pc1 = pr0 + (wr1 - wr0), pc0 + (wc1 - wc0)
                cr0, cc0 = wr0 - row0, wc0 - col0
                cr1, cc1 = cr0 + (wr1 - wr0), cc0 + (wc1 - wc0)
                if x_norm is None:
                    n_skipped += 1
                    continue
                n_valid += 1
                unet_phys, ens_phys = run_patch(models, x_norm, num_ensembles, seed_batch_size, normalizer)
                w = weight_patch[pr0:pr1, pc0:pc1]
                accum_w[cr0:cr1, cc0:cc1] += w
                accum_unet[:, cr0:cr1, cc0:cc1] += unet_phys[:, pr0:pr1, pc0:pc1] * w
                if ens_phys is not None:
                    accum_ens[:, :, cr0:cr1, cc0:cc1] += ens_phys[:, :, pr0:pr1, pc0:pc1] * w[None, None]

        def finalize(accum):
            if accum is None:
                return None
            w_ = accum_w[None] if accum.ndim == 3 else accum_w[None, None]
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.where(w_ > 0, accum / np.maximum(w_, 1e-12), np.nan).astype(np.float32)
            return out

        unet_stitched = finalize(accum_unet)
        corrdiff_stitched = finalize(accum_ens)
        unet_stitched[:, outside] = np.nan
        if corrdiff_stitched is not None:
            corrdiff_stitched[:, :, outside] = np.nan

        # Low-res ERA5 input diagnostic, sampled directly on (lat_lr,lon_lr) --
        # no stitching needed, ERA5 is a smooth continuous field.
        input_era5 = era5.build_era5_input_stack(era5_slice, lat_lr, lon_lr)
        input_era5[:, outside_lr] = np.nan

        use_filter = rain_filter_mm is not None and rain_filter_mm >= 0
        if use_filter:
            unet_for_mask = unet_stitched[TP_IDX]
            if corrdiff_stitched is not None:
                corrdiff_stitched[:, TP_IDX] = apply_rain_filter(
                    corrdiff_stitched[:, TP_IDX], unet_for_mask, rain_filter_mm
                )

        # Convert tp from the internal log10(1+mm) representation to plain
        # mm -- the output boundary, so every written field shares one
        # natural physical unit with its input counterpart (see README).
        in_ch_tp = IN_CH.index("tp_in")
        input_era5[in_ch_tp] = precip_inv(input_era5[in_ch_tp])
        unet_stitched[TP_IDX] = precip_inv(unet_stitched[TP_IDX])
        if corrdiff_stitched is not None:
            corrdiff_stitched[:, TP_IDX] = precip_inv(corrdiff_stitched[:, TP_IDX])

        # Physical consistency (see physical_constraints.py): no negative
        # ssrd/strd/tp, and ssrd = 0 wherever the sun is below the horizon.
        cos_sza_hr = solar_cos_zenith(lat_c, lon_c, dt)
        unet_stitched = apply_physical_constraints(unet_stitched, cos_sza_hr)
        if corrdiff_stitched is not None:
            corrdiff_stitched = apply_physical_constraints(corrdiff_stitched, cos_sza_hr)

        out_path = output_dir / f"coastdiff_{dt.strftime('%Y%m%dT%H%M')}.nc"
        write_output(
            out_path, dt=dt, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
            overlap=overlap, num_ensembles=num_ensembles,
            rain_filter_mm=rain_filter_mm if use_filter else -1.0,
            lat=lat_c, lon=lon_c, lat_lr=lat_lr, lon_lr=lon_lr, input_era5=input_era5,
            unet=unet_stitched, corrdiff=corrdiff_stitched,
            regression_checkpoint=models.reg_ckpt, diffusion_checkpoint=models.res_ckpt,
        )
        out_paths.append(out_path)
        log.info(f"[{ts_idx+1}/{len(timestamps)}] {dt.isoformat()}: {n_valid}/{n_patches} patches "
                 f"valid ({n_skipped} skipped) -> {out_path} ({time.time()-t0:.0f}s)")

    return out_paths
