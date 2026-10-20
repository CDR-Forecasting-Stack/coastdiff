# Architecture

## Model

Two-stage CorrDiff (NVIDIA PhysicsNeMo architecture; see "No PhysicsNeMo
dependency" below):

1. **Regression UNet** — deterministic mean prediction, 32x32 (34ch) coarse
   input -> 256x256 (8ch) high-resolution output.
2. **EDM diffusion residual model** — conditioned on the regression mean,
   samples a stochastic residual restoring small-scale detail the
   regression network smooths away. Run per ensemble member.

Both stages are loaded from checkpoint directories configured in
`src/coastdiff/configs/default.yaml`; the highest training-iteration checkpoint in each
directory is used (see `models/checkpoint_loader.find_latest_checkpoint`).

## No PhysicsNeMo dependency

CoastDiff does not need NVIDIA PhysicsNeMo installed: importing it at runtime
would be a heavyweight, somewhat fragile prerequisite (pinned `warp-lang`, a
specific clone). The model code actually needed is a small slice of PhysicsNeMo:

- the UNet backbone (`SongUNetPosEmbd` and its layer building blocks:
  conv/group-norm/attention/embedding layers),
- the two wrapper classes the checkpoints were saved as
  (`CorrDiffRegressionUNet`, `EDMPrecondSuperResolution`),
- the EDM stochastic sampler, and
- the `regression_step`/`diffusion_step` orchestration functions.

This slice (~4,500 lines) is vendored directly into
`src/coastdiff/models/vendor/` (adapted from PhysicsNeMo, Apache-2.0 — see
`NOTICE`), with PhysicsNeMo's own model-registry/checkpoint-format
machinery (the `physicsnemo.core.Module` base class, `.mdlus` file I/O)
replaced by a small local shim (`vendor/_compat.py`) and loader
(`vendor/checkpoint_io.py`). `torch` is the only ML-framework dependency
left.

**Verification.** Architecture-preserving edits like this are easy to get
subtly wrong (a missing layer, a renamed parameter), so this wasn't just
reviewed by eye:

1. Every vendored model class (`CorrDiffRegressionUNet`,
   `EDMPrecondSuperResolution`) is instantiated from the real checkpoints'
   own saved init arguments, and its `state_dict()` keys and tensor shapes
   are compared against the checkpoint's raw, unmodified `state_dict` --
   confirmed to match exactly (676/676 parameters, both checkpoints, zero
   missing/extra/mismatched-shape keys).
2. A full end-to-end run (ERA5 fetch → regression → stochastic diffusion
   sampling → stitched output) was executed on the same region, timestamp,
   and seed through both the PhysicsNeMo-backed implementation and the
   vendored one. Every output channel — `input_era5`, `unet`, and
   `corrdiff` (all ensemble members, i.e. including the random diffusion
   sampling path, not just the deterministic regression) — matched
   **bit-for-bit**.

## Input: ERA5

This package is inference-only and ERA5-only. The model was trained on HRRR
fields coarsened to a ~24 km grid (README, "How the model was trained"); at
inference the same 34 coarse channels are built from ERA5. ERA5 is fetched
directly from the public ARCO Zarr store and regridded (bilinear) onto each
patch's own coarse grid — see `data/era5.py`. Derived channels (`q_sfc`,
`tp_in`, radiation unit conversion, `cos_sza`) are computed after regridding
the raw fields, because the dew-point-to-humidity relation is nonlinear.
ERA5 and the coarsened HRRR training inputs differ physically (clouds,
radiation, aerosols, precipitation), so ERA5-driven output is not identical
in character to what the model saw in training.

## Geographic coverage: one grid, any region

No region-specific data is bundled and there is no domain check. Every
request is handled by the same code path:

- **Output grid**: `grid.build_generic_grid` builds a regular lat/lon grid
  at ~3km spacing on the fly for the requested box (latitude spacing is
  fixed, since 1° latitude is ~111km everywhere; longitude spacing is
  adjusted for the region's central latitude so cells stay close to
  3km x 3km in real distance).
- **Invariant channels** (elevation, land-sea mask): sourced from ERA5's
  own global static fields, `geopotential_at_surface` (÷ 9.80665 ->
  elevation in m) and `land_sea_mask`, regridded (bilinear) onto that same
  grid (`Era5RawRegion.global_invariants`, `data/era5.py`).

This works anywhere on Earth because the convolutional architecture has no
geographic restriction baked in: its positional embeddings
(`SongUNetPosEmbd`, `gridtype="sinusoidal"`) are computed purely from
`linspace(0, 2π, patch_size)`, i.e. position *within* a 256x256 patch, with
no dependency on absolute lat/lon.

The model was trained for regional coastal areas (ocean-dominated patches in
the HRRR domain). It can be run anywhere ERA5 has data, but skill outside the
trained coastal regimes has not been evaluated. The training grid was a
Lambert conformal projection with HRRR-native ~3km terrain and land-sea
fields; the output grid here is a regular lat/lon grid and the terrain and
land-sea fields are ERA5's ~25km ones. Each output file's `grid_source`
attribute records this (`generic_global`).

## Tiling and stitching

A requested region is tiled into overlapping 256x256-pixel patches on the
native grid (`--overlap`, default 3px). Each patch is inferred
independently and stitched back with trapezoidal-weight feathering across
the overlap band, so seams blend smoothly. NaN appears wherever the
requested region falls outside the requested box, or ERA5 data was
unavailable for that pixel.

## Continuous time ranges and parallelism

ERA5 is fetched once per run for the whole requested time span (not once
per timestamp). Each run is single-device, as CorrDiff is; there is no
in-process multi-GPU. Timestamps and sub-regions are independent, so
throughput is scaled by launching several runs at once, one GPU each, split
by time or space — see the README section "GPU usage and parallelizing".

## Rain filter

`tp` (precipitation) is the most spatially intermittent output variable,
and diffusion's stochastic texture is most valuable exactly where real rain
is occurring — and mostly just noise elsewhere. `rain_filter.py` blends the
deterministic `unet` prediction with the diffusion `corrdiff` prediction,
keyed on `unet`'s own predicted rain rate exceeding a threshold (default
0.02mm; `--rain-filter-mm` changes it, `--no-rain-filter` disables it).

## Physical constraints

`physical_constraints.py` runs after the rain filter and the `tp` conversion
to mm, on the `unet` and `corrdiff` fields only (`input_era5` is written as
fetched): `tp`, `ssrd` and `strd` are clipped to be non-negative, and `ssrd`
is set to zero wherever cos(solar zenith angle) <= 0. The cosine is computed
per output pixel from latitude, longitude and the timestamp with the same
simple declination/hour-angle formula as the model's `cos_sza` input channel
(no equation of time or refraction, so pixels within a few minutes of sunset
can be zeroed slightly early). Longwave is not zeroed at night; NaN is kept.
