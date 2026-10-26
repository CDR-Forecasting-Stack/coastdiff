# CoastDiff

Diffusion-based coastal downscaling driven by [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview). Give it a latitude/longitude box and a time
(a single hour, or a continuous range such as every hour for three days), and it produces
~3 km fields of eight surface variables from ERA5 (~25 km) with a [two-stage CorrDiff model](https://arxiv.org/abs/2309.15214): a
deterministic regression network followed by a diffusion model that restores small-scale detail.

- **Input:** ERA5 only, read live from the public ARCO ERA5 archive. Nothing to download beforehand.
- **Output:** one NetCDF file per hour, with the deterministic prediction and a diffusion ensemble.
- **Inference only:** no training or test data is included.
- **Lightweight install:** the normalization statistics are bundled; the two trained checkpoints
  (~640 MB) are downloaded once from Zenodo on first use. `torch` is the only machine-learning dependency.



## Citation

> Ranjbar, S. (2026). *CoastDiff pretrained checkpoints: two-stage CorrDiff coastal downscaling of ERA5 (regression + diffusion)* (v1.0.0). Zenodo. [https://doi.org/10.5281/zenodo.23510727](https://doi.org/10.5281/zenodo.23510727)

!!! Note
    A manuscript describing CoastDiff is in prep.


## Authors

Sadegh Ranjbar, Lucas Gloege, Elizabeth Yankovsky
Department of Earth and Planetary Sciences, Yale University, New Haven, CT, United States of America

## License

Licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)

The model code in `src/coastdiff/models/vendor/` is adapted from
[NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) and its CorrDiff implementation.
**NVIDIA Corporation is the developer of the underlying two-stage regression + diffusion
architecture, the EDM preconditioning and the sampler that this package deploys**, also licensed
under Apache 2.0. The code has been trimmed to the architecture the bundled checkpoints use and
no longer needs PhysicsNeMo installed. See [NOTICE](https://github.com/CDR-Forecasting-Stack/coastdiff/blob/main/NOTICE) for the attribution and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the adaptation was verified against the original.
