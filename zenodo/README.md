# CoastDiff pretrained checkpoints (v1.0.0)

Trained weights for **CoastDiff**, a two-stage CorrDiff model that downscales ERA5 (~25 km) to ~3 km
surface fields over coastal regions.

- Code, documentation and examples: https://github.com/CDR-Forecasting-Stack/coastdiff
- This record: https://doi.org/10.5281/zenodo.23510727

## Files

| file | stage | size | MD5 |
|---|---|---|---|
| `CorrDiffRegressionUNet.0.5000000.mdlus` | deterministic regression UNet | 319.09 MB | `fc31299cd3ef1a2af01417d875cfea48` |
| `EDMPrecondSuperResolution.0.50000000.mdlus` | EDM diffusion residual model | 319.12 MB | `79f47a77696fbf5a01cd8395adca06f0` |

The two files are a matched pair and must be used together with the `assets/stats.json`
normalization statistics shipped in the GitHub repository (same training run). The number in each
file name is the training iteration (in samples): 5,000,000 for the regression network and
50,000,000 for the diffusion model.

## Usage

Install the package from GitHub; the checkpoints are fetched from this record on first use:

```bash
pip install git+https://github.com/CDR-Forecasting-Stack/coastdiff.git
coastdiff --help
```

To place them manually, put the regression file in `checkpoints/regression/` and the diffusion file
in `checkpoints/diffusion/` of the repository (see "Updating the model" in its README).

## Model summary

- **Architecture:** NVIDIA PhysicsNeMo CorrDiff (regression UNet + EDM diffusion), adapted and
  vendored in the CoastDiff repository; PyTorch is the only ML dependency.
- **Input:** 34 ERA5 channels on a 32 x 32 grid (~24 km) plus 2 fixed fields (elevation, land-sea mask).
- **Output:** 8 channels on a 256 x 256 patch (~3 km): `2t`, `10u`, `10v`, `tp`, `ssrd`, `strd`, `sp`, `q`.
- **Training data:** NOAA HRRR analyses; 120,000 coastal patches (256 x 256 pixels, at least 70 % ocean)
  from 1,200 random hourly times in 2018-2023. Coarse inputs were Gaussian-filtered (sigma = 5 native
  pixels) and conservatively regridded to ~24 km.
- **Testing:** temporal hold-out, 10,000 patches from 2024-2025.
- **Inference:** the coarse input comes from ERA5 instead of coarsened HRRR.

## Limitations

- Trained on ocean-dominated patches in the HRRR domain (about 21-53 N, 134-61 W). It runs anywhere
  ERA5 exists, but skill outside the training regimes has not been evaluated.
- ERA5 and coarsened HRRR differ physically (clouds, radiation, aerosols, precipitation), so
  ERA5-driven output differs in character from the training inputs.
- Extreme precipitation (for example tropical-cyclone cores) is unvalidated and can be strongly
  overestimated.
- Elevation and land-sea mask come from ERA5 (~25 km) at inference, coarser than in training.

## License and attribution

- Weights: CC BY 4.0. Code: Apache-2.0 (GitHub repository).
- The architecture, EDM preconditioning and sampler are from NVIDIA PhysicsNeMo (Apache-2.0).
- Training targets are NOAA HRRR; the inference input is Copernicus ERA5 (please follow the
  Copernicus licence and cite Hersbach et al., 2020).

## Citation

Ranjbar, S. (2026). *CoastDiff pretrained checkpoints: two-stage CorrDiff coastal downscaling of ERA5 (regression + diffusion)* (v1.0.0). Zenodo.
https://doi.org/10.5281/zenodo.23510727

## Authors

Sadegh Ranjbar, Lucas Gloege, Elizabeth Yankovsky
Department of Earth and Planetary Sciences, Yale University, New Haven, CT, USA
