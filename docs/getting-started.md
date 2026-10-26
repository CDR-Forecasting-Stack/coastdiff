## Installation

Requires Python 3.11+, internet access (to read ERA5 and, once, to download the checkpoints), and
preferably one NVIDIA GPU (the CPU works but is slow). Create an environment, then install straight
from GitHub:

!!! Note
    CoastDiff will soon be available on [PyPi](https://pypi.org/)

**UV**
```bash
uv init --bare --no-pin-python
uv add git+https://github.com/CDR-Forecasting-Stack/coastdiff.git
coastdiff --help
```

**Conda**
```bash
conda create -n coastdiff python=3.11 -y && conda activate coastdiff
pip install git+https://github.com/CDR-Forecasting-Stack/coastdiff.git
coastdiff --help
```

### Development 

For development, clone the repository and run `pip install -e .` instead.

**Checkpoints.** The first run downloads the two checkpoints (~640 MB) from Zenodo
([doi:10.5281/zenodo.23510727](https://doi.org/10.5281/zenodo.23510727)) into
`~/.cache/coastdiff/checkpoints/`, verifies their SHA-256, and reuses them afterwards. To fetch them
by hand, download both `.mdlus` files from the Zenodo record into
`~/.cache/coastdiff/checkpoints/regression/` and `.../diffusion/` respectively.

`pyproject.toml` pins `torch` and `zarr` to versions verified against the bundled checkpoints and the
ERA5 archive. Keep them pinned: unpinned upgrades have broken checkpoint loading and Zarr access.
No NVIDIA PhysicsNeMo installation is needed (see [License](#authors-and-license)).

## Quick start

The package ships with a worked example: the Gulf of Mexico during the three days of a tropical
cyclone's Florida landfall, hourly.

```bash
bash examples/run_example.sh --quick      # one hour, about a minute
bash examples/run_example.sh              # all 72 hours, about 20-25 minutes on one GPU (~16 s per hour)
python examples/run_example.py [--quick]  # the same through the Python API, with a summary
```

Your own region and time:

```bash
# a single hour
coastdiff --lat-min 24 --lat-max 31 --lon-min -98 --lon-max -80 --timestamp "2026-10-09 12:00"

# every 3 hours for two days, 8 diffusion members, results in ./results
coastdiff --lat-min 30 --lat-max 45 --lon-min -130 --lon-max -115 \
          --start "2025-07-15 00:00" --end "2025-07-16 21:00" --interval 3h \
          --num-ensembles 8 --output-dir results
```

Python:

```python
from datetime import datetime, timezone
from coastdiff import run_inference

paths = run_inference(
    lat_min=24, lat_max=31, lon_min=-98, lon_max=-80,
    timestamps=[datetime(2026, 10, 9, 12, tzinfo=timezone.utc)],
    output_dir="outputs",
)
```

Longitudes are degrees east (west is negative); all times are UTC.

## Defaults

Running `coastdiff` with no arguments uses these (all can be changed):

| setting | default |
|---|---|
| region | Gulf of Mexico: 24-31 N, 98-80 W |
| time | 2026-10-08 00:00 to 2026-10-10 23:00 UTC, every hour (72 files) |
| output folder | `outputs/` (relative to where you run it) |
| diffusion ensemble members | 4 |
| patch overlap | 3 pixels |
| rain filter | on, 0.02 mm |
| device | the GPU if one is visible, otherwise the CPU |

## Model inputs and outputs

**Input (34 channels + 2 fixed fields).** For every patch the model receives ERA5 on a 32 x 32 grid
(~24 km), upsampled to the 256 x 256 grid (~3 km) the model works on, plus two fixed fields on that
fine grid. Everything comes from ERA5; the channel order is fixed by the trained weights.

| group | channels | unit | ERA5 variable |
|---|---|---|---|
| surface wind | `u10`, `v10` | m/s | 10m u/v component of wind |
| surface temperature | `t2m`, `d2m` | K | 2m temperature, 2m dewpoint temperature |
| pressure | `sp`, `msl` | Pa | surface pressure, mean sea level pressure |
| humidity | `q_sfc` | kg/kg | derived from `d2m` and `sp` |
| column water | `tcwv` | kg/m2 | total column water vapour |
| radiation | `ssrd`, `strd` | W/m2 | surface solar / thermal radiation downwards (hourly mean) |
| precipitation | `tp_in` | mm | total precipitation (1 h) |
| cloud, land | `tcc`, `lsm` | 0-1 | total cloud cover, land-sea mask |
| upper air, 1000/850/500/250 hPa | `u`, `v` (m/s), `z` (m), `t` (K), `q` (kg/kg) | | wind, geopotential height, temperature, specific humidity |
| solar geometry | `cos_sza` | 0-1 | cosine of the solar zenith angle (computed) |
| fixed fields (2) | `elev_mean` (m), `lsm_mean` (0-1) | | ERA5 surface geopotential / 9.80665, and land-sea mask |

The channels are named exactly this way in the `input_era5` output group.

**Output (8 channels, ~3 km).**

| variable | meaning | unit |
|---|---|---|
| `2t` | 2 m air temperature | K |
| `10u`, `10v` | 10 m wind components | m/s |
| `tp` | precipitation (1 h accumulation) | mm |
| `ssrd` | downward shortwave radiation at the surface | W/m2 |
| `strd` | downward longwave radiation at the surface | W/m2 |
| `sp` | surface pressure | Pa |
| `q` | near-surface specific humidity | kg/kg |

Every output field is in the same plain physical unit as its input counterpart.

## How the model was trained

CoastDiff is a two-stage CorrDiff model (NVIDIA PhysicsNeMo architecture): a regression network
predicts the mean ~3 km field from a 32 x 32 coarse input, and an EDM diffusion model samples the
residual that restores small-scale structure. It was trained on NOAA HRRR analyses (surface and
pressure-level GRIB2 files), with the 34 coarse input channels above and 8 high-resolution
target channels.

- **Training data:** 120,000 training samples, taken from 1,200 random hourly times over
  2018-2023, 100 coastal patches of 256 x 256 HRRR pixels (~3 km) per time. Each patch has an ocean
  centre pixel and is at least 70 % ocean.
- **Coarse input:** the native HRRR fields were low-pass filtered with a 50 km GCM-Filters scale and
  then conservatively regridded (xESMF) to the 32 x 32 grid (~24 km).
- **Training length:** regression network 5,000,000 samples; diffusion model 50,000,000 samples.
- **Testing:** on independent data by temporal cross-validation: training on 2018-2023, testing on
  10,000 patches from 2024-2025.

ERA5 is used only at inference, as the model's input.

## Geographic coverage and limitations

The model was trained for **regional coastal areas**: ocean-dominated patches inside the HRRR domain
(roughly 21-53 N, 134-61 W: the contiguous US and its coastal waters). CoastDiff places no
restriction on where you run it: it works for any region and any time ERA5 covers, and the Gulf of
Mexico example is just one choice. Results outside the coastal regimes it was trained on have not
been evaluated.

- **Output grid:** a regular latitude/longitude grid of ~3 km spacing is built for each request
  (latitude spacing is fixed; longitude spacing is scaled by the cosine of the central latitude).
  The training grid was a Lambert conformal projection, so cells are close to, but not exactly,
  3 km x 3 km.
- **Fixed fields:** elevation and land-sea mask come from ERA5 (~25 km). The model was trained with
  these at ~3 km, so fine terrain and coastline detail is coarser than in training.
- **Extreme precipitation:** in intense systems such as tropical-cyclone cores the model can produce
  hourly rain far above what the ~25 km ERA5 input suggests (in the Gulf example, peak values near
  180 mm per hour against about 15 mm per hour in the ERA5 input). Treat precipitation extremes
  as unvalidated.
- **Input differences:** ERA5 and the coarsened HRRR fields used in training differ physically
  (clouds, radiation, aerosols and precipitation in particular), so ERA5-driven output is not
  identical in character to what the model saw during training.

## Time ranges and ERA5 availability

`--start`, `--end` and `--interval` (for example `1h`, `6h`, `1d`) request many times at once. ERA5
is read **once** for the whole span (one request per variable, not per hour), and each hour is
regridded and run from that in-memory block, so a long range costs far less input/output than
separate single-hour runs. `--end` is inclusive.

ERA5 is published a few days after real time, and the public archive can be missing individual hours
or days (recent days are sometimes written late). Before doing any work, CoastDiff probes **every
requested hour** and, if some are missing, stops and lists exactly which ones. Add `--skip-missing`
(`skip_missing=True` in Python) to run only the hours that exist; the skipped hours are logged and no
file is written for them. The examples use `--skip-missing`, so they run even if the archive has a gap.

## Output files

One NetCDF4 file per hour: `<output-dir>/coastdiff_<YYYYMMDD>T<HHMM>.nc` with three groups.

| group | contents | dimensions |
|---|---|---|
| `input_era5` | the model's 34 coarse input channels | time, y_lr, x_lr (~24 km) |
| `unet` | regression-only (deterministic) prediction, 8 variables | time, y, x (~3 km) |
| `corrdiff` | regression + diffusion prediction, 8 variables, one slice per ensemble member | time, ensemble, y, x |

Every group carries CF-convention coordinates: a real `time` coordinate and 2-D `lat`/`lon` arrays,
so each group opens on its own:

```python
import xarray as xr
ds = xr.open_dataset("outputs/coastdiff_20261009T1200.nc", group="corrdiff")
ds["2t"]                       # (time, ensemble, y, x), kelvin
ds.lat, ds.lon                 # 2-D coordinates

# join many hours into one time series
all_hours = xr.open_mfdataset("outputs/coastdiff_*.nc", group="unet", combine="by_coords")
```

NaN marks pixels outside the requested box or without ERA5 data. Global attributes record the
region, time, checkpoints used, rain-filter threshold and a description of the fields.

## Rain filter and physical constraints

**Rain filter (on by default).** For `corrdiff`, `tp` uses the deterministic `unet` value except
where `unet` predicts more than 0.02 mm of rain, where the diffusion value is used. This avoids
diffusion noise in dry areas while keeping the textured detail where it is raining. Change the
threshold with `--rain-filter-mm`, or switch it off with `--no-rain-filter`.

**Physical constraints (always applied to `unet` and `corrdiff`; `input_era5` is left as fetched).**
- `tp`, `ssrd` and `strd` are never negative.
- `ssrd` (shortwave) is zero wherever the sun is at or below the horizon (solar zenith angle of
  90 degrees or more), computed for each output pixel from its position and the time. Longwave
  is not zeroed at night.
- Missing pixels stay NaN.

## GPU usage and parallelizing

CoastDiff is designed to use **one GPU per run**, like CorrDiff itself: it uses the GPU it can see
(or the CPU if there is none) and has no GPU-count option. To pick a GPU, restrict what the process
can see, for example `CUDA_VISIBLE_DEVICES=2 coastdiff ...`.

To go faster, run **several independent jobs at once** (one GPU each), splitting the work by time or
by space:

- **By time:** each job takes a different slice of the time range. Hours are independent, so the
  results are identical to one long run.
- **By space:** each job takes a different sub-region (a smaller `--lat-min/--lat-max/--lon-min/--lon-max`).
  Give every sub-region its **own `--output-dir`**: file names contain only the time, so jobs sharing a
  folder would overwrite each other. Each sub-region builds its own grid and its patches do not blend
  across the sub-region edge, so overlap the boxes slightly if you plan to merge them.

SLURM templates are in `scripts/`: `run_inference.sbatch` (one job) and `run_inference_array.sbatch`
(a job array that splits a time range across GPUs). Add your cluster's partition/account lines and
environment activation to them. By hand, four independent jobs:

```bash
for start in "2026-10-08 00:00" "2026-10-08 18:00" "2026-10-09 12:00" "2026-10-10 06:00"; do
  end=$(date -u -d "$start + 17 hours" "+%Y-%m-%d %H:%M")
  sbatch scripts/run_inference.sbatch --start "$start" --end "$end" --interval 1h
done
```

## Command-line options

| option | default | meaning |
|---|---|---|
| `--lat-min`, `--lat-max`, `--lon-min`, `--lon-max` | 24, 31, -98, -80 | region (degrees; west longitudes negative) |
| `--timestamp` | none | a single UTC time, e.g. `"2026-10-09 12:00"` |
| `--start`, `--end`, `--interval` | 2026-10-08 00:00, 2026-10-10 23:00, `1h` | a time range (`--end` inclusive; interval like `1h`, `6h`, `1d`) |
| `--output-dir` | `outputs` | where the NetCDF files are written |
| `--num-ensembles` | 4 | diffusion ensemble members |
| `--overlap` | 3 | pixel overlap between neighbouring patches (0 = none) |
| `--rain-filter-mm` / `--no-rain-filter` | 0.02 / off | rain-filter threshold, or disable it |
| `--skip-missing` | off | run only the hours ERA5 has, instead of stopping when some are missing |
| `--mode` | `all` | `regression` skips the diffusion model for a fast deterministic preview |
| `--sampler`, `--sampler-steps`, `--solver` | `stochastic`, model default, `euler` | diffusion sampler settings |
| `--seed`, `--seed-batch-size` | 0, 1 | random seed; ensemble members run per batch |
| `--config` | `src/coastdiff/configs/default.yaml` | alternative configuration file |

`coastdiff --help` prints the same list.

## Updating the model

The checkpoints are published on Zenodo ([doi:10.5281/zenodo.23510727](https://doi.org/10.5281/zenodo.23510727))
and cached in `~/.cache/coastdiff/checkpoints/{regression,diffusion}/`; the normalization statistics
(`src/coastdiff/assets/stats.json`) are bundled with the package. `src/coastdiff/configs/default.yaml`
points at the checkpoint **folders**, and the highest-iteration `.mdlus` file in each is chosen on
every run. To use a newly trained model, add its two `.mdlus` files to those folders (or point the
config at other folders with `--config`) **and replace `stats.json` with the statistics of the same
training run** (the statistics are not chosen automatically, and they must match the weights). Only the
inference weights (`.mdlus`) are needed, not optimizer state.

## Repository layout and tests

```
coastdiff/
  README.md  LICENSE  NOTICE  pyproject.toml
  examples/                   run_example.py, run_example.sh
  scripts/                    SLURM templates
  docs/ARCHITECTURE.md        how it works and how the model code was verified
  tests/
  src/coastdiff/
    configs/default.yaml      checkpoint folders, statistics path, output folder, rain-filter default
    assets/stats.json         per-channel mean / standard deviation (34 input, 8 output)
    cli.py                    command-line entry point
    inference.py              the pipeline: grid, ERA5, tiling, model, stitching, output
    output.py                 NetCDF writer
    rain_filter.py            physical_constraints.py   time_utils.py   config.py
    data/                     ERA5 fetch and regrid, channel/unit tables, grid and tiling, normalization
    models/                   checkpoint loading and Zenodo download; vendor/ holds the model code
```

Run the tests (the model tests need the checkpoints, downloaded on first use if absent; the rest need no network):

```bash
pip install -e ".[test]"
python -m pytest tests -q
```
