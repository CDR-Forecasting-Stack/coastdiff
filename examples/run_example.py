"""Worked example: downscale ERA5 over the Gulf of Mexico for the three days of a
tropical cyclone's Florida landfall (2026-10-08 to 2026-10-10, hourly), then print a
summary of what was written.

    python examples/run_example.py            # all 72 hours (about 20-25 min on one GPU)
    python examples/run_example.py --quick    # a single hour (about a minute)

Everything here is a default you can change: any lat/lon box and any time ERA5 covers.
"""
import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import xarray as xr

from coastdiff import run_inference

LAT_MIN, LAT_MAX = 24.0, 31.0       # degrees north
LON_MIN, LON_MAX = -98.0, -80.0     # degrees east (west is negative)
START = datetime(2026, 10, 8, 0, tzinfo=timezone.utc)
END = datetime(2026, 10, 10, 23, tzinfo=timezone.utc)

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true", help="run one timestamp only")
p.add_argument("--output-dir", default="outputs/gulf_of_mexico")
args = p.parse_args()

timestamps = [START + timedelta(hours=h) for h in range(int((END - START).total_seconds() // 3600) + 1)]
if args.quick:
    timestamps = [datetime(2026, 10, 9, 12, tzinfo=timezone.utc)]

paths = run_inference(
    lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX,
    timestamps=timestamps,
    output_dir=args.output_dir,
    num_ensembles=4,       # diffusion ensemble members per timestamp
    skip_missing=True,     # run the hours ERA5 has; hours missing from the archive are skipped and reported
)
print(f"\nWrote {len(paths)} file(s) to {args.output_dir}/")

# One file per timestamp; each has three groups: input_era5, unet, corrdiff.
ds = xr.open_dataset(paths[-1], group="corrdiff")
print(f"\nLast file: {paths[-1].name}  (time {str(ds.time.values[0])[:16]} UTC)")
print(f"Grid: {ds.sizes['y']} x {ds.sizes['x']} points, ~3 km; ensemble members: {ds.sizes.get('ensemble', 1)}")
for name in ("2t", "10u", "10v", "tp", "ssrd", "strd", "sp", "q"):
    a = ds[name]
    print(f"  {name:5s} [{a.attrs.get('units', '')}]  min {float(a.min()):10.3f}   max {float(a.max()):10.3f}")
