"""Command-line entry point (console script: `coastdiff`)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config
from .rain_filter import DEFAULT_RAIN_FILTER_MM
from .data.era5 import Era5UnavailableError
from .time_utils import generate_range, parse_timestamp

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("coastdiff.cli")

# Defaults: the Gulf of Mexico, and the three days of a tropical cyclone's
# Florida landfall (2026-10-08 to 2026-10-10, hourly) -- a worked example,
# not a restriction: any region and any time ERA5 covers can be requested.
DEFAULT_LAT_MIN, DEFAULT_LAT_MAX = 24.0, 31.0
DEFAULT_LON_MIN, DEFAULT_LON_MAX = -98.0, -80.0
DEFAULT_START, DEFAULT_END, DEFAULT_INTERVAL = "2026-10-08 00:00", "2026-10-10 23:00", "1h"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="coastdiff",
        description="ERA5-driven high-resolution (~3km) weather field inference.",
    )
    p.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    p.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    p.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    p.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    p.add_argument("--timestamp", default=None,
                    help="a single time, UTC, e.g. '2026-10-09 12:00' (overrides the default range)")
    p.add_argument("--start", default=None, help=f"start of a continuous range, UTC (default {DEFAULT_START})")
    p.add_argument("--end", default=None, help=f"end of a continuous range, UTC, inclusive (default {DEFAULT_END})")
    p.add_argument("--interval", default=DEFAULT_INTERVAL,
                    help=f"step for --start/--end, e.g. '1h', '6h', '1d' (default {DEFAULT_INTERVAL})")
    p.add_argument("--output-dir", default=None, help="default: coastdiff/configs/default.yaml's output_dir")
    p.add_argument("--config", default=None, help="default: coastdiff/configs/default.yaml")
    p.add_argument("--num-ensembles", type=int, default=4)
    p.add_argument("--seed-batch-size", type=int, default=1)
    p.add_argument("--overlap", type=int, default=3,
                    help="pixel overlap (256-native grid) between adjacent patches, 0=no blending")
    p.add_argument("--mode", choices=["all", "regression"], default="all",
                    help="'regression' skips the diffusion model for a fast UNet-only preview")
    p.add_argument("--sampler", choices=["stochastic", "deterministic"], default="stochastic")
    p.add_argument("--sampler-steps", type=int, default=None)
    p.add_argument("--solver", default="euler")
    p.add_argument("--rain-filter-mm", type=float, default=DEFAULT_RAIN_FILTER_MM)
    p.add_argument("--no-rain-filter", action="store_true")
    p.add_argument("--skip-missing", action="store_true",
                    help="run only the hours that have ERA5 data instead of stopping when some are missing")
    p.add_argument("--seed", type=int, default=0)
    return p


def resolve_timestamps(args) -> list:
    if args.timestamp and (args.start or args.end):
        raise ValueError("use either --timestamp or --start/--end, not both")
    if args.timestamp:
        return [parse_timestamp(args.timestamp)]
    if bool(args.start) != bool(args.end):
        raise ValueError("--start and --end must be given together")
    return generate_range(args.start or DEFAULT_START, args.end or DEFAULT_END, args.interval)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    try:
        timestamps = resolve_timestamps(args)
    except ValueError as e:
        log.error(str(e))
        return 2

    output_dir = Path(args.output_dir or cfg.get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    run_kwargs = dict(
        lat_min=args.lat_min, lat_max=args.lat_max, lon_min=args.lon_min, lon_max=args.lon_max,
        output_dir=output_dir, config_path=args.config,
        num_ensembles=args.num_ensembles, overlap=args.overlap, mode=args.mode, seed=args.seed,
        seed_batch_size=args.seed_batch_size, sampler=args.sampler, sampler_steps=args.sampler_steps,
        solver=args.solver, skip_missing=args.skip_missing, rain_filter_mm=(None if args.no_rain_filter else args.rain_filter_mm),
    )

    from .inference import run_inference
    try:
        run_inference(timestamps=timestamps, **run_kwargs)
    except Era5UnavailableError as e:
        log.error(str(e))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
