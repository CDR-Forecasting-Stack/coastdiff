"""CLI defaults and time parsing. No GPU/network required."""
from datetime import timezone

import pytest

from coastdiff.cli import build_parser, resolve_timestamps


def _args(*argv):
    return build_parser().parse_args(list(argv))


def test_default_region_is_gulf_of_mexico():
    a = _args()
    assert (a.lat_min, a.lat_max, a.lon_min, a.lon_max) == (24.0, 31.0, -98.0, -80.0)


def test_default_time_is_three_days_hourly():
    ts = resolve_timestamps(_args())
    assert len(ts) == 72
    assert ts[0].isoformat() == "2026-10-08T00:00:00+00:00"
    assert ts[-1].isoformat() == "2026-10-10T23:00:00+00:00"
    assert all(t.tzinfo == timezone.utc for t in ts)


def test_single_timestamp_overrides_default_range():
    ts = resolve_timestamps(_args("--timestamp", "2026-10-09 12:00"))
    assert [t.isoformat() for t in ts] == ["2026-10-09T12:00:00+00:00"]


def test_start_without_end_and_mixed_arguments_are_errors():
    with pytest.raises(ValueError):
        resolve_timestamps(_args("--start", "2026-10-08 00:00"))
    with pytest.raises(ValueError):
        resolve_timestamps(_args("--timestamp", "2026-10-09 12:00", "--start", "2026-10-08 00:00",
                                 "--end", "2026-10-09 00:00"))


def test_no_gpu_option():
    with pytest.raises(SystemExit):
        _args("--gpus", "4")
