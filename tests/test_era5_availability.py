"""ERA5 availability check against a fake store. No GPU/network required."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from coastdiff.data import era5

EPOCH = datetime(1900, 1, 1, tzinfo=timezone.utc)


class _FakeVar:
    def __init__(self, missing):
        self.missing = set(missing)

    def __getitem__(self, key):
        return np.full((2, 2), np.nan if key[0] in self.missing else 280.0, dtype=np.float32)


class _FakeStore(dict):
    def __init__(self, n_total, missing=()):
        super().__init__()
        self["time"] = type("T", (), {"shape": (n_total,), "__getitem__": lambda s, k: 0})()
        self["2m_temperature"] = _FakeVar(missing)


def _hours(*hs):
    return [EPOCH + timedelta(hours=h) for h in hs]


def test_all_present_returns_everything():
    store = _FakeStore(1000)
    assert era5.check_availability(store, _hours(0, 100, 499)) == _hours(0, 100, 499)


def test_gap_in_the_middle_is_found_and_named():
    store = _FakeStore(1000, missing=range(240, 264))                   # a whole missing day
    ts = _hours(*range(200, 300))
    with pytest.raises(era5.Era5UnavailableError) as e:
        era5.check_availability(store, ts)
    msg = str(e.value)
    assert "1900-01-11 00:00 to 1900-01-11 23:00 UTC (24 h)" in msg
    assert "--skip-missing" in msg


def test_skip_missing_returns_only_available_hours():
    store = _FakeStore(1000, missing=range(240, 264))
    ts = _hours(*range(200, 300))
    have = era5.check_availability(store, ts, skip_missing=True)
    assert len(have) == 76 and all(not (240 <= (t - EPOCH).total_seconds() / 3600 < 264) for t in have)


def test_nothing_available_is_an_error_even_with_skip_missing():
    store = _FakeStore(1000, missing=range(1000))
    with pytest.raises(era5.Era5UnavailableError):
        era5.check_availability(store, _hours(5, 6), skip_missing=True)


def test_outside_store_span_is_an_error():
    with pytest.raises(era5.Era5UnavailableError):
        era5.check_availability(_FakeStore(100), _hours(500))
