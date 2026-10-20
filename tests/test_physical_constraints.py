"""Physical-consistency constraints on the output. No GPU/network required."""
from datetime import datetime, timezone

import numpy as np

from coastdiff.data.channels import OUT_CH
from coastdiff.physical_constraints import (
    SSRD_IDX, STRD_IDX, TP_IDX, apply_physical_constraints, solar_cos_zenith,
)


def _field(shape=(2, len(OUT_CH), 4, 4), value=-1.0):
    return np.full(shape, value, dtype=np.float32)


def test_negatives_clipped_to_zero_for_ssrd_strd_tp_only():
    f = _field()
    out = apply_physical_constraints(f, np.ones((4, 4), dtype=np.float32))  # all daylight
    for idx in (SSRD_IDX, STRD_IDX, TP_IDX):
        assert (out[:, idx] == 0).all()
    other = [i for i in range(len(OUT_CH)) if i not in (SSRD_IDX, STRD_IDX, TP_IDX)]
    assert (out[:, other] == -1.0).all()          # 2t/winds/sp/q untouched
    assert (f == -1.0).all()                       # input not modified


def test_ssrd_zero_at_night_but_positive_values_kept_by_day():
    f = _field(value=300.0)
    cos = np.array([[1.0, 0.5], [0.0, 0.0]], dtype=np.float32)   # bottom row: sun at/below horizon
    cos = np.pad(cos, ((0, 2), (0, 2)), constant_values=1.0)
    out = apply_physical_constraints(f, cos)
    assert (out[:, SSRD_IDX][:, 1, :2] == 0).all()
    assert (out[:, SSRD_IDX][:, 0, :2] == 300.0).all()
    assert (out[:, STRD_IDX] == 300.0).all()      # longwave is NOT zeroed at night


def test_nan_is_preserved():
    f = _field(value=5.0)
    f[:, SSRD_IDX, 0, 0] = np.nan
    out = apply_physical_constraints(f, np.zeros((4, 4), dtype=np.float32))  # all night
    assert np.isnan(out[:, SSRD_IDX, 0, 0]).all()
    assert (out[:, SSRD_IDX, 1:, 1:] == 0).all()


def test_solar_cos_zenith_day_and_night():
    lat = np.full((1, 2), 0.0, dtype=np.float32)
    lon = np.array([[0.0, 180.0]], dtype=np.float32)
    cos = solar_cos_zenith(lat, lon, datetime(2025, 3, 20, 12, 0, tzinfo=timezone.utc))
    assert cos[0, 0] > 0.9 and cos[0, 1] == 0.0    # noon at 0E, midnight at 180E
