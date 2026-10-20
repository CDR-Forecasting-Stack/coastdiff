"""ERA5 (ARCO Zarr) input-channel builder, batched over one or more timestamps.

Builds the 34 physical input channels the model expects (coastdiff.data.
channels.IN_CH), for an arbitrary region and one or more timestamps, by
fetching the public ARCO ERA5 Zarr store and bilinearly regridding onto
whatever target lat/lon grid the caller wants (a single inference patch's
32x32 grid, typically).

For multiple timestamps, `Era5RawRegion` fetches the CONTIGUOUS ARCO
time-index span covering all of them ONCE PER VARIABLE, not once per
(timestamp, variable) pair -- the main I/O optimization this package adds
for continuous time-range requests (see README "Continuous time ranges").
Regridding a specific timestamp then slices this already-fetched block in
memory; no further network I/O.

Derived channels (q_sfc, tp_in, ssrd/strd unit conversion, cos_sza) are
computed AFTER regridding the raw fields onto the target grid, not before:
q_sfc's Magnus-formula dependence on d2m/sp is nonlinear, so "regrid then
derive" is the physically consistent order.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np

from .channels import DERIV_CH, IN_CH, PL_LEVS
from .transforms import compute_cos_sza, derive_q_surface, precip_log

ARCO_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
G = 9.80665  # standard gravity, m/s^2 -- geopotential -> geopotential height

ERA5_SFC_VARS = {
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "t2m": "2m_temperature",
    "sp": "surface_pressure",
    "msl": "mean_sea_level_pressure",
    "d2m": "2m_dewpoint_temperature",
    "tcwv": "total_column_water_vapour",
    "ssrd": "surface_solar_radiation_downwards",
    "strd": "surface_thermal_radiation_downwards",
    "tp": "total_precipitation",
    "tcc": "total_cloud_cover",
    "lsm": "land_sea_mask",
}
ERA5_PL_VARS = {
    "u": "u_component_of_wind", "v": "v_component_of_wind",
    "z": "geopotential", "t": "temperature", "q": "specific_humidity",
}

_EXPECTED_CHANNELS = (
    set(ERA5_SFC_VARS) - {"tp"} | {"tp_in", "q_sfc"}
    | {f"{v}{lev}" for v in ERA5_PL_VARS for lev in PL_LEVS}
    | set(DERIV_CH)
)
assert _EXPECTED_CHANNELS == set(IN_CH), (
    "era5.py's channel set has drifted from channels.IN_CH -- "
    f"missing={set(IN_CH) - _EXPECTED_CHANNELS} extra={_EXPECTED_CHANNELS - set(IN_CH)}"
)


def open_arco_store():
    """Anonymous read-only handle to the public ARCO ERA5 Zarr store."""
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    import gcsfs
    import zarr
    fs = gcsfs.GCSFileSystem(token="anon")
    return zarr.open(fs.get_mapper(ARCO_STORE), mode="r")


class Era5UnavailableError(ValueError):
    """Requested times are outside the span the ERA5 store has data for."""


def _has_data(store, idx: int) -> bool:
    # Times not yet written to the store read back as NaN (fill value), so one
    # small read of any always-defined field tells whether a time exists.
    x = np.asarray(store["2m_temperature"][idx, 400:402, 400:402])
    return bool(np.isfinite(x).all())


def _ranges(times):
    """Sorted datetimes -> 'YYYY-MM-DD HH:MM-HH:MM UTC (N h)' style runs of consecutive hours."""
    runs, start, prev = [], None, None
    for t in sorted(times):
        if prev is not None and (t - prev).total_seconds() == 3600:
            prev = t
            continue
        if start is not None:
            runs.append((start, prev))
        start = prev = t
    if start is not None:
        runs.append((start, prev))
    out = []
    for a_, b_ in runs:
        n = int((b_ - a_).total_seconds() // 3600) + 1
        out.append(f"{a_:%Y-%m-%d %H:%M} UTC" if n == 1 else f"{a_:%Y-%m-%d %H:%M} to {b_:%Y-%m-%d %H:%M} UTC ({n} h)")
    return "; ".join(out)


def check_availability(store, timestamps, skip_missing: bool = False, max_probes: int = 200):
    """Returns the requested timestamps that have ERA5 data. If any requested
    hour is missing, raises Era5UnavailableError naming the missing hours --
    unless `skip_missing`, in which case the missing hours are dropped (the
    caller should log them). The store can have gaps (recent days are written
    late), so every requested hour is probed with one small read; for very
    long requests, hours are probed at an even stride (<= `max_probes`)."""
    from concurrent.futures import ThreadPoolExecutor
    timestamps = list(timestamps)
    n = int(store["time"].shape[0])
    idxs = [era5_time_index(store, t) for t in timestamps]
    if min(idxs) < 0 or max(idxs) >= n:
        raise Era5UnavailableError("Requested times fall outside the span of the ERA5 store.")
    stride = max(1, -(-len(timestamps) // max_probes))
    probe = list(range(0, len(timestamps), stride))
    with ThreadPoolExecutor(max_workers=16) as ex:
        ok = list(ex.map(lambda k: _has_data(store, idxs[k]), probe))
    if stride > 1:   # unprobed hours inherit the neighbouring probe's answer
        flags = [ok[min(k // stride, len(ok) - 1)] for k in range(len(timestamps))]
    else:
        flags = ok
    missing = [t for t, f in zip(timestamps, flags) if not f]
    have = [t for t, f in zip(timestamps, flags) if f]
    if not missing:
        return timestamps
    note = ("ERA5 has no data for: " + _ranges(missing) + ". ERA5 is published with a delay of several days, "
            "and recent days can be written late.")
    if not have:
        raise Era5UnavailableError(note + " Choose other times.")
    if not skip_missing:
        raise Era5UnavailableError(note + " Choose other times, or add --skip-missing to run only the "
                                   f"{len(have)} available hour(s).")
    return have


def era5_time_index(store, dt_utc: datetime) -> int:
    epoch = datetime(1900, 1, 1, tzinfo=timezone.utc)
    target_h = int((dt_utc - epoch).total_seconds() // 3600)
    return target_h - int(store["time"][0])


def era5_latlon_box(store, lat_min, lat_max, lon_min, lon_max, pad=3):
    """Returns (lat_slice, lon_slice, era5_lat_1d, era5_lon_1d) covering the
    requested box plus `pad` extra ERA5 grid cells on each side, so bilinear
    regridding near the box edge never needs to extrapolate."""
    lats = np.array(store["latitude"])   # 90 -> -90
    lons = np.array(store["longitude"])  # 0 -> 359.75
    lon_min_360, lon_max_360 = lon_min % 360, lon_max % 360
    lat_i_hi = max(0, int(np.searchsorted(-lats, -lat_max, side="left")) - pad)
    lat_i_lo = min(len(lats), int(np.searchsorted(-lats, -lat_min, side="right")) + pad)
    lon_i_lo = max(0, int(np.searchsorted(lons, lon_min_360, side="left")) - pad)
    lon_i_hi = min(len(lons), int(np.searchsorted(lons, lon_max_360, side="right")) + pad)
    return (slice(lat_i_hi, lat_i_lo), slice(lon_i_lo, lon_i_hi),
            lats[lat_i_hi:lat_i_lo], lons[lon_i_lo:lon_i_hi])


def regrid_linear(src_lat, src_lon, values2d, tgt_lat2d, tgt_lon2d):
    """src_lat descending (N->S); flipped to ascending for
    RegularGridInterpolator. Out-of-bounds -> NaN (no extrapolation)."""
    from scipy.interpolate import RegularGridInterpolator
    src_lat_asc = src_lat[::-1]
    values_asc = values2d[::-1]
    tgt_lon_360 = tgt_lon2d % 360
    interp = RegularGridInterpolator(
        (src_lat_asc.astype(np.float64), src_lon.astype(np.float64)),
        values_asc.astype(np.float64), method="linear", bounds_error=False, fill_value=np.nan,
    )
    pts = np.stack([tgt_lat2d.ravel(), tgt_lon_360.ravel()], axis=-1)
    return interp(pts).reshape(tgt_lat2d.shape).astype(np.float32)


class Era5Slice:
    """One timestamp's worth of raw (un-regridded) ERA5 fields -- a view
    into an Era5RawRegion's batch-fetched arrays, no additional I/O."""

    __slots__ = ("dt", "raw", "era5_lat", "era5_lon")

    def __init__(self, dt, raw, era5_lat, era5_lon):
        self.dt = dt
        self.raw = raw  # dict[str, (Hlat,Wlon) float32]
        self.era5_lat = era5_lat
        self.era5_lon = era5_lon


GLOBAL_INVARIANT_VARS = {
    # Time-invariant ERA5 static fields (still stored with a time axis in
    # the ARCO store, but constant across it) -- this package's only source
    # for the model's two invariant conditioning channels (elevation,
    # land-sea mask), anywhere in the world. See grid.py:build_generic_grid
    # and docs/ARCHITECTURE.md "Geographic coverage".
    "elev_geopotential": "geopotential_at_surface",  # -> meters via /G
    "lsm_global": "land_sea_mask",
}


class Era5RawRegion:
    """Raw (un-regridded) ERA5 fields for one lat/lon box, batch-fetched
    over one or more timestamps. Fetch is the expensive part (GCS
    round-trip); regridding per timestamp/per patch afterward is cheap.
    Always also fetches the global invariant fields (elevation, land-sea
    mask; see `global_invariants()`) -- cheap, since it reuses the same
    already-open box, and every request needs them (this package has no
    other source for the model's invariant conditioning channels)."""

    def __init__(self, timestamps, store, lat_min, lat_max, lon_min, lon_max, pad=3):
        self.timestamps = list(timestamps)
        if not self.timestamps:
            raise ValueError("Era5RawRegion requires at least one timestamp")
        t_indices = [era5_time_index(store, dt) for dt in self.timestamps]
        t_lo, t_hi = min(t_indices), max(t_indices)
        t_sl = slice(t_lo, t_hi + 1)
        offsets = {dt: idx - t_lo for dt, idx in zip(self.timestamps, t_indices)}

        lat_sl, lon_sl, self.era5_lat, self.era5_lon = era5_latlon_box(
            store, lat_min, lat_max, lon_min, lon_max, pad=pad
        )

        raw_block: dict[str, np.ndarray] = {}  # name -> (T, Hlat, Wlon)
        for short, era5_name in ERA5_SFC_VARS.items():
            raw_block[short] = np.array(store[era5_name][t_sl, lat_sl, lon_sl], dtype=np.float32)
        levels = np.array(store["level"])
        for v_short, era5_name in ERA5_PL_VARS.items():
            for lev in PL_LEVS:
                lev_idx = int(np.where(levels == lev)[0][0])
                raw = np.array(store[era5_name][t_sl, lev_idx, lat_sl, lon_sl], dtype=np.float32)
                if v_short == "z":
                    raw = raw / G  # geopotential (m^2/s^2) -> geopotential height (m)
                raw_block[f"{v_short}{lev}"] = raw

        # Time-invariant, so just t_lo -- no separate GCS round-trip box needed.
        geopot = np.array(store[GLOBAL_INVARIANT_VARS["elev_geopotential"]][t_lo, lat_sl, lon_sl],
                           dtype=np.float32)
        lsm = np.array(store[GLOBAL_INVARIANT_VARS["lsm_global"]][t_lo, lat_sl, lon_sl],
                        dtype=np.float32)
        self._global_invariants = (geopot / G, lsm)  # (elevation_m, land_fraction)

        self._raw_block = raw_block
        self._offsets = offsets

    def slice_for(self, dt: datetime) -> Era5Slice:
        """One timestamp's 2D fields, sliced from the batch fetch (no I/O)."""
        i = self._offsets[dt]
        raw_2d = {k: v[i] for k, v in self._raw_block.items()}
        return Era5Slice(dt, raw_2d, self.era5_lat, self.era5_lon)

    def global_invariants(self):
        """(elevation_m, land_fraction) rasters at ERA5's native resolution
        over the fetched box, plus (era5_lat, era5_lon) coordinates for
        regridding onto the output grid."""
        elev, lsm = self._global_invariants
        return elev, lsm, self.era5_lat, self.era5_lon


def build_era5_input_stack(era5_slice: Era5Slice, tgt_lat2d: np.ndarray, tgt_lon2d: np.ndarray) -> np.ndarray:
    """Regrids one timestamp's raw fields onto (tgt_lat2d, tgt_lon2d) and
    assembles the 34-channel input stack in channels.IN_CH order."""
    def rg(short):
        return regrid_linear(era5_slice.era5_lat, era5_slice.era5_lon,
                              era5_slice.raw[short], tgt_lat2d, tgt_lon2d)

    d2m = rg("d2m")
    sp = rg("sp")
    out = {
        "u10": rg("u10"), "v10": rg("v10"), "t2m": rg("t2m"),
        "sp": sp, "msl": rg("msl"), "d2m": d2m,
        "q_sfc": derive_q_surface(d2m, sp),
        "tcwv": rg("tcwv"),
        "ssrd": np.clip(rg("ssrd") / 3600.0, 0, None),
        "strd": np.clip(rg("strd") / 3600.0, 0, None),
        "tp_in": precip_log(np.clip(rg("tp") * 1000.0, 0, None)),
        "tcc": rg("tcc"),
        "lsm": rg("lsm"),
    }
    for v_short in ERA5_PL_VARS:
        for lev in PL_LEVS:
            out[f"{v_short}{lev}"] = rg(f"{v_short}{lev}")
    out["cos_sza"] = compute_cos_sza(tgt_lat2d, tgt_lon2d, era5_slice.dt)

    return np.stack([out[ch] for ch in IN_CH], axis=0).astype(np.float32)
