"""NetCDF output writer. Every field is in a plain physical unit (K, m/s,
Pa, W/m2, kg/kg, mm) -- tp/tp_in are converted from the model's internal
log10(1+mm) representation before reaching this module."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np

from .data.channels import IN_CH, OUT_CH, UNITS


def _write_coords(group, dt: datetime, lat: np.ndarray, lon: np.ndarray,
                   y_dim: str, x_dim: str, lat_name: str, lon_name: str, suffix: str = ""):
    """Writes time/lat/lon coordinate variables into `group`. netCDF4 groups
    inherit parent-defined DIMENSIONS automatically, but not parent
    VARIABLES -- so each group needs its own copy of these coordinate
    variables to be self-describing when opened independently (e.g.
    `xr.open_dataset(path, group="unet")`), rather than only being visible
    from the file root."""
    v = group.createVariable("time", "f8", ("time",))
    v[:] = [dt.timestamp()]
    v.standard_name = "time"; v.long_name = "time"
    v.units = "seconds since 1970-01-01 00:00:00"
    v.calendar = "proleptic_gregorian"

    v = group.createVariable(lat_name, "f4", (y_dim, x_dim), zlib=True)
    v[:] = lat
    v.standard_name = "latitude"; v.long_name = f"latitude{suffix}"; v.units = "degrees_north"
    v = group.createVariable(lon_name, "f4", (y_dim, x_dim), zlib=True)
    v[:] = lon
    v.standard_name = "longitude"; v.long_name = f"longitude{suffix}"; v.units = "degrees_east"


def write_output(path: str | Path, *, dt: datetime, lat_min: float, lat_max: float,
                  lon_min: float, lon_max: float, overlap: int, num_ensembles: int,
                  rain_filter_mm: float | None, lat: np.ndarray, lon: np.ndarray,
                  lat_lr: np.ndarray, lon_lr: np.ndarray, input_era5: np.ndarray,
                  unet: np.ndarray, corrdiff: np.ndarray | None,
                  regression_checkpoint: str, diffusion_checkpoint: str | None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = nc.Dataset(path, "w", format="NETCDF4")
    ny, nx = lat.shape
    ny_lr, nx_lr = lat_lr.shape
    ds.createDimension("time", 1)
    ds.createDimension("y", ny)
    ds.createDimension("x", nx)
    ds.createDimension("y_lr", ny_lr)
    ds.createDimension("x_lr", nx_lr)
    if corrdiff is not None:
        ds.createDimension("ensemble", corrdiff.shape[0])

    # Root-level coordinates, for tools that open the whole file directly
    # (netCDF4.Dataset, ncdump, ncview, ...).
    _write_coords(ds, dt, lat, lon, "y", "x", "lat", "lon")
    v = ds.createVariable("lat_lr", "f4", ("y_lr", "x_lr"), zlib=True); v[:] = lat_lr
    v.standard_name = "latitude"; v.long_name = "latitude (coarse grid)"; v.units = "degrees_north"
    v = ds.createVariable("lon_lr", "f4", ("y_lr", "x_lr"), zlib=True); v[:] = lon_lr
    v.standard_name = "longitude"; v.long_name = "longitude (coarse grid)"; v.units = "degrees_east"

    # Each group also gets its own copy of time/lat/lon (see _write_coords'
    # docstring) so `xr.open_dataset(path, group=<name>)` is a complete,
    # self-describing dataset on its own -- no need to separately open the
    # root group to get real coordinate values.
    gi = ds.createGroup("input_era5")
    _write_coords(gi, dt, lat_lr, lon_lr, "y_lr", "x_lr", "lat_lr", "lon_lr", " (coarse grid)")
    for i, ch in enumerate(IN_CH):
        v = gi.createVariable(ch, "f4", ("time", "y_lr", "x_lr"), zlib=True, complevel=4,
                               fill_value=np.float32(np.nan))
        v[:] = input_era5[i][None]
        v.units = UNITS.get(ch, ""); v.coordinates = "lat_lr lon_lr"

    gu = ds.createGroup("unet")
    _write_coords(gu, dt, lat, lon, "y", "x", "lat", "lon")
    for i, ch in enumerate(OUT_CH):
        v = gu.createVariable(ch, "f4", ("time", "y", "x"), zlib=True, complevel=4,
                               fill_value=np.float32(np.nan))
        v[:] = unet[i][None]
        v.units = UNITS.get(ch, ""); v.coordinates = "lat lon"

    if corrdiff is not None:
        gc = ds.createGroup("corrdiff")
        _write_coords(gc, dt, lat, lon, "y", "x", "lat", "lon")
        for i, ch in enumerate(OUT_CH):
            v = gc.createVariable(ch, "f4", ("time", "ensemble", "y", "x"), zlib=True, complevel=4,
                                   fill_value=np.float32(np.nan))
            v[:] = corrdiff[:, i][None]
            v.units = UNITS.get(ch, ""); v.coordinates = "lat lon"

    ds.Conventions = "CF-1.8"
    ds.title = "coastdiff -- ERA5-driven high-resolution (~3km) weather fields"
    ds.timestamp = dt.isoformat()
    ds.input_source = "era5"
    ds.lat_min, ds.lat_max = lat_min, lat_max
    ds.lon_min, ds.lon_max = lon_min, lon_max
    ds.overlap_px = overlap
    ds.num_ensembles = num_ensembles if corrdiff is not None else 0
    ds.regression_checkpoint = str(regression_checkpoint)
    ds.diffusion_checkpoint = str(diffusion_checkpoint) if diffusion_checkpoint else "none"
    ds.rain_filter_mm = rain_filter_mm if rain_filter_mm is not None else -1.0
    ds.grid_source = "generic_global"
    ds.notes = (
        "input_era5 = the model's 34-channel coarse conditioning field (ERA5-sourced), "
        "stitched at low resolution; unet = regression-only (deterministic mean) prediction; "
        "corrdiff = regression + diffusion residual per ensemble member "
        f"(tp rain-filtered above {rain_filter_mm}mm using unet's own prediction, see rain_filter.py; "
        "omitted if rain_filter_mm < 0). "
        "All fields are in plain physical units (see each variable's 'units' attribute) -- "
        "tp/tp_in are mm, converted from the model's internal log10(1+mm) representation. "
        "unet/corrdiff ssrd, strd and tp are clipped to >= 0, and ssrd is set to 0 where the solar zenith angle is >= 90 degrees (no daylight); input_era5 is left as fetched. "
        "NaN = outside the requested bbox or ERA5 data was unavailable for that pixel. "
        "grid_source='generic_global': a regular lat/lon output grid (~3km spacing, built for "
        "this request) and ERA5-derived (~25km) elevation/land-sea mask are used as the model's "
        "invariant conditioning channels (see README 'Geographic coverage')."
    )
    ds.created = datetime.utcnow().isoformat()
    ds.close()
    return path
