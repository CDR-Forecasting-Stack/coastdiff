"""Model channel definitions: names, order, and physical units.

The order of IN_CH/OUT_CH is load-bearing — it must match the order the
checkpoints were trained with. Do not reorder.
"""

PL_LEVS = [1000, 850, 500, 250]  # hPa

# Coarse (~25km) input channels, surface (13).
SFC_CH = ["u10", "v10", "t2m", "sp", "msl", "d2m", "q_sfc",
          "tcwv", "ssrd", "strd", "tp_in", "tcc", "lsm"]

# Coarse input channels, pressure-level: u,v,z,t,q at 4 levels = 20.
PL_CH = [f"{v}{lev}" for v in ["u", "v", "z", "t", "q"] for lev in PL_LEVS]

# Derived coarse input channel.
DERIV_CH = ["cos_sza"]

IN_CH = SFC_CH + PL_CH + DERIV_CH  # 13 + 20 + 1 = 34
INVARIANT_CH = ["elev_mean", "lsm_mean"]

# High-resolution (~3km) output channels (8).
OUT_CH = ["2t", "10u", "10v", "tp", "ssrd", "strd", "sp", "q"]

# Physical units. tp/tp_in are stored internally as log10(1+mm) for model
# I/O (numerical stability at training time), but are converted back to mm
# before being written to output files — see normalization.py — so both are
# listed here in the natural unit a user sees in the output.
UNITS = {
    "u10": "m/s", "v10": "m/s", "t2m": "K", "sp": "Pa", "msl": "Pa", "d2m": "K",
    "q_sfc": "kg/kg", "tcwv": "kg/m2", "ssrd": "W/m2", "strd": "W/m2",
    "tp_in": "mm", "tcc": "0-1", "lsm": "0-1 (land fraction)",
    "2t": "K", "10u": "m/s", "10v": "m/s", "tp": "mm", "q": "kg/kg",
}
for _v in ["u", "v", "z", "t", "q"]:
    for _lev in PL_LEVS:
        UNITS[f"{_v}{_lev}"] = {"u": "m/s", "v": "m/s", "z": "m", "t": "K", "q": "kg/kg"}[_v]

# Output variable -> the coarse input channel of the same physical quantity.
INPUT_MATCH = {"2t": "t2m", "10u": "u10", "10v": "v10", "tp": "tp_in",
               "ssrd": "ssrd", "strd": "strd", "sp": "sp", "q": "q_sfc"}
