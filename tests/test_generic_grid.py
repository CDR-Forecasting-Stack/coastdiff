"""The output grid (build_generic_grid) is used for every request, anywhere
ERA5 has data -- see docs/ARCHITECTURE.md "Geographic coverage". No GPU/
network required."""
from coastdiff.data.grid import HR_SIZE, bbox_to_rowcol, build_generic_grid


def test_generic_grid_covers_requested_box():
    lat_min, lat_max, lon_min, lon_max = 45, 50, 5, 10  # the Alps
    grid = build_generic_grid(lat_min, lat_max, lon_min, lon_max)
    assert grid["latitude"].shape == grid["longitude"].shape
    assert grid["latitude"].shape[0] >= HR_SIZE
    assert grid["latitude"].shape[1] >= HR_SIZE
    assert grid["latitude"].min() <= lat_min and grid["latitude"].max() >= lat_max
    assert grid["longitude"].min() <= lon_min and grid["longitude"].max() >= lon_max
    # bbox_to_rowcol must succeed against a grid built to contain the box.
    row0, row1, col0, col1, mask = bbox_to_rowcol(grid, lat_min, lat_max, lon_min, lon_max)
    assert mask[row0:row1, col0:col1].any()


def test_generic_grid_pads_small_regions_to_min_patch_size():
    grid = build_generic_grid(1, 1.1, 1, 1.1)  # a tiny box
    assert grid["latitude"].shape[0] >= HR_SIZE
    assert grid["latitude"].shape[1] >= HR_SIZE


def test_generic_grid_works_for_any_region():
    """The whole point: no domain restriction -- CONUS, mid-ocean, and the
    poles all just work the same way."""
    for lat_min, lat_max, lon_min, lon_max in [
        (34, 38, -125, -121),    # California coast
        (-40, -35, 170, 175),    # New Zealand
        (-10, -5, -160, -155),   # open Pacific
        (60, 65, 20, 25),        # Scandinavia
    ]:
        grid = build_generic_grid(lat_min, lat_max, lon_min, lon_max)
        row0, row1, col0, col1, mask = bbox_to_rowcol(grid, lat_min, lat_max, lon_min, lon_max)
        assert mask[row0:row1, col0:col1].any()
