from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from landlab import RasterModelGrid


# ---------------------------------------------------------------------
# Edit these paths/parameters, then run:
# python 02_build_slope_area_index.py
# ---------------------------------------------------------------------

base_dir = Path("/mnt/c/Users/amehedi/Downloads/thomas")

elevation_path = base_dir / "topographic__elevation.tif"

# Preferred for this index. You can switch this to drainage_area.tif if needed.
contributing_area_path = base_dir / "topographic__specific_contributing_area.tif"
# contributing_area_path = base_dir / "drainage_area.tif"

out_tif = base_dir / "slope_area_index.tif"
out_png = base_dir / "slope_area_index.png"

ALPHA = 1.125  # slope exponent. 1.125 to 2.0.
OUT_NODATA = -9999.0


def calculate_landlab_slope(elevation, dx, dy):
    rows, cols = elevation.shape
    grid = RasterModelGrid((rows, cols), xy_spacing=(abs(dx), abs(dy)))

    elev_valid = ~np.ma.getmaskarray(elevation)
    z_raster = elevation.filled(0.0).astype(float)

    # Raster rows start at the top; Landlab rows start at the bottom.
    z_landlab = np.flipud(z_raster).ravel()
    valid_landlab = np.flipud(elev_valid).ravel()

    grid.add_field("topographic__elevation", z_landlab, at="node")
    grid.status_at_node[~valid_landlab] = grid.BC_NODE_IS_CLOSED

    slope_radians = grid.calc_slope_at_node(
        "topographic__elevation",
        method="Horn",
        ignore_closed_nodes=True,
    )
    slope_rise_run = np.tan(slope_radians).reshape((rows, cols))
    return np.flipud(slope_rise_run)


with rasterio.open(elevation_path) as elev_src:
    elevation = elev_src.read(1, masked=True).astype("float32")
    profile = elev_src.profile.copy()
    elev_shape = elev_src.shape
    elev_transform = elev_src.transform
    elev_crs = elev_src.crs
    dx, dy = elev_src.res

with rasterio.open(contributing_area_path) as area_src:
    if area_src.shape != elev_shape or area_src.transform != elev_transform or area_src.crs != elev_crs:
        raise ValueError("Contributing area and elevation rasters are not on the same grid.")

    contributing_area = area_src.read(1, masked=True).astype("float32")

slope = calculate_landlab_slope(elevation, dx, dy)
area = contributing_area.filled(np.nan)
area = np.where(area < 0, np.nan, area)

valid = (
    (~np.ma.getmaskarray(elevation))
    & (~np.ma.getmaskarray(contributing_area))
    & np.isfinite(slope)
    & np.isfinite(area)
)

slope_area_index = np.full(elev_shape, OUT_NODATA, dtype=np.float32)
slope_area_index[valid] = (area[valid] * (slope[valid] ** ALPHA)).astype(np.float32)

profile.update(dtype="float32", count=1, nodata=OUT_NODATA, compress="lzw")

with rasterio.open(out_tif, "w", **profile) as dst:
    dst.write(slope_area_index, 1)

plot_data = np.ma.masked_equal(slope_area_index, OUT_NODATA)
vmax = float(np.nanpercentile(plot_data.filled(np.nan), 98))

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(plot_data, cmap="magma", vmax=vmax)
ax.set_title(f"Slope-area index: area x slope^{ALPHA}")
ax.axis("off")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(out_png, dpi=200)
plt.show()

print("saved raster:", out_tif)
print("saved plot:", out_png)
print("elevation:", elevation_path)
print("contributing area:", contributing_area_path)
print("alpha:", ALPHA)
print("valid pixels:", int(valid.sum()))
print("index min:", float(np.nanmin(slope_area_index[valid])))
print("index mean:", float(np.nanmean(slope_area_index[valid])))
print("index p95:", float(np.nanpercentile(slope_area_index[valid], 95)))
print("index max:", float(np.nanmax(slope_area_index[valid])))
