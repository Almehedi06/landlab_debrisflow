from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from landlab import RasterModelGrid
from matplotlib.colors import BoundaryNorm, ListedColormap


# python 01_build_erosion_evidence.py
# ---------------------------------------------------------------------

base_dir = Path("/mnt/c/Users/amehedi/Downloads/ml_debris/output/thomas")

dem_diff_path = base_dir / "dem_diff.tif"
elevation_path = base_dir / "topographic__elevation.tif"

out_dir = Path("/mnt/c/Users/amehedi/Downloads")
out_tif = out_dir / "erosion_evidence.tif"
out_png = out_dir / "erosion_evidence.png"

DEM_DIFF_THRESHOLD = -0.2  # erosion evidence if dem_diff < -1.0 m
SLOPE_MIN = 0.0  # rise/run, not degrees
OUT_NODATA = 255


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


# Read DEM difference
with rasterio.open(dem_diff_path) as diff_src:
    dem_diff = diff_src.read(1, masked=True)
    profile = diff_src.profile.copy()
    diff_shape = diff_src.shape
    diff_transform = diff_src.transform
    diff_crs = diff_src.crs


# Read elevation and calculate slope
with rasterio.open(elevation_path) as elev_src:
    if elev_src.shape != diff_shape or elev_src.transform != diff_transform or elev_src.crs != diff_crs:
        raise ValueError("Elevation and dem_diff rasters are not on the same grid.")

    elevation = elev_src.read(1, masked=True).astype("float32")
    dx, dy = elev_src.res

slope = calculate_landlab_slope(elevation, dx, dy)


# Valid pixels
valid = (
    (~np.ma.getmaskarray(dem_diff))
    & (~np.ma.getmaskarray(elevation))
    & np.isfinite(slope)
)


# Pixel-level erosion evidence
erosion_evidence = (
    valid
    & (dem_diff.filled(np.nan) < DEM_DIFF_THRESHOLD)
    & (slope > SLOPE_MIN)
)


# Output raster
# 1 = erosion evidence
# 0 = no erosion evidence
# 255 = nodata
classes = np.zeros(diff_shape, dtype=np.uint8)
classes[erosion_evidence] = 1
classes[~valid] = OUT_NODATA

profile.update(dtype="uint8", count=1, nodata=OUT_NODATA, compress="lzw")

out_dir.mkdir(parents=True, exist_ok=True)
with rasterio.open(out_tif, "w", **profile) as dst:
    dst.write(classes, 1)


# Quick map
plot_data = np.ma.masked_equal(classes, OUT_NODATA)
cmap = ListedColormap(["#d9d9d9", "#d73027"])
norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(plot_data, cmap=cmap, norm=norm)
ax.set_title(
    f"Erosion evidence: dem_diff < {DEM_DIFF_THRESHOLD} m, "
    f"Landlab slope > {SLOPE_MIN}"
)
ax.axis("off")

cbar = fig.colorbar(im, ax=ax, ticks=[0, 1], fraction=0.046, pad=0.04)
cbar.ax.set_yticklabels(["0: no evidence", "1: erosion evidence"])

fig.tight_layout()
fig.savefig(out_png, dpi=200)
plt.show()


print("saved raster:", out_tif)
print("saved plot:", out_png)
print("erosion evidence pixels:", int(erosion_evidence.sum()))
