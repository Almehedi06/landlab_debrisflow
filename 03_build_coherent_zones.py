from pathlib import Path

import numpy as np
import rasterio


# ---------------------------------------------------------------------
# Edit these paths/parameters, then run:
# python 03_build_coherent_zones.py
# ---------------------------------------------------------------------

input_raster = Path("/mnt/c/Users/amehedi/Downloads/erosion_evidence.tif")
output_raster = Path("/mnt/c/Users/amehedi/Downloads/erosion_coherent_zones.tif")

TARGET_VALUE = 1
BACKGROUND_VALUE = 0
OUT_NODATA = 255

# This keeps the original raster resolution.
# Example: 5x5 on a 30 m raster checks a 150 m neighborhood.
WINDOW_SIZE = 10

# A cell becomes part of a coherent zone if at least this many target pixels
# occur inside its WINDOW_SIZE x WINDOW_SIZE neighborhood.
MIN_COUNT = 50


def focal_count(binary, window_size):
    pad = window_size // 2
    padded = np.pad(binary.astype(np.uint8), pad, mode="constant", constant_values=0)

    rows, cols = binary.shape
    counts = np.zeros(binary.shape, dtype=np.uint16)

    for r in range(window_size):
        for c in range(window_size):
            counts += padded[r : r + rows, c : c + cols]

    return counts


with rasterio.open(input_raster) as src:
    data = src.read(1)
    profile = src.profile.copy()
    nodata = src.nodata

if nodata is None:
    valid = np.ones(data.shape, dtype=bool)
else:
    valid = data != nodata

target = valid & (data == TARGET_VALUE)
target_count = focal_count(target, WINDOW_SIZE)

coherent_zone = np.full(data.shape, BACKGROUND_VALUE, dtype=np.uint8)
coherent_zone[target_count >= MIN_COUNT] = TARGET_VALUE
coherent_zone[~valid] = OUT_NODATA

profile.update(dtype="uint8", nodata=OUT_NODATA, compress="lzw")

output_raster.parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(output_raster, "w", **profile) as dst:
    dst.write(coherent_zone, 1)

print("input raster:", input_raster)
print("output raster:", output_raster)
print("window size:", WINDOW_SIZE)
print("minimum count:", MIN_COUNT)
print("target pixels before:", int(target.sum()))
print("coherent-zone pixels after:", int((coherent_zone == TARGET_VALUE).sum()))
