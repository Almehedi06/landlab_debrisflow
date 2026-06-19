from pathlib import Path

import numpy as np
import rasterio


# ---------------------------------------------------------------------
# Edit these paths/parameters, then run:
# python remove_isolated_pixels.py
# ---------------------------------------------------------------------

input_raster = Path("/mnt/c/Users/amehedi/Downloads/erosion_evidence.tif")
output_raster = Path("/mnt/c/Users/amehedi/Downloads/erosion_evidence_neighbors3.tif")

TARGET_VALUE = 1
BACKGROUND_VALUE = 0
MIN_NEIGHBORS = 4


def count_8_neighbors(binary):
    binary = binary.astype(np.uint8)
    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    rows, cols = binary.shape
    counts = np.zeros(binary.shape, dtype=np.uint8)

    for dr in range(3):
        for dc in range(3):
            if dr == 1 and dc == 1:
                continue
            counts += padded[dr : dr + rows, dc : dc + cols]

    return counts


with rasterio.open(input_raster) as src:
    data = src.read(1)
    profile = src.profile.copy()
    nodata = src.nodata

target_pixels = data == TARGET_VALUE

if nodata is None:
    valid = np.ones(data.shape, dtype=bool)
else:
    valid = data != nodata

neighbor_count = count_8_neighbors(target_pixels)
keep_target = target_pixels & (neighbor_count >= MIN_NEIGHBORS)
remove_target = target_pixels & ~keep_target

cleaned = data.copy()
cleaned[remove_target] = BACKGROUND_VALUE

output_raster.parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(output_raster, "w", **profile) as dst:
    dst.write(cleaned, 1)

print("input raster:", input_raster)
print("output raster:", output_raster)
print("target value:", TARGET_VALUE)
print("minimum neighbors:", MIN_NEIGHBORS)
print("valid pixels:", int(valid.sum()))
print("target pixels before:", int(target_pixels.sum()))
print("target pixels after:", int(keep_target.sum()))
print("target pixels removed:", int(remove_target.sum()))
