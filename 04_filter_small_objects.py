from collections import deque
from pathlib import Path

import numpy as np
import rasterio


# ---------------------------------------------------------------------
# Edit these paths/parameters, then run:
# python 04_filter_small_objects.py
# ---------------------------------------------------------------------

input_raster = Path("/mnt/c/Users/amehedi/Downloads/erosion_coherent_zones.tif")
output_raster = Path("/mnt/c/Users/amehedi/Downloads/erosion_coherent_zones_min_object10.tif")

TARGET_VALUE = 1
BACKGROUND_VALUE = 0
OUT_NODATA = 255

# Remove connected objects smaller than this many pixels.
MIN_OBJECT_PIXELS = 10

# Use 8 if diagonal touching should count as connected. Use 4 for stricter grouping.
CONNECTIVITY = 8


def neighbor_offsets(connectivity):
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        return [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    raise ValueError("CONNECTIVITY must be 4 or 8.")


def label_objects(binary, connectivity):
    rows, cols = binary.shape
    labels = np.zeros(binary.shape, dtype=np.int32)
    offsets = neighbor_offsets(connectivity)
    object_sizes = []
    current_label = 0

    for start_row, start_col in np.argwhere(binary):
        if labels[start_row, start_col] != 0:
            continue

        current_label += 1
        labels[start_row, start_col] = current_label
        queue = deque([(start_row, start_col)])
        size = 0

        while queue:
            row, col = queue.popleft()
            size += 1

            for dr, dc in offsets:
                nr = row + dr
                nc = col + dc

                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                if not binary[nr, nc] or labels[nr, nc] != 0:
                    continue

                labels[nr, nc] = current_label
                queue.append((nr, nc))

        object_sizes.append(size)

    return labels, object_sizes


with rasterio.open(input_raster) as src:
    data = src.read(1)
    profile = src.profile.copy()
    nodata = src.nodata

if nodata is None:
    valid = np.ones(data.shape, dtype=bool)
else:
    valid = data != nodata

target = valid & (data == TARGET_VALUE)
labels, object_sizes = label_objects(target, CONNECTIVITY)

keep_labels = {
    label
    for label, size in enumerate(object_sizes, start=1)
    if size >= MIN_OBJECT_PIXELS
}

keep_target = np.isin(labels, list(keep_labels))
remove_target = target & ~keep_target

cleaned = data.copy()
cleaned[remove_target] = BACKGROUND_VALUE
cleaned[~valid] = OUT_NODATA

profile.update(dtype="uint8", nodata=OUT_NODATA, compress="lzw")

output_raster.parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(output_raster, "w", **profile) as dst:
    dst.write(cleaned.astype(np.uint8), 1)

print("input raster:", input_raster)
print("output raster:", output_raster)
print("connectivity:", CONNECTIVITY)
print("minimum object pixels:", MIN_OBJECT_PIXELS)
print("objects before:", len(object_sizes))
print("objects kept:", len(keep_labels))
print("objects removed:", len(object_sizes) - len(keep_labels))
print("target pixels before:", int(target.sum()))
print("target pixels after:", int(keep_target.sum()))
print("target pixels removed:", int(remove_target.sum()))
