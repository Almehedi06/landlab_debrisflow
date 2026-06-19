from collections import deque
from pathlib import Path

import numpy as np
import rasterio


# ---------------------------------------------------------------------
# Edit these paths/parameters, then run:
# python 06_filter_zones_by_slope_area_index.py
# ---------------------------------------------------------------------

input_zone_raster = Path("/mnt/c/Users/amehedi/Downloads/erosion_zones_headwater_streams.tif")
slope_area_index_raster = Path("/mnt/c/Users/amehedi/Downloads/thomas/slope_area_index.tif")

output_raster = Path("/mnt/c/Users/amehedi/Downloads/erosion_zones_slope_area_filtered.tif")

TARGET_VALUE = 1
BACKGROUND_VALUE = 0
OUT_NODATA = 255
CONNECTIVITY = 8

# Object statistic used for filtering: "mean", "median", "p75", or "max".
OBJECT_INDEX_STAT = "mean"

# Erkan's C threshold in: specific contributing area * slope^alpha > C.
C_THRESHOLD = 10.0


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
    current_label = 0

    for start_row, start_col in np.argwhere(binary):
        if labels[start_row, start_col] != 0:
            continue

        current_label += 1
        labels[start_row, start_col] = current_label
        queue = deque([(start_row, start_col)])

        while queue:
            row, col = queue.popleft()

            for dr, dc in offsets:
                nr = row + dr
                nc = col + dc

                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                if not binary[nr, nc] or labels[nr, nc] != 0:
                    continue

                labels[nr, nc] = current_label
                queue.append((nr, nc))

    return labels


def object_stat(values, stat_name):
    if stat_name == "mean":
        return float(np.nanmean(values))
    if stat_name == "median":
        return float(np.nanmedian(values))
    if stat_name == "p75":
        return float(np.nanpercentile(values, 75))
    if stat_name == "max":
        return float(np.nanmax(values))
    raise ValueError('OBJECT_INDEX_STAT must be "mean", "median", "p75", or "max".')


with rasterio.open(input_zone_raster) as zone_src:
    zones = zone_src.read(1)
    profile = zone_src.profile.copy()
    zone_nodata = zone_src.nodata
    zone_shape = zone_src.shape
    zone_transform = zone_src.transform
    zone_crs = zone_src.crs

with rasterio.open(slope_area_index_raster) as index_src:
    if index_src.shape != zone_shape or index_src.transform != zone_transform or index_src.crs != zone_crs:
        raise ValueError("Slope-area index and zone rasters are not on the same grid.")

    slope_area_index = index_src.read(1).astype("float32")
    index_nodata = index_src.nodata

if zone_nodata is None:
    valid_zone = np.ones(zones.shape, dtype=bool)
else:
    valid_zone = zones != zone_nodata

if index_nodata is not None:
    valid_index = slope_area_index != index_nodata
else:
    valid_index = np.ones(slope_area_index.shape, dtype=bool)

valid_index &= np.isfinite(slope_area_index)
target = valid_zone & valid_index & (zones == TARGET_VALUE)

if not np.any(target):
    raise ValueError("No valid target pixels found in the input zone raster.")

labels = label_objects(target, CONNECTIVITY)
keep_labels = []
object_stats = []

for label in range(1, int(labels.max()) + 1):
    object_pixels = labels == label
    values = slope_area_index[object_pixels & valid_index]

    if values.size == 0:
        continue

    value = object_stat(values, OBJECT_INDEX_STAT)
    object_stats.append(value)

    if value >= C_THRESHOLD:
        keep_labels.append(label)

keep_target = np.isin(labels, keep_labels)

filtered = np.full(zones.shape, BACKGROUND_VALUE, dtype=np.uint8)
filtered[keep_target] = TARGET_VALUE
filtered[~valid_zone] = OUT_NODATA

profile.update(dtype="uint8", nodata=OUT_NODATA, compress="lzw")

output_raster.parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(output_raster, "w", **profile) as dst:
    dst.write(filtered, 1)

print("input zone raster:", input_zone_raster)
print("slope-area index raster:", slope_area_index_raster)
print("output raster:", output_raster)
print("object index statistic:", OBJECT_INDEX_STAT)
print("C threshold:", C_THRESHOLD)
print("zone objects before:", int(labels.max()))
print("zone objects kept:", len(keep_labels))
print("zone objects removed:", int(labels.max()) - len(keep_labels))
print("target pixels before:", int(target.sum()))
print("target pixels after:", int(keep_target.sum()))
