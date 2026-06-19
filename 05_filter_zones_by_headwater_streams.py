from collections import deque
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box


# ---------------------------------------------------------------------
# Edit these paths/parameters, then run:
# python 05_filter_zones_by_headwater_streams.py
# ---------------------------------------------------------------------

input_zone_raster = Path("/mnt/c/Users/amehedi/Downloads/erosion_coherent_zones_min_object10.tif")
flowline_path = Path("/mnt/c/Users/amehedi/Downloads/nhdplus_hr_flowlines.gpkg")

output_raster = Path("/mnt/c/Users/amehedi/Downloads/erosion_zones_headwater_streams.tif")
selected_segments_output = Path("/mnt/c/Users/amehedi/Downloads/headwater_source_segments.gpkg")
selected_buffers_output = Path("/mnt/c/Users/amehedi/Downloads/headwater_source_segment_buffers.gpkg")

TARGET_VALUE = 1
BACKGROUND_VALUE = 0
OUT_NODATA = 255

# Headwater/upland stream rules.
MAX_STREAM_ORDER = 2
USE_STARTFLAG = True

# Optional extra filters. Set either USE_* value to False if too strict.
USE_DRAINAGE_AREA_FILTER = True
MAX_DRAINAGE_AREA_SQKM = 4.0

USE_CHANNEL_SLOPE_FILTER = True
MIN_CHANNEL_SLOPE = 0.15

# Distance around selected stream segments. Units are raster CRS units, usually meters.
BUFFER_DISTANCE = 60

# Keep whole connected erosion objects if any part touches the stream buffer.
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


def numeric_column(gdf, column_name):
    if column_name not in gdf.columns:
        return pd.Series(np.nan, index=gdf.index)
    return pd.to_numeric(gdf[column_name], errors="coerce")


def write_gpkg(gdf, path):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GPKG")


with rasterio.open(input_zone_raster) as src:
    zones = src.read(1)
    profile = src.profile.copy()
    nodata = src.nodata
    raster_crs = src.crs
    raster_transform = src.transform
    raster_shape = src.shape
    raster_bounds = src.bounds

if nodata is None:
    valid = np.ones(zones.shape, dtype=bool)
else:
    valid = zones != nodata

target = valid & (zones == TARGET_VALUE)
labels = label_objects(target, CONNECTIVITY)

flowlines = gpd.read_file(flowline_path)
if flowlines.crs != raster_crs:
    flowlines = flowlines.to_crs(raster_crs)

raster_box = box(*raster_bounds)
flowlines = flowlines[flowlines.geometry.intersects(raster_box)].copy()

stream_order = numeric_column(flowlines, "streamorde")
startflag = numeric_column(flowlines, "startflag")
drainage_area = numeric_column(flowlines, "totdasqkm")
channel_slope = numeric_column(flowlines, "slope")

source_like = stream_order <= MAX_STREAM_ORDER

if USE_STARTFLAG:
    source_like = source_like | (startflag == 1)

if USE_DRAINAGE_AREA_FILTER:
    source_like = source_like & (drainage_area <= MAX_DRAINAGE_AREA_SQKM)

if USE_CHANNEL_SLOPE_FILTER:
    source_like = source_like & (channel_slope >= MIN_CHANNEL_SLOPE)

selected_segments = flowlines[source_like].copy()
selected_buffers = selected_segments.copy()
selected_buffers["geometry"] = selected_buffers.geometry.buffer(BUFFER_DISTANCE)

buffer_mask = rasterize(
    [(geom, 1) for geom in selected_buffers.geometry if geom is not None and not geom.is_empty],
    out_shape=raster_shape,
    transform=raster_transform,
    fill=0,
    dtype="uint8",
).astype(bool)

touching_labels = np.unique(labels[buffer_mask & (labels > 0)])
keep_target = np.isin(labels, touching_labels)

filtered = np.full(zones.shape, BACKGROUND_VALUE, dtype=np.uint8)
filtered[keep_target] = TARGET_VALUE
filtered[~valid] = OUT_NODATA

profile.update(dtype="uint8", nodata=OUT_NODATA, compress="lzw")

output_raster.parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(output_raster, "w", **profile) as dst:
    dst.write(filtered, 1)

write_gpkg(selected_segments, selected_segments_output)
write_gpkg(selected_buffers, selected_buffers_output)

print("input zone raster:", input_zone_raster)
print("flowlines:", flowline_path)
print("output raster:", output_raster)
print("selected segments:", selected_segments_output)
print("selected buffers:", selected_buffers_output)
print("max stream order:", MAX_STREAM_ORDER)
print("max drainage area sqkm:", MAX_DRAINAGE_AREA_SQKM if USE_DRAINAGE_AREA_FILTER else "not used")
print("min channel slope:", MIN_CHANNEL_SLOPE if USE_CHANNEL_SLOPE_FILTER else "not used")
print("buffer distance:", BUFFER_DISTANCE)
print("flowline segments in raster bounds:", len(flowlines))
print("selected stream segments:", len(selected_segments))
print("zone objects before:", int(labels.max()))
print("zone objects kept:", len(touching_labels))
print("target pixels before:", int(target.sum()))
print("target pixels after:", int(keep_target.sum()))
