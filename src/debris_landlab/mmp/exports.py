from __future__ import annotations

from pathlib import Path

import numpy as np

from debris_landlab.mmp.config import ExportConfig


def export_raster_fields(grid, config: ExportConfig, *, scenario_dir: Path) -> list[Path]:
    """Export configured grid fields as GeoTIFF and ESRI ASCII rasters."""

    if not config.enabled:
        return []
    if config.export_dir is None:
        raise ValueError("Export is enabled but export.export_dir is not configured")

    import rasterio

    template_tif = config.template_tif or (scenario_dir / "topographic__elevation.tif")
    if not template_tif.exists():
        raise FileNotFoundError(f"Export template GeoTIFF not found: {template_tif}")

    config.export_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with rasterio.open(template_tif) as src:
        profile = src.profile.copy()
        transform = src.transform
        height = src.height
        width = src.width

    profile.update(dtype="float32", count=1, nodata=config.nodata)

    for field_name, at in config.fields:
        if at == "node" and field_name not in grid.at_node:
            continue
        if at == "cell" and field_name not in grid.at_cell:
            continue

        raster = field_to_node_raster(grid, field_name, at, nodata=config.nodata)
        tif_path = config.export_dir / f"{field_name}.tif"
        asc_path = config.export_dir / f"{field_name}.asc"

        with rasterio.open(tif_path, "w", **profile) as dst:
            dst.write(raster, 1)
        _write_ascii(asc_path, raster, width=width, height=height, transform=transform, nodata=config.nodata)

        written.extend([tif_path, asc_path])

    return written


def field_to_node_raster(grid, field_name: str, at: str, *, nodata: float) -> np.ndarray:
    if at == "node":
        values = grid.at_node[field_name].astype("float32")
        return np.flipud(values.reshape(grid.shape))

    values = np.full(grid.number_of_nodes, nodata, dtype="float32")
    values[grid.node_at_cell] = grid.at_cell[field_name].astype("float32")
    return np.flipud(values.reshape(grid.shape))


def _write_ascii(path: Path, raster: np.ndarray, *, width: int, height: int, transform, nodata: float) -> None:
    with path.open("w") as dst:
        dst.write(f"ncols         {width}\n")
        dst.write(f"nrows         {height}\n")
        dst.write(f"xllcorner     {transform.c}\n")
        dst.write(f"yllcorner     {transform.f + height * transform.e}\n")
        dst.write(f"cellsize      {transform.a}\n")
        dst.write(f"NODATA_value  {nodata}\n")
        np.savetxt(dst, raster, fmt="%.6f")
