from __future__ import annotations

import argparse
import glob
from pathlib import Path
import shutil

import geopandas as gpd
import numpy as np
import rasterio
import yaml

try:
    import zarr
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    zarr = None


def _load_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve_output_dir(config_path: str | None, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    cfg = _load_config(config_path)
    out = (
        cfg.get("outputs", {}).get("root_dir")
        or cfg.get("paths", {}).get("output_dir")
        or cfg.get("paths", {}).get("output_root")
    )
    if not out:
        raise ValueError("Output directory not provided and not found in config.")
    return Path(out)


def _normalize_crs(crs: str | int | None) -> str | None:
    if crs is None:
        return None
    if isinstance(crs, int):
        return f"EPSG:{crs}"
    return str(crs)


def _resolve_crs(config_path: str | None, aoi_path: str | None, crs_arg: str | None) -> str | None:
    if crs_arg:
        return _normalize_crs(crs_arg)

    if config_path:
        cfg = _load_config(config_path)
        cfg_crs = cfg.get("grid", {}).get("crs")
        if cfg_crs:
            return _normalize_crs(cfg_crs)

    if not aoi_path and config_path:
        aoi_path = cfg.get("aoi", {}).get("aoi")

    if not aoi_path:
        return None

    gdf = gpd.read_file(aoi_path)
    if gdf.crs is None:
        return None
    epsg = gdf.crs.to_epsg()
    if epsg is not None:
        return f"EPSG:{epsg}"
    return gdf.crs.to_wkt()


def _asc_to_tif(asc_path: Path, tif_path: Path, overwrite: bool, crs: str | None) -> bool:
    if tif_path.exists() and not overwrite:
        return False

    with rasterio.open(asc_path) as src:
        data = src.read(1)
        meta = src.meta.copy()

    meta.update(driver="GTiff", count=1)
    if meta.get("nodata") is None:
        if np.any(data == -9999):
            meta["nodata"] = -9999.0
        else:
            finite = data[np.isfinite(data)]
            if finite.size and np.max(np.abs(finite)) > 1e20:
                # Common float sentinel seen in some ASC exports.
                meta["nodata"] = float(np.max(finite))
    if crs:
        meta.update(crs=crs)
    with rasterio.open(tif_path, "w", **meta) as dst:
        dst.write(data, 1)
    return True


def _asc_to_zarr(asc_path: Path, root, crs: str | None) -> None:
    with rasterio.open(asc_path) as src:
        data = src.read(1)
        meta = src.meta.copy()

    dataset = root.create_dataset(
        asc_path.stem,
        data=data,
        shape=data.shape,
        chunks=(min(512, data.shape[0]), min(512, data.shape[1])),
        overwrite=True,
    )
    transform = meta.get("transform")
    dataset.attrs.update(
        {
            "crs": crs or _normalize_crs(meta.get("crs")),
            "nodata": meta.get("nodata"),
            "dtype": str(data.dtype),
            "height": int(meta["height"]),
            "width": int(meta["width"]),
            "transform": list(transform) if transform is not None else None,
        }
    )


def export_ascii_dir_to_tifs(
    out_dir: Path,
    pattern: str = "*.asc",
    overwrite: bool = True,
    crs: str | None = None,
) -> tuple[int, int]:
    asc_files = [Path(p) for p in glob.glob(str(out_dir / pattern))]
    if not asc_files:
        return 0, 0

    exported = 0
    skipped = 0
    for asc_path in asc_files:
        tif_path = asc_path.with_suffix(".tif")
        if _asc_to_tif(asc_path, tif_path, overwrite, crs):
            exported += 1
        else:
            skipped += 1
    return exported, skipped


def export_ascii_dir_to_zarr(
    out_dir: Path,
    zarr_store: Path,
    pattern: str = "*.asc",
    overwrite: bool = True,
    crs: str | None = None,
) -> int:
    if zarr is None:
        raise ModuleNotFoundError(
            "zarr is not installed. Add zarr to the environment to export Zarr stores."
        )

    asc_files = [Path(p) for p in glob.glob(str(out_dir / pattern))]
    if not asc_files:
        return 0

    if zarr_store.exists():
        if not overwrite:
            return 0
        if zarr_store.is_dir():
            shutil.rmtree(zarr_store)
        else:
            zarr_store.unlink()

    root = zarr.open_group(str(zarr_store), mode="w")
    root.attrs.update(
        {
            "format": "debris-landlab-raster-zarr-v1",
            "crs": crs,
            "layers": [asc_path.stem for asc_path in asc_files],
        }
    )

    for asc_path in asc_files:
        _asc_to_zarr(asc_path, root, crs)

    return len(asc_files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ESRI ASCII grids in output_dir to GeoTIFF and optional Zarr."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config YAML; uses cfg['paths']['output_dir'] if set.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory containing .asc files (overrides config).",
    )
    parser.add_argument(
        "--pattern",
        default="*.asc",
        help="Glob pattern for ASCII files (default: *.asc).",
    )
    parser.add_argument(
        "--aoi",
        default=None,
        help="AOI shapefile used to set CRS for exported GeoTIFFs.",
    )
    parser.add_argument(
        "--crs",
        default=None,
        help="CRS override (e.g., EPSG:32611).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--zarr-store",
        default=None,
        help="Optional Zarr store path. If set, exports all matching ASC files to one store.",
    )
    args = parser.parse_args()

    out_dir = _resolve_output_dir(args.config, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crs = _resolve_crs(args.config, args.aoi, args.crs)

    exported, skipped = export_ascii_dir_to_tifs(
        out_dir,
        pattern=args.pattern,
        overwrite=args.overwrite,
        crs=crs,
    )
    zarr_count = 0
    if args.zarr_store:
        zarr_count = export_ascii_dir_to_zarr(
            out_dir,
            zarr_store=Path(args.zarr_store),
            pattern=args.pattern,
            overwrite=args.overwrite,
            crs=crs,
        )
    if exported == 0 and skipped == 0:
        print(f"No ASCII files found in {out_dir} matching {args.pattern}")
        return
    print(f"Exported {exported} GeoTIFF(s). Skipped {skipped}.")
    if args.zarr_store:
        print(f"Exported {zarr_count} layer(s) to Zarr: {args.zarr_store}")


if __name__ == "__main__":
    main()
