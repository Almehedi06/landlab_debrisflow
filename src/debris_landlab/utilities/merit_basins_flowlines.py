from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import pandas as pd
from shapely import union_all


MERIT_BASINS_GOOGLE_DRIVE_URL = (
    "https://drive.google.com/drive/folders/1J8vyqCnSdquY1cRI1PPsXzMBLBXKzzoW?usp=share_link"
)
MERIT_BASINS_GOOGLE_DRIVE_FOLDER_ID = "1J8vyqCnSdquY1cRI1PPsXzMBLBXKzzoW"
DEFAULT_OUTDIR = "/mnt/c/Users/amehedi/Downloads"
DEFAULT_BASENAME = "merit_basins_flowlines"
DEFAULT_SOURCE_SUBDIR = "MERIT-Hydro_v07_Basins_v01_bugfix1"
DEFAULT_SOURCE_DIR = f"{DEFAULT_OUTDIR}/{DEFAULT_SOURCE_SUBDIR}"
SHAPEFILE_EXTENSIONS = {".cpg", ".dbf", ".prj", ".shp", ".shx"}


def _read_aoi(aoi_path: str | Path) -> tuple[gpd.GeoDataFrame, Any]:
    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise ValueError(f"AOI has no features: {aoi_path}")
    if aoi.crs is None:
        raise ValueError(f"AOI has no CRS. Define one before running: {aoi_path}")

    aoi = aoi.to_crs("EPSG:4326")
    return aoi, union_all(aoi.geometry.to_numpy())


def _source_dir(outdir: Path, source_dir: str | Path | None) -> Path:
    if source_dir is not None:
        return Path(source_dir)
    return outdir / DEFAULT_SOURCE_SUBDIR


def _infer_pfaf_codes(aoi_geom: Any, pfaf_level: int | None) -> list[int]:
    centroid = aoi_geom.centroid
    lon = float(centroid.x)
    lat = float(centroid.y)

    if -170 <= lon <= -45 and 5 <= lat <= 85:
        if pfaf_level == 2:
            if lon <= -100 and lat >= 35:
                return [78]  # Western North America; keeps Pioneer/Stehekin downloads small.
            if lon <= -100:
                return [77]
            if lat >= 45:
                return [71]
            if lon <= -85:
                return [74]
            return [72]
        return [7]  # North America
    if -95 <= lon <= -25 and -60 <= lat <= 15:
        return [6]  # South America
    if -75 <= lon <= -5 and 58 <= lat <= 85:
        return [9]  # Greenland
    if -25 <= lon <= 60 and -40 <= lat <= 40:
        return [1]  # Africa
    if -25 <= lon <= 70 and 35 <= lat <= 75:
        return [2]  # Europe
    if 25 <= lon <= 180 and 45 <= lat <= 85:
        return [3]  # North Asia
    if 40 <= lon <= 150 and -15 <= lat <= 45:
        return [4]  # South Asia
    if 90 <= lon <= 180 and -55 <= lat <= 25:
        return [5]  # Oceania and Southeast Asian islands
    if lat >= 66:
        return [8]  # Arctic

    return list(range(1, 10))


def _normalize_pfaf_codes(
    pfaf_codes: str | int | Sequence[int] | None,
    aoi_geom: Any,
    pfaf_level: int | None,
) -> list[int] | None:
    if pfaf_codes is None or pfaf_codes == "auto":
        return _infer_pfaf_codes(aoi_geom, pfaf_level)
    if pfaf_codes == "all":
        return None
    if isinstance(pfaf_codes, int):
        return [pfaf_codes]
    if isinstance(pfaf_codes, str):
        return [int(part.strip()) for part in pfaf_codes.split(",") if part.strip()]
    return [int(code) for code in pfaf_codes]


def _feature_code_from_path(path: str | Path) -> str | None:
    match = re.search(r"riv_pfaf_(\d+)", Path(str(path)).name)
    return match.group(1) if match else None


def _matches_pfaf_codes(path: str | Path, pfaf_level: int | None, pfaf_codes: Sequence[int] | None) -> bool:
    if pfaf_codes is None:
        return True

    feature_code = _feature_code_from_path(path)
    if feature_code is None:
        return False

    for code in pfaf_codes:
        code_text = str(code)
        if pfaf_level == 1 and feature_code == code_text:
            return True
        if pfaf_level != 1 and feature_code.startswith(code_text):
            return True
    return False


def _zip_flowline_layers(zip_path: Path) -> list[str]:
    layers: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            path = Path(name)
            if path.name.startswith("riv_pfaf_") and path.suffix.lower() == ".shp":
                layers.append(f"zip://{zip_path}!{name}")
    return layers


def _find_flowline_layers(
    source_dir: str | Path,
    pfaf_level: int | None,
    pfaf_codes: Sequence[int] | None,
) -> list[str | Path]:
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(
            f"MERIT-Basins source directory not found: {source_dir}\n"
            "Download/extract MERIT-Basins first, then pass source_dir=...\n"
            f"Source folder: {MERIT_BASINS_GOOGLE_DRIVE_URL}"
        )

    level_part = f"pfaf_level_{pfaf_level:02d}" if pfaf_level is not None else "pfaf_level_*"

    layers: list[str | Path] = [
        path
        for path in sorted(source_dir.glob(f"**/{level_part}/riv_pfaf_*_MERIT_Hydro_v07_Basins_v01*.shp"))
        if _matches_pfaf_codes(path, pfaf_level, pfaf_codes)
    ]
    if layers:
        return layers

    for zip_path in sorted(source_dir.glob("**/*.zip")):
        if pfaf_level is not None and f"pfaf_level_{pfaf_level:02d}" not in zip_path.as_posix():
            continue
        if not _matches_pfaf_codes(zip_path, pfaf_level, pfaf_codes):
            continue
        layers.extend(_zip_flowline_layers(zip_path))

    if not layers:
        raise FileNotFoundError(
            f"No MERIT-Basins river shapefiles found under: {source_dir}\n"
            "Expected files like "
            "pfaf_level_01/riv_pfaf_07_MERIT_Hydro_v07_Basins_v01.shp "
            "or zipped archives containing those shapefiles."
        )

    return layers


def _list_google_drive_items(output_dir: Path) -> list[Any]:
    try:
        import gdown
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gdown is required to download MERIT-Basins source files. "
            "Install it with: python -m pip install gdown"
        ) from exc

    items = gdown.download_folder(
        id=MERIT_BASINS_GOOGLE_DRIVE_FOLDER_ID,
        output=str(output_dir),
        quiet=True,
        use_cookies=False,
        skip_download=True,
    )
    if not items:
        raise RuntimeError("Could not list MERIT-Basins Google Drive files.")
    return items


def _wanted_drive_item(path: str, pfaf_level: int | None, pfaf_codes: Sequence[int] | None) -> bool:
    path_obj = Path(path)
    if not path_obj.name.startswith("riv_pfaf_"):
        return False

    suffix = path_obj.suffix.lower()
    if suffix not in SHAPEFILE_EXTENSIONS and suffix != ".zip":
        return False

    if pfaf_level is not None and f"pfaf_level_{pfaf_level:02d}" not in path_obj.parts:
        return False

    return _matches_pfaf_codes(path_obj, pfaf_level, pfaf_codes)


def _download_drive_file(item: Any, output_path: Path, *, force_download: bool) -> None:
    if output_path.exists() and not force_download:
        return

    try:
        import gdown
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gdown is required to download MERIT-Basins source files. "
            "Install it with: python -m pip install gdown"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {item.path}")
    gdown.download(
        id=item.id,
        output=str(output_path),
        quiet=False,
        use_cookies=False,
        resume=True,
    )


def _ensure_source_files(
    source_dir: Path,
    *,
    pfaf_level: int | None,
    pfaf_codes: Sequence[int] | None,
    download: bool,
    force_download: bool,
) -> None:
    if source_dir.exists():
        try:
            existing = _find_flowline_layers(source_dir, pfaf_level, pfaf_codes)
        except FileNotFoundError:
            existing = []
        if existing and not force_download:
            return

    if not download:
        return

    print("MERIT-Basins source flowlines not found locally. Downloading required river files...")
    print(f"Source cache: {source_dir}")
    items = _list_google_drive_items(source_dir)
    wanted = [item for item in items if _wanted_drive_item(item.path, pfaf_level, pfaf_codes)]
    if not wanted:
        raise RuntimeError("Could not identify matching MERIT-Basins river files to download.")

    manifest_rows = []
    for item in wanted:
        output_path = source_dir / item.path
        _download_drive_file(item, output_path, force_download=force_download)
        manifest_rows.append({"id": item.id, "path": item.path, "local_path": str(output_path)})

    pd.DataFrame(manifest_rows).to_csv(source_dir / "download_manifest.csv", index=False)


def _read_intersecting_flowlines(layer: str | Path, bounds: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    try:
        data = gpd.read_file(layer, bbox=bounds)
    except Exception:
        data = gpd.read_file(layer)

    if data.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if data.crs is None:
        data = data.set_crs("EPSG:4326")
    return data.to_crs("EPSG:4326")


def _clip_flowlines(
    layers: list[str | Path],
    *,
    aoi_geom: Any,
    min_uparea_km2: float | None,
) -> gpd.GeoDataFrame:
    bounds = aoi_geom.bounds
    clip_aoi = gpd.GeoDataFrame(geometry=[aoi_geom], crs="EPSG:4326")
    clipped_frames: list[gpd.GeoDataFrame] = []

    for layer in layers:
        flowlines = _read_intersecting_flowlines(layer, bounds)
        if flowlines.empty:
            continue

        if min_uparea_km2 is not None and "uparea" in flowlines.columns:
            flowlines = flowlines[pd.to_numeric(flowlines["uparea"], errors="coerce") >= min_uparea_km2]
            if flowlines.empty:
                continue

        clipped = gpd.clip(flowlines, clip_aoi)
        if not clipped.empty:
            clipped_frames.append(clipped)

    if not clipped_frames:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    merged = gpd.GeoDataFrame(pd.concat(clipped_frames, ignore_index=True), crs="EPSG:4326")
    if "COMID" in merged.columns:
        merged = merged.drop_duplicates("COMID")
    return merged.reset_index(drop=True)


def _add_clipped_length(flowlines: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if flowlines.empty:
        flowlines["clipped_length_km"] = pd.Series(dtype="float64")
        return flowlines

    projected_crs = aoi.estimate_utm_crs()
    if projected_crs is None:
        projected_crs = "EPSG:3857"

    projected = flowlines.to_crs(projected_crs)
    flowlines["clipped_length_km"] = projected.geometry.length / 1000.0
    return flowlines


def _save_outputs(flowlines: gpd.GeoDataFrame, *, outdir: Path, basename: str) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    gpkg_path = outdir / f"{basename}.gpkg"
    geojson_path = outdir / f"{basename}.geojson"
    csv_path = outdir / f"{basename}.csv"

    flowlines.to_file(gpkg_path, layer="flowlines", driver="GPKG")
    flowlines.to_file(geojson_path, driver="GeoJSON")

    attrs = flowlines.drop(columns="geometry").copy()
    attrs.to_csv(csv_path, index=False)

    return {"gpkg": gpkg_path, "geojson": geojson_path, "csv": csv_path}


def download_merit_basins_flowlines(
    aoi_path: str | Path,
    source_dir: str | Path | None = None,
    outdir: str | Path | None = None,
    basename: str = DEFAULT_BASENAME,
    pfaf_level: int | None = 2,
    pfaf_codes: str | int | Sequence[int] | None = "auto",
    min_uparea_km2: float | None = None,
    download: bool = True,
    force_download: bool = False,
) -> gpd.GeoDataFrame:
    """
    Download/cache MERIT-Basins river files, then clip river reaches to an AOI.

    MERIT-Basins is the vector river-reach product derived from MERIT-Hydro.
    The source shapefile components are downloaded into ``source_dir``. If
    ``source_dir`` is omitted, they are cached under the output directory.
    """
    outdir = Path(DEFAULT_OUTDIR if outdir is None else outdir)
    source_dir_path = _source_dir(outdir, source_dir)
    aoi, aoi_geom = _read_aoi(aoi_path)
    selected_pfaf_codes = _normalize_pfaf_codes(pfaf_codes, aoi_geom, pfaf_level)

    _ensure_source_files(
        source_dir_path,
        pfaf_level=pfaf_level,
        pfaf_codes=selected_pfaf_codes,
        download=download,
        force_download=force_download,
    )
    layers = _find_flowline_layers(source_dir_path, pfaf_level, selected_pfaf_codes)

    print("Reading MERIT-Basins flowlines...")
    print(f"Source directory: {source_dir_path}")
    print(f"Pfaf codes: {'all' if selected_pfaf_codes is None else selected_pfaf_codes}")
    print(f"Candidate layers: {len(layers)}")
    print(f"Output directory: {outdir}")

    flowlines = _clip_flowlines(
        layers,
        aoi_geom=aoi_geom,
        min_uparea_km2=min_uparea_km2,
    )

    if flowlines.empty:
        print("No MERIT-Basins flowlines found after clipping to the AOI.")
        return flowlines

    flowlines = _add_clipped_length(flowlines, aoi)
    outputs = _save_outputs(flowlines, outdir=outdir, basename=basename)

    print("Done.")
    print(f"Flowlines: {len(flowlines)}")
    print(f"Saved: {outputs['gpkg']}")
    print(f"Saved: {outputs['geojson']}")
    print(f"Saved: {outputs['csv']}")

    return flowlines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clip MERIT-Basins river flowline shapefiles to an AOI."
    )
    parser.add_argument("--aoi-path", required=True, help="AOI vector path, for example a shapefile or GeoPackage.")
    parser.add_argument(
        "--source-dir",
        default=None,
        help=f"MERIT-Basins source cache directory. Default: <outdir>/{DEFAULT_SOURCE_SUBDIR}",
    )
    parser.add_argument("--outdir", default=None, help=f"Output directory. Default: {DEFAULT_OUTDIR}")
    parser.add_argument("--basename", default=DEFAULT_BASENAME, help="Base output filename.")
    parser.add_argument(
        "--pfaf-level",
        type=int,
        default=2,
        choices=[1, 2],
        help="Use MERIT-Basins pfaf_level_01 or pfaf_level_02 flowlines.",
    )
    parser.add_argument(
        "--all-pfaf-levels",
        action="store_true",
        help="Search all pfaf_level folders instead of one level.",
    )
    parser.add_argument(
        "--pfaf-codes",
        default="auto",
        help='Pfaf code(s) to download/read, for example "7", "71,72", "all", or "auto".',
    )
    parser.add_argument(
        "--min-uparea-km2",
        type=float,
        default=None,
        help="Optional minimum upstream drainage area filter in km2.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download missing MERIT-Basins source files.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload matching MERIT-Basins source files even if cached.",
    )
    args = parser.parse_args()

    download_merit_basins_flowlines(
        aoi_path=args.aoi_path,
        source_dir=args.source_dir,
        outdir=args.outdir,
        basename=args.basename,
        pfaf_level=None if args.all_pfaf_levels else args.pfaf_level,
        pfaf_codes=args.pfaf_codes,
        min_uparea_km2=args.min_uparea_km2,
        download=not args.no_download,
        force_download=args.force_download,
    )


__all__ = [
    "DEFAULT_BASENAME",
    "DEFAULT_OUTDIR",
    "DEFAULT_SOURCE_DIR",
    "DEFAULT_SOURCE_SUBDIR",
    "MERIT_BASINS_GOOGLE_DRIVE_FOLDER_ID",
    "MERIT_BASINS_GOOGLE_DRIVE_URL",
    "download_merit_basins_flowlines",
    "main",
]


if __name__ == "__main__":
    main()
