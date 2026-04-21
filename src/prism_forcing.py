from __future__ import annotations

import argparse
import json
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import fiona
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.transform import array_bounds
from rasterio.warp import reproject, transform_geom


PRISM_BASE_URL = "https://services.nacse.org/prism/data/get"
SUPPORTED_ELEMENTS = ("ppt", "tmin", "tmax")
DEFAULT_RESAMPLING = {
    "ppt": "nearest",
    "tmin": "bilinear",
    "tmax": "bilinear",
}
ASCII_PREFIX = {
    "ppt": "precip",
    "tmin": "tmin",
    "tmax": "tmax",
}
CSV_COLUMN = {
    "ppt": "precip_mm",
    "tmin": "tmin_c",
    "tmax": "tmax_c",
}


@dataclass
class TemplateGrid:
    crs: CRS
    transform: rasterio.Affine
    width: int
    height: int
    nodata: float
    dem_header_lines: list[str] | None


@dataclass
class PrismProcessingResult:
    forcing_df: pd.DataFrame
    output_dir: Path
    raw_dir: Path
    aligned_tif_dir: Path
    asc_dir: Path
    manifest_path: Path
    csv_path: Path


def _parse_date(value: str | date | datetime | pd.Timestamp) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def _iter_daily_dates(start_date: str | date | datetime, end_date: str | date | datetime) -> Iterable[date]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end < start:
        raise ValueError(f"end_date must be on or after start_date: {start} > {end}")

    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _resampling_enum(method: str) -> Resampling:
    lookup = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
        "mode": Resampling.mode,
    }
    key = (method or "nearest").lower()
    if key not in lookup:
        raise ValueError(f"Unsupported resampling method: {method}")
    return lookup[key]


def _prism_grid_url(region: str, resolution: str, element: str, prism_date: str) -> str:
    return f"{PRISM_BASE_URL}/{region}/{resolution}/{element}/{prism_date}"


def _release_date_url(region: str, resolution: str, element: str, prism_date: str) -> str:
    return f"{PRISM_BASE_URL}/releaseDate/{region}/{resolution}/{element}/{prism_date}?json=true"


def _release_sidecar_path(zip_path: Path) -> Path:
    return zip_path.with_name(f"{zip_path.stem}.release.json")


def _mean_release_signature(payload: dict[str, Any] | None) -> tuple[str | None, str | None]:
    if not payload:
        return (None, None)

    release_keys = ("releaseDate", "release_date", "Release date", "release-date")
    grid_count_keys = ("gridCount", "grid_count", "Grid count", "grid-count")

    release_date = next((str(payload[k]) for k in release_keys if k in payload and payload[k] is not None), None)
    grid_count = next((str(payload[k]) for k in grid_count_keys if k in payload and payload[k] is not None), None)
    return release_date, grid_count


def fetch_release_metadata(
    prism_date: str,
    *,
    element: str,
    region: str = "us",
    resolution: str = "800m",
    session: requests.Session | None = None,
    timeout: int = 60,
) -> dict[str, Any] | None:
    http = session or requests.Session()
    response = http.get(
        _release_date_url(region=region, resolution=resolution, element=element, prism_date=prism_date),
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, list):
        if not payload:
            return None
        first = payload[0]
        return first if isinstance(first, dict) else None
    if isinstance(payload, dict):
        return payload
    return None


def _download_prism_zip(
    prism_date: str,
    *,
    element: str,
    region: str,
    resolution: str,
    out_dir: Path,
    session: requests.Session,
    timeout: int,
    force_download: bool,
    check_release_dates: bool,
    sleep_seconds: float,
) -> tuple[Path, dict[str, Any] | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"prism_{element}_{region}_{resolution}_{prism_date}.zip"
    sidecar_path = _release_sidecar_path(zip_path)

    cached_release: dict[str, Any] | None = None
    if sidecar_path.exists():
        cached_release = json.loads(sidecar_path.read_text())

    if zip_path.exists() and not force_download:
        if not check_release_dates:
            return zip_path, cached_release

        remote_release = fetch_release_metadata(
            prism_date,
            element=element,
            region=region,
            resolution=resolution,
            session=session,
            timeout=timeout,
        )
        if _mean_release_signature(cached_release) == _mean_release_signature(remote_release):
            return zip_path, remote_release

    response = session.get(
        _prism_grid_url(region=region, resolution=resolution, element=element, prism_date=prism_date),
        timeout=timeout,
    )
    response.raise_for_status()
    zip_path.write_bytes(response.content)

    release_payload = cached_release
    try:
        release_payload = fetch_release_metadata(
            prism_date,
            element=element,
            region=region,
            resolution=resolution,
            session=session,
            timeout=timeout,
        )
    except requests.RequestException:
        pass

    if release_payload is not None:
        sidecar_path.write_text(json.dumps(release_payload, indent=2, sort_keys=True))

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    return zip_path, release_payload


def _extract_tif(zip_path: Path, out_dir: Path, *, force: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        tif_members = [name for name in archive.namelist() if name.lower().endswith(".tif")]
        if not tif_members:
            raise FileNotFoundError(f"No GeoTIFF found in {zip_path}")

        member = tif_members[0]
        tif_path = out_dir / Path(member).name
        if tif_path.exists() and not force:
            return tif_path

        with archive.open(member) as src, tif_path.open("wb") as dst:
            dst.write(src.read())

    return tif_path


def _read_ascii_header_lines(ascii_path: Path) -> list[str]:
    with ascii_path.open("r") as src:
        return [next(src).rstrip("\n") for _ in range(6)]


def _build_template_grid(dem_path: Path, dem_crs: str | int | None) -> TemplateGrid:
    with rasterio.open(dem_path) as src:
        crs = src.crs
        if crs is None and dem_crs is not None:
            crs = CRS.from_user_input(dem_crs)
        if crs is None:
            raise ValueError(
                "DEM has no CRS. Pass dem_crs explicitly, for example dem_crs='EPSG:32610'."
            )

        dem_header_lines = None
        if dem_path.suffix.lower() == ".asc":
            dem_header_lines = _read_ascii_header_lines(dem_path)

        nodata = src.nodata if src.nodata is not None else -9999.0
        return TemplateGrid(
            crs=crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
            nodata=float(nodata),
            dem_header_lines=dem_header_lines,
        )


def _read_aoi_shapes(aoi_path: Path, raster_crs: CRS | None) -> list[dict[str, Any]]:
    with fiona.open(aoi_path, "r") as src:
        shapes = [feature["geometry"] for feature in src]
        if not shapes:
            raise ValueError(f"No AOI features found in {aoi_path}")

        src_crs = src.crs_wkt or src.crs
        if not src_crs or raster_crs is None:
            return shapes

        src_crs_obj = CRS.from_user_input(src_crs)
        if src_crs_obj == raster_crs:
            return shapes

        return [transform_geom(src_crs_obj, raster_crs, geom) for geom in shapes]


def _clip_and_align_raster(
    src_tif: Path,
    *,
    aoi_path: Path,
    out_tif: Path,
    template: TemplateGrid,
    resampling_method: str,
) -> Path:
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_tif) as src:
        shapes = _read_aoi_shapes(aoi_path, src.crs)
        clipped, clipped_transform = mask(src, shapes, crop=True)
        src_nodata = src.nodata
        dst_nodata = template.nodata

        dst_data = np.full(
            (src.count, template.height, template.width),
            dst_nodata,
            dtype=np.float32,
        )

        reproject(
            source=clipped.astype(np.float32),
            destination=dst_data,
            src_transform=clipped_transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=template.transform,
            dst_crs=template.crs,
            dst_nodata=dst_nodata,
            resampling=_resampling_enum(resampling_method),
        )

        meta = src.meta.copy()
        meta.update(
            {
                "driver": "GTiff",
                "dtype": "float32",
                "count": src.count,
                "crs": template.crs,
                "transform": template.transform,
                "width": template.width,
                "height": template.height,
                "nodata": dst_nodata,
                "compress": "lzw",
            }
        )

    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(dst_data)

    return out_tif


def _compute_header_lines_from_template(template: TemplateGrid) -> list[str]:
    west, south, _, _ = array_bounds(template.height, template.width, template.transform)
    cellsize = abs(template.transform.a)
    return [
        f"ncols         {template.width}",
        f"nrows         {template.height}",
        f"xllcorner     {west}",
        f"yllcorner     {south}",
        f"cellsize      {cellsize}",
        f"NODATA_value  {template.nodata}",
    ]


def _write_ascii_like_dem(
    tif_path: Path,
    *,
    out_path: Path,
    template: TemplateGrid,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_lines = template.dem_header_lines or _compute_header_lines_from_template(template)

    with rasterio.open(tif_path) as src:
        array = src.read(1)
        nodata = src.nodata if src.nodata is not None else template.nodata

    array = np.where(np.isfinite(array), array, nodata)

    with out_path.open("w") as dst:
        dst.write("\n".join(header_lines) + "\n")
        for row in array:
            dst.write(" ".join(map(str, row)) + "\n")

    return out_path


def _masked_mean(raster_path: Path, fallback: float) -> float:
    with rasterio.open(raster_path) as src:
        masked = src.read(1, masked=True)
    if masked.count() == 0:
        return float(fallback)
    return float(masked.mean())


def _manifest_payload(
    *,
    start_date: date,
    end_date: date,
    dem_path: Path,
    aoi_path: Path,
    region: str,
    resolution: str,
    template: TemplateGrid,
    result_df: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "dem_path": str(dem_path),
        "aoi_path": str(aoi_path),
        "region": region,
        "resolution": resolution,
        "template": {
            "crs": str(template.crs),
            "width": template.width,
            "height": template.height,
            "transform": list(template.transform),
            "nodata": template.nodata,
        },
        "records": result_df.to_dict(orient="records"),
    }


def build_prism_forcings(
    *,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    dem_path: str | Path,
    aoi_path: str | Path,
    output_dir: str | Path,
    dem_crs: str | int | None = None,
    region: str = "us",
    resolution: str = "800m",
    force_download: bool = False,
    check_release_dates: bool = True,
    timeout: int = 120,
    sleep_seconds: float = 2.0,
) -> PrismProcessingResult:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    dem_path = Path(dem_path)
    aoi_path = Path(aoi_path)
    output_dir = Path(output_dir)

    raw_dir = output_dir / "raw"
    extracted_tif_dir = output_dir / "extracted_tif"
    aligned_tif_dir = output_dir / "aligned_tif"
    asc_dir = output_dir / "asc"
    manifest_path = output_dir / "prism_manifest.json"
    csv_path = output_dir / "forcing_daily_prism.csv"

    template = _build_template_grid(dem_path, dem_crs)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    with requests.Session() as session:
        for current_day in _iter_daily_dates(start, end):
            prism_date = current_day.strftime("%Y%m%d")
            row: dict[str, Any] = {"datetime": pd.Timestamp(current_day)}

            for element in SUPPORTED_ELEMENTS:
                zip_path, release_meta = _download_prism_zip(
                    prism_date,
                    element=element,
                    region=region,
                    resolution=resolution,
                    out_dir=raw_dir / element,
                    session=session,
                    timeout=timeout,
                    force_download=force_download,
                    check_release_dates=check_release_dates,
                    sleep_seconds=sleep_seconds,
                )

                raw_tif = _extract_tif(zip_path, extracted_tif_dir / element)
                aligned_tif = aligned_tif_dir / element / f"{ASCII_PREFIX[element]}_{prism_date}.tif"
                _clip_and_align_raster(
                    raw_tif,
                    aoi_path=aoi_path,
                    out_tif=aligned_tif,
                    template=template,
                    resampling_method=DEFAULT_RESAMPLING[element],
                )

                asc_path = asc_dir / element / f"{ASCII_PREFIX[element]}_{prism_date}.asc"
                _write_ascii_like_dem(aligned_tif, out_path=asc_path, template=template)

                row[CSV_COLUMN[element]] = _masked_mean(aligned_tif, fallback=0.0 if element == "ppt" else np.nan)
                row[f"{element}_tif_path"] = str(aligned_tif)
                row[f"{element}_asc_path"] = str(asc_path)

                if release_meta is not None:
                    release_date, grid_count = _mean_release_signature(release_meta)
                    if release_date is not None:
                        row[f"{element}_release_date"] = release_date
                    if grid_count is not None:
                        row[f"{element}_grid_count"] = grid_count

            rows.append(row)

    forcing_df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    forcing_df.to_csv(csv_path, index=False)

    manifest = _manifest_payload(
        start_date=start,
        end_date=end,
        dem_path=dem_path,
        aoi_path=aoi_path,
        region=region,
        resolution=resolution,
        template=template,
        result_df=forcing_df,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    return PrismProcessingResult(
        forcing_df=forcing_df,
        output_dir=output_dir,
        raw_dir=raw_dir,
        aligned_tif_dir=aligned_tif_dir,
        asc_dir=asc_dir,
        manifest_path=manifest_path,
        csv_path=csv_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download daily PRISM ppt/tmin/tmax, clip to AOI, align to a DEM grid, and export daily ASC files."
    )
    parser.add_argument("--start-date", required=True, help="Inclusive start date, for example 2025-12-07")
    parser.add_argument("--end-date", required=True, help="Inclusive end date, for example 2025-12-20")
    parser.add_argument("--dem-path", required=True, help="DEM path used as the template grid. ASC is supported.")
    parser.add_argument("--aoi-path", required=True, help="AOI shapefile path.")
    parser.add_argument("--output-dir", required=True, help="Output directory for PRISM forcings.")
    parser.add_argument("--dem-crs", default=None, help="Fallback CRS when the DEM file has no embedded CRS.")
    parser.add_argument("--region", default="us", help="PRISM region. Default: us")
    parser.add_argument("--resolution", default="800m", choices=["800m", "4km"], help="PRISM source resolution.")
    parser.add_argument("--force-download", action="store_true", help="Redownload PRISM grids even if cached.")
    parser.add_argument(
        "--skip-release-check",
        action="store_true",
        help="Skip checking PRISM release metadata before reusing cached files.",
    )
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=2.0,
        help="Delay between downloads to be polite to the PRISM service.",
    )
    args = parser.parse_args()

    result = build_prism_forcings(
        start_date=args.start_date,
        end_date=args.end_date,
        dem_path=args.dem_path,
        aoi_path=args.aoi_path,
        output_dir=args.output_dir,
        dem_crs=args.dem_crs,
        region=args.region,
        resolution=args.resolution,
        force_download=args.force_download,
        check_release_dates=not args.skip_release_check,
        timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
    )

    print("Saved CSV:", result.csv_path)
    print("Saved manifest:", result.manifest_path)
    print("Saved ASC root:", result.asc_dir)
    print("Saved aligned GeoTIFF root:", result.aligned_tif_dir)


__all__ = ["PrismProcessingResult", "build_prism_forcings", "fetch_release_metadata", "main"]


if __name__ == "__main__":
    main()
