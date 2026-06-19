from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely import union_all


NHDPLUS_HR_FLOWLINE_QUERY_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer/3/query"
DEFAULT_OUTDIR = "/mnt/c/Users/amehedi/Downloads"
DEFAULT_BASENAME = "nhdplus_hr_flowlines"
FEATURE_TYPE_WHERE = {
    "stream_river": "ftype = 460",
    "stream_river_artificial_path": "ftype IN (460, 558)",
    "all_network": "1=1",
}


def _read_aoi(aoi_path: str | Path) -> tuple[gpd.GeoDataFrame, Any]:
    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise ValueError(f"AOI has no features: {aoi_path}")
    if aoi.crs is None:
        raise ValueError(f"AOI has no CRS. Define one before running: {aoi_path}")

    aoi = aoi.to_crs("EPSG:4326")
    return aoi, union_all(aoi.geometry.to_numpy())


def _bbox_geometry(aoi_geom: Any) -> str:
    minx, miny, maxx, maxy = aoi_geom.bounds
    return json.dumps(
        {
            "xmin": minx,
            "ymin": miny,
            "xmax": maxx,
            "ymax": maxy,
            "spatialReference": {"wkid": 4326},
        }
    )


def _combined_where(feature_type: str, where: str | None) -> str:
    if feature_type not in FEATURE_TYPE_WHERE:
        choices = ", ".join(sorted(FEATURE_TYPE_WHERE))
        raise ValueError(f"feature_type must be one of: {choices}")

    base_where = FEATURE_TYPE_WHERE[feature_type]
    if not where:
        return base_where
    return f"({base_where}) AND ({where})"


def _request_geojson(
    session: requests.Session,
    *,
    bbox: str,
    where: str,
    out_fields: str,
    offset: int,
    page_size: int,
    timeout: int,
) -> dict[str, Any]:
    response = session.post(
        NHDPLUS_HR_FLOWLINE_QUERY_URL,
        data={
            "f": "geojson",
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "true",
            "geometry": bbox,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "orderByFields": "OBJECTID",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    if "error" in payload:
        error = payload["error"]
        code = error.get("code", "unknown")
        message = error.get("message", "Unknown ArcGIS REST error")
        details = "; ".join(error.get("details", []))
        if details:
            message = f"{message}: {details}"
        raise RuntimeError(f"NHDPlus HR service error {code}: {message}")

    return payload


def _features_to_gdf(features: list[dict[str, Any]]) -> gpd.GeoDataFrame:
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")


def _download_bbox_flowlines(
    *,
    bbox: str,
    where: str,
    out_fields: str,
    page_size: int,
    timeout: int,
    max_features: int | None,
) -> gpd.GeoDataFrame:
    if page_size < 1 or page_size > 2000:
        raise ValueError("page_size must be between 1 and 2000.")
    if max_features is not None and max_features < 1:
        raise ValueError("max_features must be >= 1 when provided.")

    frames: list[gpd.GeoDataFrame] = []
    offset = 0

    with requests.Session() as session:
        while True:
            payload = _request_geojson(
                session,
                bbox=bbox,
                where=where,
                out_fields=out_fields,
                offset=offset,
                page_size=page_size,
                timeout=timeout,
            )
            features = payload.get("features", [])
            if max_features is not None:
                remaining = max_features - sum(len(frame) for frame in frames)
                features = features[:remaining]

            if features:
                frames.append(_features_to_gdf(features))

            fetched = len(features)
            if fetched == 0:
                break
            if max_features is not None and sum(len(frame) for frame in frames) >= max_features:
                break
            if fetched < page_size and not payload.get("exceededTransferLimit", False):
                break

            offset += page_size

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")


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


def download_nhdplus_flowlines(
    aoi_path: str | Path,
    outdir: str | Path | None = None,
    basename: str = DEFAULT_BASENAME,
    feature_type: str = "stream_river",
    where: str | None = None,
    out_fields: str = "*",
    page_size: int = 2000,
    timeout: int = 120,
    max_features: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Download USGS NHDPlus HR flowlines intersecting an AOI.

    Parameters
    ----------
    aoi_path
        AOI vector path readable by GeoPandas.
    outdir
        Output directory. If omitted, files are written to
        /mnt/c/Users/amehedi/Downloads.
    basename
        Base filename for GeoPackage, GeoJSON, and CSV outputs.
    feature_type
        One of ``stream_river``, ``stream_river_artificial_path``, or
        ``all_network``.
    where
        Optional extra ArcGIS SQL where clause.
    out_fields
        ArcGIS outFields value. Use "*" for all attributes.
    page_size
        ArcGIS REST page size. The service maximum is 2000.
    timeout
        Request timeout in seconds.
    max_features
        Optional cap useful for testing very large AOIs.
    """
    outdir = Path(DEFAULT_OUTDIR if outdir is None else outdir)
    aoi, aoi_geom = _read_aoi(aoi_path)
    bbox = _bbox_geometry(aoi_geom)
    query_where = _combined_where(feature_type, where)

    print("Downloading NHDPlus HR flowlines...")
    print(f"Feature type: {feature_type}")
    print(f"Output directory: {outdir}")

    flowlines = _download_bbox_flowlines(
        bbox=bbox,
        where=query_where,
        out_fields=out_fields,
        page_size=page_size,
        timeout=timeout,
        max_features=max_features,
    )

    if flowlines.empty:
        print("No NHDPlus HR flowlines found for the AOI bounding box.")
        return flowlines

    clip_aoi = gpd.GeoDataFrame(geometry=[aoi_geom], crs="EPSG:4326")
    flowlines = gpd.clip(flowlines, clip_aoi).reset_index(drop=True)
    if flowlines.empty:
        print("No NHDPlus HR flowlines found after clipping to the AOI.")
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
        description="Download USGS NHDPlus HR flowline features intersecting an AOI."
    )
    parser.add_argument("--aoi-path", required=True, help="AOI vector path, for example a shapefile or GeoPackage.")
    parser.add_argument("--outdir", default=None, help=f"Output directory. Default: {DEFAULT_OUTDIR}")
    parser.add_argument("--basename", default=DEFAULT_BASENAME, help="Base output filename.")
    parser.add_argument(
        "--feature-type",
        default="stream_river",
        choices=sorted(FEATURE_TYPE_WHERE),
        help="Which NHDPlus HR flowline feature types to download.",
    )
    parser.add_argument("--where", default=None, help="Optional extra ArcGIS SQL where clause.")
    parser.add_argument("--out-fields", default="*", help='ArcGIS outFields value. Default: "*".')
    parser.add_argument("--page-size", type=int, default=2000, help="ArcGIS REST page size; max 2000.")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds.")
    parser.add_argument("--max-features", type=int, default=None, help="Optional cap for testing large AOIs.")
    args = parser.parse_args()

    download_nhdplus_flowlines(
        aoi_path=args.aoi_path,
        outdir=args.outdir,
        basename=args.basename,
        feature_type=args.feature_type,
        where=args.where,
        out_fields=args.out_fields,
        page_size=args.page_size,
        timeout=args.timeout,
        max_features=args.max_features,
    )


__all__ = [
    "DEFAULT_BASENAME",
    "DEFAULT_OUTDIR",
    "FEATURE_TYPE_WHERE",
    "NHDPLUS_HR_FLOWLINE_QUERY_URL",
    "download_nhdplus_flowlines",
    "main",
]


if __name__ == "__main__":
    main()
