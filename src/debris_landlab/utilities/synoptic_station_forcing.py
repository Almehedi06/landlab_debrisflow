from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely import union_all
from shapely.geometry import Point


SYNOPTIC_PRECIP_URL = "https://api.synopticdata.com/v2/stations/precip"
DEFAULT_OUTDIR = "/mnt/c/Users/amehedi/Downloads"


def _format_synoptic_time(value: str) -> str:
    """Return a Synoptic API UTC timestamp in YYYYmmddHHMM form."""
    value = str(value).strip()
    if len(value) == 12 and value.isdigit():
        return value

    timestamp = pd.to_datetime(value, utc=True)
    if pd.isna(timestamp):
        raise ValueError(f"Could not parse time value: {value}")
    return timestamp.strftime("%Y%m%d%H%M")


def _validate_interval_hours(interval_hours: int) -> int:
    interval_hours = int(interval_hours)
    if interval_hours <= 0:
        raise ValueError("interval_hours must be a positive integer.")
    if 24 % interval_hours != 0 and interval_hours % 24 != 0:
        raise ValueError(
            "Synoptic interval must be a factor or multiple of 24 hours "
            "(for example 1, 2, 3, 4, 6, 8, 12, 24, 48)."
        )
    return interval_hours


def _request_json(
    session: requests.Session,
    params: dict[str, Any],
    *,
    timeout: int,
) -> dict[str, Any]:
    response = session.get(SYNOPTIC_PRECIP_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    summary = payload.get("SUMMARY", {})
    response_code = summary.get("RESPONSE_CODE")
    if response_code not in (1, "1", 2, "2", None):
        message = summary.get("RESPONSE_MESSAGE", "Unknown Synoptic API error")
        raise RuntimeError(f"Synoptic API error {response_code}: {message}")

    return payload


def _read_aoi(aoi_path: str | Path) -> tuple[gpd.GeoDataFrame, Any]:
    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise ValueError(f"AOI has no features: {aoi_path}")
    if aoi.crs is None:
        raise ValueError(f"AOI has no CRS. Define one before running: {aoi_path}")

    aoi = aoi.to_crs("EPSG:4326")
    return aoi, union_all(aoi.geometry.to_numpy())


def _point_gdf(point_lon: float, point_lat: float) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        geometry=[Point(float(point_lon), float(point_lat))],
        crs="EPSG:4326",
    )


def _buffer_aoi(aoi: gpd.GeoDataFrame, buffer_km: float) -> tuple[Any, Any]:
    utm_crs = aoi.estimate_utm_crs()
    if utm_crs is None:
        raise ValueError("Could not estimate a projected CRS for the AOI buffer.")

    aoi_utm = aoi.to_crs(utm_crs)
    buffer_geom_utm = union_all(aoi_utm.geometry.to_numpy()).buffer(buffer_km * 1000.0)

    buffer_gdf = gpd.GeoDataFrame(geometry=[buffer_geom_utm], crs=utm_crs).to_crs("EPSG:4326")
    return buffer_gdf.geometry.iloc[0], utm_crs


def _buffer_point(point_lon: float, point_lat: float, buffer_km: float) -> tuple[Any, Any]:
    point = _point_gdf(point_lon, point_lat)
    utm_crs = point.estimate_utm_crs()
    if utm_crs is None:
        raise ValueError("Could not estimate a projected CRS for the point buffer.")

    point_utm = point.to_crs(utm_crs).geometry.iloc[0]
    buffer_geom_utm = point_utm.buffer(buffer_km * 1000.0)
    buffer_gdf = gpd.GeoDataFrame(geometry=[buffer_geom_utm], crs=utm_crs).to_crs("EPSG:4326")
    return buffer_gdf.geometry.iloc[0], utm_crs


def _validate_spatial_inputs(
    aoi_path: str | Path | None,
    point_lon: float | None,
    point_lat: float | None,
    buffer_km: float,
) -> None:
    has_aoi = aoi_path is not None
    has_lon = point_lon is not None
    has_lat = point_lat is not None

    if not has_aoi and not (has_lon and has_lat):
        raise ValueError("Provide either aoi_path or both point_lon and point_lat.")
    if has_lon != has_lat:
        raise ValueError("point_lon and point_lat must be provided together.")
    if buffer_km < 0:
        raise ValueError("buffer_km must be >= 0.")
    if not has_aoi and buffer_km == 0:
        raise ValueError("Point-only searches require buffer_km > 0.")

    if has_lon and has_lat:
        lon = float(point_lon)
        lat = float(point_lat)
        if not -180 <= lon <= 180:
            raise ValueError(f"point_lon must be between -180 and 180: {point_lon}")
        if not -90 <= lat <= 90:
            raise ValueError(f"point_lat must be between -90 and 90: {point_lat}")


def _stations_from_payload(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for station in payload.get("STATION", []):
        stid = station.get("STID")
        latitude = pd.to_numeric(station.get("LATITUDE"), errors="coerce")
        longitude = pd.to_numeric(station.get("LONGITUDE"), errors="coerce")
        if not stid or pd.isna(latitude) or pd.isna(longitude):
            continue

        rows.append(
            {
                "stid": stid,
                "name": station.get("NAME"),
                "latitude": float(latitude),
                "longitude": float(longitude),
                "elevation": station.get("ELEVATION"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["stid", "name", "latitude", "longitude", "elevation"])
    return pd.DataFrame(rows).drop_duplicates("stid")


def _rainfall_from_payload(payload: dict[str, Any], requested_interval_hours: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for station in payload.get("STATION", []):
        stid = station.get("STID")
        name = station.get("NAME")
        obs = station.get("OBSERVATIONS", {})

        for item in obs.get("precipitation", []):
            rows.append(
                {
                    "stid": stid,
                    "name": name,
                    "first_report": item.get("first_report"),
                    "last_report": item.get("last_report"),
                    "interval": item.get("interval"),
                    "requested_interval_hours": requested_interval_hours,
                    "precip_mm": item.get("total"),
                    "count": item.get("count"),
                    "report_type": item.get("report_type"),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "stid",
            "name",
            "first_report",
            "last_report",
            "interval",
            "requested_interval_hours",
            "precip_mm",
            "count",
            "report_type",
        ],
    )


def run_station_forcing(
    aoi_path: str | Path | None = None,
    point_lon: float | None = None,
    point_lat: float | None = None,
    start: str | None = None,
    end: str | None = None,
    buffer_km: float = 30,
    interval_hours: int = 1,
    outdir: str | Path | None = None,
    token: str | None = None,
    timeout: int = 120,
    include_nearby: bool = True,
) -> tuple[gpd.GeoDataFrame | None, pd.DataFrame | None]:
    """
    Find Synoptic precip stations for an AOI, a point location, or both.

    Parameters
    ----------
    aoi_path
        Optional path to an AOI vector file readable by GeoPandas. If provided,
        stations are selected inside the AOI and, by default, within buffer_km
        of the AOI.
    point_lon, point_lat
        Optional event/model point in longitude/latitude decimal degrees. If no
        AOI is provided, this point is used as the search target with buffer_km.
        If an AOI is provided, the point is used for distance sorting.
    start, end
        UTC time window. Use YYYYmmddHHMM or any pandas-parseable date/time.
    buffer_km
        Distance around the AOI or point used to include nearby stations.
    interval_hours
        Precipitation interval in hours. Synoptic accepts factors or multiples
        of 24, such as 1, 2, 3, 4, 6, 8, 12, 24, and 48.
    outdir
        Output directory for stations.csv, stations.gpkg, and rainfall.csv.
        If omitted, files are written to /mnt/c/Users/amehedi/Downloads.
    token
        Synoptic API token. If omitted, SYNOPTIC_TOKEN is read from the environment.
    timeout
        Request timeout in seconds.
    include_nearby
        When an AOI is provided, keep stations in the AOI buffer. Set False to
        keep only stations inside the AOI.
    """
    outdir = Path(DEFAULT_OUTDIR if outdir is None else outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    token = token or os.getenv("SYNOPTIC_TOKEN")
    if not token:
        raise ValueError("Set SYNOPTIC_TOKEN or pass token='your_token'.")
    if start is None or end is None:
        raise ValueError("start and end are required.")

    start_time = _format_synoptic_time(start)
    end_time = _format_synoptic_time(end)
    if end_time <= start_time:
        raise ValueError(f"end must be after start: start={start_time}, end={end_time}")
    interval_hours = _validate_interval_hours(interval_hours)
    _validate_spatial_inputs(aoi_path, point_lon, point_lat, buffer_km)

    aoi_geom = None
    point = None
    if aoi_path is not None:
        aoi, aoi_geom = _read_aoi(aoi_path)
        aoi_buffer_km = buffer_km if include_nearby else 0.0
        search_geom, utm_crs = _buffer_aoi(aoi, aoi_buffer_km)
    else:
        search_geom, utm_crs = _buffer_point(float(point_lon), float(point_lat), buffer_km)

    if point_lon is not None and point_lat is not None:
        point = _point_gdf(point_lon, point_lat)

    minx, miny, maxx, maxy = search_geom.bounds
    bbox = f"{minx},{miny},{maxx},{maxy}"

    with requests.Session() as session:
        station_payload = _request_json(
            session,
            {
                "token": token,
                "bbox": bbox,
                "start": start_time,
                "end": end_time,
                "pmode": "totals",
                "units": "precip|mm",
                "showemptystations": 1,
            },
            timeout=timeout,
        )

        stations_df = _stations_from_payload(station_payload)
        if stations_df.empty:
            print("No stations found.")
            return None, None

        stations_gdf = gpd.GeoDataFrame(
            stations_df,
            geometry=gpd.points_from_xy(stations_df.longitude, stations_df.latitude),
            crs="EPSG:4326",
        )

        stations_gdf["inside_aoi"] = False
        stations_gdf["near_aoi"] = False
        stations_gdf["within_point_buffer"] = False
        stations_gdf["category"] = "outside"

        stations_utm = stations_gdf.to_crs(utm_crs)

        if aoi_geom is not None:
            stations_gdf["inside_aoi"] = stations_gdf.geometry.apply(aoi_geom.covers)
            stations_gdf["near_aoi"] = stations_gdf.geometry.apply(search_geom.covers) & (
                ~stations_gdf["inside_aoi"]
            )

            aoi_geom_utm = gpd.GeoSeries([aoi_geom], crs="EPSG:4326").to_crs(utm_crs).iloc[0]
            stations_gdf["distance_to_aoi_km"] = stations_utm.geometry.distance(aoi_geom_utm) / 1000.0

        if point is not None:
            point_utm = point.to_crs(utm_crs).geometry.iloc[0]
            stations_gdf["distance_to_point_km"] = stations_utm.geometry.distance(point_utm) / 1000.0

            if aoi_geom is None:
                stations_gdf["within_point_buffer"] = stations_gdf.geometry.apply(search_geom.covers)

        if aoi_geom is not None:
            keep = stations_gdf["inside_aoi"].copy()
            if include_nearby:
                keep = keep | stations_gdf["near_aoi"]
            stations_gdf.loc[stations_gdf["near_aoi"], "category"] = "near_aoi"
            stations_gdf.loc[stations_gdf["inside_aoi"], "category"] = "inside_aoi"
        else:
            keep = stations_gdf["within_point_buffer"]
            stations_gdf.loc[stations_gdf["within_point_buffer"], "category"] = "near_point"

        stations_gdf = stations_gdf[keep].copy()
        if stations_gdf.empty:
            target = "AOI" if aoi_geom is not None else "point buffer"
            print(f"No stations found for the {target}.")
            return None, None

        category_priority = {"inside_aoi": 0, "near_aoi": 1, "near_point": 2}
        stations_gdf["_category_priority"] = stations_gdf["category"].map(category_priority).fillna(99)
        sort_columns = ["_category_priority"]
        if "distance_to_point_km" in stations_gdf.columns:
            sort_columns.append("distance_to_point_km")
        elif "distance_to_aoi_km" in stations_gdf.columns:
            sort_columns.append("distance_to_aoi_km")
        stations_gdf = (
            stations_gdf.sort_values(sort_columns)
            .drop(columns="_category_priority")
            .reset_index(drop=True)
        )

        rain_payload = _request_json(
            session,
            {
                "token": token,
                "stid": ",".join(stations_gdf["stid"].tolist()),
                "start": start_time,
                "end": end_time,
                "pmode": "intervals",
                "interval": interval_hours,
                "units": "precip|mm",
            },
            timeout=timeout,
        )

    rainfall_df = _rainfall_from_payload(rain_payload, interval_hours)

    stations_csv = outdir / "stations.csv"
    stations_gpkg = outdir / "stations.gpkg"
    rainfall_csv = outdir / "rainfall.csv"

    stations_gdf.to_csv(stations_csv, index=False)
    stations_gdf.to_file(stations_gpkg, driver="GPKG")
    rainfall_df.to_csv(rainfall_csv, index=False)

    print("Done.")
    print(f"Stations: {len(stations_gdf)}")
    print(f"Rainfall rows: {len(rainfall_df)}")
    print(f"Saved: {stations_csv}")
    print(f"Saved: {stations_gpkg}")
    print(f"Saved: {rainfall_csv}")

    return stations_gdf, rainfall_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find Synoptic precipitation stations for an AOI or point and download interval rainfall."
    )
    parser.add_argument(
        "--aoi-path",
        default=None,
        help="Optional AOI vector path, for example a shapefile or GeoPackage.",
    )
    parser.add_argument("--point-lon", type=float, default=None, help="Optional event/model point longitude.")
    parser.add_argument("--point-lat", type=float, default=None, help="Optional event/model point latitude.")
    parser.add_argument(
        "--start",
        required=True,
        help="UTC start time. Prefer YYYYmmddHHMM, for example 202409250000.",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="UTC end time. Prefer YYYYmmddHHMM, for example 202409280000.",
    )
    parser.add_argument("--buffer-km", type=float, default=30, help="AOI or point buffer distance in kilometers.")
    parser.add_argument("--interval-hours", type=int, default=1, help="Rainfall accumulation interval in hours.")
    parser.add_argument("--outdir", default=None, help=f"Output directory. Default: {DEFAULT_OUTDIR}")
    parser.add_argument(
        "--inside-aoi-only",
        action="store_true",
        help="When --aoi-path is set, keep only stations inside the AOI and ignore the AOI buffer.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Synoptic API token. If omitted, the SYNOPTIC_TOKEN environment variable is used.",
    )
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds.")
    args = parser.parse_args()

    run_station_forcing(
        aoi_path=args.aoi_path,
        point_lon=args.point_lon,
        point_lat=args.point_lat,
        start=args.start,
        end=args.end,
        buffer_km=args.buffer_km,
        interval_hours=args.interval_hours,
        outdir=args.outdir,
        token=args.token,
        timeout=args.timeout,
        include_nearby=not args.inside_aoi_only,
    )


__all__ = ["DEFAULT_OUTDIR", "SYNOPTIC_PRECIP_URL", "main", "run_station_forcing"]


if __name__ == "__main__":
    main()
