from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from debris_landlab.mmp.config import MMPConfig
from debris_landlab.mmp.fields import load_ascii_values, nodata_mask, read_ascii_nodata

FORCING_PATH_COLUMNS = ("ppt_asc_path", "tmin_asc_path", "tmax_asc_path")
REQUIRED_FORCING_COLUMNS = ("datetime", *FORCING_PATH_COLUMNS)


@dataclass
class DailyForcing:
    forcing_csv: Path
    manifest_path: Path | None
    asc_dir: Path | None
    forcing_df: pd.DataFrame
    rainfall_arrays: list[np.ndarray]
    tempmin_arrays: list[np.ndarray]
    tempmax_arrays: list[np.ndarray]
    ppt_asc_paths: list[Path]
    tmin_asc_paths: list[Path]
    tmax_asc_paths: list[Path]
    nodata_masks: dict[str, list[np.ndarray]]


def build_daily_forcing(config: MMPConfig, grid) -> DailyForcing:
    """Load prepared daily PRISM forcing files for the requested model window."""

    forcing_cfg = config.forcing
    forcing_csv = Path(forcing_cfg.forcing_csv)
    forcing_df = _load_forcing_index(forcing_csv, scenario_dir=config.scenario_dir)
    forcing_df = _select_forcing_window(
        forcing_df,
        forcing_cfg.start_date,
        forcing_cfg.end_date,
    )

    rainfall_arrays, ppt_paths, ppt_masks = _load_daily_asc_arrays(
        forcing_df,
        "ppt_asc_path",
        "Precipitation",
        grid,
        sanitize=forcing_cfg.sanitize_nodata,
        variable="precipitation",
        temperature_fill_method=forcing_cfg.temperature_fill_method,
    )
    tempmin_arrays, tmin_paths, tmin_masks = _load_daily_asc_arrays(
        forcing_df,
        "tmin_asc_path",
        "Temperature_min",
        grid,
        sanitize=forcing_cfg.sanitize_nodata,
        variable="temperature",
        temperature_fill_method=forcing_cfg.temperature_fill_method,
    )
    tempmax_arrays, tmax_paths, tmax_masks = _load_daily_asc_arrays(
        forcing_df,
        "tmax_asc_path",
        "Temperature_max",
        grid,
        sanitize=forcing_cfg.sanitize_nodata,
        variable="temperature",
        temperature_fill_method=forcing_cfg.temperature_fill_method,
    )

    manifest_path = (
        forcing_cfg.manifest_path
        if forcing_cfg.manifest_path is not None and forcing_cfg.manifest_path.exists()
        else None
    )
    asc_dir = forcing_csv.parent / "asc"
    return DailyForcing(
        forcing_csv=forcing_csv,
        manifest_path=manifest_path,
        asc_dir=asc_dir if asc_dir.exists() else None,
        forcing_df=forcing_df,
        rainfall_arrays=rainfall_arrays,
        tempmin_arrays=tempmin_arrays,
        tempmax_arrays=tempmax_arrays,
        ppt_asc_paths=ppt_paths,
        tmin_asc_paths=tmin_paths,
        tmax_asc_paths=tmax_paths,
        nodata_masks={
            "ppt": ppt_masks,
            "tmin": tmin_masks,
            "tmax": tmax_masks,
        },
    )


def _load_forcing_index(forcing_csv: Path, *, scenario_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not forcing_csv.exists():
        discovered = _discover_forcing_files(forcing_csv.parent)
        if not discovered.empty:
            return discovered
        raise FileNotFoundError(
            "Prepared PRISM forcing index not found. Run build-prism-forcing first: "
            f"{forcing_csv}"
        )

    forcing_df = pd.read_csv(forcing_csv)
    missing = [column for column in REQUIRED_FORCING_COLUMNS if column not in forcing_df.columns]
    if missing:
        raise ValueError(f"Prepared forcing index missing required columns: {', '.join(missing)}")

    forcing_df = forcing_df.copy()
    forcing_df["datetime"] = pd.to_datetime(forcing_df["datetime"]).dt.normalize()
    for column in FORCING_PATH_COLUMNS:
        forcing_df[column] = [
            str(_resolve_forcing_path(raw_path, forcing_csv=forcing_csv, scenario_dir=scenario_dir))
            for raw_path in forcing_df[column]
        ]
    frames.append(forcing_df)

    discovered = _discover_forcing_files(forcing_csv.parent)
    if not discovered.empty:
        frames.append(discovered)

    forcing_df = pd.concat(frames, ignore_index=True)
    forcing_df = forcing_df.sort_values("datetime").drop_duplicates("datetime", keep="first")
    forcing_df = forcing_df.reset_index(drop=True)
    return forcing_df


def _discover_forcing_files(forcing_dir: Path) -> pd.DataFrame:
    asc_dir = forcing_dir / "asc"
    ppt = _forcing_paths_by_date(asc_dir / "ppt", prefix="precip")
    tmin = _forcing_paths_by_date(asc_dir / "tmin", prefix="tmin")
    tmax = _forcing_paths_by_date(asc_dir / "tmax", prefix="tmax")

    dates = sorted(set(ppt) & set(tmin) & set(tmax))
    rows = [
        {
            "datetime": pd.Timestamp(date),
            "ppt_asc_path": str(ppt[date]),
            "tmin_asc_path": str(tmin[date]),
            "tmax_asc_path": str(tmax[date]),
        }
        for date in dates
    ]
    return pd.DataFrame(rows, columns=REQUIRED_FORCING_COLUMNS)


def _forcing_paths_by_date(directory: Path, *, prefix: str) -> dict[pd.Timestamp, Path]:
    paths: dict[pd.Timestamp, Path] = {}
    if not directory.exists():
        return paths

    for path in sorted(directory.glob(f"{prefix}_*.asc")):
        date_token = path.stem.removeprefix(f"{prefix}_")
        try:
            timestamp = pd.to_datetime(date_token, format="%Y%m%d")
        except ValueError:
            continue
        paths[timestamp.normalize()] = path
    return paths


def _resolve_forcing_path(raw_path, *, forcing_csv: Path, scenario_dir: Path) -> Path:
    if pd.isna(raw_path):
        raise ValueError(f"Prepared forcing index contains an empty path in {forcing_csv}")

    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path

    for base_dir in (forcing_csv.parent, scenario_dir, Path.cwd()):
        candidate = base_dir / path
        if candidate.exists():
            return candidate
    return forcing_csv.parent / path


def _select_forcing_window(
    forcing_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    if forcing_df.empty:
        raise ValueError("No PRISM forcing rows were loaded")

    expected_dates = pd.date_range(start_date, end_date, freq="D")
    window = forcing_df[
        (forcing_df["datetime"] >= expected_dates[0])
        & (forcing_df["datetime"] <= expected_dates[-1])
    ].copy()
    window = window.sort_values("datetime").reset_index(drop=True)
    forcing_dates = list(window["datetime"])

    if len(window) != len(expected_dates) or forcing_dates != list(expected_dates):
        found = {timestamp.date().isoformat() for timestamp in forcing_dates}
        missing = [
            timestamp.date().isoformat()
            for timestamp in expected_dates
            if timestamp.date().isoformat() not in found
        ]
        raise ValueError(
            "Prepared PRISM forcing dates do not cover the requested model window. "
            f"Missing dates: {', '.join(missing) if missing else 'none'}"
        )

    return window


def _load_daily_asc_arrays(
    forcing_df: pd.DataFrame,
    path_column: str,
    field_name: str,
    reference_grid,
    *,
    sanitize: bool,
    variable: str,
    temperature_fill_method: str,
) -> tuple[list[np.ndarray], list[Path], list[np.ndarray]]:
    arrays: list[np.ndarray] = []
    paths: list[Path] = []
    masks: list[np.ndarray] = []

    for raw_path in forcing_df[path_column]:
        asc_path = Path(raw_path)
        if not asc_path.exists():
            raise FileNotFoundError(f"Prepared forcing ASC not found: {asc_path}")

        values = load_ascii_values(asc_path, field_name).copy()
        if values.size != reference_grid.number_of_nodes:
            raise ValueError(f"Grid mismatch for {asc_path}")

        header_nodata = read_ascii_nodata(asc_path)
        bad = nodata_mask(values, header_nodata)
        if sanitize:
            values = _sanitize_forcing_values(
                values,
                bad,
                variable=variable,
                temperature_fill_method=temperature_fill_method,
            )

        arrays.append(values)
        paths.append(asc_path)
        masks.append(bad)

    return arrays, paths, masks


def _sanitize_forcing_values(
    values: np.ndarray,
    bad: np.ndarray,
    *,
    variable: str,
    temperature_fill_method: str,
) -> np.ndarray:
    out = values.copy()
    if not np.any(bad):
        return out

    if variable == "precipitation":
        out[bad] = 0.0
        return out

    valid = out[~bad]
    if valid.size == 0:
        fill_value = 0.0
    elif temperature_fill_method == "median":
        fill_value = float(np.median(valid))
    elif temperature_fill_method == "zero":
        fill_value = 0.0
    elif temperature_fill_method == "mean":
        fill_value = float(np.mean(valid))
    else:
        raise ValueError(f"Unsupported temperature fill method: {temperature_fill_method}")

    out[bad] = fill_value
    return out
