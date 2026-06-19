from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SnowResult:
    rain_depth_arrays: list[np.ndarray]
    snow_depth_arrays: list[np.ndarray]
    snow_fraction_arrays: list[np.ndarray]
    water_input_arrays: list[np.ndarray]
    swe_arrays: list[np.ndarray]
    melt_arrays: list[np.ndarray]


def partition_precipitation_phase(
    rainfall_arrays: list[np.ndarray],
    tempmin_arrays: list[np.ndarray],
    tempmax_arrays: list[np.ndarray],
    *,
    t_snow_c: float,
    t_rain_c: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Split precipitation into rain and snow with a linear transition band."""

    rain_depth_arrays: list[np.ndarray] = []
    snow_depth_arrays: list[np.ndarray] = []
    snow_fraction_arrays: list[np.ndarray] = []

    for precipitation, tmin, tmax in zip(rainfall_arrays, tempmin_arrays, tempmax_arrays):
        tavg = (tmin + tmax) / 2.0
        snow_fraction = (t_rain_c - tavg) / (t_rain_c - t_snow_c)
        snow_fraction = np.clip(snow_fraction, 0.0, 1.0)

        bad = (
            ~np.isfinite(precipitation)
            | ~np.isfinite(tmin)
            | ~np.isfinite(tmax)
        )
        snow_fraction[bad] = 0.0

        snow_depth = precipitation * snow_fraction
        rain_depth = precipitation * (1.0 - snow_fraction)
        snow_depth[bad] = 0.0
        rain_depth[bad] = 0.0

        rain_depth_arrays.append(rain_depth)
        snow_depth_arrays.append(snow_depth)
        snow_fraction_arrays.append(snow_fraction)

    return rain_depth_arrays, snow_depth_arrays, snow_fraction_arrays


def build_swe_and_water_input(
    number_of_nodes: int,
    rain_depth_arrays: list[np.ndarray],
    snow_depth_arrays: list[np.ndarray],
    tempmin_arrays: list[np.ndarray],
    tempmax_arrays: list[np.ndarray],
    *,
    melt_factor_mm_per_c_day: float,
    melt_base_temp_c: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Accumulate snow-water equivalent and return daily liquid water input."""

    water_input_arrays: list[np.ndarray] = []
    swe_arrays: list[np.ndarray] = []
    melt_arrays: list[np.ndarray] = []
    # Initialize the SWE store
    # swe_store = np.zeros(number_of_nodes, dtype=float)
    swe_store = np.full(number_of_nodes, 118.0, dtype=float)  # initial SWE in mm


    for rain, snow, tmin, tmax in zip(
        rain_depth_arrays,
        snow_depth_arrays,
        tempmin_arrays,
        tempmax_arrays,
    ):
        tavg = (tmin + tmax) / 2.0
        bad = (
            ~np.isfinite(rain)
            | ~np.isfinite(snow)
            | ~np.isfinite(tmin)
            | ~np.isfinite(tmax)
        )

        swe_store += np.where(bad, 0.0, snow)
        potential_melt = melt_factor_mm_per_c_day * np.maximum(tavg - melt_base_temp_c, 0.0)
        actual_melt = np.minimum(swe_store, potential_melt)
        swe_store -= actual_melt

        water_input = rain + actual_melt
        water_input[bad] = 0.0

        water_input_arrays.append(water_input.copy())
        swe_arrays.append(swe_store.copy())
        melt_arrays.append(actual_melt.copy())

    return water_input_arrays, swe_arrays, melt_arrays


def run_snow_model(
    number_of_nodes: int,
    rainfall_arrays: list[np.ndarray],
    tempmin_arrays: list[np.ndarray],
    tempmax_arrays: list[np.ndarray],
    *,
    t_snow_c: float,
    t_rain_c: float,
    melt_factor_mm_per_c_day: float,
    melt_base_temp_c: float,
) -> SnowResult:
    rain_depth_arrays, snow_depth_arrays, snow_fraction_arrays = partition_precipitation_phase(
        rainfall_arrays,
        tempmin_arrays,
        tempmax_arrays,
        t_snow_c=t_snow_c,
        t_rain_c=t_rain_c,
    )
    water_input_arrays, swe_arrays, melt_arrays = build_swe_and_water_input(
        number_of_nodes,
        rain_depth_arrays,
        snow_depth_arrays,
        tempmin_arrays,
        tempmax_arrays,
        melt_factor_mm_per_c_day=melt_factor_mm_per_c_day,
        melt_base_temp_c=melt_base_temp_c,
    )
    return SnowResult(
        rain_depth_arrays=rain_depth_arrays,
        snow_depth_arrays=snow_depth_arrays,
        snow_fraction_arrays=snow_fraction_arrays,
        water_input_arrays=water_input_arrays,
        swe_arrays=swe_arrays,
        melt_arrays=melt_arrays,
    )
