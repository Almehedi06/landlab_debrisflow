from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from debris_landlab.components import PotentialEvapotranspiration, SoilMoisture
from debris_landlab.mmp.config import EcohydrologyConfig
from debris_landlab.mmp.fields import add_or_update_field, cell_to_node


@dataclass
class EcohydrologyResult:
    runoff_arrays: list[np.ndarray]
    recharge_arrays: list[np.ndarray]
    soil_moisture_arrays: list[np.ndarray]
    et_arrays: list[np.ndarray]
    mean_runoff: np.ndarray
    mean_recharge: np.ndarray
    max_runoff: np.ndarray
    max_recharge: np.ndarray


def date_to_model_year_fraction(value: date | str | pd.Timestamp) -> float:
    timestamp = pd.Timestamp(value)
    days_in_year = 366.0 if timestamp.is_leap_year else 365.0
    return float((timestamp.dayofyear - 1) / days_in_year)


def run_daily_ecohydrology(
    grid,
    hydro_input_arrays: list[np.ndarray],
    tempmin_arrays: list[np.ndarray],
    tempmax_arrays: list[np.ndarray],
    forcing_dates,
    config: EcohydrologyConfig,
) -> EcohydrologyResult:
    """Run PET and root-zone soil moisture one day at a time."""

    pet = PotentialEvapotranspiration(grid, method="PenmanMonteith")
    soil_moisture = SoilMoisture(grid)

    pet._latitude = config.latitude_deg
    pet._a = config.albedo
    pet._zm = config.z_wind_m
    pet._zveg = config.zveg_m * np.ones(grid.number_of_cells)
    pet._vz = config.vwind_mps * np.ones(grid.number_of_cells)
    pet._relative_humidity = config.relative_humidity * np.ones(grid.number_of_cells)
    pet._LAI = grid.at_node["vegetation__live_leaf_area_index"]

    soil_moisture._Tb = config.storm_duration_hours
    node_at_cell = grid.node_at_cell

    runoff_arrays: list[np.ndarray] = []
    recharge_arrays: list[np.ndarray] = []
    soil_moisture_arrays: list[np.ndarray] = []
    et_arrays: list[np.ndarray] = []

    dates = list(pd.to_datetime(forcing_dates))
    if not dates:
        raise ValueError("Daily ecohydrology requires at least one forcing date")

    initial_time = (
        float(config.initial_time_years)
        if config.initial_time_years is not None
        else date_to_model_year_fraction(dates[0])
    )

    for day_index, (rainfall, tempmin, tempmax) in enumerate(
        zip(hydro_input_arrays, tempmin_arrays, tempmax_arrays)
    ):
        model_time = (
            initial_time + day_index / 365.25
            if config.initial_time_years is not None
            else date_to_model_year_fraction(dates[day_index])
        )

        add_or_update_field(grid, "Precipitation", rainfall, at="node")
        add_or_update_field(grid, "Tmin", tempmin, at="node")
        add_or_update_field(grid, "Tmax", tempmax, at="node")
        grid.at_cell["rainfall__daily_depth"][:] = rainfall[node_at_cell]

        pet._current_time = model_time
        pet._Tmin = tempmin
        pet._Tmax = tempmax
        pet.update()

        soil_moisture._current_time = model_time
        soil_moisture.update()

        recharge_arrays.append(cell_to_node(grid, grid.at_cell["soil_moisture__root_zone_leakage"]))
        runoff_arrays.append(cell_to_node(grid, grid.at_cell["surface__runoff"]))
        soil_moisture_arrays.append(
            cell_to_node(grid, grid.at_cell["soil_moisture__saturation_fraction"])
        )
        et_arrays.append(cell_to_node(grid, grid.at_cell["surface__evapotranspiration"]))

    mean_runoff = np.mean(runoff_arrays, axis=0)
    mean_recharge = np.mean(recharge_arrays, axis=0)
    max_runoff = np.maximum.reduce(runoff_arrays)
    max_recharge = np.maximum.reduce(recharge_arrays)

    return EcohydrologyResult(
        runoff_arrays=runoff_arrays,
        recharge_arrays=recharge_arrays,
        soil_moisture_arrays=soil_moisture_arrays,
        et_arrays=et_arrays,
        mean_runoff=mean_runoff,
        mean_recharge=mean_recharge,
        max_runoff=max_runoff,
        max_recharge=max_recharge,
    )
