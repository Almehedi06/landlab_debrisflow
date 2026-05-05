from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from debris_landlab.mmp.config import MMPConfig, load_mmp_config
from debris_landlab.mmp.daily_forcing import DailyForcing, build_daily_forcing
from debris_landlab.mmp.ecohydrology import EcohydrologyResult, run_daily_ecohydrology
from debris_landlab.mmp.exports import export_raster_fields
from debris_landlab.mmp.landslides import LandslideResult, run_landslide_probability
from debris_landlab.mmp.snow import SnowResult, run_snow_model
from debris_landlab.mmp.static_inputs import load_static_inputs
from debris_landlab.mmp.terrain import TerrainState, load_terrain_and_route


@dataclass
class PipelineResult:
    config: MMPConfig
    terrain: TerrainState
    forcing: DailyForcing
    snow: SnowResult
    ecohydrology: EcohydrologyResult
    landslide: LandslideResult
    exported_paths: list[Path]

    @property
    def grid(self):
        return self.terrain.grid

    def as_notebook_results(self) -> dict[str, Any]:
        """Return a notebook-friendly result dictionary matching the old workflow."""

        forcing = self.forcing
        eco = self.ecohydrology
        return {
            "forcing": forcing.forcing_df,
            "forcing_csv": str(forcing.forcing_csv),
            "forcing_manifest": str(forcing.manifest_path) if forcing.manifest_path else None,
            "forcing_asc_dir": str(forcing.asc_dir) if forcing.asc_dir else None,
            "prism_csv": str(forcing.forcing_csv),
            "prism_manifest": str(forcing.manifest_path) if forcing.manifest_path else None,
            "prism_asc_dir": str(forcing.asc_dir) if forcing.asc_dir else None,
            "ppt_asc_paths": [str(path) for path in forcing.ppt_asc_paths],
            "tmin_asc_paths": [str(path) for path in forcing.tmin_asc_paths],
            "tmax_asc_paths": [str(path) for path in forcing.tmax_asc_paths],
            "forcing_nodata_masks": forcing.nodata_masks,
            "low_ids": self.terrain.low_outlet_node_ids,
            "rainfall_arrays": forcing.rainfall_arrays,
            "tempmin_arrays": forcing.tempmin_arrays,
            "tempmax_arrays": forcing.tempmax_arrays,
            "rain_depth_arrays": self.snow.rain_depth_arrays,
            "snow_depth_arrays": self.snow.snow_depth_arrays,
            "snow_fraction_arrays": self.snow.snow_fraction_arrays,
            "water_input_arrays": self.snow.water_input_arrays,
            "swe_arrays": self.snow.swe_arrays,
            "melt_arrays": self.snow.melt_arrays,
            "mean_runoff": eco.mean_runoff,
            "mean_recharge": eco.mean_recharge,
            "max_runoff": eco.max_runoff,
            "max_recharge": eco.max_recharge,
            "runoff_arrays": eco.runoff_arrays,
            "recharge_arrays": eco.recharge_arrays,
            "soil_moisture_arrays": eco.soil_moisture_arrays,
            "ET_arrays": eco.et_arrays,
            "exported_paths": [str(path) for path in self.exported_paths],
        }


def run_pipeline_from_config(
    config_path: str | Path,
    *,
    overrides: list[str | Path] | None = None,
) -> PipelineResult:
    return run_pipeline(load_mmp_config(config_path, overrides=overrides))


def run_pipeline(config: MMPConfig) -> PipelineResult:
    terrain = load_terrain_and_route(config)
    grid = terrain.grid

    load_static_inputs(grid, config)
    forcing = build_daily_forcing(config, grid)

    if config.snow.enabled:
        snow = run_snow_model(
            grid.number_of_nodes,
            forcing.rainfall_arrays,
            forcing.tempmin_arrays,
            forcing.tempmax_arrays,
            t_snow_c=config.snow.t_snow_c,
            t_rain_c=config.snow.t_rain_c,
            melt_factor_mm_per_c_day=config.snow.melt_factor_mm_per_c_day,
            melt_base_temp_c=config.snow.melt_base_temp_c,
        )
        hydro_input_arrays = snow.water_input_arrays
    else:
        snow = _empty_snow_result(grid.number_of_nodes, forcing.rainfall_arrays)
        hydro_input_arrays = forcing.rainfall_arrays

    ecohydrology = run_daily_ecohydrology(
        grid,
        hydro_input_arrays,
        forcing.tempmin_arrays,
        forcing.tempmax_arrays,
        forcing.forcing_df["datetime"],
        config.ecohydrology,
    )

    landslide = run_landslide_probability(
        grid,
        mean_runoff=ecohydrology.mean_runoff,
        mean_recharge=ecohydrology.mean_recharge,
        max_runoff=ecohydrology.max_runoff,
        max_recharge=ecohydrology.max_recharge,
        config=config.landslide,
    )

    exported_paths = export_raster_fields(grid, config.export, scenario_dir=config.scenario_dir)
    return PipelineResult(
        config=config,
        terrain=terrain,
        forcing=forcing,
        snow=snow,
        ecohydrology=ecohydrology,
        landslide=landslide,
        exported_paths=exported_paths,
    )


def _empty_snow_result(number_of_nodes: int, rainfall_arrays: list[np.ndarray]) -> SnowResult:
    zeros = [np.zeros(number_of_nodes, dtype=float) for _ in rainfall_arrays]
    return SnowResult(
        rain_depth_arrays=[array.copy() for array in rainfall_arrays],
        snow_depth_arrays=[array.copy() for array in zeros],
        snow_fraction_arrays=[array.copy() for array in zeros],
        water_input_arrays=[array.copy() for array in rainfall_arrays],
        swe_arrays=[array.copy() for array in zeros],
        melt_arrays=[array.copy() for array in zeros],
    )
