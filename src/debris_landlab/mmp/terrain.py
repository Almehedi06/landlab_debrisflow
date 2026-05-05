from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from landlab.components import FlowAccumulator
from landlab.io import esri_ascii

from debris_landlab.mmp.config import MMPConfig
from debris_landlab.mmp.fields import add_or_update_field


@dataclass
class TerrainState:
    grid: object
    low_outlet_node_ids: np.ndarray
    drainage_area: np.ndarray
    discharge: np.ndarray
    elevation_min_core: float
    elevation_max_core: float


def load_terrain_and_route(config: MMPConfig) -> TerrainState:
    """Load DEM, apply boundary conditions, and compute routing fields."""

    dem_path = config.scenario_dir / config.terrain.dem_filename
    with dem_path.open() as src:
        grid = esri_ascii.load(src, name="topographic__elevation", at="node")

    elevation = grid.at_node["topographic__elevation"]
    grid.set_nodata_nodes_to_closed(elevation, config.terrain.dem_nodata)

    low_ids = np.where(
        (grid.status_at_node != grid.BC_NODE_IS_CLOSED)
        & (elevation <= config.terrain.low_elevation_outlet_max_m)
    )[0]
    grid.status_at_node[low_ids] = grid.BC_NODE_IS_FIXED_VALUE

    flow_accumulator = FlowAccumulator(
        grid,
        surface="topographic__elevation",
        flow_director=config.terrain.flow_director,
        runoff_rate=None,
        depression_finder=config.terrain.depression_finder,
    )
    drainage_area, discharge = flow_accumulator.accumulate_flow()

    add_or_update_field(
        grid,
        "topographic__slope",
        grid.at_node["topographic__steepest_slope"],
        at="node",
    )
    add_or_update_field(
        grid,
        "topographic__specific_contributing_area",
        drainage_area / grid.dx,
        at="node",
    )

    core_elevation = elevation[grid.core_nodes]
    return TerrainState(
        grid=grid,
        low_outlet_node_ids=low_ids,
        drainage_area=drainage_area,
        discharge=discharge,
        elevation_min_core=float(np.min(core_elevation)),
        elevation_max_core=float(np.max(core_elevation)),
    )
