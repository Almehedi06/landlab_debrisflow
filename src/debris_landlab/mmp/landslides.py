from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from landlab.components.landslides import LandslideProbability

from debris_landlab.mmp.config import LandslideConfig
from debris_landlab.mmp.fields import add_or_update_field
from debris_landlab.mmp.recharge_routing import route_recharge_field


@dataclass
class LandslideResult:
    recharge_for_landslides: np.ndarray
    recharge_std_for_landslides: np.ndarray
    landslide_component: LandslideProbability


def run_landslide_probability(
    grid,
    *,
    mean_runoff: np.ndarray,
    mean_recharge: np.ndarray,
    max_runoff: np.ndarray,
    max_recharge: np.ndarray,
    config: LandslideConfig,
) -> LandslideResult:
    """Prepare hydrologic inputs and run Landlab landslide probability."""

    if config.route_recharge:
        recharge_for_ls = route_recharge_field(grid, max_recharge, fill_sinks=False)
        add_or_update_field(grid, "routed_recharge_max", recharge_for_ls, at="node")
    else:
        recharge_for_ls = max_recharge.copy()

    recharge_for_ls = np.asarray(recharge_for_ls, dtype=float).copy()
    recharge_for_ls[~np.isfinite(recharge_for_ls)] = 0.0
    recharge_for_ls[recharge_for_ls <= 0.0] = config.recharge_floor_mm_per_day
    recharge_std_for_ls = recharge_for_ls * config.recharge_std_fraction

    add_or_update_field(grid, "groundwater__recharge_mean", recharge_for_ls, at="node")
    add_or_update_field(
        grid,
        "groundwater__recharge_standard_deviation",
        recharge_std_for_ls,
        at="node",
    )
    add_or_update_field(grid, "groundwater__runoff_mean", max_runoff, at="node")
    add_or_update_field(grid, "mean_runoff", mean_runoff, at="node")
    add_or_update_field(grid, "mean_recharge", mean_recharge, at="node")
    add_or_update_field(grid, "max_recharge", max_recharge, at="node")

    landslide_probability = LandslideProbability(
        grid,
        number_of_iterations=config.number_of_iterations,
        groundwater__recharge_distribution="lognormal_spatial",
        groundwater__recharge_mean=recharge_for_ls,
        groundwater__recharge_standard_deviation=recharge_std_for_ls,
        seed=0 if config.seed is None else int(config.seed),
    )
    landslide_probability.calculate_landslide_probability()

    return LandslideResult(
        recharge_for_landslides=recharge_for_ls,
        recharge_std_for_landslides=recharge_std_for_ls,
        landslide_component=landslide_probability,
    )
