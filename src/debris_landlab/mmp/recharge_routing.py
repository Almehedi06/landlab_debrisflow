from __future__ import annotations

import numpy as np
from landlab.components import FlowAccumulator, SinkFillerBarnes


def route_recharge_field(
    grid,
    local_recharge,
    *,
    surface: str = "topographic__elevation",
    flow_director: str = "FlowDirectorD8",
    depression_finder: str = "DepressionFinderAndRouter",
    fill_sinks: bool = True,
    min_recharge: float = 1.0e-4,
) -> np.ndarray:
    """Route local recharge to an upslope-area-averaged recharge field."""

    recharge = np.asarray(local_recharge, dtype=float).copy()
    recharge[~np.isfinite(recharge)] = 0.0
    recharge[recharge <= 0.0] = min_recharge
    grid.add_field("water__unit_flux_in", recharge, at="node", clobber=True)

    if fill_sinks:
        sink_filler = SinkFillerBarnes(
            grid,
            surface,
            method="D8",
            fill_flat=True,
            ignore_overfill=False,
        )
        sink_filler.run_one_step()

    flow_accumulator = FlowAccumulator(
        grid,
        surface=surface,
        flow_director=flow_director,
        depression_finder=depression_finder,
    )
    drainage_area, _ = flow_accumulator.accumulate_flow()

    drainage_area_safe = drainage_area.copy()
    drainage_area_safe[drainage_area_safe <= 0.0] = grid.dx * grid.dy
    routed_recharge = grid.at_node["surface_water__discharge"] / drainage_area_safe

    grid.add_field("routed_recharge", routed_recharge, at="node", clobber=True)
    grid.add_field("diff_recharge", routed_recharge - recharge, at="node", clobber=True)
    return routed_recharge
