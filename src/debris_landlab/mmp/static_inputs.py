from __future__ import annotations

import numpy as np

from debris_landlab.mmp.config import MMPConfig
from debris_landlab.mmp.fields import add_or_update_field, load_node_field

COHESION_FIELDS = (
    "soil__minimum_total_cohesion",
    "soil__mode_total_cohesion",
    "soil__maximum_total_cohesion",
)


def update_cohesion_fields(grid, reduction_by_burn: dict[int, float] | None = None) -> None:
    burn = grid.at_node["burn__severity"].astype(int)
    multiplier = np.ones(grid.number_of_nodes, dtype=float)
    if reduction_by_burn is not None:
        for burn_class, reduction in reduction_by_burn.items():
            multiplier[burn == int(burn_class)] = 1.0 - float(reduction)

    for field in COHESION_FIELDS:
        backup_field = f"{field}_pre"
        if backup_field not in grid.at_node:
            add_or_update_field(grid, backup_field, grid.at_node[field].copy(), at="node")
        values = grid.at_node[backup_field].copy() * multiplier
        add_or_update_field(grid, field, values, at="node")


def load_static_inputs(grid, config: MMPConfig) -> None:
    """Load and derive static scenario inputs required by hydrology and landslides."""

    static_cfg = config.static_inputs
    base_dir = config.scenario_dir
    layers = static_cfg.layers
    nodata = static_cfg.field_nodata

    load_node_field(
        grid,
        base_dir,
        layers["soil__thickness"],
        "soil__thickness",
        nodata_value=nodata.get("soil__thickness"),
    )
    load_node_field(
        grid,
        base_dir,
        layers["soil__density"],
        "soil__density",
        nodata_value=nodata.get("soil__density"),
    )
    load_node_field(
        grid,
        base_dir,
        layers["soil__internal_friction_angle"],
        "soil__internal_friction_angle",
        nodata_value=nodata.get("soil__internal_friction_angle"),
    )
    load_node_field(grid, base_dir, layers["porosity"], "porosity")
    load_node_field(grid, base_dir, layers["field__capacity"], "field__capacity")
    load_node_field(grid, base_dir, layers["wilting__point"], "wilting__point")
    load_node_field(
        grid,
        base_dir,
        layers["soil__saturated_hydraulic_conductivity"],
        "soil__saturated_hydraulic_conductivity",
        nodata_value=nodata.get("soil__saturated_hydraulic_conductivity"),
    )
    load_node_field(
        grid,
        base_dir,
        layers["vegetation__plant_functional_type"],
        "vegetation__plant_functional_type",
        nodata_value=nodata.get("vegetation__plant_functional_type"),
        dtype=int,
    )
    load_node_field(
        grid,
        base_dir,
        layers["soil__maximum_total_cohesion"],
        "soil__maximum_total_cohesion",
        nodata_value=nodata.get("soil__maximum_total_cohesion"),
    )
    load_node_field(
        grid,
        base_dir,
        layers["soil__mode_total_cohesion"],
        "soil__mode_total_cohesion",
        nodata_value=nodata.get("soil__mode_total_cohesion"),
    )
    load_node_field(
        grid,
        base_dir,
        layers["soil__minimum_total_cohesion"],
        "soil__minimum_total_cohesion",
        nodata_value=nodata.get("soil__minimum_total_cohesion"),
    )
    load_node_field(grid, base_dir, layers["burn__severity"], "burn__severity")

    open_nodes = grid.status_at_node != grid.BC_NODE_IS_CLOSED
    soil_thickness = grid.at_node["soil__thickness"]
    soil_thickness[open_nodes & (soil_thickness <= 0.0)] = static_cfg.min_soil_thickness_m

    ksat = grid.at_node["soil__saturated_hydraulic_conductivity"]
    transmissivity = static_cfg.transmissivity_multiplier * ksat * soil_thickness
    transmissivity[transmissivity <= 0.0] = static_cfg.min_transmissivity_m2_per_day
    add_or_update_field(grid, "soil__transmissivity", transmissivity, at="node")

    pft = grid.at_node["vegetation__plant_functional_type"].astype(int)
    pft[pft < 0] = 0
    grid.at_node["vegetation__plant_functional_type"][:] = pft
    add_or_update_field(
        grid,
        "vegetation__plant_functional_type",
        pft[grid.node_at_cell].astype(int),
        at="cell",
    )

    default_lai = static_cfg.lai_by_pft.get(3, 1.0)
    lai = np.full(grid.number_of_nodes, default_lai, dtype=float)
    for pft_class, value in static_cfg.lai_by_pft.items():
        lai[pft == int(pft_class)] = float(value)
    add_or_update_field(grid, "vegetation__live_leaf_area_index", lai, at="node")
    add_or_update_field(
        grid,
        "vegetation__live_leaf_area_index",
        lai[grid.node_at_cell],
        at="cell",
    )
    add_or_update_field(grid, "vegetation__cover_fraction", lai / 4.0, at="node")
    add_or_update_field(
        grid,
        "vegetation__cover_fraction",
        (lai / 4.0)[grid.node_at_cell],
        at="cell",
    )

    burn = grid.at_node["burn__severity"].astype(int)
    burn[~np.isin(burn, [2, 3, 4])] = 1
    grid.at_node["burn__severity"][:] = burn

    with np.errstate(divide="ignore", invalid="ignore"):
        initial_saturation = (
            0.5 * (grid.at_node["field__capacity"] - grid.at_node["wilting__point"])
            + grid.at_node["wilting__point"]
        ) / grid.at_node["porosity"]
    initial_saturation[~np.isfinite(initial_saturation)] = 0.0
    initial_saturation = np.clip(initial_saturation, 0.0, 1.0)

    add_or_update_field(
        grid,
        "soil_moisture__initial_saturation_fraction",
        initial_saturation,
        at="node",
    )
    add_or_update_field(
        grid,
        "soil_moisture__initial_saturation_fraction",
        initial_saturation[grid.node_at_cell],
        at="cell",
    )
    add_or_update_field(
        grid,
        "saturated__hydraulic_conductivity",
        ksat[grid.node_at_cell],
        at="cell",
    )
    add_or_update_field(grid, "Porosity", grid.at_node["porosity"][grid.node_at_cell], at="cell")
    add_or_update_field(
        grid,
        "field_capacity_saturation",
        grid.at_node["field__capacity"][grid.node_at_cell],
        at="cell",
    )
    add_or_update_field(
        grid,
        "wilting_point_saturation",
        grid.at_node["wilting__point"][grid.node_at_cell],
        at="cell",
    )
    add_or_update_field(
        grid,
        "rainfall__daily_depth",
        np.zeros(grid.number_of_cells, dtype=float),
        at="cell",
    )

    update_cohesion_fields(
        grid,
        static_cfg.cohesion_reduction_by_burn if static_cfg.apply_cohesion_reduction else None,
    )
