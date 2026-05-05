from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np


def add_or_update_field(grid, name: str, values, *, at: str = "node"):
    """Add a Landlab field or overwrite an existing one in place."""

    values = np.asarray(values)
    container = grid.at_node if at == "node" else grid.at_cell
    if name in container:
        container[name][:] = values
    else:
        grid.add_field(name, values.copy(), at=at, clobber=True)
    return container[name]


def read_ascii_nodata(path: str | Path, default: float | None = None) -> float | None:
    """Read an ESRI ASCII NODATA_value header when present."""

    with Path(path).open("r") as src:
        for _ in range(6):
            line = src.readline()
            if not line:
                break
            parts = line.split()
            if parts and parts[0].lower() == "nodata_value":
                return float(parts[1])
    return default


def load_ascii_values(path: str | Path, field_name: str) -> np.ndarray:
    from landlab.io import esri_ascii

    with Path(path).open() as src:
        temp_grid = esri_ascii.load(src, name=field_name)
    return temp_grid.at_node[field_name]


def nodata_mask(values, nodata_values: float | Iterable[float] | None = None) -> np.ndarray:
    values = np.asarray(values)
    mask = ~np.isfinite(values)
    if nodata_values is None:
        return mask
    if isinstance(nodata_values, (int, float)):
        nodata_values = [float(nodata_values)]
    for nodata in nodata_values:
        mask |= np.isclose(values, float(nodata), rtol=0.0, atol=1.0e-12)
    return mask


def close_nodes_by_mask(grid, mask: np.ndarray) -> None:
    grid.status_at_node[np.asarray(mask, dtype=bool)] = grid.BC_NODE_IS_CLOSED


def load_node_field(
    grid,
    base_dir: str | Path,
    filename: str,
    field_name: str,
    *,
    nodata_value: float | None = None,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    dtype=None,
):
    """Load an ESRI ASCII file as a node field and close configured nodata nodes."""

    raw = load_ascii_values(Path(base_dir) / filename, field_name)
    values = raw.copy()
    if transform is not None:
        values = transform(values)
    if dtype is not None:
        values = values.astype(dtype)
    add_or_update_field(grid, field_name, values, at="node")
    if nodata_value is not None:
        close_nodes_by_mask(grid, nodata_mask(raw, nodata_value))
    return raw, grid.at_node[field_name]


def cell_to_node(grid, cell_values, *, fill_value: float = 0.0) -> np.ndarray:
    out = np.full(grid.number_of_nodes, fill_value, dtype=float)
    out[grid.node_at_cell] = np.asarray(cell_values)
    return out


def node_field_to_cell(grid, node_field: str, cell_field: str | None = None) -> np.ndarray:
    from landlab.grid.mappers import map_node_to_cell

    values = map_node_to_cell(grid, node_field)
    if cell_field is not None:
        add_or_update_field(grid, cell_field, values, at="cell")
    return values
