from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _as_path(value: str | Path | None, *, base_dir: Path | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def _as_scenario_path(value: str | Path | None, *, scenario_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = scenario_dir / path
    return path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as src:
        data = yaml.safe_load(src) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _float_key_map(values: dict[Any, Any]) -> dict[int, float]:
    return {int(key): float(value) for key, value in values.items()}


@dataclass(frozen=True)
class PathConfig:
    scenario_dir: Path
    aoi_path: Path | None = None
    prism_output_dir: Path | None = None


@dataclass(frozen=True)
class TerrainConfig:
    dem_filename: str = "topographic__elevation.asc"
    dem_nodata: float = -999999.0
    low_elevation_outlet_max_m: float = 337.0
    flow_director: str = "FlowDirectorD8"
    depression_finder: str = "DepressionFinderAndRouter"


@dataclass(frozen=True)
class StaticInputConfig:
    layers: dict[str, str]
    field_nodata: dict[str, float] = field(default_factory=dict)
    min_soil_thickness_m: float = 0.01
    transmissivity_multiplier: float = 2.5
    min_transmissivity_m2_per_day: float = 0.01
    lai_by_pft: dict[int, float] = field(
        default_factory=lambda: {0: 1.5, 1: 2.0, 2: 4.0, 3: 1.0}
    )
    apply_cohesion_reduction: bool = False
    cohesion_reduction_by_burn: dict[int, float] = field(
        default_factory=lambda: {1: 0.0, 2: 0.15, 3: 0.35, 4: 0.60}
    )


@dataclass(frozen=True)
class ForcingConfig:
    start_date: str
    end_date: str
    forcing_csv: Path
    manifest_path: Path | None = None
    sanitize_nodata: bool = True
    temperature_fill_method: str = "mean"


@dataclass(frozen=True)
class SnowConfig:
    enabled: bool = True
    t_snow_c: float = -1.1
    t_rain_c: float = 3.0
    melt_factor_mm_per_c_day: float = 5.0
    melt_base_temp_c: float = 0.0


@dataclass(frozen=True)
class EcohydrologyConfig:
    latitude_deg: float = 47.7
    albedo: float = 0.2
    zveg_m: float = 0.5
    z_wind_m: float = 2.0
    vwind_mps: float = 3.04
    relative_humidity: float = 0.8
    initial_time_years: float | None = None
    storm_duration_hours: float = 24.0


@dataclass(frozen=True)
class LandslideConfig:
    number_of_iterations: int = 1000
    route_recharge: bool = True
    recharge_floor_mm_per_day: float = 0.01
    recharge_std_fraction: float = 0.1
    seed: int | None = None


@dataclass(frozen=True)
class ExportConfig:
    enabled: bool = False
    export_dir: Path | None = None
    template_tif: Path | None = None
    nodata: float = -999999.0
    fields: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class MMPConfig:
    project_name: str
    paths: PathConfig
    terrain: TerrainConfig
    static_inputs: StaticInputConfig
    forcing: ForcingConfig
    snow: SnowConfig
    ecohydrology: EcohydrologyConfig
    landslide: LandslideConfig
    export: ExportConfig
    raw: dict[str, Any] = field(repr=False)

    @property
    def scenario_dir(self) -> Path:
        return self.paths.scenario_dir

    @property
    def prism_output_dir(self) -> Path:
        return self.paths.prism_output_dir or self.forcing.forcing_csv.parent


DEFAULT_STATIC_LAYERS = {
    "soil__thickness": "soil__thickness.asc",
    "soil__density": "soil__density.asc",
    "soil__internal_friction_angle": "soil__internal_friction_angle.asc",
    "porosity": "porosity.asc",
    "field__capacity": "field__capacity.asc",
    "wilting__point": "wilting__point.asc",
    "soil__saturated_hydraulic_conductivity": "soil__saturated_hydraulic_conductivity.asc",
    "vegetation__plant_functional_type": "vegetation__plant_functional_type.asc",
    "soil__maximum_total_cohesion": "soil__maximum_total_cohesion.asc",
    "soil__mode_total_cohesion": "soil__mode_total_cohesion.asc",
    "soil__minimum_total_cohesion": "soil__minimum_total_cohesion.asc",
    "burn__severity": "burn__severity.asc",
}


DEFAULT_EXPORT_FIELDS = [
    ("groundwater__recharge_mean", "node"),
    ("groundwater__recharge_standard_deviation", "node"),
    ("groundwater__runoff_mean", "node"),
    ("mean_runoff", "node"),
    ("mean_recharge", "node"),
    ("max_recharge", "node"),
    ("routed_recharge_max", "node"),
    ("landslide__probability_of_failure", "node"),
    ("soil__probability_of_saturation", "node"),
    ("soil__mean_relative_wetness", "node"),
]


def load_mmp_config(config_path: str | Path, overrides: list[str | Path] | None = None) -> MMPConfig:
    """Load a YAML MMP workflow config and optional layered overrides."""

    path = Path(config_path).expanduser().resolve()
    data = _load_yaml(path)
    for override in overrides or []:
        data = _deep_merge(data, _load_yaml(Path(override).expanduser().resolve()))
    return parse_mmp_config(data, base_dir=path.parent)


def parse_mmp_config(data: dict[str, Any], *, base_dir: Path | None = None) -> MMPConfig:
    project = data.get("project", {})
    project_name = str(project.get("name", "mmp_landslide"))

    paths_data = data.get("paths", {})
    scenario_dir = _as_path(paths_data.get("scenario_dir"), base_dir=base_dir)
    aoi_path = _as_path(paths_data.get("aoi_path"), base_dir=base_dir)
    if scenario_dir is None:
        raise ValueError("MMP config missing paths.scenario_dir")

    prism_output_dir = _as_path(paths_data.get("prism_output_dir"), base_dir=base_dir)
    paths = PathConfig(
        scenario_dir=scenario_dir,
        aoi_path=aoi_path,
        prism_output_dir=prism_output_dir,
    )

    terrain = TerrainConfig(**data.get("terrain", {}))

    static_data = data.get("static_inputs", {})
    layers = dict(DEFAULT_STATIC_LAYERS)
    layers.update(static_data.get("layers", {}))
    static_inputs = StaticInputConfig(
        layers=layers,
        field_nodata={str(k): float(v) for k, v in static_data.get("field_nodata", {}).items()},
        min_soil_thickness_m=float(static_data.get("min_soil_thickness_m", 0.01)),
        transmissivity_multiplier=float(static_data.get("transmissivity_multiplier", 2.5)),
        min_transmissivity_m2_per_day=float(static_data.get("min_transmissivity_m2_per_day", 0.01)),
        lai_by_pft=_float_key_map(static_data.get("lai_by_pft", {0: 1.5, 1: 2.0, 2: 4.0, 3: 1.0})),
        apply_cohesion_reduction=bool(static_data.get("apply_cohesion_reduction", False)),
        cohesion_reduction_by_burn=_float_key_map(
            static_data.get("cohesion_reduction_by_burn", {1: 0.0, 2: 0.15, 3: 0.35, 4: 0.60})
        ),
    )

    forcing_data = data.get("forcing", data.get("prism", {}))
    if "start_date" not in forcing_data or "end_date" not in forcing_data:
        raise ValueError("MMP config missing forcing.start_date or forcing.end_date")

    default_forcing_dir = prism_output_dir or (scenario_dir / "prism_forcing")
    default_forcing_csv = default_forcing_dir / "forcing_daily_prism.csv"
    forcing_csv = (
        _as_scenario_path(forcing_data.get("forcing_csv"), scenario_dir=scenario_dir)
        or default_forcing_csv
    )
    manifest_path = (
        _as_scenario_path(forcing_data.get("manifest_path"), scenario_dir=scenario_dir)
        or (forcing_csv.parent / "prism_manifest.json")
    )
    forcing = ForcingConfig(
        start_date=str(forcing_data["start_date"]),
        end_date=str(forcing_data["end_date"]),
        forcing_csv=forcing_csv,
        manifest_path=manifest_path,
        sanitize_nodata=bool(forcing_data.get("sanitize_nodata", True)),
        temperature_fill_method=str(forcing_data.get("temperature_fill_method", "mean")),
    )

    snow = SnowConfig(**data.get("snow", {}))
    ecohydrology = EcohydrologyConfig(**data.get("ecohydrology", {}))
    landslide = LandslideConfig(**data.get("landslide", {}))

    export_data = data.get("export", {})
    export_dir = _as_path(export_data.get("export_dir"), base_dir=base_dir)
    template_tif = _as_path(export_data.get("template_tif"), base_dir=base_dir)
    export = ExportConfig(
        enabled=bool(export_data.get("enabled", False)),
        export_dir=export_dir,
        template_tif=template_tif,
        nodata=float(export_data.get("nodata", -999999.0)),
        fields=[tuple(item) for item in export_data.get("fields", DEFAULT_EXPORT_FIELDS)],
    )

    return MMPConfig(
        project_name=project_name,
        paths=paths,
        terrain=terrain,
        static_inputs=static_inputs,
        forcing=forcing,
        snow=snow,
        ecohydrology=ecohydrology,
        landslide=landslide,
        export=export,
        raw=data,
    )
