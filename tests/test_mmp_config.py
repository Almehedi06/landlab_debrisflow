from debris_landlab.mmp.config import load_mmp_config


def test_load_mmp_config_defaults():
    cfg = load_mmp_config("config/mmp_landslide.yaml")

    assert cfg.project_name == "pioneer_mmp_landslide"
    assert cfg.terrain.dem_filename == "topographic__elevation.asc"
    assert cfg.forcing.start_date == "2025-12-07"
    assert cfg.forcing.forcing_csv.name == "forcing_daily_prism.csv"
    assert cfg.snow.enabled is True
    assert cfg.landslide.number_of_iterations == 1000
    assert cfg.prism_output_dir.name == "prism_forcing"


def test_load_mmp_config_override():
    cfg = load_mmp_config(
        "config/mmp_landslide.yaml",
        overrides=["config/scenarios/mmp_cohesion_burnsev_reduction.yaml"],
    )

    assert cfg.project_name == "pioneer_mmp_landslide_cohesion_burnsev_reduction"
    assert cfg.static_inputs.apply_cohesion_reduction is True
    assert cfg.static_inputs.cohesion_reduction_by_burn[4] == 0.60
