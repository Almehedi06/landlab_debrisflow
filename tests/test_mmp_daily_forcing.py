from types import SimpleNamespace

import numpy as np
import pytest

from debris_landlab.mmp import daily_forcing


def test_build_daily_forcing_reads_prepared_csv(tmp_path, monkeypatch):
    asc_dir = tmp_path / "prism_forcing" / "asc"
    ppt_path = asc_dir / "ppt" / "precip_20250101.asc"
    tmin_path = asc_dir / "tmin" / "tmin_20250101.asc"
    tmax_path = asc_dir / "tmax" / "tmax_20250101.asc"
    for path in (ppt_path, tmin_path, tmax_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    forcing_csv = tmp_path / "prism_forcing" / "forcing_daily_prism.csv"
    forcing_csv.write_text(
        "datetime,ppt_asc_path,tmin_asc_path,tmax_asc_path\n"
        "2025-01-01,asc/ppt/precip_20250101.asc,asc/tmin/tmin_20250101.asc,"
        "asc/tmax/tmax_20250101.asc\n"
    )

    arrays = {
        "Precipitation": np.array([5.0, -999999.0]),
        "Temperature_min": np.array([-999999.0, 1.0]),
        "Temperature_max": np.array([2.0, 4.0]),
    }
    monkeypatch.setattr(daily_forcing, "load_ascii_values", lambda _path, field: arrays[field])
    monkeypatch.setattr(daily_forcing, "read_ascii_nodata", lambda _path: -999999.0)

    config = SimpleNamespace(
        scenario_dir=tmp_path,
        forcing=SimpleNamespace(
            start_date="2025-01-01",
            end_date="2025-01-01",
            forcing_csv=forcing_csv,
            manifest_path=forcing_csv.parent / "prism_manifest.json",
            sanitize_nodata=True,
            temperature_fill_method="mean",
        ),
    )
    grid = SimpleNamespace(number_of_nodes=2)

    result = daily_forcing.build_daily_forcing(config, grid)

    np.testing.assert_allclose(result.rainfall_arrays[0], [5.0, 0.0])
    np.testing.assert_allclose(result.tempmin_arrays[0], [1.0, 1.0])
    np.testing.assert_allclose(result.tempmax_arrays[0], [2.0, 4.0])
    assert result.forcing_csv == forcing_csv
    assert result.asc_dir == asc_dir


def test_build_daily_forcing_reports_missing_prepared_csv(tmp_path):
    config = SimpleNamespace(
        scenario_dir=tmp_path,
        forcing=SimpleNamespace(
            start_date="2025-01-01",
            end_date="2025-01-01",
            forcing_csv=tmp_path / "missing.csv",
            manifest_path=None,
            sanitize_nodata=True,
            temperature_fill_method="mean",
        ),
    )
    grid = SimpleNamespace(number_of_nodes=2)

    with pytest.raises(FileNotFoundError, match="Run build-prism-forcing first"):
        daily_forcing.build_daily_forcing(config, grid)
