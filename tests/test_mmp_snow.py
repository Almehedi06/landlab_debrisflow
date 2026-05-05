import numpy as np

from debris_landlab.mmp.snow import build_swe_and_water_input, partition_precipitation_phase


def test_partition_precipitation_phase_linear_transition():
    precipitation = [np.array([10.0, 10.0, 10.0])]
    tmin = [np.array([-5.0, 0.0, 5.0])]
    tmax = [np.array([-3.0, 2.0, 7.0])]

    rain, snow, snow_fraction = partition_precipitation_phase(
        precipitation,
        tmin,
        tmax,
        t_snow_c=-1.0,
        t_rain_c=3.0,
    )

    np.testing.assert_allclose(snow_fraction[0], [1.0, 0.5, 0.0])
    np.testing.assert_allclose(snow[0], [10.0, 5.0, 0.0])
    np.testing.assert_allclose(rain[0], [0.0, 5.0, 10.0])


def test_build_swe_and_water_input_carries_snowpack():
    rain = [np.array([0.0]), np.array([0.0])]
    snow = [np.array([10.0]), np.array([0.0])]
    tmin = [np.array([-5.0]), np.array([1.0])]
    tmax = [np.array([-3.0]), np.array([3.0])]

    water_input, swe, melt = build_swe_and_water_input(
        1,
        rain,
        snow,
        tmin,
        tmax,
        melt_factor_mm_per_c_day=2.0,
        melt_base_temp_c=0.0,
    )

    np.testing.assert_allclose(water_input[0], [0.0])
    np.testing.assert_allclose(swe[0], [10.0])
    np.testing.assert_allclose(melt[1], [4.0])
    np.testing.assert_allclose(water_input[1], [4.0])
    np.testing.assert_allclose(swe[1], [6.0])
