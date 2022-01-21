import pytest

import numpy as np

from echopype.utils.uwa import calc_absorption, calc_sound_speed


# Tolerance used here are set empirically
# so the test captures a sort of current status
@pytest.mark.parametrize(
    "frequency, temperature, salinity, pressure, pH, tolerance",
    [
        (18e3, 27, 35, 10, 8, 3e-5),
        (18e3, 27, 35, 100, 8, 3e-5),
        (38e3, 27, 35, 10, 8, 2.1e-4),
        (38e3, 10, 35, 10, 8, 2.1e-4),
        (120e3, 27, 35, 10, 8, 3e-5),
        (200e3, 27, 35, 10, 8, 3.1e-3),
        (455e3, 20, 35, 10, 8, 7.4e-3),
        (1e6, 10, 35, 10, 8, 2.49e-2),
    ],
)
def test_absorption(frequency, temperature, salinity, pressure, pH, tolerance):
    abs_dB_m = dict()
    for fm in ["AM", "FG", "AZFP"]:
        abs_dB_m[fm] = calc_absorption(
            frequency=frequency,
            temperature=temperature,
            salinity=salinity,
            pressure=pressure,
            pH=pH,
            formula_source=fm,
        )
    assert np.abs(abs_dB_m["AM"] - abs_dB_m["FG"]) < tolerance


# These tests are just to make sure the code runs
# and the difference between formula is not too dramatic,
# instead of trying to verify the actual numbers
# as we don't know the source of the AZFP formula.
@pytest.mark.parametrize(
    "temperature, salinity, pressure, tolerance",
    [
        (27, 35, 10, 0.07),
        (27, 35, 100, 0.07),
        (5, 35, 3500, 0.5),
    ],
)
def test_sound_speed(temperature, salinity, pressure, tolerance):
    c = dict()
    for fm in ["Mackenzie", "AZFP"]:
        c[fm] = calc_sound_speed(
            temperature=temperature,
            salinity=salinity,
            pressure=pressure,
            formula_source=fm
        )
    assert np.abs(c["Mackenzie"] - c["AZFP"]) < tolerance
