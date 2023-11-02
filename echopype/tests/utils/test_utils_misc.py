import pytest

import numpy as np
from echopype.utils.misc import depth_from_pressure


def test_depth_from_pressure():
    # A single pressure value and defaults for the other arguments
    pressure = 100.0
    depth = depth_from_pressure(pressure)
    assert np.isclose(depth, 99.2954)

    # Array of pressure and list of latitude values
    pressure = np.array([100.0, 101.0, 101.0])
    latitude = [0.0, 30.0, 50.0]
    depth = depth_from_pressure(pressure, latitude)
    assert np.allclose(depth, [99.4265, 100.2881, 100.1096])

    # Scalars specified for all 3 arguments
    pressure = 1000.0
    latitude = 0.0
    atm_pres_surf = 10.1325  # standard atm pressure at sea level
    depth = depth_from_pressure(pressure, latitude, atm_pres_surf)
    assert np.isclose(depth, 982.0882)

    # ValueError triggered by argument arrays having different lengths
    pressure = np.array([100.0, 101.0, 101.0])
    latitude = [0.0, 30.0]
    with pytest.raises(ValueError) as excinfo:
        depth = depth_from_pressure(pressure, latitude)
    assert str(excinfo.value) == "Sequence shape or size does not match pressure"
