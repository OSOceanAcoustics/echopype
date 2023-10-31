import numpy as np
from echopype.utils.misc import depth_from_pressure


def test_depth_from_pressure():
    # Test with a single pressure value
    pressure = 10000.0
    latitude = 30.0
    depth = depth_from_pressure(pressure, latitude)
    assert np.isclose(depth, 9712.653)

    # Test with an array of pressure values
    pressure = np.array([10000.0, 10001.0, 10002.0])
    latitude = 30.0
    depth = depth_from_pressure(pressure, latitude)
    assert np.allclose(depth, [9712.653, 9713.604, 9714.556])

    # Test with a different latitude value
    pressure = 10000.0
    latitude = 34.0
    depth = depth_from_pressure(pressure, latitude)
    assert np.isclose(depth, 9709.439)

