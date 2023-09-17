import pytest
from echopype.convert.utils.ek import COMPLEX_VAR, _get_power_dims

@pytest.fixture()
def dgram_zarr_vars():
    return {'power': ['timestamp', 'channel'], 'angle': ['timestamp', 'channel']}

def test__get_power_dims(dgram_zarr_vars):
    power_dims = _get_power_dims(dgram_zarr_vars)
    assert isinstance(power_dims, list)
    assert sorted(power_dims) == sorted(['timestamp', 'channel'])

def test__extract_datagram_dfs():
    ...