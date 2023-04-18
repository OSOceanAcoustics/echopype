import numpy as np
import xarray as xr
import echopype as ep
import pytest
from numpy.random import default_rng


def test_remove_noise():
    """Test remove_noise on toy data"""

    # Parameters for fake data
    nchan, npings, nrange_samples = 1, 10, 100
    chan = np.arange(nchan).astype(str)
    ping_index = np.arange(npings)
    range_sample = np.arange(nrange_samples)
    data = np.ones(nrange_samples)

    # Insert noise points
    np.put(data, 30, -30)
    np.put(data, 60, -30)
    # Add more pings
    data = np.array([data] * npings)
    # Make DataArray
    Sv = xr.DataArray(
        [data],
        coords=[
            ('channel', chan),
            ('ping_time', ping_index),
            ('range_sample', range_sample),
        ],
    )
    Sv.name = "Sv"
    ds_Sv = Sv.to_dataset()

    ds_Sv = ds_Sv.assign(
        echo_range=xr.DataArray(
            np.array([[np.linspace(0, 10, nrange_samples)] * npings]),
            coords=Sv.coords,
        )
    )
    ds_Sv = ds_Sv.assign(sound_absorption=0.001)
    # Run noise removal
    ds_Sv = ep.clean.remove_noise(
        ds_Sv, ping_num=2, range_sample_num=5, SNR_threshold=0
    )

    # Test if noise points are nan
    assert np.isnan(
        ds_Sv.Sv_corrected.isel(channel=0, ping_time=0, range_sample=30)
    )
    assert np.isnan(
        ds_Sv.Sv_corrected.isel(channel=0, ping_time=0, range_sample=60)
    )

    # Test remove noise on a normal distribution
    np.random.seed(1)
    data = np.random.normal(
        loc=-100, scale=2, size=(nchan, npings, nrange_samples)
    )
    # Make Dataset to pass into remove_noise
    Sv = xr.DataArray(
        data,
        coords=[
            ('channel', chan),
            ('ping_time', ping_index),
            ('range_sample', range_sample),
        ],
    )
    Sv.name = "Sv"
    ds_Sv = Sv.to_dataset()
    # Attach required echo_range and sound_absorption values
    ds_Sv = ds_Sv.assign(
        echo_range=xr.DataArray(
            np.array([[np.linspace(0, 3, nrange_samples)] * npings]),
            coords=Sv.coords,
        )
    )
    ds_Sv = ds_Sv.assign(sound_absorption=0.001)
    # Run noise removal
    ds_Sv = ep.clean.remove_noise(
        ds_Sv, ping_num=2, range_sample_num=5, SNR_threshold=0
    )
    null = ds_Sv.Sv_corrected.isnull()
    # Test to see if the right number of points are removed before the range gets too large
    assert (
        np.count_nonzero(null.isel(channel=0, range_sample=slice(None, 50)))
        == 6
    )


def test_remove_noise_no_sound_absorption():
    """
    Tests remove_noise on toy data that does
    not have sound absorption as a variable.
    """

    pytest.xfail(f"Tests for remove_noise have not been implemented" +
                 " when no sound absorption is provided!")
