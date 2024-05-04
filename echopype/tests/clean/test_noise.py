import numpy as np
import xarray as xr
import echopype as ep
import pytest
from numpy.random import default_rng


@pytest.mark.integration
def test_mask_attenuated_noise_value_errors():
    """Test `mask_attenuated_noise` values errors."""
    # Parse and calibrate
    ed = ep.open_raw(
        "echopype/test_data/ek60/from_echopy/JR161-D20061118-T010645.raw",
        sonar_model="EK60"
    )
    ds_Sv = ep.calibrate.compute_Sv(ed)

    # Attempt to create masks without depth
    r0, r1, n, threshold = 180, 280, 30, -6 # units: (m, m, pings, dB)
    with pytest.raises(ValueError):
        ep.clean.mask_attenuated_noise(
            ds_Sv,
            r0,
            r1,
            n,
            threshold
        )
    
    # Add depth
    ds_Sv = ep.consolidate.add_depth(ds_Sv)

    # Attempt to create mask with `r0 > r1`
    with pytest.raises(ValueError):
        ep.clean.mask_attenuated_noise(
            ds_Sv,
            r0=180,
            r1=170,
            n=n,
            threshold=threshold
        )


@pytest.mark.integration
def test_mask_attenuated_noise_outside_searching_range():
    """Test `mask_attenuated_noise` values errors."""
    # Parse, calibrate, and add_depth
    ed = ep.open_raw(
        "echopype/test_data/ek60/from_echopy/JR161-D20061118-T010645.raw",
        sonar_model="EK60"
    )
    ds_Sv = ep.calibrate.compute_Sv(ed)
    ds_Sv = ep.consolidate.add_depth(ds_Sv)

    # Create masks with out of bounds r0 and r1 values
    r0, r1, n, threshold = 1800, 2800, 30, -6 # units: (m, m, pings, dB)
    attenuated_mask, unfeasible_mask = ep.clean.mask_attenuated_noise(
        ds_Sv,
        r0,
        r1,
        n,
        threshold
    )
    
    # Check outputs
    assert attenuated_mask.dims == unfeasible_mask.dims == ds_Sv["Sv"].dims
    assert np.allclose(attenuated_mask, xr.zeros_like(ds_Sv["Sv"], dtype=bool))
    assert np.allclose(unfeasible_mask, xr.full_like(ds_Sv["Sv"], True, dtype=bool))


@pytest.mark.integration
@pytest.mark.parametrize(
    ("chunk"),
    [
        (False),
        (True),
    ],
)
def test_mask_attenuated_noise_against_echopy(chunk):
    """Test `attenuated_noise` to see if Echopype output matches echopy output masks."""
    # Parse, calibrate, and add depth
    ed = ep.open_raw(
        "echopype/test_data/ek60/from_echopy/JR161-D20061118-T010645.raw",
        sonar_model="EK60"
    )
    ds_Sv = ep.calibrate.compute_Sv(ed)
    ds_Sv = ep.consolidate.add_depth(ds_Sv)

    if chunk:
        # Chunk dataset
        ds_Sv = ds_Sv.chunk("auto")

    # Create masks
    r0, r1, n, threshold = 180, 280, 30, -6 # units: (m, m, pings, dB)
    attenuated_mask, unfeasible_mask = ep.clean.mask_attenuated_noise(
        ds_Sv,
        r0,
        r1,
        n,
        threshold
    )

    # Grab echopy attenuated masks
    echopy_attenuated_masks = xr.open_dataset(
        "echopype/test_data/ek60/from_echopy/JR161-D20061118-T010645_echopy_attenuated_masks.zarr",
        engine="zarr"
    )

    # Check that Echopype masks match echopy masks
    assert np.allclose(
        echopy_attenuated_masks["attenuated_mask"],
        attenuated_mask.isel(channel=0).transpose("range_sample", "ping_time")
    )
    assert np.allclose(
        echopy_attenuated_masks["unfeasible_mask"],
        unfeasible_mask.isel(channel=0).transpose("range_sample", "ping_time")
    )


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
