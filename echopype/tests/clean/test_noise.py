import numpy as np
import xarray as xr
import echopype as ep
import pytest

from echopype.clean.utils import calc_transient_noise_pooled_Sv, downsample_upsample_along_depth
from echopype.utils.compute import _lin2log, _log2lin


@pytest.mark.integration
def test_mask_functions_with_no_depth():
    """Test mask functions when the depth variable is not within `ds_Sv`."""
    # Open raw and calibrate
    ed = ep.open_raw(
        "echopype/test_data/ek60/from_echopy/JR230-D20091215-T121917.raw",
        sonar_model="EK60"
    )
    ds_Sv = ep.calibrate.compute_Sv(ed)

    # `depth` is not contained in `ds_Sv`. Ensure that `ValueError` is raised
    # for all masking functions.
    with pytest.raises(ValueError):
        ep.clean.mask_attenuated_signal(ds_Sv)
    with pytest.raises(ValueError):
        ep.clean.mask_impulse_noise(ds_Sv)
    with pytest.raises(ValueError):
        ep.clean.mask_transient_noise(ds_Sv)


@pytest.mark.integration
def test_mask_functions_dimensions():
    """Test mask functions' output dimensions."""
    # Open raw, calibrate, and add depth
    ed = ep.open_raw(
        "echopype/test_data/ek60/from_echopy/JR230-D20091215-T121917.raw",
        sonar_model="EK60"
    )
    ds_Sv = ep.calibrate.compute_Sv(ed)
    ds_Sv = ep.consolidate.add_depth(ds_Sv)

    # Select Sv subset
    ds_Sv = ds_Sv.isel(ping_time=slice(294, 300), range_sample=slice(1794,1800))

    # Compute masks and check that dimensions match `ds_Sv`
    attenuated_signal_mask = ep.clean.mask_attenuated_signal(ds_Sv)
    impulse_noise_mask = ep.clean.mask_impulse_noise(ds_Sv)
    transient_noise_mask = ep.clean.mask_transient_noise(ds_Sv)
    for mask in [attenuated_signal_mask, impulse_noise_mask, transient_noise_mask]:
        assert ds_Sv["channel"].equals(mask["channel"])
        assert np.allclose(ds_Sv["range_sample"].data, mask["range_sample"].data)
        assert ds_Sv["ping_time"].equals(mask["ping_time"])


@pytest.mark.integration
def test_transient_mask_noise_func_error_and_warnings(caplog):
    """Check if appropriate warnings and errors are raised for transient noise mask func input."""
    # Open raw, calibrate, and add depth
    ed = ep.open_raw(
        "echopype/test_data/ek60/from_echopy/JR230-D20091215-T121917.raw",
        sonar_model="EK60"
    )
    ds_Sv = ep.calibrate.compute_Sv(ed)
    ds_Sv = ep.consolidate.add_depth(ds_Sv)

    # Select Sv subset
    ds_Sv = ds_Sv.isel(ping_time=slice(290, 300), range_sample=slice(1794,1800))

    # Set window args
    depth_bin="0.2m"
    num_side_pings=2
    exclude_above = 250

    ### Check for `nanmedian` warning:

    # Turn on logger verbosity
    ep.utils.log.verbose(override=False)

    # Compute transient noise mask
    ep.clean.mask_transient_noise(
        ds_Sv,
        depth_bin=depth_bin,
        num_side_pings=num_side_pings,
        exclude_above=exclude_above,
        func="nanmedian",
    )

    # Check for expected warning
    expected_warning = (
        "Consider using `func=nanmean`. `func=nanmedian` is an incredibly slow operation due to "
        "the overhead sorting."
    )
    assert any(expected_warning in record.message for record in caplog.records)

    # Turn off logger verbosity
    ep.utils.log.verbose(override=True)

    # Check for func value error:
    with pytest.raises(ValueError, match="Input `func` is `nanmode`. `func` must be `nanmean` or `nanmedian`."):
        ep.clean.mask_transient_noise(
            ds_Sv,
            depth_bin=depth_bin,
            num_side_pings=num_side_pings,
            exclude_above=exclude_above,
            func="nanmode",
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("chunk", "func"),
    [
        (False, "nanmean"),
        (True, "nanmean"),
        (False, "nanmedian"),
        (True, "nanmedian"),
    ],
)
def test_calc_transient_noise_pooled_Sv_values(chunk, func):
    """
    Manually check if the pooled Sv for transient noise masking contains 
    the correct nan boundary and the correct bin aggregate values.
    """
    # Open raw, calibrate, and add depth
    ed = ep.open_raw(
        "echopype/test_data/ek60/from_echopy/JR230-D20091215-T121917.raw",
        sonar_model="EK60"
    )
    ds_Sv = ep.calibrate.compute_Sv(ed)
    ds_Sv = ep.consolidate.add_depth(ds_Sv)

    # Select Sv subset
    ds_Sv = ds_Sv.isel(ping_time=slice(294, 300), range_sample=slice(1794,1800))

    if chunk:
        # Chunk calibrated Sv
        ds_Sv = ds_Sv.chunk("auto")

    # Set window args
    depth_bin = 0.2 # depth values ~0.2m apart per range sample
    num_side_pings=2
    exclude_above = 250

    # Compute pooled Sv
    pooled_Sv = calc_transient_noise_pooled_Sv(
        ds_Sv, func, depth_bin, num_side_pings, exclude_above
    ).compute()

    # Compute min and max values
    depth_values_min = ds_Sv["depth"].min()
    depth_values_max = ds_Sv["depth"].max()
    ping_time_index_min = 0
    ping_time_index_max = len(ds_Sv["ping_time"])

    # Create ping time indices array
    ping_time_indices = xr.DataArray(
        np.arange(len(ds_Sv["ping_time"]), dtype=int),
        dims=["ping_time"],
        coords=[ds_Sv["ping_time"]],
        name="ping_time_indices",
    )

    # Check appropriate NaN boundaries
    within_mask = (
        (ds_Sv["depth"] - depth_bin >= depth_values_min) &
        (ds_Sv["depth"] + depth_bin <= depth_values_max) &
        (ds_Sv["depth"] - depth_bin >= exclude_above) &
        (ping_time_indices - num_side_pings >= ping_time_index_min) &
        (ping_time_indices + num_side_pings <= ping_time_index_max)
    )
    unique_pool_boundary_values = np.unique(
        pooled_Sv.where(~within_mask)
    )

    # Check that NaN is the only pool boundary unique value
    assert np.isclose(unique_pool_boundary_values, np.array([np.nan]), equal_nan=True)

    # Compute Sv
    ds_Sv["Sv"] = ds_Sv["Sv"].compute()

    # Manually check correct binning and aggregation values
    assert np.isclose(
        pooled_Sv.isel(channel=0, ping_time=2, range_sample=2).data,
        _lin2log(
            _log2lin(ds_Sv["Sv"].isel(channel=0, ping_time=slice(0, 5), range_sample=slice(1, 4))).pipe(
                np.nanmean if func == "nanmean" else np.nanmedian
            )
        ),
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.isclose(
        pooled_Sv.isel(channel=1, ping_time=3, range_sample=3).data,
        _lin2log(
            _log2lin(ds_Sv["Sv"].isel(channel=1, ping_time=slice(1, 6), range_sample=slice(2, 5))).pipe(
                np.nanmean if func == "nanmean" else np.nanmedian
            )
        ),
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.isclose(
        pooled_Sv.isel(channel=2, ping_time=4, range_sample=3).data,
        _lin2log(
            _log2lin(ds_Sv["Sv"].isel(channel=2, ping_time=slice(2, 7), range_sample=slice(2, 5))).pipe(
                np.nanmean if func == "nanmean" else np.nanmedian
            )
        ),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("chunk"),
    [
        (False),
        (True),
    ],
)
def test_downsample_upsample_along_depth(chunk):
    """Test downsample bins and upsample repeating values"""
    # Open raw, calibrate, and add depth
    ed = ep.open_raw(
        "echopype/test_data/ek60/from_echopy/JR230-D20091215-T121917.raw",
        sonar_model="EK60"
    )
    ds_Sv = ep.calibrate.compute_Sv(ed)
    ds_Sv = ep.consolidate.add_depth(ds_Sv)

    # Subset `ds_Sv`
    ds_Sv = ds_Sv.isel(ping_time=slice(100, 120), range_sample=slice(0, 100))

    if chunk:
        # Chunk calibrated Sv
        ds_Sv = ds_Sv.chunk("auto")

    # Run downsampling and upsampling
    downsampled_Sv, upsampled_Sv = downsample_upsample_along_depth(ds_Sv, 2)

    # Compute DataArrays
    downsampled_Sv = downsampled_Sv.compute()
    upsampled_Sv = upsampled_Sv.compute()
    original_resolution_depth = ds_Sv["depth"].compute()

    # Check for appropriate binning behavior
    # Test every depth bin
    for depth_bin_index in range(len(downsampled_Sv["depth_bins"])):
        # Test every channel
        for channel_index in range(len(downsampled_Sv["channel"])):
            # Test every ping time
            for ping_time_index in range(len(downsampled_Sv["ping_time"])):
                # Check that manual and flox downsampled bin Sv are equal
                flox_downsampled_bin_Sv = downsampled_Sv.isel(
                    channel=channel_index, ping_time=ping_time_index, depth_bins=depth_bin_index
                ).data
                flox_downsampled_bin_Sv_indices = np.where(
                    upsampled_Sv.isel(channel=channel_index, ping_time=ping_time_index).data == flox_downsampled_bin_Sv
                )[0]
                manual_downsampled_bin_Sv = _lin2log(
                    np.nanmean(
                        ds_Sv["Sv"].compute().isel(
                        channel=channel_index, ping_time=ping_time_index, range_sample=flox_downsampled_bin_Sv_indices
                        ).pipe(_log2lin)
                    )
                ).data
                assert np.isclose(manual_downsampled_bin_Sv, flox_downsampled_bin_Sv)

                # Check that depth bins encapsulated the correct original resolution depth values
                manual_depth_array = original_resolution_depth.isel(
                    channel=channel_index, ping_time=ping_time_index, range_sample=flox_downsampled_bin_Sv_indices
                ).data
                flox_depth_bin = downsampled_Sv["depth_bins"].data[depth_bin_index]
                for manual_depth in manual_depth_array:
                    if not np.isnan(manual_depth):
                        assert flox_depth_bin.left <= manual_depth < flox_depth_bin.right


@pytest.mark.integration
@pytest.mark.parametrize(
    ("chunk"),
    [
        (False),
        (True),
    ],
)
def test_impulse_noise_mask_values(chunk):
    """Manually check if impulse noise mask removes impulse noise values."""
    # Open raw, calibrate, and add depth
    ed = ep.open_raw(
        "echopype/test_data/ek60/from_echopy/JR230-D20091215-T121917.raw",
        sonar_model="EK60"
    )
    ds_Sv = ep.calibrate.compute_Sv(ed)
    ds_Sv = ep.consolidate.add_depth(ds_Sv)

    if chunk:
        # Chunk calibrated Sv
        ds_Sv = ds_Sv.chunk("auto")

    # Subset `ds_Sv`
    ds_Sv = ds_Sv.isel(ping_time=slice(100, 120), range_sample=slice(0, 100))

    # Create impulse noise mask
    impulse_noise_mask = ep.clean.mask_impulse_noise(ds_Sv, "2m")

    # Compute upsampled data
    _, upsampled_Sv = downsample_upsample_along_depth(ds_Sv, 2)
    upsampled_Sv = upsampled_Sv.compute()

    # Remove impulse noise from Sv
    ds_Sv["upsampled_Sv_cleaned_of_impulse_noise"] = xr.where(
        impulse_noise_mask,
        np.nan,
        upsampled_Sv
    ).compute()

    # Iterate through all channels
    for channel_index in range(len(ds_Sv["channel"])):
        # Iterate through every 50th range sample
        for range_sample_index in range(0, len(ds_Sv["range_sample"]), 50):
            # Iterate through every third ping time
            for ping_time_index in range(1, len(ds_Sv["ping_time"]) - 1, 3):
                # Grab range sample row array
                row_array = ds_Sv["upsampled_Sv_cleaned_of_impulse_noise"].isel(
                    channel=channel_index,
                    ping_time=slice(ping_time_index - 1, ping_time_index + 2),
                    range_sample=range_sample_index
                ).data
                # Compute left and right subtraction values
                left_subtracted_value = row_array[1] - row_array[0]
                right_subtracted_value = row_array[1] - row_array[2]
                # Check negation of impulse condition if middle array and subtraction values are not NaN
                if not (
                    np.isnan(row_array[1]) or np.isnan(left_subtracted_value) or np.isnan(right_subtracted_value)
                ):
                    assert (row_array[1] - row_array[0] <= 10.0 or row_array[1] - row_array[2] <= 10.0)


@pytest.mark.integration
def test_mask_attenuated_signal_limit_error():
    """Test `mask_attenuated_signal` limit error."""
    # Parse, calibrate, and add depth
    ed = ep.open_raw(
        "echopype/test_data/ek60/from_echopy/JR161-D20061118-T010645.raw",
        sonar_model="EK60"
    )
    ds_Sv = ep.calibrate.compute_Sv(ed)
    ds_Sv = ep.consolidate.add_depth(ds_Sv)

    # Attempt to create mask with `upper_limit_sl > lower_limit_sl`
    with pytest.raises(ValueError):
        ep.clean.mask_attenuated_signal(
            ds_Sv,
            upper_limit_sl=180,
            lower_limit_sl=170,
        )


@pytest.mark.integration
def test_mask_attenuated_signal_outside_searching_range():
    """Test `mask_attenuated_signal` values errors."""
    # Parse, calibrate, and add_depth
    ed = ep.open_raw(
        "echopype/test_data/ek60/from_echopy/JR161-D20061118-T010645.raw",
        sonar_model="EK60"
    )
    ds_Sv = ep.calibrate.compute_Sv(ed)
    ds_Sv = ep.consolidate.add_depth(ds_Sv)

    # Create mask
    upper_limit_sl, lower_limit_sl, num_pings, attenuation_signal_threshold = 1800, 2800, 30, -6 # units: (m, m, pings, dB)
    attenuated_mask = ep.clean.mask_attenuated_signal(
        ds_Sv,
        upper_limit_sl,
        lower_limit_sl,
        num_pings,
        attenuation_signal_threshold
    )
    
    # Check outputs
    assert np.allclose(attenuated_mask, xr.zeros_like(ds_Sv["Sv"], dtype=bool))


@pytest.mark.integration
@pytest.mark.parametrize(
    ("chunk"),
    [
        (False),
        (True),
    ],
)
def test_mask_attenuated_signal_against_echopy(chunk):
    """Test `attenuated_signal` to see if Echopype output matches echopy output mask."""
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

    # Create mask
    upper_limit_sl, lower_limit_sl, num_pings, attenuation_signal_threshold = 180, 280, 30, -6 # units: (m, m, pings, dB)
    attenuated_mask = ep.clean.mask_attenuated_signal(
        ds_Sv,
        upper_limit_sl,
        lower_limit_sl,
        num_pings,
        attenuation_signal_threshold
    )

    # Grab echopy attenuated signal mask
    echopy_attenuated_mask = xr.open_dataset(
        "echopype/test_data/ek60/from_echopy/JR161-D20061118-T010645_echopy_attenuated_masks.zarr",
        engine="zarr"
    )

    # Check that Echopype 38kHz mask matches echopy mask
    assert np.allclose(
        echopy_attenuated_mask["attenuated_mask"],
        attenuated_mask.isel(channel=0).transpose("range_sample", "ping_time")
    )


def test_remove_background_noise():
    """Test remove_background_noise on toy data"""

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
    ds_Sv = ep.clean.remove_background_noise(
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
    # Make Dataset to pass into remove_background_noise
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
    ds_Sv = ep.clean.remove_background_noise(
        ds_Sv, ping_num=2, range_sample_num=5, SNR_threshold=0
    )
    null = ds_Sv.Sv_corrected.isnull()
    # Test to see if the right number of points are removed before the range gets too large
    assert (
        np.count_nonzero(null.isel(channel=0, range_sample=slice(None, 50)))
        == 6
    )


def test_remove_background_noise_no_sound_absorption():
    """
    Tests remove_background_noise on toy data that does
    not have sound absorption as a variable.
    """

    pytest.xfail(f"Tests for remove_background_noise have not been implemented" +
                 " when no sound absorption is provided!")
