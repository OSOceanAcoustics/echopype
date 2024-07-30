import numpy as np
import xarray as xr
import echopype as ep
import pytest
import pandas as pd

from echopype.clean.utils import (
    extract_dB,
    pool_Sv,
    index_binning_pool_Sv,
    index_binning_downsample_upsample_along_depth,
    downsample_upsample_along_depth,
)
from echopype.utils.compute import _lin2log, _log2lin


@pytest.mark.unit
def test_extract_dB():
    """Test for correct outputs and errors of `extract_dB`."""
    # Check correct values
    assert extract_dB("10dB") == 10.0
    assert extract_dB("-10dB") == -10.0
    assert extract_dB("10.0dB") == 10.0
    assert extract_dB("-10.0dB") == -10.0
    assert extract_dB("10db") == 10.0 # b doesn't need to be capitalized
    # Check for invalid string
    with pytest.raises(ValueError):
        extract_dB("10")
    with pytest.raises(ValueError):
        extract_dB("10 db")
    with pytest.raises(ValueError):
        extract_dB("10db1")
    # Check invalid type
    with pytest.raises(TypeError):
        extract_dB(10)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("range_var"),
    [
        ("depth"),
        ("echo_range"),
    ],
)
def test_mask_functions_with_no_vertical_range_variables(range_var):
    """Test mask functions when no vertical range variables are in `ds_Sv`."""
    # Open raw and calibrate
    ed = ep.open_raw(
        "echopype/test_data/ek60/from_echopy/JR230-D20091215-T121917.raw",
        sonar_model="EK60"
    )
    ds_Sv = ep.calibrate.compute_Sv(ed)
    
    if range_var == "echo_range":
        # Drop `echo_range`
        ds_Sv = ds_Sv.drop_vars("echo_range")

    # `range_var` is not contained in `ds_Sv`. Ensure that `ValueError` is raised
    # for all masking functions.
    with pytest.raises(ValueError):
        ep.clean.mask_attenuated_signal(ds_Sv, range_var=range_var)
    with pytest.raises(ValueError):
        ep.clean.mask_impulse_noise(ds_Sv, range_var=range_var)
    with pytest.raises(ValueError):
        ep.clean.mask_transient_noise(ds_Sv, range_var=range_var)


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
    depth_bin = "0.2m"
    num_side_pings = 2
    exclude_above = "250m"

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
        "`func=nanmedian` is an incredibly slow operation due to the overhead sorting. "
        "We plan to add the Fielding Transient Noise Filter in the future"
        "described here: https://github.com/OSOceanAcoustics/echopype/issues/1352"
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
        (False, np.nanmean),
        (True, np.nanmean),
        (False, np.nanmedian),
        (True, np.nanmedian),
    ],
)
def test_pool_Sv_values(chunk, func):
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
    range_var = "depth"

    # Compute pooled Sv
    pooled_Sv = pool_Sv(
        ds_Sv, func, depth_bin, num_side_pings, exclude_above, range_var,
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

    # Check correct binning and aggregation values
    for channel_index in range(len(ds_Sv["channel"])):
        for ping_time_index in range(len(ds_Sv["ping_time"])):
            for range_sample_index in range(len(ds_Sv["range_sample"])):
                # Grab pooled value
                pooled_value = pooled_Sv.isel(
                    channel=channel_index,
                    ping_time=ping_time_index,
                    range_sample=range_sample_index
                ).data
                if not np.isnan(pooled_value):
                    assert np.isclose(
                        pooled_value,
                        # Compute pooled value using the fact that the depth bin covers
                        # 3 depth values (1 below, middle, 1 above) and that the ping bin
                        # covers 5 ping values (2 below, middle, 2 above).
                        _lin2log(
                            _log2lin(
                                ds_Sv["Sv"].isel(
                                    channel=channel_index,
                                    ping_time=slice(ping_time_index-2, ping_time_index+3),
                                    range_sample=slice(range_sample_index-1, range_sample_index+2)
                                )
                            ).pipe(func)
                        ),
                        rtol=1e-10,
                        atol=1e-10,
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
def test_transient_noise_mask_values(chunk, func):
    """Manually check if impulse noise mask removes transient noise values."""
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
    depth_bin = "0.2m" # depth values ~0.2m apart per range sample
    num_side_pings = 2
    exclude_above = "250m"
    transient_noise_threshold_str = "12dB"
    transient_noise_threshold = 12

    # Compute transient noise mask
    transient_noise_mask = ep.clean.mask_transient_noise(
        ds_Sv, func, depth_bin, num_side_pings, exclude_above, transient_noise_threshold_str
    ).compute()

    # Remove transient noise from Sv
    ds_Sv["Sv_cleaned_of_transient_noise"] = xr.where(
        transient_noise_mask,
        np.nan,
        ds_Sv["Sv"]
    ).compute()

    # Compute Sv
    ds_Sv["Sv"] = ds_Sv["Sv"].compute()

    # Check if transient noise values have been removed:
    for channel_index in range(len(ds_Sv["channel"])):
        for ping_time_index in range(len(ds_Sv["ping_time"])):
            for range_sample_index in range(len(ds_Sv["range_sample"])):
                # Grab cleaned value
                Sv_value = ds_Sv["Sv_cleaned_of_transient_noise"].isel(
                    channel=channel_index,
                    ping_time=ping_time_index,
                    range_sample=range_sample_index
                ).data
                # Compute pooled value using the fact that the depth bin covers
                # 3 depth values (1 below, middle, 1 above) and that the ping bin
                # covers 5 ping values (2 below, middle, 2 above).
                pooled_value = _lin2log(
                    _log2lin(
                        ds_Sv["Sv"].isel(
                            channel=channel_index,
                            ping_time=slice(ping_time_index-2, ping_time_index+3),
                            range_sample=slice(range_sample_index-1, range_sample_index+2)
                        )
                    ).pipe(
                        np.nanmean if func == "nanmean" else np.nanmedian
                    )
                )
                # Check negation of transient noise condition only when both values are not NaN:
                if not np.isnan(Sv_value) and not np.isnan(pooled_value):
                    assert Sv_value - pooled_value <= transient_noise_threshold


@pytest.mark.integration
@pytest.mark.parametrize(
    ("chunk", "func"),
    [
        (False, np.nanmean),
        (True, np.nanmean),
        (False, np.nanmedian),
        (True, np.nanmedian),
    ],
)
def test_index_binning_pool_Sv_values(chunk, func):
    """
    Manually check if the index binning pooled Sv for transient noise masking does the
    correct reflection computation. This is tested using `np.pad` to extend the Sv boundary
    via reflecting around the existing Sv edge values, and doing the windowing aggregation
    based off of this extended Sv.
    """
    # Open raw, calibrate, and add depth
    ed = ep.open_raw("echopype/test_data/ek60/from_echopy/JR161-D20061118-T010645.raw", sonar_model="EK60")
    ds_Sv = ep.calibrate.compute_Sv(ed)
    ds_Sv = ep.consolidate.add_depth(ds_Sv)

    # Select Sv subset
    ds_Sv = ds_Sv.isel(ping_time=slice(100,120), range_sample=slice(990,1020))

    if chunk:
        # Chunk calibrated Sv
        ds_Sv = ds_Sv.chunk("auto")

    # Set window args
    depth_bin = 1
    num_side_pings = 2
    exclude_above = 186
    range_var = "depth"
    chunk_dict = {}

    # Compute pooled Sv using index binning
    pooled_Sv = index_binning_pool_Sv(
        ds_Sv, func, depth_bin, num_side_pings, exclude_above, range_var, chunk_dict
    ).compute()

    # Compute `ds_Sv` prior to using it for manual testing
    ds_Sv = ds_Sv.compute()

    # Iterate through channels
    for channel_index in range(len(ds_Sv["channel"])):
        # Grab single channel Sv
        chan_Sv = ds_Sv.isel(channel=channel_index)
        chan_pooled_Sv = pooled_Sv.isel(channel=channel_index)

        # Compute number of range sample indices that are needed to encapsulate the `depth_bin`
        # value per channel.
        num_range_sample_indices = np.ceil(
            depth_bin / np.nanmean(np.diff(chan_Sv["depth"], axis=1), axis=(0,1))
        ).astype(int)

        # Remove values close to top
        min_range_sample = (chan_Sv[range_var] <= exclude_above).argmin().values
        chan_Sv = chan_Sv.isel(range_sample=slice(min_range_sample, None))
        chan_pooled_Sv = chan_pooled_Sv.isel(range_sample=slice(min_range_sample, None))

        # Pad `chan_ds_Sv` using symmetric (equivalent to Dask Generic Filter reflect)
        padded_chan_Sv = chan_Sv.pad(
            pad_width={
                "ping_time": (
                    num_side_pings,
                    num_side_pings
                ),
                "range_sample": (
                    num_range_sample_indices,
                    num_range_sample_indices
                )
            },
            mode="symmetric"
        )

        # Check correct binning and aggregation values
        for ping_time_index in range(
            num_side_pings,
            len(padded_chan_Sv["ping_time"]) - num_side_pings
        ):
            for range_sample_index in range(
                num_range_sample_indices,
                len(padded_chan_Sv["range_sample"]) - num_range_sample_indices
            ): 
                # Grab pooled value
                pooled_value = chan_pooled_Sv.isel(
                    ping_time=ping_time_index - num_side_pings,
                    range_sample=range_sample_index - num_range_sample_indices
                ).data
                # Check that manually computed pool value matches Dask-Image's
                # generic filter output
                assert np.allclose(
                    pooled_value,
                    _lin2log(
                        _log2lin(
                            padded_chan_Sv["Sv"].isel(
                                ping_time=slice(
                                    ping_time_index - num_side_pings,
                                    ping_time_index + num_side_pings + 1
                                ),
                                range_sample=slice(
                                    range_sample_index - num_range_sample_indices,
                                    range_sample_index + num_range_sample_indices + 1
                                )
                            )
                        ).pipe(func)
                    ),
                    rtol=1e-10,
                    atol=1e-10,
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
def test_index_binning_transient_noise_mask_values(chunk, func):
    """Manually check if impulse noise mask removes transient noise values when using index binning."""
    # Open raw, calibrate, and add depth
    ed = ep.open_raw("echopype/test_data/ek60/from_echopy/JR161-D20061118-T010645.raw", sonar_model="EK60")
    ds_Sv = ep.calibrate.compute_Sv(ed)
    ds_Sv = ep.consolidate.add_depth(ds_Sv)

    # Select Sv subset
    ds_Sv = ds_Sv.isel(ping_time=slice(100,120), range_sample=slice(990,1020))

    if chunk:
        # Chunk calibrated Sv
        ds_Sv = ds_Sv.chunk("auto")

    # Set window args
    depth_bin_str = "1m"
    depth_bin = 1
    num_side_pings = 2
    exclude_above = "186m"
    transient_noise_threshold_str = "12dB"
    transient_noise_threshold = 12

    # Compute transient noise mask
    transient_noise_mask = ep.clean.mask_transient_noise(
        ds_Sv,
        func,
        depth_bin_str,
        num_side_pings,
        exclude_above,
        transient_noise_threshold_str,
        use_index_binning=True
    ).compute()

    # Compute `ds_Sv` prior to using it for manual testing
    ds_Sv = ds_Sv.compute()

    # Remove transient noise from Sv
    ds_Sv["Sv_cleaned_of_transient_noise"] = xr.where(
        transient_noise_mask,
        np.nan,
        ds_Sv["Sv"]
    )

    # Check if transient noise values have been removed:
    for channel_index in range(len(ds_Sv["channel"])):
        # Grab single channel Sv
        chan_ds_Sv = ds_Sv.isel(channel=channel_index)

        # Compute number of range sample indices that are needed to encapsulate the `depth_bin`
        # value per channel.
        num_range_sample_indices = np.ceil(
            depth_bin / np.nanmean(np.diff(chan_ds_Sv["depth"], axis=1), axis=(0,1))
        ).astype(int)

        # Iterate through ping time indices
        for ping_time_index in range(len(ds_Sv["ping_time"])):
            # Iterate through range sample indices
            for range_sample_index in range(len(ds_Sv["range_sample"])):
                # Grab cleaned value
                Sv_value = ds_Sv["Sv_cleaned_of_transient_noise"].isel(
                    channel=channel_index,
                    ping_time=ping_time_index,
                    range_sample=range_sample_index
                ).data
                # Manually compute pooled value
                pooled_value = _lin2log(
                    _log2lin(
                        ds_Sv["Sv"].isel(
                            channel=channel_index,
                            ping_time=slice(
                                ping_time_index - num_side_pings,
                                ping_time_index + 1 + num_side_pings
                            ),
                            range_sample=slice(
                                range_sample_index - num_range_sample_indices,
                                range_sample_index + 1 + num_range_sample_indices
                            )
                        )
                    ).pipe(
                        np.nanmean if func == "nanmean" else np.nanmedian
                    )
                )
                # Check negation of transient noise condition only when both values are not NaN:
                if not np.isnan(Sv_value) and not np.isnan(pooled_value):
                    assert Sv_value - pooled_value <= transient_noise_threshold


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
    downsampled_Sv, upsampled_Sv = downsample_upsample_along_depth(ds_Sv, 2, "depth")

    # Compute DataArrays
    downsampled_Sv = downsampled_Sv.compute()
    upsampled_Sv = upsampled_Sv.compute()
    original_resolution_depth = ds_Sv["depth"].compute()

    # Check for appropriate binning behavior
    # Test every channel
    for channel_index in range(len(downsampled_Sv["channel"])):
        # Test every depth bin
        for depth_bin_index in range(len(downsampled_Sv["depth_bins"])):
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
                )
                assert np.isclose(manual_downsampled_bin_Sv, flox_downsampled_bin_Sv, atol=1e-10, rtol=1e-10)

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
def test_index_binning_downsample_upsample_along_depth(chunk):
    """Test index binning downsampled-upsampled values."""
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
    depth_bin = 2
    upsampled_Sv = index_binning_downsample_upsample_along_depth(ds_Sv, depth_bin, "depth")

    # Compute DataArray
    upsampled_Sv = upsampled_Sv.compute()

    # Iterate through all channels
    for channel_index in range(len(upsampled_Sv["channel"])):
        # Compute number of range sample indices that are needed to encapsulate the `depth_bin`
        # value per channel.
        num_range_sample_indices = np.ceil(
            depth_bin / np.nanmean(
                np.diff(ds_Sv["depth"].isel(channel=0), axis=1), axis=(0,1)
            )
        ).astype(int)
        # Iterate through all ping times
        for ping_time_index in range(len(upsampled_Sv["ping_time"])):
            for range_bin_index in range(
                np.ceil(
                    len(upsampled_Sv["range_sample"]) / num_range_sample_indices
                ).astype(int)
            ):
                # Grab upsampled values
                upsampled_values = upsampled_Sv.isel(
                    channel=channel_index,
                    ping_time=ping_time_index,
                    range_sample=slice(
                        num_range_sample_indices * range_bin_index,
                        num_range_sample_indices * (range_bin_index + 1)
                    )
                ).data
                # Manually compute bin value
                manual_value = _lin2log(
                    np.nanmean(
                        ds_Sv["Sv"].isel(
                            channel=channel_index,
                            ping_time=ping_time_index,
                            range_sample=slice(
                                num_range_sample_indices * range_bin_index,
                                num_range_sample_indices * (range_bin_index + 1)
                            )
                        ).pipe(_log2lin)
                    )
                )
                assert np.isclose(
                    np.unique(upsampled_values),
                    manual_value,
                    atol=1e-10,
                    rtol=1e-10,
                )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("chunk, use_index_binning"),
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_impulse_noise_mask_values(chunk, use_index_binning):
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
    depth_bin = 2
    depth_bin_str = "2m"
    num_side_pings = 2
    impulse_noise_threshold_str = "10.0dB"
    impulse_noise_threshold = 10.0
    range_var = "depth"
    impulse_noise_mask = ep.clean.mask_impulse_noise(
        ds_Sv,
        depth_bin_str,
        num_side_pings,
        impulse_noise_threshold_str,
        range_var,
        use_index_binning
    ).compute()

    # Compute upsampled data
    if not use_index_binning:
        _, upsampled_Sv = downsample_upsample_along_depth(ds_Sv, depth_bin, range_var)
    else:
        upsampled_Sv = index_binning_downsample_upsample_along_depth(ds_Sv, depth_bin, range_var)
    upsampled_Sv = upsampled_Sv.compute()

    # Remove impulse noise from Sv
    ds_Sv["upsampled_Sv_cleaned_of_impulse_noise"] = xr.where(
        impulse_noise_mask,
        np.nan,
        upsampled_Sv
    )

    # Iterate through every channel
    for channel_index in range(len(ds_Sv["channel"])):
        # Iterate through every range sample
        for range_sample_index in range(len(ds_Sv["range_sample"])):
            # Iterate through every valid ping time
            for ping_time_index in range(
                num_side_pings,
                len(ds_Sv["ping_time"]) - num_side_pings
            ):
                # Grab range sample row array
                row_array = ds_Sv["upsampled_Sv_cleaned_of_impulse_noise"].isel(
                    channel=channel_index,
                    ping_time=slice(
                        ping_time_index - num_side_pings,
                        ping_time_index + num_side_pings + 1
                    ),
                    range_sample=range_sample_index
                ).data
                # Compute left and right subtraction values
                left_subtracted_value = row_array[num_side_pings] - row_array[0]
                right_subtracted_value = row_array[num_side_pings] - row_array[num_side_pings * 2]
                # Check negation of impulse condition if middle array value and subtraction values are not NaN
                if not (
                    np.isnan(row_array[num_side_pings]) or np.isnan(left_subtracted_value) or np.isnan(right_subtracted_value)
                ):
                    assert (left_subtracted_value <= impulse_noise_threshold or right_subtracted_value <= impulse_noise_threshold)


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
            upper_limit_sl="180m",
            lower_limit_sl="170m",
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
    upper_limit_sl, lower_limit_sl, num_side_pings, attenuation_signal_threshold = (
        "1800m", "2800m", 15, "-6dB"
    ) # units: (m, m, pings, dB)
    attenuated_mask = ep.clean.mask_attenuated_signal(
        ds_Sv,
        upper_limit_sl,
        lower_limit_sl,
        num_side_pings,
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
    upper_limit_sl, lower_limit_sl, num_side_pings, attenuation_signal_threshold = (
        "180m", "280m", 30, "-6dB"
    ) # units: (m, m, pings, dB)
    attenuated_mask = ep.clean.mask_attenuated_signal(
        ds_Sv,
        upper_limit_sl,
        lower_limit_sl,
        num_side_pings,
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


@pytest.mark.unit
def test_estimate_background_noise_upsampling():
    """Test for the correct upsampling behavior in `estimate_background_noise`"""
    # Open Raw and Calibrate
    ed = ep.open_raw(
        "echopype/test_data/ek60/ncei-wcsd/Summer2017-D20170615-T190214.raw",
        sonar_model="EK60"
    )
    ds_Sv = ep.calibrate.compute_Sv(ed)

    # Estimate background noise
    Sv_noise = ep.clean.estimate_background_noise(
        ds_Sv, ping_num=2, range_sample_num=5,
    )

    # Check that the correct upsampling has occurred
    for channel_idx in range(len(ds_Sv["channel"])):
        for range_sample_idx in range(len(ds_Sv["range_sample"])):
            # Grab vector that just has ping time dimension
            Sv_noise_vector = Sv_noise.isel(channel=channel_idx, range_sample=range_sample_idx)

            # Check that the values repeat every 2 values:

            # Grab numpy values
            data_np = Sv_noise_vector.values

            # Handle odd length by removing the last element
            if len(data_np) % 2 != 0:
                data_np = data_np[:-1]

            # Reshape so that the pairs are columnwise next to each other
            pairs = data_np.reshape(-1, 2)

            # Assert that the pairs are equal
            assert np.allclose(pairs[:,0], pairs[:,1])


@pytest.mark.integration
def test_remove_background_noise():
    """Test remove_background_noise on toy data"""

    # Parameters for fake data
    nchan, npings, nrange_samples = 1, 10, 100
    chan = np.arange(nchan).astype(str)
    ping_time = pd.date_range(
        start='2015-06-30T17:30:10.000000000',
        end='2015-06-30T17:30:25.000000000',
        periods=npings,
    )
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
            ('ping_time', ping_time),
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
        ds_Sv, ping_num=2, range_sample_num=5, SNR_threshold="0dB"
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
            ('ping_time', ping_time),
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
        ds_Sv, ping_num=2, range_sample_num=5, SNR_threshold="0dB"
    )
    null = ds_Sv.Sv_corrected.isnull()
    # Test to see if the right number of points are removed before the range gets too large
    assert (
        np.count_nonzero(null.isel(channel=0, range_sample=slice(None, 50)))
        == 6
    )
