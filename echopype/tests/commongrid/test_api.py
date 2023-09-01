import pytest
import echopype as ep
import numpy as np
import pandas as pd

# NASC Tests
@pytest.mark.integration
@pytest.mark.skip(reason="NASC is not implemented yet")
def test_compute_NASC(test_data_samples):
    pass

# MVBS Tests
@pytest.mark.integration
def test_compute_MVBS_index_binning(ds_Sv_er_regular, regular_data_params):
    """Test compute_MVBS_index_binning on mock data"""

    ping_num = 3  # number of pings to average over
    range_sample_num = 7  # number of range_samples to average over
    nchan = regular_data_params["channel_len"]
    npings = regular_data_params["ping_time_len"]
    nrange_samples = regular_data_params["depth_len"]

    # Binned MVBS test
    ds_MVBS = ep.commongrid.compute_MVBS_index_binning(
        ds_Sv_er_regular, range_sample_num=range_sample_num, ping_num=ping_num
    )

    # Shape test
    data_binned_shape = np.ceil(
        (nchan, npings / ping_num, nrange_samples / range_sample_num)
    ).astype(int)
    assert np.all(ds_MVBS.Sv.shape == data_binned_shape)

    # Expected values compute
    # average should be done in linear domain
    da_sv = 10 ** (ds_Sv_er_regular["Sv"] / 10)
    expected = 10 * np.log10(
        da_sv
        .coarsen(ping_time=ping_num, range_sample=range_sample_num, boundary="pad")
        .mean(skipna=True)
    )

    # Test all values in MVBS
    assert np.array_equal(ds_MVBS.Sv.data, expected.data)
    
@pytest.mark.unit
@pytest.mark.parametrize(["range_meter_bin", "ping_time_bin"], [
    (5, "10S"),
    ("10m", 10),
    ("10km", "10S"),
    ("10", "10S")
])
def test_compute_MVBS_bin_inputs_fail(ds_Sv_er_regular, range_meter_bin, ping_time_bin):
    expected_error = ValueError
    if isinstance(range_meter_bin, int) or isinstance(ping_time_bin, int):
        expected_error = TypeError
        match = r'\w+ must be a string'
    elif "m" not in range_meter_bin:
        match = r'Found incompatible units \(\w+\) *'
    elif "km" in range_meter_bin:
        match = "range_meter_bin must be less than 1 kilometer."
    
    with pytest.raises(expected_error, match=match):
        ep.commongrid.compute_MVBS(
            ds_Sv_er_regular,
            range_meter_bin=range_meter_bin,
            ping_time_bin=ping_time_bin
        )
    
@pytest.mark.unit
def test_compute_MVBS_w_latlon(ds_Sv_er_regular_w_latlon, lat_attrs, lon_attrs):
    """Testing for compute_MVBS with latitude and longitude"""
    from echopype.consolidate.api import POSITION_VARIABLES
    ds_MVBS = ep.commongrid.compute_MVBS(
        ds_Sv_er_regular_w_latlon,
        range_meter_bin="5m",
        ping_time_bin="10S"
    )
    for var in POSITION_VARIABLES:
        # Check to ensure variable is in dataset
        assert var in ds_MVBS.data_vars
        # Check for correct shape, which is just ping time
        assert ds_MVBS[var].shape == ds_MVBS.ping_time.shape
        
        # Check if attributes match
        if var == "latitude":
            assert ds_MVBS[var].attrs == lat_attrs
        elif var == "longitude":
            assert ds_MVBS[var].attrs == lon_attrs
        
    
@pytest.mark.unit
@pytest.mark.parametrize("range_var", ["my_range", "echo_range", "depth"])
def test_compute_MVBS_invalid_range_var(ds_Sv_er_regular, range_var):
    """Test compute MVBS range_var on mock data"""

    if range_var == "my_range":
        with pytest.raises(
            ValueError,
            match="range_var must be one of 'echo_range' or 'depth'."
        ):
            ep.commongrid.compute_MVBS(ds_Sv_er_regular, range_var=range_var)
    elif range_var == "depth":
        with pytest.raises(
            ValueError,
            match=f"range_var '{range_var}' does not exist in the input dataset."
        ):
            ep.commongrid.compute_MVBS(ds_Sv_er_regular, range_var=range_var)
    else:
        pass

@pytest.mark.integration
def test_compute_MVBS(test_data_samples):
    """
    Test running through from open_raw to compute_MVBS.
    """
    (
        filepath,
        sonar_model,
        azfp_xml_path,
        range_kwargs,
    ) = test_data_samples
    ed = ep.open_raw(filepath, sonar_model, azfp_xml_path)
    if ed.sonar_model.lower() == 'azfp':
        avg_temperature = ed["Environment"]['temperature'].values.mean()
        env_params = {
            'temperature': avg_temperature,
            'salinity': 27.9,
            'pressure': 59,
        }
        range_kwargs['env_params'] = env_params
        if 'azfp_cal_type' in range_kwargs:
            range_kwargs.pop('azfp_cal_type')
    Sv = ep.calibrate.compute_Sv(ed, **range_kwargs)
    ping_time_bin = "20S"
    ds_MVBS = ep.commongrid.compute_MVBS(Sv, ping_time_bin=ping_time_bin)
    assert ds_MVBS is not None
    
    # Test to see if ping_time was resampled correctly
    expected_ping_time = (
        Sv["ping_time"]
        .resample(ping_time=ping_time_bin, skipna=True)
        .asfreq()
        .indexes["ping_time"]
    )
    assert np.array_equal(ds_MVBS.ping_time.data, expected_ping_time.values)

@pytest.mark.integration
@pytest.mark.parametrize(
    ("er_type"),
    [
        ("regular"),
        ("irregular"),
    ],
)
def test_compute_MVBS_range_output(request, er_type):
    """Tests the output of compute_MVBS on regular and irregular data."""
    # set jitter=0 to get predictable number of ping within each echo_range groups
    if er_type == "regular":
        ds_Sv = request.getfixturevalue("ds_Sv_er_regular")
    else:
        ds_Sv = request.getfixturevalue("ds_Sv_er_irregular")
    
    ds_MVBS = ep.commongrid.compute_MVBS(ds_Sv, range_meter_bin="5m", ping_time_bin="10S")

    if er_type == "regular":
        expected_len = (
            ds_Sv["channel"].size,  # channel
            np.ceil(np.diff(ds_Sv["ping_time"][[0, -1]].astype(int))/1e9 / 10),  # ping_time
            np.ceil(ds_Sv["echo_range"].max()/5),  # depth
        )
        assert ds_MVBS["Sv"].shape == expected_len
    else:
        assert (
            ds_MVBS["Sv"]
            .isel(ping_time=slice(None, 3))
            .dropna(dim="echo_range").shape
        ) == (2, 3, 10)  # full array, no NaN
        assert (
            ds_MVBS["Sv"]
            .isel(ping_time=slice(3, 12))
            .dropna(dim="echo_range").shape
        ) == (2, 9, 7)  # bottom bins contain NaN
        assert (
            ds_MVBS["Sv"]
            .isel(ping_time=slice(12, None))
            .dropna(dim="echo_range").shape
        ) == (2, 6, 3)  # bottom bins contain NaN
