import numpy as np
import pytest

from echopype import open_raw
from echopype.convert.parse_bi500 import ParseBI500

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def bi500_output(test_path):
    """Open BI500 test data once for all tests in this module."""
    bi500_path = test_path["BI500"]

    return open_raw(
        raw_file=str(bi500_path),
        sonar_model="BI500",
    )


def test_bi500_invalid_file_path(tmp_path):
    """Verify that an error is raised when the folder lacks required BI500 files."""
    invalid_path = tmp_path / "not_bi500_folder"
    invalid_path.mkdir()
    (invalid_path / "readme.txt").write_text("not bi500 data")

    with pytest.raises(ValueError, match="Expecting a folder"):
        open_raw(
            raw_file=str(invalid_path),
            sonar_model="BI500",
        )


def test_bi500_echodata_structure(bi500_output):
    """Verify the BI500 EchoData structure and provenance."""
    echodata, _ = bi500_output
    beam = echodata["Sonar/Beam_group1"]

    assert echodata.sonar_model == "BI500"
    assert len(beam.channel) == 1

    # BI500 values are already calibrated and must not be exposed
    # as raw backscatter samples.
    assert "backscatter_r" not in beam
    assert "backscatter_r_bottom" not in beam

    assert echodata["Platform"].ping_time.shape == (3323,)

    assert echodata["Provenance"].nation_code.values == 31
    assert echodata["Provenance"].ship_code.values == 445
    assert echodata["Provenance"].survey_code.values == 2000008


def test_bi500_beam_group_dimensions(bi500_output):
    """Verify Sonar/Beam_group1 dimensions and metadata variables."""
    echodata, _ = bi500_output
    beam = echodata["Sonar/Beam_group1"]

    assert beam.sizes["channel"] == 1
    assert beam.sizes["ping_time"] == 3323
    assert beam.sizes["ping_time_vlog"] > 0

    assert beam["frequency_nominal"].item() == 11990.0
    assert beam["transceiver_channel_number"].item() == 1
    # TODO: Add beam_type once it can be populated from the BI500 data
    assert "beam_type" not in beam
    assert beam["channel"].item() == "BI500-F11990-T01"

    ping_time_vars = [
        "echogram_type",
        "pelagic_upper",
        "pelagic_lower",
        "bottom_upper",
        "bottom_lower",
    ]
    for variable in ping_time_vars:
        assert variable in beam
        if variable == "echogram_type":
            assert beam[variable].dims == ("ping_time",)
            assert beam[variable].shape == (3323,)
        else:
            assert beam[variable].dims == ("channel", "ping_time")
            assert beam[variable].shape == (1, 3323)

    vlog_size = beam.sizes["ping_time_vlog"]
    vlog_vars = [
        "echogram_type_vlog",
        "pelagic_upper_vlog",
        "pelagic_lower_vlog",
        "bottom_upper_vlog",
        "bottom_lower_vlog",
    ]
    for variable in vlog_vars:
        assert variable in beam
        if variable == "echogram_type_vlog":
            assert beam[variable].dims == ("ping_time_vlog",)
            assert beam[variable].shape == (vlog_size,)
        else:
            assert beam[variable].dims == ("channel", "ping_time_vlog")
            assert beam[variable].shape == (1, vlog_size)


def test_bi500_platform_group(bi500_output):
    """Verify Platform group dimensions and vlog alignment."""
    echodata, _ = bi500_output
    platform = echodata["Platform"]
    beam = echodata["Sonar/Beam_group1"]

    assert platform.sizes["ping_time"] == 3323
    assert platform.sizes["ping_time_vlog"] == beam.sizes["ping_time_vlog"]

    ping_time_vars = [
        "latitude",
        "longitude",
        "bottom_depth",
        "vessel_log_distance",
    ]
    for variable in ping_time_vars:
        assert variable in platform
        assert platform[variable].dims == ("ping_time",)
        assert platform[variable].shape == (3323,)

    vlog_size = platform.sizes["ping_time_vlog"]
    vlog_vars = [
        "latitude_vlog",
        "longitude_vlog",
        "bottom_depth_vlog",
        "vessel_log_distance_vlog",
    ]
    for variable in vlog_vars:
        assert variable in platform
        assert platform[variable].dims == ("ping_time_vlog",)
        assert platform[variable].shape == (vlog_size,)

    np.testing.assert_array_equal(
        platform["ping_time"].values,
        beam["ping_time"].values,
    )


def test_bi500_environment_and_vendor_groups(bi500_output):
    """Verify Environment and Vendor_specific group contents."""
    echodata, _ = bi500_output

    environment = echodata["Environment"]
    assert environment.sizes == {"channel": 1}
    assert environment["absorption_indicative"].dims == ("channel",)
    assert environment["absorption_indicative"].shape == (1,)
    assert np.isnan(environment["absorption_indicative"].item())
    assert np.isnan(environment["sound_speed_indicative"].values)

    vendor = echodata["Vendor_specific"]
    for variable in [
        "start_latitude",
        "start_longitude",
        "start_distance",
        "stop_latitude",
        "stop_longitude",
        "stop_distance",
    ]:
        assert variable in vendor
        assert vendor[variable].dims == ()
        assert np.isfinite(vendor[variable].values)


def test_bi500_sonar_metadata(bi500_output):
    """Verify Sonar group metadata attributes."""
    echodata, _ = bi500_output
    sonar = echodata["Sonar"]

    assert sonar.attrs["sonar_manufacturer"] == "Bergen Integrator"
    assert sonar.attrs["sonar_model"] == "BI500"
    assert sonar.attrs["sonar_software_name"] == "BI500"
    assert sonar.attrs["sonar_type"] == "echosounder"
    assert isinstance(sonar.attrs["sonar_software_version"], str)
    assert sonar.attrs["sonar_software_version"]


def test_bi500_calibrated_coordinate_alignment(bi500_output):
    """Verify calibrated dataset coordinates align with EchoData groups."""
    echodata, ds_cal = bi500_output
    beam = echodata["Sonar/Beam_group1"]
    platform = echodata["Platform"]

    np.testing.assert_array_equal(ds_cal["ping_time"].values, platform["ping_time"].values)
    np.testing.assert_array_equal(ds_cal["ping_time"].values, beam["ping_time"].values)
    np.testing.assert_array_equal(ds_cal["channel"].values, beam["channel"].values)

    assert ds_cal.sizes["range_sample"] == 500
    assert ds_cal.sizes["range_sample_bottom"] == 150
    np.testing.assert_array_equal(ds_cal["range_sample"].values, np.arange(500))
    np.testing.assert_array_equal(ds_cal["range_sample_bottom"].values, np.arange(150))

    window_vars = [
        "pelagic_upper",
        "pelagic_lower",
        "bottom_upper",
        "bottom_lower",
    ]
    for variable in window_vars:
        assert ds_cal[variable].dims == ("channel", "ping_time")
        assert ds_cal[variable].shape == (1, 3323)
        np.testing.assert_array_equal(
            ds_cal[variable].values,
            beam[variable].values,
        )


def test_bi500_calibrated_echograms(bi500_output):
    """Verify calibrated BI500 Sv products."""
    _, ds_cal = bi500_output

    assert {"Sv", "Sv_bottom"}.issubset(ds_cal.data_vars)

    sv = ds_cal["Sv"]
    sv_bottom = ds_cal["Sv_bottom"]

    assert sv.dims == (
        "channel",
        "ping_time",
        "range_sample",
    )
    assert sv_bottom.dims == (
        "channel",
        "ping_time",
        "range_sample_bottom",
    )

    assert sv.shape == (1, 3323, 500)
    assert sv_bottom.shape == (1, 3323, 150)

    assert np.isfinite(sv.values).any()
    assert np.isfinite(sv_bottom.values).any()

    assert sv.attrs["units"] == "dB"
    assert sv_bottom.attrs["units"] == "dB"

    assert ds_cal.channel.values[0] == "BI500-F11990-T01"
    assert ds_cal["frequency_nominal"].shape == (1,)

    assert ds_cal.attrs["source_sonar_model"] == "BI500"
    assert ds_cal.attrs["processing_function"] == "open_raw"


def test_bi500_depth(bi500_output):
    """Verify reconstructed BI500 depth products."""
    _, ds_cal = bi500_output

    depth = ds_cal["depth"]
    depth_bottom = ds_cal["depth_bottom"]

    assert depth.shape == ds_cal["Sv"].shape
    assert depth_bottom.shape == ds_cal["Sv_bottom"].shape

    assert depth.attrs["units"] == "m"
    assert depth.attrs["positive"] == "down"

    assert depth_bottom.attrs["units"] == "m"
    assert depth_bottom.attrs["positive"] == "down"

    assert np.all(
        np.diff(
            depth.isel(channel=0).values,
            axis=1,
        )
        > 0
    )
    assert np.all(
        np.diff(
            depth_bottom.isel(channel=0).values,
            axis=1,
        )
        > 0
    )


def test_bi500_depth_within_window_bounds(bi500_output):
    """Verify reconstructed depths stay within BI500 window bounds."""
    echodata, ds_cal = bi500_output

    depth = ds_cal["depth"].isel(channel=0)
    pelagic_upper = ds_cal["pelagic_upper"].isel(channel=0)
    pelagic_lower = ds_cal["pelagic_lower"].isel(channel=0)

    assert np.all(depth.min(dim="range_sample") >= pelagic_upper)
    assert np.all(depth.max(dim="range_sample") <= pelagic_lower)

    bottom_depth = echodata["Platform"]["bottom_depth"]
    bottom_upper = ds_cal["bottom_upper"].isel(channel=0)
    bottom_lower = ds_cal["bottom_lower"].isel(channel=0)
    depth_bottom = ds_cal["depth_bottom"].isel(channel=0)

    window_start = bottom_depth - bottom_upper
    window_stop = bottom_depth - bottom_lower

    assert np.all(depth_bottom.min(dim="range_sample_bottom") >= window_start)
    assert np.all(depth_bottom.max(dim="range_sample_bottom") <= window_stop)


def test_bi500_single_targets(bi500_output):
    """Verify BI500 single-target products and ping linkage."""
    _, ds_cal = bi500_output

    assert ds_cal.sizes["single_target"] == 8724

    required_target_variables = [
        "single_target_identifier",
        "ping_index",
        "single_target_ping_time",
        "single_target_depth",
        "single_target_alongship_angle",
        "single_target_athwartship_angle",
        "uncompensated_TS",
        "compensated_TS",
    ]

    for variable in required_target_variables:
        assert variable in ds_cal
        assert ds_cal[variable].dims == ("single_target",)
        assert ds_cal[variable].shape == (8724,)

    assert np.isfinite(ds_cal["single_target_depth"].values).any()
    assert np.isfinite(ds_cal["uncompensated_TS"].values).any()
    assert np.isfinite(ds_cal["compensated_TS"].values).any()

    assert ds_cal["single_target_depth"].attrs["units"] == "m"
    assert ds_cal["single_target_depth"].attrs["positive"] == "down"
    assert ds_cal["single_target_alongship_angle"].attrs["units"] == "arc_degree"
    assert ds_cal["single_target_athwartship_angle"].attrs["units"] == "arc_degree"
    assert ds_cal["uncompensated_TS"].attrs["units"] == "dB"
    assert ds_cal["compensated_TS"].attrs["units"] == "dB"

    ping_index = ds_cal["ping_index"].values

    assert np.issubdtype(ping_index.dtype, np.integer)
    assert ping_index.min() >= 0
    assert ping_index.max() < ds_cal.sizes["ping_time"]

    np.testing.assert_array_equal(
        ds_cal["single_target_ping_time"].values,
        ds_cal["ping_time"].values[ping_index],
    )

def test_bi500_calibrated_structure(bi500_output):
    """BI500 calibrated output follows the standard Sv dimensional structure."""
    _, ds_cal = bi500_output

    assert ds_cal["Sv"].dims == (
        "channel",
        "ping_time",
        "range_sample",
    )
    assert ds_cal["depth"].dims == ds_cal["Sv"].dims

    assert ds_cal["frequency_nominal"].dims == ("channel",)

    # BI500 provides absolute depth, not transducer-relative range
    assert "echo_range" not in ds_cal

def test_bi500_groups_multiple_channels_from_one_acquisition():
    """Verify BI500 companion files are grouped by frequency/transceiver."""
    parser = ParseBI500.__new__(ParseBI500)

    prefix = "N031-S445-S2000008"
    acquisition = "D20000921-T034142"
    channel_sets = [
        ("011990", "01"),
        ("037879", "02"),
        ("119048", "03"),
    ]

    files = [
        f"/tmp/{prefix}-F{frequency}-T{transceiver}-{acquisition}{file_type}"
        for frequency, transceiver in channel_sets
        for file_type in ("-Data", "-Info", "-Ping")
    ]

    parser._group_file_sets(files)

    assert parser.acquisition_key == (prefix, "20000921", "034142")
    assert len(parser.file_set_map) == 3

    for frequency, transceiver in channel_sets:
        file_type_map = parser.file_set_map[(frequency, transceiver)]
        assert set(file_type_map) == {"-Data", "-Info", "-Ping"}


def test_bi500_rejects_multiple_acquisitions():
    """Verify one open_raw BI500 input cannot mix acquisition periods."""
    parser = ParseBI500.__new__(ParseBI500)

    files = [
        "/tmp/N031-S445-S2000008-F011990-T01-D20000921-T034142-Data",
        "/tmp/N031-S445-S2000008-F011990-T01-D20000921-T034142-Info",
        "/tmp/N031-S445-S2000008-F011990-T01-D20000921-T034142-Ping",
        "/tmp/N031-S445-S2000008-F037879-T02-D20000922-T034142-Data",
        "/tmp/N031-S445-S2000008-F037879-T02-D20000922-T034142-Info",
        "/tmp/N031-S445-S2000008-F037879-T02-D20000922-T034142-Ping",
    ]

    with pytest.raises(ValueError, match="single acquisition"):
        parser._group_file_sets(files)