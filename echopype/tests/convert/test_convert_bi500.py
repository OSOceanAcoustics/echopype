import numpy as np
import pytest

from echopype import open_raw

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def bi500_output(test_path):
    """Open BI500 test data once for all tests in this module."""
    bi500_path = test_path["BI500"]

    return open_raw(
        raw_file=str(bi500_path),
        sonar_model="BI500",
    )


def test_bi500_echodata_structure(bi500_output):
    """Verify the BI500 EchoData structure and provenance."""
    echodata, _ = bi500_output

    assert echodata.sonar_model == "BI500"
    assert len(echodata["Sonar/Beam_group1"].channel) == 1

    beam = echodata["Sonar/Beam_group1"]

    # BI500 values are already calibrated and must not be exposed
    # as raw backscatter samples.
    assert "backscatter_r" not in beam
    assert "backscatter_r_bottom" not in beam

    assert echodata["Platform"].ping_time.shape == (3323,)

    assert echodata["Provenance"].nation_code.values == 31
    assert echodata["Provenance"].ship_code.values == 445
    assert echodata["Provenance"].survey_code.values == 2000008


def test_bi500_calibrated_echograms(bi500_output):
    """Verify calibrated BI500 Sv products."""
    _, ds_cal = bi500_output

    assert "Sv" in ds_cal
    assert "Sv_bottom" in ds_cal

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


def test_bi500_echo_range(bi500_output):
    """Verify reconstructed BI500 echo-range products."""
    _, ds_cal = bi500_output

    echo_range = ds_cal["echo_range"]
    echo_range_bottom = ds_cal["echo_range_bottom"]

    assert echo_range.dims == (
        "channel",
        "ping_time",
        "range_sample",
    )
    assert echo_range_bottom.dims == (
        "channel",
        "ping_time",
        "range_sample_bottom",
    )

    assert echo_range.shape == ds_cal["Sv"].shape
    assert echo_range_bottom.shape == ds_cal["Sv_bottom"].shape

    assert echo_range.attrs["units"] == "m"
    assert echo_range_bottom.attrs["units"] == "m"

    assert np.all(
        np.diff(
            echo_range.isel(channel=0).values,
            axis=1,
        )
        > 0
    )
    assert np.all(
        np.diff(
            echo_range_bottom.isel(channel=0).values,
            axis=1,
        )
        > 0
    )


def test_bi500_single_targets(bi500_output):
    """Verify BI500 single-target products and ping linkage."""
    _, ds_cal = bi500_output

    assert ds_cal.sizes["single_target"] == 8724

    required_target_variables = [
        "single_target_identifier",
        "ping_index",
        "single_target_ping_time",
        "single_target_range",
        "single_target_alongship_angle",
        "single_target_athwartship_angle",
        "uncompensated_TS",
        "compensated_TS",
    ]

    for variable in required_target_variables:
        assert variable in ds_cal
        assert ds_cal[variable].dims == ("single_target",)
        assert ds_cal[variable].shape == (8724,)

    assert np.isfinite(ds_cal["single_target_range"].values).any()
    assert np.isfinite(ds_cal["uncompensated_TS"].values).any()
    assert np.isfinite(ds_cal["compensated_TS"].values).any()

    assert ds_cal["single_target_range"].attrs["units"] == "m"
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