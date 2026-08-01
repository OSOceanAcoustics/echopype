import numpy as np
import pytest

from echopype import open_raw

pytestmark = pytest.mark.integration


def test_convert_bi500(test_path):
    """Verify BI500 conversion returns EchoData and calibrated products."""
    bi500_path = test_path["BI500"]

    echodata, ds_cal = open_raw(
        raw_file=str(bi500_path),
        sonar_model="BI500",
    )

    # EchoData structure
    assert echodata.sonar_model == "BI500"
    assert len(echodata["Sonar/Beam_group1"].channel) == 1

    beam = echodata["Sonar/Beam_group1"]

    # BI500 values are already calibrated and must not be exposed
    # as raw backscatter samples.
    assert "backscatter_r" not in beam
    assert "backscatter_r_bottom" not in beam

    # Calibrated echogram products
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

    # Single-target products
    assert ds_cal.sizes["single_target"] == 8724

    required_target_variables = [
        "single_target_identifier",
        "ping_index",
        "single_target_ping_time",
        "single_target_range",
        "single_target_alongship_angle",
        "single_target_athwartship_angle",
        "Sp",
        "TS",
    ]

    for variable in required_target_variables:
        assert variable in ds_cal
        assert ds_cal[variable].dims == ("single_target",)
        assert ds_cal[variable].shape == (8724,)

    assert np.isfinite(ds_cal["single_target_range"].values).any()
    assert np.isfinite(ds_cal["Sp"].values).any()
    assert np.isfinite(ds_cal["TS"].values).any()

    assert ds_cal["single_target_range"].attrs["units"] == "m"
    assert ds_cal["single_target_alongship_angle"].attrs["units"] == "arc_degree"
    assert ds_cal["single_target_athwartship_angle"].attrs["units"] == "arc_degree"
    assert ds_cal["Sp"].attrs["units"] == "dB"
    assert ds_cal["TS"].attrs["units"] == "dB"

    # Check target-to-ping linkage
    ping_index = ds_cal["ping_index"].values

    assert np.issubdtype(ping_index.dtype, np.integer)
    assert ping_index.min() >= 0
    assert ping_index.max() < ds_cal.sizes["ping_time"]

    np.testing.assert_array_equal(
        ds_cal["single_target_ping_time"].values,
        ds_cal["ping_time"].values[ping_index],
    )

    # General coordinates and metadata
    assert echodata["Platform"].ping_time.shape == (3323,)
    assert ds_cal.channel.values[0] == "BI500-F11990-T01"
    assert ds_cal["frequency_nominal"].shape == (1,)

    assert ds_cal.attrs["source_sonar_model"] == "BI500"
    assert ds_cal.attrs["processing_function"] == "open_raw"

    # Provenance
    assert echodata["Provenance"].nation_code.values == 31
    assert echodata["Provenance"].ship_code.values == 445
    assert echodata["Provenance"].survey_code.values == 2000008