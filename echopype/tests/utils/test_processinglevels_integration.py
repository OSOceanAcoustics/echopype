import sys
import pytest

import numpy as np
import xarray as xr
import echopype as ep


pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Test data not available on windows tests")


@pytest.mark.xfail(
    reason=(
        "Not sure why this fails. Will list this as an issue to address "
        "in issue #1332, since this is outside of the scope of PR #1331."
    )
)
@pytest.mark.parametrize(
    ["sonar_model", "path_model", "raw_and_xml_paths", "extras"],
    [
        pytest.param(
            "EK60",
            "EK60",
            ("Winter2017-D20170115-T150122.raw", None),
            {},
            # marks=pytest.mark.skipif(sys.platform == "win32", reason="Test data not available on windows tests"),
        ),
        pytest.param(
            "AZFP",
            "AZFP",
            ("17082117.01A", "17041823.XML"),
            {"longitude": -60.0, "latitude": 45.0, "salinity": 27.9, "pressure": 59},
            # marks=pytest.mark.skipif(sys.platform == "win32", reason="Test data not available on windows tests"),
        ),
    ],
)
def test_raw_to_mvbs(
        sonar_model,
        path_model,
        raw_and_xml_paths,
        extras,
        test_path
):
    # Prepare the Sv dataset
    raw_path = test_path[path_model] / raw_and_xml_paths[0]
    if raw_and_xml_paths[1]:
        xml_path = test_path[path_model] / raw_and_xml_paths[1]
    else:
        xml_path = None

    def _presence_test(test_ds, processing_level):
        assert "processing_level" in test_ds.attrs
        assert "processing_level_url" in test_ds.attrs
        assert test_ds.attrs["processing_level"] == processing_level

    def _absence_test(test_ds):
        assert "processing_level" not in test_ds.attrs
        assert "processing_level_url" not in test_ds.attrs

    # ---- Convert raw file and update_platform
    def _var_presence_notnan_test(name):
        if name in ed['Platform'].data_vars and not ed["Platform"][name].isnull().all():
            return True
        else:
            return False

    ed = ep.open_raw(raw_path, xml_path=xml_path, sonar_model=sonar_model)
    if _var_presence_notnan_test("longitude") and _var_presence_notnan_test("latitude"):
        _presence_test(ed["Top-level"], "Level 1A")
    elif "longitude" in extras and "latitude" in extras:
        _absence_test(ed["Top-level"])
        point_ds = xr.Dataset(
            {
                "latitude": (["time"], np.array([float(extras["latitude"])])),
                "longitude": (["time"], np.array([float(extras["longitude"])])),
            },
            coords={
                "time": (["time"], np.array([ed["Sonar/Beam_group1"]["ping_time"].values.min()]))
            },
        )
        ed.update_platform(point_ds, variable_mappings={"latitude": "latitude", "longitude": "longitude"})
        _presence_test(ed["Top-level"], "Level 1A")
    else:
        _absence_test(ed["Top-level"])
        raise RuntimeError(
            "Platform latitude and longitude are not present and cannot be added "
            "using update_platform based on test raw file and included parameters."
        )

    # ---- Calibrate and add_latlon
    env_params = None
    if sonar_model == "AZFP":
        # AZFP data require external salinity and pressure
        env_params = {
            "temperature": ed["Environment"]["temperature"].values.mean(),
            "salinity": extras["salinity"],
            "pressure": extras["pressure"],
        }

    ds = ep.calibrate.compute_Sv(echodata=ed, env_params=env_params)
    _absence_test(ds)

    Sv_ds = ep.consolidate.add_location(ds=ds, echodata=ed)
    assert "longitude" in Sv_ds.data_vars and "latitude" in Sv_ds.data_vars
    _presence_test(Sv_ds, "Level 2A")

    # ---- Noise removal
    denoised_ds = ep.clean.remove_background_noise(Sv_ds, ping_num=10, range_sample_num=20)
    _presence_test(denoised_ds, "Level 2B")

    # ---- apply_mask based on frequency differencing
    def _freqdiff_applymask(test_ds):
        # frequency_differencing expects a dataarray variable named "Sv". For denoised Sv,
        # rename Sv to Sv_raw and Sv_corrected to Sv before passing ds to frequency_differencing
        if "Sv_corrected" in test_ds.data_vars:
            out_ds = test_ds.rename_vars(name_dict={"Sv": "Sv_raw", "Sv_corrected": "Sv"})
        else:
            out_ds = test_ds
        freqAB = list(out_ds.frequency_nominal.values[:2])
        freqABEq = str(freqAB[0]) + "Hz" + "-" + str(freqAB[1]) + "Hz" + ">" + str(5) + "dB"
        freqdiff_da = ep.mask.frequency_differencing(source_Sv=out_ds, freqABEq=freqABEq)

        # Apply mask to multi-channel Sv
        return ep.mask.apply_mask(source_ds=out_ds, var_name="Sv", mask=freqdiff_da)

    # On Sv w/o noise removal
    ds = _freqdiff_applymask(Sv_ds)
    _presence_test(ds, "Level 3A")

    # On denoised Sv
    ds = _freqdiff_applymask(denoised_ds)
    _presence_test(ds, "Level 3B")

    # ---- Compute MVBS
    # compute_MVBS expects a variable named "Sv"
    # ds = ds.rename_vars(name_dict={"Sv": "Sv_unmasked", "Sv_ch0": "Sv"})
    mvbs_ds = ep.commongrid.compute_MVBS(ds, range_bin="30m", ping_time_bin='1min')
    _presence_test(mvbs_ds, "Level 3B")
