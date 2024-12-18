"""test_convert_ad2cp.py

This module test conversion of two sets of .ad2cp files.
Files under "normal" contain default data variables,
whereas files under "raw" additionally contain the IQ samples.
"""
import glob
from echopype.convert import is_AD2CP

import xarray as xr
import numpy as np
import netCDF4
import pytest
from tempfile import TemporaryDirectory
from pathlib import Path
import glob

from echopype import open_raw, open_converted
from echopype.testing import TEST_DATA_FOLDER

from echopype.convert.parse_ad2cp import is_AD2CP

@pytest.fixture
def ocean_contour_export_dir(test_path):
    return test_path["AD2CP"] / "ocean-contour"


@pytest.fixture
def ocean_contour_export_076_dir(ocean_contour_export_dir):
    return ocean_contour_export_dir / "076"


@pytest.fixture
def ocean_contour_export_090_dir(ocean_contour_export_dir):
    return ocean_contour_export_dir / "090"


@pytest.fixture
def output_dir():
    return "/echopype_test-export"


def pytest_generate_tests(metafunc):
    ad2cp_path = TEST_DATA_FOLDER / "ad2cp"
    test_file_dir = (
        ad2cp_path / "normal"
    )  # "normal" files do not have IQ samples
    raw_test_file_dir = ad2cp_path / "raw"  # "raw" files contain IQ samples
    ad2cp_files = test_file_dir.glob("**/*.ad2cp")
    raw_ad2cp_files = raw_test_file_dir.glob("**/*.ad2cp")
    if "filepath" in metafunc.fixturenames:
        metafunc.parametrize(
            argnames="filepath",
            argvalues=ad2cp_files,
            ids=lambda f: str(f.name),
        )

    if "filepath_raw" in metafunc.fixturenames:
        metafunc.parametrize(
            argnames="filepath_raw",
            argvalues=raw_ad2cp_files,
            ids=lambda f: str(f.name),
        )


@pytest.fixture
def filepath(request):
    return request.param


@pytest.fixture
def filepath_raw(request):
    return request.param


@pytest.fixture
def absolute_tolerance():
    return 1e-6


def test_convert(filepath, output_dir):
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir + output_dir)
        print("converting", filepath)
        echodata = open_raw(raw_file=str(filepath), sonar_model="AD2CP")
        echodata.to_netcdf(save_path=output_dir)


def test_convert_raw(
    filepath_raw,
    output_dir,
    ocean_contour_export_090_dir,
    ocean_contour_export_076_dir,
    absolute_tolerance,
):
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir + output_dir)

        print("converting raw", filepath_raw)
        echodata = open_raw(raw_file=str(filepath_raw), sonar_model="AD2CP")
        echodata.to_netcdf(save_path=output_dir)

        _check_raw_output(
            filepath_raw,
            output_dir,
            ocean_contour_export_090_dir,
            ocean_contour_export_076_dir,
            absolute_tolerance,
        )


def _check_raw_output(
    filepath_raw,
    output_dir,
    ocean_contour_export_090_dir,
    ocean_contour_export_076_dir,
    absolute_tolerance,
):
    print("checking raw", filepath_raw)
    echodata = open_converted(
        converted_raw_path=output_dir.joinpath(
            filepath_raw.with_suffix(".nc").name
        )
    )
    if "090" in filepath_raw.parts:
        ocean_contour_converted_config_path = (
            ocean_contour_export_090_dir.joinpath(
                filepath_raw.with_suffix(
                    filepath_raw.suffix + ".00000.nc"
                ).name
            )
        )
        ocean_contour_converted_transmit_data_path = (
            ocean_contour_converted_config_path
        )
        ocean_contour_converted_data_path = ocean_contour_converted_config_path
    else:
        ocean_contour_converted_config_path = (
            ocean_contour_export_076_dir
            / filepath_raw.with_suffix("").name
            / "Raw Echo 1_1000 kHz_001.nc"
        )
        ocean_contour_converted_transmit_data_path = (
            ocean_contour_export_076_dir
            / filepath_raw.with_suffix("").name
            / "Raw Echo 1_1000 kHz Tx_001.nc"
        )
        ocean_contour_converted_data_path = (
            ocean_contour_export_076_dir
            / filepath_raw.with_suffix("").name
            / "Raw Echo 1_1000 kHz_001.nc"
        )
    if not all(
        (
            ocean_contour_converted_config_path.exists(),
            ocean_contour_converted_transmit_data_path.exists(),
            ocean_contour_converted_data_path.exists(),
        )
    ):
        pass
    else:
        # check pulse compression
        base = xr.open_dataset(
            str(ocean_contour_converted_config_path), group="Config"
        )
        pulse_compressed = 0
        for i in range(1, 4):
            if "090" in filepath_raw.parts:
                if base.attrs[f"echo_pulseComp{i}"]:
                    pulse_compressed = i
                    break
            else:
                if base.attrs[f"Instrument_echo_pulseComp{i}"]:
                    pulse_compressed = i
                    break
        for i in range(1, len(echodata["Sonar"]["beam_group"]) + 1):
            if "pulse_compressed" in echodata[f"Sonar/Beam_group{i}"]:
                pulse_compressed_vector = np.zeros(3)
                pulse_compressed_vector[pulse_compressed - 1] = 1
                assert (echodata[f"Sonar/Beam_group{i}"]["pulse_compressed"] == pulse_compressed_vector).all()
        base.close()

        # check raw data transmit samples
        try:
            netCDF4.Dataset(str(ocean_contour_converted_transmit_data_path))[
                "Data/RawEcho1_1000kHzTx"
            ]
        except IndexError:
            # no transmit data in this dataset
            pass
        else:
            base = xr.open_dataset(
                str(ocean_contour_converted_transmit_data_path),
                group="Data/RawEcho1_1000kHzTx",
            )
            if "090" in filepath_raw.parts:
                for i in range(1, len(echodata["Sonar"]["beam_group"]) + 1):
                    if "transmit_pulse_r" in echodata[f"Sonar/Beam_group{i}"]:
                        assert np.allclose(
                            echodata[f"Sonar/Beam_group{i}"][
                                "transmit_pulse_r"
                            ].data.flatten(),
                            base["DataI"].data.flatten(),
                            atol=absolute_tolerance,
                        )
                        assert np.allclose(
                            echodata[f"Sonar/Beam_group{i}"][
                                "transmit_pulse_i"
                            ].data.flatten(),
                            base["DataQ"].data.flatten(),
                            atol=absolute_tolerance,
                        )
            else:
                for i in range(1, len(echodata["Sonar"]["beam_group"]) + 1):
                    if "transmit_pulse_r" in echodata[f"Sonar/Beam_group{i}"]:
                        # note the underscore
                        assert np.allclose(
                            echodata[f"Sonar/Beam_group{i}"][
                                "transmit_pulse_r"
                            ].data.flatten(),
                            base["Data_I"].data.flatten(),
                            atol=absolute_tolerance,
                        )
                        assert np.allclose(
                            echodata[f"Sonar/Beam_group{i}"][
                                "transmit_pulse_i"
                            ].data.flatten(),
                            base["Data_Q"].data.flatten(),
                            atol=absolute_tolerance,
                        )
            base.close()

        # check raw data samples
        base = xr.open_dataset(
            str(ocean_contour_converted_data_path),
            group="Data/RawEcho1_1000kHz",
        )
        if "090" in filepath_raw.parts:
            for i in range(1, len(echodata["Sonar"]["beam_group"]) + 1):
                if "backscatter_r" in echodata[f"Sonar/Beam_group{i}"]:
                    assert np.allclose(
                        echodata[f"Sonar/Beam_group{i}"]["backscatter_r"].data.flatten(),
                        base["DataI"].data.flatten(),
                        atol=absolute_tolerance,
                    )
                    assert np.allclose(
                        echodata[f"Sonar/Beam_group{i}"]["backscatter_i"].data.flatten(),
                        base["DataQ"].data.flatten(),
                        atol=absolute_tolerance,
                    )
        else:
            for i in range(1, len(echodata["Sonar"]["beam_group"]) + 1):
                if "transmit_pulse_r" in echodata[f"Sonar/Beam_group{i}"]:
                    # note the transpose
                    assert np.allclose(
                        echodata[f"Sonar/Beam_group{i}"]["backscatter_r"].data.flatten(),
                        base["Data_I"].data.T.flatten(),
                        atol=absolute_tolerance,
                    )
                    assert np.allclose(
                        echodata[f"Sonar/Beam_group{i}"]["backscatter_i"].data.flatten(),
                        base["Data_Q"].data.T.flatten(),
                        atol=absolute_tolerance,
                    )
        base.close()