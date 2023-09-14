"""``pytest`` configuration."""

import os
import subprocess
import pytest

import xarray as xr

import echopype as ep
from echopype.testing import TEST_DATA_FOLDER


@pytest.fixture(scope="session")
def dump_output_dir():
    return TEST_DATA_FOLDER / "dump"


@pytest.fixture(scope="session")
def test_path():
    return {
        "ROOT": TEST_DATA_FOLDER,
        "EA640": TEST_DATA_FOLDER / "ea640",
        "EK60": TEST_DATA_FOLDER / "ek60",
        "EK80": TEST_DATA_FOLDER / "ek80",
        "EK80_NEW": TEST_DATA_FOLDER / "ek80_new",
        "ES70": TEST_DATA_FOLDER / "es70",
        "ES80": TEST_DATA_FOLDER / "es80",
        "AZFP": TEST_DATA_FOLDER / "azfp",
        "AD2CP": TEST_DATA_FOLDER / "ad2cp",
        "EK80_CAL": TEST_DATA_FOLDER / "ek80_bb_with_calibration",
        "EK80_EXT": TEST_DATA_FOLDER / "ek80_ext",
        "ECS": TEST_DATA_FOLDER / "ecs",
    }


@pytest.fixture(scope="session")
def minio_bucket():
    return dict(
        client_kwargs=dict(endpoint_url="http://localhost:9000/"),
        key="minioadmin",
        secret="minioadmin",
    )


@pytest.fixture(scope="session")
def setup_test_data_jr230():
    file_name = "JR230-D20091215-T121917.raw"
    return _setup_file(file_name)


@pytest.fixture(scope="session")
def setup_test_data_jr161():
    file_name = "JR161-D20061118-T010645.raw"
    return _setup_file(file_name)


@pytest.fixture(scope="session")
def setup_test_data_jr179():
    file_name = "JR179-D20080410-T150637.raw"
    return _setup_file(file_name)


def _setup_file(file_name):
    test_data_path = os.path.join(TEST_DATA_FOLDER, file_name)
    FTP_MAIN = "ftp://ftp.bas.ac.uk"
    FTP_PARTIAL_PATH = "/rapidkrill/ek60/"
    if not os.path.exists(TEST_DATA_FOLDER):
        os.mkdir(TEST_DATA_FOLDER)
    if not os.path.exists(test_data_path):
        ftp_file_path = FTP_MAIN + FTP_PARTIAL_PATH + file_name
        subprocess.run(["wget", ftp_file_path, "-O", test_data_path])

    return test_data_path


# Separate Sv dataset fixtures for each file


@pytest.fixture(scope="session")
def sv_dataset_jr230(setup_test_data_jr230) -> xr.DataArray:
    return _get_sv_dataset(setup_test_data_jr230)


@pytest.fixture(scope="session")
def sv_dataset_jr161(setup_test_data_jr161) -> xr.DataArray:
    return _get_sv_dataset(setup_test_data_jr161)


@pytest.fixture(scope="session")
def sv_dataset_jr179(setup_test_data_jr179) -> xr.DataArray:
    return _get_sv_dataset(setup_test_data_jr179)


@pytest.fixture(scope="session")
def complete_dataset_jr179(setup_test_data_jr179):
    Sv = _get_sv_dataset(setup_test_data_jr179, enriched=True, waveform="CW", encode="power")
    return Sv


def _get_sv_dataset(file_path, enriched: bool = False, waveform: str = "CW", encode: str = "power"):
    ed = ep.open_raw(file_path, sonar_model="ek60")
    Sv = ep.calibrate.compute_Sv(ed).compute()
    if enriched is True:
        Sv = ep.consolidate.add_splitbeam_angle(Sv, ed, waveform, encode)
    return Sv
