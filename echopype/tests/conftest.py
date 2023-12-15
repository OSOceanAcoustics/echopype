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


def _get_sv_dataset(file_path):
    ed = ep.open_raw(file_path, sonar_model="ek60")
    Sv = ep.calibrate.compute_Sv(ed).compute()
    return Sv


@pytest.fixture(scope="session")
def sv_ek80():
    base_url = "noaa-wcsd-pds.s3.amazonaws.com/"
    path = "data/raw/Sally_Ride/SR1611/EK80/"
    file_name = "D20161109-T163350.raw"

    local_path = os.path.join(TEST_DATA_FOLDER, file_name)
    if os.path.isfile(local_path):
        ed = ep.open_raw(
            local_path,
            sonar_model="EK80",
        )
    else:
        raw_file_address = base_url + path + file_name
        rf = raw_file_address  # Path(raw_file_address)
        ed = ep.open_raw(
            f"https://{rf}",
            sonar_model="EK80",
        )
    Sv = ep.calibrate.compute_Sv(ed, waveform_mode="CW", encode_mode="complex").compute()
    return Sv
