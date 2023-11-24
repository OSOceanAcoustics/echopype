"""``pytest`` configuration."""
from ftplib import FTP

import os
import subprocess
import pytest

import xarray as xr

import echopype as ep
from echopype.testing import TEST_DATA_FOLDER


def _setup_file(file_name):
    FTP_MAIN = "ftp.bas.ac.uk"
    FTP_PARTIAL_PATH = "rapidkrill/ek60/"
    with FTP(FTP_MAIN) as ftp:
        ftp.login()
        print(TEST_DATA_FOLDER)
        download_ftp_file(ftp, FTP_PARTIAL_PATH, file_name, TEST_DATA_FOLDER)
    return os.path.join(TEST_DATA_FOLDER, file_name)


def download_ftp_file(ftp, remote_path, file_name, local_path):
    # Construct the full paths
    remote_file_path = os.path.join(remote_path, file_name)
    local_file_path = os.path.join(local_path, file_name)

    try:
        # Ensure the local directory exists
        os.makedirs(local_path, exist_ok=True)

        # Check if the file already exists locally
        if not os.path.exists(local_file_path):
            with open(local_file_path, "wb") as local_file:
                ftp.retrbinary("RETR " + remote_file_path, local_file.write)
        else:
            print(f"File {local_file_path} already exists. Skipping download.")

    except Exception as e:
        print(f"Error downloading {remote_file_path}. Error: {e}")


def _get_sv_dataset(file_path, enriched: bool = False, waveform: str = "CW", encode: str = "power"):
    ed = ep.open_raw(file_path, sonar_model="ek60")
    Sv = ep.calibrate.compute_Sv(ed).compute()
    if enriched is True:
        Sv = ep.consolidate.add_splitbeam_angle(Sv, ed, waveform, encode)
    return Sv


def _get_raw_dataset(file_path):
    ed = ep.open_raw(file_path, sonar_model="ek60")
    return ed


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


@pytest.fixture(scope="session")
def raw_dataset_jr179(setup_test_data_jr179):
    ed = _get_raw_dataset(setup_test_data_jr179)
    return ed


@pytest.fixture(scope="session")
def ed_ek_60_for_Sv():
    bucket = "ncei-wcsd-archive"
    base_path = "data/raw/Bell_M._Shimada/SH1707/EK60/"
    filename = "Summer2017-D20170620-T011027.raw"
    rawdirpath = base_path + filename

    s3raw_fpath = f"s3://{bucket}/{rawdirpath}"
    storage_opts = {"anon": True}
    ed = ep.open_raw(s3raw_fpath, sonar_model="EK60", storage_options=storage_opts)  # type: ignore
    return ed


@pytest.fixture(scope="session")
def ek60_Sv(ed_ek_60_for_Sv):
    sv_echopype_EK60 = ep.calibrate.compute_Sv(ed_ek_60_for_Sv).compute()
    return sv_echopype_EK60
