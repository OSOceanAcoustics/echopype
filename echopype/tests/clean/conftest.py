"""``pytest`` configuration."""

import os
import subprocess
import pytest

import fsspec
import xarray as xr

import echopype as ep
from echopype.testing import TEST_DATA_FOLDER


@pytest.fixture(scope="module")
def setup_test_data_jr230():
    file_name = "JR230-D20091215-T121917.raw"
    return _setup_file(file_name)


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def sv_dataset_jr230(setup_test_data_jr230) -> xr.DataArray:
    return _get_sv_dataset(setup_test_data_jr230)


@pytest.fixture(scope="module")
def sv_dataset_jr161(setup_test_data_jr161) -> xr.DataArray:
    return _get_sv_dataset(setup_test_data_jr161)


def _get_sv_dataset(file_path):
    ed = ep.open_raw(file_path, sonar_model="ek60")
    Sv = ep.calibrate.compute_Sv(ed).compute()
    return Sv
