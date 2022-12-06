"""``pytest`` configuration."""

import pytest

import fsspec

from echopype.testing import TEST_DATA_FOLDER


@pytest.fixture(scope="session")
def dump_output_dir():
    return TEST_DATA_FOLDER / "dump"


@pytest.fixture(scope="session")
def test_path():
    return {
        'ROOT': TEST_DATA_FOLDER,
        'EA640': TEST_DATA_FOLDER / "ea640",
        'EK60': TEST_DATA_FOLDER / "ek60",
        'EK80': TEST_DATA_FOLDER / "ek80",
        'EK80_NEW': TEST_DATA_FOLDER / "ek80_new",
        'ES70': TEST_DATA_FOLDER / "es70",
        'ES80': TEST_DATA_FOLDER / "es80",
        'AZFP': TEST_DATA_FOLDER / "azfp",
        'AD2CP': TEST_DATA_FOLDER / "ad2cp",
        'EK80_CAL': TEST_DATA_FOLDER / "ek80_bb_with_calibration",
    }


@pytest.fixture(scope="session")
def ek60_test_data(test_path):
    files = [
        ("ncei-wcsd", "Summer2017-D20170620-T011027.raw"),
        ("ncei-wcsd", "Summer2017-D20170620-T014302.raw"),
        ("ncei-wcsd", "Summer2017-D20170620-T021537.raw"),
    ]
    return [test_path["EK60"].joinpath(*f) for f in files]


@pytest.fixture(scope="session")
def minio_bucket():
    return dict(
        client_kwargs=dict(endpoint_url="http://localhost:9000/"),
        key="minioadmin",
        secret="minioadmin",
    )
