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
def minio_bucket():
    common_storage_options = dict(
        client_kwargs=dict(endpoint_url="http://localhost:9000/"),
        key="minioadmin",
        secret="minioadmin",
    )
    bucket_name = "ooi-raw-data"
    fs = fsspec.filesystem(
        "s3",
        **common_storage_options,
    )
    test_data = "data"
    if not fs.exists(test_data):
        fs.mkdir(test_data)

    if not fs.exists(bucket_name):
        fs.mkdir(bucket_name)

    # Load test data into bucket
    for d in TEST_DATA_FOLDER.iterdir():
        source_path = f'echopype/test_data/{d.name}'
        fs.put(source_path, f'{test_data}/{d.name}', recursive=True)

    return common_storage_options
