"""``pytest`` configuration."""

import pytest

import fsspec

from echopype.testing import TEST_DATA_FOLDER


def pytest_generate_tests(metafunc):
    ek80_new_path = TEST_DATA_FOLDER / "ek80_new"
    ek80_files = ek80_new_path.glob("**/*.raw")
    if "ek80_file" in metafunc.fixturenames:
        metafunc.parametrize(
            "ek80_file", ek80_files, ids=lambda f: str(f.name)
        )


@pytest.fixture(scope="session")
def ek80_file(request):
    return request.param


@pytest.fixture(scope="session")
def dump_output_dir():
    return TEST_DATA_FOLDER / "dump"


@pytest.fixture(scope="session")
def test_path():
    ek60_path = TEST_DATA_FOLDER / "ek60"
    ek80_path = TEST_DATA_FOLDER / "ek80"
    ek80_new_path = TEST_DATA_FOLDER / "ek80_new"
    azfp_path = TEST_DATA_FOLDER / "azfp"
    ad2cp_path = TEST_DATA_FOLDER / "ad2cp"
    return {
        'ROOT': TEST_DATA_FOLDER,
        'EK60': ek60_path,
        'EK80': ek80_path,
        'EK80_NEW': ek80_new_path,
        'AZFP': azfp_path,
        'AD2CP': ad2cp_path
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
