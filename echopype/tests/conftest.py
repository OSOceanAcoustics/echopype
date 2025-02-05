"""``pytest`` configuration."""

import pytest
import pooch
import os
from pathlib import Path
from zipfile import ZipFile


from echopype.testing import TEST_DATA_FOLDER

ECHOPYPE_RESOURCES = pooch.create(
    path=pooch.os_cache("echopype"),
    base_url="https://github.com/oftfrfbf/echopype-test-data/releases/download/{version}/",
    version="2025.2.1",
    registry={
        "legacy_data.zip": "sha256:14e0ef5715716aa7f42ca148c9aea660a36313dad12d419f827ddbd22d6bc902",
    },
)


def unpack(fname, action, pup):
    unzipped = Path(fname.split(".zip")[0]).parent  # + ".unzipped"
    unzipped_child = fname.split(".zip")[0]
    if action in ("update", "download") or not os.path.exists(unzipped_child):
        with ZipFile(fname, "r") as zip_file:
            zip_file.extractall(path=unzipped)
    return unzipped


def fetch_zipped_file(file_path):
    fname = ECHOPYPE_RESOURCES.fetch(
        fname=file_path,
        processor=unpack,
        # progressbar=True,
    )
    return Path(fname).joinpath(Path(file_path).stem)


legacy_data = fetch_zipped_file("legacy_data.zip")


@pytest.fixture(scope="session")
def dump_output_dir():
    return TEST_DATA_FOLDER / "dump"


@pytest.fixture(scope="session")
def test_path():
    return {
        "ROOT": TEST_DATA_FOLDER,
        "EA640": TEST_DATA_FOLDER / "ea640",
        "EK60": TEST_DATA_FOLDER / "ek60",
        "EK60_MISSING_CHANNEL_POWER": TEST_DATA_FOLDER / "ek60_missing_channel_power",
        "EK80": TEST_DATA_FOLDER / "ek80",
        "EK80_NEW": TEST_DATA_FOLDER / "ek80_new",
        "ES60": TEST_DATA_FOLDER / "es60",
        "ES70": TEST_DATA_FOLDER / "es70",
        "ES80": TEST_DATA_FOLDER / "es80",
        "AZFP": TEST_DATA_FOLDER / "azfp",
        "AZFP6": TEST_DATA_FOLDER / "azfp6",
        "AD2CP": TEST_DATA_FOLDER / "ad2cp",
        "EK80_CAL": TEST_DATA_FOLDER / "ek80_bb_with_calibration",
        "EK80_EXT": TEST_DATA_FOLDER / "ek80_ext",
        "ECS": TEST_DATA_FOLDER / "ecs",
        "LEGACY_DATA": legacy_data,
    }


@pytest.fixture(scope="session")
def minio_bucket():
    return dict(
        client_kwargs=dict(endpoint_url="http://localhost:9000/"),
        key="minioadmin",
        secret="minioadmin",
    )
