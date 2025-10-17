"""``pytest`` configuration."""

import pytest


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
        "EK60_CAL_CHUNKS": TEST_DATA_FOLDER / "ek60_calibrate_chunks",
        "EK60_MISSING_CHANNEL_POWER": TEST_DATA_FOLDER / "ek60_missing_channel_power",
        "EK80": TEST_DATA_FOLDER / "ek80",
        "EK80_NEW": TEST_DATA_FOLDER / "ek80_new",
        "ES60": TEST_DATA_FOLDER / "es60",
        "ES70": TEST_DATA_FOLDER / "es70",
        "ES80": TEST_DATA_FOLDER / "es80",
        "AZFP": TEST_DATA_FOLDER / "azfp",
        "AZFP6": TEST_DATA_FOLDER / "azfp6",
        "AD2CP": TEST_DATA_FOLDER / "ad2cp",
        "EK80_MULTIPLEX": TEST_DATA_FOLDER / "ek80_bb_complex_multiplex",
        "EK80_DUPE_PING": TEST_DATA_FOLDER / "ek80_duplicate_ping_times",
        "EK80_MISSING_SOUND": TEST_DATA_FOLDER / "ek80_missing_sound_velocity_profile",
        "EK80_INVALID_ENV": TEST_DATA_FOLDER / "ek80_invalid_env_datagrams",
        "EK80_SEQUENCE": TEST_DATA_FOLDER / "ek80_sequence",
        "EK80_CAL": TEST_DATA_FOLDER / "ek80_bb_with_calibration",
        "EK80_EXT": TEST_DATA_FOLDER / "ek80_ext",
        "EK80_MULTI": TEST_DATA_FOLDER / "ek80_bb_complex_multiplex",
        "ECS": TEST_DATA_FOLDER / "ecs",
        "LEGACY_DATATREE": TEST_DATA_FOLDER / "legacy_datatree",
    }


@pytest.fixture(scope="session")
def minio_bucket():
    return dict(
        client_kwargs=dict(endpoint_url="http://localhost:9000/"),
        key="minioadmin",
        secret="minioadmin",
    )
