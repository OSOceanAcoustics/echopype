"""``pytest`` configuration."""

import pytest
from pathlib import Path

import fsspec


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
    test_data_path = Path(__file__).parent.parent.joinpath(Path("test_data"))
    for d in test_data_path.iterdir():
        source_path = f'echopype/test_data/{d.name}'
        fs.put(source_path, f'{test_data}/{d.name}', recursive=True)

    return common_storage_options
