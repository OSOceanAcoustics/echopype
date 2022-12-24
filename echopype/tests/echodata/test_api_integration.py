import fsspec
from echopype import open_converted
from echopype.echodata import EchoData

def test_open_converted(ek60_converted_zarr, minio_bucket):  # noqa
    def _check_path(zarr_path):
        storage_options = {}
        if zarr_path.startswith("s3://"):
            storage_options = dict(
                client_kwargs=dict(endpoint_url="http://localhost:9000/"),
                key="minioadmin",
                secret="minioadmin",
            )
        return storage_options

    storage_options = {}
    if not isinstance(ek60_converted_zarr, fsspec.FSMap):
        storage_options = _check_path(str(ek60_converted_zarr))

    try:
        ed = open_converted(
            ek60_converted_zarr, storage_options=storage_options
        )
        assert isinstance(ed, EchoData) is True
    except Exception as e:
        if (
            isinstance(ek60_converted_zarr, str)
            and ek60_converted_zarr.startswith("s3://")
            and ek60_converted_zarr.endswith(".nc")
        ):
            assert isinstance(e, ValueError) is True
