"""``pytest`` configuration."""
from http.server import HTTPServer
from moto.server import ThreadedMotoServer
import threading
import contextlib

import os
import pytest

import fsspec

from echopype.testing import TEST_DATA_FOLDER, RangeRequestHandler

HTTP_SERVER_PORT = 8080
MOTO_SERVER_PORT = 9000
AWS_ACCESS_KEY_ID = "motoadmin"
AWS_SECRET_ACCESS_KEY = "motoadmin"

@contextlib.contextmanager
def motoserve():
    server = ThreadedMotoServer(port=MOTO_SERVER_PORT)
    server.start()
    try:
        yield "http://127.0.0.1:%i" % MOTO_SERVER_PORT
    finally:
        server.stop()

@contextlib.contextmanager
def serve():
    server_address = ("", HTTP_SERVER_PORT)
    httpd = HTTPServer(server_address, RangeRequestHandler)
    th = threading.Thread(target=httpd.serve_forever)
    th.daemon = True
    th.start()
    try:
        yield "http://127.0.0.1:%i" % HTTP_SERVER_PORT
    finally:
        httpd.socket.close()
        httpd.shutdown()
        th.join()


@pytest.fixture(scope="module")
def http_server():
    with serve() as s:
        yield s

@pytest.fixture(scope="module")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY


@pytest.fixture(scope="module")
def moto_server(aws_credentials, common_storage_options):
    with motoserve() as s:
        fs = fsspec.filesystem(
            "s3",
            **common_storage_options,
        )
        test_data = "data"
        if not fs.exists(test_data):
            fs.mkdir(test_data)

        # Load test data into bucket
        for d in TEST_DATA_FOLDER.iterdir():
            fs.put(str(d), f"{test_data}/{d.name}", recursive=True)

        yield s

        print("Tear down.")


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
        'EK80_EXT': TEST_DATA_FOLDER / "ek80_ext",
        'ECS': TEST_DATA_FOLDER / "ecs",
    }


@pytest.fixture(scope="session")
def common_storage_options():
    return dict(
        client_kwargs=dict(endpoint_url="http://localhost:9000/"),
        key=AWS_ACCESS_KEY_ID,
        secret=AWS_SECRET_ACCESS_KEY,
    )
