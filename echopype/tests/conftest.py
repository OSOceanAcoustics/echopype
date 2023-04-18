"""``pytest`` configuration."""
from http.server import SimpleHTTPRequestHandler, HTTPServer
import socketserver
import threading
import contextlib

import pytest

import fsspec

from echopype.testing import TEST_DATA_FOLDER

HTTP_SERVER_PORT = 8080

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=TEST_DATA_FOLDER, **kwargs)

@contextlib.contextmanager
def serve():
    server_address = ("", HTTP_SERVER_PORT)
    httpd = HTTPServer(server_address, Handler)
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
def minio_bucket():
    return dict(
        client_kwargs=dict(endpoint_url="http://localhost:9000/"),
        key="minioadmin",
        secret="minioadmin",
    )
