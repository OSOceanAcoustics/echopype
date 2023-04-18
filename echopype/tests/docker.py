import pytest
import sys
import requests

from requests.exceptions import ConnectionError

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Don't test on anything else for now")

def is_responsive(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
    except ConnectionError:
        return False


@pytest.fixture(scope="session")
def http_service(docker_ip, docker_services):
    """Ensure that HTTP service is up and responsive."""

    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("httpserver", 80)
    url = "http://{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive(url)
    )
    return url


def test_service(http_service):
    status = 200
    response = requests.get(http_service + "/")

    assert response.status_code == status
    assert "It works!" in response.text
