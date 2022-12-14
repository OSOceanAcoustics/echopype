import pytest

from echopype.testing import get_mock_echodata


@pytest.fixture
def mock_echodata():
    return get_mock_echodata()
