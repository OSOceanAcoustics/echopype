import pytest

from collections import defaultdict

import numpy as np
import pandas as pd

from echopype.testing import _gen_ping_data_dict_power_angle, _gen_ping_data_dict_complex

@pytest.fixture
def mock_ping_data_dict_power_angle():
    # TODO: Parameterize this fixture
    return _gen_ping_data_dict_power_angle()

@pytest.fixture
def mock_ping_data_dict_complex():
    # TODO: Parameterize this fixture
    return _gen_ping_data_dict_complex()
