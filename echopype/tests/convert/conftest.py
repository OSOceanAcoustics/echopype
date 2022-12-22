import pytest

from echopype.testing import TEST_DATA_FOLDER


# AZFP ----------------------

@pytest.fixture
def azfp_conversion_file(test_path):
    azfp_01a_path = test_path["AZFP"] / '17082117.01A'
    azfp_xml_path = test_path["AZFP"] / '17041823.XML'

    return azfp_01a_path, azfp_xml_path


# EK60 ----------------------

# EK80 ----------------------

def pytest_generate_tests(metafunc):
    ek80_new_path = TEST_DATA_FOLDER / "ek80_new"
    ek80_new_files = ek80_new_path.glob("**/*.raw")
    if "ek80_new_file" in metafunc.fixturenames:
        metafunc.parametrize(
            "ek80_new_file", ek80_new_files, ids=lambda f: str(f.name)
        )


@pytest.fixture
def ek80_new_file(request):
    return request.param


@pytest.fixture
def ek80cw_bb_conversion_file(test_path):
    """EK80 CW power/angle BB data"""
    return test_path["EK80"] / "D20170912-T234910.raw"
