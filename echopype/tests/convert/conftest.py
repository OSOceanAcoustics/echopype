import pytest


# AZFP ----------------------

@pytest.fixture
def azfp_conversion_file(test_path):
    azfp_01a_path = test_path["AZFP"] / '17082117.01A'
    azfp_xml_path = test_path["AZFP"] / '17041823.XML'

    return azfp_01a_path, azfp_xml_path
