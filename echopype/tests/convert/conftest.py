import pytest

from echopype.testing import TEST_DATA_FOLDER


def _create_path_str(test_folder, paths):
    return test_folder.joinpath(*paths).absolute()


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


# For test_convert_source_target_locs_integration.py ----------------------

@pytest.fixture(
    params=[
        None,
        "/",
        "/tmp.zarr",
        "/tmp.nc",
        "s3://ooi-raw-data/dump/",
        "s3://ooi-raw-data/dump/tmp.zarr",
        "s3://ooi-raw-data/dump/tmp.nc",
    ],
    ids=[
        "None",
        "folder_string",
        "zarr_file_string",
        "netcdf_file_string",
        "s3_folder_string",
        "s3_zarr_file_string",
        "s3_netcdf_file_string",
    ],
)
def output_save_path(request):
    return request.param


@pytest.fixture(params=["zarr", "netcdf4"])
def export_engine(request):
    return request.param


# TODO: (Emilio) Overhaul so the file name and sensor model are stated only once,
#  followed by the various source paths?
@pytest.fixture(
    params=[
        [("ncei-wcsd", "Summer2017-D20170615-T190214.raw"), "EK60"],
        [
            "s3://data/ek60/ncei-wcsd/Summer2017-D20170615-T190214.raw",
            "EK60",
        ],
        [
            [
                "http://localhost:8080/data/ek60/ncei-wcsd/Summer2017-D20170615-T190214.raw",
                "http://localhost:8080/data/ek60/ncei-wcsd/Summer2017-D20170615-T190843.raw",
            ],
            "EK60",
        ],
        [("D20151202-T020259.raw",), "ES70"],
        ["s3://data/es70/D20151202-T020259.raw", "ES70"],
        [
            [
                "http://localhost:8080/data/es70/D20151202-T020259.raw",
            ],
            "ES70",
        ],
        [("WBT-D20210620-T012250.raw",), "ES80"],
        [("WBT-but-internally-marked-as-EK80-D20210710-T204029.raw",), "ES80"],
        ["s3://data/es80/WBT-D20210620-T012250.raw", "ES80"],
        [
            [
                "http://localhost:8080/data/es80/WBT-D20210620-T012250.raw",
            ],
            "ES80",
        ],
        [("ea640_test.raw",), "EA640"],
        ["s3://data/ea640/ea640_test.raw", "EA640"],
        [
            [
                "http://localhost:8080/data/ea640/ea640_test.raw",
            ],
            "EA640",
        ],
        [("echopype-test-D20211005-T001135.raw",), "EK80"],
        [
            "http://localhost:8080/data/ek80_new/echopype-test-D20211005-T001135.raw",
            "EK80",
        ],
        ["s3://data/ek80_new/echopype-test-D20211005-T001135.raw", "EK80"],
    ],
    ids=[
        "ek60_file_path_string",
        "ek60_s3_file_string",
        "ek60_multiple_http_file_string",
        "es70_file_path_string",
        "es70_s3_file_string",
        "es70_multiple_http_file_string",
        "es80_file_path_string_WBT",
        "es80_file_path_string_WBT_EK80",
        "es80_s3_file_string",
        "es80_multiple_http_file_string",
        "ea640_file_path_string",
        "ea640_s3_file_string",
        "ea640_multiple_http_file_string",
        "ek80_file_path_string",
        "ek80_http_file_string",
        "ek80_s3_file_string",
    ],
)
def ek_input_params(request, test_path):
    path, model = request.param
    key = model
    if model == "EK80":
        key = f"{model}_NEW"

    if isinstance(path, tuple):
        path = _create_path_str(test_path[key], path)

    return [path, model]


@pytest.fixture(
    params=[
        ("ooi", "17032923.01A"),
        "http://localhost:8080/data/azfp/ooi/17032923.01A",
    ],
    ids=["file_path_string", "http_file_string"],
)
def azfp_input_paths(request, test_path):
    if isinstance(request.param, tuple):
        return _create_path_str(test_path["AZFP"], request.param)
    return request.param


@pytest.fixture(
    params=[
        ("ooi", "17032922.XML"),
        "http://localhost:8080/data/azfp/ooi/17032922.XML",
    ],
    ids=["xml_file_path_string", "xml_http_file_string"],
)
def azfp_xml_paths(request, test_path):
    if isinstance(request.param, tuple):
        return _create_path_str(test_path["AZFP"], request.param)
    return request.param


@pytest.fixture(
    params=[
        ("AZFP", "azfp", ("ooi", "17032923.01A"), ("ooi", "17032922.XML")),
        (
            "EK60",
            "ek60",
            ("DY1801_EK60-D20180211-T164025.raw",),
            None,
        ),
        (
            "ES70",
            "es70",
            ("D20151202-T020259.raw",),
            None,
        ),
        (
            "ES80",
            "es80",
            ("WBT-D20210620-T012250.raw",),
            None,
        ),
        (
            "EA640",
            "ea640",
            ("ea640_test.raw",),
            None,
        ),
        (
            "EK80_NEW",
            "ek80",
            ("echopype-test-D20211004-T235757.raw",),
            None,
        ),
        (
            "AD2CP",
            "ad2cp",
            ("raw", "076", "rawtest.076.00000.ad2cp"),
            None,
        ),
    ],
    ids=["azfp", "ek60", "es70", "es80", "ea640", "ek80", "ad2cp"],
)
def convert_time_encodings_params(request, test_path):
    path_model, sonar_model, raw_file, xml_path = request.param
    return [path_model, sonar_model, raw_file, xml_path]
