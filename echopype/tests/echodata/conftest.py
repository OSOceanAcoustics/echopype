import pytest
import fsspec


@pytest.fixture
def azfp_path(test_path):
    return test_path['AZFP']


@pytest.fixture
def ek60_path(test_path):
    return test_path['EK60']


@pytest.fixture
def ek80_path(test_path):
    return test_path['EK80']


@pytest.fixture(scope="class")
def single_ek60_zarr(test_path):
    return (
        test_path['EK60']
        / "ncei-wcsd"
        / "Summer2017-D20170615-T190214__NEW.zarr"
    )


@pytest.fixture(
    params=[
        single_ek60_zarr,
        (str, "ncei-wcsd", "Summer2017-D20170615-T190214.zarr"),
        (None, "ncei-wcsd", "Summer2017-D20170615-T190214__NEW.nc"),
        "s3://data/ek60/ncei-wcsd/Summer2017-D20170615-T190214.nc",
        "http://localhost:8080/data/ek60/ncei-wcsd/Summer2017-D20170615-T190214.zarr",
        "s3://data/ek60/ncei-wcsd/Summer2017-D20170615-T190214.zarr",
        fsspec.get_mapper(
            "s3://data/ek60/ncei-wcsd/Summer2017-D20170615-T190214.zarr",
            **dict(
                client_kwargs=dict(endpoint_url="http://localhost:9000/"),
                key="minioadmin",
                secret="minioadmin",
            ),
        ),
    ],
    ids=[
        "ek60_zarr_path",
        "ek60_zarr_path_string",
        "ek60_netcdf_path",
        "ek60_netcdf_s3_string",
        "ek60_zarr_http_string",
        "ek60_zarr_s3_string",
        "ek60_zarr_s3_FSMap",
    ],
)
def ek60_converted_zarr(request, test_path):
    if isinstance(request.param, tuple):
        desired_type, *paths = request.param
        if desired_type is not None:
            return desired_type(test_path['EK60'].joinpath(*paths))
        else:
            return test_path['EK60'].joinpath(*paths)
    else:
        return request.param


@pytest.fixture(
    params=[
        (
            ("EK60", "ncei-wcsd", "Summer2017-D20170615-T190214.raw"),
            "EK60",
            None,
            None,
            "CW",
            "power",
        ),
        (
            ("EK80_NEW", "D20211004-T233354.raw"),
            "EK80",
            None,
            None,
            "CW",
            "power",
        ),
        (
            ("EK80_NEW", "echopype-test-D20211004-T235930.raw"),
            "EK80",
            None,
            None,
            "BB",
            "complex",
        ),
        (
            ("EK80_NEW", "D20211004-T233115.raw"),
            "EK80",
            None,
            None,
            "CW",
            "complex",
        ),
        (
            ("ES70", "D20151202-T020259.raw"),
            "ES70",
            None,
            None,
            None,
            None,
        ),
        (
            ("AZFP", "ooi", "17032923.01A"),
            "AZFP",
            ("AZFP", "ooi", "17032922.XML"),
            "Sv",
            None,
            None,
        ),
        (
            ("AZFP", "ooi", "17032923.01A"),
            "AZFP",
            ("AZFP", "ooi", "17032922.XML"),
            "TS",
            None,
            None,
        ),
        (
            ("AD2CP", "raw", "090", "rawtest.090.00001.ad2cp"),
            "AD2CP",
            None,
            None,
            None,
            None,
        ),
    ],
    ids=[
        "ek60_cw_power",
        "ek80_cw_power",
        "ek80_bb_complex",
        "ek80_cw_complex",
        "es70",
        "azfp_sv",
        "azfp_sp",
        "ad2cp",
    ],
)
def compute_range_samples(request, test_path):
    (
        filepath,
        sonar_model,
        azfp_xml_path,
        azfp_cal_type,
        ek_waveform_mode,
        ek_encode_mode,
    ) = request.param
    if sonar_model.lower() == 'es70':
        pytest.xfail(
            reason="Not supported at the moment",
        )
    path_model, *paths = filepath
    filepath = test_path[path_model].joinpath(*paths)

    if azfp_xml_path is not None:
        path_model, *paths = azfp_xml_path
        azfp_xml_path = test_path[path_model].joinpath(*paths)
    return (
        filepath,
        sonar_model,
        azfp_xml_path,
        azfp_cal_type,
        ek_waveform_mode,
        ek_encode_mode,
    )


@pytest.fixture(
    params=[
        {
            "path_model": "EK60",
            "raw_path": "Winter2017-D20170115-T150122.raw",
        },
        {
            "path_model": "EK80",
            "raw_path": "D20170912-T234910.raw",
        },
    ],
    ids=[
        "ek60_winter2017",
        "ek80_summer2017",
    ],
)
def range_check_files(request, test_path):
    return (
        request.param["path_model"],
        test_path[request.param["path_model"]].joinpath(
            request.param['raw_path']
        ),
    )


@pytest.fixture(
    params=[
        (
            {
                "randint_low": 10,
                "randint_high": 5000,
                "num_datasets": 20,
                "group": "test_group",
                "zarr_name": "combined_echodatas.zarr",
                "delayed_ds_list": False,
            }
        ),
        (
            {
                "randint_low": 10,
                "randint_high": 5000,
                "num_datasets": 20,
                "group": "test_group",
                "zarr_name": "combined_echodatas.zarr",
                "delayed_ds_list": True,
            }
        ),
    ],
    ids=["in-memory-ds_list", "lazy-ds_list"],
    scope="module",
)
def append_ds_list_params(request):
    return list(request.param.values())


@pytest.fixture
def ek60_test_data(test_path):
    files = [
        ("ncei-wcsd", "Summer2017-D20170620-T011027.raw"),
        ("ncei-wcsd", "Summer2017-D20170620-T014302.raw"),
        ("ncei-wcsd", "Summer2017-D20170620-T021537.raw"),
    ]
    return [test_path["EK60"].joinpath(*f) for f in files]


@pytest.fixture
def ek60_diff_range_sample_test_data(test_path):
    files = [
        ("ncei-wcsd", "SH1701", "TEST-D20170114-T202932.raw"),
        ("ncei-wcsd", "SH1701", "TEST-D20170114-T203337.raw"),
        ("ncei-wcsd", "SH1701", "TEST-D20170114-T203853.raw"),
    ]
    return [test_path["EK60"].joinpath(*f) for f in files]


@pytest.fixture
def ek80_test_data(test_path):
    files = [
        ("echopype-test-D20211005-T000706.raw",),
        ("echopype-test-D20211005-T000737.raw",),
        ("echopype-test-D20211005-T000810.raw",),
        ("echopype-test-D20211005-T000843.raw",),
    ]
    return [test_path["EK80_NEW"].joinpath(*f) for f in files]


@pytest.fixture
def ek80_broadband_same_range_sample_test_data(test_path):
    files = [
        ("ncei-wcsd", "SH1707", "Reduced_D20170826-T205615.raw"),
        ("ncei-wcsd", "SH1707", "Reduced_D20170826-T205659.raw"),
        ("ncei-wcsd", "SH1707", "Reduced_D20170826-T205742.raw"),
    ]
    return [test_path["EK80"].joinpath(*f) for f in files]


@pytest.fixture
def ek80_narrowband_diff_range_sample_test_data(test_path):
    files = [
        ("ncei-wcsd", "SH2106", "EK80", "Reduced_Hake-D20210701-T130426.raw"),
        ("ncei-wcsd", "SH2106", "EK80", "Reduced_Hake-D20210701-T131325.raw"),
        ("ncei-wcsd", "SH2106", "EK80", "Reduced_Hake-D20210701-T131621.raw"),
    ]
    return [test_path["EK80"].joinpath(*f) for f in files]


@pytest.fixture
def azfp_test_data(test_path):

    # TODO: in the future we should replace these files with another set of
    #  similarly small set of files, for example the files from the location below:
    #  "https://rawdata.oceanobservatories.org/files/CE01ISSM/R00015/instrmts/dcl37/ZPLSC_sn55076/DATA/202109/*"
    #  This is because we have lost track of where the current files came from,
    #  since the filenames does not contain the site identifier.
    files = [
        ("ooi", "18100407.01A"),
        ("ooi", "18100408.01A"),
        ("ooi", "18100409.01A"),
    ]
    return [test_path["AZFP"].joinpath(*f) for f in files]


@pytest.fixture
def azfp_test_xml(test_path):
    return test_path["AZFP"].joinpath(*("ooi", "18092920.XML"))


@pytest.fixture(
    params=[
        {"sonar_model": "EK60", "xml_file": None, "files": "ek60_test_data"},
        {
            "sonar_model": "EK60",
            "xml_file": None,
            "files": "ek60_diff_range_sample_test_data",
        },
        {
            "sonar_model": "AZFP",
            "xml_file": "azfp_test_xml",
            "files": "azfp_test_data",
        },
        {
            "sonar_model": "EK80",
            "xml_file": None,
            "files": "ek80_broadband_same_range_sample_test_data",
        },
        {
            "sonar_model": "EK80",
            "xml_file": None,
            "files": "ek80_narrowband_diff_range_sample_test_data",
        },
    ],
    ids=[
        "ek60",
        "ek60_diff_range_sample",
        "azfp",
        "ek80_bb_same_range_sample",
        "ek80_nb_diff_range_sample",
    ],
)
def raw_datasets(request):
    files = request.param["files"]
    xml_file = request.param["xml_file"]
    if xml_file is not None:
        xml_file = request.getfixturevalue(xml_file)

    files = request.getfixturevalue(files)

    return (
        files,
        request.param['sonar_model'],
        xml_file,
        request.node.callspec.id,
    )
