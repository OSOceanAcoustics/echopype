from textwrap import dedent

import fsspec

import echopype
from echopype.echodata import EchoData
from echopype import open_converted

import pytest
import xarray as xr
import numpy as np


@pytest.fixture(scope="module")
def single_ek60_zarr(test_path):
    return (
        test_path['EK60'] / "ncei-wcsd" / "Summer2017-D20170615-T190214.zarr"
    )


@pytest.fixture(
    params=[
        single_ek60_zarr,
        (str, "ncei-wcsd", "Summer2017-D20170615-T190214.zarr"),
        (None, "ncei-wcsd", "Summer2017-D20170615-T190214.nc"),
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
            "complex",
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
            "Sp",
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
        "ek60_cw_complex",
        "ek80_cw_power",
        "ek80_bb_complex",
        "ek80_cw_complex",
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
            "path_model": "EK80",
            "raw_path": (
                "saildrone",
                "SD2019_WCS_v05-Phase0-D20190617-T125959-0.raw",
            ),
            "extra_data_path": (
                "saildrone",
                "saildrone-gen_5-fisheries-acoustics-code-sprint-sd1039-20190617T130000-20190618T125959-1_hz-v1.1595357449818.nc",
            ),
        },
    ],
    ids=["ek80_saildrone"],
)
def update_platform_samples(request, test_path):
    return (
        test_path[request.param["path_model"]].joinpath(*request.param['raw_path']),
        test_path[request.param["path_model"]].joinpath(*request.param['extra_data_path'])
    )


class TestEchoData:
    @pytest.fixture(scope="class")
    def converted_zarr(self, single_ek60_zarr):
        return single_ek60_zarr

    def test_constructor(self, converted_zarr):
        ed = EchoData(converted_raw_path=converted_zarr)
        expected_groups = [
            'top',
            'environment',
            'platform',
            'provenance',
            'sonar',
            'beam',
            'vendor',
        ]

        assert ed.sonar_model == 'EK60'
        assert ed.converted_raw_path == converted_zarr
        assert ed.storage_options == {}
        for group in expected_groups:
            assert isinstance(getattr(ed, group), xr.Dataset)

    def test_repr(self, converted_zarr):
        zarr_path_string = str(converted_zarr.absolute())
        expected_repr = dedent(
            f"""\
            EchoData: standardized raw data from {zarr_path_string}
              > top: (Top-level) contains metadata about the SONAR-netCDF4 file format.
              > environment: (Environment) contains information relevant to acoustic propagation through water.
              > platform: (Platform) contains information about the platform on which the sonar is installed.
              > nmea: (Platform/NMEA) contains information specific to the NMEA protocol.
              > provenance: (Provenance) contains metadata about how the SONAR-netCDF4 version of the data were obtained.
              > sonar: (Sonar) contains specific metadata for the sonar system.
              > beam: (Beam) contains backscatter data and other beam or channel-specific data.
              > vendor: (Vendor specific) contains vendor-specific information about the sonar and the data."""
        )
        ed = EchoData(converted_raw_path=converted_zarr)
        actual = "\n".join(x.rstrip() for x in repr(ed).split("\n"))
        assert expected_repr == actual

    def test_repr_html(self, converted_zarr):
        zarr_path_string = str(converted_zarr.absolute())
        ed = EchoData(converted_raw_path=converted_zarr)
        assert hasattr(ed, "_repr_html_")
        html_repr = ed._repr_html_().strip()
        assert (
            f"""<div class="xr-obj-type">EchoData: standardized raw data from {zarr_path_string}</div>"""
            in html_repr
        )

        with xr.set_options(display_style="text"):
            html_fallback = ed._repr_html_().strip()

        assert html_fallback.startswith(
            "<pre>EchoData"
        ) and html_fallback.endswith("</pre>")


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


def test_compute_range(compute_range_samples):
    (
        filepath,
        sonar_model,
        azfp_xml_path,
        azfp_cal_type,
        ek_waveform_mode,
        ek_encode_mode,
    ) = compute_range_samples
    ed = echopype.open_raw(filepath, sonar_model, azfp_xml_path)
    env_params = {"sound_speed": 343}

    if sonar_model == "AD2CP":
        try:
            ed.compute_range(
                env_params, ek_waveform_mode="CW", azfp_cal_type="Sv"
            )
        except ValueError:
            return
        else:
            raise AssertionError
    else:
        range = ed.compute_range(
            env_params,
            azfp_cal_type,
            ek_waveform_mode,
        )
        assert isinstance(range, xr.DataArray)


def test_update_platform(update_platform_samples):
    raw_file, extra_platform_data_file = update_platform_samples
    extra_platform_data_file_name = extra_platform_data_file.name

    ed = echopype.open_raw(raw_file, "EK80")

    updated = ["pitch", "roll", "latitude", "longitude", "water_level"]
    for variable in updated:
        assert np.isnan(ed.platform[variable].values).all()

    extra_platform_data = xr.open_dataset(extra_platform_data_file)
    ed.update_platform(
        extra_platform_data,
        extra_platform_data_file_name=extra_platform_data_file_name,
    )

    for variable in updated:
        assert not np.isnan(ed.platform[variable].values).all()

    # times have max interval of 2s
    # check times are > min(ed.beam["ping_time"]) - 2s
    assert (
        ed.platform["location_time"]
        > ed.beam["ping_time"].min() - np.timedelta64(2, "s")
    ).all()
    # check there is only 1 time < min(ed.beam["ping_time"])
    assert (
        np.count_nonzero(
            ed.platform["location_time"] < ed.beam["ping_time"].min()
        )
        == 1
    )
    # check times are < max(ed.beam["ping_time"]) + 2s
    assert (
        ed.platform["location_time"]
        < ed.beam["ping_time"].max() + np.timedelta64(2, "s")
    ).all()
    # check there is only 1 time > max(ed.beam["ping_time"])
    assert (
        np.count_nonzero(
            ed.platform["location_time"] > ed.beam["ping_time"].max()
        )
        == 1
    )
