from textwrap import dedent

import fsspec

from datatree import DataTree
from zarr.errors import GroupNotFoundError

import echopype
from echopype.calibrate.env_params import EnvParams
from echopype.echodata import EchoData
from echopype import open_converted

import pytest
import xarray as xr
import numpy as np


@pytest.fixture(scope="module")
def single_ek60_zarr(test_path):
    return (
        test_path['EK60'] / "ncei-wcsd" / "Summer2017-D20170615-T190214__NEW.zarr"
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
        test_path[request.param["path_model"]].joinpath(request.param['raw_path'])
    )


class TestEchoData:
    @pytest.fixture(scope="class")
    def converted_zarr(self, single_ek60_zarr):
        return single_ek60_zarr

    def test_constructor(self, converted_zarr):
        ed = EchoData.from_file(converted_raw_path=converted_zarr)
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
            <EchoData: standardized raw data from {zarr_path_string}>
            Top-level: contains metadata about the SONAR-netCDF4 file format.
            ├── Environment: contains information relevant to acoustic propagation through water.
            ├── Platform: contains information about the platform on which the sonar is installed.
            │   └── NMEA: contains information specific to the NMEA protocol.
            ├── Provenance: contains metadata about how the SONAR-netCDF4 version of the data were obtained.
            ├── Sonar: contains sonar system metadata and sonar beam groups.
            │   └── Beam_group1: contains backscatter data (either complex samples or uncalibrated power samples) and other beam or channel-specific data, including split-beam angle data when they exist.
            └── Vendor_specific: contains vendor-specific information about the sonar and the data."""
        )
        ed = EchoData.from_file(converted_raw_path=converted_zarr)
        actual = "\n".join(x.rstrip() for x in repr(ed).split("\n"))
        assert expected_repr == actual

    def test_repr_html(self, converted_zarr):
        zarr_path_string = str(converted_zarr.absolute())
        ed = EchoData.from_file(converted_raw_path=converted_zarr)
        assert hasattr(ed, "_repr_html_")
        html_repr = ed._repr_html_().strip()
        assert (
            f"""<div class="xr-obj-type">EchoData: standardized raw data from {zarr_path_string}</div>"""
            in html_repr
        )

        with xr.set_options(display_style="text"):
            html_fallback = ed._repr_html_().strip()

        assert html_fallback.startswith(
            "<pre>&lt;EchoData"
        ) and html_fallback.endswith("</pre>")

    def test_setattr(self, converted_zarr):
        sample_data = xr.Dataset({"x": [0, 0, 0]})
        sample_data2 = xr.Dataset({"y": [0, 0, 0]})
        ed = EchoData.from_file(converted_raw_path=converted_zarr)
        current_ed_beam = ed.beam
        current_ed_top = ed.top
        ed.beam = sample_data
        ed.top = sample_data2

        assert ed.beam.equals(sample_data) is True
        assert ed.beam.equals(ed['Sonar/Beam_group1']) is True
        assert ed.beam.equals(current_ed_beam) is False

        assert ed.top.equals(sample_data2) is True
        assert ed.top.equals(ed['Top-level']) is True
        assert ed.top.equals(current_ed_top) is False

    def test_getitem(self, converted_zarr):
        ed = EchoData.from_file(converted_raw_path=converted_zarr)
        beam = ed['Sonar/Beam_group1']
        assert isinstance(beam, xr.Dataset)
        try:
            ed['MyGroup']
        except Exception as e:
            assert isinstance(e, GroupNotFoundError)

        ed._tree = None
        try:
            ed['Sonar']
        except Exception as e:
            assert isinstance(e, ValueError)

    def test_getattr(self, converted_zarr):
        ed = EchoData.from_file(converted_raw_path=converted_zarr)
        expected_groups = {
            'top': 'Top-level',
            'environment': 'Environment',
            'platform': 'Platform',
            'nmea': 'Platform/NMEA',
            'provenance': 'Provenance',
            'sonar': 'Sonar',
            'beam': 'Sonar/Beam_group1',
            'vendor': 'Vendor_specific',
        }
        for group, path in expected_groups.items():
            ds = getattr(ed, group)
            assert ds.equals(ed[path])

    def test_setitem(self, converted_zarr):
        ed = EchoData.from_file(converted_raw_path=converted_zarr)
        ed['Sonar/Beam_group1'] = ed['Sonar/Beam_group1'].rename({'beam': 'beam_newname'})

        assert sorted(ed['Sonar/Beam_group1'].dims.keys()) == ['beam_newname', 'channel', 'ping_time', 'range_sample']

    def test_get_dataset(self, converted_zarr):
        ed = EchoData.from_file(converted_raw_path=converted_zarr)
        node = DataTree()
        result = ed._EchoData__get_dataset(node)

        ed_node = ed._tree['Sonar']
        ed_result = ed._EchoData__get_dataset(ed_node)

        assert result is None
        assert isinstance(ed_result, xr.Dataset)

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
    print(ed.platform)
    rng = np.random.default_rng(0)
    stationary_env_params = EnvParams(
        xr.Dataset(
            data_vars={
                "pressure": ("time3", np.arange(50)),
                "salinity": ("time3", np.arange(50)),
                "temperature": ("time3", np.arange(50)),
            },
            coords={
                "time3": np.arange("2017-06-20T01:00", "2017-06-20T01:25", np.timedelta64(30, "s"), dtype="datetime64[ns]")
            }
        ),
        data_kind="stationary"
    )
    if "time3" in ed.platform and sonar_model != "AD2CP":
        ed.compute_range(stationary_env_params, azfp_cal_type, ek_waveform_mode)
    else:
        try:
            ed.compute_range(stationary_env_params, ek_waveform_mode="CW", azfp_cal_type="Sv")
        except ValueError:
            pass
        else:
            raise AssertionError


    mobile_env_params = EnvParams(
        xr.Dataset(
            data_vars={
                "pressure": ("time", np.arange(100)),
                "salinity": ("time", np.arange(100)),
                "temperature": ("time", np.arange(100)),
            },
            coords={
                "latitude": ("time", rng.random(size=100) + 44),
                "longitude": ("time", rng.random(size=100) - 125),
            }
        ),
        data_kind="mobile"
    )
    if "latitude" in ed.platform and "longitude" in ed.platform and sonar_model != "AD2CP" and not np.isnan(ed.platform["time1"]).all():
        ed.compute_range(mobile_env_params, azfp_cal_type, ek_waveform_mode)
    else:
        try:
            ed.compute_range(mobile_env_params, ek_waveform_mode="CW", azfp_cal_type="Sv")
        except ValueError:
            pass
        else:
            raise AssertionError

    env_params = {"sound_speed": 343}
    if sonar_model == "AD2CP":
        try:
            ed.compute_range(
                env_params, ek_waveform_mode="CW", azfp_cal_type="Sv"
            )
        except ValueError:
            pass  # AD2CP is not currently supported in ed.compute_range
        else:
            raise AssertionError
    else:
        echo_range = ed.compute_range(
            env_params,
            azfp_cal_type,
            ek_waveform_mode,
        )
        assert isinstance(echo_range, xr.DataArray)


def test_nan_range_entries(range_check_files):
    sonar_model, ek_file = range_check_files
    echodata = echopype.open_raw(ek_file, sonar_model=sonar_model)
    if sonar_model == "EK80":
        ds_Sv = echopype.calibrate.compute_Sv(echodata, waveform_mode='BB', encode_mode='complex')
        range_output = echodata.compute_range(env_params=[], ek_waveform_mode='BB')
        nan_locs_backscatter_r = ~echodata.beam.backscatter_r.isel(beam=0).drop("beam").isnull()
    else:
        ds_Sv = echopype.calibrate.compute_Sv(echodata)
        range_output = echodata.compute_range(env_params=[])
        nan_locs_backscatter_r = ~echodata.beam.backscatter_r.isel(beam=0).drop("beam").isnull()

    nan_locs_Sv_range = ~ds_Sv.echo_range.isnull()
    nan_locs_range = ~range_output.isnull()
    assert xr.Dataset.equals(nan_locs_backscatter_r, nan_locs_range)
    assert xr.Dataset.equals(nan_locs_backscatter_r, nan_locs_Sv_range)


@pytest.mark.parametrize(
    ["ext_type", "sonar_model", "updated", "path_model", "raw_path", "platform_data"],
    [
        (
                "external-trajectory",
                "EK80",
                ("pitch", "roll", "longitude", "latitude"),
                "EK80",
                (
                        "saildrone",
                        "SD2019_WCS_v05-Phase0-D20190617-T125959-0.raw",
                ),
                (
                        "saildrone",
                        "saildrone-gen_5-fisheries-acoustics-code-sprint-sd1039-20190617T130000-20190618T125959-1_hz-v1.1595357449818.nc",  #noqa
                ),
        ),
        (
                "fixed-location",
                "EK60",
                ("longitude", "latitude"),
                "EK60",
                (
                        "ooi",
                        "CE02SHBP-MJ01C-07-ZPLSCB101_OOI-D20191201-T000000.raw"
                ),
                (-100.0, -50.0),
        ),
    ],
)
def test_update_platform(
        ext_type,
        sonar_model,
        updated,
        path_model,
        raw_path,
        platform_data,
        test_path
):
    raw_file = test_path[path_model] / raw_path[0] / raw_path[1]
    ed = echopype.open_raw(raw_file, sonar_model=sonar_model)

    for variable in updated:
        assert np.isnan(ed.platform[variable].values).all()

    if ext_type == "external-trajectory":
        extra_platform_data_file_name = platform_data[1]
        extra_platform_data = xr.open_dataset(
            test_path[path_model] / platform_data[0] / extra_platform_data_file_name
        )
    elif ext_type == "fixed-location":
        extra_platform_data_file_name = None
        extra_platform_data = xr.Dataset(
            {
                "longitude": (["time"], np.array([float(platform_data[0])])),
                "latitude": (["time"], np.array([float(platform_data[1])])),
            },
            coords={
                "time": (["time"], np.array([ed['Sonar/Beam_group1'].ping_time.values.min()]))
            },
        )

    ed.update_platform(
        extra_platform_data,
        extra_platform_data_file_name=extra_platform_data_file_name,
    )

    for variable in updated:
        assert not np.isnan(ed.platform[variable].values).all()

    # times have max interval of 2s
    # check times are > min(ed.beam["ping_time"]) - 2s
    assert (
        ed.platform["time1"]
        > ed.beam["ping_time"].min() - np.timedelta64(2, "s")
    ).all()
    # check there is only 1 time < min(ed.beam["ping_time"])
    assert (
        np.count_nonzero(
            ed.platform["time1"] < ed.beam["ping_time"].min()
        )
        <= 1
    )
    # check times are < max(ed.beam["ping_time"]) + 2s
    assert (
        ed.platform["time1"]
        < ed.beam["ping_time"].max() + np.timedelta64(2, "s")
    ).all()
    # check there is only 1 time > max(ed.beam["ping_time"])
    assert (
        np.count_nonzero(
            ed.platform["time1"] > ed.beam["ping_time"].max()
        )
        <= 1
    )
