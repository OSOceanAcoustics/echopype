from textwrap import dedent

import fsspec
from pathlib import Path
import shutil

from datatree import DataTree
from zarr.errors import GroupNotFoundError

import echopype
from echopype.echodata import EchoData
from echopype import open_converted
from echopype.calibrate.calibrate_ek import CalibrateEK60, CalibrateEK80

import pytest
import xarray as xr
import numpy as np

from utils import get_mock_echodata, check_consolidated


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
    expected_groups = (
        'Top-level',
        'Environment',
        'Platform',
        'Platform/NMEA',
        'Provenance',
        'Sonar',
        'Sonar/Beam_group1',
        'Vendor_specific',
    )

    @pytest.fixture(scope="class")
    def mock_echodata(self):
        return get_mock_echodata()

    @pytest.fixture(scope="class")
    def converted_zarr(self, single_ek60_zarr):
        return single_ek60_zarr

    def create_ed(self, converted_raw_path):
        return EchoData.from_file(converted_raw_path=converted_raw_path)

    def test_constructor(self, converted_zarr):
        ed = self.create_ed(converted_zarr)

        assert ed.sonar_model == 'EK60'
        assert ed.converted_raw_path == converted_zarr
        assert ed.storage_options == {}
        for group in self.expected_groups:
            assert isinstance(ed[group], xr.Dataset)

    def test_group_paths(self, converted_zarr):
        ed = self.create_ed(converted_zarr)
        assert ed.group_paths == self.expected_groups

    def test_nbytes(self, converted_zarr):
        ed = self.create_ed(converted_zarr)
        assert isinstance(ed.nbytes, float)
        assert ed.nbytes == 4688692.0

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
            │   └── Beam_group1: contains backscatter power (uncalibrated) and other beam or channel-specific data, including split-beam angle data when they exist.
            └── Vendor_specific: contains vendor-specific information about the sonar and the data."""
        )
        ed = self.create_ed(converted_raw_path=converted_zarr)
        actual = "\n".join(x.rstrip() for x in repr(ed).split("\n"))
        assert expected_repr == actual

    def test_repr_html(self, converted_zarr):
        zarr_path_string = str(converted_zarr.absolute())
        ed = self.create_ed(converted_raw_path=converted_zarr)
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
        ed = self.create_ed(converted_raw_path=converted_zarr)
        current_ed_beam = ed["Sonar/Beam_group1"]
        current_ed_top = ed['Top-level']
        ed["Sonar/Beam_group1"] = sample_data
        ed['Top-level'] = sample_data2

        assert ed["Sonar/Beam_group1"].equals(sample_data) is True
        assert ed["Sonar/Beam_group1"].equals(current_ed_beam) is False

        assert ed['Top-level'].equals(sample_data2) is True
        assert ed['Top-level'].equals(current_ed_top) is False

    def test_getitem(self, converted_zarr):
        ed = self.create_ed(converted_raw_path=converted_zarr)
        beam = ed['Sonar/Beam_group1']
        assert isinstance(beam, xr.Dataset)
        assert ed['MyGroup'] is None

        ed._tree = None
        try:
            ed['Sonar']
        except Exception as e:
            assert isinstance(e, ValueError)

    def test_setitem(self, converted_zarr):
        ed = self.create_ed(converted_raw_path=converted_zarr)
        ed['Sonar/Beam_group1'] = ed['Sonar/Beam_group1'].rename({'beam': 'beam_newname'})

        assert sorted(ed['Sonar/Beam_group1'].dims.keys()) == ['beam_newname', 'channel', 'ping_time', 'range_sample']

        try:
            ed['SomeRandomGroup'] = 'Testing value'
        except Exception as e:
            assert isinstance(e, GroupNotFoundError)

    def test_get_dataset(self, converted_zarr):
        ed = self.create_ed(converted_raw_path=converted_zarr)
        node = DataTree()
        result = ed._EchoData__get_dataset(node)

        ed_node = ed._tree['Sonar']
        ed_result = ed._EchoData__get_dataset(ed_node)

        assert result is None
        assert isinstance(ed_result, xr.Dataset)

    @pytest.mark.parametrize("consolidated", [True, False])
    def test_to_zarr_consolidated(self, mock_echodata, consolidated):
        """
        Tests to_zarr consolidation. Currently, this test uses a mock EchoData object that only
        has attributes. The consolidated flag provided will be used in every to_zarr call (which
        is used to write each EchoData group to zarr_path).
        """
        zarr_path = Path('test.zarr')
        mock_echodata.to_zarr(str(zarr_path), consolidated=consolidated, overwrite=True)

        check = True if consolidated else False
        zmeta_path = zarr_path / ".zmetadata"

        assert zmeta_path.exists() is check

        if check is True:
            check_consolidated(mock_echodata, zmeta_path)

        # clean up the zarr file
        shutil.rmtree(zarr_path)


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


# def test_compute_range(compute_range_samples):
#     (
#         filepath,
#         sonar_model,
#         azfp_xml_path,
#         azfp_cal_type,
#         ek_waveform_mode,
#         ek_encode_mode,
#     ) = compute_range_samples
#     ed = echopype.open_raw(filepath, sonar_model, azfp_xml_path)
#     rng = np.random.default_rng(0)
#     stationary_env_params = EnvParams(
#         xr.Dataset(
#             data_vars={
#                 "pressure": ("time3", np.arange(50)),
#                 "salinity": ("time3", np.arange(50)),
#                 "temperature": ("time3", np.arange(50)),
#             },
#             coords={
#                 "time3": np.arange("2017-06-20T01:00", "2017-06-20T01:25", np.timedelta64(30, "s"), dtype="datetime64[ns]")
#             }
#         ),
#         data_kind="stationary"
#     )
#     if "time3" in ed["Platform"] and sonar_model != "AD2CP":
#         ed.compute_range(stationary_env_params, azfp_cal_type, ek_waveform_mode)
#     else:
#         try:
#             ed.compute_range(stationary_env_params, ek_waveform_mode="CW", azfp_cal_type="Sv")
#         except ValueError:
#             pass
#         else:
#             raise AssertionError


#     mobile_env_params = EnvParams(
#         xr.Dataset(
#             data_vars={
#                 "pressure": ("time", np.arange(100)),
#                 "salinity": ("time", np.arange(100)),
#                 "temperature": ("time", np.arange(100)),
#             },
#             coords={
#                 "latitude": ("time", rng.random(size=100) + 44),
#                 "longitude": ("time", rng.random(size=100) - 125),
#             }
#         ),
#         data_kind="mobile"
#     )
#     if "latitude" in ed["Platform"] and "longitude" in ed["Platform"] and sonar_model != "AD2CP" and not np.isnan(ed["Platform"]["time1"]).all():
#         ed.compute_range(mobile_env_params, azfp_cal_type, ek_waveform_mode)
#     else:
#         try:
#             ed.compute_range(mobile_env_params, ek_waveform_mode="CW", azfp_cal_type="Sv")
#         except ValueError:
#             pass
#         else:
#             raise AssertionError

#     env_params = {"sound_speed": 343}
#     if sonar_model == "AD2CP":
#         try:
#             ed.compute_range(
#                 env_params, ek_waveform_mode="CW", azfp_cal_type="Sv"
#             )
#         except ValueError:
#             pass  # AD2CP is not currently supported in ed.compute_range
#         else:
#             raise AssertionError
#     else:
#         echo_range = ed.compute_range(
#             env_params,
#             azfp_cal_type,
#             ek_waveform_mode,
#         )
#         assert isinstance(echo_range, xr.DataArray)


def test_nan_range_entries(range_check_files):
    sonar_model, ek_file = range_check_files
    echodata = echopype.open_raw(ek_file, sonar_model=sonar_model)
    if sonar_model == "EK80":
        ds_Sv = echopype.calibrate.compute_Sv(echodata, waveform_mode='BB', encode_mode='complex')
        cal_obj = CalibrateEK80(
            echodata, env_params=None, cal_params=None, ecs_file=None, waveform_mode="BB", encode_mode="complex",
        )
        range_output = cal_obj.range_meter
        # broadband complex data EK80 file: always need to drop "beam" dimension
        nan_locs_backscatter_r = ~echodata["Sonar/Beam_group1"].backscatter_r.isel(beam=0).drop_vars("beam").isnull()
    else:
        # EK60 file does not need dropping "beam" dimension
        ds_Sv = echopype.calibrate.compute_Sv(echodata)
        cal_obj = CalibrateEK60(echodata, env_params={}, cal_params=None, ecs_file=None)
        range_output = cal_obj.range_meter
        nan_locs_backscatter_r = ~echodata["Sonar/Beam_group1"].backscatter_r.isnull()

    nan_locs_Sv_range = ~ds_Sv.echo_range.isnull()
    nan_locs_range = ~range_output.isnull()
    assert xr.Dataset.equals(nan_locs_backscatter_r, nan_locs_range)
    assert xr.Dataset.equals(nan_locs_backscatter_r, nan_locs_Sv_range)


@pytest.mark.integration
@pytest.mark.parametrize(
    ["ext_type", "sonar_model", "variable_mappings", "path_model", "raw_path", "platform_data"],
    [
        (
            "external-trajectory",
            "EK80",
            # variable_mappings dictionary as {Platform_var_name: external-data-var-name}
            {"pitch": "PITCH", "roll": "ROLL", "longitude": "longitude", "latitude": "latitude"},
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
            # variable_mappings dictionary as {Platform_var_name: external-data-var-name}
            {"longitude": "longitude", "latitude": "latitude"},
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
        variable_mappings,
        path_model,
        raw_path,
        platform_data,
        test_path
):
    raw_file = test_path[path_model] / raw_path[0] / raw_path[1]
    ed = echopype.open_raw(raw_file, sonar_model=sonar_model)

    # Test that the variables in Platform are all empty (nan)
    for variable in variable_mappings.keys():
        assert np.isnan(ed["Platform"][variable].values).all()

    # Prepare the external data
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

    # Run update_platform
    ed.update_platform(
        extra_platform_data,
        variable_mappings=variable_mappings,
        extra_platform_data_file_name=extra_platform_data_file_name,
    )

    for variable in variable_mappings.keys():
        assert not np.isnan(ed["Platform"][variable].values).all()

    # times have max interval of 2s
    # check times are > min(ed["Sonar/Beam_group1"]["ping_time"]) - 2s
    assert (
        ed["Platform"]["time3"]
        > ed["Sonar/Beam_group1"]["ping_time"].min() - np.timedelta64(2, "s")
    ).all()
    # check there is only 1 time < min(ed["Sonar/Beam_group1"]["ping_time"])
    assert (
        np.count_nonzero(
            ed["Platform"]["time3"] < ed["Sonar/Beam_group1"]["ping_time"].min()
        )
        <= 1
    )
    # check times are < max(ed["Sonar/Beam_group1"]["ping_time"]) + 2s
    assert (
        ed["Platform"]["time3"]
        < ed["Sonar/Beam_group1"]["ping_time"].max() + np.timedelta64(2, "s")
    ).all()
    # check there is only 1 time > max(ed["Sonar/Beam_group1"]["ping_time"])
    assert (
        np.count_nonzero(
            ed["Platform"]["time3"] > ed["Sonar/Beam_group1"]["ping_time"].max()
        )
        <= 1
    )


@pytest.mark.integration
def test_update_platform_multidim(test_path):
    raw_file = test_path["EK60"] / "ooi" / "CE02SHBP-MJ01C-07-ZPLSCB101_OOI-D20191201-T000000.raw"
    ed = echopype.open_raw(raw_file, sonar_model="EK60")

    extra_platform_data = xr.Dataset(
        {
            "lon": (["time"], np.array([-100.0])),
            "lat": (["time"], np.array([-50.0])),
            "pitch": (["time_pitch"], np.array([0.1])),
            "waterlevel": ([], float(10)),
        },
        coords={
            "time": (["time"], np.array([ed['Sonar/Beam_group1'].ping_time.values.min()])),
            "time_pitch": (
                ["time_pitch"],
                # Adding a time delta is not necessary, but it may be handy if we later
                # want to expand the scope of this test
                np.array([ed['Sonar/Beam_group1'].ping_time.values.min()]) + np.timedelta64(5, "s")
            )
        },
    )

    platform_preexisting_dims = ed["Platform"].dims

    variable_mappings = {
        "longitude": "lon",
        "latitude": "lat",
        "pitch": "pitch",
        "water_level": "waterlevel"
    }
    ed.update_platform(extra_platform_data, variable_mappings=variable_mappings)

    # Updated variables are not all nan
    for variable in variable_mappings.keys():
        assert not np.isnan(ed["Platform"][variable].values).all()

    # Number of dimensions in Platform group and addition of time3 and time4
    assert len(ed["Platform"].dims) == len(platform_preexisting_dims) + 2
    assert "time3" in ed["Platform"].dims
    assert "time4" in ed["Platform"].dims

    # Dimension assignment
    assert ed["Platform"]["longitude"].dims[0] == ed["Platform"]["latitude"].dims[0]
    assert ed["Platform"]["pitch"].dims[0] != ed["Platform"]["longitude"].dims[0]
    assert ed["Platform"]["longitude"].dims[0] not in platform_preexisting_dims
    assert ed["Platform"]["pitch"].dims[0] not in platform_preexisting_dims
    # scalar variable
    assert len(ed["Platform"]["water_level"].dims) == 0


@pytest.mark.integration
@pytest.mark.parametrize(
    ["variable_mappings"],
    [
        pytest.param(
            # lat and lon both exist, but aligned on different time dimension: should fail
            {"longitude": "lon", "latitude": "lat"},
            marks=pytest.mark.xfail(strict=True, reason="Fail since lat and lon not on the same time dimension")
        ),
        pytest.param(
            # only lon exists: should fail
            {"longitude": "lon"},
            marks=pytest.mark.xfail(strict=True, reason="Fail since only lon exists without lat")
        ),
    ],
    ids=[
        "lat_lon_diff_time",
        "lon_only"
    ]
)
def test_update_platform_latlon(test_path, variable_mappings):
    raw_file = test_path["EK60"] / "ooi" / "CE02SHBP-MJ01C-07-ZPLSCB101_OOI-D20191201-T000000.raw"
    ed = echopype.open_raw(raw_file, sonar_model="EK60")

    if "latitude" in variable_mappings:
        extra_platform_data = xr.Dataset(
            {
                "lon": (["time1"], np.array([-100.0])),
                "lat": (["time2"], np.array([-50.0])),
            },
            coords={
                "time1": (["time1"], np.array([ed['Sonar/Beam_group1'].ping_time.values.min()])),
                "time2": (["time2"], np.array([ed['Sonar/Beam_group1'].ping_time.values.min()]) + np.timedelta64(5, "s")),
            },
        )
    else:
        extra_platform_data = xr.Dataset(
            {
                "lon": (["time"], np.array([-100.0])),
            },
            coords={
                "time": (["time"], np.array([ed['Sonar/Beam_group1'].ping_time.values.min()])),
            },
        )

    ed.update_platform(extra_platform_data, variable_mappings=variable_mappings)


@pytest.mark.filterwarnings("ignore:No variables will be updated")
def test_update_platform_no_update(test_path):
    raw_file = test_path["EK60"] / "ooi" / "CE02SHBP-MJ01C-07-ZPLSCB101_OOI-D20191201-T000000.raw"
    ed = echopype.open_raw(raw_file, sonar_model="EK60")

    extra_platform_data = xr.Dataset(
        {
            "lon": (["time"], np.array([-100.0])),
            "lat": (["time"], np.array([-50.0])),
        },
        coords={
            "time": (["time"], np.array([ed['Sonar/Beam_group1'].ping_time.values.min()])),
        },
    )

    # variable names in mappings different from actual external dataset
    variable_mappings = {"longitude": "longitude", "latitude": "latitude"}

    ed.update_platform(extra_platform_data, variable_mappings=variable_mappings)


@pytest.mark.integration
def test_update_platform_latlon_notimestamp(test_path):
    raw_file = test_path["EK60"] / "ooi" / "CE02SHBP-MJ01C-07-ZPLSCB101_OOI-D20191201-T000000.raw"
    ed = echopype.open_raw(raw_file, sonar_model="EK60")

    extra_platform_data = xr.Dataset(
        {
            "lon": ([], float(-100.0)),
            "lat": ([], float(-50.0)),
        }
    )

    platform_preexisting_dims = ed["Platform"].dims

    # variable names in mappings different from actual external dataset
    variable_mappings = {"longitude": "lon", "latitude": "lat"}

    ed.update_platform(extra_platform_data, variable_mappings=variable_mappings)

    # Updated variables are not all nan
    for variable in variable_mappings.keys():
        assert not np.isnan(ed["Platform"][variable].values).all()

    # Number of dimensions in Platform group should be as previous
    assert len(ed["Platform"].dims) == len(platform_preexisting_dims)

    # Dimension assignment
    assert ed["Platform"]["longitude"].dims[0] == ed["Platform"]["latitude"].dims[0]
    assert ed["Platform"]["longitude"].dims[0] in platform_preexisting_dims
    assert ed["Platform"]["latitude"].dims[0] in platform_preexisting_dims
    assert ed['Platform']['longitude'].coords['time1'].values[0] == ed['Sonar/Beam_group1'].ping_time.data[0]


@pytest.mark.unit
@pytest.mark.parametrize(
    "chunk_dict",
    [
        ({"ping_time": 10, "range_sample": 100}),
        ({"time1": 10, "time2": 10}),
    ],
)
def test_echodata_chunk(chunk_dict):
    # Parse Raw File
    ed = echopype.open_raw(
        "echopype/test_data/ek60/DY1801_EK60-D20180211-T164025.raw",
        sonar_model="EK60"
    )

    # Chunk Echodata object
    ed = ed.chunk(chunk_dict)

    # Iterate through groups
    ed_group_map = ed.group_map
    for key in ed_group_map.keys():
        echodata_group = ed_group_map[key]["ep_group"]
        if echodata_group is not None:
            group = ed[echodata_group]
            if group is not None:
                # Grab group chunk dict
                group_chunk_dict = group.chunks

                # Get shared dimensions
                group_dims = set(list(group_chunk_dict.keys()))
                chunk_dims = set(chunk_dict.keys())
                shared_dims = group_dims & chunk_dims

                # Check that all but the last chunk are equal to the desired chunk size
                # Check that the last chunk is equal or less than the desired chunk size
                for key in shared_dims:
                    desired_chunk_size = chunk_dict[key]
                    chunk_sizes = group_chunk_dict[key]
                    for chunk_size in chunk_sizes[:-1]:
                        assert chunk_size == desired_chunk_size
                    assert chunk_sizes[-1] <= desired_chunk_size
