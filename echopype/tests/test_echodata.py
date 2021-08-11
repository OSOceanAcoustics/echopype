from textwrap import dedent

import fsspec

import echopype
from echopype.testing import TEST_DATA_FOLDER
from echopype.echodata import EchoData
from echopype import open_converted

import pytest
import xarray as xr

ek60_path = TEST_DATA_FOLDER / "ek60"
ek80_path = TEST_DATA_FOLDER / "ek80"
azfp_path = TEST_DATA_FOLDER / "azfp"
ad2cp_path = TEST_DATA_FOLDER / "ad2cp"


class TestEchoData:
    converted_zarr = (
        ek60_path / "ncei-wcsd" / "Summer2017-D20170615-T190214.zarr"
    )

    def test_constructor(self):
        ed = EchoData(converted_raw_path=self.converted_zarr)
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
        assert ed.converted_raw_path == self.converted_zarr
        assert ed.storage_options == {}
        for group in expected_groups:
            assert isinstance(getattr(ed, group), xr.Dataset)

    def test_repr(self):
        zarr_path_string = str(self.converted_zarr.absolute())
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
        ed = EchoData(converted_raw_path=self.converted_zarr)
        actual = "\n".join(x.rstrip() for x in repr(ed).split("\n"))
        assert expected_repr == actual


@pytest.mark.parametrize(
    "converted_zarr",
    [
        (ek60_path / "ncei-wcsd" / "Summer2017-D20170615-T190214.zarr"),
        str(ek60_path / "ncei-wcsd" / "Summer2017-D20170615-T190214.zarr"),
        (
            ek60_path / "ncei-wcsd" / "Summer2017-D20170615-T190214.nc"
        ),  # netcdf test
        "s3://data/ek60/ncei-wcsd/Summer2017-D20170615-T190214.nc",  # netcdf test
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
)
def test_open_converted(
    converted_zarr,
    minio_bucket  # noqa
):
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
    if not isinstance(converted_zarr, fsspec.FSMap):
        storage_options = _check_path(str(converted_zarr))

    try:
        ed = open_converted(converted_zarr, storage_options=storage_options)
        assert isinstance(ed, EchoData) is True
    except Exception as e:
        if (
            isinstance(converted_zarr, str)
            and converted_zarr.startswith("s3://")
            and converted_zarr.endswith(".nc")
        ):
            assert isinstance(e, ValueError) is True


@pytest.mark.parametrize(
    ("filepath", "sonar_model", "azfp_xml_path", "azfp_cal_type", "ek_waveform_mode", "ek_encode_mode"),
    [
        (ek60_path / "ncei-wcsd" / "Summer2017-D20170615-T190214.raw", "EK60", None, None, "CW", "complex"),
        (ek80_path / "D20190822-T161221.raw", "EK80", None, None, "CW", "power"),
        (ek80_path / "D20170912-T234910.raw", "EK80", None, None, "BB", "complex"),
        (azfp_path / "ooi" / "17032923.01A", "AZFP", azfp_path / "ooi" / "17032922.XML", "Sv", None, None),
        (azfp_path / "ooi" / "17032923.01A", "AZFP", azfp_path / "ooi" / "17032922.XML", "Sp", None, None),
        (ad2cp_path / "raw" / "090" / "rawtest.090.00001.ad2cp", "AD2CP", None, None, None, None)
    ]
)
def test_compute_range(filepath, sonar_model, azfp_xml_path, azfp_cal_type, ek_waveform_mode, ek_encode_mode):
    ed = echopype.open_raw(filepath, sonar_model, azfp_xml_path)
    env_params = {"sound_speed": 343}

    if sonar_model == "AD2CP":
        try:
            ed.compute_range(env_params, ek_waveform_mode="CW", azfp_cal_type="Sv")
        except ValueError:
            return
        else:
            raise AssertionError
    else:
        range = ed.compute_range(env_params, azfp_cal_type, ek_waveform_mode, )
        assert isinstance(range, xr.DataArray)
