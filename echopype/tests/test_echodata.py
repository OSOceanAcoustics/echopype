from textwrap import dedent

from echopype.testing import TEST_DATA_FOLDER
from echopype.echodata import EchoData
import xarray as xr

ek60_path = TEST_DATA_FOLDER / "ek60"


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
              > Top-level: contains metadata about the SONAR-netCDF4 file format.
              > Environment: contains information relevant to acoustic propagation through water.
              > Platform: contains information about the platform on which the sonar is installed.
              > Provenance: contains metadata about how the SONAR-netCDF4 version of the data were obtained.
              > Sonar: contains specific metadata for the sonar system.
              > Beam: contains backscatter data and other beam or channel-specific data.
              > Vendor specific: contains vendor-specific information about the sonar and the data."""
        )
        ed = EchoData(converted_raw_path=self.converted_zarr)
        actual = "\n".join(x.rstrip() for x in repr(ed).split("\n"))
        assert expected_repr == actual
