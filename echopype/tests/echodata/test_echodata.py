from textwrap import dedent
from pathlib import Path
import shutil

from datatree import DataTree
from zarr.errors import GroupNotFoundError

from echopype.echodata.echodata import EchoData

import pytest
import xarray as xr
import numpy as np

from echopype.testing import (
    _check_consolidated,
)


class TestEchoData:
    expected_groups = {
        'Top-level',
        'Environment',
        'Platform',
        'Platform/NMEA',
        'Provenance',
        'Sonar',
        'Sonar/Beam_group1',
        'Vendor_specific',
    }

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
        ed['Sonar/Beam_group1'] = ed['Sonar/Beam_group1'].rename(
            {'beam': 'beam_newname'}
        )

        assert sorted(ed['Sonar/Beam_group1'].dims.keys()) == [
            'beam_newname',
            'channel',
            'ping_time',
            'range_sample',
        ]

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

    @staticmethod
    def test__harmonize_env_param_time():
        # Scalar
        p = 10.05
        assert EchoData._harmonize_env_param_time(p=p) == 10.05

        # time1 length=1, should return length=1 numpy array
        p = xr.DataArray(
            data=[1],
            coords={
                "time1": np.array(
                    ["2017-06-20T01:00:00"], dtype="datetime64[ns]"
                )
            },
            dims=["time1"],
        )
        assert EchoData._harmonize_env_param_time(p=p) == 1

        # time1 length>1, interpolate to tareget ping_time
        p = xr.DataArray(
            data=np.array([0, 1]),
            coords={
                "time1": np.arange(
                    "2017-06-20T01:00:00",
                    "2017-06-20T01:00:31",
                    np.timedelta64(30, "s"),
                    dtype="datetime64[ns]",
                )
            },
            dims=["time1"],
        )
        # ping_time target is identical to time1
        ping_time_target = p["time1"].rename({"time1": "ping_time"})
        p_new = EchoData._harmonize_env_param_time(
            p=p, ping_time=ping_time_target
        )
        assert (p_new["ping_time"] == ping_time_target).all()
        assert (p_new.data == p.data).all()
        # ping_time target requires actual interpolation
        ping_time_target = xr.DataArray(
            data=[1],
            coords={
                "ping_time": np.array(
                    ["2017-06-20T01:00:15"], dtype="datetime64[ns]"
                )
            },
            dims=["ping_time"],
        )
        p_new = EchoData._harmonize_env_param_time(
            p=p, ping_time=ping_time_target["ping_time"]
        )
        assert p_new["ping_time"] == ping_time_target["ping_time"]
        assert p_new.data == 0.5

    @pytest.mark.parametrize("consolidated", [True, False])
    def test_to_zarr_consolidated(self, mock_echodata, consolidated):
        """
        Tests to_zarr consolidation. Currently, this test uses a mock EchoData object that only
        has attributes. The consolidated flag provided will be used in every to_zarr call (which
        is used to write each EchoData group to zarr_path).
        """
        zarr_path = Path('test.zarr')
        mock_echodata.to_zarr(
            str(zarr_path), consolidated=consolidated, overwrite=True
        )

        check = True if consolidated else False
        zmeta_path = zarr_path / ".zmetadata"

        assert zmeta_path.exists() is check

        if check is True:
            _check_consolidated(mock_echodata, zmeta_path)

        # clean up the zarr file
        shutil.rmtree(zarr_path)
