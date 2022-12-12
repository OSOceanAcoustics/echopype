import pytest
from echopype.utils.data_proc_lvls import validate_source_Sv
import xarray as xr
import os


@pytest.mark.parametrize(
    ("source_Sv_input", "storage_options_input", "true_file_type"),
    [
        pytest.param(42, {}, None,
                     marks=pytest.mark.xfail(
                         strict=True,
                         reason='This test should fail because Source_Sv is not of the correct type.')
                     ),
        pytest.param(xr.DataArray(), {}, None,
                     marks=pytest.mark.xfail(
                         strict=True,
                         reason='This test should fail because Source_Sv is not of the correct type.')
                     ),
        pytest.param({}, 42, None,
                     marks=pytest.mark.xfail(
                         strict=True,
                         reason='This test should fail because storage_options is not of the correct type.')
                     ),
        (xr.Dataset(attrs={"test": 42}), {}, None),
        (os.path.join('folder', 'new_test.nc'), {}, 'netcdf4'),
        (os.path.join('folder', 'new_test.zarr'), {}, 'zarr')
    ]

)
def test_validate_source_Sv(source_Sv_input, storage_options_input, true_file_type):
    """
    Tests that ``validate_source_Sv`` has the appropriate outputs. It should
    be noted that an exhaustive combination of ``source_Sv`` and ``storage_options``
    has not been implemented when ``source_Sv`` is a path. These are tested in
    ``test_utils_io.py::test_validate_output_path``.
    """

    source_Sv_output, file_type_output = validate_source_Sv(source_Sv_input, storage_options_input)

    if isinstance(source_Sv_input, xr.Dataset):
        assert source_Sv_output.identical(source_Sv_input)
        assert file_type_output is None
    else:
        assert isinstance(source_Sv_output, str)
        assert file_type_output == true_file_type



