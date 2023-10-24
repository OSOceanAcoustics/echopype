import numpy as np
import pytest
import echopype.utils.mask_transformation_xr as ep
import xarray as xr


def test_lin():
    # Prepare input and expected output
    variable = xr.DataArray([10, 20, 30])
    expected_output = xr.DataArray([10, 100, 1000])

    # Apply function
    output = ep.lin(variable)

    # Assert output is as expected
    xr.testing.assert_equal(output, expected_output)


def test_log():
    # Prepare input and expected output
    variable = xr.DataArray([10, 100, 0, -10])
    expected_output = xr.DataArray([10, 20, -999, -999])

    # Apply function
    output = ep.log(variable)
    # Assert output is as expected
    truth = output == expected_output
    assert truth.all()
    # Test with a single positive number
    assert ep.log(10) == 10 * np.log10(10)

    # Test with a single negative number (should return -999)
    assert ep.log(-10) == -999

    # Test with zero (should return -999)
    assert ep.log(0) == -999

    # Test with a list of numbers
    truth = ep.log([10, 20, -10, 0]) == xr.DataArray(
        [
            10 * np.log10(10),
            10 * np.log10(20),
            -999,
            -999,
        ]
    )
    assert truth.all()

    # Test with an integer numpy array
    int_array = xr.DataArray([10, 20, -10, 0])
    truth = ep.log(int_array) == xr.DataArray([10 * np.log10(10), 10 * np.log10(20), -999, -999])
    assert truth.all()

    # Test with a single number in a numpy array
    assert ep.log(np.array([10])) == 10 * np.log10(10)

    # Test with a single negative number in a numpy array (should return -999)
    assert ep.log(np.array([-10])) == -999

    # Test with a single zero in a numpy array (should return -999)
    assert ep.log(np.array([0])) == -999


def test_downsample_exceptions():
    data = np.arange(24).reshape(4, 6)
    dims = ["x", "y"]
    coords = {"x": np.arange(4), "y": np.arange(6)}
    dataset = xr.DataArray(data=data, dims=dims, coords=coords)

    with pytest.raises(Exception, match="Operation not in approved list"):
        ep.downsample(dataset, {"x": 2}, "kind")
    with pytest.raises(Exception, match="Coordinate z not in dataset coordinates"):
        ep.downsample(dataset, {"z": 2}, "mean")


@pytest.mark.parametrize(
    "coordinates,operation,is_log,shape,value",
    [
        ({"range_sample": 2}, "mean", False, (3, 572, 1891), -10.82763365585262),
        ({"range_sample": 2, "ping_time": 2}, "mean", False, (3, 286, 1891), -11.018715656585043),
        ({"range_sample": 2}, "sum", False, (3, 572, 1891), -21.65526731170524),
        ({"range_sample": 2}, "mean", True, (3, 572, 1891), -10.825779607785),
    ],
)
def test_downsample(sv_dataset_jr230, coordinates, operation, is_log, shape, value):
    source_Sv = sv_dataset_jr230.copy(deep=True)["Sv"]
    res = ep.downsample(source_Sv, coordinates, operation, is_log)
    assert res.values.shape == shape
    assert res.values[-1, -1, -1] == value


def test_upsample():
    data = np.array([[3.5, 4.5, 5.5, 6.5, 7.5], [13.5, 14.5, 15.5, 16.5, 17.5]])
    data_2 = np.arange(25).reshape(5, 5)
    data_3 = np.array(
        [
            [3.5, 4.5, 5.5, 6.5, 7.5],
            [3.5, 4.5, 5.5, 6.5, 7.5],
            [13.5, 14.5, 15.5, 16.5, 17.5],
            [13.5, 14.5, 15.5, 16.5, 17.5],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )
    dims = ["x", "y"]
    coords_1 = {"x": [1, 4], "y": [1, 3, 5, 7, 9]}
    coords_2 = {"x": [1, 2, 3, 4, 5], "y": [1, 3, 5, 7, 9]}
    ds_1 = xr.DataArray(data=data, dims=dims, coords=coords_1)
    ds_2 = xr.DataArray(data=data_2, dims=dims, coords=coords_2)
    ds_3 = xr.DataArray(data=data_3, dims=dims, coords=coords_2)
    ds_4 = ep.upsample(ds_1, ds_2)
    assert ds_3.equals(ds_4)


def test_line_to_square():
    row = [False, False, True, False]
    one = xr.DataArray(data=[row], dims=["x", "y"], coords={"x": [1], "y": [1, 2, 3, 4]})
    two = xr.DataArray(
        data=[row, row, row], dims=["x", "y"], coords={"x": [1, 2, 3], "y": [1, 2, 3, 4]}
    )
    res = ep.line_to_square(one, two, dim="x")
    print(res)
    assert res.shape == two.shape
