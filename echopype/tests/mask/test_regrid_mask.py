import pandas as pd
import xarray as xr
import numpy as np

import pytest

from echopype.mask import regrid_mask


@pytest.mark.parametrize(
    ("dtype", "func"),
    [
        ("int", "logical-AND"),
        ("int", "logical-OR"),
        ("bool", "logical-AND"),
        ("bool", "logical-OR"),
    ]
)
def test_regrid_non_boundary_mask_2D(dtype, func):
    """
    Test mask regridding for a 2D array by checking logical-AND and logical-OR outputs.
    """
    # Create mask array
    input_array = np.array(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, True],
            [False, False, True, False],
        ]
    )

    # Create expected array
    if func == "logical-AND":
        expected_array = np.array(
            [
                [True,  False],
                [False, False],
            ]
        )
    elif func == "logical-OR":
        expected_array = np.array(
            [
                [True,  False],
                [False, True],
            ]
        )
    if dtype == "int":
        input_array = input_array.astype(np.int64)
        expected_array = expected_array.astype(np.int64)
    mask = xr.DataArray(
        input_array,
        dims=("ping_time", "depth"),
    )
    mask = mask.assign_coords(
        {
            "ping_time": pd.to_datetime(
                [
                    "2020-01-01T01:00:01.000000000",
                    "2020-01-01T01:00:19.000000000",
                    "2020-01-01T01:00:21.000000000",
                    "2020-01-01T01:00:39.000000000",
                ]
            ),
            "depth": [1.0, 19.0, 21.0, 39.0]
        }
    )

    # Regrid mask
    mask_regridded_da = regrid_mask(
        mask,
        range_da=mask["depth"],
        range_bin="20m",
        ping_time_bin="20s",
        func=func,
    )

    # Check that expected result is what is received
    mask_expected_da = xr.DataArray(
        expected_array,
        dims=("ping_time", "depth"),
    )
    mask_expected_da = mask_expected_da.assign_coords(
        {
            "ping_time": pd.to_datetime(
                [
                    "2020-01-01T01:00:00.000000000",
                    "2020-01-01T01:00:20.000000000",
                ]
            ),
            "depth": [0.0, 20.0]
        }
    )
    assert mask_regridded_da.equals(mask_expected_da)


@pytest.mark.parametrize(
    ("closed", "func"),
    [
        ("left", "logical-AND"),
        ("left", "logical-OR"),
        ("right", "logical-AND"),
        ("right", "logical-OR"),
    ]
)
def test_regrid_boundary_mask_2D(closed, func):
    """
    Test 2d mask regridding for when points are on boundaries.
    """
    # Create mask array
    input_array = np.array(
        [
            [True, True, False, False, False],
            [True, True, False, False, False],
            [False, False, True, True, False],
            [False, False, True, False, False],
            [False, False, False, False, False],
        ]
    )

    # Create expected array
    if func == "logical-AND" and closed == "left":
        expected_array = np.array(
            [
                [True,  False, False],
                [False, False, False],
                [False, False, False],
            ]
        )
    elif func == "logical-OR" and closed == "left":
        expected_array = np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, False],
            ]
        )
    elif func == "logical-AND" and closed == "right":
        expected_array = np.array(
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ]
        )
    elif func == "logical-OR" and closed == "right":
        expected_array = np.array(
            [
                [True, True, False],
                [True, False, False],
                [False, False, False],
            ]
        )
    mask = xr.DataArray(
        input_array,
        dims=("ping_time", "depth"),
    )
    mask = mask.assign_coords(
        {
            "ping_time": pd.to_datetime(
                [
                    "2020-01-01T01:00:00.000000000",
                    "2020-01-01T01:00:10.000000000",
                    "2020-01-01T01:00:20.000000000",
                    "2020-01-01T01:00:30.000000000",
                    "2020-01-01T01:00:40.000000000",
                ]
            ),
            "depth": [0.0, 10.0, 20.0, 30.0, 40.0]
        }
    )

    # Regrid mask
    mask_regridded_da = regrid_mask(
        mask,
        range_da=mask["depth"],
        range_bin="20m",
        ping_time_bin="20s",
        func=func,
        closed=closed,
    )

    # Check that expected result is what is received
    mask_expected_da = xr.DataArray(
        expected_array,
        dims=("ping_time", "depth"),
    ).assign_coords(
        {
            "ping_time": pd.to_datetime(
                [
                    "2020-01-01T01:00:00",
                    "2020-01-01T01:00:20",
                    "2020-01-01T01:00:40",
                ]
            ),
            "depth": [0.0, 20.0, 40.0],
        }
    )
    assert mask_regridded_da.equals(mask_expected_da)


@pytest.mark.integration
def test_regrid_mask_3D():
    """
    Test mask regridding for a 3D array.
    """
    # Create mask
    mask = xr.DataArray(
        np.array(
            [
                [
                    [True, True, False, False],
                    [True, True, False, False],
                    [False, False, True, True],
                    [False, False, True, False],
                ],
                [
                    [False, True, False, False],
                    [True, True, False, False],
                    [False, False, True, True],
                    [False, False, True, True],
                ]
            ]
        ),
        dims=("region_id", "ping_time", "depth"),
    )
    mask = mask.assign_coords(
        {
            "region_id": [1, 2],
            "ping_time": pd.to_datetime(
                [
                    "2020-01-01T01:00:01.000000000",
                    "2020-01-01T01:00:19.000000000",
                    "2020-01-01T01:00:21.000000000",
                    "2020-01-01T01:00:39.000000000",
                ]
            ),
            "depth": [1.0, 19.0, 21.0, 39.0],
        }
    )

    # Regrid mask
    mask_regridded_da = regrid_mask(
        mask,
        range_da=mask["depth"],
        range_bin="20m",
        ping_time_bin="20s",
        third_dim="region_id",
        func="logical-OR",
        range_var_max="39.0m",
    )

    # Check that expected result is what is received
    mask_expected_da = xr.DataArray(
        np.array(
            [
                [
                    [True, False],
                    [False, True],
                ],
                [
                    [True, False],
                    [False, True],
                ]
            ]
        ),
        dims=("region_id", "ping_time", "depth"),
    )
    mask_expected_da = mask_expected_da.assign_coords(
        {
            "region_id": [1, 2],
            "ping_time": pd.to_datetime(
                [
                    "2020-01-01T01:00:00.000000000",
                    "2020-01-01T01:00:20.000000000",
                ]
            ),
            "depth": [0.0, 20.0]
        }
    )
    assert mask_regridded_da.equals(mask_expected_da)


@pytest.mark.integration
def test_regrid_mask_errors():
    "Test that errors are raised correctly."

    # Create valid mask
    input_array = np.array(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, True],
            [False, False, True, False],
        ]
    )
    mask = xr.DataArray(
        input_array,
        dims=("ping_time", "depth"),
    )
    mask = mask.assign_coords(
        {
            "ping_time": pd.to_datetime(
                [
                    "2020-01-01T01:00:10.000000000",
                    "2020-01-01T01:00:20.000000000",
                    "2020-01-01T01:00:30.000000000",
                    "2020-01-01T01:00:40.000000000",
                ]
            ),
            "depth": [10, 20, 30, 40]
        }
    )

    # Test errors

    with pytest.raises(ValueError, match="Passing in reindex=.*only allowed when method='map_reduce'"):
        regrid_mask(
            mask,
            range_da=mask["depth"],
            method="blockwise",
            reindex=False,
        )

    with pytest.raises(TypeError, match="ping_time_bin must be a string"):
        regrid_mask(
            mask,
            range_da=mask["depth"],
            ping_time_bin=20,
        )

    with pytest.raises(ValueError, match="'func' must be 'logical-AND' or 'logical-OR'."):
        regrid_mask(
            mask,
            range_da=mask["depth"],
            func="invalid_func",
        )

    with pytest.raises(ValueError, match="Mask must have only 2 dimensions unless 'third_dim' is specified."):
        regrid_mask(
            mask.expand_dims("region_id"),
            range_da=mask["depth"],
            third_dim=None,
        )
    
    with pytest.raises(ValueError, match="Mask must contain the specified 'region_id' as a dimension."):
        regrid_mask(
            mask,
            range_da=mask["depth"],
            third_dim="region_id",
        )

    with pytest.raises(ValueError, match="Mask must have 3 dimensions when 'third_dim' is specified."):
        regrid_mask(
            mask.expand_dims("region_id_1").expand_dims("region_id_2"),
            range_da=mask["depth"],
            third_dim="region_id_1",
        )

    # Create invalid mask
    input_invalid_array = np.array(
        [
            [1, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, np.nan], # NaN also invalid for binary mask
        ]
    )
    invalid_mask = xr.DataArray(
        input_invalid_array,
        dims=("ping_time", "depth"),
    )
    invalid_mask = invalid_mask.assign_coords(
        {
            "ping_time": pd.to_datetime(
                [
                    "2020-01-01T01:00:10.000000000",
                    "2020-01-01T01:00:20.000000000",
                    "2020-01-01T01:00:30.000000000",
                    "2020-01-01T01:00:40.000000000",
                ]
            ),
            "depth": [10, 20, 30, 40]
        }
    )

    # Test invalid mask
    with pytest.raises(ValueError, match = "Mask must be binary True/False or 1/0."):
        regrid_mask(
            invalid_mask,
            range_da=mask["depth"],
        )
