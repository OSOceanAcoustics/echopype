import numpy as np
import pytest
import re
import echopype.utils.mask_transformation


def test_lin():
    # Prepare input and expected output
    variable = np.array([10, 20, 30])
    expected_output = np.array([10, 100, 1000])

    # Apply function
    output = echopype.utils.mask_transformation.lin(variable)

    # Assert output is as expected
    np.testing.assert_almost_equal(output, expected_output)


def test_log():
    # Prepare input and expected output
    variable = np.array([10, 100, 0, -10])
    expected_output = np.array([10, 20, -999, -999])

    # Apply function
    output = echopype.utils.mask_transformation.log(variable)

    # Assert output is as expected
    np.testing.assert_almost_equal(output, expected_output, decimal=4)
    # Test with a single positive number
    assert echopype.utils.mask_transformation.log(10) == 10 * np.log10(10)

    # Test with a single negative number (should return -999)
    assert echopype.utils.mask_transformation.log(-10) == -999

    # Test with zero (should return -999)
    assert echopype.utils.mask_transformation.log(0) == -999

    # Test with a list of numbers
    assert echopype.utils.mask_transformation.log([10, 20, -10, 0]) == [
        10 * np.log10(10),
        10 * np.log10(20),
        -999,
        -999,
    ]

    # Test with a numpy array of numbers
    assert np.array_equal(
        echopype.utils.mask_transformation.log(np.array([10, 20, -10, 0])),
        np.array([10 * np.log10(10), 10 * np.log10(20), -999, -999]),
    )

    # Test with an integer numpy array
    int_array = np.array([10, 20, -10, 0])
    assert np.array_equal(
        echopype.utils.mask_transformation.log(int_array),
        np.array([10 * np.log10(10), 10 * np.log10(20), -999, -999]),
    )

    # Test with a single number in a numpy array
    assert echopype.utils.mask_transformation.log(np.array([10])) == 10 * np.log10(10)

    # Test with a single negative number in a numpy array (should return -999)
    assert echopype.utils.mask_transformation.log(np.array([-10])) == -999

    # Test with a single zero in a numpy array (should return -999)
    assert echopype.utils.mask_transformation.log(np.array([0])) == -999


def test_dim2ax():
    # Prepare input and expected output
    dim = np.array([0, 2, 4, 6, 8, 10])
    ax = np.array([0, 1, 2, 3, 4, 5])
    dimrs = np.array([0, 3, 6, 9])
    expected_output = np.array([0, 1.5, 3.0, 4.5])

    # Apply function
    output = echopype.utils.mask_transformation.dim2ax(dim, ax, dimrs)

    # Assert output is as expected
    np.testing.assert_almost_equal(output, expected_output)

    # Sample data
    dim = np.array([0, 2, 4, 6, 8, 10])
    ax = np.array([0, 1, 2, 3, 4, 5])
    dimrs = np.array([0, 3, 6, 9])

    # Expected output
    expected = np.array([0, 1.5, 3, 4.5])

    # Assertions to check the output
    assert np.array_equal(echopype.utils.mask_transformation.dim2ax(dim, ax, dimrs), expected)

    # Sample data
    dim = np.array([0, 10])
    ax = np.array([0, 1])
    dimrs = np.array([0, 5, 10])

    # Expected output
    expected = np.array([0, 0.5, 1])

    # Assertions to check the output
    assert np.array_equal(echopype.utils.mask_transformation.dim2ax(dim, ax, dimrs), expected)

    # Sample data
    dim = np.array([0, 2, 4, 6, 8, 10])
    ax = np.array([0, 1, 2, 3, 4, 5])
    dimrs = np.array([-1, 3, 6, 11])

    # Test with invalid dimrs
    with pytest.raises(
        Exception, match="resampling dimension can not exceed the original dimension limits"
    ):
        echopype.utils.mask_transformation.dim2ax(dim, ax, dimrs)

    # Sample data
    epoch = np.datetime64("1970-01-01T00:00:00")
    dim = np.array([epoch, epoch + np.timedelta64(2, "D"), epoch + np.timedelta64(4, "D")])
    ax = np.array([0, 1, 2])
    dimrs = np.array([epoch, epoch + np.timedelta64(3, "D")])

    # Expected output
    expected = np.array([0, 1.5])

    # Assertions to check the output
    assert np.array_equal(echopype.utils.mask_transformation.dim2ax(dim, ax, dimrs), expected)


def test_oned():
    # Creating a mock data
    data = np.array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]
    )

    # Example dimension
    dim = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    # Resampling dimension intervals
    rvals = np.array([0.5, 2.5, 4.5])

    # Resampling axis
    axis = 0

    # Expected output
    expected_data = np.array([[3.5, 4.5, 5.5, 6.5, 7.5], [13.5, 14.5, 15.5, 16.5, 17.5]])
    expected_dim = np.array([0.5, 2.5, 4.5])
    expected_percentage = np.array(
        [[100.0, 100.0, 100.0, 100.0, 100.0], [100.0, 100.0, 100.0, 100.0, 100.0]]
    )

    # Calling the oned function
    result_data, result_dim, result_percentage = echopype.utils.mask_transformation.oned(
        data, dim, rvals, axis, log_var=False, operation="mean"
    )

    # Asserting the output
    np.testing.assert_array_equal(result_data, expected_data)
    np.testing.assert_array_equal(result_dim, expected_dim)
    np.testing.assert_array_equal(result_percentage, expected_percentage)

    # Sample data
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dim = np.array([1, 2, 3])
    rvals = np.array([1, 3])

    # Test with axis set to 1
    resampled_data, resampled_dim, percentage = echopype.utils.mask_transformation.oned(
        data, dim, rvals, axis=1
    )
    # Assertions to check the output
    assert np.array_equal(resampled_data, np.array([[1.5], [4.5], [7.5]]))
    assert np.array_equal(resampled_dim, rvals)
    assert np.array_equal(percentage, np.array([[100.0], [100.0], [100.0]]))

    # Test with data width not equal to dim length
    with pytest.raises(Exception, match="data width and j dimension length must be equal"):
        echopype.utils.mask_transformation.oned(np.array([[1, 2], [4, 5]]), dim, rvals, axis=1)


def test_oned_exceptions():
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    dim = np.array([0, 1])
    rvals = np.array([2])
    axis = 3

    # Testing invalid axis
    with pytest.raises(Exception, match="axis must be 0 or 1"):
        echopype.utils.mask_transformation.oned(
            data, dim, rvals, axis, log_var=False, operation="mean"
        )

    # Testing invalid resampling intervals
    rvals = np.array([2])
    axis = 0
    with pytest.raises(Exception, match="length of resampling intervals must be >2"):
        echopype.utils.mask_transformation.oned(
            data, dim, rvals, axis, log_var=False, operation="mean"
        )

    # Testing invalid dim range
    rvals = np.array([0, 0.5, 2])
    with pytest.raises(Exception, match="resampling intervals must be within dim range"):
        echopype.utils.mask_transformation.oned(
            data, dim, rvals, axis, log_var=False, operation="mean"
        )

    # Testing unrecognised operation
    rvals = np.array([0, 0.5, 1])
    with pytest.raises(Exception, match="Operation not recognised"):
        echopype.utils.mask_transformation.oned(
            data, dim, rvals, axis, log_var=False, operation="invalid operation"
        )


def test_full_function():
    # Resampled data
    data = np.array([[3.5, 4.5, 5.5, 6.5, 7.5], [13.5, 14.5, 15.5, 16.5, 17.5]])
    # Axis data representing indices
    jax = np.arange(len(data[0]))
    # Dimension values
    dim = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    # Resampling dimension intervals
    rvals = np.array([0.5, 2.5, 4.5])

    # Expected data
    expected_data = np.array(
        [
            [3.5, 4.5, 5.5, 6.5, 7.5],
            [3.5, 4.5, 5.5, 6.5, 7.5],
            [13.5, 14.5, 15.5, 16.5, 17.5],
            [13.5, 14.5, 15.5, 16.5, 17.5],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )

    expected_mask = np.array(
        [
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
            [True, True, True, True, True],
        ]
    )

    # Output data
    output_data, output_mask = echopype.utils.mask_transformation.full(data, rvals, jax, dim, jax)

    np.testing.assert_array_equal(output_data, expected_data)
    np.testing.assert_array_equal(output_mask, expected_mask)


def test_full_function_exceptions():
    data = np.array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]
    )
    dim = [0.5, 1.5, 2.5, 3.5, 4.5]

    with pytest.raises(Exception, match="i resampling interval length must be >2"):
        echopype.utils.mask_transformation.full(data, [0.5], [0.5, 2.5, 4.5], dim, dim)
    with pytest.raises(Exception, match="j resampling interval length must be >2"):
        echopype.utils.mask_transformation.full(data, [0.5, 2.5, 4.5], [0.5], dim, dim)
    # Sample data and dimensions
    datar = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    idim = np.array([1, 2, 3])
    jdim = np.array([1, 2, 3])
    irvals_valid = np.array([1, 2, 3])
    jrvals_valid = np.array([1, 2, 3])
    irvals_invalid = np.array([1, 2, 3, 4, 5])
    jrvals_invalid = np.array([1, 2, 3, 4, 5])

    # Test with valid resampling intervals
    echopype.utils.mask_transformation.full(datar, irvals_valid, jrvals_valid, idim, jdim)

    # Test with invalid i resampling intervals
    with pytest.raises(
        Exception, match=re.escape("i resampling intervals length can't exceed data height + 1")
    ):
        echopype.utils.mask_transformation.full(datar, irvals_invalid, jrvals_valid, idim, jdim)

    # Test with invalid j resampling intervals
    with pytest.raises(
        Exception, match=re.escape("j resampling intervals length can't exceed data width + 1")
    ):
        echopype.utils.mask_transformation.full(datar, irvals_valid, jrvals_invalid, idim, jdim)

    # Test with both invalid i and j resampling intervals
    with pytest.raises(Exception):
        echopype.utils.mask_transformation.full(datar, irvals_invalid, jrvals_invalid, idim, jdim)

    # Sample data and parameters
    datar = np.array([[1, 2], [3, 4]])
    idim = np.array([1, 2])
    jdim = np.array([1, 2])
    irvals = np.array([1, 2])
    jrvals = np.array([1, 2])

    # Call the function
    data, mask_ = echopype.utils.mask_transformation.full(datar, irvals, jrvals, idim, jdim)

    # Assertions to check the output
    # (You can add more assertions based on expected output of the function)
    assert not mask_.any()  # Assuming that the mask should be all False for this input

    # Sample data and parameters
    datar = np.array([[1, 2], [3, 4]])
    idim = np.array([1, 2, 3])
    jdim = np.array([1, 2, 3])
    irvals = np.array([1, 3])
    jrvals = np.array([1, 3])

    # Call the function
    data, mask_ = echopype.utils.mask_transformation.full(datar, irvals, jrvals, idim, jdim)

    # Assertions to check the output
    expected_data = np.array([[1, 1, np.nan], [1, 1, np.nan], [np.nan, np.nan, np.nan]])
    assert np.allclose(data, expected_data, equal_nan=True)

    # Check the mask
    expected_mask = np.array([[False, False, True], [False, False, True], [True, True, True]])
    assert np.array_equal(mask_, expected_mask)
    # Sample data and parameters
    datar = np.array([[1, 2], [3, 4]])
    idim = np.array([1, 2])
    jdim = np.array([1, 2, 3])
    irvals = np.array([1, 2])  # This ensures idiff is False
    jrvals = np.array([1, 3])  # This ensures jdiff is True

    # Call the function
    data, mask_ = echopype.utils.mask_transformation.full(datar, irvals, jrvals, idim, jdim)

    # Assertions to check the output
    expected_data = np.array([[1, 1, np.nan], [3, 3, np.nan]])
    assert np.allclose(data, expected_data, equal_nan=True)


def test_twod():
    data = np.array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]
    )

    # Example dimensions
    idim = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    jdim = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    # Resampling dimension intervals
    irvals = np.array([0.5, 2.5, 4.5])
    jrvals = np.array([0.5, 2.5, 4.5])

    # Expected output
    expected_data = np.array([[4, 6], [14, 16]])
    expected_idim = np.array([0.5, 2.5])
    expected_jdim = np.array([0.5, 2.5])
    expected_percentage = np.array([[100, 100], [100, 100]])

    # call function
    (
        result_data,
        result_idim,
        result_jdim,
        result_percentage,
    ) = echopype.utils.mask_transformation.twod(
        data, idim, jdim, irvals, jrvals, log_var=False, operation="mean"
    )

    # Asserting the output
    np.testing.assert_array_equal(result_data, expected_data)
    np.testing.assert_array_equal(result_idim, expected_idim)
    np.testing.assert_array_equal(result_jdim, expected_jdim)
    np.testing.assert_array_equal(result_percentage, expected_percentage)


def test_twod_exceptions():
    data = np.array([[1, 2], [5, 6]])
    idim = np.array([0, 4])
    jdim = np.array([3, 6])
    rvals = np.array([2])

    # Testing invalid resampling intervals
    with pytest.raises(Exception, match="length of i resampling intervals must be >2"):
        echopype.utils.mask_transformation.twod(
            data, idim, jdim, rvals, rvals, log_var=False, operation="mean"
        )
        # Testing invalid resampling intervals
    with pytest.raises(Exception, match="length of j resampling intervals must be >2"):
        echopype.utils.mask_transformation.twod(
            data, idim, jdim, np.array([2, 2]), rvals, log_var=False, operation="mean"
        )

    # Testing invalid dim range
    rvals = np.array([-1, -0.5, 0])
    with pytest.raises(Exception, match="i resampling intervals must be within idim range"):
        echopype.utils.mask_transformation.twod(
            data, idim, jdim, rvals, np.array([4, 4.5, 5]), log_var=False, operation="mean"
        )
    # Testing invalid dim range
    rvals = np.array([0, 0.5, 2])
    with pytest.raises(Exception, match="j resampling intervals must be within jdim range"):
        echopype.utils.mask_transformation.twod(
            data, idim, jdim, np.array([1, 1.5, 2]), rvals, log_var=False, operation="mean"
        )
    # Testing unrecognised operation
    with pytest.raises(Exception, match="Operation not recognised"):
        echopype.utils.mask_transformation.twod(
            data,
            idim,
            jdim,
            np.array([1, 1.5, 2]),
            np.array([4, 4.5, 5]),
            log_var=False,
            operation="kind",
        )

    # Sample data
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    idim = np.array([1, 2, 3])
    jdim = np.array([1, 2, 3])

    # Test with data height not equal to idim length
    with pytest.raises(Exception, match="data height and idim length must be the same"):
        echopype.utils.mask_transformation.twod(
            np.array([[1, 2], [3, 4]]), idim, jdim, np.array([1, 3]), np.array([1, 3])
        )

    # Sample data
    idim = np.array([1, 2, 3])
    jdim = np.array([1, 2])

    # Test with data width not equal to jdim length
    with pytest.raises(Exception, match="data width and jdim length must be the same"):
        echopype.utils.mask_transformation.twod(
            np.array([[1, 2, 3], [4, 5, 6], [4, 5, 6]]),
            idim,
            jdim,
            np.array([1, 3]),
            np.array([1, 3]),
        )
