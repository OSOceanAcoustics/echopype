import numpy as np
import pytest
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


def test_oned():
    # Creating a mock data
    data = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ])

    # Example dimension
    dim = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    # Resampling dimension intervals
    rvals = np.array([0.5, 2.5, 4.5])

    # Resampling axis
    axis = 0

    # Expected output
    expected_data = np.array([[3.5, 4.5, 5.5, 6.5, 7.5],
                              [13.5, 14.5, 15.5, 16.5, 17.5]])
    expected_dim = np.array([0.5, 2.5, 4.5])
    expected_percentage = np.array([[100., 100., 100., 100., 100.],
                                    [100., 100., 100., 100., 100.]])

    # Calling the oned function
    result_data, result_dim, result_percentage = echopype.utils.mask_transformation.oned(data, dim, rvals, axis,
                                                                                         log_var=False,
                                                                                         operation='mean')

    # Asserting the output
    np.testing.assert_array_equal(result_data, expected_data)
    np.testing.assert_array_equal(result_dim, expected_dim)
    np.testing.assert_array_equal(result_percentage, expected_percentage)


def test_oned_exceptions():
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    dim = np.array([0, 1])
    rvals = np.array([2])
    axis = 3

    # Testing invalid axis
    with pytest.raises(Exception, match='axis must be 0 or 1'):
        echopype.utils.mask_transformation.oned(data, dim, rvals, axis, log_var=False, operation='mean')

    # Testing invalid resampling intervals
    rvals = np.array([2])
    axis = 0
    with pytest.raises(Exception, match='length of resampling intervals must be >2'):
        echopype.utils.mask_transformation.oned(data, dim, rvals, axis, log_var=False, operation='mean')

    # Testing invalid dim range
    rvals = np.array([0, 0.5, 2])
    with pytest.raises(Exception, match='resampling intervals must be within dim range'):
        echopype.utils.mask_transformation.oned(data, dim, rvals, axis, log_var=False, operation='mean')

    # Testing unrecognised operation
    rvals = np.array([0, 0.5, 1])
    with pytest.raises(Exception, match='Operation not recognised'):
        echopype.utils.mask_transformation.oned(data, dim, rvals, axis, log_var=False, operation='invalid operation')


def test_full_function():
    # Resampled data
    data = np.array([[3.5, 4.5, 5.5, 6.5, 7.5],
                     [13.5, 14.5, 15.5, 16.5, 17.5]])
    # Axis data representing indices
    jax = np.arange(len(data[0]))
    # Dimension values
    dim = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    # Resampling dimension intervals
    rvals = np.array([0.5, 2.5, 4.5])

    # Expected data
    expected_data = np.array([[3.5, 4.5, 5.5, 6.5, 7.5],
                              [3.5, 4.5, 5.5, 6.5, 7.5],
                              [13.5, 14.5, 15.5, 16.5, 17.5],
                              [13.5, 14.5, 15.5, 16.5, 17.5],
                              [np.nan, np.nan, np.nan, np.nan, np.nan]])

    expected_mask = np.array([[False, False, False, False, False],
                              [False, False, False, False, False],
                              [False, False, False, False, False],
                              [False, False, False, False, False],
                              [True, True, True, True, True]])

    # Output data
    output_data, output_mask = echopype.utils.mask_transformation.full(data, rvals, jax, dim, jax)

    np.testing.assert_array_equal(output_data, expected_data)
    np.testing.assert_array_equal(output_mask, expected_mask)


def test_full_function_exceptions():
    data = np.array([[1, 2, 3, 4, 5],
                     [6, 7, 8, 9, 10],
                     [11, 12, 13, 14, 15],
                     [16, 17, 18, 19, 20],
                     [21, 22, 23, 24, 25]])
    dim = [0.5, 1.5, 2.5, 3.5, 4.5]

    with pytest.raises(Exception, match='i resampling interval length must be >2'):
        echopype.utils.mask_transformation.full(data, [0.5], [0.5, 2.5, 4.5], dim, dim)
    with pytest.raises(Exception, match='j resampling interval length must be >2'):
        echopype.utils.mask_transformation.full(data, [0.5, 2.5, 4.5], [0.5], dim, dim)


def test_twod() :
    data = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ])

    #Example dimensions
    idim = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    jdim = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    #Resampling dimension intervals
    irvals = np.array([0.5, 2.5, 4.5])
    jrvals = np.array([0.5, 2.5, 4.5])

    #Expected output
    expected_data = np.array([[4, 6], [14, 16]])
    expected_idim = np.array([0.5, 2.5])
    expected_jdim = np.array([0.5, 2.5])
    expected_percentage = np.array([[100, 100], [100, 100]])

    #call function
    result_data, result_idim, result_jdim, result_percentage = (
        echopype.utils.mask_transformation.twod(data, idim, jdim, irvals, jrvals, log_var=False, operation="mean"))

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
    with pytest.raises(Exception, match='length of resampling intervals must be >2'):
        echopype.utils.mask_transformation.twod(data, idim, jdim, rvals, log_var=False, operation='mean')

    # Testing invalid dim range
    rvals = np.array([0, 0.5, 2])
    with pytest.raises(Exception, match='resampling intervals must be within dim range'):
        echopype.utils.mask_transformation.twod(data, idim, jdim, rvals, log_var=False, operation='mean')

    # Testing unrecognised operation
    rvals = np.array([0, 0.5, 1])
    with pytest.raises(Exception, match='Operation not recognised'):
        echopype.utils.mask_transformation.twod(data, idim, jdim, rvals, log_var=False, operation='kind')