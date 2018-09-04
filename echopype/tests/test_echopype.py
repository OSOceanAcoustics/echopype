import os.path as op
import numpy as np
import numpy.testing as npt
# import echopype as ep

# data_path = op.join(ep.__path__[0], 'data')


def test_try():
    """
    Trying out test functions
    """
    my_data = np.array([[0.1, 2], [0.1, 1], [0.2, 2], [0.2, 2], [0.3, 1], [0.3, 1]])
    npt.assert_equal(my_data[0, :], np.array([0.1, 2]))


def test_try2():
    """
    Trying out test functions
    """
    my_data = np.array([[0.1, 2], [0.1, 1], [0.2, 2], [0.2, 2], [0.3, 1], [0.3, 1]])
    npt.assert_equal(my_data[2, :], np.array([0.2, 2]))