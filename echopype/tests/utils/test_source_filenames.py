from pathlib import Path

import numpy as np

from echopype.utils.prov import _sanitize_source_files


def test_scalars():
    """One or more scalar values"""
    path1 = "/my/path1"
    path2 = Path("/my/path2")

    # Single scalars
    assert _sanitize_source_files(path1) == [path1]
    assert _sanitize_source_files(path2) == [str(path2)]
    # List of scalars
    assert _sanitize_source_files([path1, path2]) == [path1, str(path2)]


def test_mixed():
    """A scalar value and a list or ndarray"""
    path1 = "/my/path1"
    path2 = Path("/my/path2")
    # Mixed-type list
    path_list1 = [path1, path2]
    # String-type ndarray
    path_list2 = np.array([path1, str(path2)])

    # A scalar and a list
    target_path_list = [path1, path1, str(path2)]
    assert _sanitize_source_files([path1, path_list1]) == target_path_list
    # A scalar and an ndarray
    assert _sanitize_source_files([path1, path_list2]) == target_path_list
