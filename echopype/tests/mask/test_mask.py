import pytest

import numpy as np
import xarray as xr
import tempfile
import os.path

import echopype as ep
from echopype.mask.api import _check_source_Sv_freq_diff

from typing import List, Union


def get_mock_freq_diff_data(n: int, add_chan: bool, add_freq_nom: bool):
    """
    Creates an in-memory mock Sv Dataset.

    Parameters
    ----------
    n: int
        The number of rows (``ping_time``) and columns (``range_sample``) of
        each channel matrix
    add_chan: bool
        If True the ``channel`` dimension will be named "channel", else it will
        be named "data_coord"
    add_freq_nom: bool
        If True the ``frequency_nominal`` variable will be added to the Dataset

    Notes
    -----
    The mock Sv Data is created in such a way where ``mat_A - mat_B`` will be
    the identity matrix.
    """

    # matrix representing freqB
    mat_B = np.arange(n ** 2).reshape(n, n) - np.identity(n)

    # matrix representing freqA
    mat_A = np.arange(n ** 2).reshape(n, n)

    # set channel coordinate name (used for testing purposes)
    if not add_chan:
        channel_coord_name = "data_coord"
    else:
        channel_coord_name = "channel"

    # create mock Sv DataArray
    mock_Sv_da = xr.DataArray(data=np.stack([mat_A, mat_B]),
                              coords={channel_coord_name: ['chan1', 'chan2'], "ping_time": np.arange(n),
                                      "range_sample": np.arange(n)})

    # create data variables for the Dataset
    data_vars = {"Sv": mock_Sv_da}

    if add_freq_nom:
        # create mock frequency_nominal and add it to the Dataset variables
        mock_freq_nom = xr.DataArray(data=np.array([1.0, 2.0]), coords={channel_coord_name: ['chan1', 'chan2']})
        data_vars["frequency_nominal"] = mock_freq_nom

    # create mock Dataset with Sv and frequency_nominal
    mock_Sv_ds = xr.Dataset(data_vars=data_vars)

    return mock_Sv_ds


@pytest.mark.parametrize(
    ("n", "add_chan", "add_freq_nom", "source_Sv_is_path", "freqAB", "chanAB"),
    [
        (5, True, True, True, [1.0, 2.0], None),
        (5, True, True, True, None, ['chan1', 'chan2']),
        (5, True, True, False, [1.0, 2.0], None),
        (5, True, True, False, None, ['chan1', 'chan2']),
        pytest.param(5, False, True, False, [1.0, 2.0], None,
                     marks=pytest.mark.xfail(strict=True,
                                             reason="This should fail because the Dataset "
                                                    "will not have the channel coordinate.")),
        pytest.param(5, True, False, False, [1.0, 2.0], None,
                     marks=pytest.mark.xfail(strict=True,
                                             reason="This should fail because the Dataset "
                                                    "will not have the frequency_nominal variable.")),
        pytest.param(5, True, True, False, [1.0, 3.0], None,
                     marks=pytest.mark.xfail(strict=True,
                                             reason="This should fail because not all selected frequencies"
                                                    "are in the frequency_nominal variable.")),
        pytest.param(5, True, True, False, None, ['chan1', 'chan3'],
                     marks=pytest.mark.xfail(strict=True,
                                             reason="This should fail because not all selected channels"
                                                    "are in the channel coordinate.")),
    ],
    ids=["path_input_freqAB_provided", "path_input_chanAB_provided",
         "dataset_input_freqAB_provided", "dataset_input_chanAB_provided", "dataset_no_channel",
         "dataset_no_frequency_nominal", "dataset_missing_freqAB_in_freq_nom",
         "dataset_missing_chanAB_in_channel"]
)
def test_check_source_Sv_freq_diff(n: int, add_chan: bool, add_freq_nom: bool,
                                   source_Sv_is_path: bool,
                                   freqAB: List[float],
                                   chanAB: List[str]):
    """
    Test the inputs ``source_Sv, freqAB, chanAB`` for ``_check_source_Sv_freq_diff``.

    Parameters
    ----------
    n: int
        The number of rows (``ping_time``) and columns (``range_sample``) of
        each channel matrix
    add_chan: bool
        If True the ``channel`` dimension will be named "channel", else it will
        be named "data_coord"
    add_freq_nom: bool
        If True the ``frequency_nominal`` variable will be added to the Dataset
    source_Sv_is_path: bool
        If True a mock in-memory Dataset is created then this Dataset is written
        to a zarr file and ``source_Sv`` is set to the zarr path, else
        ``source_Sv`` will be set to the mock Dataset.
    freqAB: list of float, optional
        The pair of nominal frequencies to be used for frequency-differencing, where
        the first element corresponds to ``freqA`` and the second element corresponds
        to ``freqB``
    chanAB: list of float, optional
        The pair of channels that will be used to select the nominal frequencies to be
        used for frequency-differencing, where the first element corresponds to ``freqA``
        and the second element corresponds to ``freqB``
    """

    ds = get_mock_freq_diff_data(n, add_chan, add_freq_nom)

    if source_Sv_is_path:

        # create temporary directory for zarr store
        temp_zarr_dir = tempfile.TemporaryDirectory()
        zarr_path = os.path.join(temp_zarr_dir.name, "test.zarr")

        # save source_Sv to zarr and set the variable to zarr_path
        ds.to_zarr(zarr_path)
        source_Sv_input = zarr_path

    else:
        source_Sv_input = ds

    source_Sv_output = _check_source_Sv_freq_diff(source_Sv_input, storage_options={},
                                                  freqAB=freqAB, chanAB=chanAB)

    # ensure that ds is equal to the source_Sv produced
    assert ds.identical(source_Sv_output)

    if source_Sv_is_path:
        # remove temporary directory
        temp_zarr_dir.cleanup()


@pytest.mark.parametrize(
    ("n", "source_Sv_is_path", "freqAB", "chanAB", "diff", "operator", "mask_truth"),
    [
        (5, False, [1.0, 2.0], None, 1.0, "==", np.identity(5)),
        (5, False, None, ['chan1', 'chan2'], 1.0, "==", np.identity(5)),
        (5, False, [2.0, 1.0], None, 1.0, "==", np.zeros((5, 5))),
        (5, False, None, ['chan2', 'chan1'], 1.0, "==", np.zeros((5, 5))),
        (5, False, [1.0, 2.0], None, 1.0, ">=", np.identity(5)),
        (5, False, None, ['chan1', 'chan2'], 1.0, ">=", np.identity(5)),
        (5, False, [1.0, 2.0], None, 1.0, ">", np.zeros((5, 5))),
        (5, False, None, ['chan1', 'chan2'], 1.0, ">", np.zeros((5, 5))),
        (5, False, [1.0, 2.0], None, 1.0, "<=", np.ones((5, 5))),
        (5, False, None, ['chan1', 'chan2'], 1.0, "<=", np.ones((5, 5))),
        (5, False, [1.0, 2.0], None, 1.0, "<", np.ones((5, 5)) - np.identity(5)),
        (5, False, None, ['chan1', 'chan2'], 1.0, "<", np.ones((5, 5)) - np.identity(5)),
    ],
    ids=["freqAB_sel_op_equals", "chanAB_sel_op_equals", "reverse_freqAB_sel_op_equals",
         "reverse_chanAB_sel_op_equals", "freqAB_sel_op_ge", "chanAB_sel_op_ge",
         "freqAB_sel_op_greater", "chanAB_sel_op_greater", "freqAB_sel_op_le",
         "chanAB_sel_op_le", "freqAB_sel_op_less", "chanAB_sel_op_less"]
)
def test_frequency_differencing(n: int, source_Sv_is_path: bool,
                                freqAB: List[float], chanAB: List[str],
                                diff: Union[float, int], operator: str,
                                mask_truth: np.ndarray):
    """
    Tests that the output values of ``frequency_differencing`` are what we
    expect, the output is a DataArray, and that the name of the DataArray is correct.

    Parameters
    ----------
    n: int
        The number of rows (``ping_time``) and columns (``range_sample``) of
        each channel matrix
    freqAB: list of float, optional
        The pair of nominal frequencies to be used for frequency-differencing, where
        the first element corresponds to ``freqA`` and the second element corresponds
        to ``freqB``
    chanAB: list of float, optional
        The pair of channels that will be used to select the nominal frequencies to be
        used for frequency-differencing, where the first element corresponds to ``freqA``
        and the second element corresponds to ``freqB``
    diff: float or int
        The threshold of Sv difference between frequencies
    operator: {">", "<", "<=", ">=", "=="}
        The operator for the frequency-differencing
    mask_truth: np.ndarray
        The truth value for the output mask, provided the given inputs
    """

    # obtain mock Sv Dataset
    mock_Sv_ds = get_mock_freq_diff_data(n, add_chan=True, add_freq_nom=True)

    # obtain the frequency-difference mask for mock_Sv_ds
    out = ep.mask.frequency_differencing(source_Sv=mock_Sv_ds, storage_options={}, freqAB=freqAB,
                                         chanAB=chanAB,
                                         operator=operator, diff=diff)

    # ensure that the output values are correct
    assert np.all(out == mask_truth)

    # ensure that the output is a DataArray
    assert isinstance(out, xr.DataArray)

    # test that the output DataArray is correctly names
    assert out.name == "mask"





