import pytest

import numpy as np
import xarray as xr
import dask.array

import echopype as ep
import echopype.mask
from echopype.mask.api import _check_source_Sv_freq_diff

from typing import List, Union


def get_mock_freq_diff_data(n: int, n_chan_freq: int, add_chan: bool,
                            add_freq_nom: bool) -> xr.Dataset:
    """
    Creates an in-memory mock Sv Dataset.

    Parameters
    ----------
    n: int
        The number of rows (``ping_time``) and columns (``range_sample``) of
        each channel matrix
    n_chan_freq: int
        Determines the size of the ``channel`` coordinate and ``frequency_nominal``
        variable. To create mock data with known outcomes for ``frequency_differencing``,
        this value must be greater than or equal to 3.
    add_chan: bool
        If True the ``channel`` dimension will be named "channel", else it will
        be named "data_coord"
    add_freq_nom: bool
        If True the ``frequency_nominal`` variable will be added to the Dataset

    Returns
    -------
    mock_Sv_ds: xr.Dataset
        A mock Sv dataset to be used for ``frequency_differencing`` tests. The Sv
        data values for the channel coordinate ``chan1`` will be equal to ``mat_A``,
        ``chan3`` will be equal to ``mat_B``, and all other channel coordinates
        will retain the value of ``np.identity(n)``.

    Notes
    -----
    The mock Sv Data is created in such a way where ``mat_A - mat_B`` will be
    the identity matrix.
    """

    if n_chan_freq < 3:
        raise RuntimeError("The input n_chan_freq must be greater than or equal to 3!")

    # matrix representing freqB
    mat_B = np.arange(n ** 2).reshape(n, n) - np.identity(n)

    # matrix representing freqA
    mat_A = np.arange(n ** 2).reshape(n, n)

    # construct channel values
    chan_vals = ['chan' + str(i) for i in range(1, n_chan_freq+1)]

    # construct mock Sv data
    mock_Sv_data = [mat_A, np.identity(n), mat_B] + [np.identity(n) for i in range(3, n_chan_freq)]

    # set channel coordinate name (used for testing purposes)
    if not add_chan:
        channel_coord_name = "data_coord"
    else:
        channel_coord_name = "channel"

    # create mock Sv DataArray
    mock_Sv_da = xr.DataArray(data=np.stack(mock_Sv_data),
                              coords={channel_coord_name: chan_vals, "ping_time": np.arange(n),
                                      "range_sample": np.arange(n)})

    # create data variables for the Dataset
    data_vars = {"Sv": mock_Sv_da}

    if add_freq_nom:
        # construct frequency_values
        freq_vals = [float(i) for i in range(1, n_chan_freq + 1)]

        # create mock frequency_nominal and add it to the Dataset variables
        mock_freq_nom = xr.DataArray(data=freq_vals, coords={channel_coord_name: chan_vals})
        data_vars["frequency_nominal"] = mock_freq_nom

    # create mock Dataset with Sv and frequency_nominal
    mock_Sv_ds = xr.Dataset(data_vars=data_vars)

    return mock_Sv_ds


def get_mock_source_ds_apply_mask(n: int, n_chan: int, is_delayed: bool) -> xr.Dataset:
    """
    Constructs a mock ``source_ds`` Dataset input for the
    ``apply_mask`` function.

    Parameters
    ----------
    n: int
        The number of rows (``x``) and columns (``y``) of
        each channel matrix
    n_chan: int
        Determines the size of the ``channel`` coordinate
    is_delayed: bool
        If True, the returned Dataset variables ``var1`` and ``var2`` will be
        a Dask arrays, else they will be in-memory arrays

    Returns
    -------
    xr.Dataset
        A Dataset with coordinates ``channel, x, y`` and
        variables ``var1, var2`` (with the created coordinates). The
        variables will contain square matrices of ones for each ``channel``.
    """

    # construct channel values
    chan_vals = ['chan' + str(i) for i in range(1, n_chan + 1)]

    # construct mock variable data for each channel
    if is_delayed:
        mock_var_data = [dask.array.ones((n, n)) for i in range(n_chan)]
    else:
        mock_var_data = [np.ones((n, n)) for i in range(n_chan)]

    # create mock var1 and var2 DataArrays
    mock_var1_da = xr.DataArray(data=np.stack(mock_var_data),
                                coords={"channel": ("channel", chan_vals, {"long_name": "channel name"}),
                                        "x": np.arange(n), "y": np.arange(n)},
                                attrs={"long_name": "variable 1"})
    mock_var2_da = xr.DataArray(data=np.stack(mock_var_data),
                                coords={"channel": ("channel", chan_vals, {"long_name": "channel name"}),
                                        "x": np.arange(n),
                                        "y": np.arange(n)},
                                attrs={"long_name": "variable 2"})

    # create mock Dataset
    mock_ds = xr.Dataset(data_vars={"var1": mock_var1_da, "var2": mock_var2_da})

    return mock_ds


@pytest.mark.parametrize(
    ("n", "n_chan_freq", "add_chan", "add_freq_nom", "freqAB", "chanAB"),
    [
        (5, 3, True, True, [1.0, 3.0], None),
        (5, 3, True, True, None, ['chan1', 'chan3']),
        pytest.param(5, 3, False, True, [1.0, 3.0], None,
                     marks=pytest.mark.xfail(strict=True,
                                             reason="This should fail because the Dataset "
                                                    "will not have the channel coordinate.")),
        pytest.param(5, 3, True, False, [1.0, 3.0], None,
                     marks=pytest.mark.xfail(strict=True,
                                             reason="This should fail because the Dataset "
                                                    "will not have the frequency_nominal variable.")),
        pytest.param(5, 3, True, True, [1.0, 4.0], None,
                     marks=pytest.mark.xfail(strict=True,
                                             reason="This should fail because not all selected frequencies"
                                                    "are in the frequency_nominal variable.")),
        pytest.param(5, 3, True, True, None, ['chan1', 'chan4'],
                     marks=pytest.mark.xfail(strict=True,
                                             reason="This should fail because not all selected channels"
                                                    "are in the channel coordinate.")),
    ],
    ids=["dataset_input_freqAB_provided", "dataset_input_chanAB_provided", "dataset_no_channel",
         "dataset_no_frequency_nominal", "dataset_missing_freqAB_in_freq_nom",
         "dataset_missing_chanAB_in_channel"]
)
def test_check_source_Sv_freq_diff(n: int, n_chan_freq: int, add_chan: bool, add_freq_nom: bool,
                                   freqAB: List[float],
                                   chanAB: List[str]):
    """
    Test the inputs ``source_Sv, freqAB, chanAB`` for ``_check_source_Sv_freq_diff``.

    Parameters
    ----------
    n: int
        The number of rows (``ping_time``) and columns (``range_sample``) of
        each channel matrix
    n_chan_freq: int
        Determines the size of the ``channel`` coordinate and ``frequency_nominal``
        variable. To create mock data with known outcomes for ``frequency_differencing``,
        this value must be greater than or equal to 3.
    add_chan: bool
        If True the ``channel`` dimension will be named "channel", else it will
        be named "data_coord"
    add_freq_nom: bool
        If True the ``frequency_nominal`` variable will be added to the Dataset
    freqAB: list of float, optional
        The pair of nominal frequencies to be used for frequency-differencing, where
        the first element corresponds to ``freqA`` and the second element corresponds
        to ``freqB``
    chanAB: list of float, optional
        The pair of channels that will be used to select the nominal frequencies to be
        used for frequency-differencing, where the first element corresponds to ``freqA``
        and the second element corresponds to ``freqB``
    """

    source_Sv = get_mock_freq_diff_data(n, n_chan_freq, add_chan, add_freq_nom)

    _check_source_Sv_freq_diff(source_Sv, freqAB=freqAB, chanAB=chanAB)


@pytest.mark.parametrize(
    ("n", "n_chan_freq", "freqAB", "chanAB", "diff", "operator", "mask_truth"),
    [
        (5, 4, [1.0, 3.0], None, 1.0, "==", np.identity(5)),
        (5, 4, None, ['chan1', 'chan3'], 1.0, "==", np.identity(5)),
        (5, 4, [3.0, 1.0], None, 1.0, "==", np.zeros((5, 5))),
        (5, 4, None, ['chan3', 'chan1'], 1.0, "==", np.zeros((5, 5))),
        (5, 4, [1.0, 3.0], None, 1.0, ">=", np.identity(5)),
        (5, 4, None, ['chan1', 'chan3'], 1.0, ">=", np.identity(5)),
        (5, 4, [1.0, 3.0], None, 1.0, ">", np.zeros((5, 5))),
        (5, 4, None, ['chan1', 'chan3'], 1.0, ">", np.zeros((5, 5))),
        (5, 4, [1.0, 3.0], None, 1.0, "<=", np.ones((5, 5))),
        (5, 4, None, ['chan1', 'chan3'], 1.0, "<=", np.ones((5, 5))),
        (5, 4, [1.0, 3.0], None, 1.0, "<", np.ones((5, 5)) - np.identity(5)),
        (5, 4, None, ['chan1', 'chan3'], 1.0, "<", np.ones((5, 5)) - np.identity(5)),
    ],
    ids=["freqAB_sel_op_equals", "chanAB_sel_op_equals", "reverse_freqAB_sel_op_equals",
         "reverse_chanAB_sel_op_equals", "freqAB_sel_op_ge", "chanAB_sel_op_ge",
         "freqAB_sel_op_greater", "chanAB_sel_op_greater", "freqAB_sel_op_le",
         "chanAB_sel_op_le", "freqAB_sel_op_less", "chanAB_sel_op_less"]
)
def test_frequency_differencing(n: int, n_chan_freq: int,
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
    n_chan_freq: int
        Determines the size of the ``channel`` coordinate and ``frequency_nominal``
        variable. To create mock data with known outcomes for ``frequency_differencing``,
        this value must be greater than or equal to 3.
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
    mock_Sv_ds = get_mock_freq_diff_data(n, n_chan_freq, add_chan=True, add_freq_nom=True)

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


@pytest.mark.parametrize(
    ("n", "n_chan", "var_name", "mask", "is_delayed", "var_masked_truth"),
    [
        (2, 1, "var1", np.identity(2), False, np.array([[1, 2.0], [2.0, 1]])),
        (2, 1, "var1", [np.identity(2), np.array([[0, 1], [0, 1]])], False, np.array([[2.0, 2.0], [2.0, 1]])),
        (2, 1, "var1", np.identity(2), True, np.array([[1, 2.0], [2.0, 1]])),
    ],
    ids=["single_mask", "list_mask_all_np", "single_mask_ds_delayed"]
)
def test_apply_mask(n: int, n_chan: int, var_name: str,
                    mask: Union[np.ndarray, List[np.ndarray]],
                    var_masked_truth: np.ndarray, is_delayed: bool):
    """


    Parameters
    ----------
    n: int

    n_chan: int

    var_name: str

    mask: np.ndarray or list of np.ndarray

    var_masked_truth: np.ndarray

    is_delayed: bool
        If True, makes all variables in constructed mock Dataset Dask arrays,
        else they will be in-memory arrays

    """

    # TODO: test different fill_value
    fill_value = 2.0  # having this be a number, rather than nan makes for easier testing

    mock_ds = get_mock_source_ds_apply_mask(n, n_chan, is_delayed)

    # TODO: test path input for mask

    # TODO: create test for list of masks
    if isinstance(mask, list):
        mask_list = []
        for elem in mask:
            mask_list.append(xr.DataArray(data=np.stack([elem for i in range(n_chan)]),
                                          coords=mock_ds.coords))
        mask = mask_list

    else:
        mask = xr.DataArray(data=np.stack([mask for i in range(n_chan)]),
                            coords=mock_ds.coords)

    var_masked_truth = xr.DataArray(data=np.stack([var_masked_truth for i in range(n_chan)]),
                                    coords=mock_ds[var_name].coords, attrs=mock_ds[var_name].attrs)
    var_masked_truth.name = mock_ds[var_name].name

    masked_ds = echopype.mask.apply_mask(source_ds=mock_ds, var_name=var_name, mask=mask,
                                         fill_value=fill_value, storage_options_ds={},
                                         storage_options_mask={})

    # TODO: check that masked_ds has the same structure as mock_ds

    # check that masked_ds[var_name] == var_masked_truth
    assert masked_ds[var_name].identical(var_masked_truth)

    # check that the output Dataset has lazy elements, if the input was lazy
    if is_delayed:
        assert isinstance(masked_ds[var_name].data, dask.array.Array)
