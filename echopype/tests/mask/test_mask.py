import pathlib

import pytest

import numpy as np
import xarray as xr
import dask.array
import tempfile
import os

import echopype as ep
import echopype.mask
from echopype.mask.api import _validate_and_collect_mask_input, _check_var_name_fill_value
from echopype.mask.freq_diff import (
    _parse_freq_diff_eq,
    _check_freq_diff_source_Sv,
)

from typing import List, Union, Optional


def get_mock_freq_diff_data(
    n: int,
    n_chan_freq: int,
    add_chan: bool,
    add_freq_nom: bool,
    dask_array: bool = False,
    chunks: dict = None,
) -> xr.Dataset:
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

    if dask_array:
        if chunks is None:
            raise RuntimeError("The input chunks must be provided if dask_array is True.")

    # matrix representing freqB
    mat_B = np.arange(n**2).reshape(n, n) - np.identity(n)

    # matrix representing freqA
    mat_A = np.arange(n**2).reshape(n, n)

    # construct channel values
    chan_vals = ["chan" + str(i) for i in range(1, n_chan_freq + 1)]

    # construct mock Sv data
    mock_Sv_data = [mat_A, np.identity(n), mat_B] + [np.identity(n) for i in range(3, n_chan_freq)]

    # set channel coordinate name (used for testing purposes)
    if not add_chan:
        channel_coord_name = "data_coord"
    else:
        channel_coord_name = "channel"

    # create mock Sv DataArray
    mock_Sv_da = xr.DataArray(
        data=np.stack(mock_Sv_data),
        coords={
            channel_coord_name: chan_vals,
            "ping_time": np.arange(n),
            "range_sample": np.arange(n),
        },
    )

    # create data variables for the Dataset
    if dask_array:
        data_vars = {"Sv": mock_Sv_da.chunk(chunks=chunks)}
    else:
        data_vars = {"Sv": mock_Sv_da}

    if add_freq_nom:
        # construct frequency_values
        freqs = [1, 1e3, 2, 2e3]
        freq_vals = [float(i) for i in freqs]

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
        The ``ping_time`` and ``range_sample`` dimensions of each channel matrix
    n_chan: int
        Size of the ``channel`` coordinate
    is_delayed: bool
        If True, the returned Dataset variables ``var1`` and ``var2`` will be
        a Dask arrays, else they will be in-memory arrays

    Returns
    -------
    xr.Dataset
        A Dataset containing data variables ``var1, var2`` with coordinates
        ``('channel', 'ping_time', 'range_sample')``.
        The variables are square matrices of ones for each ``channel``.
    """

    # construct channel values
    chan_vals = ["chan" + str(i) for i in range(1, n_chan + 1)]

    # construct mock variable data for each channel
    if is_delayed:
        mock_var_data = [dask.array.ones((n, n)) for i in range(n_chan)]
    else:
        mock_var_data = [np.ones((n, n)) for i in range(n_chan)]

    # create mock var1 and var2 DataArrays
    mock_var1_da = xr.DataArray(
        data=np.stack(mock_var_data),
        coords={
            "channel": ("channel", chan_vals, {"long_name": "channel name"}),
            "ping_time": np.arange(n),
            "range_sample": np.arange(n),
        },
        attrs={"long_name": "variable 1"},
    )
    mock_var2_da = xr.DataArray(
        data=np.stack(mock_var_data),
        coords={
            "channel": ("channel", chan_vals, {"long_name": "channel name"}),
            "ping_time": np.arange(n),
            "range_sample": np.arange(n),
        },
        attrs={"long_name": "variable 2"},
    )

    # create mock Dataset
    mock_ds = xr.Dataset(data_vars={"var1": mock_var1_da, "var2": mock_var2_da})

    return mock_ds


def create_input_mask(
    mask: Union[np.ndarray, List[np.ndarray]],
    mask_file: Optional[Union[str, List[str]]],
    mask_coords: Union[xr.core.coordinates.DataArrayCoordinates, dict],
):
    """
    A helper function that correctly constructs the mask input, so it can be
    used for ``apply_mask`` related tests.

    Parameters
    ----------
    mask: np.ndarray or list of np.ndarray
        The mask(s) that should be applied to ``var_name``
    mask_file: str or list of str, optional
        If provided, the ``mask`` input will be written to a temporary directory
        with file name ``mask_file``. This will then be used in ``apply_mask``.
    mask_coords: xr.core.coordinates.DataArrayCoordinates or dict
        The DataArray coordinates that should be used for each mask DataArray created
    """

    # initialize temp_dir
    temp_dir = None

    # make input numpy array masks into DataArrays
    if isinstance(mask, list):
        # initialize final mask
        mask_out = []

        # create temporary directory if mask_file is provided
        if any([isinstance(elem, str) for elem in mask_file]):
            # create temporary directory for mask_file
            temp_dir = tempfile.TemporaryDirectory()

        for mask_ind in range(len(mask)):
            # form DataArray from given mask data
            mask_da = xr.DataArray(
                data=[mask[mask_ind]], coords=mask_coords, name="mask_" + str(mask_ind)
            )

            if mask_file[mask_ind] is None:
                # set mask value to the DataArray given
                mask_out.append(mask_da)
            else:
                # write DataArray to temporary directory
                zarr_path = os.path.join(temp_dir.name, mask_file[mask_ind])
                mask_da.to_dataset().to_zarr(zarr_path)

                if isinstance(mask_file[mask_ind], pathlib.Path):
                    # make zarr_path into a Path object
                    zarr_path = pathlib.Path(zarr_path)

                # set mask value to created path
                mask_out.append(zarr_path)

    elif isinstance(mask, np.ndarray):
        # form DataArray from given mask data
        mask_da = xr.DataArray(data=[mask], coords=mask_coords, name="mask_0")

        if mask_file is None:
            # set mask to the DataArray formed
            mask_out = mask_da
        else:
            # create temporary directory for mask_file
            temp_dir = tempfile.TemporaryDirectory()

            # write DataArray to temporary directory
            zarr_path = os.path.join(temp_dir.name, mask_file)
            mask_da.to_dataset().to_zarr(zarr_path)

            if isinstance(mask_file, pathlib.Path):
                # make zarr_path into a Path object
                zarr_path = pathlib.Path(zarr_path)

            # set mask index to path
            mask_out = zarr_path

    return mask_out, temp_dir


@pytest.mark.parametrize(
    ("n", "n_chan_freq", "add_chan", "add_freq_nom", "freqAB", "chanAB"),
    [
        (5, 4, True, True, [1000.0, 2.0], None),
        (5, 4, True, True, None, ["chan1", "chan3"]),
        pytest.param(
            5,
            4,
            False,
            True,
            [1.0, 2000000.0],
            None,
            marks=pytest.mark.xfail(
                strict=True,
                reason="This should fail because the Dataset "
                "will not have the channel coordinate.",
            ),
        ),
        pytest.param(
            5,
            4,
            True,
            False,
            [1.0, 2.0],
            None,
            marks=pytest.mark.xfail(
                strict=True,
                reason="This should fail because the Dataset "
                "will not have the frequency_nominal variable.",
            ),
        ),
        pytest.param(
            5,
            4,
            True,
            True,
            [1.0, 4.0],
            None,
            marks=pytest.mark.xfail(
                strict=True,
                reason="This should fail because not all selected frequencies"
                "are in the frequency_nominal variable.",
            ),
        ),
        pytest.param(
            5,
            4,
            True,
            True,
            None,
            ["chan1", "chan9"],
            marks=pytest.mark.xfail(
                strict=True,
                reason="This should fail because not all selected channels"
                "are in the channel coordinate.",
            ),
        ),
    ],
    ids=[
        "dataset_input_freqAB_provided",
        "dataset_input_chanAB_provided",
        "dataset_no_channel",
        "dataset_no_frequency_nominal",
        "dataset_missing_freqAB_in_freq_nom",
        "dataset_missing_chanAB_in_channel",
    ],
)
def test_check_freq_diff_source_Sv(
    n: int,
    n_chan_freq: int,
    add_chan: bool,
    add_freq_nom: bool,
    freqAB: List[float],
    chanAB: List[str],
):
    """
    Test the inputs ``source_Sv, freqAB, chanAB`` for ``_check_freq_diff_source_Sv``.

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

    _check_freq_diff_source_Sv(source_Sv, freqAB=freqAB, chanAB=chanAB)


@pytest.mark.parametrize(
    ("freqABEq", "chanABEq"),
    [
        ("1.0Hz-2.0Hz==1.0dB", None),
        (None, '"chan1"-"chan3"==1.0dB'),
        ("1.0 kHz - 2.0 MHz>=1.0 dB", None),
        (None, '"chan2-12 89" - "chan4 89-12" >= 1.0 dB'),
        pytest.param(
            "1.0kHz-2.0 kHz===1.0dB",
            None,
            marks=pytest.mark.xfail(
                strict=True, reason="This should fail because " "the operator is incorrect."
            ),
        ),
        pytest.param(
            None,
            '"chan1"-"chan3"===1.0 dB',
            marks=pytest.mark.xfail(
                strict=True, reason="This should fail because " "the operator is incorrect."
            ),
        ),
        pytest.param(
            "1.0 MHz-1.0MHz==1.0dB",
            None,
            marks=pytest.mark.xfail(
                strict=True, reason="This should fail because the " "frequencies are the same."
            ),
        ),
        pytest.param(
            None,
            '"chan1"-"chan1"==1.0 dB',
            marks=pytest.mark.xfail(
                strict=True, reason="This should fail because the " "channels are the same."
            ),
        ),
        pytest.param(
            "1.0 Hz-2.0==1.0dB",
            None,
            marks=pytest.mark.xfail(
                strict=True,
                reason="This should fail because unit of one of " "the frequency is missing.",
            ),
        ),
        pytest.param(
            None,
            '"chan1"-"chan3"==1.0',
            marks=pytest.mark.xfail(
                strict=True, reason="This should fail because unit of the " "difference is missing."
            ),
        ),
    ],
    ids=[
        "input_freqABEq_provided",
        "input_chanABEq_provided",
        "input_freqABEq_different_units",
        "input_chanABEq_provided",
        "input_freqABEq_wrong_operator",
        "input_chanABEq_wrong_operator",
        "input_freqABEq_duplicate_frequencies",
        "input_chanABEq_duplicate_channels",
        "input_freqABEq_missing_unit",
        "input_chanABEq_missing_unit",
    ],
)
def test_parse_freq_diff_eq(freqABEq: str, chanABEq: str):
    """
    Tests the inputs ``freqABEq, chanABEq`` for ``_parse_freq_diff_eq``.

    Parameters
    ----------
    freqABEq: string, optional
        The frequency differencing criteria.
    chanABEq: string, optional
        The frequency differencing criteria in terms of channel names where channel names
        in the criteria are enclosed in double quotes.
    """
    freq_vals = [1.0, 2.0, 1e3, 2e3, 1e6, 2e6]
    chan_vals = ["chan1", "chan3", "chan2-12 89", "chan4 89-12"]
    operator_vals = [">=", "=="]
    diff_val = 1.0
    freqAB, chanAB, operator, diff = _parse_freq_diff_eq(freqABEq=freqABEq, chanABEq=chanABEq)
    if freqAB is not None:
        for freq in freqAB:
            assert freq in freq_vals
    if chanAB is not None:
        for chan in chanAB:
            assert chan in chan_vals
    assert operator in operator_vals
    assert diff == diff_val


@pytest.mark.parametrize(
    (
        "n",
        "n_chan_freq",
        "freqABEq",
        "chanABEq",
        "mask_truth",
        "dask_array",
        "chunks",
    ),
    [
        (5, 4, "1.0Hz-2.0Hz== 1.0dB", None, np.identity(5), None, None),
        (5, 4, None, '"chan1"-"chan3" == 1.0 dB', np.identity(5), None, None),
        (5, 4, "2.0 Hz - 1.0 Hz==1.0 dB", None, np.zeros((5, 5)), None, None),
        (5, 4, None, '"chan3" - "chan1"==1.0 dB', np.zeros((5, 5)), None, None),
        (5, 4, "1.0 Hz-2.0Hz>=1.0dB", None, np.identity(5), None, None),
        (5, 4, None, '"chan1" - "chan3" >= 1.0 dB', np.identity(5), None, None),
        (5, 4, "1.0 kHz - 2.0 kHz > 1.0dB", None, np.zeros((5, 5)), None, None),
        (5, 4, None, '"chan1"-"chan3">1.0 dB', np.zeros((5, 5)), None, None),
        (5, 4, "1.0kHz-2.0 kHz<=1.0dB", None, np.ones((5, 5)), None, None),
        (5, 4, None, '"chan1" - "chan3" <= 1.0 dB', np.ones((5, 5)), None, None),
        (5, 4, "1.0 Hz-2.0Hz<1.0dB", None, np.ones((5, 5)) - np.identity(5), None, None),
        (5, 4, None, '"chan1"-"chan3"< 1.0 dB', np.ones((5, 5)) - np.identity(5), None, None),
        # Dask Arrays
        (
            500,
            4,
            "1.0Hz-2.0Hz== 1.0dB",
            None,
            np.identity(500),
            True,
            {
                "ping_time": 10,
                "range_sample": 10,
            },
        ),
        (
            500,
            4,
            None,
            '"chan1"-"chan3" == 1.0 dB',
            np.identity(500),
            True,
            {
                "ping_time": 25,
                "range_sample": 25,
            },
        ),
        (
            1000,
            4,
            "2.0 Hz - 1.0 Hz==1.0 dB",
            None,
            np.zeros((1000, 1000)),
            True,
            {
                "ping_time": 200,
                "range_sample": 200,
            },
        ),
        (
            750,
            4,
            None,
            '"chan3" - "chan1"==1.0 dB',
            np.zeros((750, 750)),
            True,
            {
                "ping_time": 60,
                "range_sample": 40,
            },
        ),
        (
            5,
            4,
            "1.0 Hz-2.0Hz>=1.0dB",
            None,
            np.identity(5),
            True,
            {
                "ping_time": -1,
                "range_sample": -1,
            },
        ),
        (
            600,
            4,
            None,
            '"chan1" - "chan3" >= 1.0 dB',
            np.identity(600),
            True,
            {
                "ping_time": 120,
                "range_sample": 60,
            },
        ),
        (
            700,
            4,
            "1.0 kHz - 2.0 kHz > 1.0dB",
            None,
            np.zeros((700, 700)),
            True,
            {
                "ping_time": 100,
                "range_sample": 100,
            },
        ),
        (
            800,
            4,
            None,
            '"chan1"-"chan3">1.0 dB',
            np.zeros((800, 800)),
            True,
            {
                "ping_time": 120,
                "range_sample": 150,
            },
        ),
        (
            5,
            4,
            "1.0kHz-2.0 kHz<=1.0dB",
            None,
            np.ones((5, 5)),
            True,
            {
                "ping_time": -1,
                "range_sample": -1,
            },
        ),
        (
            500,
            4,
            None,
            '"chan1" - "chan3" <= 1.0 dB',
            np.ones((500, 500)),
            True,
            {
                "ping_time": 50,
                "range_sample": 50,
            },
        ),
        (
            500,
            4,
            "1.0 Hz-2.0Hz<1.0dB",
            None,
            np.ones((500, 500)) - np.identity(500),
            True,
            {
                "ping_time": 100,
                "range_sample": 100,
            },
        ),
        (
            10000,
            4,
            None,
            '"chan1"-"chan3"< 1.0 dB',
            np.ones((10000, 10000)) - np.identity(10000),
            True,
            {
                "ping_time": 1000,
                "range_sample": 1000,
            },
        ),
    ],
    ids=[
        "freqAB_sel_op_equals",
        "chanAB_sel_op_equals",
        "reverse_freqAB_sel_op_equals",
        "reverse_chanAB_sel_op_equals",
        "freqAB_sel_op_ge",
        "chanAB_sel_op_ge",
        "freqAB_sel_op_greater",
        "chanAB_sel_op_greater",
        "freqAB_sel_op_le",
        "chanAB_sel_op_le",
        "freqAB_sel_op_less",
        "chanAB_sel_op_less",
        # Dask Arrays
        "freqAB_sel_op_equals_dask",
        "chanAB_sel_op_equals_dask",
        "reverse_freqAB_sel_op_equals_dask",
        "reverse_chanAB_sel_op_equals_dask",
        "freqAB_sel_op_ge_dask",
        "chanAB_sel_op_ge_dask",
        "freqAB_sel_op_greater_dask",
        "chanAB_sel_op_greater_dask",
        "freqAB_sel_op_le_dask",
        "chanAB_sel_op_le_dask",
        "freqAB_sel_op_less_dask",
        "chanAB_sel_op_less_dask",
    ],
)
def test_frequency_differencing(
    n: int,
    n_chan_freq: int,
    freqABEq: str,
    chanABEq: str,
    mask_truth: np.ndarray,
    dask_array: bool,
    chunks: dict,
):
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
    freqABEq: string, optional
        The frequency differencing criteria.
    chanABEq: string, optional
        The frequency differencing criteria in terms of channel names where channel names
        in the criteria are enclosed in double quotes.
    mask_truth: np.ndarray
        The truth value for the output mask, provided the given inputs
    dask_array: bool
        The boolean value to decide if the mock Sv Dataset is dask or not
    chunks: dict
        The chunk sizes along ping_time and range_sample dimension
    """

    # obtain mock Sv Dataset
    mock_Sv_ds = get_mock_freq_diff_data(
        n,
        n_chan_freq,
        add_chan=True,
        add_freq_nom=True,
        dask_array=dask_array,
        chunks=chunks,
    )

    if dask_array:
        assert mock_Sv_ds["Sv"].chunks != None

    # obtain the frequency-difference mask for mock_Sv_ds
    out = ep.mask.frequency_differencing(
        source_Sv=mock_Sv_ds, storage_options={}, freqABEq=freqABEq, chanABEq=chanABEq
    )

    if dask_array:
        # ensure that the output values are correct
        assert np.all(out.data.compute() == mask_truth)
    else:
        # ensure that the output values are correct
        assert np.all(out == mask_truth)

    # ensure that the output is a DataArray
    assert isinstance(out, xr.DataArray)

    # test that the output DataArray is correctly names
    assert out.name == "mask"


@pytest.mark.parametrize(
    ("n", "n_chan", "mask_np", "mask_file", "storage_options_mask"),
    [
        (5, 1, np.identity(5), None, {}),
        (5, 1, [np.identity(5), np.identity(5)], [None, None], {}),
        (5, 1, [np.identity(5), np.identity(5)], [None, None], [{}, {}]),
        (5, 1, np.identity(5), "path/to/mask.zarr", {}),
        (5, 1, [np.identity(5), np.identity(5)], ["path/to/mask0.zarr", "path/to/mask1.zarr"], {}),
        (5, 1, np.identity(5), pathlib.Path("path/to/mask.zarr"), {}),
        (
            5,
            1,
            [np.identity(5), np.identity(5), np.identity(5)],
            [None, "path/to/mask0.zarr", pathlib.Path("path/to/mask1.zarr")],
            {},
        ),
    ],
    ids=[
        "mask_da",
        "mask_list_da_single_storage",
        "mask_list_da_list_storage",
        "mask_str_path",
        "mask_list_str_path",
        "mask_pathlib",
        "mask_mixed_da_str_pathlib",
    ],
)
def test_validate_and_collect_mask_input(
    n: int,
    n_chan: int,
    mask_np: Union[np.ndarray, List[np.ndarray]],
    mask_file: Optional[Union[str, pathlib.Path, List[Union[str, pathlib.Path]]]],
    storage_options_mask: Union[dict, List[dict]],
):
    """
    Tests the allowable types for the mask input and corresponding storage options.

    Parameters
    ----------
    n: int
        The number of rows (``x``) and columns (``y``) of
        each channel matrix
    n_chan: int
        Determines the size of the ``channel`` coordinate
    mask_np: np.ndarray or list of np.ndarray
        The mask(s) that should be applied to ``var_name``
    mask_file: str or list of str, optional
        If provided, the ``mask`` input will be written to a temporary directory
        with file name ``mask_file``. This will then be used in ``apply_mask``.
    storage_options_mask: dict or list of dict, default={}
        Any additional parameters for the storage backend, corresponding to the
        path provided for ``mask``

    Notes
    -----
    The input for ``storage_options_mask`` will only contain the value `{}` or a list of
    empty dictionaries as other options are already tested in
    ``test_utils_io.py::test_validate_output_path`` and are therefore not included here.
    """

    # construct channel values
    chan_vals = ["chan" + str(i) for i in range(1, n_chan + 1)]

    # create coordinates that will be used by all DataArrays created
    coords = {
        "channel": ("channel", chan_vals, {"long_name": "channel name"}),
        "ping_time": np.arange(n),
        "range_sample": np.arange(n),
    }

    # create input mask and obtain temporary directory, if it was created
    mask, _ = create_input_mask(mask_np, mask_file, coords)

    mask_out = _validate_and_collect_mask_input(
        mask=mask, storage_options_mask=storage_options_mask
    )

    if isinstance(mask_out, list):
        for ind, da in enumerate(mask_out):
            # create known solution for mask
            mask_da = xr.DataArray(
                data=[mask_np[ind] for i in range(n_chan)], coords=coords, name="mask_" + str(ind)
            )

            assert da.identical(mask_da)
    else:
        # create known solution for mask
        mask_da = xr.DataArray(data=[mask_np for i in range(n_chan)], coords=coords, name="mask_0")
        assert mask_out.identical(mask_da)


@pytest.mark.parametrize(
    ("n", "n_chan", "var_name", "fill_value"),
    [
        pytest.param(
            4,
            2,
            2.0,
            np.nan,
            marks=pytest.mark.xfail(
                strict=True, reason="This should fail because the var_name is not a string."
            ),
        ),
        pytest.param(
            4,
            2,
            "var3",
            np.nan,
            marks=pytest.mark.xfail(
                strict=True,
                reason="This should fail because mock_ds will " "not have var_name=var3 in it.",
            ),
        ),
        pytest.param(
            4,
            2,
            "var1",
            "1.0",
            marks=pytest.mark.xfail(
                strict=True, reason="This should fail because fill_value is an incorrect type."
            ),
        ),
        (4, 2, "var1", 1),
        (4, 2, "var1", 1.0),
        (2, 1, "var1", np.identity(2)[None, :]),
        (
            2,
            1,
            "var1",
            xr.DataArray(
                data=np.array([[[1.0, 0], [0, 1]]]),
                coords={"channel": ["chan1"], "ping_time": [0, 1], "range_sample": [0, 1]},
            ),
        ),
        pytest.param(
            4,
            2,
            "var1",
            np.identity(2),
            marks=pytest.mark.xfail(
                strict=True, reason="This should fail because fill_value is not the right shape."
            ),
        ),
        pytest.param(
            4,
            2,
            "var1",
            xr.DataArray(
                data=np.array([[1.0, 0], [0, 1]]),
                coords={"ping_time": [0, 1], "range_sample": [0, 1]},
            ),
            marks=pytest.mark.xfail(
                strict=True, reason="This should fail because fill_value is not the right shape."
            ),
        ),
    ],
    ids=[
        "wrong_var_name_type",
        "no_var_name_ds",
        "wrong_fill_value_type",
        "fill_value_int",
        "fill_value_float",
        "fill_value_np_array",
        "fill_value_DataArray",
        "fill_value_np_array_wrong_shape",
        "fill_value_DataArray_wrong_shape",
    ],
)
def test_check_var_name_fill_value(
    n: int, n_chan: int, var_name: str, fill_value: Union[int, float, np.ndarray, xr.DataArray]
):
    """
    Ensures that the function ``_check_var_name_fill_value`` is behaving as expected.

    Parameters
    ----------
    n: int
        The number of rows (``x``) and columns (``y``) of
        each channel matrix
    n_chan: int
        Determines the size of the ``channel`` coordinate
    var_name: {"var1", "var2"}
        The variable name in the mock Dataset to apply the mask to
    fill_value: int, float, np.ndarray, or xr.DataArray
        Value(s) at masked indices
    """

    # obtain mock Dataset containing var_name
    mock_ds = get_mock_source_ds_apply_mask(n, n_chan, is_delayed=False)

    _check_var_name_fill_value(source_ds=mock_ds, var_name=var_name, fill_value=fill_value)


@pytest.mark.parametrize(
    (
        "n",
        "n_chan",
        "var_name",
        "mask",
        "mask_file",
        "fill_value",
        "is_delayed",
        "var_masked_truth",
        "no_channel",
    ),
    [
        # single_mask_default_fill
        (
            2,
            1,
            "var1",
            np.identity(2),
            None,
            np.nan,
            False,
            np.array([[1, np.nan], [np.nan, 1]]),
            False,
        ),
        # single_mask_default_fill_no_channel
        (
            2,
            1,
            "var1",
            np.identity(2),
            None,
            np.nan,
            False,
            np.array([[1, np.nan], [np.nan, 1]]),
            True,
        ),
        # single_mask_float_fill
        (2, 1, "var1", np.identity(2), None, 2.0, False, np.array([[1, 2.0], [2.0, 1]]), False),
        # single_mask_np_array_fill
        (
            2,
            1,
            "var1",
            np.identity(2),
            None,
            np.array([[[np.nan, np.nan], [np.nan, np.nan]]]),
            False,
            np.array([[1, np.nan], [np.nan, 1]]),
            False,
        ),
        # single_mask_DataArray_fill
        (
            2,
            1,
            "var1",
            np.identity(2),
            None,
            xr.DataArray(
                data=np.array([[[np.nan, np.nan], [np.nan, np.nan]]]),
                coords={"channel": ["chan1"], "ping_time": [0, 1], "range_sample": [0, 1]},
            ),
            False,
            np.array([[1, np.nan], [np.nan, 1]]),
            False,
        ),
        # list_mask_all_np
        (
            2,
            1,
            "var1",
            [np.identity(2), np.array([[0, 1], [0, 1]])],
            [None, None],
            2.0,
            False,
            np.array([[2.0, 2.0], [2.0, 1]]),
            False,
        ),
        # single_mask_ds_delayed
        (2, 1, "var1", np.identity(2), None, 2.0, True, np.array([[1, 2.0], [2.0, 1]]), False),
        # single_mask_as_path
        (
            2,
            1,
            "var1",
            np.identity(2),
            "test.zarr",
            2.0,
            True,
            np.array([[1, 2.0], [2.0, 1]]),
            False,
        ),
        # list_mask_all_path
        (
            2,
            1,
            "var1",
            [np.identity(2), np.array([[0, 1], [0, 1]])],
            ["test0.zarr", "test1.zarr"],
            2.0,
            False,
            np.array([[2.0, 2.0], [2.0, 1]]),
            False,
        ),
        # list_mask_some_path
        (
            2,
            1,
            "var1",
            [np.identity(2), np.array([[0, 1], [0, 1]])],
            ["test0.zarr", None],
            2.0,
            False,
            np.array([[2.0, 2.0], [2.0, 1]]),
            False,
        ),
    ],
    ids=[
        "single_mask_default_fill",
        "single_mask_default_fill_no_channel",
        "single_mask_float_fill",
        "single_mask_np_array_fill",
        "single_mask_DataArray_fill",
        "list_mask_all_np",
        "single_mask_ds_delayed",
        "single_mask_as_path",
        "list_mask_all_path",
        "list_mask_some_path",
    ],
)
def test_apply_mask(
    n: int,
    n_chan: int,
    var_name: str,
    mask: Union[np.ndarray, List[np.ndarray]],
    mask_file: Optional[Union[str, List[str]]],
    fill_value: Union[int, float, np.ndarray, xr.DataArray],
    is_delayed: bool,
    var_masked_truth: np.ndarray,
    no_channel: bool,
):
    """
    Ensures that ``apply_mask`` functions correctly.

    Parameters
    ----------
    n: int
        The number of rows (``x``) and columns (``y``) of
        each channel matrix
    n_chan: int
        Determines the size of the ``channel`` coordinate
    var_name: {"var1", "var2"}
        The variable name in the mock Dataset to apply the mask to
    mask: np.ndarray or list of np.ndarray
        The mask(s) that should be applied to ``var_name``
    mask_file: str or list of str, optional
        If provided, the ``mask`` input will be written to a temporary directory
        with file name ``mask_file``. This will then be used in ``apply_mask``.
    fill_value: int, float, np.ndarray, or xr.DataArray
        Value(s) at masked indices
    var_masked_truth: np.ndarray
        The true value of ``var_name`` values after the mask has been applied
    is_delayed: bool
        If True, makes all variables in constructed mock Dataset Dask arrays,
        else they will be in-memory arrays
    """

    # obtain mock Dataset containing var_name
    mock_ds = get_mock_source_ds_apply_mask(n, n_chan, is_delayed)

    # create input mask and obtain temporary directory, if it was created
    mask, temp_dir = create_input_mask(mask, mask_file, mock_ds.coords)

    # create DataArray form of the known truth value
    var_masked_truth = xr.DataArray(
        data=np.stack([var_masked_truth for i in range(n_chan)]),
        coords=mock_ds[var_name].coords,
        attrs=mock_ds[var_name].attrs,
    )
    var_masked_truth.name = mock_ds[var_name].name

    if no_channel:
        mock_ds = mock_ds.isel(channel=0)
        mask = mask.isel(channel=0)
        var_masked_truth = var_masked_truth.isel(channel=0)

    # apply the mask to var_name
    masked_ds = echopype.mask.apply_mask(
        source_ds=mock_ds,
        var_name=var_name,
        mask=mask,
        fill_value=fill_value,
        storage_options_ds={},
        storage_options_mask={},
    )

    # check that masked_ds[var_name] == var_masked_truth
    assert masked_ds[var_name].equals(var_masked_truth)

    # check that the output Dataset has lazy elements, if the input was lazy
    if is_delayed:
        assert isinstance(masked_ds[var_name].data, dask.array.Array)

    if temp_dir:
        # remove the temporary directory, if it was created
        temp_dir.cleanup()


@pytest.mark.parametrize(
    ("source_has_ch", "mask_has_ch"),
    [
        (True, True),
        (False, True),
        (True, False),
        (False, False),
    ],
    ids=[
        "source_with_ch_mask_with_ch",
        "source_no_ch_mask_with_ch",
        "source_with_ch_mask_no_ch",
        "source_no_ch_mask_no_ch",
    ],
)
def test_apply_mask_channel_variation(source_has_ch, mask_has_ch):
    source_ds = get_mock_source_ds_apply_mask(2, 3, False)
    var_name = "var1"

    if mask_has_ch:
        mask = xr.DataArray(
            np.array([np.identity(2)]),
            coords={"channel": ["chA"], "ping_time": np.arange(2), "range_sample": np.arange(2)},
            attrs={"long_name": "mask_with_channel"},
        )
    else:
        mask = xr.DataArray(
            np.identity(2),
            coords={"ping_time": np.arange(2), "range_sample": np.arange(2)},
            attrs={"long_name": "mask_no_channel"},
        )

    if source_has_ch:
        masked_ds = echopype.mask.apply_mask(source_ds, mask, var_name)
    else:
        source_ds[f"{var_name}_ch0"] = source_ds[var_name].isel(channel=0).squeeze()
        var_name = f"{var_name}_ch0"
        masked_ds = echopype.mask.apply_mask(source_ds, mask, var_name)

    # Output dimension will be the same as source
    if source_has_ch:
        truth_da = xr.DataArray(
            np.array([[[1, np.nan], [np.nan, 1]]] * 3),
            coords={
                "channel": ["chan1", "chan2", "chan3"],
                "ping_time": np.arange(2),
                "range_sample": np.arange(2),
            },
            attrs=source_ds[var_name].attrs,
        )
    else:
        truth_da = xr.DataArray(
            [[1, np.nan], [np.nan, 1]],
            coords={"ping_time": np.arange(2), "range_sample": np.arange(2)},
            attrs=source_ds[var_name].attrs,
        )

    assert masked_ds[var_name].equals(truth_da)
