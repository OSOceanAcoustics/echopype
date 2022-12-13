import pytest

import numpy as np
import xarray as xr
import tempfile
import os.path

import echopype as ep
from echopype.mask.api import _check_source_Sv_freq_diff

from typing import List


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
    ]

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

    Notes
    -----
    The following checks are made:

    - If ``source_Sv_is_path=True`` then makes sure that

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



# @pytest.mark.parametrize(
#     ("n", "add_chan", "add_freq_nom"),
#     [
#
#     ]
#
# )
# def test_frequency_difference_mask(n, add_chan, add_freq_nom):
#     """
#     Tests specifically the mask generated by ``frequency_nominal``
#
#     """
#
#     mock_Sv_da, mock_Sv_ds = get_mock_freq_diff_data(n, add_chan, add_freq_nom)
#
#     out = ep.mask.frequency_difference(source_Sv=mock_Sv_ds, storage_options={}, freqAB=None,
#                                        chanAB=['chan1', 'chan2'],
#                                        operator="==", diff=1.0)
#
#     assert np.all(out == np.identity(n))





