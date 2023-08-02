import pytest
import numpy as np
import xarray as xr

import echopype.mask


@pytest.mark.parametrize(
    "n, n_chan_freq, channel, r0, r1, method, expected",
    [
        (
            4, 4, 'chan1', 1, np.nan, 'below',
            np.array([
                [False, False, True, True],
                [False, False, True, True],
                [False, False, True, True],
                [False, False, True, True]
            ])
        ),
        (
            4, 4, 'chan1', 1, np.nan, 'above',
            np.array([
                [True, False, False, False],
                [True, False, False, False],
                [True, False, False, False],
                [True, False, False, False]
            ])
        ),
        (
            4, 4, 'chan1', 1, 3, 'inside',
            np.array([
                [False,  True,  True, True],
                [False,  True,  True, True],
                [False,  True,  True, True],
                [False,  True,  True, True]
            ])

        ),
        (
            4, 4, 'chan1', 1, 3, 'outside',
            np.array([
                [True, False, False, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, False, False, True]
            ])
        ),
    ]
)

def test_get_range_mask(n: int, n_chan_freq: int, channel: str, r0: float, r1: float, method: str, expected: np.ndarray):
    def get_mock_range_mask_data(n: int, n_chan_freq: int) -> xr.Dataset:
        """
        Creates an in-memory mock Sv Dataset.

        Parameters
        ----------
        n: int
            The number of rows (``ping_time``) and columns (``range_sample``) of
            each channel matrix
        n_chan_freq: int
            Determines the size of the ``channel`` coordinate

        Returns
        -------
        mock_Sv_ds: xr.Dataset
            A mock Sv dataset to be used for ``mask_range`` tests. The Sv
            data values for the channel coordinate ``chan1`` will be equal to ``mat_A``,
            ``chan3`` will be equal to ``mat_B``, and all other channel coordinates
            will retain the value of ``np.identity(n)``.

        """

        if n_chan_freq < 1:
            raise RuntimeError("The input n_chan_freq must be greater than or equal to 1!")

        # matrix representing freqB
        mat_B = np.arange(n ** 2).reshape(n, n) - np.identity(n)

        # matrix representing freqA
        mat_A = np.arange(n ** 2).reshape(n, n)

        # construct channel values
        chan_vals = ['chan' + str(i) for i in range(1, n_chan_freq + 1)]

        # construct mock Sv data
        mock_Sv_data = [mat_A, np.identity(n), mat_B] + [np.identity(n) for i in range(3, n_chan_freq)]

        # set channel coordinate name
        channel_coord_name = "channel"

        # create mock Sv DataArray
        mock_Sv_da = xr.DataArray(data=np.stack(mock_Sv_data),
                                  coords={channel_coord_name: chan_vals, "ping_time": np.arange(n),
                                          "range_sample": np.arange(n)})

        # create data variables for the Dataset
        data_vars = {"Sv": mock_Sv_da}


        # create mock Dataset with Sv
        mock_Sv_ds = xr.Dataset(data_vars=data_vars)

        return mock_Sv_ds


    Sv_ds = get_mock_range_mask_data(n, n_chan_freq)
    result = echopype.mask.get_range_mask(Sv_ds, channel=channel, r0=r0, r1=r1, method=method)
    np.testing.assert_array_equal(result.values, expected)