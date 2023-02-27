"""
Contains core functions needed to compute the MVBS of an input dataset.
"""

import warnings
from typing import Tuple, Union

import dask.array
import numpy as np
import xarray as xr


def get_bin_indices(
    echo_range: np.ndarray, bins_er: np.ndarray, times: np.ndarray, bins_time: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Obtains the bin index of ``echo_range`` and ``times`` based
    on the binning ``bins_er`` and ``bins_time``, respectively.

    Parameters
    ----------
    echo_range: np.ndarray
        2D array of echo range values
    bins_er: np.ndarray
        1D array (used by np.digitize) representing the binning required for ``echo_range``
    times: np.ndarray
        1D array corresponding to the time values that should be binned
    bins_time: np.ndarray
        1D array (used by np.digitize) representing the binning required for ``times``

    Returns
    -------
    digitized_echo_range: np.ndarray
        2D array of bin indices for ``echo_range``
    bin_time_ind: np.ndarray
        1D array of bin indices for ``times``
    """

    # get bin index for each echo range value
    digitized_echo_range = np.digitize(echo_range, bins_er, right=False)

    # turn datetime into integers, so we can use np.digitize
    if isinstance(times, dask.array.Array):
        times_i8 = times.compute().data.view("i8")
    else:
        times_i8 = times.view("i8")

    # turn datetime into integers, so we can use np.digitize
    bins_time_i8 = bins_time.view("i8")

    # get bin index for each time
    bin_time_ind = np.digitize(times_i8, bins_time_i8, right=False)

    return digitized_echo_range, bin_time_ind


def bin_and_mean_echo_range(
    arr: Union[np.ndarray, dask.array.Array], digitized_echo_range: np.ndarray, n_bin_er: int
) -> Union[np.ndarray, dask.array.Array]:
    """
    Bins and means ``arr`` with respect to the ``echo_range`` bins.

    Parameters
    ----------
    arr: np.ndarray or dask.array.Array
        2D array (dimension: [``echo_range`` x ``ping_time``]) to bin  along ``echo_range``
        and compute mean of each bin
    digitized_echo_range: np.ndarray
        2D array of bin indices for ``echo_range``
    n_bin_er: int
        The number of echo range bins

    Returns
    -------
    er_means: np.ndarray or dask.array.Array
        2D array representing the bin and mean of ``arr`` along ``echo_range``
    """

    binned_means = []
    for bin_er in range(1, n_bin_er):
        # Catch a known warning that can occur, which does not impact the results
        with warnings.catch_warnings():
            # ignore warnings caused by taking a mean of an array filled with NaNs
            warnings.filterwarnings(action="ignore", message="Mean of empty slice")

            # bin and mean echo_range dimension
            er_selected_data = np.nanmean(arr[:, digitized_echo_range == bin_er], axis=1)

        # collect all echo_range bins
        binned_means.append(er_selected_data)

    # create full echo_range binned array
    er_means = np.vstack(binned_means)

    return er_means


def get_unequal_rows(mat: np.ndarray, row: np.ndarray) -> np.ndarray:
    """
    Obtains those row indices of ``mat`` that are not equal
    to ``row``.

    Parameters
    ----------
    mat: np.ndarray
        2D array with the same column dimension as the number
        of elements in ``row``
    row: np.ndarray
        1D array with the same number of element elements as
        the column dimension of ``mat``

    Returns
    -------
    row_ind_not_equal: np.ndarray
        The row indices of ``mat`` that are not equal to ``row``

    Notes
    -----
    Elements with NaNs are considered equal if they are in the same position.
    """

    # compare row against all rows in mat (allowing for NaNs to be equal)
    element_nan_equal = (mat == row) | (np.isnan(mat) & np.isnan(row))

    # determine if mat row is equal to row
    row_not_equal = np.logical_not(np.all(element_nan_equal, axis=1))

    if isinstance(row_not_equal, dask.array.Array):
        row_not_equal = row_not_equal.compute()

    # get those row indices that are not equal to row
    row_ind_not_equal = np.argwhere(row_not_equal).flatten()

    return row_ind_not_equal


def if_all_er_steps_identical(er_chan: Union[xr.DataArray, np.ndarray]) -> bool:
    """
    A comprehensive check that determines if all ``echo_range`` values
    along ``ping_time`` have the same step size. If they do not have
    the same step sizes, then grouping of the ``echo_range`` values
    will be necessary.

    Parameters
    ----------
    er_chan: xr.DataArray or np.ndarray
        2D array containing the ``echo_range`` values for each ``ping_time``

    Returns
    -------
    bool
        True, if grouping of ``echo_range`` along ``ping_time`` is necessary, otherwise False

    Notes
    -----
    ``er_chan`` should have rows corresponding to ``ping_time`` and columns
    corresponding to ``range_sample``
    """

    # grab the in-memory numpy echo_range values, if necessary
    if isinstance(er_chan, xr.DataArray):
        er_chan = er_chan.values

    # grab the first ping_time that is not filled with NaNs
    ping_index = 0
    while np.all(np.isnan(er_chan[ping_index, :])):
        ping_index += 1

    # determine those rows of er_chan that are not equal to the row ping_index
    unequal_ping_ind = get_unequal_rows(er_chan, er_chan[ping_index, :])

    if len(unequal_ping_ind) > 0:
        # see if all unequal_ping_ind are filled with NaNs
        all_nans = np.all(np.all(np.isnan(er_chan[unequal_ping_ind, :]), axis=1))

        if all_nans:
            # All echo_range values have the same step size
            return False
        else:
            # Some echo_range values have different step sizes
            return True
    else:
        # All echo_range values have the same step size
        return False


def if_last_er_steps_identical(er_chan: Union[xr.DataArray, np.ndarray]) -> bool:
    """
    An alternative (less comprehensive) check that determines if all
    ``echo_range`` values along ``ping_time`` have the same step size.
    If they do not have the same step sizes, then grouping of the
    ``echo_range`` values will be necessary.

    Parameters
    ----------
    er_chan: xr.DataArray or np.ndarray
        2D array containing the ``echo_range`` values for each ``ping_time``

    Returns
    -------
    bool
        True, if grouping of ``echo_range`` along ``ping_time`` is necessary, otherwise False

    Notes
    -----
    It is possible that this method will incorrectly determine if grouping
    is necessary.

    ``er_chan`` should have rows corresponding to ``ping_time`` and columns
    corresponding to ``range_sample``
    """

    # determine the number of NaNs in each ping and find the unique number of NaNs
    unique_num_nans = np.unique(np.isnan(er_chan.data).sum(axis=1))

    # compute the results, if necessary, to allow for downstream checks
    if isinstance(unique_num_nans, dask.array.Array):
        unique_num_nans = unique_num_nans.compute()

    # determine if any value is not 0 or er_chan.shape[1]
    unexpected_num_nans = False in np.logical_or(
        unique_num_nans == 0, unique_num_nans == er_chan.shape[1]
    )

    if unexpected_num_nans:
        # echo_range varies with ping_time
        return True
    else:
        # make sure that the final echo_range value for each ping_time is the same (account for NaN)
        num_non_nans = np.logical_not(np.isnan(np.unique(er_chan.data[:, -1]))).sum()

        # compute the results, if necessary, to allow for downstream checks
        if isinstance(num_non_nans, dask.array.Array):
            num_non_nans = num_non_nans.compute()

        if num_non_nans > 1:
            # echo_range varies with ping_time
            return True
        else:
            # echo_range does not vary with ping_time
            return False


def is_er_grouping_needed(
    echo_range: Union[xr.DataArray, np.ndarray], comprehensive_er_check: bool
) -> bool:
    """
    Determines if ``echo_range`` values along ``ping_time`` can change and
    thus need to be grouped.

    Parameters
    ----------
    echo_range: xr.DataArray or np.ndarray
        2D array containing the ``echo_range`` values for each ``ping_time``
    comprehensive_er_check: bool
        If True, a more comprehensive check will be completed to determine if ``echo_range``
        grouping along ``ping_time`` is needed, otherwise a less comprehensive check will be done

    Returns
    -------
    bool
        If True grouping of ``echo_range`` will be required, else it will not
        be necessary
    """

    if comprehensive_er_check:
        return if_all_er_steps_identical(echo_range)
    else:
        return if_last_er_steps_identical(echo_range)


def group_dig_er_bin_mean_echo_range(
    arr: Union[np.ndarray, dask.array.Array],
    digitized_echo_range: Union[np.ndarray, dask.array.Array],
    n_bin_er: int,
) -> Union[np.ndarray, dask.array.Array]:
    """
    Groups the rows of ``arr`` such that they have the same corresponding
    row values in ``digitized_echo_range``, then applies ``bin_and_mean_echo_range``
    on each group, and lastly assembles the correctly ordered ``er_means`` array
    representing the bin and mean of ``arr`` with respect to ``echo_range``.

    Parameters
    ----------
    arr: dask.array.Array or np.ndarray
        The 2D array whose values should be binned
    digitized_echo_range: dask.array.Array or np.ndarray
        2D array of bin indices for ``echo_range``
    n_bin_er: int
        The number of echo range bins

    Returns
    -------
    er_means: dask.array.Array or np.ndarray
        The bin and mean of ``arr`` with respect to ``echo_range``
    """

    # compute bin indices to allow for downstream processes (mainly axis argument in unique)
    if isinstance(digitized_echo_range, dask.array.Array):
        digitized_echo_range = digitized_echo_range.compute()

    # determine the unique rows of digitized_echo_range and the inverse
    unique_er_bin_ind, unique_inverse = np.unique(digitized_echo_range, axis=0, return_inverse=True)

    # create groups of row indices using the unique inverse
    grps_same_ind = [
        np.argwhere(unique_inverse == grp).flatten() for grp in np.unique(unique_inverse)
    ]

    # for each group bin and mean arr along echo_range
    # note: the values appended may not be in the correct final order
    binned_er = []
    for count, grp in enumerate(grps_same_ind):
        binned_er.append(
            bin_and_mean_echo_range(arr[grp, :], unique_er_bin_ind[count, :], n_bin_er)
        )

    # construct er_means and put the columns in the correct order
    binned_er_array = np.hstack(binned_er)
    correct_column_ind = np.argsort(np.concatenate(grps_same_ind))
    er_means = binned_er_array[:, correct_column_ind]

    return er_means


def bin_and_mean_2d(
    arr: Union[dask.array.Array, np.ndarray],
    bins_time: np.ndarray,
    bins_er: np.ndarray,
    times: np.ndarray,
    echo_range: np.ndarray,
    comprehensive_er_check: bool = True,
) -> np.ndarray:
    """
    Bins and means ``arr`` based on ``times`` and ``echo_range``,
    and their corresponding bins. If ``arr`` is ``Sv`` then this
    will compute the MVBS.

    Parameters
    ----------
    arr: dask.array.Array or np.ndarray
        The 2D array whose values should be binned
    bins_time: np.ndarray
        1D array (used by np.digitize) representing the binning required for ``times``
    bins_er: np.ndarray
        1D array (used by np.digitize) representing the binning required for ``echo_range``
    times: np.ndarray
        1D array corresponding to the time values that should be binned
    echo_range: np.ndarray
        2D array of echo range values
    comprehensive_er_check: bool
        If True, a more comprehensive check will be completed to determine if ``echo_range``
        grouping along ``ping_time`` is needed, otherwise a less comprehensive check will be done

    Returns
    -------
    final_reduced: np.ndarray
        The final binned and mean ``arr``, if ``arr`` is ``Sv`` then this is the MVBS

    Notes
    -----
    This function assumes that ``arr`` has rows corresponding to
    ``ping_time`` and columns corresponding to ``echo_range``.

    This function should not be run if the number of ``echo_range`` values
    vary amongst ``ping_times``. This should not occur for our current use
    of echopype-generated Sv data.
    """

    # get the number of echo range and time bins
    n_bin_er = len(bins_er)
    n_bin_time = len(bins_time)

    # obtain the bin indices for echo_range and times
    digitized_echo_range, bin_time_ind = get_bin_indices(echo_range, bins_er, times, bins_time)

    # determine if grouping of echo_range values with the same step size is necessary
    er_grouping_needed = is_er_grouping_needed(echo_range, comprehensive_er_check)

    if er_grouping_needed:
        # groups, bins, and means arr with respect to echo_range
        er_means = group_dig_er_bin_mean_echo_range(arr, digitized_echo_range, n_bin_er)
    else:
        # bin and mean arr with respect to echo_range
        er_means = bin_and_mean_echo_range(arr, digitized_echo_range[0, :], n_bin_er)

    # if er_means is a dask array we compute it so the graph does not get too large
    if isinstance(er_means, dask.array.Array):
        er_means = er_means.compute()

    # create final reduced array i.e. MVBS
    final = np.empty((n_bin_time, n_bin_er - 1))
    for bin_time in range(1, n_bin_time + 1):
        # obtain er_mean indices corresponding to the time bin
        indices = np.argwhere(bin_time_ind == bin_time).flatten()

        if len(indices) == 0:
            # fill values with NaN, if there are no values in the bin
            final[bin_time - 1, :] = np.nan
        else:
            # bin and mean the er_mean time bin
            final[bin_time - 1, :] = np.nanmean(er_means[:, indices], axis=1)

    return final


def get_MVBS_along_channels(
    ds_Sv: xr.Dataset, echo_range_interval: np.ndarray, ping_interval: np.ndarray
) -> np.ndarray:
    """
    Computes the MVBS of ``ds_Sv`` along each channel for the given
    intervals.

    Parameters
    ----------
    ds_Sv: xr.Dataset
        A Dataset containing ``Sv`` and ``echo_range`` data with coordinates
        ``channel``, ``ping_time``, and ``range_sample``
    echo_range_interval: np.ndarray
        1D array (used by np.digitize) representing the binning required for ``echo_range``
    ping_interval: np.ndarray
        1D array (used by np.digitize) representing the binning required for ``ping_time``

    Returns
    -------
    np.ndarray
        The MVBS value of the input ``ds_Sv`` for all channels

    Notes
    -----
    If the values in ``ds_Sv`` are delayed then the binning and mean of ``Sv`` with
    respect to ``echo_range`` will take place, then the delayed result will be computed,
    and lastly the binning and mean with respect to ``ping_time`` will be completed. It
    is necessary to apply a compute midway through this method because Dask graph layers
    get too large and this makes downstream operations very inefficient.
    """

    all_MVBS = []
    for chan in ds_Sv.channel:
        # squeeze to remove "channel" dim if present
        # TODO: not sure why not already removed for the AZFP case. Investigate.
        ds = ds_Sv.sel(channel=chan).squeeze()

        # average should be done in linear domain
        sv = 10 ** (ds["Sv"] / 10)

        # get MVBS for channel in linear domain
        chan_MVBS = bin_and_mean_2d(
            sv.data,
            bins_time=ping_interval,
            bins_er=echo_range_interval,
            times=sv.ping_time.data,
            echo_range=ds["echo_range"],
            comprehensive_er_check=True,
        )

        # apply inverse mapping to get back to the original domain and store values
        all_MVBS.append(10 * np.log10(chan_MVBS))

    # collect the MVBS values for each channel
    return np.stack(all_MVBS, axis=0)
