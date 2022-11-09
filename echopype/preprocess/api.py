"""
Functions for enhancing the spatial and temporal coherence of data.
"""

from typing import Tuple, Union

import dask.array
import numpy as np
import pandas as pd
import xarray as xr

from ..utils.prov import echopype_prov_attrs
from .noise_est import NoiseEst


def _check_range_uniqueness(da):
    """
    Check if range (``echo_range``) changes across ping in a given frequency channel.
    """
    # squeeze to remove "channel" dim if present
    # TODO: not sure why not already removed for the AZFP case. Investigate.
    da = da.squeeze()

    # remove pings with NaN entries if exist
    # since goal here is to check uniqueness
    if np.unique(da.isnull(), axis=0).shape[0] != 1:
        da = da.dropna(dim="ping_time", how="any")

    # remove padded NaN entries if exist for all pings
    da = da.dropna(dim="range_sample", how="all")

    ping_time_idx = np.argwhere([dim == "ping_time" for dim in da.dims])[0][0]
    if np.unique(da, axis=ping_time_idx).shape[ping_time_idx] == 1:
        return xr.DataArray(data=True, coords={"channel": da["channel"].values})
    else:
        return xr.DataArray(data=False, coords={"channel": da["channel"].values})


def _freq_MVBS(ds, rint, pbin):
    # squeeze to remove "channel" dim if present
    # TODO: not sure why not already removed for the AZFP case. Investigate.
    ds = ds.squeeze()

    # average should be done in linear domain
    sv = 10 ** (ds["Sv"] / 10)

    # set 1D coordinate using the 1st ping echo_range since identical for all pings
    # remove padded NaN entries if exist for all pings
    er = (
        ds["echo_range"]
        .dropna(dim="range_sample", how="all")
        .dropna(dim="ping_time")
        .isel(ping_time=0)
    )

    # use first ping er as indexer for sv
    sv = sv.sel(range_sample=er.range_sample.values)
    sv.coords["echo_range"] = (["range_sample"], er.values)
    sv = sv.swap_dims({"range_sample": "echo_range"})
    sv_groupby_bins = (
        sv.resample(ping_time=pbin, skipna=True)
        .mean(skipna=True)
        .groupby_bins("echo_range", bins=rint, right=False, include_lowest=True)
        .mean(skipna=True)
    )
    sv_groupby_bins.coords["echo_range"] = (["echo_range_bins"], rint[:-1])
    sv_groupby_bins = sv_groupby_bins.swap_dims({"echo_range_bins": "echo_range"}).drop_vars(
        "echo_range_bins"
    )
    return 10 * np.log10(sv_groupby_bins)


def _set_MVBS_attrs(ds):
    """
    Attach common attributes.

    Parameters
    ----------
    ds : xr.Dataset
        dataset containing MVBS
    """
    ds["ping_time"].attrs = {
        "long_name": "Ping time",
        "standard_name": "time",
        "axis": "T",
    }

    ds["Sv"].attrs = {
        "long_name": "Mean volume backscattering strength (MVBS, mean Sv re 1 m-1)",
        "units": "dB",
        "actual_range": [
            round(float(ds["Sv"].min().values), 2),
            round(float(ds["Sv"].max().values), 2),
        ],
    }


def compute_MVBS(ds_Sv, range_meter_bin=20, ping_time_bin="20S"):
    """
    Compute Mean Volume Backscattering Strength (MVBS)
    based on intervals of range (``echo_range``) and ``ping_time`` specified in physical units.

    Output of this function differs from that of ``compute_MVBS_index_binning``, which computes
    bin-averaged Sv according to intervals of ``echo_range`` and ``ping_time`` specified as
    index number.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        dataset containing Sv and ``echo_range`` [m]
    range_meter_bin : Union[int, float]
        bin size along ``echo_range`` in meters, default to ``20``
    ping_time_bin : str
        bin size along ``ping_time``, default to ``20S``

    Returns
    -------
    A dataset containing bin-averaged Sv
    """

    if not ds_Sv["echo_range"].groupby("channel").apply(_check_range_uniqueness).all():
        raise ValueError(
            "echo_range variable changes across pings in at least one of the frequency channels."
        )

    # Groupby freq in case of different echo_range (from different sampling intervals)
    range_interval = np.arange(0, ds_Sv["echo_range"].max() + range_meter_bin, range_meter_bin)
    ds_MVBS = (
        ds_Sv.groupby("channel")
        .apply(_freq_MVBS, args=(range_interval, ping_time_bin))
        .to_dataset()
    )
    # Added this check to support the test in test_process.py::test_compute_MVBS
    if "filenames" in ds_MVBS.variables:
        ds_MVBS = ds_MVBS.drop_vars("filenames")

    # ping_time_bin parsing and conversions
    # Need to convert between pd.Timedelta and np.timedelta64 offsets/frequency strings
    # https://xarray.pydata.org/en/stable/generated/xarray.Dataset.resample.html
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.resample.html
    # https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html
    # https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.resolution_string.html
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    # https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    timedelta_units = {
        "d": {"nptd64": "D", "unitstr": "day"},
        "h": {"nptd64": "h", "unitstr": "hour"},
        "t": {"nptd64": "m", "unitstr": "minute"},
        "min": {"nptd64": "m", "unitstr": "minute"},
        "s": {"nptd64": "s", "unitstr": "second"},
        "l": {"nptd64": "ms", "unitstr": "millisecond"},
        "ms": {"nptd64": "ms", "unitstr": "millisecond"},
        "u": {"nptd64": "us", "unitstr": "microsecond"},
        "us": {"nptd64": "ms", "unitstr": "millisecond"},
        "n": {"nptd64": "ns", "unitstr": "nanosecond"},
        "ns": {"nptd64": "ms", "unitstr": "millisecond"},
    }
    ping_time_bin_td = pd.Timedelta(ping_time_bin)
    # res = resolution (most granular time unit)
    ping_time_bin_resunit = ping_time_bin_td.resolution_string.lower()
    ping_time_bin_resvalue = int(
        ping_time_bin_td / np.timedelta64(1, timedelta_units[ping_time_bin_resunit]["nptd64"])
    )
    ping_time_bin_resunit_label = timedelta_units[ping_time_bin_resunit]["unitstr"]

    # Attach attributes
    _set_MVBS_attrs(ds_MVBS)
    ds_MVBS["echo_range"].attrs = {"long_name": "Range distance", "units": "m"}
    ds_MVBS["Sv"] = ds_MVBS["Sv"].assign_attrs(
        {
            "cell_methods": (
                f"ping_time: mean (interval: {ping_time_bin_resvalue} {ping_time_bin_resunit_label} "  # noqa
                "comment: ping_time is the interval start) "
                f"echo_range: mean (interval: {range_meter_bin} meter "
                "comment: echo_range is the interval start)"
            ),
            "binning_mode": "physical units",
            "range_meter_interval": str(range_meter_bin) + "m",
            "ping_time_interval": ping_time_bin,
            "actual_range": [
                round(float(ds_MVBS["Sv"].min().values), 2),
                round(float(ds_MVBS["Sv"].max().values), 2),
            ],
        }
    )

    prov_dict = echopype_prov_attrs(process_type="processing")
    prov_dict["processing_function"] = "preprocess.compute_MVBS"
    ds_MVBS = ds_MVBS.assign_attrs(prov_dict)
    ds_MVBS["frequency_nominal"] = ds_Sv["frequency_nominal"]  # re-attach frequency_nominal

    return ds_MVBS


def compute_MVBS_v2(ds_Sv, range_meter_bin=20, ping_time_bin="20S"):
    """
    Compute Mean Volume Backscattering Strength (MVBS)
    based on intervals of range (``echo_range``) and ``ping_time`` specified in physical units.

    Output of this function differs from that of ``compute_MVBS_index_binning``, which computes
    bin-averaged Sv according to intervals of ``echo_range`` and ``ping_time`` specified as
    index number.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        dataset containing Sv and ``echo_range`` [m]
    range_meter_bin : Union[int, float]
        bin size along ``echo_range`` in meters, default to ``20``
    ping_time_bin : str
        bin size along ``ping_time``, default to ``20S``

    Returns
    -------
    A dataset containing bin-averaged Sv
    """

    # TODO: remove this as it is no longer necessary
    # if not ds_Sv["echo_range"].groupby("channel").apply(_check_range_uniqueness).all():
    #     raise ValueError(
    #         "echo_range variable changes across pings in at least one of the frequency channels."
    #     )

    # create bin information for echo_range
    range_interval = np.arange(0, ds_Sv["echo_range"].max() + range_meter_bin, range_meter_bin)

    # create bin information needed for ping_time
    ping_interval = (
        ds_Sv.ping_time.resample(ping_time=ping_time_bin, skipna=True).asfreq().ping_time.values
    )

    # calculate the MVBS along each channel
    MVBS_values = get_MVBS_along_channels(ds_Sv, range_interval, ping_interval)

    # create MVBS dataset
    ds_MVBS = xr.Dataset(
        data_vars={"Sv": (["channel", "ping_time", "echo_range"], MVBS_values)},
        coords={
            "ping_time": ping_interval,
            "channel": ds_Sv.channel,
            "echo_range": range_interval[:-1],
        },
    )

    # Added this check to support the test in test_process.py::test_compute_MVBS
    if "filenames" in ds_MVBS.variables:
        ds_MVBS = ds_MVBS.drop_vars("filenames")

    # ping_time_bin parsing and conversions
    # Need to convert between pd.Timedelta and np.timedelta64 offsets/frequency strings
    # https://xarray.pydata.org/en/stable/generated/xarray.Dataset.resample.html
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.resample.html
    # https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html
    # https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.resolution_string.html
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    # https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    timedelta_units = {
        "d": {"nptd64": "D", "unitstr": "day"},
        "h": {"nptd64": "h", "unitstr": "hour"},
        "t": {"nptd64": "m", "unitstr": "minute"},
        "min": {"nptd64": "m", "unitstr": "minute"},
        "s": {"nptd64": "s", "unitstr": "second"},
        "l": {"nptd64": "ms", "unitstr": "millisecond"},
        "ms": {"nptd64": "ms", "unitstr": "millisecond"},
        "u": {"nptd64": "us", "unitstr": "microsecond"},
        "us": {"nptd64": "ms", "unitstr": "millisecond"},
        "n": {"nptd64": "ns", "unitstr": "nanosecond"},
        "ns": {"nptd64": "ms", "unitstr": "millisecond"},
    }
    ping_time_bin_td = pd.Timedelta(ping_time_bin)
    # res = resolution (most granular time unit)
    ping_time_bin_resunit = ping_time_bin_td.resolution_string.lower()
    ping_time_bin_resvalue = int(
        ping_time_bin_td / np.timedelta64(1, timedelta_units[ping_time_bin_resunit]["nptd64"])
    )
    ping_time_bin_resunit_label = timedelta_units[ping_time_bin_resunit]["unitstr"]

    # Attach attributes
    _set_MVBS_attrs(ds_MVBS)
    ds_MVBS["echo_range"].attrs = {"long_name": "Range distance", "units": "m"}
    ds_MVBS["Sv"] = ds_MVBS["Sv"].assign_attrs(
        {
            "cell_methods": (
                f"ping_time: mean (interval: {ping_time_bin_resvalue} {ping_time_bin_resunit_label} "  # noqa
                "comment: ping_time is the interval start) "
                f"echo_range: mean (interval: {range_meter_bin} meter "
                "comment: echo_range is the interval start)"
            ),
            "binning_mode": "physical units",
            "range_meter_interval": str(range_meter_bin) + "m",
            "ping_time_interval": ping_time_bin,
            "actual_range": [
                round(float(ds_MVBS["Sv"].min().values), 2),
                round(float(ds_MVBS["Sv"].max().values), 2),
            ],
        }
    )

    prov_dict = echopype_prov_attrs(process_type="processing")
    prov_dict["processing_function"] = "preprocess.compute_MVBS"
    ds_MVBS = ds_MVBS.assign_attrs(prov_dict)
    ds_MVBS["frequency_nominal"] = ds_Sv["frequency_nominal"]  # re-attach frequency_nominal

    return ds_MVBS


def compute_MVBS_index_binning(ds_Sv, range_sample_num=100, ping_num=100):
    """
    Compute Mean Volume Backscattering Strength (MVBS)
    based on intervals of ``range_sample`` and ping number (``ping_num``) specified in index number.

    Output of this function differs from that of ``compute_MVBS``, which computes
    bin-averaged Sv according to intervals of range (``echo_range``) and ``ping_time`` specified
    in physical units.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        dataset containing ``Sv`` and ``echo_range`` [m]
    range_sample_num : int
        number of samples to average along the ``range_sample`` dimension, default to 100
    ping_num : int
        number of pings to average, default to 100

    Returns
    -------
    A dataset containing bin-averaged Sv
    """
    da_sv = 10 ** (ds_Sv["Sv"] / 10)  # average should be done in linear domain
    da = 10 * np.log10(
        da_sv.coarsen(ping_time=ping_num, range_sample=range_sample_num, boundary="pad").mean(
            skipna=True
        )
    )

    # Attach attributes and coarsened echo_range
    da.name = "Sv"
    ds_MVBS = da.to_dataset()
    ds_MVBS.coords["range_sample"] = (
        "range_sample",
        np.arange(ds_MVBS["range_sample"].size),
        {"long_name": "Along-range sample number, base 0"},
    )  # reset range_sample to start from 0
    ds_MVBS["echo_range"] = (
        ds_Sv["echo_range"]
        .coarsen(  # binned echo_range (use first value in each average bin)
            ping_time=ping_num, range_sample=range_sample_num, boundary="pad"
        )
        .min(skipna=True)
    )
    _set_MVBS_attrs(ds_MVBS)
    ds_MVBS["Sv"] = ds_MVBS["Sv"].assign_attrs(
        {
            "cell_methods": (
                f"ping_time: mean (interval: {ping_num} pings "
                "comment: ping_time is the interval start) "
                f"range_sample: mean (interval: {range_sample_num} samples along range "
                "comment: range_sample is the interval start)"
            ),
            "comment": "MVBS binned on the basis of range_sample and ping number specified as index numbers",  # noqa
            "binning_mode": "sample number",
            "range_sample_interval": f"{range_sample_num} samples along range",
            "ping_interval": f"{ping_num} pings",
            "actual_range": [
                round(float(ds_MVBS["Sv"].min().values), 2),
                round(float(ds_MVBS["Sv"].max().values), 2),
            ],
        }
    )

    prov_dict = echopype_prov_attrs(process_type="processing")
    prov_dict["processing_function"] = "preprocess.compute_MVBS_index_binning"
    ds_MVBS = ds_MVBS.assign_attrs(prov_dict)
    ds_MVBS["frequency_nominal"] = ds_Sv["frequency_nominal"]  # re-attach frequency_nominal

    return ds_MVBS


def estimate_noise(ds_Sv, ping_num, range_sample_num, noise_max=None):
    """
    Remove noise by using estimates of background noise
    from mean calibrated power of a collection of pings.

    See ``remove_noise`` for reference.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        dataset containing ``Sv`` and ``echo_range`` [m]
    ping_num : int
        number of pings to obtain noise estimates
    range_sample_num : int
        number of samples along the ``range_sample`` dimension to obtain noise estimates
    noise_max : float
        the upper limit for background noise expected under the operating conditions

    Returns
    -------
    A DataArray containing noise estimated from the input ``ds_Sv``
    """
    noise_obj = NoiseEst(ds_Sv=ds_Sv.copy(), ping_num=ping_num, range_sample_num=range_sample_num)
    noise_obj.estimate_noise(noise_max=noise_max)
    return noise_obj.Sv_noise


def remove_noise(ds_Sv, ping_num, range_sample_num, noise_max=None, SNR_threshold=3):
    """
    Remove noise by using estimates of background noise
    from mean calibrated power of a collection of pings.

    Reference: De Robertis & Higginbottom. 2007.
    A post-processing technique to estimate the signal-to-noise ratio
    and remove echosounder background noise.
    ICES Journal of Marine Sciences 64(6): 1282–1291.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        dataset containing ``Sv`` and ``echo_range`` [m]
    ping_num : int
        number of pings to obtain noise estimates
    range_sample_num : int
        number of samples along the ``range_sample`` dimension to obtain noise estimates
    noise_max : float
        the upper limit for background noise expected under the operating conditions
    SNR_threshold : float
        acceptable signal-to-noise ratio, default to 3 dB

    Returns
    -------
    The input dataset with additional variables, including
    the corrected Sv (``Sv_corrected``) and the noise estimates (``Sv_noise``)
    """
    noise_obj = NoiseEst(ds_Sv=ds_Sv.copy(), ping_num=ping_num, range_sample_num=range_sample_num)
    noise_obj.remove_noise(noise_max=noise_max, SNR_threshold=SNR_threshold)
    ds_Sv = noise_obj.ds_Sv

    prov_dict = echopype_prov_attrs(process_type="processing")
    prov_dict["processing_function"] = "preprocess.remove_noise"
    ds_Sv = ds_Sv.assign_attrs(prov_dict)

    return ds_Sv


def regrid():
    return 1


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


def bin_and_mean_echo_range(arr, digitized_echo_range, n_bin_er):

    # TODO: document!

    binned_means = []
    for bin_er in range(1, n_bin_er):
        # bin and mean echo_range dimension
        er_selected_data = np.nanmean(arr[:, digitized_echo_range == bin_er], axis=1)

        # collect all echo_range bins
        binned_means.append(er_selected_data)

    # create full echo_range binned array
    er_means = np.vstack(binned_means)

    return er_means


def get_unequal_rows(mat, row):
    # compare row against all rows in mat (allowing for NaNs to be equal)
    element_nan_equal = (mat == row) | (np.isnan(mat) & np.isnan(row))

    # get those row indices that are not equal to row
    row_not_equal = np.argwhere(np.logical_not(np.all(element_nan_equal, axis=1))).flatten()

    return row_not_equal


def is_grouping_needed_comprehensive(er_chan):

    if isinstance(er_chan, xr.DataArray):
        er_chan = er_chan.values

    # grab the first ping_time that is not filled with NaNs
    ping_index = 0
    while np.all(np.isnan(er_chan[ping_index, :])):
        ping_index += 1

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


def is_grouping_needed_less_comprehensive(er_chan):
    """


    Parameters
    ----------
    er_chan

    Notes
    -----
    Although this method is slightly faster than ``is_grouping_needed_comprehensive``,
    it is possible that this method will incorrectly determine if grouping
    is necessary.
    """

    # determine the number of NaNs in each ping and find the unique number of NaNs
    unique_num_nans = np.unique(np.isnan(er_chan.data).sum(axis=1))

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

        if isinstance(num_non_nans, dask.array.Array):
            num_non_nans = num_non_nans.compute()

        if num_non_nans > 1:
            # echo_range varies with ping_time
            return True
        else:
            # echo_range does not vary with ping_time
            return False


def bin_and_mean_2d(
    arr: Union[dask.array.Array, np.ndarray],
    bins_time: np.ndarray,
    bins_er: np.ndarray,
    times: np.ndarray,
    echo_range: np.ndarray,
    comp_er_check: bool = True,
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
    comp_er_check: bool
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
    vary amongst ``ping_times``.
    """

    # get the number of echo range and time bins
    n_bin_er = len(bins_er)
    n_bin_time = len(bins_time)

    # obtain the bin indices for echo_range and times
    digitized_echo_range, bin_time_ind = get_bin_indices(echo_range, bins_er, times, bins_time)

    # TODO: clean up the below code!

    if comp_er_check:
        grouping_needed = is_grouping_needed_comprehensive(echo_range)
    else:
        grouping_needed = is_grouping_needed_less_comprehensive(echo_range)

        # TODO: put the below into its own function
    if grouping_needed:

        print("grouping necessary")

        # compute bin indices to allow for downstream processes (mainly axis argument in unique)
        if isinstance(digitized_echo_range, dask.array.Array):
            digitized_echo_range = digitized_echo_range.compute()

        unique_er_bin_ind, unique_inverse = np.unique(
            digitized_echo_range, axis=0, return_inverse=True
        )
        grps_same_ind = [
            np.argwhere(unique_inverse == grp).flatten() for grp in np.unique(unique_inverse)
        ]

        binned_er = []  # the values appended may not be in the correct order
        for count, grp in enumerate(grps_same_ind):
            binned_er.append(
                bin_and_mean_echo_range(arr[grp, :], unique_er_bin_ind[count, :], n_bin_er)
            )

        # put the columns in the correct order
        binned_er_array = np.hstack(binned_er)
        correct_column_ind = np.argsort(np.concatenate(grps_same_ind))
        er_means = binned_er_array[:, correct_column_ind]

    else:
        print("no grouping necessary")
        er_means = bin_and_mean_echo_range(arr, digitized_echo_range[0, :], n_bin_er)

    # if er_means is a dask array we compute it so the graph does not get too large
    if isinstance(er_means, dask.array.Array):
        er_means = er_means.compute()

    # create final reduced array
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
        )

        # apply inverse mapping to get back to the original domain and store values
        all_MVBS.append(10 * np.log10(chan_MVBS))

    # collect the MVBS values for each channel
    return np.stack(all_MVBS, axis=0)
