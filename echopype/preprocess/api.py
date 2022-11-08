"""
Functions for enhancing the spatial and temporal coherence of data.
"""

from typing import List, Tuple, Union

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
    # ping_interval = np.array(list(ds_Sv.ping_time.resample(ping_time=ping_time_bin,
    # skipna=True).groups.keys()))

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
    # return MVBS_values


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


def mean_temp_arr(
    n_bin_er: int,
    temp_arr: Union[dask.array.Array, np.ndarray],
    dig_er_subset: Union[dask.array.Array, np.ndarray],
) -> List[Union[dask.array.Array, np.ndarray]]:
    """
    Bins the data in ``temp_arr`` with respect to the
    ``echo_range`` bin and means the resulting bin.

    Parameters
    ----------
    n_bin_er: int
        The number of echo range bins
    temp_arr: dask.array.Array or np.ndarray
        Array of Sv values at the ``ping_time`` bin index being considered
    dig_er_subset: dask.array.Array or np.ndarray
        Array representing the digitized (bin indices) for ``echo_range`` at
        the ``ping_time`` bin index being considered

    Returns
    -------
    means: list of dask.array.Array or np.ndarray

    Notes
    -----
    It is necessary for this to be a function because we may need to
    delay it.
    """

    means = []
    for bin_er in range(1, n_bin_er):
        means.append(np.nanmean(temp_arr[dig_er_subset == bin_er], axis=0))

    return means


def bin_and_mean_2d(
    arr: Union[dask.array.Array, np.ndarray],
    bins_time: np.ndarray,
    bins_er: np.ndarray,
    times: np.ndarray,
    echo_range: np.ndarray,
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

    Returns
    -------
    final_reduced: np.ndarray
        The final binned and mean ``arr``, if ``arr`` is ``Sv`` then this is the MVBS

    Notes
    -----
    This function assumes that ``arr`` has rows corresponding to
    ``ping_time`` and columns corresponding to ``echo_range``.

    This function allows the number of ``echo_range`` values to
    vary amongst ``ping_times``.
    """

    # determine if array to bin is lazy or not
    # is_lazy = False
    # if isinstance(arr, dask.array.Array):
    #     is_lazy = True

    # get the number of echo range and time bins
    n_bin_er = len(bins_er)
    n_bin_time = len(bins_time)
    print(f"n_bin_time = {n_bin_time}")

    # obtain the bin indices for echo_range and times
    digitized_echo_range, bin_time_ind = get_bin_indices(echo_range, bins_er, times, bins_time)

    binned_means = []
    for bin_er in range(1, n_bin_er):
        er_selected_data = np.nanmean(arr[:, digitized_echo_range == bin_er], axis=1)

        binned_means.append(er_selected_data)

    er_means = np.vstack(binned_means).compute()

    final = np.empty((n_bin_time, n_bin_er - 1))
    # for bin_time in range(1, n_bin_time + 1):
    for bin_time in range(1, n_bin_time):
        indices = np.argwhere(bin_time_ind == bin_time).flatten()

        if len(indices) == 0:
            final[bin_time - 1, :] = np.nanmean(er_means[:, :], axis=1)  # TODO: look into this
        else:
            final[bin_time - 1, :] = np.nanmean(er_means[:, indices], axis=1)

    return final

    # all_means = []
    # for bin_time in range(1, n_bin_time + 1):
    #
    #     # get the indices of time in bin index bin_time
    #     indices_time = np.argwhere(bin_time_ind == bin_time).flatten()
    #
    #     # select only those array values that are in the time bin being considered
    #     temp_arr = arr[indices_time, :]
    #
    #     # bin and mean with respect to echo_range bins
    #     if is_lazy:
    #         all_means.append(
    #             dask.delayed(mean_temp_arr)(
    #                 n_bin_er, temp_arr, digitized_echo_range[indices_time, :]
    #             )
    #         )
    #     else:
    #         all_means.append(
    #             mean_temp_arr(n_bin_er, temp_arr, digitized_echo_range[indices_time, :])
    #         )

    # if is_lazy:
    #     # compute all constructs means
    #     all_means = dask.compute(all_means)[0]  # TODO: make this into persist in the future?
    #
    # # construct final reduced form of arr
    # final_reduced = np.array(all_means)
    #
    # return final_reduced


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
    respect to ``echo_range`` will take place, then the binning and mean with respect to
    ``ping_time`` will be a delayed operation, and lastly the delayed values will be
    computed. It is necessary to apply a compute at the end of this method because Dask
    graph layers get too large and this makes downstream operations very inefficient.
    """

    all_MVBS = []
    for chan in ds_Sv.channel:

        # squeeze to remove "channel" dim if present
        # TODO: not sure why not already removed for the AZFP case. Investigate.
        ds = ds_Sv.sel(channel=chan).squeeze()

        echo_range = (
            ds["echo_range"]
            .dropna(dim="range_sample", how="all")
            .dropna(dim="ping_time")
            .isel(ping_time=0)
            .values
        )

        # average should be done in linear domain
        sv = 10 ** (ds["Sv"] / 10)

        # get MVBS for channel in linear domain
        chan_MVBS = bin_and_mean_2d(
            sv.data,
            bins_time=ping_interval,
            bins_er=echo_range_interval,
            times=sv.ping_time.data,
            echo_range=echo_range,  # ds["echo_range"].data,
        )

        # all_MVBS.append(chan_MVBS)
        # all_MVBS.extend(chan_MVBS)
        # return all_MVBS

        # apply inverse mapping to get back to the original domain and store values
        all_MVBS.append(10 * np.log10(chan_MVBS))

    # collect the MVBS values for each channel
    return np.stack(all_MVBS, axis=0)
