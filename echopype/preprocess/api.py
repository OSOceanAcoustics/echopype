"""
Functions for enhancing the spatial and temporal coherence of data.
"""

import numpy as np
import pandas as pd

from .noise_est import NoiseEst


def _check_range_uniqueness(ds):
    """Check if range (``echo_range``) changes across ping in a given frequency channel."""
    return (
        ds["echo_range"].isel(ping_time=0).dropna(dim="range_bin")
        == ds["echo_range"].dropna(dim="range_bin")
    ).all()


def compute_MVBS(ds_Sv, range_meter_bin=20, ping_time_bin="20S"):
    """Compute Mean Volume Backscattering Strength (MVBS)
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

    if not ds_Sv.groupby("frequency").apply(_check_range_uniqueness).all():
        raise ValueError(
            "echo_range variable changes across pings in at least one of the frequency channel."
        )

    def _freq_MVBS(ds, rint, pbin):
        sv = 10 ** (ds["Sv"] / 10)  # average should be done in linear domain
        sv.coords["range_meter"] = (
            ["range_bin"],
            ds_Sv["echo_range"].isel(frequency=0, ping_time=0).data,
        )
        sv = sv.swap_dims({"range_bin": "range_meter"})
        sv_groupby_bins = (
            sv.groupby_bins("range_meter", bins=rint, right=False, include_lowest=True)
            .mean()
            .resample(ping_time=pbin, skipna=True)
            .mean()
        )
        sv_groupby_bins.coords["echo_range"] = (["range_meter_bins"], rint[:-1])
        sv_groupby_bins = sv_groupby_bins.swap_dims({"range_meter_bins": "echo_range"})
        sv_groupby_bins = sv_groupby_bins.drop_vars("range_meter_bins")
        return 10 * np.log10(sv_groupby_bins)

    # Groupby freq in case of different echo_range (from different sampling intervals)
    range_interval = np.arange(
        0, ds_Sv["echo_range"].max() + range_meter_bin, range_meter_bin
    )
    ds_MVBS = ds_Sv.groupby("frequency").apply(
        _freq_MVBS, args=(range_interval, ping_time_bin)
    ).to_dataset()

    # ping_time_bin parsing and conversions
    # Need to convert between pd.Timedelta and np.timedelta64 offsets/frequency strings
    # https://xarray.pydata.org/en/stable/generated/xarray.Dataset.resample.html
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.resample.html
    # https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html
    # https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.resolution_string.html
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    # https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    timedelta_units = {
        'd': {'nptd64': 'D', 'unitstr': 'day'},
        'h': {'nptd64': 'h', 'unitstr': 'hour'},
        't': {'nptd64': 'm', 'unitstr': 'minute'},
        'min': {'nptd64': 'm', 'unitstr': 'minute'},
        's': {'nptd64': 's', 'unitstr': 'second'},
        'l': {'nptd64': 'ms', 'unitstr': 'millisecond'},
        'ms': {'nptd64': 'ms', 'unitstr': 'millisecond'},
        'u': {'nptd64': 'us', 'unitstr': 'microsecond'},
        'us': {'nptd64': 'ms', 'unitstr': 'millisecond'},
        'n': {'nptd64': 'ns', 'unitstr': 'nanosecond'},
        'ns': {'nptd64': 'ms', 'unitstr': 'millisecond'},
    }
    ping_time_bin_td = pd.Timedelta(ping_time_bin)
    # res = resolution (most granular time unit)
    ping_time_bin_resunit = ping_time_bin_td.resolution_string.lower()
    ping_time_bin_resvalue = int(
            ping_time_bin_td / np.timedelta64(1, timedelta_units[ping_time_bin_resunit]['nptd64'])
    )
    ping_time_bin_resunit_label = timedelta_units[ping_time_bin_resunit]['unitstr']

    # Attach attributes
    ds_MVBS = ds_MVBS.rename({'ping_time': 'time'})
    ds_MVBS["time"].attrs = {
        "long_name": "Time",
        "standard_name": "time",
        "axis": "T",
        "comment": "From ping_time",
    }
    ds_MVBS["echo_range"].attrs = {
        "long_name": "Range distance",
        "units": "m"
    }
    ds_MVBS["echo_range"].attrs = {"long_name": "Range distance", "units": "m"}
    MVBS = ds_MVBS["Sv"]
    ds_MVBS["Sv"].attrs = {
        "long_name": "Mean volume backscattering strength (MVBS, mean Sv)",
        "units": "dB re 1 m-1",
        "actual_range": [
            round(float(MVBS.min().values), 2),
            round(float(MVBS.max().values), 2),
        ],
        "cell_methods": (
            f"time: mean (interval: {ping_time_bin_resvalue} {ping_time_bin_resunit_label}"
            " comment: time is the interval start)"
            f" echo_range: mean (interval: {range_meter_bin} meter"
            " comment: echo_range is the interval start)"
        ),
        "binning_mode": "physical units",
        "range_meter_interval": str(range_meter_bin) + "m",
        "ping_time_interval": ping_time_bin,
    }

    return ds_MVBS


def compute_MVBS_index_binning(ds_Sv, range_bin_num=100, ping_num=100):
    """Compute Mean Volume Backscattering Strength (MVBS)
    based on intervals of ``range_bin`` and ping number specified in index number.

    Output of this function differs from that of ``compute_MVBS``, which computes
    bin-averaged Sv according to intervals of range (``echo_range``) and ``ping_time`` specified
    in physical units.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        dataset containing ``Sv`` and ``echo_range`` [m]
    range_bin_num : int
        number of samples to average along the ``range_bin`` dimension, default to 100
    ping_num : int
        number of pings to average, default to 100

    Returns
    -------
    A dataset containing bin-averaged Sv
    """
    ds_Sv["sv"] = 10 ** (ds_Sv["Sv"] / 10)  # average should be done in linear domain
    da = 10 * np.log10(
        ds_Sv["sv"]
        .coarsen(ping_time=ping_num, range_bin=range_bin_num, boundary="pad")
        .mean(skipna=True)
    )

    # Attach coarsened echo_range
    da.name = "Sv"
    ds_out = da.to_dataset()
    ds_out["echo_range"] = (
        ds_Sv["echo_range"]
        .coarsen(  # binned echo_range (use first value in each bin)
            ping_time=ping_num, range_bin=range_bin_num, boundary="pad"
        )
        .min(skipna=True)
    )
    ds_out.coords["range_bin"] = (
        "range_bin",
        np.arange(ds_out["range_bin"].size),
    )  # reset range_bin to start from 0

    # Attach attributes
    ds_out.attrs = {
        "binning_mode": "index",
        "range_bin_num": range_bin_num,
        "ping_num": ping_num,
    }

    return ds_out


def estimate_noise(ds_Sv, ping_num, range_bin_num, noise_max=None):
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
    range_bin_num : int
        number of samples along the ``range_bin`` dimension to obtain noise estimates
    noise_max : float
        the upper limit for background noise expected under the operating conditions

    Returns
    -------
    A DataArray containing noise estimated from the input ``ds_Sv``
    """
    noise_obj = NoiseEst(
        ds_Sv=ds_Sv.copy(), ping_num=ping_num, range_bin_num=range_bin_num
    )
    noise_obj.estimate_noise(noise_max=noise_max)
    return noise_obj.Sv_noise


def remove_noise(ds_Sv, ping_num, range_bin_num, noise_max=None, SNR_threshold=3):
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
    range_bin_num : int
        number of samples along the ``range_bin`` dimension to obtain noise estimates
    noise_max : float
        the upper limit for background noise expected under the operating conditions
    SNR_threshold : float
        acceptable signal-to-noise ratio, default to 3 dB

    Returns
    -------
    The input dataset with additional variables, including
    the corrected Sv (``Sv_corrected``) and the noise estimates (``Sv_noise``)
    """
    noise_obj = NoiseEst(
        ds_Sv=ds_Sv.copy(), ping_num=ping_num, range_bin_num=range_bin_num
    )
    noise_obj.remove_noise(noise_max=noise_max, SNR_threshold=SNR_threshold)
    return noise_obj.ds_Sv


def regrid():
    return 1
