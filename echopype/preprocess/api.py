"""
Functions for enhancing the spatial and temporal coherence of data.
"""

import numpy as np
import pandas as pd

from ..utils.prov import echopype_prov_attrs
from .noise_est import NoiseEst


def _check_range_uniqueness(ds):
    """Check if range (``echo_range``) changes across ping in a given frequency channel."""
    return (
        ds["echo_range"].isel(ping_time=0).dropna(dim="range_sample")
        == ds["echo_range"].dropna(dim="range_sample")
    ).all()


def _set_MVBS_attrs(ds):
    """Attach common attributes

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

    if not ds_Sv.groupby("channel").apply(_check_range_uniqueness).all():
        raise ValueError(
            "echo_range variable changes across pings in at least one of the frequency channels."
        )

    # TODO: right now this computation is brittle as it takes echo_range
    #  from only the lowest frequency to make it the range for all channels.
    #  This should be implemented different to allow non-uniform echo_range.

    # get indices of sorted frequency_nominal values. This is necessary
    # because the frequency_nominal values are not always in ascending order.
    sorted_freq_ind = np.argsort(ds_Sv.frequency_nominal)

    def _freq_MVBS(ds, rint, pbin):
        sv = 10 ** (ds["Sv"] / 10)  # average should be done in linear domain
        sv.coords["range_meter"] = (
            ["range_sample"],
            ds_Sv["echo_range"].isel(channel=sorted_freq_ind[0], ping_time=0).data,
        )
        sv = sv.swap_dims({"range_sample": "range_meter"})
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
    range_interval = np.arange(0, ds_Sv["echo_range"].max() + range_meter_bin, range_meter_bin)
    ds_MVBS = (
        ds_Sv.groupby("channel")
        .apply(_freq_MVBS, args=(range_interval, ping_time_bin))
        .to_dataset()
    )

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
    """Compute Mean Volume Backscattering Strength (MVBS)
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
