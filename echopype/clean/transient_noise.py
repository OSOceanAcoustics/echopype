"""
Algorithms for masking transient noise.

    Copyright (c) 2020 Echopy

    This code is licensed under the MIT License.
    See https://opensource.org/license/mit/ for details.
    Original code sourced from:
    https://github.com/open-ocean-sounding/echopy/blob/master/echopy/processing/mask_transient.py

"""
__authors__ = [
    "Alejandro Ariza",  # wrote ryan(), fielding()
    "Mihai Boldeanu",  # adapted the mask transient noise algorithms to echopype
    "Ruxandra Valcu",  # modified ryan and fielding to run off xarray functionality
]

import warnings

import numpy as np
import xarray as xr

# from ..utils.mask_transformation import lin as _lin, log as _log
from ..utils.mask_transformation_xr import (
    dask_nanmean,
    dask_nanmedian,
    lin as _lin,
    line_to_square,
    log as _log,
)

RYAN_DEFAULT_PARAMS = {
    #    "m": 5,
    #    "n": 20,
    "m": 5,
    "n": 5,
    "thr": 20,
    "excludeabove": 250,
    "operation": "mean",
    "dask_chunking": {"ping_time": 100},
}
FIELDING_DEFAULT_PARAMS = {
    "r0": 200,
    "r1": 1000,
    "n": 20,
    "thr": [2, 0],
    "roff": 250,
    "jumps": 5,
    "maxts": -35,
    "start": 0,
    "dask_chunking": {"ping_time": 100},
}


def _ryan(source_Sv: xr.DataArray, desired_channel: str, parameters: dict = RYAN_DEFAULT_PARAMS):
    """
    Mask transient noise as in:

        Ryan et al. (2015) ‘Reducing bias due to noise and attenuation in
        open-ocean echo integration data’, ICES Journal of Marine Science,
        72: 2482–2493.

    This mask is based on the assumption that Sv values which exceed the median
    value in a surrounding region of m metres by n pings must be due to
    transient noise. Sv values are removed if exceed a threshold. Masking is
    excluded above 250 m by default to avoid the removal of aggregated biota.

    Args:
    Args:
        source_Sv (xr.DataArray): Sv array
        selected_channel (str): name of the channel to process
        parameters(dict): dict of parameters, containing:
            m (int): height of surrounding region (m)
            n (int): width of surrounding region (pings)
            thr (int): user-defined threshold for comparisons (dB)
            excludeabove (int): range above which masking is excluded (m)
            operation (str): type of average operation:
                'mean'
                'percentileXX'
                'median'
                'mode'#not in numpy
            dask_chunking(dict): specify what dask chunking to use

    Returns:
        xarray.DataArray: xr.DataArray with mask indicating the presence of transient noise.
    """
    parameter_names = ("m", "n", "thr", "excludeabove", "operation", "dask_chunking")
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be m, n, thr, excludeabove, operation, dask_chunking, are"
            + str(parameters.keys())
        )
    m = parameters["m"]
    n = parameters["n"]
    thr = parameters["thr"]
    excludeabove = parameters["excludeabove"]
    operation = parameters["operation"]
    dask_chunking = parameters["dask_chunking"]

    om = {"mean": dask_nanmean, "median": dask_nanmedian}

    if operation not in om.keys():
        raise ValueError("Wrong operation type {}".format(operation))

    channel_Sv = source_Sv.sel(channel=desired_channel)
    Sv = channel_Sv["Sv"]
    range = channel_Sv["echo_range"][0]

    # calculate offsets
    ping_offset = n * 2 + 1
    range_offset = abs(range - m).argmin(dim="range_sample").values
    range_offset = int(range_offset) * 2 + 1

    r0 = abs(range - excludeabove).argmin(dim="range_sample").values

    Sv = Sv.chunk(dask_chunking)
    # mask = Sv > 0  # just to create it at the right size

    # create averaged/median value block
    block = (
        _lin(Sv)
        .rolling(ping_time=ping_offset, range_sample=range_offset, center=True)
        .reduce(om[operation])
        .compute()
    )
    block = _log(block)

    mask = Sv - block > thr

    mask = mask.where(~(mask["range_sample"] < r0), False)
    mask = ~mask
    mask = mask.drop("channel")

    return mask


def _fielding(
    source_Sv: xr.DataArray, desired_channel: str, parameters: dict = FIELDING_DEFAULT_PARAMS
):
    """
    Mask transient noise with method proposed by Fielding et al (unpub.).

    A comparison is made ping by ping with respect to a block in a reference
    layer set at far range, where transient noise mostly occurs. If the ping
    median is greater than the block median by a user-defined threshold, the
    ping will be masked all the way up, until transient noise disappears, or
    until it gets the minimum range allowed by the user.

       transient                 transient             ping
         noise                     noise             evaluated
           |                         |                  |
    ______ | _______________________ | ____________.....V.....____________
          |||  far range interval   |||            .  block  .            |
    _____|||||_____________________|||||___________...........____________|

    When transient noise is detected, comparisons start to be made in the same
    ping but moving vertically every x meters (jumps). Pings with transient
    noise will be masked up to where the ping is similar to the block according
    with a secondary threshold or until it gets the exclusion range depth.

    Args:
        source_Sv (xr.DataArray): Sv array
        selected_channel (str): name of the channel to process
        parameters(dict): dict of parameters, containing:
            r0    (int  ): range below which transient noise is evaluated (m).
            r1    (int  ): range above which transient noise is evaluated (m).
            n     (int  ): n of preceding & subsequent pings defining the block.
            thr   (int  ): user-defined threshold for side-comparisons (dB).
            maxts (int  ): max transient noise permitted, prevents to interpret
                           seabed as transient noise (dB).
            jumps (int  ): height of vertical steps (m).
            dask_chunking(dict): dask array chunking parameters

    Returns:
        xarray.DataArray: xr.DataArray with mask indicating the presence of transient noise.
    """
    parameter_names = ("r0", "r1", "n", "thr", "maxts", "jumps", "dask_chunking")
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be r0, r1, n, thr, maxts, jumps, dask_chunking are"
            + str(parameters.keys())
        )
    r0 = parameters["r0"]
    r1 = parameters["r1"]
    n = parameters["n"]
    thr = parameters["thr"]
    maxts = parameters["maxts"]
    jumps = parameters["jumps"]
    dask_chunking = parameters["dask_chunking"]

    channel_Sv = source_Sv.sel(channel=desired_channel)
    Sv = channel_Sv["Sv"]
    r = channel_Sv["echo_range"][0]

    # raise errors if wrong arguments
    if r0 > r1:
        raise Exception("Minimum range has to be shorter than maximum range")

    # return empty mask if searching range is outside the echosounder range
    if (r0 > r[-1]) or (r1 < r[0]):
        # Raise a warning to inform the user
        warnings.warn(
            "The searching range is outside the echosounder range. "
            "A default mask with all True values is returned, "
            "which won't mask any data points in the dataset."
        )
        return xr.DataArray(
            np.ones_like(Sv, dtype=bool),
            dims=("ping_time", "range_sample"),
            coords={"ping_time": source_Sv.ping_time, "range_sample": source_Sv.range_sample},
        )

    # get upper and lower range indexes
    up = abs(r - r0).argmin(dim="range_sample").values
    lw = abs(r - r1).argmin(dim="range_sample").values

    # get minimum range index admitted for processing
    # rmin = abs(r - roff).argmin(dim="range_sample").values
    # get scaling factor index
    sf = abs(r - jumps).argmin(dim="range_sample").values

    range_mask = (Sv.range_sample >= up) & (Sv.range_sample <= lw)
    Sv_range = Sv.where(range_mask, np.nan)

    # get columns in which no processing can be done - question, do we want to mask them out?
    nan_mask = Sv_range.isnull()
    nan_mask = nan_mask.reduce(np.any, dim="range_sample")
    nan_mask[0:n] = False
    nan_mask[-n:] = False

    ping_median = _log(_lin(Sv_range).median(dim="range_sample", skipna=True))
    ping_75q = _log(_lin(Sv_range).reduce(np.nanpercentile, q=75, dim="range_sample"))

    block = Sv_range[:, up:lw]
    block_list = [block.shift({"ping_time": i}) for i in range(-n, n)]
    concat_block = xr.concat(block_list, dim="range_sample")
    block_median = _log(_lin(concat_block).median(dim="range_sample", skipna=True))

    # identify columns in which noise can be found
    noise_column = (ping_75q < maxts) & ((ping_median - block_median) < thr[0])

    noise_column_mask = xr.DataArray(
        data=line_to_square(noise_column, Sv, "range_sample").transpose(),
        dims=Sv.dims,
        coords=Sv.coords,
    )

    # Chunk the data if not already chunked
    Sv_range_chunked = Sv_range.chunk(dask_chunking)

    # Apply rolling operation and reduce using the custom Dask-aware function
    ping_median = _lin(Sv_range_chunked).rolling(range_sample=sf).reduce(dask_nanmedian)
    block_median = (
        _lin(Sv_range_chunked).rolling(range_sample=sf, ping_time=2 * n + 1).reduce(dask_nanmedian)
    )

    # Compute the results
    ping_median = _log(ping_median.compute())
    block_median = _log(block_median.compute())

    height_mask = ping_median - block_median < thr[1]

    height_noise_mask = height_mask | noise_column_mask

    flipped_mask = height_noise_mask.isel(range_sample=slice(None, None, -1))
    flipped_mask["range_sample"] = height_mask["range_sample"]
    neg_mask = ~height_noise_mask

    # propagate break upward
    flipped_mask = neg_mask.isel(range_sample=slice(None, None, -1))
    flipped_mask["range_sample"] = height_mask["range_sample"]
    flipped_mask = ~flipped_mask
    ft = len(flipped_mask.range_sample) - flipped_mask.argmax(dim="range_sample")

    first_true_indices = xr.DataArray(
        line_to_square(ft, flipped_mask, dim="range_sample").transpose(),
        dims=("ping_time", "range_sample"),
        coords={"ping_time": channel_Sv.ping_time, "range_sample": channel_Sv.range_sample},
    )

    indices = xr.DataArray(
        line_to_square(height_noise_mask["range_sample"], height_noise_mask, dim="ping_time"),
        dims=("ping_time", "range_sample"),
        coords={"ping_time": channel_Sv.ping_time, "range_sample": channel_Sv.range_sample},
    )

    noise_spike_mask = height_noise_mask.where(indices > first_true_indices, True)

    mask = noise_spike_mask
    mask = mask.drop("channel")

    # uncomment if we want to mask out the columns where no processing could be done,
    # mask = nan_full_mask & noise_spike_mask
    return mask
