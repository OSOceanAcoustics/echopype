import warnings

import numpy as np
import xarray as xr

from ..utils.mask_transformation_xr import (
    lin as _lin,
    line_to_square,
    log as _log,
    rolling_median_block,
)

# import dask.array as da


DEFAULT_RYAN_PARAMS = {
    "r0": 180,
    "r1": 280,
    "n": 30,
    "thr": -6,
    "start": 0,
    "dask_chunking": {"ping_time": 100, "range_sample": 100},
}
DEFAULT_ARIZA_PARAMS = {"offset": 20, "thr": (-40, -35), "m": 20, "n": 50}


def _ryan(source_Sv: xr.DataArray, desired_channel: str, parameters=DEFAULT_RYAN_PARAMS):
    """
    Locate attenuated signal and create a mask following the attenuated signal
    filter as in:

        Ryan et al. (2015) ‘Reducing bias due to noise and attenuation in
        open-ocean echo integration data’, ICES Journal of Marine Science,
        72: 2482–2493.

    Scattering Layers (SLs) are continuous high signal-to-noise regions with
    low inter-ping variability. But attenuated pings create gaps within SLs.

       attenuation                attenuation       ping evaluated
    ______ V _______________________ V ____________.....V.....____________
          | |   scattering layer    | |            .  block  .            |
    ______| |_______________________| |____________...........____________|

    The filter takes advantage of differences with preceding and subsequent
    pings to detect and mask attenuation. A comparison is made ping by ping
    with respect to a block of the reference layer. The entire ping is masked
    if the ping median is less than the block median by a user-defined
    threshold value.

    Args:
        source_Sv (xr.DataArray): Sv array
        selected_channel (str): name of the channel to process
        parameters(dict): dict of parameters, containing:
            r0 (int): upper limit of SL (m).
            r1 (int): lower limit of SL (m).
            n (int): number of preceding & subsequent pings defining the block.
            thr (int): user-defined threshold value (dB).
            start (int): ping index to start processing.

    Returns:
        xr.DataArray: boolean array with AS mask, with ping_time and range_sample dims
    """
    parameter_names = ("r0", "r1", "n", "thr", "start", "dask_chunking")
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be r0, r1, n, thr, start, dask_chunking are"
            + str(parameters.keys())
        )
    r0 = parameters["r0"]
    r1 = parameters["r1"]
    n = parameters["n"]
    thr = parameters["thr"]
    dask_chunking = parameters["dask_chunking"]
    # start = parameters["start"]

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

    # find indexes for upper and lower SL limits
    up = abs(r - r0).argmin(dim="range_sample").item()
    lw = abs(r - r1).argmin(dim="range_sample").item()

    layer_mask = (Sv["range_sample"] >= up) & (Sv["range_sample"] <= lw)
    layer_Sv = Sv.where(layer_mask)

    layer_Sv_chunked = layer_Sv.chunk(dask_chunking)

    block_median = rolling_median_block(layer_Sv_chunked.data, window_half_size=n, axis=0)
    ping_median = layer_Sv_chunked.median(dim="range_sample", skipna=True)

    # Creating the mask based on the threshold
    mask_condition = (ping_median - block_median) > thr
    mask = mask_condition.reindex_like(layer_Sv, method="nearest").fillna(True)

    ret_mask = xr.DataArray(
        data=line_to_square(mask, Sv, "range_sample").transpose(),
        dims=("ping_time", "range_sample"),
        coords={"ping_time": source_Sv.ping_time, "range_sample": source_Sv.range_sample},
    )
    return ret_mask


def _ariza(source_Sv, desired_channel, parameters=DEFAULT_ARIZA_PARAMS):
    """
    Mask attenuated pings by looking at seabed breaches.

    Ariza et al. (2022) 'Acoustic seascape partitioning through functional data analysis',
    Journal of Biogeography, 00, 1– 15. https://doi.org/10.1111/jbi.14534
    Args:
        source_Sv (xr.DataArray): Sv array
        selected_channel (str): name of the channel to process
        parameters(dict): dict of parameters, containing:
            thr (int):
            seabed_mask: (xr.DataArray) - externally created seabed mask

    Returns:
        xr.DataArray: boolean array with AS mask, with ping_time and range_sample dims
    """
    parameter_names = ("thr", "seabed")
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be thr, m, n, offset, seabed, are" + str(parameters.keys())
        )
    # m = parameters["m"]
    # n = parameters["n"]
    thr = parameters["thr"]
    # offset = parameters["offset"]
    seabed = parameters["seabed"]

    channel_Sv = source_Sv.sel(channel=desired_channel)
    Sv = channel_Sv["Sv"]

    Sv_sb = Sv.copy(deep=True).where(seabed, np.isnan)
    seabed_percentile = _log(_lin(Sv_sb).reduce(dim="range_sample", func=np.nanpercentile, q=95))
    mask = line_to_square(seabed_percentile < thr[0], Sv, dim="range_sample").transpose()
    mask = mask.drop("channel")
    return_mask = xr.DataArray(
        mask,
        dims=("ping_time", "range_sample"),
        coords={"ping_time": source_Sv.ping_time, "range_sample": source_Sv.range_sample},
    )
    return return_mask
