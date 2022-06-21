import numpy as np
import xarray as xr


def swap_dims_channel_frequency(ds: xr.Dataset) -> xr.Dataset:
    """
    Use frequency_nominal in place of channel to be dataset dimension and coorindate.

    This is useful because the nominal transducer frequencies are commonly used to
    refer to data collected from a specific transducer.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset for which the dimension will be swapped

    Returns
    -------
    The input dataset with the dimension swapped

    Note
    ----
    This operation is only possible when there are no duplicated frequencies present in the file.
    """
    # Only possible if no duplicated frequencies
    if np.unique(ds["frequency_nominal"]).size == ds["frequency_nominal"].size:
        return (
            ds
            .set_coords('frequency_nominal')
            .swap_dims({"channel": "frequency_nominal"})
            .reset_coords('channel')
        )
    else:
        raise ValueError(
            "Duplicated transducer nominal frequencies exist in the file. "
            "Operation is not valid."
        )
