from typing import List, Optional

import numpy as np
import xarray as xr

from ..echodata import EchoData
from ..utils.log import _init_logger

logger = _init_logger(__name__)


def _clean_ping_time(ping_time_old, local_win_len=100):
    ping_time_old_diff = np.diff(ping_time_old)
    neg_idx = np.argwhere(
        ping_time_old_diff < np.timedelta64(0, "ns")
    )  # indices with negative diff
    if neg_idx.size != 0:
        ni = neg_idx[0][0]
        local_win = np.arange(-local_win_len, local_win_len)
        local_pt_diff = ping_time_old_diff[ni + local_win]
        local_pt_median = np.median(
            np.delete(local_pt_diff, local_win_len)
        )  # median after removing negative element
        ping_time_new = np.hstack(
            (
                ping_time_old[: ni + 1],
                ping_time_old[ni]
                + np.cumsum(np.hstack((local_pt_median, ping_time_old_diff[(ni + 1) :]))),
            )
        )
        return _clean_ping_time(ping_time_new, local_win_len=local_win_len)
    else:
        return ping_time_old  # no negative diff


def coerce_increasing_time(
    ds: xr.Dataset, time_name: str = "ping_time", local_win_len: int = 100
) -> None:
    """
    Coerce a time coordinate so that it always flows forward. If coercion
    is necessary, the input `ds` will be directly modified.

    Parameters
    ----------
    ds : xr.Dataset
        a dataset for which the time coordinate needs to be corrected
    time_name : str
        name of the time coordinate to be corrected
    local_win_len : int
        half length of the local window within which the median pinging interval
        is used to infer the correct next ping time

    Returns
    -------
    the input dataset but with specified time coordinate coerced to flow forward

    Notes
    -----
    This is to correct for problems sometimes observed in EK60 data
    where a time coordinate (``ping_time`` or ``time1``) would suddenly
    go backward for one ping, but then the rest of the pinging interval
    would remain undisturbed.
    """

    ds[time_name].values[:] = _clean_ping_time(ds[time_name].values, local_win_len=local_win_len)


def exist_reversed_time(ds, time_name):
    """Test for occurrence of time reversal in specified datetime coordinate variable.

    Parameters
    ----------
    ds : xr.Dataset
        a dataset for which the time coordinate will be tested
    time_name : str
        name of the time coordinate to be tested

    Returns
    -------
    `True` if at least one time reversal is found, `False` otherwise.
    """
    return (np.diff(ds[time_name]) < np.timedelta64(0, "ns")).any()


def check_and_correct_reversed_time(
    combined_group: xr.Dataset, time_str: str, ed_group: str
) -> Optional[xr.DataArray]:
    """
    Makes sure that the time coordinate ``time_str`` in
    ``combined_group`` is in the correct order and corrects
    it, if it is not. If coercion is necessary, the input
    `combined_group` will be directly modified.

    Parameters
    ----------
    combined_group : xr.Dataset
        Dataset representing a combined EchoData group
    time_str : str
        Name of time coordinate to be checked and corrected
    ed_group : str
        Name of ``EchoData`` group name

    Returns
    -------
    old_time : xr.DataArray or None
        If correction is necessary, returns the time before
        reversal correction, otherwise returns None

    Warns
    -----
    UserWarning
        If a time reversal is detected
    """

    if time_str in combined_group and exist_reversed_time(combined_group, time_str):
        logger.warning(
            f"{ed_group} {time_str} reversal detected; {time_str} will be corrected"  # noqa
            " (see https://github.com/OSOceanAcoustics/echopype/pull/297)"
        )
        old_time = combined_group[time_str].copy()
        coerce_increasing_time(combined_group, time_name=time_str)
    else:
        old_time = None

    return old_time


def create_old_time_array(group: str, old_time_in: xr.DataArray) -> xr.DataArray:
    """
    Creates an old time array with the appropriate values, name,
    attributes, and encoding.

    Parameters
    ----------
    group: str
        The name of the ``EchoData`` group that contained
        the old time
    old_time_in: xr.DataArray
        The uncorrected old time

    Returns
    -------
    old_time_array: xr.DataArray
        The newly created old time array
    """

    # make a copy, so we don't change the source array
    old_time = old_time_in.copy()

    # get name of old time and dim for Provenance group
    ed_name = group.replace("-", "_").replace("/", "_").lower()
    old_time_name = ed_name + "_old_" + old_time.name

    old_time_name_dim = old_time_name + "_dim"

    # construct old time attributes
    attributes = old_time.attrs
    attributes["comment"] = f"Uncorrected {old_time.name} from the combined group {group}."

    # create old time array
    old_time_array = xr.DataArray(
        data=old_time.values, dims=[old_time_name_dim], attrs=attributes, name=old_time_name
    )

    # set encodings
    old_time_array.encoding = old_time.encoding

    return old_time_array


def orchestrate_reverse_time_check(
    ed_comb: EchoData,
    zarr_store: str,
    possible_time_dims: List[str],
    storage_options: dict,
    consolidated: bool = True,
) -> None:
    """
    Performs a reverse time check of all groups and
    each time dimension within the group. If a reversed
    time is found it will be corrected in ``ed_comb``,
    updated in the zarr store, the old time will be
    added to the ``Provenance`` group in ``ed_comb``,
    the old time will be written to the zarr store,
    and the attribute ``reversed_ping_times`` in the
    ``Provenance`` group will be set to ``1``.

    Parameters
    ----------
    ed_comb: EchoData
        ``EchoData`` object that has been constructed from
        combined ``EchoData`` objects
    zarr_store: str
        The zarr store containing the ``ed_comb`` data
    possible_time_dims: list of str
        All possible time dimensions that can occur within
        ``ed_comb``, which should be checked
    storage_options: dict
        Additional keywords to pass to the filesystem class.
    consolidated : bool
        Flag to consolidate zarr metadata.
        Defaults to ``True``

    Notes
    -----
    If correction is necessary, ``ed_comb`` will be
    directly modified.
    """

    # set Provenance attribute to zero in ed_comb
    ed_comb["Provenance"].attrs["reversed_ping_times"] = 0

    # set Provenance attribute to zero in zarr (Dataset needed for metadata creation)
    only_attrs_ds = xr.Dataset(attrs=ed_comb["Provenance"].attrs)
    only_attrs_ds.to_zarr(
        zarr_store,
        group="Provenance",
        mode="a",
        storage_options=storage_options,
        consolidated=consolidated,
    )

    for group in ed_comb.group_paths:
        if group != "Platform/NMEA":
            # Platform/NMEA is skipped because we found that the times which correspond to
            # other non-GPS messages are often out of order and correcting them is not
            # possible with the current implementation of _clean_ping_time in qc.api due
            # to excessive recursion. There is also no obvious advantage in correcting
            # the order of these timestamps.

            # get all time dimensions of the group
            ed_comb_time_dims = set(ed_comb[group].dims).intersection(possible_time_dims)

            for time in ed_comb_time_dims:
                old_time = check_and_correct_reversed_time(
                    combined_group=ed_comb[group], time_str=time, ed_group=group
                )

                if old_time is not None:
                    old_time_array = create_old_time_array(group, old_time)

                    # put old times in Provenance and modify attribute
                    ed_comb["Provenance"][old_time_array.name] = old_time_array
                    ed_comb["Provenance"].attrs["reversed_ping_times"] = 1

                    # save old time to zarr store
                    old_time_ds = old_time_array.to_dataset()
                    old_time_ds.attrs = ed_comb["Provenance"].attrs
                    old_time_ds.to_zarr(
                        zarr_store,
                        group="Provenance",
                        mode="a",
                        storage_options=storage_options,
                        consolidated=consolidated,
                    )

                    # save corrected time to zarr store
                    ed_comb[group][[time]].to_zarr(
                        zarr_store,
                        group=group,
                        mode="r+",
                        storage_options=storage_options,
                        consolidated=consolidated,
                    )
