import operator as op
import pathlib
from typing import List, Optional, Union

import numpy as np
import xarray as xr

from ..utils.data_proc_lvls import validate_source_Sv

# lookup table with key string operator and value as corresponding Python operator
str2ops = {
    ">": op.gt,
    "<": op.lt,
    "<=": op.le,
    ">=": op.ge,
    "==": op.eq,
}


def _check_freq_diff_non_data_inputs(
    freqAB: Optional[List[float]] = None,
    chanAB: Optional[List[str]] = None,
    operator: str = ">",
    diff: Union[float, int] = None,
) -> None:
    """
    Checks that the non-data related inputs of ``frequency_difference`` (i.e. ``freqAB``,
    ``chanAB``, ``operator``, ``diff``) were correctly provided.

    Parameters
    ----------
    freqAB: list of float, optional
        The pair of nominal frequencies to be used for frequency-differencing, where
        the first element corresponds to ``freqA`` and the second element corresponds
        to ``freqB``
    chanAB: list of float, optional
        The pair of channels that will be used to select the nominal frequencies to be
        used for frequency-differencing, where the first element corresponds to ``freqA``
        and the second element corresponds to ``freqB``
    operator: {">", "<", "<=", ">=", "=="}
        The operator for the frequency-differencing
    diff: float or int
        The threshold of Sv difference between frequencies
    """

    # check that either freqAB or chanAB are provided and they are a list of length 2
    if (freqAB is None) and (chanAB is None):
        raise RuntimeError("Either freqAB or chanAB must be given!")
    elif (freqAB is not None) and (chanAB is not None):
        raise RuntimeError("Only freqAB or chanAB must be given, but not both!")
    elif freqAB is not None:
        if not isinstance(freqAB, list):
            raise TypeError("freqAB must be a list!")
        elif len(set(freqAB)) != 2:
            raise RuntimeError("freqAB must be a list of length 2 with unique elements!")
    else:
        if not isinstance(chanAB, list):
            raise TypeError("chanAB must be a list!")
        elif len(set(chanAB)) != 2:
            raise RuntimeError("chanAB must be a list of length 2 with unique elements!")

    # check that operator is a string and a valid operator
    if not isinstance(operator, str):
        raise TypeError("operator must be a string!")
    else:
        if operator not in [">", "<", "<=", ">=", "=="]:
            raise RuntimeError("Invalid operator!")

    # ensure that diff is a float or an int
    if not isinstance(diff, (float, int)):
        raise TypeError("diff must be a float or int!")


def _check_source_Sv_freq_diff(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    storage_options: Optional[dict],
    freqAB: Optional[List[float]] = None,
    chanAB: Optional[List[str]] = None,
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Ensures that ``source_Sv`` is of the correct type, it exists if it is a
    path, and contains ``channel`` as a coordinate and ``frequency_nominal`` as a
    variable.

    Parameters
    ----------
    source_Sv: xr.Dataset or str or pathlib.Path
        If a Dataset this value contains the Sv data to create a
        mask for, else it specifies the path to a zarr or netcdf file
    storage_options: dict, optional
        Any additional parameters for the storage backend, corresponding to the
        path provided for ``source_Sv``
    freqAB: list of float, optional
        The pair of nominal frequencies to be used for frequency-differencing, where
        the first element corresponds to ``freqA`` and the second element corresponds
        to ``freqB``
    chanAB: list of float, optional
        The pair of channels that will be used to select the nominal frequencies to be
        used for frequency-differencing, where the first element corresponds to ``freqA``
        and the second element corresponds to ``freqB``

    Returns
    -------
    source_Sv: xr.Dataset
        A Dataset containing the Sv data
    """

    source_Sv, file_type = validate_source_Sv(source_Sv, storage_options)

    if isinstance(source_Sv, str):

        # open up Dataset using source_Sv path
        source_Sv = xr.open_dataset(source_Sv, engine=file_type, chunks="auto", **storage_options)

    # check that channel and frequency nominal are in source_Sv
    if "channel" not in source_Sv.coords:
        raise RuntimeError("The Dataset defined by source_Sv must have channel as a coordinate!")
    elif "frequency_nominal" not in source_Sv.variables:
        raise RuntimeError(
            "The Dataset defined by source_Sv must have frequency_nominal as a variable!"
        )

    def get_positions(var_name, input_list):

        # obtain position of input in either frequency_nominal or channel
        inputA_pos = np.argwhere(source_Sv[var_name].values == input_list[0]).flatten()
        inputB_pos = np.argwhere(source_Sv[var_name].values == input_list[1]).flatten()

        return inputA_pos, inputB_pos

    # check that the element of freqAB or chanAB are in frequency_nominal or channel, respectively
    if freqAB is not None:
        inputA_pos, inputB_pos = get_positions(var_name="frequency_nominal", input_list=freqAB)
    else:
        inputA_pos, inputB_pos = get_positions(var_name="channel", input_list=chanAB)

    # TODO: add tests for the below block
    if (len(inputA_pos) != 1) or (len(inputB_pos) != 1):

        if freqAB is not None:
            raise RuntimeError(
                "A provided frequency is either not contained in "
                "frequency_nominal or frequency_nominal has repeated "
                "frequencies, which is not allowed!"
            )
        else:
            raise RuntimeError(
                "A provided channel is either not contained in "
                "the channel coordinate or the coordinate has repeated "
                "channels, which is not allowed!"
            )

    return source_Sv


def frequency_difference(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    storage_options: Optional[dict] = {},
    freqAB: Optional[List[float]] = None,
    chanAB: Optional[List[str]] = None,
    operator: str = ">",
    diff: Union[float, int] = None,
) -> xr.DataArray:
    """
    Create a mask based on the differences of Sv values using a pair of
    frequencies. This method is often referred to as the "frequency-differencing"
    or "dB-differencing" method.

    Parameters
    ----------
    source_Sv: xr.Dataset or str or pathlib.Path
        If a Dataset this value contains the Sv data to create a mask for,
        else it specifies the path to a zarr or netcdf file containing
        a Dataset. This input must correspond to a Dataset that has the
        coordinate ``channel`` and variables ``frequency_nominal`` and ``Sv``.
    storage_options: dict, optional
        Any additional parameters for the storage backend, corresponding to the
        path provided for ``source_Sv``
    freqAB: list of float, optional
        The pair of nominal frequencies to be used for frequency-differencing, where
        the first element corresponds to ``freqA`` and the second element corresponds
        to ``freqB`` (see notes below for more details)
    chanAB: list of float, optional
        The pair of channels that will be used to select the nominal frequencies to be
        used for frequency-differencing, where the first element corresponds to ``freqA``
        and the second element corresponds to ``freqB`` (see notes below for more details)
    operator: {">", "<", "<=", ">=", "=="}
        The operator for the frequency-differencing
    diff: float or int
        The threshold of Sv difference between frequencies

    Returns
    -------
    xr.DataArray
        A DataArray containing the mask for the Sv data. Regions satisfying the thresholding
        criteria are filled with ``True``, else the regions are filled with ``False``.

    Raises
    ------
    RuntimeError
        If neither ``freqAB`` or ``chanAB`` are given
    RuntimeError
        If both ``freqAB`` and ``chanAB`` are given
    TypeError
        If any input is not of the correct type
    RuntimeError
        If either ``freqAB`` or ``chanAB`` are provided and not of length 2
    RuntimeError
        If ``freqAB`` contains values that are repeated or not contained
        in ``frequency_nominal``
    RuntimeError
        If operator is not one of the following: ``">", "<", "<=", ">=", "=="``
    RuntimeError
        If the path provided for ``source_Sv`` is not a valid path
    RuntimeError
        If ``freqAB`` is provided and the Dataset produced by ``source_Sv`` does not
        contain the coordinate ``channel`` and variable ``frequency_nominal``
    RuntimeError
        If ``chanAB`` is provided and the Dataset or DataArray produced by ``source_Sv``
        does not contain the coordinate ``channel``

    Notes
    -----
    TODO: mention it is freqA - freqB operator diff, thus if operator = "<" and diff = "5"
        we would have freqA - freqB < 5


    Either ``freqAB`` or ``chanAB`` must be provided, but both parameters cannot be
    given.

    * return a mask as xarray Dataset (xr.where())

    Examples
    --------

    ep.mask.frequency_difference()


    """

    # check that non-data related inputs were correctly provided
    _check_freq_diff_non_data_inputs(freqAB, chanAB, operator, diff)

    # check the source_Sv input
    source_Sv = _check_source_Sv_freq_diff(source_Sv, storage_options, freqAB, chanAB)

    # determine chanA and chanB
    if freqAB is not None:

        # obtain position of frequency provided in frequency_nominal
        freqA_pos = np.argwhere(source_Sv.frequency_nominal.values == freqAB[0]).flatten()[0]
        freqB_pos = np.argwhere(source_Sv.frequency_nominal.values == freqAB[1]).flatten()[0]

        # get channel corresponding to frequency provided
        chanA = source_Sv.channel.isel(channel=freqA_pos)
        chanB = source_Sv.channel.isel(channel=freqB_pos)

    else:
        # get individual channels
        chanA = chanAB[0]
        chanB = chanAB[1]

    # get the left-hand side of condition
    lhs = source_Sv["Sv"].sel(channel=chanA) - source_Sv["Sv"].sel(channel=chanB)

    # create mask using operator lookup table
    da = xr.where(str2ops[operator](lhs, diff), True, False)

    # assign a name to DataArray
    da.name = "frequency_difference_mask"

    return da
