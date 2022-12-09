import operator as op
import pathlib
from typing import List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from echopype.utils.io import check_file_existence, get_file_format, validate_output_path

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
        elif len(freqAB) != 2:
            raise RuntimeError("freqAB must be a list of length 2!")
    else:
        if not isinstance(chanAB, list):
            raise TypeError("chanAB must be a list!")
        elif len(chanAB) != 2:
            raise RuntimeError("chanAB must be a list of length 2!")

    # check that operator is a string and a valid operator
    if not isinstance(operator, str):
        raise TypeError("operator must be a string!")
    else:
        if operator not in [">", "<", "<=", ">=", "=="]:
            raise RuntimeError("Invalid operator!")

    # ensure that diff is a float or an int
    if not isinstance(diff, (float, int)):
        raise TypeError("diff must be a float or int!")


# TODO: validate_source_Sv is likely to used in other places,
#  should we move it somewhere else?
def validate_source_Sv(
    source_Sv: Union[xr.Dataset, xr.DataArray, str, pathlib.Path], storage_options: Optional[dict]
) -> Tuple[Union[xr.Dataset, str, xr.DataArray], Optional[str]]:
    """
    This function ensures that ``source_Sv`` is of the correct
    type and validates the path of ``source_Sv``, if it is provided.

    Parameters
    ----------
    source_Sv: xr.Dataset or xr.DataArray or str or pathlib.Path
        If a Dataset or DataArray this value contains the Sv data,
        else it specifies the path to a zarr or netcdf file
    storage_options: dict, optional
        Any additional parameters for the storage backend, corresponding to the
        path provided for ``source_Sv``

    Returns
    -------
    source_Sv: xr.Dataset or str
        A Dataset or DataArray that contains the Sv data (which will be the same as
        the input) or a validated path to a zarr or netcdf file
    file_type: {"netcdf4", "zarr"}, optional
        If ``source_Sv`` is a path then corresponds to the file type of input, else
        is ``None``
    """

    # initialize file_type
    file_type = None

    # make sure that storage_options is of the appropriate type
    if not isinstance(storage_options, dict):
        raise TypeError("storage_options must be a dict!")

    # check that source_Sv is of the correct type, if it is a path validate
    # the path and open the dataset using xarray
    if not isinstance(source_Sv, (xr.Dataset, xr.DataArray, str, pathlib.Path)):
        raise TypeError("source_Sv must be a Dataset or DataArray or str or pathlib.Path!")
    elif isinstance(source_Sv, (str, pathlib.Path)):

        # determine if we obtained a zarr or netcdf file
        file_type = get_file_format(source_Sv)

        # validate source_Sv if it is a path
        source_Sv = validate_output_path(
            source_file="blank",  # will be unused since source_Sv cannot be none
            engine=file_type,
            output_storage_options=storage_options,
            save_path=source_Sv,
        )

        # check that the path exists
        check_file_existence(file_path=source_Sv, storage_options=storage_options)

    return source_Sv, file_type


def _check_source_Sv_freq_diff(
    source_Sv: Union[xr.Dataset, xr.DataArray, str, pathlib.Path],
    storage_options: Optional[dict],
    freqAB: Optional[List[float]] = None,
    chanAB: Optional[List[str]] = None,
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Ensures that ``source_Sv`` is of the correct type, it exists if it is a
    path, contains ``channel`` as a coordinate and ``frequency_nominal`` as a
    variable if ``freqAB`` is not ``None``, and contains ``channel`` as a
    coordinate if ``chanAB`` is not ``None``.

    Parameters
    ----------
    source_Sv: xr.Dataset or xr.DataArray or str or pathlib.Path
        If a Dataset or DataArray this value contains the Sv data to create a
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
    source_Sv: xr.Dataset or xr.DataArray
        A Dataset or DataArray containing the Sv data
    """

    source_Sv, file_type = validate_source_Sv(source_Sv, storage_options)

    if isinstance(source_Sv, str):

        # TODO: check that this works with zarr and netcdf (create test)
        # open up Dataset using source_Sv path
        source_Sv = xr.open_dataset(source_Sv, engine=file_type, chunks="auto", **storage_options)

    # TODO: create a mock test for chanAB/freqAB code block checks below
    # check that channel and frequency nominal are in source_Sv, if freqAB is used
    if freqAB is not None:
        if "channel" not in source_Sv.coords:
            raise RuntimeError(
                "The Dataset defined by source_Sv must have channel as a coordinate!"
            )
        elif not isinstance(source_Sv, xr.Dataset):
            raise RuntimeError("source_Sv must be a Dataset, if freqAB is provided!")
        elif "frequency_nominal" not in source_Sv.variables:
            raise RuntimeError(
                "The Dataset defined by source_Sv must have frequency_nominal as a variable!"
            )

        # obtain position of frequency provided in frequency_nominal
        freqA_pos = np.argwhere(source_Sv.frequency_nominal.values == freqAB[0]).flatten()
        freqB_pos = np.argwhere(source_Sv.frequency_nominal.values == freqAB[1]).flatten()

        # TODO: create test for this!
        # check that freqA and freqB are contained in frequency_nominal and unique
        if (len(freqA_pos) != 1) or (len(freqB_pos) != 1):
            raise RuntimeError(
                "A provided frequency is either not contained in "
                "frequency_nominal or frequency_nominal has repeated "
                "frequencies, which is not allowed!"
            )

    # check that channel is in source_SV, if chanAB is used
    if (chanAB is not None) and ("channel" not in source_Sv.coords):
        raise RuntimeError("The Dataset defined by source_Sv must have channel as a coordinate!")

    return source_Sv


def frequency_difference(
    source_Sv: Union[xr.Dataset, xr.DataArray, str, pathlib.Path],
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
    source_Sv: xr.Dataset xr.DataArray or str or pathlib.Path
        If a Dataset or DataArray this value contains the Sv data to create a
        mask for, else it specifies the path to a zarr or netcdf file containing
        a Dataset. If ``source_Sv`` corresponds to a Dataset, it must have the
        coordinate ``channel`` and variable ``frequency_nominal``, if it
        corresponds to a DataArray it must have the coordinate ``channel``.
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
    if isinstance(source_Sv, xr.Dataset):
        lhs = source_Sv["Sv"].sel(channel=chanA) - source_Sv["Sv"].sel(channel=chanB)
    else:
        lhs = source_Sv.sel(channel=chanA) - source_Sv.sel(channel=chanB)

    # create mask using operator lookup table
    return xr.where(str2ops[operator](lhs, diff), True, False)
