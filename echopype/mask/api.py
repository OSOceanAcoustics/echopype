import operator
import pathlib
from typing import List, Optional, Union

import xarray as xr

from echopype.utils.io import check_file_existence, get_file_format, validate_output_path

# lookup table with key string operator and value as corresponding Python operator
str2ops = {
    ">": operator.gt,
    "<": operator.lt,
    "<=": operator.le,
    ">=": operator.ge,
    "==": operator.eq,
}


def frequency_difference(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    storage_options: Optional[dict] = {},
    freqAB: Optional[List[float]] = None,
    chanAB: Optional[List[str]] = None,
    operator: str = ">",
    diff: Union[float, int] = None,
):
    """
    Create a mask based on the differences of Sv values using a pair of
    frequencies. This method is often referred to as the "frequency-differencing"
    or "dB-differencing" method.

    Parameters
    ----------
    source_Sv: xr.Dataset or str or pathlib.Path
        If a Dataset this value contains the Sv data to create a
        mask for, else it specifies the path to a zarr or netcdf file. This must
        point to a Dataset that has the coordinate ``channel`` and variable `
        `frequency_nominal``.
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
    xr.Dataset
        A Dataset containing the mask. Regions satisfying the thresholding criteria
        are filled with ``True``, else the regions are filled with ``False``.

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
        If operator is not one of the following: ``">", "<", "<=", ">=", "=="``
    RuntimeError
        If the path provided for ``source_Sv`` is not a valid path
    RuntimeError
        If ``freqAB`` is provided and the Dataset produced by ``source_Sv`` does not
        contain the coordinate ``channel`` and variable ``frequency_nominal``
    RuntimeError
        If ``chanAB`` is provided and the Dataset produced by ``source_Sv`` does not
        contain the coordinate ``channel``


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

    # make sure that storage_options is of the appropriate type
    if not isinstance(storage_options, dict):
        raise TypeError("storage_options must be a dict!")

    # check that source_Sv is of the correct type, if it is a path validate
    # the path and open the dataset using xarray
    if not isinstance(source_Sv, (xr.Dataset, str, pathlib.Path)):
        raise TypeError("source_Sv must be a Dataset or str or pathlib.Path!")
    elif isinstance(source_Sv, (str, pathlib.Path)):

        # determine if we obtained a zarr or netcdf file
        file_type = get_file_format(source_Sv)

        # validate source_Sv if it is a path
        validated_path = validate_output_path(
            source_file="blank",  # will be unused since sourve_Sv cannot be none
            engine=file_type,
            output_storage_options=storage_options,
            save_path=source_Sv,
        )

        # check that the path exists
        check_file_existence(file_path=validated_path, storage_options=storage_options)

        # TODO: check that this works with zarr and netcdf (create test)
        # open up Dataset using source_Sv path
        source_Sv = xr.open_dataset(
            validated_path, engine=file_type, chunks="auto", **storage_options
        )

    # TODO: create a mock test for chanAB/freqAB code block checks below

    # check that channel and frequency nominal are in source_Sv, if freqAB is used
    if freqAB is not None:
        if "channel" not in source_Sv.coords:
            raise RuntimeError(
                "The Dataset defined by source_Sv must have channel as a coordinate!"
            )
        elif "frequency_nominal" not in source_Sv.variables:
            raise RuntimeError(
                "The Dataset defined by source_Sv must have frequency_nominal as a variable!"
            )

    # check that channel is in source_SV, if chanAB is used
    if (chanAB is not None) and ("channel" not in source_Sv.coords):
        raise RuntimeError("The Dataset defined by source_Sv must have channel as a coordinate!")

    # TODO: do we want to allow for source_Sv to be a xr.DataArray?

    # TODO: use lookup table
    # print(str2ops["<"](1, 1))
