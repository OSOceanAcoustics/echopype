import operator as op
import pathlib
from typing import List, Optional, Union

import numpy as np
import xarray as xr

from ..utils.io import validate_source_ds_da

# lookup table with key string operator and value as corresponding Python operator
str2ops = {
    ">": op.gt,
    "<": op.lt,
    "<=": op.le,
    ">=": op.ge,
    "==": op.eq,
}


def validate_and_collect_mask_input(
    mask: Union[
        Union[xr.DataArray, str, pathlib.Path], List[Union[xr.DataArray, str, pathlib.Path]]
    ],
    storage_options_mask: Union[dict, List[dict]],
) -> Union[xr.DataArray, List[xr.DataArray]]:
    """
    Validate that the input ``mask`` and associated ``storage_options_mask`` are correctly
    provided to ``apply_mask``. Additionally, form the mask input that should be used
    in the core routine of ``apply_mask``.

    Parameters
    ----------
    mask: xr.DataArray, str, pathlib.Path, or a list of these datatypes
        The mask(s) to be applied. Can be a single input or list that corresponds to a
        DataArray or a path. If a path is provided this should point to a zarr or netcdf
        file with only one data variable in it.
    storage_options_mask: dict or list of dict, default={}
        Any additional parameters for the storage backend, corresponding to the
        path provided for ``mask``. If ``mask`` is a list, then this input should either
        be a list of dictionaries or a single dictionary with storage options that
        correspond to all elements in ``mask`` that are paths.

    Returns
    -------
    xr.DataArray or list of xr.DataArray
        If the ``mask`` input is a single value, then the corresponding DataArray will be
        returned, else a list of DataArrays corresponding to the input masks will be returned

    Raises
    ------
    ValueError
        If ``mask`` is a single-element and ``storage_options_mask`` is not a single dict
    TypeError
        If ``storage_options_mask`` is not a list of dict or a dict
    """

    if isinstance(mask, list):

        # if storage_options_mask is not a list create a list of
        # length len(mask) with elements storage_options_mask
        if not isinstance(storage_options_mask, list):

            if not isinstance(storage_options_mask, dict):
                raise TypeError("storage_options_mask must be a list of dict or a dict!")

            storage_options_mask = [storage_options_mask] * len(mask)
        else:
            # ensure all element of storage_options_mask are a dict
            if not all([isinstance(elem, dict) for elem in storage_options_mask]):
                raise TypeError("storage_options_mask must be a list of dict or a dict!")

        for mask_ind in range(len(mask)):

            # validate the mask type or path (if it is provided)
            mask_val, file_type = validate_source_ds_da(
                mask[mask_ind], storage_options_mask[mask_ind]
            )

            # replace mask element path with its corresponding DataArray
            if isinstance(mask_val, str):
                # open up DataArray using mask path
                mask[mask_ind] = xr.open_dataarray(
                    mask_val, engine=file_type, chunks={}, **storage_options_mask[mask_ind]
                )

    else:

        if not isinstance(storage_options_mask, dict):
            raise ValueError(
                "The provided input storage_options_mask should be a single "
                "dict because mask is a single value!"
            )

        # validate the mask type or path (if it is provided)
        mask, file_type = validate_source_ds_da(mask, storage_options_mask)

        if isinstance(mask, str):
            # open up DataArray using mask path
            mask = xr.open_dataarray(mask, engine=file_type, chunks={}, **storage_options_mask)

    return mask


def _check_var_name_fill_value(
    source_ds: xr.Dataset, var_name: str, fill_value: Union[int, float, np.ndarray, xr.DataArray]
) -> None:
    """
    Ensures that the inputs ``var_name`` and ``fill_value`` for the function
    ``apply_mask`` were appropriately provided.

    Parameters
    ----------
    source_ds: xr.Dataset
        A Dataset that contains the variable ``var_name``
    var_name: str
        The variable name in ``source_ds`` that the mask should be applied to
    fill_value: int or float or np.ndarray or xr.DataArray
        Specifies the value(s) at false indices

    Raises
    ------
    TypeError
        If ``var_name`` or ``fill_value`` are not an accepted type
    ValueError
        If the Dataset ``source_ds`` does not contain ``var_name``
    ValueError
        If ``fill_value`` is an array and not the same shape as ``var_name``
    """

    # check the type of var_name
    if not isinstance(var_name, str):
        raise TypeError("The input var_name must be a string!")

    # ensure var_name is in source_ds
    if var_name not in source_ds.variables:
        raise ValueError("The Dataset source_ds does not contain the variable var_name!")

    # check the type of fill_value
    if not isinstance(fill_value, (int, float, np.ndarray, xr.DataArray)):
        raise TypeError(
            "The input fill_value must be of type int or " "float or np.ndarray or xr.DataArray!"
        )

    # make sure that fill_values is the same shape as var_name, if it is an array
    if isinstance(fill_value, (np.ndarray, xr.DataArray)) and (
        fill_value.shape != source_ds[var_name].shape
    ):
        raise ValueError("If fill_value is an array is must be of the same shape as var_name!")


def apply_mask(
    source_ds: Union[xr.Dataset, str, pathlib.Path],
    mask: Union[
        Union[xr.DataArray, str, pathlib.Path], List[Union[xr.DataArray, str, pathlib.Path]]
    ],
    var_name: str = "Sv",
    fill_value: Union[int, float, np.ndarray, xr.DataArray] = np.nan,
    storage_options_ds: dict = {},
    storage_options_mask: Union[dict, List[dict]] = {},
) -> xr.Dataset:
    """
    Applies the provided mask(s) to the variable ``var_name``
    in the provided Dataset ``source_ds``.

    Parameters
    ----------
    source_ds: xr.Dataset, str, or pathlib.Path
        Points to a Dataset that contains the variable the mask should be applied to
    mask: xr.DataArray, str, pathlib.Path, or a list of these datatypes
        The mask(s) to be applied. Can be a single input or list that corresponds to
        a DataArray or a path. If a path is provided this should point to a zarr or
        netcdf file with only one data variable in it.
    var_name: str, default="Sv"
        The variable name in ``source_ds`` that the mask should be applied to
    fill_value: int, float, np.ndarray, or xr.DataArray, default=np.nan
        Value(s) at masked indices
    storage_options_ds: dict, default={}
        Any additional parameters for the storage backend, corresponding to the
        path provided for ``source_ds``
    storage_options_mask: dict or list of dict, default={}
        Any additional parameters for the storage backend, corresponding to the
        path provided for ``mask``. If ``mask`` is a list, then this input should either
        be a list of dictionaries or a single dictionary with storage options that
        correspond to all elements in ``mask`` that are paths.

    Returns
    -------
    xr.Dataset
        A Dataset with the same format of ``source_ds`` with the mask(s) applied to ``var_name``

    Notes
    -----
    If the input ``mask`` is a list, then a logical AND will be used to produce the final
    mask that will be applied to ``var_name``.
    """

    # validate the source_ds type or path (if it is provided)
    source_ds, file_type = validate_source_ds_da(source_ds, storage_options_ds)

    if isinstance(source_ds, str):
        # open up Dataset using source_ds path
        source_ds = xr.open_dataset(source_ds, engine=file_type, chunks={}, **storage_options_ds)

    # validate and form the mask input to be used downstream
    mask = validate_and_collect_mask_input(mask, storage_options_mask)

    # ensure that var_name and fill_value were correctly provided
    _check_var_name_fill_value(source_ds, var_name, fill_value)

    # select data only, if fill_value is a DataArray (necessary since
    # xr.where(keep_attrs=True) is not functioning correctly)
    if isinstance(fill_value, xr.DataArray):
        fill_value = fill_value.data

    # obtain final mask to be applied to var_name
    if isinstance(mask, list):
        # perform a logical AND element-wise operation across the masks
        final_mask = np.logical_and.reduce(mask)

        # xr.where has issues with attrs when final_mask is an array, so we make it a DataArray
        final_mask = xr.DataArray(final_mask, coords=mask[0].coords)
    else:
        final_mask = mask

    # sanity check to make sure final_mask is the same shape as source_ds[var_name]
    if final_mask.shape != source_ds[var_name].shape:
        raise ValueError("Final constructed mask is not the same shape as source_ds[var_name]!")

    # apply the mask to var_name
    var_name_masked = xr.where(final_mask, x=source_ds[var_name], y=fill_value, keep_attrs=True)

    # obtain a shallow copy of source_ds
    output_ds = source_ds.copy(deep=False)

    # replace var_name with var_name_masked
    output_ds[var_name] = var_name_masked

    # TODO: add provenance or attributes specifying that a mask was applied here!

    return output_ds


def _check_freq_diff_non_data_inputs(
    freqAB: Optional[List[float]] = None,
    chanAB: Optional[List[str]] = None,
    operator: str = ">",
    diff: Union[float, int] = None,
) -> None:
    """
    Checks that the non-data related inputs of ``frequency_differencing`` (i.e. ``freqAB``,
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
        raise ValueError("Either freqAB or chanAB must be given!")
    elif (freqAB is not None) and (chanAB is not None):
        raise ValueError("Only freqAB or chanAB must be given, but not both!")
    elif freqAB is not None:
        if not isinstance(freqAB, list):
            raise TypeError("freqAB must be a list!")
        elif len(set(freqAB)) != 2:
            raise ValueError("freqAB must be a list of length 2 with unique elements!")
    else:
        if not isinstance(chanAB, list):
            raise TypeError("chanAB must be a list!")
        elif len(set(chanAB)) != 2:
            raise ValueError("chanAB must be a list of length 2 with unique elements!")

    # check that operator is a string and a valid operator
    if not isinstance(operator, str):
        raise TypeError("operator must be a string!")
    else:
        if operator not in [">", "<", "<=", ">=", "=="]:
            raise ValueError("Invalid operator!")

    # ensure that diff is a float or an int
    if not isinstance(diff, (float, int)):
        raise TypeError("diff must be a float or int!")


def _check_source_Sv_freq_diff(
    source_Sv: xr.Dataset,
    freqAB: Optional[List[float]] = None,
    chanAB: Optional[List[str]] = None,
) -> None:
    """
    Ensures that ``source_Sv`` contains ``channel`` as a coordinate and
    ``frequency_nominal`` as a variable, the provided list input
    (``freqAB`` or ``chanAB``) are contained in the coordinate ``channel``
    or variable ``frequency_nominal``, and ``source_Sv`` does not have
    repeated values for ``channel`` and ``frequency_nominal``.

    Parameters
    ----------
    source_Sv: xr.Dataset
        A Dataset that contains the Sv data to create a mask for
    freqAB: list of float, optional
        The pair of nominal frequencies to be used for frequency-differencing, where
        the first element corresponds to ``freqA`` and the second element corresponds
        to ``freqB``
    chanAB: list of float, optional
        The pair of channels that will be used to select the nominal frequencies to be
        used for frequency-differencing, where the first element corresponds to ``freqA``
        and the second element corresponds to ``freqB``
    """

    # check that channel and frequency nominal are in source_Sv
    if "channel" not in source_Sv.coords:
        raise ValueError("The Dataset defined by source_Sv must have channel as a coordinate!")
    elif "frequency_nominal" not in source_Sv.variables:
        raise ValueError(
            "The Dataset defined by source_Sv must have frequency_nominal as a variable!"
        )

    # make sure that the channel and frequency_nominal values are not repeated in source_Sv
    if len(set(source_Sv.channel.values)) < source_Sv.channel.size:
        raise ValueError(
            "The provided source_Sv contains repeated channel values, this is not allowed!"
        )

    if len(set(source_Sv.frequency_nominal.values)) < source_Sv.frequency_nominal.size:
        raise ValueError(
            "The provided source_Sv contains repeated frequency_nominal "
            "values, this is not allowed!"
        )

    # check that the elements of freqAB are in frequency_nominal
    if (freqAB is not None) and (not all([freq in source_Sv.frequency_nominal for freq in freqAB])):
        raise ValueError(
            "The provided list input freqAB contains values that "
            "are not in the frequency_nominal variable!"
        )

    # check that the elements of chanAB are in channel
    if (chanAB is not None) and (not all([chan in source_Sv.channel for chan in chanAB])):
        raise ValueError(
            "The provided list input chanAB contains values that are "
            "not in the channel coordinate!"
        )


def frequency_differencing(
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
        to ``freqB``. Only one of ``freqAB`` and ``chanAB`` should be provided, and not both.
    chanAB: list of float, optional
        The pair of channels that will be used to select the nominal frequencies to be
        used for frequency-differencing, where the first element corresponds to ``freqA``
        and the second element corresponds to ``freqB``. Only one of ``freqAB`` and ``chanAB``
        should be provided, and not both.
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
    ValueError
        If neither ``freqAB`` or ``chanAB`` are given
    ValueError
        If both ``freqAB`` and ``chanAB`` are given
    TypeError
        If any input is not of the correct type
    ValueError
        If either ``freqAB`` or ``chanAB`` are provided and the list
        does not contain 2 distinct elements
    ValueError
        If ``freqAB`` contains values that are not contained in ``frequency_nominal``
    ValueError
        If ``chanAB`` contains values that not contained in ``channel``
    ValueError
        If ``operator`` is not one of the following: ``">", "<", "<=", ">=", "=="``
    ValueError
        If the path provided for ``source_Sv`` is not a valid path
    ValueError
        If ``freqAB`` or ``chanAB`` is provided and the Dataset produced by ``source_Sv``
        does not contain the coordinate ``channel`` and variable ``frequency_nominal``

    Notes
    -----
    This function computes the frequency differencing as follows:
    ``Sv_freqA - Sv_freqB operator diff``. Thus, if ``operator = "<"``
    and ``diff = "5"`` the following would be calculated:
    ``Sv_freqA - Sv_freqB < 5``.

    Examples
    --------
    Compute frequency-differencing mask using a mock Dataset and channel selection:

    >>> n = 5 # set the number of ping times and range samples
    ...
    >>> # create mock Sv data
    >>> Sv_da = xr.DataArray(data=np.stack([np.arange(n**2).reshape(n,n), np.identity(n)]),
    ...                      coords={"channel": ['chan1', 'chan2'],
    ...                              "ping_time": np.arange(n), "range_sample":np.arange(n)})
    ...
    >>> # obtain mock frequency_nominal data
    >>> freq_nom = xr.DataArray(data=np.array([1.0, 2.0]),
    ...                         coords={"channel": ['chan1', 'chan2']})
    ...
    >>> # construct mock Sv Dataset
    >>> Sv_ds = xr.Dataset(data_vars={"Sv": Sv_da, "frequency_nominal": freq_nom})
    ...
    >>> # compute frequency-differencing mask using channel names
    >>> echopype.mask.frequency_differencing(source_Sv=mock_Sv_ds, storage_options={}, freqAB=None,
    ...                                      chanAB = ['chan1', 'chan2'],
    ...                                      operator = ">=", diff=10.0)
    <xarray.DataArray 'mask' (ping_time: 5, range_sample: 5)>
    array([[False, False, False, False, False],
           [False, False, False, False, False],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True]])
    Coordinates:
      * ping_time     (ping_time) int64 0 1 2 3 4
      * range_sample  (range_sample) int64 0 1 2 3 4
    """

    # check that non-data related inputs were correctly provided
    _check_freq_diff_non_data_inputs(freqAB, chanAB, operator, diff)

    # validate the source_Sv type or path (if it is provided)
    source_Sv, file_type = validate_source_ds_da(source_Sv, storage_options)

    if isinstance(source_Sv, str):
        # open up Dataset using source_Sv path
        source_Sv = xr.open_dataset(source_Sv, engine=file_type, chunks={}, **storage_options)

    # check the source_Sv with respect to channel and frequency_nominal
    _check_source_Sv_freq_diff(source_Sv, freqAB, chanAB)

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
    da.name = "mask"

    return da
