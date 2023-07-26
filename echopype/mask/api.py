import datetime
import operator as op
import pathlib
from typing import List, Optional, Union

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d


from ..utils.io import validate_source_ds_da
from ..utils.prov import add_processing_level, echopype_prov_attrs, insert_input_processing_level

# lookup table with key string operator and value as corresponding Python operator
str2ops = {
    ">": op.gt,
    "<": op.lt,
    "<=": op.le,
    ">=": op.ge,
    "==": op.eq,
}


def _validate_source_ds(source_ds, storage_options_ds):
    """
    Validate the input ``source_ds`` and the associated ``storage_options_mask``.
    """
    # Validate the source_ds type or path (if it is provided)
    source_ds, file_type = validate_source_ds_da(source_ds, storage_options_ds)

    if isinstance(source_ds, str):
        # open up Dataset using source_ds path
        source_ds = xr.open_dataset(source_ds, engine=file_type, chunks={}, **storage_options_ds)

    # Check source_ds coordinates
    if "ping_time" not in source_ds or "range_sample" not in source_ds:
        raise ValueError("'source_ds' must have coordinates 'ping_time' and 'range_sample'!")

    return source_ds


def _validate_and_collect_mask_input(
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
            if isinstance(mask_val, (str, pathlib.Path)):
                # open up DataArray using mask path
                mask[mask_ind] = xr.open_dataarray(
                    mask_val, engine=file_type, chunks={}, **storage_options_mask[mask_ind]
                )

            # check mask coordinates
            # the coordinate sequence matters, so fix the tuple form
            allowed_dims = [
                ("ping_time", "range_sample"),
                ("channel", "ping_time", "range_sample"),
            ]
            if mask[mask_ind].dims not in allowed_dims:
                raise ValueError("All masks must have dimensions ('ping_time', 'range_sample')!")

    else:
        if not isinstance(storage_options_mask, dict):
            raise ValueError(
                "The provided input storage_options_mask should be a single "
                "dict because mask is a single value!"
            )

        # validate the mask type or path (if it is provided)
        mask, file_type = validate_source_ds_da(mask, storage_options_mask)

        if isinstance(mask, (str, pathlib.Path)):
            # open up DataArray using mask path
            mask = xr.open_dataarray(mask, engine=file_type, chunks={}, **storage_options_mask)

    return mask


def _check_var_name_fill_value(
    source_ds: xr.Dataset, var_name: str, fill_value: Union[int, float, np.ndarray, xr.DataArray]
) -> Union[int, float, np.ndarray, xr.DataArray]:
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

    Returns
    -------
    fill_value: int or float or np.ndarray or xr.DataArray
        fill_value with sanitized dimensions

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

    # make sure that fill_values is the same shape as var_name
    if isinstance(fill_value, (np.ndarray, xr.DataArray)):
        if isinstance(fill_value, xr.DataArray):
            fill_value = fill_value.data.squeeze()  # squeeze out length=1 channel dimension
        elif isinstance(fill_value, np.ndarray):
            fill_value = fill_value.squeeze()  # squeeze out length=1 channel dimension

        source_ds_shape = (
            source_ds[var_name].isel(channel=0).shape
            if "channel" in source_ds[var_name].coords
            else source_ds[var_name].shape
        )

        if fill_value.shape != source_ds_shape:
            raise ValueError(
                f"If fill_value is an array it must be of the same shape as {var_name}!"
            )

    return fill_value


def _variable_prov_attrs(
    masked_da: xr.DataArray, source_mask: Union[xr.DataArray, List[xr.DataArray]]
) -> dict:
    """
    Extract and compose masked Sv provenance attributes from the masked Sv and the
    masks used to generate it.

    Parameters
    ----------
    masked_da: xr.DataArray
        Masked Sv
    source_mask: Union[xr.DataArray, List[xr.DataArray]]
        Individual mask or list of masks used to create the masked Sv

    Returns
    -------
    dict
        Dictionary of provenance attributes (attribute name and value) for the intended variable.
    """
    # Modify core variable attributes
    attrs = {
        "long_name": "Volume backscattering strength, masked (Sv re 1 m-1)",
        "actual_range": [
            round(float(masked_da.min().values), 2),
            round(float(masked_da.max().values), 2),
        ],
    }
    # Add history attribute
    history_attr = f"{datetime.datetime.utcnow()} +00:00. " "Created masked Sv dataarray."  # noqa
    attrs = {**attrs, **{"history": history_attr}}

    # Add attributes from the mask DataArray, if present
    # Handle only a single mask. If not passed to apply_mask as a single DataArray,
    # will use the first mask of the list passed to  apply_mask
    # TODO: Expand it to handle attributes from multiple masks
    if isinstance(source_mask, xr.DataArray) or (
        isinstance(source_mask, list) and isinstance(source_mask[0], xr.DataArray)
    ):
        use_mask = source_mask[0] if isinstance(source_mask, list) else source_mask
        if len(use_mask.attrs) > 0:
            mask_attrs = use_mask.attrs.copy()
            if "history" in mask_attrs:
                # concatenate the history string as new line
                attrs["history"] += f"\n{mask_attrs['history']}"
                mask_attrs.pop("history")
            attrs = {**attrs, **mask_attrs}

    return attrs


@add_processing_level("L3*")
def apply_mask(
    source_ds: Union[xr.Dataset, str, pathlib.Path],
    mask: Union[xr.DataArray, str, pathlib.Path, List[Union[xr.DataArray, str, pathlib.Path]]],
    var_name: str = "Sv",
    fill_value: Union[int, float, np.ndarray, xr.DataArray] = np.nan,
    storage_options_ds: dict = {},
    storage_options_mask: Union[dict, List[dict]] = {},
) -> xr.Dataset:
    """
    Applies the provided mask(s) to the Sv variable ``var_name``
    in the provided Dataset ``source_ds``.

    Parameters
    ----------
    source_ds: xr.Dataset, str, or pathlib.Path
        Points to a Dataset that contains the variable the mask should be applied to
    mask: xr.DataArray, str, pathlib.Path, or a list of these datatypes
        The mask(s) to be applied.
        Can be a single input or list that corresponds to a DataArray or a path.
        Each entry in the list must have dimensions ``('ping_time', 'range_sample')``.
        Multi-channel masks are not currently supported.
        If a path is provided this should point to a zarr or netcdf file with only
        one data variable in it.
        If the input ``mask`` is a list, a logical AND will be used to produce the final
        mask that will be applied to ``var_name``.
    var_name: str, default="Sv"
        The Sv variable name in ``source_ds`` that the mask should be applied to.
        This variable needs to have coordinates ``ping_time`` and ``range_sample``,
        and can optionally also have coordinate ``channel``.
        In the case of a multi-channel Sv data variable, the ``mask`` will be broadcast
        to all channels.
    fill_value: int, float, np.ndarray, or xr.DataArray, default=np.nan
        Value(s) at masked indices.
        If ``fill_value`` is of type ``np.ndarray`` or ``xr.DataArray``,
        it must have the same shape as each entry of ``mask``.
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
    """

    # Validate the source_ds
    source_ds = _validate_source_ds(source_ds, storage_options_ds)

    # Validate and form the mask input to be used downstream
    mask = _validate_and_collect_mask_input(mask, storage_options_mask)

    # Check var_name and sanitize fill_value dimensions if an array
    fill_value = _check_var_name_fill_value(source_ds, var_name, fill_value)

    # Obtain final mask to be applied to var_name
    if isinstance(mask, list):
        # perform a logical AND element-wise operation across the masks
        final_mask = np.logical_and.reduce(mask)

        # xr.where has issues with attrs when final_mask is an array, so we make it a DataArray
        final_mask = xr.DataArray(final_mask, coords=mask[0].coords)
    else:
        final_mask = mask

    # Sanity check: final_mask should be of the same shape as source_ds[var_name]
    #               along the ping_time and range_sample dimensions
    def get_ch_shape(da):
        return da.isel(channel=0).shape if "channel" in da.dims else da.shape

    # Below operate on the actual data array to be masked
    source_da = source_ds[var_name]

    source_da_shape = get_ch_shape(source_da)
    final_mask_shape = get_ch_shape(final_mask)

    if final_mask_shape != source_da_shape:
        raise ValueError(
            f"The final constructed mask is not of the same shape as source_ds[{var_name}] "
            "along the ping_time and range_sample dimensions!"
        )

    # final_mask is always an xr.DataArray with at most length=1 channel dimension
    if "channel" in final_mask.dims:
        final_mask = final_mask.isel(channel=0)

    # Make sure fill_value and final_mask are expanded in dimensions
    if "channel" in source_da.dims:
        if isinstance(fill_value, np.ndarray):
            fill_value = np.array([fill_value] * source_da["channel"].size)
        final_mask = np.array([final_mask.data] * source_da["channel"].size)

    # Apply the mask to var_name
    # Somehow keep_attrs=True errors out here, so will attach later
    var_name_masked = xr.where(final_mask, x=source_da, y=fill_value)

    # Obtain a shallow copy of source_ds
    output_ds = source_ds.copy(deep=False)

    # Replace var_name with var_name_masked
    output_ds[var_name] = var_name_masked
    output_ds[var_name] = output_ds[var_name].assign_attrs(source_da.attrs)

    # Add or modify variable and global (dataset) provenance attributes
    output_ds[var_name] = output_ds[var_name].assign_attrs(
        _variable_prov_attrs(output_ds[var_name], mask)
    )

    process_type = "mask"
    prov_dict = echopype_prov_attrs(process_type=process_type)
    prov_dict[f"{process_type}_function"] = "mask.apply_mask"

    output_ds = output_ds.assign_attrs(prov_dict)

    output_ds = insert_input_processing_level(output_ds, input_ds=source_ds)

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
    chanAB: list of strings, optional
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

        # get channels corresponding to frequencies provided
        chanA = str(source_Sv.channel.isel(channel=freqA_pos).values)
        chanB = str(source_Sv.channel.isel(channel=freqB_pos).values)

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

    # assign provenance attributes
    mask_attrs = {"mask_type": "frequency differencing"}
    history_attr = (
        f"{datetime.datetime.utcnow()} +00:00. "
        "Mask created by mask.frequency_differencing. "
        f"Operation: Sv['{chanA}'] - Sv['{chanB}'] {operator} {diff}"
    )

    da = da.assign_attrs({**mask_attrs, **{"history": history_attr}})

    return da


def get_attenuation_mask(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    mask_type: str = "ryan",
    r0=180, r1=280, n=30, thr=6
) -> xr.DataArray:
    """
    Create a mask based on the identified signal attenuations of Sv values at 38KHz. 
    This method is based on:
    Ryan et al. (2015) ‘Reducing bias due to noise and attenuation in 
        open-ocean echo integration data’, ICES Journal of Marine Science,
        72: 2482–2493.
    and,
    Ariza et al. (2022) 'Acoustic seascape partitioning through functional data analysis',
    Journal of Biogeography, 00, 1– 15. https://doi.org/10.1111/jbi.14534
    

    Parameters
    ----------
    source_Sv: xr.Dataset or str or pathlib.Path
        If a Dataset this value contains the Sv data to create a mask for,
        else it specifies the path to a zarr or netcdf file containing
        a Dataset. This input must correspond to a Dataset that has the
        coordinate ``channel`` and variables ``frequency_nominal`` and ``Sv``.
    mask_type: str with either "ryan" or "ariza" based on the prefered method for signal attenuation mask generation
    Returns
    -------
    xr.DataArray
        A DataArray containing the mask for the Sv data. Regions satisfying the thresholding
        criteria are filled with ``True``, else the regions are filled with ``False``.

    Raises
    ------
    ValueError
        If neither ``ryan`` or ``azira`` are given

    Notes
    -----


    Examples
    --------

    """
    assert mask_type in ['ryan', 'ariza'], "mask_type must be either 'ryan' or 'ariza'"
    
    Sv = source_Sv['Sv'].values[0]
    r = source_Sv['echo_range'].values[0,0]
    if mask_type == "ryan":
        mask = _get_attenuation_mask_ryan(Sv, r, r0=r0, r1=r1, n=n, thr=thr)
    elif mask_type == "ariza":
        mask = _get_attenuation_mask_ariza(Sv, r)
    else:
        raise ValueError(
                "The provided mask_type must be ryan or ariza!"
                )
                
    return_mask = xr.DataArray(mask,
                                    dims=("ping_time", 
                                          "range_sample"),
                                    coords={"ping_time": source_Sv.ping_time,
                                            "range_sample": source_Sv.range_sample}
                                   )
    return return_mask
    
    
def _get_attenuation_mask_ryan(Sv, r, r0=400, r1=500, n=30, thr=8, start=0):
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
        Sv (float): 2D array with Sv data to be masked (dB). 
        r (float):  1D array with range data (m).
        r0 (int): upper limit of SL (m).
        r1 (int): lower limit of SL (m).
        n (int): number of preceding & subsequent pings defining the block.
        thr (int): user-defined threshold value (dB).
        start (int): ping index to start processing.
        
    Returns:
        list: 2D boolean array with AS mask and 2D boolean array with mask
              indicating where AS detection was unfeasible.
    """
    
     # raise errors if wrong arguments
    if r0>r1:
        raise Exception('Minimum range has to be shorter than maximum range')
    
    # return empty mask if searching range is outside the echosounder range
    if (r0>r[-1]) or (r1<r[0]):
        mask  = np.zeros_like(Sv, dtype=bool) 
        mask_ = np.zeros_like(Sv, dtype=bool) 
        return mask, mask_ 
    
    # turn layer boundaries into arrays with length = Sv.shape[1]
    r0 = np.ones(Sv.shape[1])*r0
    r1 = np.ones(Sv.shape[1])*r1
    
    # start masking process    
    mask_ = np.zeros(Sv.shape, dtype=bool)
    mask = np.zeros(Sv.shape, dtype=bool) 
    # find indexes for upper and lower SL limits
    up = np.argmin(abs(r - r0))
    lw = np.argmin(abs(r - r1))    
    for j in range(start, len(Sv)):
        
        
            # TODO: now indexes are the same at every loop, but future 
            # versions will have layer boundaries with variable range
            # (need to implement mask_layer.py beforehand!)
        
        # mask where AS evaluation is unfeasible (e.g. edge issues, all-NANs)
        if (j-n<0) | (j+n>len(Sv)-1) | np.all(np.isnan(Sv[j,up:lw])):        
            mask_[j, :] = True
            
        
        # compare ping and block medians otherwise & mask ping if too different
        else:
            pingmedian  = _log(np.nanmedian(_lin(Sv[j, up:lw])))
            blockmedian = _log(np.nanmedian(_lin(Sv[(j-n):(j+n),up:lw ])))
            
            if (pingmedian-blockmedian)<thr:            
                mask[j, :] = True
                
    final_mask = np.logical_not(mask[start:, :] | mask_[start:, :]) 
    return final_mask

def _get_attenuation_mask_ariza(Sv, r, offset=20, thr=(-40,-35), m=20, n=50):
    """
    Mask attenuated pings by looking at seabed breaches.
    
    Ariza et al. (2022) 'Acoustic seascape partitioning through functional data analysis',
    Journal of Biogeography, 00, 1– 15. https://doi.org/10.1111/jbi.14534
    """
    
    # get ping array
    p = np.arange(len(Sv))
    # set to NaN shallow waters and data below the Sv threshold
    Sv_ = Sv.copy()
    Sv_[0:np.nanargmin(abs(r - offset)), :] = np.nan
    Sv_[Sv_<-thr[0]] = np.nan
    
    # bin Sv
    # TODO: update to 'twod' and 'full' funtions   
    # DID    
    irvals = np.linspace(p[0],p[-1],num=int((p[-1]-p[0])/n)+1)
    jrvals = np.linspace(r[0],r[-1],num=int((r[-1]-r[0])/m)+1)
    Sv_bnd, p_bnd, r_bnd = _twod(Sv_, p, r, irvals, jrvals, operation='mean')[0:3]
    print(r_bnd)
    print(p_bnd)
    print("originals:")
    print(r)
    print(p)
    Sv_bnd = _full(Sv_bnd, r_bnd, p_bnd, r, p)
    
    # label binned Sv data features
    Sv_lbl = label(~np.isnan(Sv_bnd))
    labels = np.unique(Sv_lbl)
    labels = np.delete(labels, np.where(labels==0))
    
    # list the median values for each Sv feature
    val = []
    for lbl in labels:
        val.append(_log(np.nanmedian(_lin(Sv_bnd[Sv_lbl==lbl]))))
    
    # keep the feature with a median above the Sv threshold (~seabed)
    # and set the rest of the array to NaN
    if val:
        if np.nanmax(val)>thr[1]:
            labels = labels[val!=np.nanmax(val)]
            for lbl in labels:
                Sv_bnd[Sv_lbl==lbl] = np.nan
        else:
            Sv_bnd[:] = np.nan
    else:
        Sv_bnd[:] = np.nan
        
    # remove everything in the original Sv array that is not seabed
    Sv_sb = Sv.copy()
    Sv_sb[np.isnan(Sv_bnd)] = np.nan
    
    # compute the percentile 90th for each ping, at the range at which 
    # the seabed is supposed to be.    
    seabed_percentile = _log(np.nanpercentile(_lin(Sv_sb), 95, axis=0))
    
    # get mask where this value falls bellow a Sv threshold (seabed breaches)
    mask = seabed_percentile<thr[0]
    mask = np.tile(mask, [len(Sv), 1])    
    
    return mask
    
    
def _lin(variable):
    """
    Turn variable into the linear domain.     
    
    Args:
        variable (float): array of elements to be transformed.    
    
    Returns:
        float:array of elements transformed
    """
    
    lin    = 10**(variable/10)
    return   lin  

def _log(variable):
    """
    Turn variable into the logarithmic domain. This function will return -999
    in the case of values less or equal to zero (undefined logarithm). -999 is
    the convention for empty water or vacant sample in fisheries acoustics. 
    
    Args:
        variable (float): array of elements to be transformed.    
    
    Returns:
        float: array of elements transformed
    """
    
    # convert variable to float array
    back2single = False
    back2list   = False
    back2int    = False    
    if not isinstance(variable, np.ndarray):
        if isinstance(variable, list):
            variable  = np.array(variable)
            back2list = True
        else:
            variable    = np.array([variable])
            back2single = True            
    if variable.dtype=='int64':
        variable   = variable*1.0
        back2int = True
    
    # compute logarithmic value except for zeros values, which will be -999 dB    
    mask           = np.ma.masked_less_equal(variable, 0).mask
    variable[mask] = np.nan
    log            = 10*np.log10(variable)
    log[mask]      = -999
    
    # convert back to original data format and return
    if back2int:
        log = np.int64(log)
    if back2list:
        log = log.tolist()
    if back2single:
        log = log[0]       
    return log
    
    
    
def _full(datar, irvals, jrvals, idim, jdim):
    """
    Turn resampled data back to full resolution, according to original i and j
    full resolution dimensions.
    
    Args:
        datar  (float): 2D array with resampled data.
        irvals (float): 1D array with i resampling intervals.
        jdimr  (float): 1D array with i resampling intervals.
        idim   (float): 1D array with full resolution i axis.
        jdim   (float): 1D array with full resolution j axis.
        
    Returns:
        float: 2D array with data resampled at full resolution.
        bool : 2D array with mask indicating valid values in data resampled
               at full resolution.
    """
    
    
    # check for appropiate inputs
    if len(irvals)<2:
        raise Exception('i resampling interval length must be >2')
    if len(jrvals)<2:
        raise Exception('j resampling interval length must be >2')
    if len(datar)+1<len(irvals):
        raise Exception('i resampling intervals length can\'t exceed data height + 1')
    if len(datar[0])+1<len(jrvals):
        raise Exception('j resampling intervals length can\'t exceed data width + 1') 
        
    # get i/j axes from i/j dimensions and i/j intervals
    iax   = np.arange(len(idim))
    jax   = np.arange(len(jdim))
    iaxrs = dim2ax(idim, iax, irvals)    
    jaxrs = dim2ax(jdim, jax, jrvals)
    
    # check whether i/j resampled axes and i/j intervals are different
    idiff = True
    if len(iaxrs)==len(iax):
        if (iaxrs==iax).all():
            idiff = False
    jdiff = True
    if len(jaxrs)==len(jax):
        if (jaxrs==jax).all():
            jdiff = False
    
    # preallocate full resolution data array 
    data = np.zeros((len(iax), len(jax)))*np.nan
        
    # if i/j axes are different, resample back to full along i/j dimensions
    if idiff&jdiff:
        for i in range(len(iaxrs)-1):            
            idx = np.where((iaxrs[i]<=iax) & (iax<iaxrs[i+1]))[0]
            for j in range(len(jaxrs)-1):        
                jdx = np.where((jaxrs[j]<=jax) & (jax<jaxrs[j+1]))[0]
                if idx.size*jdx.size > 0:
                    data[idx[0]:idx[-1]+1, jdx[0]:jdx[-1]+1]= datar[i,j]
    
    # if only i axis is different, resample back to full along i dimension
    elif idiff & (not jdiff):
        for i in range(len(iaxrs)-1):            
            idx = np.where((iaxrs[i]<=iax) & (iax<iaxrs[i+1]))[0]
            if idx.size>0:
                data[idx[0]:idx[-1]+1, :]= datar[i, :]
        
    # if only j axis is different, resample back to full along j dimension
    elif (not idiff) & jdiff:        
        for j in range(len(jaxrs)-1):        
            jdx = np.where((jaxrs[j]<=jax) & (jax<jaxrs[j+1]))[0]
            if jdx.size > 0:
                data[:, jdx[0]:jdx[-1]+1]= datar[:, j].reshape(-1,1)
        
    # if i/j resampled & i/j full are the same, data resampled & data are equal
    else:
        warnings.warn("Array already at full resolution!", RuntimeWarning)
        data= datar.copy()
    
    # get mask indicating where data couldn't be resampled back
    mask_ = np.zeros_like(data, dtype=bool)
    i1= np.where(iax<iaxrs[-1])[0][-1] + 1
    j1= np.where(jax<jaxrs[-1])[0][-1] + 1
    if idiff:
        mask_[i1:,   :] = True
    if jdiff:
        mask_[  :, j1:] = True
    
    return data, mask_
    
    
def _twod(data, idim, jdim, irvals, jrvals, log=False, operation='mean'):
    """
    Resample down an array along the two dimensions, i and j.
    
    Args:
        data   (float): 2D array with data to be resampled.
        idim   (float): i vertical dimension.
        jdim   (float): j horizontal dimension.
        irvals (float): i resampling intervals for i vertical dimension.
        jrvals (float): j resampling intervals for j horizontal dimension.
        log    (bool ): if True, data is considered logarithmic and it will 
                        be converted to linear during calculations.
        operation(str): type of resampling operation. Accepts "mean" or "sum".
                    
    Returns:
        float: 2D resampled data array
        float: 1D resampled i vertical dimension
        float: 1D resampled j horizontal dimension
        float: 2D array with percentage of valid samples included on each
               resampled cell.
    """ 
    
    # check for appropiate inputs
    if len(irvals)<2:
        raise Exception('length of i resampling intervals must be  >2')
    if len(jrvals)<2:
        raise Exception('length of j resampling intervals must be >2')
    if len(data)!=len(idim):
        raise Exception('data height and idim length must be the same')
    if len(data[0])!=len(jdim):
        raise Exception('data width and jdim length must be the same')        
    for irval, jrval in zip(irvals, jrvals):
        if (irval<idim[0]) | (idim[-1]<irval):
            raise Exception('i resampling intervals must be within idim range')
        if (jrval<jdim[0]) | (jdim[-1]<jrval):
            raise Exception('j resampling intervals must be within jdim range')
        
    # convert data to linear, if logarithmic
    if log is True:
        data = _lin(data)
    
    # get i/j axes from i/j dimensions and i/j intervals
    iax   = np.arange(len(idim))
    jax   = np.arange(len(jdim))
    iaxrs = dim2ax(idim, iax, irvals)    
    jaxrs = dim2ax(jdim, jax, jrvals)
    
    # declare new array to allocate resampled values, and new array to
    # alllocate the percentage of values used for resampling
    datar     = np.zeros((len(iaxrs)-1, len(jaxrs)-1))*np.nan
    percentage = np.zeros((len(iaxrs)-1, len(jaxrs)-1))*np.nan
    
    # iterate along-range
    for i in range(len(iaxrs)-1):
        
        # get i indexes to locate the samples for the binning operation
        idx0     = np.where(iax-iaxrs[i+0]<=0)[0][-1]        
        idx1     = np.where(iaxrs[i+1]-iax> 0)[0][-1]
        idx      = np.arange(idx0, idx1+1)
        
        # get i weights as the sum of the sample proportions taken
        iweight0 = 1 - abs(iax[idx0]-iaxrs[i+0])
        iweight1 = abs(iax[idx1]-iaxrs[i+1])
        if len(idx)>1:
            iweights = np.r_[iweight0, np.ones(len(idx)-2), iweight1]
        else:
            iweights = np.array([iweight0-1 + iweight1])
        
        # iterate along-pings
        for j in range(len(jaxrs)-1):
            
            # get j indexes to locate the samples for the binning operation
            jdx0     = np.where(jax-jaxrs[j+0]<=0)[0][-1]        
            jdx1     = np.where(jaxrs[j+1]-jax> 0)[0][-1]
            jdx      = np.arange(jdx0, jdx1+1)
            
            # get j weights as the sum of the sample proportions taken
            jweight0 = 1 - abs(jax[jdx0]-jaxrs[j+0])
            jweight1 = abs(jax[jdx1]-jaxrs[j+1])
            if len(jdx)>1:
                jweights = np.r_[jweight0, np.ones(len(jdx)-2), jweight1]
            else:
                jweights = np.array([jweight0-1 + jweight1])
                            
            # get data and weight 2D matrices for the binning operation
            d = data[idx[0]:idx[-1]+1, jdx[0]:jdx[-1]+1]
            w = np.multiply.outer(iweights, jweights)
                      
            # if d is an all-NAN array, return NAN as the weighted operation
            # and zero as the percentage of valid numbers used for binning
            if np.isnan(d).all():
                datar    [i, j] = np.nan
                percentage[i, j] = 0
            
            #compute weighted operation & percentage of valid numbers otherwise
            else:                    
                w_ = w.copy()
                w_[np.isnan(d)] = np.nan
                if operation=='mean':
                    datar   [i, j]  = np.nansum(d*w_)/np.nansum(w_)
                elif operation=='sum':
                    datar   [i, j]  = np.nansum(d*w_)
                else:
                    raise Exception('Operation not recognised')                        
                percentage[i, j]  = np.nansum(  w_)/np.nansum(w )*100                        
    
    # convert back to logarithmic, if data was logarithmic
    if log is True:
        datar = _log(datar)
    
    # get resampled dimensions from resampling intervals
    idimr = irvals[:-1]
    jdimr = jrvals[:-1]
    
    # return data
    return datar, idimr, jdimr, percentage
    
def dim2ax(dim, ax, dimrs):
    """
    It gives you a new resampled axis based on a known dimension/axis pair, and 
    a new resampled dimesion.
    
    Args:
        dim   (float): 1D array with original dimension.
        ax    (int  ): 1D array with original axis.
        dimrs (float): 1D array with resampled dimension.
    
    Returns:
        float: 1D array with resampled axis.
    
    Notes:
        Dimension refers to range, time, latitude, distance, etc., and axis
        refer to dimension indexes such as sample or ping number.
        
    Example:
                                                         (resampled)
        seconds dimension | ping axis       seconds dimension | ping axis
        ------------------·----------       ------------------·----------
                       0  | 0                             0   | 0
                       2  | 1                             3   | 1.5
                       4  | 2          ==>                6   | 3.0
                       6  | 3                             9   | 4.5
                       8  | 4                             -   | -
                      10  | 4                             -   | -
    """
    
    # check that resampled dimension doesn't exceed the limits
    # of the original one
    if (dimrs[0]<dim[0]) | (dimrs[-1]>dim[-1]):
        raise Exception('resampling dimension can not exceed ' +
                        'the original dimension limits') 
        
    # convert variables to float64 if they are in datetime64 format
    epoch = np.datetime64('1970-01-01T00:00:00')    
    if 'datetime64' in str(dim[0].dtype):        
        dim = np.float64(dim-epoch)        
    if 'datetime64' in str(dimrs[0].dtype):
        dimrs = np.float64(dimrs-epoch)
    
    # get interpolated new y variable    
    f    = interp1d(dim, ax)
    axrs = f(dimrs)
            
    return axrs