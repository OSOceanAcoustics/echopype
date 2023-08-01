import datetime
import operator as op
import pathlib
from typing import List, Optional, Union

from scipy.signal import convolve2d
import scipy.ndimage as nd
from skimage.morphology import remove_small_objects
from skimage.morphology import erosion
from skimage.morphology import dilation
from skimage.measure import label

import numpy as np
import xarray as xr

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

def get_seabed_mask(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    mask_type: str = "deltaSv",
    **kwargs
) -> xr.DataArray:
    """
    Create a mask based on the identified seabed at 38KHz. 
    This method is based on:
    Ryan et al. (2015) ‘Reducing bias due to noise and attenuation in 
        open-ocean echo integration data’, ICES Journal of Marine Science,
        72: 2482–2493.

    Parameters
    ----------
    source_Sv: xr.Dataset or str or pathlib.Path
        If a Dataset this value contains the Sv data to create a mask for,
        else it specifies the path to a zarr or netcdf file containing
        a Dataset. This input must correspond to a Dataset that has the
        coordinate ``channel`` and variables ``frequency_nominal`` and ``Sv``.
    mask_type: str with either 'deltaSv','maxSv','ariza','experimental','blackwell', 'blackwell_mod' based on the prefered method for signal attenuation mask generation
    Returns
    -------
    xr.DataArray
        A DataArray containing the mask for the Sv data. Regions satisfying the thresholding
        criteria are filled with ``True``, else the regions are filled with ``False``.

    Raises
    ------
    ValueError
        If neither 'deltaSv','maxSv','ariza','experimental','blackwell', 'blackwell_mod' are given

    Notes
    -----


    Examples
    --------

    """
    assert mask_type in ['deltaSv','maxSv','ariza','experimental','blackwell', 'blackwell_mod'], "mask_type must be either 'deltaSv','maxSv','ariza','experimental','blackwell', 'blackwell_mod'"
    
    Sv = source_Sv['Sv'].values[0].T
    r = source_Sv['echo_range'].values[0,0]
    if mask_type == "blackwell":
        # Define a list of the keyword arguments your function can handle
        valid_args = {'theta', 'phi','r0', 'r1', 'tSv',
                      'ttheta', 'tphi', 'wtheta', 'wphi'
                      }
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_seabed_mask_blackwell(Sv, r, m=5, **filtered_kwargs)
    elif mask_type == "blackwell_mod":
        #Define a list of the keyword arguments your function can handle
        valid_args = {'theta', 'phi','r0', 'r1', 'tSv',
                      'ttheta', 'tphi', 'wtheta', 'wphi', 
                      'rlog', 'tpi', 'freq', 'rank'
                      }
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_seabed_mask_blackwell_mod(Sv, r, **filtered_kwargs )
    elif mask_type == "ariza":
        #Define a list of the keyword arguments your function can handle
        valid_args = {'r0', 'r1','roff','thr','ec','ek','dc','dk'}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_seabed_mask_ariza(Sv, r, **filtered_kwargs )
    elif mask_type == "experimental":
        #Define a list of the keyword arguments your function can handle
        valid_args = {'r0', 'r1','roff','ns','nd','thr'}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_seabed_mask_experimental(Sv, r, **filtered_kwargs )
    elif mask_type == "deltaSv":
        #Define a list of the keyword arguments your function can handle
        valid_args = {'r0', 'r1','roff','thr'}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_seabed_mask_deltaSv(Sv, r, **filtered_kwargs )
    elif mask_type == "maxSv":
        #Define a list of the keyword arguments your function can handle
        valid_args = {'r0', 'r1','roff','thr'}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_seabed_mask_maxSv(Sv, r, **filtered_kwargs )
    else:
        raise ValueError(
                "The provided 'mask_type' must be be either 'deltaSv','maxSv','ariza','experimental','blackwell', 'blackwell_mod'!"
                )
    
    mask = np.logical_not(mask.T)
    return_mask = xr.DataArray(mask,
                                    dims=("ping_time", 
                                          "range_sample"),
                                    coords={"ping_time": source_Sv.ping_time,
                                            "range_sample": source_Sv.range_sample}
                                   )
    return return_mask
def _get_seabed_mask_maxSv(Sv, r, r0=10, r1=1000, roff=10, thr=(-40, -60)):
    """
    Initially detects the seabed as the ping sample with the strongest Sv value, 
    as long as it exceeds a dB threshold. Then it searchs up along the ping 
    until Sv falls below a secondary (lower) dB threshold, where the final 
    seabed is set.
    
    Args:
        Sv (float): 2D Sv array (dB).
        r (float): 1D range array (m).
        r0 (int): minimum range below which the search will be performed (m).
        r1 (int): maximum range above which the search will be performed (m).
        roff (int): seabed range offset (m).
        thr (tuple): 2 integers with 1st and 2nd Sv threshold (dB).

    Returns:
        bool: 2D array with seabed mask.     
    """
        
    
    # get offset and range indexes
    roff = np.nanargmin(abs(r-roff))
    r0 = np.nanargmin(abs(r - r0))
    r1 = np.nanargmin(abs(r - r1))
    
    # get indexes for maximum Sv along every ping,
    idx = np.int64(np.zeros(Sv.shape[1]))
    idx[~np.isnan(Sv).all(axis=0)] = np.nanargmax(
            Sv[r0:r1, ~np.isnan(Sv).all(axis=0)], axis=0) + r0
    
    # indexes with maximum Sv < main threshold are discarded (=0)
    maxSv = Sv[idx, range(len(idx))]
    maxSv[np.isnan(maxSv)] = -999
    idx[maxSv < thr[0]] = 0
    
    # mask seabed, proceed only with acepted seabed indexes (!=0)
    idx = idx
    mask  = np.zeros(Sv.shape, dtype=bool)
    for j, i in enumerate(idx):
        if i!=0:
            
            # decrease indexes until Sv mean falls below the 2nd threshold
            if np.isnan(Sv[i-5:i, j]).all():
                Svmean = thr[1]+1
            else:      
                Svmean = _log(np.nanmean(_lin(Sv[i-5:i, j])))
            
            while (Svmean>thr[1]) & (i>=5):
                i -= 1
                       
            # subtract range offset & mask all the way down 
            i -= roff
            if i<0:
                i = 0
            mask[i:, j] = True
    
    return mask

def _get_seabed_mask_deltaSv(Sv, r, r0=10, r1=1000, roff=0, thr=20):
    """
    Examines the difference in Sv over a 2-samples moving window along
    every ping, and returns the range of the first value that exceeded 
    a user-defined dB threshold (likely, the seabed).
    
    Args:
        Sv (float): 2D Sv array (dB).
        r (float): 1D range array (m).
        r0 (int): minimum range below which the search will be performed (m).
        r1 (int): maximum range above which the search will be performed (m).
        roff (int): seabed range offset (m).
        thr (int): threshold value (dB).
        start (int): ping index to start processing.

    Returns:
        bool: 2D array with seabed mask. 
    """
    # get offset as number of samples 
    roff = np.nanargmin(abs(r-roff))
    
    # compute Sv difference along every ping
    Svdiff = np.diff(Sv, axis=0)
    dummy = np.zeros((1, Svdiff.shape[1])) * np.nan
    Svdiff = np.r_[dummy, Svdiff]
    
    # get range indexes  
    r0 = np.nanargmin(abs(r-r0))
    r1 = np.nanargmin(abs(r-r1))
    
    # get indexes for the first value above threshold, along every ping
    idx = np.nanargmax((Svdiff[r0:r1, :]>thr), axis=0) + r0
    
    # mask seabed, proceed only with acepted seabed indexes (!=0)
    idx = idx
    mask = np.zeros(Sv.shape, dtype=bool)
    for j, i in enumerate(idx):
        if i != 0: 
            
            # subtract range offset & mask all the way down
            i -= roff
            if i<0:
                i = 0
            mask[i:, j] = True        

    return mask    
def _get_seabed_mask_blackwell(Sv, r,theta, phi,
              r0=10, r1=1000, 
              tSv=-75, ttheta=702, tphi=282, 
              wtheta=28 , wphi=52):
    """
    Detects and mask seabed using the split-beam angle and Sv, based in 
    "Blackwell et al (2019), Aliased seabed detection in fisheries acoustic
    data". Complete article here: https://arxiv.org/abs/1904.10736
    
    Args:
        Sv (float): 2D numpy array with Sv data (dB)
        theta (float): 2D numpy array with the along-ship angle (degrees)
        phi (float): 2D numpy array with the athwart-ship angle (degrees)
        r (float): 1D range array (m)
        r0 (int): minimum range below which the search will be performed (m) 
        r1 (int): maximum range above which the search will be performed (m)
        tSv (float): Sv threshold above which seabed is pre-selected (dB)
        ttheta (int): Theta threshold above which seabed is pre-selected (dB)
        tphi (int): Phi threshold above which seabed is pre-selected (dB)
        wtheta (int): window's size for mean square operation in Theta field
        wphi (int): window's size for mean square operation in Phi field
                
    Returns:
        bool: 2D array with seabed mask
    """
    
    # delimit the analysis within user-defined range limits 
    r0         = np.nanargmin(abs(r - r0))
    r1         = np.nanargmin(abs(r - r1)) + 1
    Svchunk    = Sv[r0:r1, :]
    thetachunk = theta[r0:r1, :]
    phichunk   = phi[r0:r1, :]
    
    # get blur kernels with theta & phi width dimensions 
    ktheta = np.ones((wtheta, wtheta))/wtheta**2
    kphi   = np.ones((wphi  , wphi  ))/wphi  **2
    
    # perform mean square convolution and mask if above theta & phi thresholds
    thetamaskchunk = convolve2d(thetachunk, ktheta, 'same',
                                boundary='symm')**2 > ttheta
    phimaskchunk   = convolve2d(phichunk,  kphi, 'same',
                                boundary='symm')**2 > tphi
    anglemaskchunk = thetamaskchunk | phimaskchunk
        
    # if aliased seabed, mask Sv above the Sv median of angle-masked regions
    if anglemaskchunk.any():
        Svmedian_anglemasked = log(np.nanmedian(lin(Svchunk[anglemaskchunk])))
        if np.isnan(Svmedian_anglemasked):
            Svmedian_anglemasked = np.inf
        if Svmedian_anglemasked < tSv:
            Svmedian_anglemasked = tSv                            
        Svmaskchunk = Svchunk > Svmedian_anglemasked
    
        # label connected items in Sv mask
        items = nd.label(Svmaskchunk, nd.generate_binary_structure(2,2))[0]
        
        # get items intercepted by angle mask (likely, the seabed)
        intercepted = list(set(items[anglemaskchunk]))  
        if 0 in intercepted: 
            intercepted.remove(intercepted==0)
            
        # combine angle-intercepted items in a single mask 
        maskchunk = np.zeros(Svchunk.shape, dtype=bool)
        for i in intercepted:
            maskchunk = maskchunk | (items==i)
    
        # add data above r0 and below r1 (removed in first step)
        above = np.zeros((r0, maskchunk.shape[1]), dtype=bool)
        below = np.zeros((len(r) - r1, maskchunk.shape[1]), dtype=bool)
        mask  = np.r_[above, maskchunk, below]
        anglemask = np.r_[above, anglemaskchunk, below] # TODO remove
    
    # return empty mask if aliased-seabed was not detected in Theta & Phi    
    else:
        mask = np.zeros_like(Sv, dtype=bool)

    return mask, anglemask

def _get_seabed_mask_blackwell_mod(Sv, r, theta, phi, r0=10, r1=1000, tSv=-75, ttheta=702, 
                  tphi=282, wtheta=28 , wphi=52, 
                  rlog=None, tpi=None, freq=None, rank=50):
    """
    Detects and mask seabed using the split-beam angle and Sv, based in 
    "Blackwell et al (2019), Aliased seabed detection in fisheries acoustic
    data". Complete article here: https://arxiv.org/abs/1904.10736
    
    This is a modified version from the original algorithm. It includes extra
    arguments to evaluate whether aliased seabed items can occur, given the 
    true seabed detection range, and the possibility of tuning the percentile's
    rank.
    
    Args:
        Sv (float): 2D numpy array with Sv data (dB)
        theta (float): 2D numpy array with the along-ship angle (degrees)
        phi (float): 2D numpy array with the athwart-ship angle (degrees)
        r (float): 1D range array (m)
        r0 (int): minimum range below which the search will be performed (m) 
        r1 (int): maximum range above which the search will be performed (m)
        tSv (float): Sv threshold above which seabed is pre-selected (dB)
        ttheta (int): Theta threshold above which seabed is pre-selected (dB)
        tphi (int): Phi threshold above which seabed is pre-selected (dB)
        wtheta (int): window's size for mean square operation in Theta field
        wphi (int): window's size for mean square operation in Phi field
        rlog (float): Maximum logging range of the echosounder (m)
        tpi (float): Transmit pulse interval, or ping rate (s)
        freq (int): frequecy (kHz)
        rank (int): Rank for percentile operation: [0, 100]
                
    Returns:
        bool: 2D array with seabed mask
    """
    
    # raise errors if wrong arguments
    if r0>r1:
        raise Exception('Minimum range has to be shorter than maximum range')
    
    # return empty mask if searching range is outside the echosounder range
    if (r0>r[-1]) or (r1<r[0]):
        return np.zeros_like(Sv, dtype=bool)
    
    # delimit the analysis within user-defined range limits 
    i0         = np.nanargmin(abs(r - r0))
    i1         = np.nanargmin(abs(r - r1)) + 1
    Svchunk    = Sv   [i0:i1, :]
    thetachunk = theta[i0:i1, :]
    phichunk   = phi  [i0:i1, :]
    
    # get blur kernels with theta & phi width dimensions 
    ktheta = np.ones((wtheta, wtheta))/wtheta**2
    kphi   = np.ones((wphi  , wphi  ))/wphi  **2
    
    # perform mean square convolution and mask if above theta & phi thresholds
    thetamaskchunk = convolve2d(thetachunk, ktheta, 'same',
                                boundary='symm')**2 > ttheta
    phimaskchunk   = convolve2d(phichunk,  kphi, 'same',
                                boundary='symm')**2 > tphi
    anglemaskchunk = thetamaskchunk | phimaskchunk
    
    # remove aliased seabed items when estimated True seabed can not be 
    # detected below the logging range
    if (rlog is not None) and (tpi is not None) and (freq is not None):
        items       = label(anglemaskchunk)
        item_labels = np.unique(label(anglemaskchunk))[1:]
        for il in item_labels:   
            item    = items==il
            ritem   = np.nanmean(r[i0:i1][np.where(item)[0]])
            rseabed = _aliased2seabed(ritem , rlog, tpi, freq)
            if rseabed==[]:
                anglemaskchunk[item] = False

    anglemaskchunk = anglemaskchunk & (Svchunk>tSv)
        
    # if aliased seabed, mask Sv above the Sv median of angle-masked regions
    if anglemaskchunk.any():
        Svmedian_anglemasked = log(
            np.nanpercentile(lin(Svchunk[anglemaskchunk]), rank))
        if np.isnan(Svmedian_anglemasked):
            Svmedian_anglemasked = np.inf
        if Svmedian_anglemasked < tSv:
            Svmedian_anglemasked = tSv                            
        Svmaskchunk = Svchunk > Svmedian_anglemasked
    
        # label connected items in Sv mask
        items = nd.label(Svmaskchunk, nd.generate_binary_structure(2,2))[0]
        
        # get items intercepted by angle mask (likely, the seabed)
        intercepted = list(set(items[anglemaskchunk]))  
        if 0 in intercepted: 
            intercepted.remove(intercepted==0)
            
        # combine angle-intercepted items in a single mask 
        maskchunk = np.zeros(Svchunk.shape, dtype=bool)
        for i in intercepted:
            maskchunk = maskchunk | (items==i)
    
        # add data above r0 and below r1 (removed in first step)
        above = np.zeros((i0, maskchunk.shape[1]), dtype=bool)
        below = np.zeros((len(r) - i1, maskchunk.shape[1]), dtype=bool)
        mask  = np.r_[above, maskchunk, below]
    
    # return empty mask if aliased-seabed was not detected in Theta & Phi    
    else:
        mask = np.zeros_like(Sv, dtype=bool)

    return mask

def _aliased2seabed(aliased, rlog, tpi, f, c=1500, 
                   rmax={18:7000, 38:2800, 70:1100, 120:850, 200:550}):
    """ 
    Estimate true seabed, given the aliased seabed range. It might provide
    a list of ranges, corresponding to seabed reflections from several pings
    before, or provide an empty list if true seabed occurs within the logging 
    range or beyond the maximum detection range.
    
    Args:
      aliased (float): Range of aliased seabed (m).
      rlog (float): Maximum logging range (m).
      tpi (float): Transmit pulse interval (s).
      f (int): Frequency (kHz).
      c (int): Sound speed in seawater (m s-1). Defaults to 1500.
      rmax (dict): Maximum seabed detection range per frequency. Defaults 
                   to {18:7000, 38:2800, 70:1100, 120:850, 200:550}.
  
    Returns:
        float: list with estimated seabed ranges, reflected from preceeding 
        pings (ping -1, ping -2, ping -3, etc.).
        
    """   
    ping    = 0
    seabed  = 0
    seabeds = []
    while seabed<=rmax[f]:
        ping   = ping + 1
        seabed = (c*tpi*ping)/2 + aliased
        if (seabed>rlog) & (seabed<rmax[f]):
            seabeds.append(seabed)
            
    return seabeds 

def _seabed2aliased(seabed, rlog, tpi, f, c=1500, 
                   rmax={18:7000, 38:2800, 70:1100, 120:850, 200:550}):
    """
    Estimate aliased seabed range, given the true seabed range. The answer will
    be 'None' if true seabed occurs within the logging range or if it's beyond 
    the detection limit of the echosounder.

    Args:
        seabed (float): True seabed range (m).
        rlog (float): Maximum logging range (m).
        tpi (float): Transmit pulse interval (s).
        f (int): frequency (kHz).
        c (float): Sound speed in seawater (m s-1). Defaults to 1500.
        rmax (dict): Maximum seabed detection range per frequency. Defaults 
                     to {18:7000, 38:2800, 70:1100, 120:850, 200:550}.

    Returns:
        float: Estimated range of aliased seabed (m

    """
    if (not seabed<rlog) and (not seabed>rmax[f]):
        aliased = ((2*seabed) % (c*tpi)) / 2
    else:
        aliased = None
        
    return aliased

def _get_seabed_mask_experimental(Sv, r, r0=10, r1=1000, roff=0, thr=(-30,-70), ns=150, nd=3):
    """
    Mask Sv above a threshold to get a potential seabed mask. Then, the mask is
    dilated to fill seabed breaches, and small objects are removed to prevent 
    masking high Sv features that are not seabed (e.g. fish schools or spikes).    
    Once this is done, the mask is built up until Sv falls below a 2nd
    threshold, Finally, the mask is extended all the way down. 
    
    Args:
        Sv (float): 2D Sv array (dB).
        r (float): 1D range array (m).
        r0 (int): minimum range below which the search will be performed (m). 
        r1 (int): maximum range above which the search will be performed (m).
        roff (int): seabed range offset (m).
        thr (tuple): 2 integers with 1st and 2nd Sv threshold (dB).
        ns (int): maximum number of samples for an object to be removed.
        nd (int): number of dilations performed to the seabed mask.
           
    Returns:
        bool: 2D array with seabed mask.
    """

    # get indexes for range offset and range limits 
    roff = np.nanargmin(abs(r - roff))
    r0 = np.nanargmin(abs(r - r0))
    r1 = np.nanargmin(abs(r - r1)) + 1
    
    # mask Sv above the first Sv threshold
    mask = Sv[r0:r1, :] > thr[0]
    maskabove = np.zeros((r0, mask.shape[1]), dtype =bool)
    maskbelow = np.zeros((len(r) - r1, mask.shape[1]), dtype=bool)
    mask  = np.r_[maskabove, mask, maskbelow]     
    
    # remove small to prevent other high Sv features to be masked as seabed 
    # (e.g fish schools, impulse noise not properly masked. etc)
    mask = remove_small_objects(mask, ns)
    
    # dilate mask to fill seabed breaches
    # (e.g. attenuated pings or gaps from previous masking) 
    kernel = np.ones((3,5))
    mask = cv2.dilate(np.uint8(mask), kernel, iterations=nd)
    mask = np.array(mask, dtype = 'bool')
        
    # proceed with the following only if seabed was detected
    idx = np.argmax(mask, axis=0)
    for j, i in enumerate(idx):
        if i != 0:
            
            # rise up seabed until Sv falls below the 2nd threshold
            while (log(np.nanmean(lin(Sv[i-5:i, j]))) > thr[1]) & (i>=5):
                i -= 1
                   
            # subtract range offset & mask all the way down 
            i -= roff
            if i<0:
                i = 0
            mask[i:, j] = True  
    
#    # dilate again to ensure not leaving seabed behind
#    kernel = np.ones((3,3))
#    mask = cv2.dilate(np.uint8(mask), kernel, iterations = 2)
#    mask = np.array(mask, dtype = 'bool')

    return mask

def _get_seabed_mask_ariza(Sv, r, r0=10, r1=1000, roff=0,
          thr=-40, ec=1, ek=(1,3), dc=10, dk=(3,7)):
   
    """
    Mask Sv above a threshold to get potential seabed features. These features
    are eroded first to get rid of fake seabeds (spikes, schools, etc.) and
    dilated afterwards to fill in seabed breaches. Seabed detection is coarser
    than other methods (it removes water nearby the seabed) but the seabed line
    never drops when a breach occurs. Suitable for pelagic assessments and
    reconmended for non-supervised processing.
    
    Args:
        Sv (float): 2D Sv array (dB).
        r (float): 1D range array (m).
        r0 (int): minimum range below which the search will be performed (m). 
        r1 (int): maximum range above which the search will be performed (m).
        roff (int): seabed range offset (m).
        thr (int): Sv threshold above which seabed might occur (dB).
        ec (int): number of erosion cycles.
        ek (int): 2-elements tuple with vertical and horizontal dimensions
                  of the erosion kernel.
        dc (int): number of dilation cycles.
        dk (int): 2-elements tuple with vertical and horizontal dimensions
                  of the dilation kernel.
           
    Returns:
        bool: 2D array with seabed mask.
    """
    
     # raise errors if wrong arguments
    if r0>r1:
        raise Exception('Minimum range has to be shorter than maximum range')
    
    # return empty mask if searching range is outside the echosounder range
    if (r0>r[-1]) or (r1<r[0]):
        return np.zeros_like(Sv, dtype=bool)
    
    # get indexes for range offset and range limits
    r0   = np.nanargmin(abs(r - r0))
    r1   = np.nanargmin(abs(r - r1))
    roff = np.nanargmin(abs(r - roff))
    
    # set to -999 shallow and deep waters (prevents seabed detection)
    Sv_ = Sv.copy()
    Sv_[ 0:r0, :] = -999
    Sv_[r1:  , :] = -999
    
    # return empty mask if there is nothing above threshold
    if not (Sv_>thr).any():
        
        mask = np.zeros_like(Sv_, dtype=bool)
        return mask
    
    # search for seabed otherwise    
    else:
        
        # potential seabed will be everything above the threshold, the rest
        # will be set as -999
        seabed          = Sv_.copy()
        seabed[Sv_<thr] = -999
        
        # run erosion cycles to remove fake seabeds (e.g: spikes, small shoals)
        for i in range(ec):
            seabed = erosion(seabed, np.ones(ek))
        
        # run dilation cycles to fill seabed breaches   
        for i in range(dc):
            seabed = dilation(seabed, np.ones(dk))
        
        # mask as seabed everything greater than -999 
        mask = seabed>-999        
        
        # if seabed occur in a ping...
        idx = np.argmax(mask, axis=0)
        for j, i in enumerate(idx):
            if i != 0:
                
                # ...apply range offset & mask all the way down 
                i -= roff
                if i<0:
                    i = 0
                mask[i:, j] = True 
                
    return mask