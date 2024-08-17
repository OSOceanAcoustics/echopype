import datetime
import operator as op
import pathlib
from typing import List, Optional, Union

import dask
import dask.array
import numpy as np
import xarray as xr

from ..utils.io import validate_source
from ..utils.prov import add_processing_level, echopype_prov_attrs, insert_input_processing_level
from .freq_diff import _check_freq_diff_source_Sv, _parse_freq_diff_eq

# lookup table with key string operator and value as corresponding Python operator
str2ops = {
    ">": op.gt,
    "<": op.lt,
    "<=": op.le,
    ">=": op.ge,
    "==": op.eq,
}


def _check_mask_dim_alignment(source_ds, mask, var_name):
    """
    Check that mask aligns with source_ds.
    """
    # Grab dimensions of mask
    if isinstance(mask, list):
        mask_dims = set()
        for mask_indiv in mask:
            for dim in mask_indiv.dims:
                mask_dims.add(dim)
    else:
        mask_dims = set(mask.dims)

    # Grab dimensions of the target variable in source ds
    target_variable_dims = set(source_ds[var_name].dims)

    # Raise ValueError if the mask has channel dim but the target variable doesn't
    if "channel" in mask_dims and "channel" not in target_variable_dims:
        raise ValueError("'channel' is a dimension in mask but not a dimension in source.")

    # If non-channel dimensions don't align raise ValueError
    mask_dims.discard("channel")
    target_variable_dims.discard("channel")
    if mask_dims != target_variable_dims:
        raise ValueError(
            f"The dimensions of mask: ({mask_dims}) do not match "
            f"the dimensions of source ({target_variable_dims}) "
            "when not considering 'channel'."
        )

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
            mask_val, file_type = validate_source(mask[mask_ind], storage_options_mask[mask_ind])

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
                ("ping_time", "depth"),
                ("ping_time", "echo_range"),
                ("channel", "ping_time", "range_sample"),
                ("channel", "ping_time", "depth"),
                ("channel", "ping_time", "echo_range"),
            ]
            if mask[mask_ind].dims not in allowed_dims:
                raise ValueError(
                    "Masks must have one of the following dimensions: "
                    "('ping_time', 'range_sample'), "
                    "('ping_time', 'depth'), "
                    "('ping_time', 'echo_range'), "
                    "('channel', 'ping_time', 'range_sample'), "
                    "('channel', 'ping_time', 'depth')"
                    "('channel', 'ping_time', 'echo_range')"
                )

        # Check for the channel dimension consistency
        channel_dim_shapes = set()
        for mask_indiv in mask:
            if "channel" in mask_indiv.dims:
                for mask_chan_ind in range(len(mask_indiv["channel"])):
                    channel_dim_shapes.add(mask_indiv.isel(channel=mask_chan_ind).shape)
        if len(channel_dim_shapes) > 1:
            raise ValueError("All masks must have the same shape in the 'channel' dimension.")

    else:
        if not isinstance(storage_options_mask, dict):
            raise ValueError(
                "The provided input storage_options_mask should be a single "
                "dict because mask is a single value!"
            )

        # validate the mask type or path (if it is provided)
        mask, file_type = validate_source(mask, storage_options_mask)

        if isinstance(mask, (str, pathlib.Path)):
            # open up DataArray using mask path
            mask = xr.open_dataarray(mask, engine=file_type, chunks={}, **storage_options_mask)

    return mask


def _check_var_name_fill_value(
    source_ds: xr.Dataset, var_name: str, fill_value: Union[int, float, xr.DataArray]
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
    fill_value: int, float, or xr.DataArray
        Specifies the value(s) at false indices

    Returns
    -------
    fill_value: int, float, or xr.DataArray
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
    if not isinstance(fill_value, (int, float, xr.DataArray)):
        raise TypeError("The input fill_value must be of type int, float, or xr.DataArray!")

    # make sure that fill_values is the same shape as var_name
    if isinstance(fill_value, xr.DataArray):
        fill_value = fill_value.data.squeeze()  # squeeze out length=1 channel dimension

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
    fill_value: Union[int, float, xr.DataArray] = np.nan,
    storage_options_ds: dict = {},
    storage_options_mask: Union[dict, List[dict]] = {},
) -> xr.Dataset:
    """
    Applies the provided mask(s) to the Sv variable ``var_name``
    in the provided Dataset ``source_ds``.

    The code allows for these 3 cases of `source_ds` and `mask` dimensions:

    1) No channel in both `source_ds` and `mask`,
    but they have matching `ping_time` and
    `depth` (or `range_sample`) dimensions.
    2) `source_ds` and `mask` both have matching `channel`,
    `ping_time`, and `depth` (or `range_sample`) dimensions.
    3) `source_ds` has the channel dimension and `mask` doesn't,
    but they have matching
    `ping_time` and `depth` (or `range_sample`) dimensions.

    If a user only wants to apply masks to a subset of the channels in `source_ds`,
    they could put 1s to allow all data entries in the other channels.

    Parameters
    ----------
    source_ds: xr.Dataset, str, or pathlib.Path
        Points to a Dataset that contains the variable the mask should be applied to
    mask: xr.DataArray, str, pathlib.Path, or a list of these datatypes
        The mask(s) to be applied.
        Can be a individual input or a list that corresponds to a DataArray or a path.
        Each individual input or entry in the list must contain dimensions
        ``('ping_time', 'range_sample')`` or dimensions ``('ping_time', 'depth')``.
        The mask can also contain the dimension ``channel``.
        If a path is provided this should point to a zarr or netcdf file with only
        one data variable in it.
        If the input ``mask`` is a list, a logical AND will be used to produce the final
        mask that will be applied to ``var_name``.
    var_name: str, default="Sv"
        The Sv variable name in ``source_ds`` that the mask should be applied to.
        This variable needs to have coordinates ``('ping_time', 'range_sample')`` or
        coordinates ``('ping_time', 'depth')``, and can optionally also have coordinate
        ``channel``.
        In the case of a multi-channel Sv data variable, the ``mask`` will be broadcast
        to all channels.
    fill_value: int, float, or xr.DataArray, default=np.nan
        Value(s) at masked indices.
        If ``fill_value`` is of type ``xr.DataArray`` it must have the same shape as each
        entry of ``mask``.
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
    # Validate the source_ds type or path (if it is provided)
    source_ds, file_type = validate_source(source_ds, storage_options_ds)

    if isinstance(source_ds, str):
        # open up Dataset using source_ds path
        source_ds = xr.open_dataset(source_ds, engine=file_type, chunks={}, **storage_options_ds)

    # Validate and form the mask input to be used downstream
    mask = _validate_and_collect_mask_input(mask, storage_options_mask)

    # Validate the source_ds and make sure it aligns with the mask input
    source_ds = _check_mask_dim_alignment(source_ds, mask, var_name)

    # Check var_name and sanitize fill_value dimensions if an array
    fill_value = _check_var_name_fill_value(source_ds, var_name, fill_value)

    # Obtain final mask to be applied to var_name
    if isinstance(mask, list):
        # Broadcast all input masks together before combining them
        broadcasted_masks = xr.broadcast(*mask)

        # Perform a logical AND element-wise operation across the masks
        final_mask = np.logical_and.reduce(broadcasted_masks)

        # xr.where has issues with attrs when final_mask is an array, so we make it a DataArray
        final_mask = xr.DataArray(final_mask, coords=broadcasted_masks[0].coords)
    else:
        final_mask = mask

    # Operate on the actual data array to be masked
    source_da = source_ds[var_name]

    # The final_mask should be of the same shape as source_ds[var_name]
    # along the ping_time and range_sample dimensions.
    source_da_chan_shape = (
        source_da.isel(channel=0).shape if "channel" in source_da.dims else source_da.shape
    )
    final_mask_chan_shape = (
        final_mask.isel(channel=0).shape if "channel" in final_mask.dims else final_mask.shape
    )
    if final_mask_chan_shape != source_da_chan_shape:
        raise ValueError(
            f"The final constructed mask is not of the same shape as source_ds[{var_name}] "
            "along the ping_time, and range_sample dimensions!"
        )
    # If final_mask has dim channel then source_da must have dim channel
    if "channel" in final_mask.dims and "channel" not in source_da.dims:
        raise ValueError(
            "The final constructed mask has the channel dimension, "
            f"so source_ds[{var_name}] must also have the channel dimension."
        )
    # If final_mask and source_da both have channel dimension, then they must
    # have the same number of channels.
    elif "channel" in final_mask.dims and "channel" in source_da.dims:
        if len(final_mask["channel"]) != len(source_da["channel"]):
            raise ValueError(
                f"If both the final constructed mask and source_ds[{var_name}] "
                "have the channel dimension, that dimension should match between the two."
            )

    # Turn NaN in final_mask to False, otherwise xr.where treats as True
    final_mask = final_mask.fillna(False)

    # Apply the mask to var_name
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

    # Use the original dimension order
    output_ds[var_name] = output_ds[var_name].transpose(*source_da.dims)

    # Attribute handling
    process_type = "mask"
    prov_dict = echopype_prov_attrs(process_type=process_type)
    prov_dict[f"{process_type}_function"] = "mask.apply_mask"
    output_ds = output_ds.assign_attrs(prov_dict)
    output_ds = insert_input_processing_level(output_ds, input_ds=source_ds)

    return output_ds


def frequency_differencing(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    storage_options: Optional[dict] = {},
    freqABEq: Optional[str] = None,
    chanABEq: Optional[str] = None,
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
    freqABEq: string, optional
        The frequency differencing criteria.
        Only one of ``freqAB`` and ``chanAB`` should be provided, and not both.
    chanAB: string, optional
        The frequency differencing criteria in terms of channel names where channel names
        in the criteria are enclosed in double quotes. Only one of ``freqAB`` and ``chanAB``
        should be provided, and not both.

    Returns
    -------
    xr.DataArray
        A DataArray containing the mask for the Sv data. Regions satisfying the thresholding
        criteria are filled with ``True``, else the regions are filled with ``False``.

    Raises
    ------
    ValueError
        If neither ``freqABEq`` or ``chanABEq`` are given
    ValueError
        If both ``freqABEq`` and ``chanABEq`` are given
    TypeError
        If any input is not of the correct type
    ValueError
        If either ``freqABEq`` or ``chanABEq`` are provided and the extracted
        ``freqAB`` or ``chanAB`` does not contain 2 distinct elements
    ValueError
        If ``freqABEq`` contains values that are not contained in ``frequency_nominal``
    ValueError
        If ``chanABEq`` contains values that not contained in ``channel``
    ValueError
        If ``operator`` is not one of the following: ``">", "<", "<=", ">=", "=="``
    ValueError
        If the path provided for ``source_Sv`` is not a valid path
    ValueError
        If ``freqABEq`` or ``chanABEq`` is provided and the Dataset produced by ``source_Sv``
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
    >>> echopype.mask.frequency_differencing(source_Sv=Sv_ds, storage_options={},
    ...                                      freqABEq=None, chanABEq = '"chan1" - "chan2">=10.0dB')
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
    # _check_freq_diff_non_data_inputs(freqAB, chanAB, operator, diff)
    freqAB, chanAB, operator, diff = _parse_freq_diff_eq(freqABEq, chanABEq)

    # validate the source_Sv type or path (if it is provided)
    source_Sv, file_type = validate_source(source_Sv, storage_options)

    if isinstance(source_Sv, str):
        # open up Dataset using source_Sv path
        source_Sv = xr.open_dataset(source_Sv, engine=file_type, chunks={}, **storage_options)

    # check the source_Sv with respect to channel and frequency_nominal
    _check_freq_diff_source_Sv(source_Sv, freqAB, chanAB)

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

    def _get_lhs(
        Sv_block: np.ndarray, chanA_idx: int, chanB_idx: int, chan_dim_idx: int = 0
    ) -> np.ndarray:
        """Get left-hand side of condition"""

        def _sel_channel(chan_idx):
            return tuple(
                [chan_idx if i == chan_dim_idx else slice(None) for i in range(Sv_block.ndim)]
            )

        # get the left-hand side of condition (lhs)
        return Sv_block[_sel_channel(chanA_idx)] - Sv_block[_sel_channel(chanB_idx)]

    def _create_mask(lhs: np.ndarray, diff: float) -> np.ndarray:
        """Create mask using operator lookup table"""
        return xr.where(str2ops[operator](lhs, diff), True, False)

    # Get the Sv data array
    Sv_data_array = source_Sv["Sv"]

    # Determine channel index based on names
    channels = list(source_Sv["channel"].to_numpy())
    chanA_idx = channels.index(chanA)
    chanB_idx = channels.index(chanB)
    # Get the channel dimension index for filtering
    chan_dim_idx = Sv_data_array.dims.index("channel")

    # If Sv data is not dask array
    if not isinstance(Sv_data_array.variable._data, dask.array.Array):
        # get the left-hand side of condition
        lhs = _get_lhs(Sv_data_array, chanA_idx, chanB_idx, chan_dim_idx=chan_dim_idx)

        # create mask using operator lookup table
        da = _create_mask(lhs, diff)
    # If Sv data is dask array
    else:
        # Get the final data array template
        template = Sv_data_array.isel(channel=0).drop_vars("channel")

        dask_array_data = Sv_data_array.data
        # Perform block wise computation
        dask_array_result = (
            dask_array_data
            # Compute the left-hand side of condition
            # drop the first axis (channel) as it is dropped in the result
            .map_blocks(
                _get_lhs,
                chanA_idx,
                chanB_idx,
                chan_dim_idx=chan_dim_idx,
                dtype=dask_array_data.dtype,
                drop_axis=0,
            )
            # create mask using operator lookup table
            .map_blocks(_create_mask, diff)
        )

        # Create DataArray of the result
        da = xr.DataArray(
            data=dask_array_result,
            coords=template.coords,
        )

    xr_dataarray_attrs = {
        "mask_type": "frequency differencing",
        "history": f"{datetime.datetime.utcnow()} +00:00. "
        "Mask created by mask.frequency_differencing. "
        f"Operation: Sv['{chanA}'] - Sv['{chanB}'] {operator} {diff}",
    }

    # assign a name to DataArray
    da.name = "mask"

    # assign provenance attributes
    da = da.assign_attrs(xr_dataarray_attrs)

    return da
