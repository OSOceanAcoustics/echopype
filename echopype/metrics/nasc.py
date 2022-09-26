from typing import Tuple, Union

import xarray as xr


def _get_cell_size(da: xr.DataArray, cell_height: str, cell_width: str) -> Tuple:
    """
    Get the number of elements in height and width for each NASC cell.

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing Sv from a single channel or frequency.
    cell_height : str
        Height of NASC cell in meters (e.g., "10m") or number of pixels (e.g., "100px")
    cell_width : str
        Width of NASC cell in meters (e.g., "10m") or number of pixels (e.g., "100px")

    Returns
    -------
    A tuple containing the number of elements in height and width for each NASC cell.
    """
    pass


def _PRC_NASC(da: xr.DataArray, mask_all: xr.DataArray, cell_height_px: int, cell_width_px: int):
    """
    Compute NASC using the Echoview PRC_NASC routine.

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing Sv from a single channel or frequency.
    mask_all : xr.DataArray
        DataArray containing 1s for included regions and 0s for excluded regions
    cell_height_px : int
        Height of an output NASC cell in terms of the number of elements in ``da``
    cell_width_px : int
        Width of an output NASC cell in terms of the number of elements in ``da``

    Returns
    -------
    An xarray DataArray containing the final NASC.
    """
    # Check if dimensions of da and mask_all are the same

    # Construct NASC output

    pass


def compute_NASC(
    ds_Sv: xr.Dataset,
    masks: Tuple,
    cell_height: str,
    cell_width: str,
    channel: Union[str, None] = None,
    frequency: Union[float, int, None] = None,
) -> xr.DataArray:
    """
    Export the Nautical Area Scattering Coefficient (NASC) by combining Sv data and masks.

    NASC is calculated using the Echoview PRC_NASC routine, where PRC stands for
    "Proportion that the Region contributes to Cell."
    See https://support.echoview.com/WebHelp/Reference/Algorithms/Analysis_variables/PRC_ABC_and_PRC_NASC.htm

    Parameters
    ----------
    ds_Sv : xr.Dataset
        Dataset with Sv
    masks : tuple of masks
        A tuple of masks (xarray DataArrays of the same dimension as the Sv data
        filled with 1s and 0s) to be used to generate NASC.
        Elements in the masks with values outside of 0s and 1s will be converted to 1s.
        Element-wise AND operations will be used to combine all masks.
    channel : str
        The channel to use for computing NASC.
        One of ``channel`` or ``frequency`` must be specified.
        An warning will be issued if both ``chanenl`` and ``frequency`` are specified.
        ``channel`` takes priority over ``frequency`` if both are specified.
    frequency : int, float
        The frequency to use for computing NASC.

    Returns
    -------
    xr.DataArray
    """  # noqa

    # Check masks
    for m in masks:
        # if m contains values outside of 0s and 1s:
        #     convert all non-0 values to 1
        continue

    # Assemble overall mask
    # mask_all = element-wise AND operations across all masks
    mask_all = masks  # placeholder

    # Select channel
    if channel is not None:
        if frequency is not None:
            # raise warning saying that the specified channel will take priority over frequency
            pass
        da = ds_Sv.sel(channel=channel)
    else:
        if frequency is None:
            raise ValueError("One of channel or frequency must be specified!")
        else:
            # make frequency_nominal a coordinate
            da = ds_Sv.sel(frequency_nominal=frequency)

    # Get cell height and width in element unit
    cell_height_px, cell_width_px = _get_cell_size(da, cell_height, cell_width)

    # Compute NASC
    # - element-wise AND operations between m_all and da
    # - implement Echoview PRC_NASC routine
    nasc = _PRC_NASC(da, mask_all=mask_all, cell_height_px=cell_height, cell_width_px=cell_width_px)

    return nasc
