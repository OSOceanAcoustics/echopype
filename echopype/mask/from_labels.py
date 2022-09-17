import xarray as xr


def boundary(da: xr.DataArray, label) -> xr.DataArray:
    """
    Create a mask based on a boundary label.

    The label is usually a line denoting the seafloor or the sea surface,
    or such lines modified to exclude specific features in the data (e.g., bubbles).

    Parameters
    ----------
    da : xr.DataArray
        A DataArray containing the Sv data to create a mask for
    label : echoregions
        An echoregions line object

    Returns
    -------
    A DataArray containing the mask.
    Areas to be included are filled with 1s, and areas to be excluded are filled with 0s.
    """
    pass


def region(da: xr.DataArray, label) -> xr.DataArray:
    """
    Create a mask based on region labels.

    The labels are usually polygons marking the presence of a certain type of organism.

    Parameters
    ----------
    da : xr.DataArray
        A DataArray containing the Sv data to create a mask for
    label : echoregions
        An echoregions region object

    Returns
    -------
    A DataArray containing the mask.
    Areas included in the regions are filled with 1s, and areas to be excluded are filled with 0s.
    """
    pass
