import pathlib
from typing import List, Optional, Union

import xarray as xr


def frequency_difference(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    storage_options: Optional[dict] = None,
    freqAB: Optional[List[float]] = None,
    chanAB: Optional[List[str]] = None,
    operator: str = ">",
    diff: float = None,
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
    diff:
        The threshold of Sv difference between frequencies

    Returns
    -------
    xr.Dataset
        A Dataset containing the mask. Regions satisfying the thresholding criteria
        are filled with ``True``, else the regions are filled with ``False``.

    Notes
    -----
    TODO: mention it is freqA - freqB operator diff, thus if operator = "<" and diff = "5"
        we would have freqA - freqB < 5


    Either ``freqAB`` or ``chanAB`` must be provided, but both parameters cannot be
    given.

    * `operator` possible options: `>, <, <=, >=, ==`
    * Make sure to put in docs that freqAB has first element as A and second is B
    * return a mask as xarray Dataset (xr.where())

    Examples
    --------

    ep.mask.frequency_difference()


    """

    # TODO: check
    #  - freqAB must be list of float of length 2
    #  - chanAB must be a list of str of length 2
    #  - only freqAB or chanAB must be provided, but not both
    #  - channel and frequency_nominal both exist
    #  -

    if not isinstance(source_Sv, (xr.Dataset, str, pathlib.Path)):
        raise TypeError("source_Sv must be a Dataset or str or pathlib.Path!")

    if not isinstance(storage_options, (dict, type(None))):
        raise TypeError("storage_options must be dict or None!")

    if not isinstance(freqAB, list):
        raise TypeError("freqAB must be a list!")

    if not isinstance(chanAB, list):
        raise TypeError("chanAB must be a list!")

    if not isinstance(operator, str):
        raise TypeError("operator must be a string!")

    if not isinstance(diff, float):
        raise TypeError("diff must be a float!")
