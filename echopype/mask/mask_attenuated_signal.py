import pathlib
from typing import Union

import numpy as np
import xarray as xr

from ..utils.mask_transformation import log as _log
from ..utils.mask_transformation import lin as _lin
from ..utils.mask_transformation import full as _full
from ..utils.mask_transformation import twod as _twod

from skimage.measure import label

def get_attenuation_mask(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    mask_type: str = "ryan",
    **kwargs
    
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
        # Define a list of the keyword arguments your function can handle
        valid_args = {'r0', 'r1', 'n','thr', 'start'}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_attenuation_mask_ryan(Sv, r, **filtered_kwargs)
    elif mask_type == "ariza":
        # Define a list of the keyword arguments your function can handle
        valid_args = {'offset', 'thr','m', 'n'}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_attenuation_mask_ariza(Sv, r, **filtered_kwargs)
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
    

def _get_attenuation_mask_ryan(Sv, r, r0=180, r1=280, n=30, thr=6, start=0):
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
    Sv_[Sv_<thr[0]] = np.nan
    # bin Sv
    # TODO: update to 'twod' and 'full' funtions   
    # DID    
    irvals = np.round(np.linspace(p[0],p[-1],num=int((p[-1]-p[0])/n)+1))
    jrvals = np.linspace(r[0],r[-1],num=int((r[-1]-r[0])/m)+1)
    Sv_bnd, p_bnd, r_bnd = _twod(Sv_, p, r, irvals, jrvals, operation="mean")[0:3]
    Sv_bnd = _full(Sv_bnd, p_bnd, r_bnd, p, r)[0]
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
