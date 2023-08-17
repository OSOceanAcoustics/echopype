import pathlib
from typing import List, Optional, Union

import numpy as np
import xarray as xr

from ..utils.mask_transformation import log as _log
from ..utils.mask_transformation import lin as _lin

def get_transient_noise_mask(
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

    Parameters
    ----------
    source_Sv: xr.Dataset or str or pathlib.Path
        If a Dataset this value contains the Sv data to create a mask for,
        else it specifies the path to a zarr or netcdf file containing
        a Dataset. This input must correspond to a Dataset that has the
        coordinate ``channel`` and variables ``frequency_nominal`` and ``Sv``.
    mask_type: str with either "ryan" or "fielding" based on the prefered method for signal attenuation mask generation
    Returns
    -------
    xr.DataArray
        A DataArray containing the mask for the Sv data. Regions satisfying the thresholding
        criteria are filled with ``True``, else the regions are filled with ``False``.

    Raises
    ------
    ValueError
        If neither ``ryan`` or ``fielding`` are given

    Notes
    -----


    Examples
    --------

    """
    assert mask_type in ['ryan', 'fielding'], "mask_type must be either 'ryan' or 'fielding'"
    
    Sv = source_Sv['Sv'].values[0]
    r = source_Sv['echo_range'].values[0,0]
    if mask_type == "ryan":
        # Define a list of the keyword arguments your function can handle
        valid_args = {'m', 'n', 'thr','excludeabove', 'operation'}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_transient_noise_mask_ryan(Sv, r, m=5, **filtered_kwargs)
    elif mask_type == "fielding":
        #Define a list of the keyword arguments your function can handle
        valid_args = {'r0', 'r1','roff','n','thr'}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_transient_noise_mask_fielding(Sv, r, **filtered_kwargs )
    else:
        raise ValueError(
                "The provided mask_type must be ryan or fielding!"
                )
                
    mask = np.logical_not(mask)
    return_mask = xr.DataArray(mask,
                                    dims=("ping_time", 
                                          "range_sample"),
                                    coords={"ping_time": source_Sv.ping_time,
                                            "range_sample": source_Sv.range_sample}
                                   )
    return return_mask
    
def _get_transient_noise_mask_ryan(Sv, r, m=50, n=20, thr=20,
         excludeabove=250, operation='percentile15'):
    """
    Mask transient noise as in:
        
        Ryan et al. (2015) ‘Reducing bias due to noise and attenuation in 
        open-ocean echo integration data’, ICES Journal of Marine Science,
        72: 2482–2493.
    
    This mask is based on the assumption that Sv values which exceed the median
    value in a surrounding region of m metres by n pings must be due to 
    transient noise. Sv values are removed if exceed a threshold. Masking is
    excluded above 250 m by default to avoid the removal of aggregated biota.

    Args:
        Sv (float): 2D numpy array with Sv data to be masked (dB) 
        r (float): 1D numpy array with range data (m)
        m (int): height of surrounding region (m) 
        n (int): width of surrounding region (pings)
        threshold (int): user-defined threshold for comparisons (dB)
        excludeabove (int): range above which masking is excluded (m)
        operation (str): type of average operation:
            'mean'
            'percentileXX'
            'median'
            'mode'#not in numpy
        
    Returns:
        bool: 2D numpy array mask (transient noise = True) 
    """
    # offsets for i and j indexes 
    ioff = n
    joff = np.argmin(abs(r - m))
    
    
    # preclude processing above a user-defined range
    r0 = np.argmin(abs(r - excludeabove))    

    # mask if Sv sample greater than averaged block
    # TODO: find out a faster method. The iteration below is too slow.
    mask = np.ones(Sv.shape, dtype = bool)
    mask[:, 0:r0] = False    
    
    
    for i in range(len(Sv)):
        
        for j in range(r0, len(Sv[0])):
            # proceed only if enough room for setting the block
            if (i-ioff >= 0) & (i+ioff < len(Sv)) & (j-joff >= 0) & (j+joff < len(Sv[0])):               
                sample = Sv[i, j]
                if operation == "mean":
                    block = _log(np.nanmean(_lin(Sv[i-ioff : i+ioff,j-joff : j+joff])))   
                elif operation == "median":
                    block = _log(np.nanmedian(_lin(Sv[i-ioff : i+ioff,j-joff : j+joff])))   
                else:
                    block = _log(np.nanpercentile(_lin(Sv[i-ioff : i+ioff,j-joff : j+joff]), int(operation[-2:])))           
                mask[i, j] = sample - block > thr
    
    
    return mask

def _get_transient_noise_mask_fielding(Sv, r, r0=200, r1=1000, n=20, thr=[2,0], roff=250, jumps=5, maxts=-35, start=0):
    """
    Mask transient noise with method proposed by Fielding et al (unpub.).
    
    A comparison is made ping by ping with respect to a block in a reference 
    layer set at far range, where transient noise mostly occurs. If the ping 
    median is greater than the block median by a user-defined threshold, the 
    ping will be masked all the way up, until transient noise dissapears, or 
    until it gets the minimum range allowed by the user.
    
       transient                 transient             ping      
         noise                     noise             evaluated   
           |                         |                  |        
    ______ | _______________________ | ____________.....V.....____________
          |||  far range interval   |||            .  block  .            |
    _____|||||_____________________|||||___________...........____________|
    
    When transient noise is detected, comparisons start to be made in the same 
    ping but moving vertically every x meters (jumps). Pings with transient
    noise will be masked up to where the ping is similar to the block according
    with a secondary threshold or until it gets the exclusion range depth.
    
    Args:
        Sv    (float): 2D numpy array with Sv data to be masked (dB).
        r     (float): 1D numpy array with range data (m).
        r0    (int  ): range below which transient noise is evaluated (m).
        r1    (int  ): range above which transient noise is evaluated (m).
        n     (int  ): n of preceeding & subsequent pings defining the block.
        thr   (int  ): user-defined threshold for side-comparisons (dB).
        roff  (int  ): range above which masking is excluded (m).
        maxts (int  ): max transient noise permited, prevents to interpret 
                       seabed as transient noise (dB). 
        jumps (int  ): height of vertical steps (m).
        start (int  ): ping index to start processing.
        
    Returns:
        list: 2D boolean array with TN mask and 2D boolean array with mask
              indicating where TN detection was unfeasible.
    """ 

     # raise errors if wrong arguments
    if r0>r1:
        raise Exception('Minimum range has to be shorter than maximum range')
    
    # return empty mask if searching range is outside the echosounder range
    if (r0>r[-1]) or (r1<r[0]):
        mask  = np.zeros_like(Sv, dtype=bool) 
        mask_ = np.zeros_like(Sv, dtype=bool) 
        return mask, mask_              
        
    # get upper and lower range indexes   
    up = np.argmin(abs(r - r0))
    lw = np.argmin(abs(r - r1))
    
    # get minimum range index admitted for processing
    rmin = np.argmin(abs(r - roff))
    # get scaling factor index
    sf = np.argmin(abs(r - jumps))
    # start masking process
    mask_ = np.zeros(Sv.shape, dtype=bool)
    mask  = np.zeros(Sv.shape, dtype=bool)   
    for j in range(start, len(Sv)):
        
        # mask where TN evaluation is unfeasible (e.g. edge issues, all-NANs) 
        if (j-n<0) | (j+n>len(Sv)-1) | np.all(np.isnan(Sv[j, up:lw])):        
            mask_[j, :] = True
        # evaluate ping and block averages otherwise
        
        else:
            
            pingmedian  = _log(np.nanmedian(_lin(Sv[j,up:lw])))
            pingp75     = _log(np.nanpercentile(_lin(Sv[j,up:lw]), 75))
            blockmedian = _log(np.nanmedian(_lin(Sv[j-n:j+n, up:lw])))
            
            # if ping median below 'maxts' permited, and above enough from the
            # block median, mask all the way up until noise dissapears
            if (pingp75<maxts) & ((pingmedian-blockmedian)>thr[0]): 
                r0, r1 = lw-sf, lw
                while r0>rmin:
                    pingmedian = _log(np.nanmedian(_lin(Sv[j,r0:r1])))
                    blockmedian= _log(np.nanmedian(_lin(Sv[j-n:j+n,r0:r1])))
                    r0, r1 = r0-sf, r1-sf
                    if (pingmedian-blockmedian)<thr[1]:
                        break 
                mask[j, r0:] = True

    final_mask =  mask[:, start:] | mask_[:, start:]         
    return final_mask
