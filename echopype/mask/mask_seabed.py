import datetime
import pathlib
import operator as op
import numpy as np
import xarray as xr
from typing import List, Optional, Union

import scipy.ndimage as nd
from scipy.interpolate import interp1d
from scipy.signal import convolve2d

from skimage.morphology import remove_small_objects
from skimage.morphology import erosion,dilation, square
from skimage.measure import label

from ..utils.io import validate_source_ds_da
from ..utils.prov import add_processing_level, echopype_prov_attrs, insert_input_processing_level
from ..utils.mask_tranformation import log,lin,dim2ax,full,twod

def get_seabed_mask(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    mask_type: str = "ariza",
    **kwargs
    
) -> xr.DataArray:
    """
    Create a mask based on the identified signal attenuations of Sv values at 38KHz. 
    This method is based on:
    XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    Parameters
    ----------
    source_Sv: xr.Dataset or str or pathlib.Path
        If a Dataset this value contains the Sv data to create a mask for,
        else it specifies the path to a zarr or netcdf file containing
        a Dataset. This input must correspond to a Dataset that has the
        coordinate ``channel`` and variables ``frequency_nominal`` and ``Sv``.
    mask_type: str with either "ariza", "experimental", "blackwell_mod", "blackwell", "deltaSv", "maxSv" based on the prefered method for signal attenuation mask generation
    Returns
    -------
    xr.DataArray
        A DataArray containing the mask for the Sv data. Regions satisfying the thresholding
        criteria are filled with ``True``, else the regions are filled with ``False``.

    Raises
    ------
    ValueError
        If neither "ariza", "experimental", "blackwell_mod", "blackwell", "deltaSv", "maxSv" are given

    Notes
    -----


    Examples
    --------

    """
    assert mask_type in ["ariza", "experimental", "blackwell_mod", "blackwell", "deltaSv", "maxSv"], "mask_type must be either 'ariza', 'experimental', 'blackwell', 'maxSv', 'deltaSv'"
    
    Sv = source_Sv['Sv'].values[0].T
    r = source_Sv['echo_range'].values[0,0]
    if mask_type == "ariza":
        # Define a list of the keyword arguments your function can handle
        valid_args = {'r0', 'r1', 'roff','thr', 'ec', 'ek', 'dc', 'dk'}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_seabed_mask_ariza(Sv, r, **filtered_kwargs)
    elif mask_type == "experimental":
        # Define a list of the keyword arguments your function can handle
        valid_args = {'r0', 'r1', 'roff', 'thr', 'ns', 'nd'}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_seabed_mask_experimental(Sv, r, **filtered_kwargs)
    elif mask_type == "blackwell":
        # Define a list of the keyword arguments your function can handle
        valid_args = {'theta', 'phi','r0', 'r1', 'tSv', 'ttheta', 'tphi', 'wtheta' , 'wphi'}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_seabed_mask_blackwell(Sv, r, **filtered_kwargs)
    elif mask_type == "blackwell_mod":
        # Define a list of the keyword arguments your function can handle
        valid_args = {'theta', 'phi','r0', 'r1', 'tSv', 'ttheta', 'tphi', 'wtheta' , 'wphi',
                      'rlog', 'tpi', 'freq', 'rank'}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_seabed_mask_blackwell_mod(Sv, r, **filtered_kwargs)
    elif mask_type == "deltaSv":
        # Define a list of the keyword arguments your function can handle
        valid_args = {'r0', 'r1', 'roff','thr'}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_seabed_mask_deltaSv(Sv, r, **filtered_kwargs)
    elif mask_type == "maxSv":
        # Define a list of the keyword arguments your function can handle
        valid_args = {'r0', 'r1', 'roff','thr'}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_seabed_mask_maxSv(Sv, r, **filtered_kwargs)
    else:
        raise ValueError(
                "The provided mask_type must be 'ariza', 'experimental', 'blackwell', 'maxSv' or 'deltaSv'!"
                )
                
    mask = np.logical_not(mask.T)
    return_mask = xr.DataArray(mask,
                                    dims=("ping_time", 
                                          "range_sample"),
                                    coords={"ping_time": source_Sv.ping_time,
                                            "range_sample": source_Sv.range_sample}
                                   )
    return return_mask


def _get_seabed_mask_maxSv(Sv, r, r0=10, r1=1000, roff=0, thr=(-40, -60)):
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
                Svmean = log(np.nanmean(lin(Sv[i-5:i, j])))
            
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

def _get_seabed_mask_blackwell(Sv, r, theta=None, phi=None,
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

    return mask#, anglemask

def _get_seabed_mask_blackwell_mod(Sv, r, theta=None, phi=None, r0=10, r1=1000, tSv=-75, ttheta=702, 
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
            rseabed = aliased2seabed(ritem , rlog, tpi, freq)
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

def aliased2seabed(aliased, rlog, tpi, f, c=1500, 
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

def seabed2aliased(seabed, rlog, tpi, f, c=1500, 
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

def _get_seabed_mask_experimental(Sv, r,
                 r0=10, r1=1000, roff=0, thr=(-30,-70), ns=150, nd=3):
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
    #mask = cv2.dilate(np.uint8(mask), kernel, iterations=nd)
    mask = dilation(np.uint8(mask), square(nd))
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