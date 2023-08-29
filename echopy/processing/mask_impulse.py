#!/usr/bin/env python3
"""
Algorithms for masking Impulse noise.
    
Copyright (c) 2020 Echopy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__authors__ = ['Alejandro Ariza'   # wrote ryan(), ryan_iterable(), and wang()
               ]
__credits__ = ['Rob Blackwell'     # supervised the code and provided ideas
               'Sophie Fielding'   # supervised the code and provided ideas               
               ]

import numpy as np
from skimage.morphology import erosion
from skimage.morphology import dilation
from scipy.ndimage.filters import median_filter as medianf
from echopy.processing import resample as rs
from echopy.utils import transform as tf

def ryan(Sv, iax, m, n=1, thr=10):
    """
    Mask impulse noise following the two-sided comparison method described
    in:        
        Ryan et al. (2015) ‘Reducing bias due to noise and attenuation in 
        open-ocean echo integration data’, ICES Journal of Marine Science,
        72: 2482–2493.

    Args:
        Sv  (float)    : 2D array with Sv data to be masked (dB).
        iax (int/float): 1D array with i axis data (n samples or range).
        m   (int/float): vertical binning length (n samples or range).
        n   (int)      : number of pings either side for comparisons.
        thr (int/float): user-defined threshold value (dB).
        
    Returns:
        bool: 2D array with IN mask
        bool: 2D array with mask indicating valid IN mask samples.  
    """    
        
    # resample down vertically    
    iax_ = np.arange(iax[0], iax[-1], m)
    Sv_  = rs.oned(Sv, iax, iax_, 0, log=True)[0]
    
    # resample back to full resolution
    jax = np.arange(len(Sv[0]))
    Sv_, mask_ = rs.full(Sv_, iax_, jax, iax, jax)
        
    #side comparison (±n)
    dummy               = np.zeros((iax.shape[0], n))*np.nan     
    comparison_forward  = Sv_ - np.c_[Sv_[:, n:], dummy]
    comparison_backward = Sv_ - np.c_[dummy, Sv_[:, 0:-n]]
    
    # get IN mask            
    comparison_forward[np.isnan(comparison_forward)] = np.inf 
    maskf = comparison_forward>thr               
    comparison_backward[np.isnan(comparison_backward)] = np.inf 
    maskb = comparison_backward>thr
    mask  = maskf & maskb   
    
    # get second mask indicating valid samples in IN mask    
    mask_[:, 0:n] = False
    mask_[:, -n:] = False    
    
    return mask, mask_

def ryan_iterable(Sv, iax, m, n=(1,2), thr=10):
    """
    Modified from "ryan" so that the parameter "n" can be provided multiple
    times. It enables the algorithm to iterate and perform comparisons at
    different n distances. Resulting masks at each iteration are combined in
    a single mask. By setting multiple n distances the algorithm can detect 
    spikes adjacent each other.
    
    Args:
        Sv  (float)     : 2D array with Sv data to be masked (dB).
        iax (int, float): 1D array with i axis data (n samples or range).
        m   (int, float): vertical binning length (n samples or range).
        n   (int)       : number of pings either side for comparisons.
        thr (int,float) : user-defined threshold value (dB).
        
    Returns:
        bool: 2D array with IN mask
        bool: 2D array with mask indicating valid IN mask samples.
    """    
        
    # resample down vertically    
    iax_ = np.arange(iax[0], iax[-1], m)
    Sv_  = rs.oned(Sv, iax, iax_, 0, log=True)[0]
    
    # resample back to full resolution
    jax = np.arange(len(Sv[0]))
    Sv_, mask_ = rs.full(Sv_, iax_, jax, iax, jax)
    
    # perform side comparisons and combine masks in one unique mask
    mask = np.zeros_like(Sv, dtype=bool)
    for i in n:
        dummy    = np.zeros((iax.shape[0], i)); dummy[:] = np.nan     
        forward  = Sv_ - np.c_[Sv_[:,i:], dummy]
        backward = Sv_ - np.c_[dummy, Sv_[:, 0:-i]]
        maskf    = np.ma.masked_greater(forward, thr).mask
        maskb    = np.ma.masked_greater(backward, thr).mask
        mask     = mask | (maskf&maskb)
    
    # get second mask indicating valid samples in IN mask    
    mask_[:, 0:max(n)] = True
    mask_[:, -max(n):] = True  
    
    return mask, mask_

def wang(Sv, thr=(-70,-40), erode=[(3,3)], dilate=[(5,5),(7,7)],
         median=[(7,7)]):
    """
    Clean impulse noise from Sv data following the method decribed by:
        
        Wang et al. (2015) ’A noise removal algorithm for acoustic data with
        strong interference based on post-processing techniques’, CCAMLR
        SG-ASAM: 15/02.
        
    This algorithm runs different cycles of erosion, dilation, and median
    filtering to clean impulse noise from Sv. Note that this function
    returns a clean/corrected Sv array, instead of a boolean array indicating
    the occurrence of impulse noise.
        
    Args:
        Sv     (float)    : 2D numpy array with Sv data (dB).
        thr    (int/float): 2-element tupple with bottom/top Sv thresholds (dB)
        erode  (int)      : list of 2-element tupples indicating the window's
                            size for each erosion cycle.
        dilate (int)      : list of 2-element tupples indicating the window's
                            size for each dilation cycle.
        median (int)      : list of 2-element tupples indicating the window's
                            size for each median filter cycle.
                      
    Returns:
        float             : 2D array with clean Sv data.
        bool              : 2D array with mask indicating valid clean Sv data.
    """

    # set weak noise and strong interference as vacant samples (-999)
    Sv_thresholded                          = Sv.copy()
    Sv_thresholded[(Sv<thr[0])|(Sv>thr[1])] = -999
    
    # remaining weak interferences will take neighbouring vacant values 
    # by running erosion cycles
    Sv_eroded = Sv.copy()
    for e in erode:
        Sv_eroded = erosion(Sv_thresholded, np.ones(e))
    
    # the last step might have turned interferences inside biology into vacant
    # samples, this is solved by running dilation cycles
    Sv_dilated = Sv_eroded.copy()
    for d in dilate:
        Sv_dilated = dilation(Sv_dilated, np.ones(d))
    
    # dilation has modified the Sv value of biological features, so these are
    # now corrected to corresponding Sv values before the erosion/dilation
    Sv_corrected1           = Sv_dilated.copy()
    mask_bio                = (Sv_dilated>=thr[0]) & (Sv_dilated<thr[1])
    Sv_corrected1[mask_bio] = Sv_thresholded[mask_bio]
    
    # compute median convolution in Sv corrected array
    Sv_median = Sv_corrected1.copy()
    for m in median:
        Sv_median = tf.log(medianf(tf.lin(Sv_median), footprint=np.ones(m)))
    
    # any vacant sample inside biological features will be corrected with
    # the median of corresponding neighbouring samples      
    Sv_corrected2                       = Sv_corrected1.copy()
    mask_bio                            = (Sv>=thr[0]) & (Sv<thr[1])
    mask_vacant                         = Sv_corrected1==-999
    Sv_corrected2[mask_vacant&mask_bio] = Sv_median[mask_vacant&mask_bio]
    
    # get mask indicating edges, where swarms analysis couldn't be performed 
    mask_ = np.ones_like(Sv_corrected2, dtype=bool)
    idx   = int((max([e[0], d[0]])-1)/2)
    jdx   = int((max([e[1], d[1]])-1)/2)
    mask_[idx:-idx, jdx:-jdx] = False
    
    return Sv_corrected2, mask_

def other():
    """
    Note to contributors:
        Other algorithms for masking impulse noise must be named with the
        author or method name. If already published, the full citation must be
        provided. Please, add "unpub." otherwise. E.g: Smith et al. (unpub.)
        
        Please, check contribute.md to follow our coding and documenting style.
    """