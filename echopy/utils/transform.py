#!/usr/bin/env python3
"""
Algorithms for tranformation of acoustic units.
    
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

__authors__ = ['Alejandro Ariza'   # wrote lin(), log(), Sv2sa(), Sv2NASC(),
               ]                   # pos2dis(), and dis2speed()

__credits__ = ['Rob Blackwell'     # supervised the code and provided ideas
               'Sophie Fielding'   # supervised the code and provided ideas               
               ]

import numpy as np
from echopy.processing import resample as rs
from geopy.distance import distance

def lin(variable):
    """
    Turn variable into the linear domain.     
    
    Args:
        variable (float): array of elements to be transformed.    
    
    Returns:
        float:array of elements transformed
    """
    
    lin    = 10**(variable/10)
    return   lin  

def log(variable):
    """
    Turn variable into the logarithmic domain. This function will return -999
    in the case of values less or equal to zero (undefined logarithm). -999 is
    the convention for empty water or vacant sample in fisheries acoustics. 
    
    Args:
        variable (float): array of elements to be transformed.    
    
    Returns:
        float: array of elements transformed
    """    
    if not isinstance(variable, (np.ndarray)):
        variable = np.array([variable])
        
    if isinstance(variable, int):
        variable = np.float64(variable)
        
    mask           = np.ma.masked_less_equal(variable, 0).mask
    variable[mask] = np.nan
    log            = 10*np.log10(variable)
    log[mask]      = -999
    return           log

def Sv2sa(Sv, r, r0, r1, operation='mean'):
    """
    Compute Area backscattering coefficient (m2 m-2), by integrating Sv in a
    given range interval.
    
    Args:
        Sv (float)    : 2D array with Sv data (dB m-1)
        r  (float)    : 1D array with range data (m)
        r0 (int/float): Top range limit (m)
        r1 (int/float): Bottom range limit (m)
        method (str)  : Method for calculating sa. Accepts "mean" or "sum".
        
    Returns:
        float: 1D array with area backscattering coefficient data.
        float: 1D array with the percentage of vertical samples integrated.
    """
       
    if operation=='mean':
        svmean, pc, m_ = rs.oned(lin(Sv), r, np.array([r0, r1]),
                                 operation='mean')
        integration_range = r1-r0
        sa = svmean*integration_range
        
    elif operation=='sum':
        svsum, pc, m_ = rs.oned(lin(Sv), r, np.array([r0, r1]),
                                operation='sum')
        sample_height = (r[-1]-r[0])/len(r)
        sa = svsum*sample_height
    
    return sa, pc, m_

def Sv2NASC(Sv, r, r0, r1, operation='mean'):
    """
    Compute Nautical Area Scattering Soefficient (m2 nmi-2), by integrating Sv
    in a given range interval.
    
    Args:
        Sv (float)    : 2D array with Sv data (dB m-1)
        r  (float)    : 1D array with range data (m)
        r0 (int/float): Top range limit (m)
        r1 (int/float): Bottom range limit (m)
        method (str)  : Method for calculating sa. Accepts "mean" or "sum"
        
    Returns:
        float: 1D array with Nautical Area Scattering Coefficient data.
        float: 1D array with the percentage of vertical samples integrated.
    """
   
    # get r0 and r1 indexes
    r0 = np.argmin(abs(r-r0))
    r1 = np.argmin(abs(r-r1))
    
    # get number and height of samples 
    ns     = len(r[r0:r1])
    sh = np.r_[np.diff(r), np.nan]
    sh = np.tile(sh.reshape(-1,1), (1,len(Sv[0])))[r0:r1,:]
    
    # compute NASC    
    sv = lin(Sv[r0:r1, :])
    if operation=='mean':    
        NASC = np.nanmean(sv * sh, axis=0) * ns * 4*np.pi*1852**2
    elif operation=='sum':
        NASC = np.nansum (sv * sh, axis=0)      * 4*np.pi*1852**2
    else:
        raise Exception('Method not recognised')
    
    # compute percentage of valid values (not NAN) behind every NASC integration    
    per = (len(sv) - np.sum(np.isnan(sv*sh), axis=0)) / len(sv) * 100
    
    # correct sa with the proportion of valid values
    NASC = NASC/(per/100)
        
    return NASC, per

def pos2dis(lon, lat, units='nm'):
    """
    Return cumulated distance from longitude and latitude positions, in
    nautical miles or kilometres.
    
    Args:
        lon (float): 1D array with longitude data (decimal degrees)
        lat (float): 1D array with latitude data (decimal degrees)
        units (str): distance unit to return, accepts 'nm' and 'km'
                     
    Returns
        float: 1D array with cumulated distance 
    """

    # calculate distance
    dis = np.zeros(len(lon))*np.nan
    
    if units=='nm':
        for i in range(len(dis)-1):
            if np.isnan(lat[i]) | np.isnan(lon[i]) | np.isnan(lat[i+1]) | np.isnan(lon[i+1]):
                dis[i] = np.nan
            else:
                dis[i] = distance((lat[i], lon[i]), (lat[i+1], lon[i+1])).nm
    elif units=='km':
        for i in range(len(dis)-1):
            if np.isnan(lat[i]) | np.isnan(lon[i]) | np.isnan(lat[i+1]) | np.isnan(lon[i+1]):
                dis[i] = np.nan
            else:
                dis[i] = distance((lat[i], lon[i]), (lat[i+1], lon[i+1])).km
    else:
        raise Exception('Units not recognised')
    
    # calculate cumulated distance
    cumdis                = np.nancumsum(dis)
    cumdis[np.isnan(dis)] = np.nan
    
    return cumdis

def dis2speed(t, dis):
    """
    Return speed in distance travelled per hour.
    
    Args:
        t   (datetime64[ms]): 1D array with time.
        dis (float         ): 1D array with distance travelled.
        
    Returns:
        float: 1D array with speed data.
    """
    
    # divide by one hour (=3600 x 1000 milliseconds)
    speed = np.diff(dis) / (np.float64(np.diff(t))/1000) *3600
    speed = np.r_[np.nan, speed]
    
    return speed