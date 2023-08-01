"""
Algorithms for resampling data arrays.
    
Copyright (c) XXXX

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

# import modules
import warnings
import numpy as np
from scipy.interpolate import interp1d
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
        svmean, pc, m_ = oned(lin(Sv), r, np.array([r0, r1]),
                                 operation='mean')
        integration_range = r1-r0
        sa = svmean*integration_range
        
    elif operation=='sum':
        svsum, pc, m_ = oned(lin(Sv), r, np.array([r0, r1]),
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
def twod(data, idim, jdim, irvals, jrvals, log=False, operation='mean'):
    """
    Resample down an array along the two dimensions, i and j.
    
    Args:
        data   (float): 2D array with data to be resampled.
        idim   (float): i vertical dimension.
        jdim   (float): j horizontal dimension.
        irvals (float): i resampling intervals for i vertical dimension.
        jrvals (float): j resampling intervals for j horizontal dimension.
        log    (bool ): if True, data is considered logarithmic and it will 
                        be converted to linear during calculations.
        operation(str): type of resampling operation. Accepts "mean" or "sum".
                    
    Returns:
        float: 2D resampled data array
        float: 1D resampled i vertical dimension
        float: 1D resampled j horizontal dimension
        float: 2D array with percentage of valid samples included on each
               resampled cell.
    """ 
    
    # check for appropiate inputs
    if len(irvals)<2:
        raise Exception('length of i resampling intervals must be  >2')
    if len(jrvals)<2:
        raise Exception('length of j resampling intervals must be >2')
    if len(data)!=len(idim):
        raise Exception('data height and idim length must be the same')
    if len(data[0])!=len(jdim):
        raise Exception('data width and jdim length must be the same')        
    for irval, jrval in zip(irvals, jrvals):
        if (irval<idim[0]) | (idim[-1]<irval):
            raise Exception('i resampling intervals must be within idim range')
        if (jrval<jdim[0]) | (jdim[-1]<jrval):
            raise Exception('j resampling intervals must be within jdim range')
        
    # convert data to linear, if logarithmic
    if log is True:
        data = lin(data)
    
    # get i/j axes from i/j dimensions and i/j intervals
    iax   = np.arange(len(idim))
    jax   = np.arange(len(jdim))
    iaxrs = dim2ax(idim, iax, irvals)    
    jaxrs = dim2ax(jdim, jax, jrvals)
    
    # declare new array to allocate resampled values, and new array to
    # alllocate the percentage of values used for resampling
    datar     = np.zeros((len(iaxrs)-1, len(jaxrs)-1))*np.nan
    percentage = np.zeros((len(iaxrs)-1, len(jaxrs)-1))*np.nan
    
    # iterate along-range
    for i in range(len(iaxrs)-1):
        
        # get i indexes to locate the samples for the binning operation
        idx0     = np.where(iax-iaxrs[i+0]<=0)[0][-1]        
        idx1     = np.where(iaxrs[i+1]-iax> 0)[0][-1]
        idx      = np.arange(idx0, idx1+1)
        
        # get i weights as the sum of the sample proportions taken
        iweight0 = 1 - abs(iax[idx0]-iaxrs[i+0])
        iweight1 = abs(iax[idx1]-iaxrs[i+1])
        if len(idx)>1:
            iweights = np.r_[iweight0, np.ones(len(idx)-2), iweight1]
        else:
            iweights = np.array([iweight0-1 + iweight1])
        
        # iterate along-pings
        for j in range(len(jaxrs)-1):
            
            # get j indexes to locate the samples for the binning operation
            jdx0     = np.where(jax-jaxrs[j+0]<=0)[0][-1]        
            jdx1     = np.where(jaxrs[j+1]-jax> 0)[0][-1]
            jdx      = np.arange(jdx0, jdx1+1)
            
            # get j weights as the sum of the sample proportions taken
            jweight0 = 1 - abs(jax[jdx0]-jaxrs[j+0])
            jweight1 = abs(jax[jdx1]-jaxrs[j+1])
            if len(jdx)>1:
                jweights = np.r_[jweight0, np.ones(len(jdx)-2), jweight1]
            else:
                jweights = np.array([jweight0-1 + jweight1])
                            
            # get data and weight 2D matrices for the binning operation
            d = data[idx[0]:idx[-1]+1, jdx[0]:jdx[-1]+1]
            w = np.multiply.outer(iweights, jweights)
                      
            # if d is an all-NAN array, return NAN as the weighted operation
            # and zero as the percentage of valid numbers used for binning
            if np.isnan(d).all():
                datar    [i, j] = np.nan
                percentage[i, j] = 0
            
            #compute weighted operation & percentage of valid numbers otherwise
            else:                    
                w_ = w.copy()
                w_[np.isnan(d)] = np.nan
                if operation=='mean':
                    datar   [i, j]  = np.nansum(d*w_)/np.nansum(w_)
                elif operation=='sum':
                    datar   [i, j]  = np.nansum(d*w_)
                else:
                    raise Exception('Operation not recognised')                        
                percentage[i, j]  = np.nansum(  w_)/np.nansum(w )*100                        
    
    # convert back to logarithmic, if data was logarithmic
    if log is True:
        datar = log(datar)
    
    # get resampled dimensions from resampling intervals
    idimr = irvals[:-1]
    jdimr = jrvals[:-1]
    
    # return data
    return datar, idimr, jdimr, percentage

def oned(data, dim, rvals, axis, log_var=False, operation='mean'):
    """
    Resample down an array along i or j dimension.
    
    Args:
        data  (float) : 2D array with data to be resampled.
        dim   (float) : original dimension.
        rvals (float) : resampling dimension intervals.
        axis  (int  ) : resampling axis (0= i vertical ax, 1= j horizontal ax).
        log   (bool ) : if True, data is considered logarithmic and it will 
                        be converted to linear during the calculations.
        operation(str): type of resampling operation. Accepts "mean" or "sum".
                    
    Returns:
        float: 2D resampled data array
        float: 1D resampled array corresponding to either i or j dimension.
        float: 2D array with percentage of valid samples included on each
               resampled cell.
    """
    
    # check if appropiate axis input
    if axis>1:
        raise Exception('axis must be 0 or 1')
        
    # check if appropiate resampled dimension
    if len(rvals)<2:
        raise Exception('length of resampling intervals must be >2')
    
    # check if intervals are within the dimension range of values 
    for rval in rvals:
        if (rval<dim[0]) | (dim[-1]<rval):
            raise Exception('resampling intervals must be within dim range')
        
    # convert data to linear, if logarithmic
    if log_var is True:
        data = lin(data)
        
    # get axis from dimension
    ax   = np.arange(len(dim))   
    axrs = dim2ax(dim, ax, rvals)
    
    # proceed along i dimension
    if axis==0:
        iax   = ax
        iaxrs = axrs
        
        # check data and axis match
        if len(data)!=len(iax):
            raise Exception('data height and i dimension length must be equal')
        
        # declare new array to allocate resampled values, and new array to
        # alllocate the percentage of values used for resampling
        datar     = np.zeros((len(iaxrs)-1, len(data[0])))*np.nan
        percentage = np.zeros((len(iaxrs)-1, len(data[0])))*np.nan
        
        # iterate along i dimension
        for i in range(len(iaxrs)-1):
            
            # get i indexes to locate the samples for the resampling operation
            idx0     = np.where(iax-iaxrs[i+0]<=0)[0][-1]        
            idx1     = np.where(iaxrs[i+1]-iax> 0)[0][-1]
            idx      = np.arange(idx0, idx1+1)
            
            # get i weights as the sum of the proportions of samples taken
            iweight0 = 1 - abs(iax[idx0]-iaxrs[i+0])
            iweight1 = abs(iax[idx1]-iaxrs[i+1])
            if len(idx)>1:
                iweights = np.r_[iweight0, np.ones(len(idx)-2), iweight1]
            else:
                iweights = np.array([iweight0-1 + iweight1])
                
            # get data and weight 2D matrices for the resampling operation
            d = data[idx[0]:idx[-1]+1, :]
            w = np.multiply.outer(iweights, np.ones(len(data[0])))
                      
            # if d is an all-NAN array, return NAN as the weighted operation
            # and zero as the percentage of valid numbers used for binning
            if np.isnan(d).all():
                datar     [i, :] = np.nan
                percentage[i, :] = 0
            
            # compute weighted operation and percentage valid numbers otherwise
            else:
                w_             =w.copy()
                w_[np.isnan(d)]=np.nan
                if operation=='mean':
                    datar    [i,:]=np.nansum(d*w_,axis=0)/np.nansum(w_,axis=0)
                elif operation=='sum':
                    datar    [i,:]= np.nansum(d*w_,axis=0)
                else:
                    raise Exception('Operation not recognised')
                percentage[i,:]=np.nansum(w_  ,axis=0)/np.nansum(w ,axis=0)*100                        
        
        # convert back to logarithmic, if data was logarithmic
        if log_var is True:
            datar = log(datar)
        
        # get resampled dimension from resampling interval
        dimr = rvals
        
        # return data
        return datar, dimr, percentage
    
    # proceed along j dimension
    if axis==1:
        jax   = ax
        jaxrs = axrs
        
        # check data and axis match
        if len(data[0])!=len(jax):
            raise Exception('data width and j dimension lenght must be equal')
        
        # declare new array to allocate resampled values, and new array to
        # alllocate the percentage of values used for resampling
        datar      = np.zeros((len(data), len(jaxrs)-1))*np.nan
        percentage = np.zeros((len(data), len(jaxrs)-1))*np.nan
        
        # iterate along j dimension
        for j in range(len(jaxrs)-1):
            
            # get j indexes to locate the samples for the resampling operation
            jdx0     = np.where(jax-jaxrs[j+0]<=0)[0][-1]        
            jdx1     = np.where(jaxrs[j+1]-jax> 0)[0][-1]
            jdx      = np.arange(jdx0, jdx1+1)
            
            # get j weights as the sum of the proportions of samples taken
            jweight0 = 1 - abs(jax[jdx0]-jaxrs[j+0])
            jweight1 = abs(jax[jdx1]-jaxrs[j+1])
            if len(jdx)>1:
                jweights = np.r_[jweight0, np.ones(len(jdx)-2), jweight1]
            else:
                jweights = np.array([jweight0-1 + jweight1])
                
            # get data and weight 2D matrices for the resampling operation
            d = data[:, jdx[0]:jdx[-1]+1]
            w = np.multiply.outer(np.ones(len(data)), jweights)
                      
            # if d is an all-NAN array, return NAN as the weighted operation
            # and zero as the percentage of valid numbers used for resampling
            if np.isnan(d).all():
                datar     [:, j] = np.nan
                percentage[:, j] = 0
            
            # compute weighted operation and percentage valid numbers otherwise
            else:
                w_             =w.copy()
                w_[np.isnan(d)]=np.nan
                if operation=='mean':
                    datar     [:,j]=np.nansum(d*w_,axis=1)/np.nansum(w_,axis=1)
                elif operation=='sum':
                    datar     [:,j]=np.nansum(d*w_,axis=1)
                else:
                    raise Exception('Operation not recognised')
                percentage[:,j]=np.nansum(w_  ,axis=1)/np.nansum(w ,axis=1)*100                        
        
        # convert back to logarithmic, if data was logarithmic
        if log_var is True:
            datar = log(datar)
        
        # get resampled dimension from resampling intervals
        dimr = rvals
        
        # return data
        return datar, dimr, percentage

def full(datar, irvals, jrvals, idim, jdim):
    """
    Turn resampled data back to full resolution, according to original i and j
    full resolution dimensions.
    
    Args:
        datar  (float): 2D array with resampled data.
        irvals (float): 1D array with i resampling intervals.
        jdimr  (float): 1D array with i resampling intervals.
        idim   (float): 1D array with full resolution i axis.
        jdim   (float): 1D array with full resolution j axis.
        
    Returns:
        float: 2D array with data resampled at full resolution.
        bool : 2D array with mask indicating valid values in data resampled
               at full resolution.
    """
    
    
    # check for appropiate inputs
    if len(irvals)<2:
        raise Exception('i resampling interval length must be >2')
    if len(jrvals)<2:
        raise Exception('j resampling interval length must be >2')
    if len(datar)+1<len(irvals):
        raise Exception('i resampling intervals length can\'t exceed data height + 1')
    if len(datar[0])+1<len(jrvals):
        raise Exception('j resampling intervals length can\'t exceed data width + 1') 
        
    # get i/j axes from i/j dimensions and i/j intervals
    iax   = np.arange(len(idim))
    jax   = np.arange(len(jdim))
    iaxrs = dim2ax(idim, iax, irvals)    
    jaxrs = dim2ax(jdim, jax, jrvals)
    
    # check whether i/j resampled axes and i/j intervals are different
    idiff = True
    if len(iaxrs)==len(iax):
        if (iaxrs==iax).all():
            idiff = False
    jdiff = True
    if len(jaxrs)==len(jax):
        if (jaxrs==jax).all():
            jdiff = False
    
    # preallocate full resolution data array 
    data = np.zeros((len(iax), len(jax)))*np.nan
        
    # if i/j axes are different, resample back to full along i/j dimensions
    if idiff&jdiff:
        for i in range(len(iaxrs)-1):            
            idx = np.where((iaxrs[i]<=iax) & (iax<iaxrs[i+1]))[0]
            for j in range(len(jaxrs)-1):        
                jdx = np.where((jaxrs[j]<=jax) & (jax<jaxrs[j+1]))[0]
                if idx.size*jdx.size > 0:
                    data[idx[0]:idx[-1]+1, jdx[0]:jdx[-1]+1]= datar[i,j]
    
    # if only i axis is different, resample back to full along i dimension
    elif idiff & (not jdiff):
        for i in range(len(iaxrs)-1):            
            idx = np.where((iaxrs[i]<=iax) & (iax<iaxrs[i+1]))[0]
            if idx.size>0:
                data[idx[0]:idx[-1]+1, :]= datar[i, :]
        
    # if only j axis is different, resample back to full along j dimension
    elif (not idiff) & jdiff:        
        for j in range(len(jaxrs)-1):        
            jdx = np.where((jaxrs[j]<=jax) & (jax<jaxrs[j+1]))[0]
            if jdx.size > 0:
                data[:, jdx[0]:jdx[-1]+1]= datar[:, j].reshape(-1,1)
        
    # if i/j resampled & i/j full are the same, data resampled & data are equal
    else:
        warnings.warn("Array already at full resolution!", RuntimeWarning)
        data= datar.copy()
    
    # get mask indicating where data couldn't be resampled back
    mask_ = np.zeros_like(data, dtype=bool)
    i1= np.where(iax<iaxrs[-1])[0][-1] + 1
    j1= np.where(jax<jaxrs[-1])[0][-1] + 1
    if idiff:
        mask_[i1:,   :] = True
    if jdiff:
        mask_[  :, j1:] = True
    
    return data, mask_

def dim2ax(dim, ax, dimrs):
    """
    It gives you a new resampled axis based on a known dimension/axis pair, and 
    a new resampled dimesion.
    
    Args:
        dim   (float): 1D array with original dimension.
        ax    (int  ): 1D array with original axis.
        dimrs (float): 1D array with resampled dimension.
    
    Returns:
        float: 1D array with resampled axis.
    
    Notes:
        Dimension refers to range, time, latitude, distance, etc., and axis
        refer to dimension indexes such as sample or ping number.
        
    Example:
                                                         (resampled)
        seconds dimension | ping axis       seconds dimension | ping axis
        ------------------·----------       ------------------·----------
                       0  | 0                             0   | 0
                       2  | 1                             3   | 1.5
                       4  | 2          ==>                6   | 3.0
                       6  | 3                             9   | 4.5
                       8  | 4                             -   | -
                      10  | 4                             -   | -
    """
    
    # check that resampled dimension doesn't exceed the limits
    # of the original one
    if (dimrs[0]<dim[0]) | (dimrs[-1]>dim[-1]):
        raise Exception('resampling dimension can not exceed ' +
                        'the original dimension limits') 
        
    # convert variables to float64 if they are in datetime64 format
    epoch = np.datetime64('1970-01-01T00:00:00')    
    if 'datetime64' in str(dim[0].dtype):        
        dim = np.float64(dim-epoch)        
    if 'datetime64' in str(dimrs[0].dtype):
        dimrs = np.float64(dimrs-epoch)
    
    # get interpolated new y variable    
    f    = interp1d(dim, ax)
    axrs = f(dimrs)
            
    return axrs
