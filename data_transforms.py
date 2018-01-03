# coding=utf-8

#     National Oceanic and Atmospheric Administration (NOAA)
#     Alaskan Fisheries Science Center (AFSC)
#     Resource Assessment and Conservation Engineering (RACE)
#     Midwater Assessment and Conservation Engineering (MACE)

#  THIS SOFTWARE AND ITS DOCUMENTATION ARE CONSIDERED TO BE IN THE PUBLIC DOMAIN
#  AND THUS ARE AVAILABLE FOR UNRESTRICTED PUBLIC USE. THEY ARE FURNISHED "AS IS."
#  THE AUTHORS, THE UNITED STATES GOVERNMENT, ITS INSTRUMENTALITIES, OFFICERS,
#  EMPLOYEES, AND AGENTS MAKE NO WARRANTY, EXPRESS OR IMPLIED, AS TO THE USEFULNESS
#  OF THE SOFTWARE AND DOCUMENTATION FOR ANY PURPOSE. THEY ASSUME NO RESPONSIBILITY
#  (1) FOR THE USE OF THE SOFTWARE AND DOCUMENTATION; OR (2) TO PROVIDE TECHNICAL
#  SUPPORT TO USERS.

u'''
.. module:: pyecholab2.instruments.util.data_transforms.py

data_transforms implements common functions used by echolab2 "data objects".
Data objects are classes that store acoustic data collected at discrete
intervals (i.e. "pings"). This can be "raw" data as stored by the
instruments.EK60.RawData class or "processed" data as stored by the
processing.ProcessedData class.


useful functions:



| Developed by:  Rick Towler   <rick.towler@noaa.gov>
| National Oceanic and Atmospheric Administration (NOAA)
| Alaska Fisheries Science Center (AFSC)
| Midwater Assesment and Conservation Engineering Group (MACE)
|
| Author:
|       Rick Towler   <rick.towler@noaa.gov>
| Maintained by:
|       Rick Towler   <rick.towler@noaa.gov>

'''


import logging
import numpy as np

log = logging.getLogger(__name__)



def delete(data_obj, start_ping=None, end_ping=None, start_time=None,
        end_time=None, remove=True):
    '''
    delete deletes data from an echolab2 data object by ping over the range
    defined by the start and end pings/times. If remove is True, the data
    arrays are shrunk, if False the arrays stay the same size and the data
    values are set to NaNs (or appropriate value based on type)
    '''
    #  determine the indices of the pings we're deleting
    idx = get_indices(data_obj, start_time=start_time, end_time=end_time,
            start_ping=start_ping, end_ping=end_ping)


def append(data_obj, obj_to_append):
    '''
    append appends another echolab2 data object to this one. The objects must
    be instances of the same class and share the same frequency.
    '''

    #  append simply inserts at the end of our internal array.
    insert(data_obj,obj_to_append, ping_number=data_obj.ping_number[-1])


def insert(data_obj, obj_to_insert, ping_number=None, ping_time=None,
        insert_after=True):
    '''
    insert inserts the data from the provided echolab2 data object into this object.
    The insertion point is specified by ping number or time. After inserting
    data, the ping_number property is updated and the ping numbers from the insertion
    point onward will be re-numbered accordingly.

    Note that internally echolab2 RawData objects store data by ping number as read,
    appended, or inserted. Data is not sorted by time until the user calls a get_*
    method. Consequently if ping times are not ordered and/or repeating, specifying
    a ping time as an insetion point may not yield the desired results since the
    insertion index will be the *first* index that satisfies the condition.

    By default, the insert is *after* the provided ping number or time. Set the
    insert_after keyword to False to insert *before* the provided ping number or time.

    '''

    #  check that we have been given an insetion point
    if ping_number is None and ping_time is None:
        raise ValueError('Either ping_number or ping_time needs to be defined to ' +
                'specify an insertion point.')

    #  make sure that data_object is an EK60.RawData raw data object
    if (not isinstance(data_obj, obj_to_insert.__class__)):
        raise TypeError('The object you are inserting/appending must be an instance of ' +
            str(data_obj.__class__))

    #  make sure that the frequencies match - we don't allow insrting/appending of different frequencies
    if (data_obj.frequency[0] != obj_to_insert.frequency[0]):
        raise TypeError('The frequency of the object you are inserting/appending ' +
                'does not match the  frequency of this object. Frequencies must match to ' +
                'append or insert.')

    #  determine the index of the insertion point
    idx = get_indices(data_obj, start_time=ping_time, end_time=ping_time,
            start_ping=ping_number, end_ping=ping_number)[0]

    #  check if we're inserting before or after the provided insert point and adjust as necessary
    if (insert_after):
        #  we're inserting *after* - increment the index by 1
        idx += 1

    #  get some info about the shape of the data we're inserting
    my_pings = data_obj.ping_number.shape[0]
    new_pings = obj_to_insert.ping_number.shape[0]
    my_samples = data_obj.power.shape[1]
    new_samples = obj_to_insert.power.shape[1]

    #  check if we need to vertically resize one of the arrays - we resize the smaller to
    #  the size of the larger array. It will automatically be padded with NaNs
    if (my_samples < new_samples):
        #  resize our data arrays - check if we have a limit on the max number of samples
        if (hasattr(data_obj, 'max_sample_number') and (data_obj.max_sample_number)):
            #  we have the attribue and a value is set - chech if the new array exceeds our max_sample_count
            if (new_samples > data_obj.max_sample_number):
                #  it does - we have to change our new_samples
                new_samples = data_obj.max_sample_number
                #  and vertically trim the array we're inserting
                resize_arrays(obj_to_insert, new_pings, new_samples)
        #  vertically resize the object we're inserting into
        resize_arrays(data_obj, my_pings, new_samples)
    elif (my_samples > new_samples):
        #  resize the object we're inserting
        resize_arrays(obj_to_insert, new_pings, new_samples)

    #  work thru our data properties inserting the data from obj_to_insert
    for attribute in data_obj._data_attributes:

        #  check if we have data for this attribute
        if (not hasattr(data_obj, attribute)):
            #  data_obj does not have this attribute, move along
            continue

        #  get a reference to our data_obj's attribute
        data = getattr(data_obj, attribute)

        #  check if the obj_to_insert shares this attribute
        if (hasattr(obj_to_insert, attribute)):
            #  get a reference to our obj_to_insert's attribute
            data_to_insert = getattr(obj_to_insert, attribute)

            #  handle lists and numpy arrays appropriately
            if isinstance(data, list):
                #  this attribute is a list - create the new list
                new_data = data[0:idx] + data_to_insert + data[idx:]
                #  and update our property
                setattr(data_obj, attribute, new_data)

            elif isinstance(data, np.ndarray):
                #  this attribute is a numpy array - we have to handle the 2d and 1d differently
                if (data.ndim == 1):
                    #  concatenate the 1d data
                    if (attribute == 'ping_number'):
                        #  update ping_number so it is sequential
                        new_data = np.arange(data.shape[0]+data_to_insert.shape[0]) + 1
                    else:
                        new_data = np.concatenate((data[0:idx], data_to_insert, data[idx:]))
                elif (data.ndim == 2):
                    #  concatenate the 2d data
                    new_data = np.vstack((data[0:idx,:], data_to_insert, data[idx:,:]))
                else:
                    #  at some point do we handle 3d arrays?
                    pass

                #  update this attribute
                setattr(data_obj, attribute, new_data)

    #  now update our global properties
    if (obj_to_insert.channel_id not in data_obj.channel_id):
        data_obj.channel_id += obj_to_insert.channel_id
    data_obj.n_pings = data_obj.ping_number.shape[0]


def trim(data_obj, length, n_samples=None):
    '''
    trim deletes pings from an echolab2 data object to a given length
    '''

    #  work thru our list of attributes to find a 2d array and get the sample number

    if (n_samples == None):
        n_samples = -1
        for attr_name in data_obj._data_attributes:
            #  get a reference to this attribute
            if (hasattr(data_obj, attr_name)):
                attr = getattr(data_obj, attr_name)
            else:
                continue
            #  get the nunber of samples if this is a 2d array
            if (isinstance(attr, np.ndarray) and (n_samples < 0) and (attr.ndim == 2)):
                n_samples = attr.shape[1]
                break

    #  resize keeping the sample number the same
    resize_arrays(data_obj, length, n_samples)


def get_indices(data_obj, start_ping=None, end_ping=None, start_time=None,
        end_time=None, time_order=True):
    '''
    get_indices returns an index array containing the indices contained in the range
    defined by the times and/or ping numbers provided. By default the indexes are in time
    order. If time_order is set to False, the data will be returned in the order they
    occur in the data arrays.
    '''

    #  if starts and/or ends are omitted, assume fist and last respectively
    if (start_ping == start_time == None):
        start_ping = data_obj.ping_number[0]
    if (end_ping == end_time == None):
        end_ping = data_obj.ping_number[-1]

    #  get the primary index
    if (time_order):
        #  return indices in time order
        primary_index = data_obj.ping_time.argsort()
    else:
        #  return indices in ping order
        primary_index = data_obj.ping_number - 1

    #  generate a boolean mask of the values to return
    if (start_time):
        mask = data_obj.ping_time[primary_index] >= start_time
    elif (start_ping >= 0):
        mask = data_obj.ping_number[primary_index] >= start_ping
    if (end_time):
        mask = np.logical_and(mask, data_obj.ping_time[primary_index] <= end_time)
    elif (end_ping >= 0):
        mask = np.logical_and(mask, data_obj.ping_number[primary_index] <= end_ping)

    #  and return the indices that are included in the specified range
    return primary_index[mask]


def vertical_resample(data, sample_intervals, unique_sample_intervals, resample_interval,
        sample_offsets, min_sample_offset, is_power=True, dtype='float32'):
    '''
    vertical_resample er, vertically resamples sample data given a target sample interval.
    This function also shifts samples vertically based on their sample offset so they
    are positioned correctly relative to each other. The first sample in the resulting
    array will have an offset that is the minimum of all offsets in the data.
    '''

    #  determine the number of pings in the new array
    n_pings = data.shape[0]

    # check if we need to substitute our resample_interval value
    if (resample_interval == 0):
        #  resample to the shortest sample interval in our data
        resample_interval = min(unique_sample_intervals)
    elif (resample_interval == 1):
        #  resample to the longest sample interval in our data
        resample_interval = max(unique_sample_intervals)

    #  generate a vector of sample counts - generalized method that works with both
    #  RawData and ProcessedData classes that finds the first non-NaN value searching
    #  from the "bottom up"
    sample_counts = data.shape[1] - np.argmax(~np.isnan(np.fliplr(data)), axis=1)

    #  create a couple of dictionaries to store resampling parameters by sample interval
    #  they will be used again when we fill the output array with the resampled data.
    resample_factor = {}
    rows_this_interval = {}
    sample_offsets_this_interval = {}

    #  determine number of samples in the output array - to do this we must loop thru
    #  the sample intervals, determine the resampling factor, then find the maximum sample
    #  count at that sample interval (taking into account the sample's offset) and multiply
    #  by the resampling factor to determine the max number of samples for that sample interval.
    new_sample_dims = 0
    for sample_interval in unique_sample_intervals:
        #  determine the resampling factor
        if (resample_interval > sample_interval):
            #  we're reducing resolution - determine the number of samples to average
            resample_factor[sample_interval] = resample_interval / sample_interval
        else:
            #  we're increasing resolution - determine the number of samples to expand
            resample_factor[sample_interval] = sample_interval / resample_interval

        #  determine the rows in this subset with this sample interval
        rows_this_interval[sample_interval] = np.where(sample_intervals == sample_interval)[0]

        #  determine the net vertical shift for the samples with this sample interval
        sample_offsets_this_interval[sample_interval] = sample_offsets[rows_this_interval[sample_interval]] - \
                min_sample_offset

        #  and determine the maximum number of samples for this sample interval - this has to
        #  be done on a row-by-row basis since sample number can change on the fly. We include
        #  the sample offset to ensure we have room to shift our samples vertically by the offset
        max_samples_this_sample_int = max(sample_counts[rows_this_interval[sample_interval]] +
                sample_offsets_this_interval[sample_interval])
        max_dim_this_sample_int = int(round(max_samples_this_sample_int * resample_factor[sample_interval]))
        if (max_dim_this_sample_int > new_sample_dims):
                new_sample_dims = max_dim_this_sample_int

    #  emit some info to the logger
    log.info("Vertically resampling " + str(data.shape) + " array to " +
            str((n_pings, new_sample_dims)))
    log.info("New sample interval is " + str(resample_interval * 1000 * 1000) + " us.")

    #  now that we know the dimensions of the output array create the it and fill with NaNs
    resampled_data = np.empty((n_pings, new_sample_dims), dtype=dtype, order='C')
    resampled_data.fill(np.nan)

    #  and fill it with data - We loop thru the sample intervals and within an interval extract slices
    #  of data that share the same number of samples (to reduce looping). We then determine if we're
    #  expanding or shrinking the number of samples. If expanding we simply replicate existing sample
    #  data to fill out the expaned array. If reducing, we take the mean of the samples. Power data is
    #  converted to linear units before the mean is computed and then transformed back.
    for sample_interval in unique_sample_intervals:
        #  determine the unique sample_counts for this sample interval
        unique_sample_counts = np.unique(sample_counts[rows_this_interval[sample_interval]])
        for count in unique_sample_counts:
            #  determine if we're reducing, expanding, or keeping the same number of samples
            if (resample_interval > sample_interval):
                #  we're reducing the number of samples

                #  if we're resampling power convert power to linear units
                if (is_power):
                    this_data = np.power(data[rows_this_interval[sample_interval]]
                            [sample_counts[rows_this_interval[sample_interval]] == count] / 20.0, 10.0)

                #  reduce the number of samples by taking the mean
                this_data =  np.mean(this_data.reshape(-1, int(resample_factor[sample_interval])), axis=1)

                if (is_power):
                    #  convert power back to log units
                    this_data = 20.0 * np.log10(this_data)

            elif (resample_interval < sample_interval):
                #  we're increasing the number of samples

                #  replicate the values to fill out the higher resolution array
                this_data = np.repeat(data[rows_this_interval[sample_interval]]
                        [sample_counts[rows_this_interval[sample_interval]] == count][:,0:count],
                        int(resample_factor[sample_interval]), axis=1)

            else:
                #  no change in resolution for this sample interval
                this_data = data[rows_this_interval[sample_interval]] \
                        [sample_counts[rows_this_interval[sample_interval]] == count]


            #  generate the index array for this sample interval/sample count chunk of data
            rows_this_interval_count = rows_this_interval[sample_interval] \
                    [sample_counts[rows_this_interval[sample_interval]] == count]

            #  assign new values to output array - at the same time we will shift the data by sample offset
            unique_sample_offsets = np.unique(sample_offsets_this_interval[sample_interval])
            for offset in unique_sample_offsets:
                resampled_data[rows_this_interval_count,
                        offset:offset + this_data.shape[1]]  = this_data

    #  return the resampled data and the sampling interval used
    return (resampled_data, resample_interval)


def vertical_shift(data, sample_offsets, unique_sample_offsets, min_sample_offset,
        dtype='float32'):
    '''
    vertical_shift adjusts the output array size and pads the top of the
    samples array to vertically shift the positions of the sample data in the output
    array. Pings with offsets greater than the minimum will be padded on the top,
    shifting them into their correct location relative to the other pings.

    The result is an output array with samples that are properly aligned vertically
    relative to each other with a sample offset that is constant and equal to the
    minimum of the original sample offsets.

    This function is only called if our data has a constant sample interval but
    varying sample offsets. If the data has multiple sample intervals the offset
    adjustment is done in vertical_resample.
    '''

    #  determine the new array size
    new_sample_dims = data.shape[1] + max(sample_offsets) - min_sample_offset

    #  create the new array
    shifted_data = np.empty((data.shape[0], new_sample_dims), dtype=dtype, order='C')
    shifted_data.fill(np.nan)

    #  and fill it looping over the different sample offsets
    for offset in unique_sample_offsets:
        rows_this_offset = np.where(sample_offsets == offset)[0]
        start_index = offset - min_sample_offset
        end_index = start_index + data.shape[1]
        shifted_data[rows_this_offset, start_index:end_index] = data[rows_this_offset, 0:data.shape[1]]

    return shifted_data


def resize_arrays(data_object, new_ping_dim, new_sample_dim):
    '''
    resize_arrays iterates thru the provided list of attributes and resizes them in the
    instance of the object provided given the new array dimensions.
    '''

    #  initialize arrays to store the "old" data dimensions
    old_ping_dim = -1
    old_sample_dim = -1

    def resize2d(data, ping_dim, sample_dim):
        '''
        resize2d returns a new array of the specified dimensions with the data from
        the provided array copied into it. This funciton is used when we need to resize
        2d arrays along the minor axis as ndarray.resize and numpy.resize don't maintain
        the order of the data in these cases.
        '''

        #  if the minor axis is changing we have to either concatenate or copy into a new resized
        #  array. I'm taking the second approach for now as I don't think there are performance
        #  differences between the two approaches.

        #  create a new array
        new_array = np.empty((ping_dim, sample_dim))
        #  fill it with NaNs
        new_array.fill(np.nan)
        #  copy the data into our new array
        new_array[0:data.shape[0], 0:data.shape[1]] = data
        #  and return it
        return new_array

    #  work thru our list of attributes
    for attr_name in data_object._data_attributes:

        #  get a reference to this attribute
        if (hasattr(data_object, attr_name)):
            attr = getattr(data_object, attr_name)
        else:
            continue

        #  check if this data attribute is a list ans skip over if so
        if (isinstance(attr, list)):
            continue

        #  determine the "old" dimensions
        if ((old_ping_dim < 0) and (attr.ndim == 1)):
            old_ping_dim = attr.shape[0]
        if ((old_sample_dim < 0) and (attr.ndim == 2)):
            old_sample_dim = attr.shape[1]

        #  resize the 1d arrays
        if ((attr.ndim == 1) and (new_ping_dim != old_ping_dim)):
            #  resize this 1d attribute
            #attr.resize((new_ping_dim))
            attr = np.resize(attr,(new_ping_dim))
        elif (attr.ndim == 2):
            #  resize this 2d array
            if (new_sample_dim == old_sample_dim):
                #  if the minor axes isn't changing we can use ndarray.resize()
                #attr.resize((new_ping_dim, new_sample_dim))
                attr = np.resize(attr,(new_ping_dim, new_sample_dim))
            else:
                #  if the minor axes is changing we need to use our resize2d function
                attr = resize2d(attr, new_ping_dim, new_sample_dim)

        setattr(data_object, attr_name, attr)
