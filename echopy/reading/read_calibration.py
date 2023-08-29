#!/usr/bin/env python3
"""
Modules for reading calibration files.
    
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

__authors__ = ['Alejandro Ariza'   # wrote environment(), ices(), lobes(), and
               ]                   # lobes2params(), echoview()
__credits__ = ['Rob Blackwell'     # supervised the code and provided ideas
               'Sophie Fielding'   # supervised the code and provided ideas               
               ]
import numpy as np
import toml

def environment(file, transect):
    """
    Read environmental metadata, temperature and salinity, from the file
    "transects.toml".
    
    Args:
        file     (str): path to transects.toml.
        transect (str): Name of transect.
        
    Returns:
        float: Temperature (degrees Celsius)
        float: Salinity    (PSU)
    """

    data     = toml.load(file)
    try:
        data = [x for x in data['transect'] if x['name']==transect][0]
    except IndexError:
        raise Exception('Transect name doesn\'t exist')    
    
    temperature = data['temperature']
    salinity    = data['salinity']
    
    return temperature, salinity
        

def ices(calfile, frequency):
    """
    Read calibration parameters from a ICES metadata toml file 
    
    Args:
        calfile (str): path/to/calibration_file.
        frequency (int): frequency you want to read.
                
    Returns:
        object with calibration parameters 
    """
    class params(object):
        pass
    
    #find data_processing attributes for the requested frequency
    data = toml.load(calfile)   
    data = [x for x in data['data_processing'] if x['frequency'] == frequency][0]
    
    # populate params object with data_processing attributes
    params.frequency                = np.float64(data['frequency']*1000            ) # Hz
    params.transmit_power           = np.float64(data['transceiver_power']         ) # watts
    params.pulse_length             = np.float64(data['transmit_pulse_length']/1000) # s
    params.gain                     = np.float64(data['on_axis_gain']              ) # dB
    params.sa_correction            = np.float64(data['Sacorrection']              ) # dB   
    params.absorption_coefficient   = None                                           # dB m-1
    params.sound_velocity           = None                                           # m s-1
    params.equivalent_beam_angle    = np.float64(data['psi']                       ) # dB
    params.angle_beam_athwartship   = np.float64(data['beam_angle_major']          ) # deg
    params.angle_beam_alongship     = np.float64(data['beam_angle_minor']          ) # deg
    params.angle_offset_athwartship = np.float64(data['beam_angle_offset_major']   ) # deg  
    params.angle_offset_alongship   = np.float64(data['beam_angle_offset_minor']   ) # deg
        
    return params
 
def lobes(file):
    """
    Reads LOBES calibration text files and returns variables.
    
    Parameters
    ----------
    file: str
        path/to/lobes/calibration/file.txt

    Returns
    -------
    d: dict
        Contains data variables from the calibration LOBES file.

    """
    
    # open LOBES calibration file and extract variables, line by line
    d = {}
    f = open(file, 'r')    
    while 1:
        l = f.readline()
        if not ('#' in l[0]):
            break
        
        if '#  Calibration  Version' in l:
            d['calibration_version'] = l.strip('#  Calibration  Version').strip()
            
        if '#  Date:' in l:
            d['calibration_date'] = l.strip('#  Date:').strip()
        
        if '#  Comments:' in l:
            l = f.readline()
            d['calibration_coments'] = l.strip('#').strip()
            
        if '#  Reference Target:' in l:
            l = f.readline()
            d['reference_TS']           = float(l.split()[2])
            d['reference_min_distance'] = float(l.split()[6])
            l = f.readline()
            d['reference_TS_deviation'] = float(l.split()[3])
            d['reference_max_distance'] = float(l.split()[7])
                    
        if '#  Transducer:' in l:
            d['instrument_transducer_model']                        = l.split()[2]
            d['instrument_transducer_serial']                       = l.split()[5]        
            l = f.readline()
            d['instrument_transducer_frequency']                    = int(l.split()[2])/1e3
            d['instrument_transducer_beam_type']                    = l.split()[5]        
            l = f.readline()
            d['instrument_transducer_gain']                         = float(l.split()[2])
            d['instrument_transducer_psi']                          = float(l.split()[8])        
            l = f.readline()
            d['instrument_transducer_beam_angle_major_sensitivity'] = float(l.split()[4])
            d['instrument_transducer_beam_angle_minor_sensitivity'] = float(l.split()[8])                
            l = f.readline()
            d['instrument_transducer_beam_angle_major']             = float(l.split()[4])
            d['instrument_transducer_beam_angle_minor']             = float(l.split()[9])        
            l = f.readline()
            d['instrument_transducer_beam_angle_major_offset']      = float(l.split()[4])
            d['instrument_transducer_beam_angle_minor_offset']      = float(l.split()[9])
            
        if'#  Transceiver:' in l:
            d['instrument_transceiver_model']          = l.split()[-1]
            d['instrument_transceiver_serial']         = l.split()[5]        
            l = f.readline()
            d['data_processing_transmit_pulse_length'] = float(l.split()[3])
            d['data_range_axis_interval_value']        = float(l.split()[7])
            d['data_range_axis_interval_type']         = 'Range (metres)'
            l = f.readline()
            d['data_processing_transceiver_power']     = float(l.split()[2])
            d['data_processing_bandwidth']             = float(l.split()[6])
            
        if '#  Sounder Type:' in l:
            l = f.readline()
            d['data_processing_software_version'] = l.split('#')[1].strip()
            
        if '#  TS Detection:' in l:
            l = f.readline()
            d['target_detection_minimum_value']           = float(l.split()[3])
            d['target_detection_minimum_spacing']         = float(l.split()[7])
            l = f.readline()
            d['target_detection_beam_compensation']       = float(l.split()[4])
            d['target_detection_minimum_ecolength']       = float(l.split()[8])
            l = f.readline()
            d['target_detection_maximum_phase_deviation'] = float(l.split()[4])
            d['target_detection_maximum_echolength']      = float(l.split()[7])
            
        if '#  Environment:' in l:
            l                       = f.readline()
            d['calibration_absorption']  = float(l.split()[3])/1000
            d['calibration_sound_speed'] = float(l.split()[7])
            
        if '#  Beam Model results:' in l:
            l = f.readline()
            d['data_processing_on_axis_gain']           = float(l.split()[4])
            d['data_processing_on_axis_gain_units']     = 'dB'
            d['data_processing_Sacorrection']           = float(l.split()[8])
            l = f.readline()
            d['data_procesing_beam_angle_major']        = float(l.split()[5])
            d['data_procesing_beam_angle_minor']        = float(l.split()[11])
            l = f.readline()
            d['data_procesing_beam_angle_major_offset'] = float(l.split()[5])
            d['data_procesing_beam_angle_minor_offset'] = float(l.split()[11])
            
    # return data dictionary    
    return d 

def lobes2params(file):
    """
    Read lobes text file contents, and allocate variables in a params object,
    following pyEcholab structure.

    Parameters
    ----------
    file: str
          path/to/calibration/lobes/file.txt

    Returns
    -------
    params: object
            Contains calibration parameters, following pyEcholab structure.
    """
    
    # read lobes file content
    d = lobes(file)
    
    # allocate variables in params object and return
    class params(object):
        frequency               =d['instrument_transducer_frequency']*1e3      # Hz
        transmit_power          =d['data_processing_transceiver_power']        # W
        pulse_length            =d['data_processing_transmit_pulse_length']/1e3# s
        gain                    =d['data_processing_on_axis_gain']             # dB
        sa_correction           =d['data_processing_Sacorrection']             # dB   
        angle_beam_athwartship  =d['data_procesing_beam_angle_major']          # deg
        angle_beam_alongship    =d['data_procesing_beam_angle_minor']          # deg
        angle_offset_athwartship=d['data_procesing_beam_angle_major_offset']   # deg  
        angle_offset_alongship  =d['data_procesing_beam_angle_minor_offset']   # deg
        absorption_coefficient  =None                                          # dB m-1
        sound_velocity          =None                                          # m s-1
    return params

def echoview(calfile, channel):
    """
    Read calibration parameters from an echoview calibration file
    
    Args:
        calfile (str): path/to/calibration_file.
        channel (int): channel you want to read.
                
    Returns:
        object with calibration parameters
    """
    
    # create object to populate parameters
    class params(object):
        pass
    
    # open calibration file
    f = open(calfile, 'r')
    
    # read all lines in the file
    line = ' '    
    while line:       
        line = f.readline()
        
        # look for parameters after finding the requested transducer channel
        if line == 'SourceCal T' + str(channel) + '\n':           
            while line:                
                line = f.readline()
                
                if line != '\n':
                    
                    if line.split()[0] == 'AbsorptionCoefficient':        
                        params.absorption_coefficient = np.float64(line.split()[2]) # dB s-1
                        
                    if line.split()[0] == 'EK60SaCorrection':        
                        params.sa_correction = np.float64(line.split()[2]) # dB
                        
                    if line.split()[0] == 'Ek60TransducerGain':        
                        params.gain = np.float64(line.split()[2]) # dB
                        
                    if line.split()[0] == 'Frequency':        
                        params.frequency = np.float64(line.split()[2])*1000 # Hz
                        
                    if line.split()[0] == 'MajorAxis3dbBeamAngle':        
                        params.angle_beam_athwartship = np.float64(line.split()[2]) # deg
                        
                    if line.split()[0] == 'MajorAxisAngleOffset':        
                        params.angle_offset_athwartship = np.float64(line.split()[2]) # deg
                        
                    if line.split()[0] == 'MajorAxisAngleSensitivity':        
                        params.angle_sensitivity_athwartship = np.float64(line.split()[2]) #
                        
                    if line.split()[0] == 'MinorAxis3dbBeamAngle':        
                        params.angle_beam_alongship = np.float64(line.split()[2]) # deg
                        
                    if line.split()[0] == 'MinorAxisAngleOffset':        
                        params.angle_offset_alongship = np.float64(line.split()[2]) # deg
                        
                    if line.split()[0] == 'MinorAxisAngleSensitivity':        
                        params.angle_sensitivity_alongship = np.float64(line.split()[2]) #
                        
                    if line.split()[0] == 'SoundSpeed':        
                        params.sound_velocity = np.float64(line.split()[2]) # m s-1
                        
                    if line.split()[0] == 'TransmittedPower':        
                        params.transmit_power = np.float64(line.split()[2]) # watts
                        
                    if line.split()[0] == 'TransmittedPulseLength':        
                        params.pulse_length = np.float64(line.split()[2])/1000 # s
                        
                    if line.split()[0] == 'TvgRangeCorrection':        
                        params.tvg_range_correction = line.split()[2] # str
                        
                    if line.split()[0] == 'TwoWayBeamAngle':        
                        params.equivalent_beam_angle = np.float64(line.split()[2]) # dB
                
                # break when empty line accours 
                else:                    
                    break
            
            # stop reading    
            break
    
    # return parameters    
    return params

def other():
    """    
    Note to contributors:
        Further calibration file readers must be named with the file's
        name or format.
        
        Please, check contribute.md to follow our coding and documenting style.
    """