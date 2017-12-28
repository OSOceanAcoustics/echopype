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
.. module:: pyecholab.instruments.util.unit_conversion


useful functions:

    power_to_Sv
    Sv_to_power

    power_to_Sp
    Sp_to_power

    indexed_to_electrical_angle
    indexed_to_physical_angle

    electrical_to_indexed_angle
    electrical_to_physical_angle

    physical_to_electrical_angle
    physical_to_indexed_angle


| Developed by:  Zac Berkowitz <zac.berkowitz@gmail.com> under contract for
| National Oceanic and Atmospheric Administration (NOAA)
| Alaska Fisheries Science Center (AFSC)
| Midwater Assesment and Conservation Engineering Group (MACE)
|
| Author:
|       Zac Berkowitz <zac.berkowitz@gmail.com>
| Maintained by:
|       Rick Towler   <rick.towler@noaa.gov>

'''


import logging
import numpy as np

#REQUIRED_CAL_PARAMS = ['frequency', 'sound_velocity', 'sample_interval', 'gain',
#                       'absorption_coefficient', 'equivalent_beam_angle', 'transmit_power',
#                       'pulse_length']
#
#OPTIONAL_CAL_PARAMS = [('sa_correction', 0.0), ('offset', 0)]

__all__ = ['power_to_Sp', 'power_to_Sv', 'Sp_to_power', 'Sv_to_power',
           'indexed_to_electrical_angle', 'indexed_to_physical_angle',
           'electrical_to_indexed_angle', 'physical_to_indexed_angle',
           'electrical_to_physical_angle', 'physical_to_electrical_angle']

log = logging.getLogger(__name__)

#def check_cal_data(cal):
#    '''
#    Checks that all required calibration data is present and calculates
#    the following derived quantities:
#
#    dR:         Sample Thickness
#    wlength:     Wavelength
#
#    The dictionary `cal` is required to have the following keys:
#
#        frequency
#        sound_velocity
#        sample_interval
#        gain
#        absorption_coefficient
#        equivalent_beam_angle
#        transmit_power
#        pulse_length
#
#    Additionally, the following optional keys are checked for.  If not present,
#    the listed default value is used:
#
#        sa_correction    (0.0)
#        offset            (0)        Sample offset of first value
#
#    :param cal: Calibration data
#    :type cal:  dict
#    '''
#    missing_required_params = []
#    #missing_optional_params = []
#
#    for param in REQUIRED_CAL_PARAMS:
#        if not cal.has_key(param):
#            missing_required_params.append(param)
#
#    if missing_required_params != []:
#        err_string = 'Calibration data missing required field(s): ', str(missing_required_params)
#        log.error(err_string)
#        raise ValueError(err_string)
#
#    for param, default in OPTIONAL_CAL_PARAMS:
#        if not cal.has_key(param):
#            warn_string = 'Calibration data missing optional field: %s  Using default value %s' %(param, str(default))
#            log.warning(warn_string)
#
#            cal[param] = default
#
#    #Calculate derived quantities
#
#    #Sample Thickness:
#    dR = cal['sound_velocity'] * cal['sample_interval'] / 2.0
#
#    #Wavelength
#    wlength = cal['sound_velocity'] / float(cal['frequency']) # float cast ensures non-integer arithmatic
#
#    #Convenicne constant for calculating CSv and CSp
#    beta = cal['transmit_power'] * (10**(cal['gain'] / 10.0) * wlength)**2 / (16 * np.pi**2)
#
#    #CSv
#    CSv = 10 * np.log10(beta / 2.0 * cal['sound_velocity'] * cal['pulse_length'] * 10**(cal['equivalent_beam_angle'] / 10.0))
#
#    #CSp
#    CSp = 10 * np.log10(beta)
#
#    return dR, wlength, CSv, CSp


def get_range_vector(num_samples, sample_interval, sound_speed, sample_offset,
        tvg_correction=2):
#    '''
#    Constructs an [num_sample x num_ping] corrected range matrix for use
#    in power conversions.
#
#    :param num_samples: number of samples
#    :type num_samples: int
#
#    :param num_pings: number of pings
#    :type num_pings: int
#
#    :param sample_offset:  first sample's offset value
#    :type sample_offset: int
#
#    :param sample_thickness:  sample thickness (sound_velocity * sample_interval / 2)
#    :type sample_thickness: float
#
#    :param tvg_correction:  Time-varying gain factor (typically = 2)
#    :param tvg_correction: int
#    '''
#
    if tvg_correction is None:
        tvg_correction = 2

    #  calculate the thickness of samples with this sound speed
    thickness = sample_interval * sound_speed / 2.0
    #  calculate the range vector
    range = (np.arange(0, num_samples) + sample_offset) * thickness
    #  apply TVG range correction
    range = range - (tvg_correction * thickness)
    #  zero negative ranges
    range[range < 0] = 0


    return range


#def make_RANGE_vector(num_samples, num_pings, sample_offset, sample_thickness, tvg_correction=2):
#    '''
#    Constructs an [num_sample x num_ping] corrected range matrix for use
#    in power conversions.
#
#    :param num_samples: number of samples
#    :type num_samples: int
#
#    :param num_pings: number of pings
#    :type num_pings: int
#
#    :param sample_offset:  first sample's offset value
#    :type sample_offset: int
#
#    :param sample_thickness:  sample thickness (sound_velocity * sample_interval / 2)
#    :type sample_thickness: float
#
#    :param tvg_correction:  Time-varying gain factor (typically = 2)
#    :param tvg_correction: int
#    '''
#
#    if tvg_correction is None:
#        tvg_correction = 2
#
#    sample_range = (np.arange(0, num_samples) + sample_offset) * sample_thickness
#    corrected_range = sample_range - (tvg_correction * sample_thickness)
#    corrected_range[corrected_range < 0] = 0
#
#    # RANGE_MATRIX = np.repeat(np.reshape(corrected_range, (-1, 1)), num_pings, axis=1).astype('float32')
#
#    return corrected_range

def _calc_beta(gain, sound_velocity, frequency, transmit_power):
    '''

    :param gain:
    :type gain: float

    :param sound_velocity:
    :type sound_veloicty: float

    :param frequency:
    :type frequency: float

    :param transmit_power:
    :type transmit_power: float

    Calculates the conveniece parameter `beta` used for calculation of the
    CSv and CSp constants

    '''

#    if isinstance(calibration, dict):
#        sound_velocity  = calibration.get('sound_velocity', data['sound_velocity'])
#        frequency       = calibration.get('frequency', data['frequency'])
#        transmit_power  = calibration.get('transmit_power', data['transmit_power'])
#        gain            = calibration.get('gain', data['gain'])
#        eba             = calibration.get('equivalent_beam_angle', eba)
#
#    else:
#        sound_velocity  = data['sound_velocity']
#        frequency       = data['frequency']
#        transmit_power  = data['transmit_power']
#        gain            = data['gain']


    #Convenicne constant for calculating CSv and CSp
    wlength = sound_velocity / (1.0 * frequency)
    beta = transmit_power * (10**(gain / 10.0) * wlength)**2 / (16 * np.pi**2)

    return beta


def _calc_CSv(gain, sound_velocity, frequency, transmit_power, eba, pulse_length):
    '''
    :param gain:
    :type gain: float

    :param sound_velocity:
    :type sound_veloicty: float

    :param frequency:
    :type frequency: float

    :param transmit_power:
    :type transmit_power: float

    :param eba:  Equivalent Beam Angle
    :type eba: float

    :param pulse_length:
    :type pulse_length: float


    Calculates the CSv constant used in power <-> Sv conversions
    '''
    beta = _calc_beta(gain, sound_velocity, frequency, transmit_power)

    CSv = 10 * np.log10(beta / 2.0 * sound_velocity * pulse_length * 10**(eba / 10.0))

    return CSv


def _calc_CSp(gain, sound_velocity, frequency, transmit_power):
    '''
    :param gain:
    :type gain: float

    :param sound_velocity:
    :type sound_veloicty: float

    :param frequency:
    :type frequency: float

    :param transmit_power:
    :type transmit_power: float

    Calculates the CSp constant used in power <-> Sp conversions

    '''

    beta = _calc_beta(gain, sound_velocity, frequency, transmit_power)
    CSp = 10 * np.log10(beta)

    return CSp


def _range_vector(offset, count, sound_velocity, sample_interval, tvg_correction=2.0):
    '''
    :param offset:
    :type offset: int

    :param count:
    :type count: int

    :param sound_velocity:
    :type sound_velocity: float

    :param sample_interval:
    :type sample_interval: float

    :param tvg_correction:
    :type tvg_correcction: float

    Calculates the the tvg-corrected range vector used in Sp and Sv conversions
    '''

    dR = sound_velocity * sample_interval / 2.0
    sample_range = (np.arange(0, count) + offset) * dR
    corrected_range = sample_range - (tvg_correction * dR)
    corrected_range[corrected_range < 0] = 0


#    return sample_range
    return corrected_range


def power_to_Sv(data, gain, eba, sa_correction, calibration=None,
                tvg_correction=2.0, linear=False, raw=False):
    '''
    :param data: Raw power (on dB scale) to calculate Sv from
    :type data: pd.Series

    :param gain: Gain
    :type gain: float


    :param eba: Equivalent Beam Angle
    :type eba: float

    :param sa_correction:  Sa correction coefficient
    :type sa_correction: float

    :param calibration: Calibration information
    :type calibration: dict

    :param tvg_correction:  Number of samples for time-varying gain correction (default 2)
    :type tvg_correction: int

    :param linear:  Use linear units for Sv (calculate sv)
    :type linear: bool

    :param raw: Data is raw indexed power
    :type raw: bool

    Converts raw power (in dB) to Sv

    `raw`
    The 'power' data in Simrad .raw files is an indexed datatype.
        db_power = indexed_power * 10log_10(2.0) / 256

    Setting raw=True performs this conversion step before calculating Sv


    The paramter `calibration` is a dictionary any of the following keys:

        absorption_coefficient
        sound_velocity
        frequency
        transmit_power
        pulse_length
        offset
        count
        sample_interval

    Converts raw power (in dB) to Sv, with preference to values contained in
    `calibration` instead of the default values contained in the ping meta
    information.

    WARNING:  The following parameters are KNOWN TO CHANGE during long surveys
        sample_interval
        pulse_length

    DO NOT provide 'calibrated' values for these keys UNLESS you know what
    you're doing!

    '''

    #Empty arrays -- return empty result
    if np.prod(data.shape) == 0:
        return np.empty((0,))

    if isinstance(calibration, dict):
        absorption_coefficient = calibration.get('absorption_coefficient', data['absorption_coefficient'])
        sound_velocity  = calibration.get('sound_velocity', data['sound_velocity'])
        frequency       = calibration.get('frequency', data['frequency'])
        transmit_power  = calibration.get('transmit_power', data['transmit_power'])
        pulse_length    = calibration.get('pulse_length', data['pulse_length'])
        offset          = calibration.get('offset', data['offset'])
        count           = calibration.get('count', data['count'])
        sample_interval = calibration.get('sample_interval', data['sample_interval'])

    else:
        sound_velocity  = data['sound_velocity']
        frequency       = data['frequency']
        transmit_power  = data['transmit_power']
        pulse_length    = data['pulse_length']
        absorption_coefficient = data['absorption_coefficient']
        offset          = data['offset']
        count           = data['count']
        sample_interval = data['sample_interval']

    CSv = _calc_CSv(gain, sound_velocity, frequency, transmit_power, eba, pulse_length)
    range_ = _range_vector(offset, count, sound_velocity, sample_interval, tvg_correction)
    tvg = range_.copy()
    tvg[tvg == 0] = 1
    tvg = 20 * np.log10(tvg)
    tvg[tvg < 0] = 0

    #TODO: CHECK IF POWER DATA IS ALREADY CONVERTED FROM INDEXED POWER BY ZAC'S READING CODE

    if raw:
        raw_factor = 10.0 * np.log10(2.0) / 256.0
    else:
        raw_factor = 1

    convert = lambda x: x * raw_factor + tvg + 2 * absorption_coefficient * range_ - \
        CSv - 2 * sa_correction

    if data['power'].ndim == 1:
        if linear:
            Sv = 10 **(convert(data['power']) / 10.0)

        else:
            Sv = convert(data['power'])

    elif data['power'].ndim == 2:
        Sv = np.empty_like(data['power'], dtype=np.float)
        for ping in range(data['power'].shape[1]):

            if linear:
                Sv[:, ping] = 10 **(convert(data['power'][:, ping]) / 10.0)

            else:
                Sv[:, ping] = convert(data['power'][:, ping])

    else:
        raise ValueError('Expected a 1- or 2-dimensional array')

    return Sv


def Sv_to_power(data, gain, eba, sa_correction, tvg_correction=2.0,
                calibration=None, linear=False, raw=True):
    '''
    :param data: Sv (on dB scale) or sv (linear scale) to calculate power from
    :type data: pd.Series

    :param eba: Equivalent Beam Angle
    :type eba: float

    :param sa_correction:  Sa correction coefficient
    :type sa_correction: float

    :param calibration: Calibration information
    :type calibration: dict

    :param tvg_correction:  Number of samples for time-varying gain correction (default 2)
    :type tvg_correction: int

    :param linear:  Use linear units for Sv (calculate sv)
    :type linear: bool

    :param raw: Return raw indexed power
    :type raw: bool

    Converts Sv to raw power (in dB)

    `raw`
    The 'power' data in Simrad .raw files is an indexed datatype.
        db_power = indexed_power * 10log_10(2.0) / 256

    The paramter `calibration` is a dictionary any of the following keys:

        absorption_coefficient
        sound_velocity
        frequency
        transmit_power
        pulse_length
        offset
        count
        sample_interval

    Converts Sv -> raw power, with preference to values contained in
    `calibration` instead of the default values contained in the ping meta
    information.

    WARNING:  The following parameters are KNOWN TO CHANGE during long surveys
        sample_interval
        pulse_length

    DO NOT provide 'calibrated' values for these keys UNLESS you know what
    you're doing!

    '''

    if isinstance(calibration, dict):
        absorption_coefficient = calibration.get('absorption_coefficient', data['absorption_coefficient'])
        sound_velocity  = calibration.get('sound_velocity', data['sound_velocity'])
        frequency       = calibration.get('frequency', data['frequency'])
        transmit_power  = calibration.get('transmit_power', data['transmit_power'])
        pulse_length    = calibration.get('pulse_length', data['pulse_length'])
        offset          = calibration.get('offset', data['offset'])
        count           = calibration.get('count', data['count'])
        sample_interval = calibration.get('sample_interval', data['sample_interval'])

    else:
        sound_velocity  = data['sound_velocity']
        frequency       = data['frequency']
        transmit_power  = data['transmit_power']
        pulse_length    = data['pulse_length']
        absorption_coefficient = data['absorption_coefficient']
        offset          = data['offset']
        count           = data['count']
        sample_interval = data['sample_interval']

    CSv = _calc_CSv(gain, sound_velocity, frequency, transmit_power, eba, pulse_length)
    range_ = _range_vector(offset, count, sound_velocity, sample_interval, tvg_correction)
    tvg = range_.copy()
    tvg[tvg == 0] = 1
    tvg = 20 * np.log10(tvg)
    tvg[tvg < 0] = 0

    if linear:
        sv_key = 'sv'
    else:
        sv_key = 'Sv'

    if raw:
        raw_factor = 10.0 * np.log10(2.0) / 256.0
    else:
        raw_factor = 1.0

    convert = lambda x: (x - tvg - 2 * absorption_coefficient * range_ + \
        CSv + 2 * sa_correction) / raw_factor

    if data[sv_key].ndim == 1:
        if linear:
            power = convert(10 * np.log10(data[sv_key]))

        else:
            power = convert(data[sv_key])


    elif data[sv_key].ndim == 2:

        power = np.empty_like(data[sv_key], dtype=np.float)
        for ping in range(data[sv_key].shape[1]):

            if linear:
                power[:, ping] = convert(10 * np.log10(data[sv_key][:, ping]))

            else:
                power[:, ping] = convert(data[sv_key][:, ping])

    else:
        raise ValueError('Expected a 1- or 2-dimensional array')


    return power


def power_to_Sp(data, gain, tvg_correction=0.0, calibration=None, linear=False, raw=False):
    '''
    :param data: Raw power (on dB scale) to calculate Sp from
    :type data: pd.Series


    :param calibration: Calibration information
    :type calibration: dict

    :param tvg_correction:  Number of samples for time-varying gain correction (default 0)
    :type tvg_correction: int

    :param linear:  Use linear units for Sv (calculate sv)
    :type linear: bool

    :param raw: Data is raw indexed power
    :type raw: bool

    Converts raw power (in dB) to Sp

    `raw`
    The 'power' data in Simrad .raw files is an indexed datatype.
        db_power = indexed_power * 10log_10(2.0) / 256

    The paramter `calibration` is a dictionary any of the following keys:

        absorption_coefficient
        sound_velocity
        frequency
        transmit_power
        gain
        offset
        count
        sample_interval

    Converts Sv -> raw power, with preference to values contained in
    `calibration` instead of the default values contained in the ping meta
    information.

    WARNING:  The following parameters are KNOWN TO CHANGE during long surveys
        sample_interval
        pulse_length

    DO NOT provide 'calibrated' values for these keys UNLESS you know what
    you're doing!

    '''
    if isinstance(calibration, dict):
        absorption_coefficient = calibration.get('absorption_coefficient', data['absorption_coefficient'])
        sound_velocity  = calibration.get('sound_velocity', data['sound_velocity'])
        frequency       = calibration.get('frequency', data['frequency'])
        transmit_power  = calibration.get('transmit_power', data['transmit_power'])
        gain            = calibration.get('gain', gain)
        offset          = calibration.get('offset', data['offset'])
        count           = calibration.get('count', data['count'])
        sample_interval = calibration.get('sample_interval', data['sample_interval'])

    else:
        absorption_coefficient = data['absorption_coefficient']
        sound_velocity  = data['sound_velocity']
        frequency       = data['frequency']
        transmit_power  = data['transmit_power']
        offset          = data['offset']
        count           = data['count']
        sample_interval = data['sample_interval']

    CSp = _calc_CSp(gain, sound_velocity, frequency, transmit_power)
    range_ = _range_vector(offset, count, sound_velocity, sample_interval, tvg_correction)
    tvg = range_.copy()
    tvg[tvg == 0] = 1
    tvg = 40 * np.log10(tvg)
    tvg[tvg < 0] = 0

    if raw:
        raw_factor = 10.0 * np.log10(2.0) / 256.0
    else:
        raw_factor = 1

    convert = lambda x: x * raw_factor + tvg + 2 * absorption_coefficient * range_ - CSp

    if data['power'].ndim == 1:
        if linear:
            Sp = 10 ** (convert(data['power']) / 10.0)

        else:
            Sp = convert(data['power'])

    elif data['power'].ndim == 2:

        Sp = np.empty_like(data['power'], dtype=np.float)
        for ping in range(data['power'].shape[1]):

            if linear:
                Sp[:, ping] = 10 ** (convert(data['power'][:, ping]) / 10.0)

            else:
                Sp[:, ping] = convert(data['power'][:, ping])

    else:
        raise ValueError('Expected a 1- or 2-dimensional array')

    return Sp


def Sp_to_power(data, tvg_correction=2.0, calibration=None, linear=False,
                raw=True):
    '''
    :param data: Sp (dB scale) or sp (linear) to calculate raw power from
    :type data: pd.Series


    :param calibration: Calibration information
    :type calibration: dict

    :param tvg_correction:  Number of samples for time-varying gain correction (default 2)
    :type tvg_correction: int

    :param linear:  Use linear units for Sv (calculate sv)
    :type linear: bool

    :param raw: Return raw indexed power
    :type raw: bool

    Converts Sp to raw power (in dB)

    `raw`
    The 'power' data in Simrad .raw files is an indexed datatype.
        db_power = indexed_power * 10log_10(2.0) / 256

    The paramter `calibration` is a dictionary any of the following keys:

        absorption_coefficient
        sound_velocity
        frequency
        transmit_power
        gain
        offset
        count
        sample_interval

    Converts Sv -> raw power, with preference to values contained in
    `calibration` instead of the default values contained in the ping meta
    information.

    WARNING:  The following parameters are KNOWN TO CHANGE during long surveys
        sample_interval
        pulse_length

    DO NOT provide 'calibrated' values for these keys UNLESS you know what
    you're doing!

    '''
    #Empty arrays -- return empty result
    if np.prod(data.shape) == 0:
        return np.empty((0,))

    if isinstance(calibration, dict):
        absorption_coefficient = calibration.get('absorption_coefficient', data['absorption_coefficient'])
        sound_velocity  = calibration.get('sound_velocity', data['sound_velocity'])
        frequency       = calibration.get('frequency', data['frequency'])
        transmit_power  = calibration.get('transmit_power', data['transmit_power'])
        gain            = calibration.get('gain', data['gain'])
        offset          = calibration.get('offset', data['offset'])
        count           = calibration.get('count', data['count'])
        sample_interval = calibration.get('sample_interval', data['sample_interval'])

    else:
        absorption_coefficient = data['absorption_coefficient']
        sound_velocity  = data['sound_velocity']
        frequency       = data['frequency']
        transmit_power  = data['transmit_power']
        gain            = data['gain']
        offset          = data['offset']
        count           = data['count']
        sample_interval = data['sample_interval']

    CSp = _calc_CSp(gain, sound_velocity, frequency, transmit_power)
    range_ = _range_vector(offset, count, sound_velocity, sample_interval, tvg_correction)
    tvg = range_.copy()
    tvg[tvg == 0] = 1
    tvg = 40 * np.log10(tvg)
    tvg[tvg < 0] = 0

    if linear:
        sp_key = 'sp'
    else:
        sp_key = 'Sp'

    if raw:
        raw_factor = 10.0 * np.log10(2.0) / 256.0
    else:
        raw_factor = 1.0

    convert = lambda x: (x - tvg - 2 * absorption_coefficient * range_ + CSp) / raw_factor

    if data[sp_key].ndim == 1:
        if linear:
            power = convert(10 * np.log10(data[sp_key]))

        else:
            power = convert(data[sp_key])


    elif data[sp_key].ndim == 2:

        power = np.empty_like(data[sp_key], dtype=np.float)
        for ping in range(data[sp_key].shape[1]):

            if linear:
                power[:, ping] = convert(10 * np.log10(data[sp_key][:, ping]))
            else:
                power[:, ping] = convert(data[sp_key][:, ping])

    else:
        raise ValueError('Expected a 1- or 2-dimensional array')

    return power


#  functions dealing with indexed angles have been removed. We operate with either
#  electrical angles or physical angles and convert directly between the two.

def electrical_to_physical_angle(data, sensitivity, offset):
    '''
    Convert electrical angles to physical angles

    :param data:
    :type data:  numpy.ndarray

    :param sensitivity:  Angle sensitivity from calibration
    :type sensitivity: float

    :param offset:  Angle offset from calibration
    :type offset: float
    '''

    return ((data * 180.0 / 128.0) / sensitivity) - offset


def physical_to_electrical_angle(data, sensitivity, offset):
    '''
    Convert physical angles to electrical angles

    :param data:
    :type data:  numpy.ndarray

    :param sensitivity:  Angle sensitivity from calibration
    :type sensitivity: float

    :param offset:  Angle offset from calibration
    :type offset: float
    '''

    return (((data + offset) * sensitivity) * 128.0) / 180.
