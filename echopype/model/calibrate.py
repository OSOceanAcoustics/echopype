"""
Methods for calibrating sonar echo data based on converted netCDF file via xarray.

Detail of calibration may differ for data from different sonar systems.

The previous incarnation for EK60 raw calibration is under /ref_code/zplsc_b.py
"""

import numpy as np
import xarray as xr
import echopype.model as epm


def calibrate(epm):
    """
    Calibrate sonar data by calling manufacturer specific function.

    :param epm: an object of class EchoData
    :return da_Sv: volume backscattering strength an xarray DataArray
    """

    if 'EK60' in epm.ds_toplevel.keywords:
        print('Calibrating data from Simrad EK60 echosounder...')
        return calibrate_ek60(epm)


def calibrate_ek60(epm, tvg_correction_factor=2):
    """
    Perform echo-integration to get volume backscattering strength (Sv)
    from EK60 power data.

    :param epm: an object of class EchoData
    :return da_Sv: volume backscattering strength an xarray DataArray
    """

    # Loop through each frequency for calibration
    Sv = np.zeros(epm.beam.backscatter_r.shape)
    for f_seq, freq in enumerate(epm.beam.frequency.values):
        # Params from env group
        c = epm.env.sound_speed_indicative.sel(frequency=freq).values
        alpha = epm.env.absorption_indicative.sel(frequency=freq).values

        # Params from beam group
        t = epm.beam.sample_interval.sel(frequency=freq).values
        gain = epm.beam.gain_correction.sel(frequency=freq).values
        phi = epm.beam.equivalent_beam_angle.sel(frequency=freq).values
        pt = epm.beam.transmit_power.sel(frequency=freq).values
        tau = epm.beam.transmit_duration_nominal.sel(frequency=freq).values
        Sac = 2 * epm.beam.sa_correction.sel(frequency=freq).values

        # Derived params
        dR = c*t/2  # sample thickness
        wvlen = c/freq  # wavelength

        # Calc gain
        CSv = 10 * np.log10((pt * (10 ** (gain / 10))**2 *
                             wvlen**2 * c * tau * 10**(phi / 10)) /
                            (32 * np.pi ** 2))

        # Get TVG
        range_vec = np.arange(epm.beam.range_bin.size) * dR
        range_vec = range_vec - (tvg_correction_factor * dR)
        range_vec[range_vec < 0] = 0

        TVG = np.empty(range_vec.shape)
        TVG[range_vec != 0] = np.real(20 * np.log10(range_vec[range_vec != 0]))
        TVG[range_vec == 0] = 0

        # Get absorption
        ABS = 2 * alpha * range_vec

        # Compute Sv
        Sv[f_seq, :, :] = epm.beam.backscatter_r.sel(frequency=freq).values + \
                          TVG + ABS - CSv - Sac

    # Assemble an xarray DataArray
    da_Sv = xr.DataArray(Sv,
                         dims=['frequency', 'ping_time', 'range_bin'],
                         coords={'frequency': epm.beam.frequency,
                                 'ping_time': epm.beam.ping_time,
                                 'range_bin': epm.beam.range_bin},
                         attrs={'tvg_correction_factor': tvg_correction_factor})
    return da_Sv

