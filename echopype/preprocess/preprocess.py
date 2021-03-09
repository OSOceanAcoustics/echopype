"""
Functions for enhancing the spatial and temporal coherence of data.
"""

import numpy as np
from ..utils import uwa


def compute_MVBS():
    return 1


def regrid():
    return 1


def remove_noise(Sv, env_params, ping_num, range_bin_num, SNR=3):
    """Remove noise by using estimates of background noise.

    The background noise is estimated from the minimum mean calibrated power level
    along each column of tiles.

    Reference: De Robertis & Higginbottom, 2007, ICES Journal of Marine Sciences

    Parameters
    ----------
    Sv : xarray.Dataset
        dataset containing Sv and range [m]
    env_params : dict
        environmental parameters, either the sound absorption coefficients for all frequencies,
        or salinity, temperature, pressure for computing the sound absorption coefficients
    range_bin_num : int
        number of sample intervals along range to obtain noise estimates
    ping_num : int
        number of pings to obtain noise estimates
    SNR : float
        acceptable signal-to-noise ratio

    Returns
    -------
    Sv_corr : xarray.Dataset
        dataset containing the denoised Sv, range, and noise estimates

    """
    # TODO: @leewujung: incorporate an user-specified upper limit of noise level

    # Obtain sound absorption coefficients
    if 'sound_absorption' not in env_params:
        sound_absorption = uwa.calc_absorption(
            frequency=Sv.frequency,
            temperature=env_params['temperature'],
            salinity=env_params['salinity'],
            pressure=env_params['pressure'],
        )
    else:
        if env_params['sound_absorption'].frequency == Sv.frequency:
            sound_absorption = env_params['sound_absorption']
        else:
            raise ValueError("Mismatch in the frequency dimension for sound_absorption and Sv!")

    # Transmission loss
    spreading_loss = 20 * np.log10(Sv['range'].where(Sv['range'] >= 1, other=1))
    absorption_loss = 2 * sound_absorption * Sv['range']

    # Noise estimates
    power_cal = Sv['Sv'] - spreading_loss - absorption_loss  # calibrated power
    power_cal_binned_avg = 10 * np.log10(  # binned averages of calibrated power
        (10 ** (power_cal / 10)).coarsen(
            ping_time=ping_num,
            range_bin=range_bin_num,
            boundary='pad'
        ).mean())
    noise_est = power_cal_binned_avg.min(dim='range_bin')
    noise_est['ping_time'] = power_cal['ping_time'][::ping_num]
    Sv_noise = (noise_est.reindex({'ping_time': power_cal['ping_time']}, method='ffill')  # forward fill empty index
                + spreading_loss + absorption_loss)

    # Sv corrected for noise
    fac = 10 ** (Sv['Sv'] / 10) - 10 ** (Sv_noise / 10)
    Sv_corr = 10 * np.log10(fac.where(fac > 0, other=np.nan))
    Sv_corr = Sv_corr.where(Sv_corr - Sv_noise > SNR, other=np.nan)

    Sv_corr.name = 'Sv_clean'
    Sv_corr = Sv_corr.to_dataset()

    # Attach calculated range into data set
    Sv_corr['range'] = Sv['range'].copy()
    Sv_corr['Sv_noise'] = Sv_noise

    return Sv_corr
