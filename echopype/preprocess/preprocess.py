"""
Functions for enhancing the spatial and temporal coherence of data.
"""

import numpy as np
from ..utils import uwa


def _check_range_uniqueness(ds):
    """Check if range changes across ping in a given frequency channel.
    """
    return (ds['range'].isel(ping_time=0) == ds['range']).all()


def compute_MVBS(ds_Sv, range_bin_bin=10, ping_time_bin=10):
    """Compute Mean Volume Backscattering Strength (MVBS) based on physical units.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        calibrated Sv dataset
    range_bin_bin : Union[int, float]
        bin size along ``range`` in meters
    ping_time_bin : Union[int, float]
        bin size along ``ping_time`` in seconds

    Returns
    -------
    a data set containing bin-averaged Sv
    """

    if not ds_Sv.groupby('frequency').apply(_check_range_uniqueness).all():
        raise ValueError("`range` variable changes across pings in at least one of the frequency channel.")

    def _freq_MVBS(ds, rint, pbin):
        sv = 10 ** (ds['Sv'] / 10)  # average should be done in linear domain
        sv.coords['range_meter'] = (['range_bin'], ds_Sv['range'].isel(frequency=0, ping_time=0))
        sv = sv.swap_dims({'range_bin': 'range_meter'})
        sv_groupby_bins = (
            sv.groupby_bins('range_meter', bins=rint, right=False, include_lowest=True).mean()
                .resample(ping_time=str(pbin) + 'S', skipna=True).mean()
        )
        sv_groupby_bins.coords['range'] = (['range_meter_bins'], range_interval[:-1])
        sv_groupby_bins = sv_groupby_bins.swap_dims({'range_meter_bins': 'range'})
        sv_groupby_bins = sv_groupby_bins.drop_vars('range_meter_bins')
        return 10 * np.log10(sv_groupby_bins)

    # Groupby freq in case of different range (from different sampling intervals)
    range_interval = np.arange(0, ds_Sv['range'].max() + range_bin_bin, range_bin_bin)
    MVBS = ds_Sv.groupby('frequency').apply(_freq_MVBS, args=(range_interval, ping_time_bin))

    # Attach attributes
    MVBS = MVBS.to_dataset()
    MVBS.attrs = {
        'mode': 'physical units',
        'range_meter_bin': str(range_bin_bin) + 'm',
        'ping_time_bin': str(ping_time_bin) + 'S'
    }

    return MVBS


def regrid():
    return 1


def remove_noise(ds_Sv, env_params, ping_num, range_bin_num, SNR=3, save_noise_est=False):
    """Remove noise by using estimates of background noise.

    The background noise is estimated from the minimum mean calibrated power level
    along each column of tiles.

    Reference: De Robertis & Higginbottom, 2007, ICES Journal of Marine Sciences

    Parameters
    ----------
    ds_Sv : xarray.Dataset
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
    save_noise_est : bool
        whether to save noise estimates in output, default to `False`

    Returns
    -------
    Sv_corr : xarray.Dataset
        dataset containing the denoised Sv, range, and noise estimates

    """
    # TODO: @leewujung: incorporate an user-specified upper limit of noise level

    # Obtain sound absorption coefficients
    if 'sound_absorption' not in env_params:
        sound_absorption = uwa.calc_absorption(
            frequency=ds_Sv.frequency,
            temperature=env_params['temperature'],
            salinity=env_params['salinity'],
            pressure=env_params['pressure'],
        )
        p_to_store = ['temperature', 'salinity', 'pressure']
    else:
        if env_params['sound_absorption'].frequency == ds_Sv.frequency:
            sound_absorption = env_params['sound_absorption']
            p_to_store = ['sound_absorption']
        else:
            raise ValueError("Mismatch in the frequency dimension for sound_absorption and Sv!")

    # Transmission loss
    spreading_loss = 20 * np.log10(ds_Sv['range'].where(ds_Sv['range'] >= 1, other=1))
    absorption_loss = 2 * sound_absorption * ds_Sv['range']

    # Noise estimates
    power_cal = ds_Sv['Sv'] - spreading_loss - absorption_loss  # calibrated power
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
    fac = 10 ** (ds_Sv['Sv'] / 10) - 10 ** (Sv_noise / 10)
    Sv_corr = 10 * np.log10(fac.where(fac > 0, other=np.nan))
    Sv_corr = Sv_corr.where(Sv_corr - Sv_noise > SNR, other=np.nan)

    Sv_corr.name = 'Sv_clean'
    Sv_corr = Sv_corr.to_dataset()

    # Attach other variables and attributes to dataset
    Sv_corr['range'] = ds_Sv['range'].copy()
    if save_noise_est:
        Sv_corr['Sv_noise'] = Sv_noise
    Sv_corr.attrs['noise_est_ping_num'] = ping_num
    Sv_corr.attrs['noise_est_range_bin_num'] = range_bin_num
    Sv_corr.attrs['SNR'] = SNR
    for p in p_to_store:
        Sv_corr.attrs[p] = env_params[p]

    return Sv_corr
