"""
Functions for enhancing the spatial and temporal coherence of data.
"""

import numpy as np
from ..utils import uwa


def _check_range_uniqueness(ds):
    """Check if range changes across ping in a given frequency channel.
    """
    return (ds['range'].isel(ping_time=0) == ds['range']).all()


def compute_MVBS(ds_Sv, range_meter_bin=20, ping_time_bin='20S'):
    """Compute Mean Volume Backscattering Strength (MVBS) based on physical units.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        dataset containing Sv and range [m]
    range_meter_bin : Union[int, float]
        bin size along ``range`` in meters, default to ``20``
    ping_time_bin : Union[int, float]
        bin size along ``ping_time``, default to ``20S``

    Returns
    -------
    A dataset containing bin-averaged Sv
    """

    if not ds_Sv.groupby('frequency').apply(_check_range_uniqueness).all():
        raise ValueError("`range` variable changes across pings in at least one of the frequency channel.")

    def _freq_MVBS(ds, rint, pbin):
        sv = 10 ** (ds['Sv'] / 10)  # average should be done in linear domain
        sv.coords['range_meter'] = (['range_bin'], ds_Sv['range'].isel(frequency=0, ping_time=0))
        sv = sv.swap_dims({'range_bin': 'range_meter'})
        sv_groupby_bins = (
            sv.groupby_bins('range_meter', bins=rint, right=False, include_lowest=True).mean()
                .resample(ping_time=pbin, skipna=True).mean()
        )
        sv_groupby_bins.coords['range'] = (['range_meter_bins'], rint[:-1])
        sv_groupby_bins = sv_groupby_bins.swap_dims({'range_meter_bins': 'range'})
        sv_groupby_bins = sv_groupby_bins.drop_vars('range_meter_bins')
        return 10 * np.log10(sv_groupby_bins)

    # Groupby freq in case of different range (from different sampling intervals)
    range_interval = np.arange(0, ds_Sv['range'].max() + range_meter_bin, range_meter_bin)
    MVBS = ds_Sv.groupby('frequency').apply(_freq_MVBS, args=(range_interval, ping_time_bin))

    # Attach attributes
    MVBS = MVBS.to_dataset()
    MVBS.attrs = {
        'mode': 'physical units',
        'range_meter_bin': str(range_meter_bin) + 'm',
        'ping_time_bin': ping_time_bin
    }

    return MVBS


def remove_noise(ds_Sv, ping_num, range_bin_num, noise_max=None, SNR_threshold=3):
    """Remove noise by using estimates of background noise from mean calibrated power of a collection of pings.

    Reference: De Robertis & Higginbottom. 2007.
    A post-processing technique to estimate the signal-to-noise ratio and
    remove echosounder background noise.
    ICES Journal of Marine Sciences 64(6): 1282–1291.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        dataset containing Sv and range [m]
    ping_num : int
        number of pings to obtain noise estimates
    range_bin_num : int
        number of samples along range to obtain noise estimates
    noise_max : float
        the upper limit for background noise expected under the operating conditions
    SNR_threshold : float
        acceptable signal-to-noise ratio, default to 3 dB

    Returns
    -------
    The input dataset with additional variables, including
    the corrected Sv (``Sv_corrected``)
    and the noise estimates (``Sv_noise``)
    """
    # Obtain sound absorption coefficients
    if 'sound_absorption' not in ds_Sv:
        sound_absorption = uwa.calc_absorption(
            frequency=ds_Sv.frequency,
            temperature=ds_Sv['temperature'],
            salinity=ds_Sv['salinity'],
            pressure=ds_Sv['pressure'],
        )
    else:
        sound_absorption = ds_Sv['sound_absorption']

    # Transmission loss
    spreading_loss = 20 * np.log10(ds_Sv['range'].where(ds_Sv['range'] >= 1, other=1))
    absorption_loss = 2 * sound_absorption * ds_Sv['range']

    # Noise estimates
    power_cal = 10 ** ((ds_Sv['Sv'] - spreading_loss - absorption_loss) / 10)  # calibrated power without TVG, linear
    power_cal_binned_avg = 10 * np.log10(  # binned averages of calibrated power
        power_cal.coarsen(
            ping_time=ping_num,
            range_bin=range_bin_num,
            boundary='pad'
        ).mean()
    )
    noise = power_cal_binned_avg.min(dim='range_bin', skipna=True)
    noise['ping_time'] = power_cal['ping_time'][::ping_num]  # align ping_time to first of each ping collection
    if noise_max is not None:
        noise = noise.where(noise < noise_max, noise_max)  # limit max noise level
    Sv_noise = (
            noise.reindex({'ping_time': power_cal['ping_time']}, method='ffill')  # forward fill empty index
            + spreading_loss
            + absorption_loss
    )

    # Sv corrected for noise
    fac = 10 ** (ds_Sv['Sv'] / 10) - 10 ** (Sv_noise / 10)  # linear domain
    Sv_corr = 10 * np.log10(fac.where(fac > 0, other=np.nan))
    Sv_corr = Sv_corr.where(Sv_corr - Sv_noise > SNR_threshold, other=np.nan)  # other=-999 proposed in the paper

    # Assemble output dataset
    ds_out = ds_Sv.copy()
    ds_out['Sv_corr'] = Sv_corr
    ds_out['Sv_noise'] = Sv_noise
    ds_out = ds_out.assign_attrs(
        {
            'noise_ping_num': ping_num,
            'noise_range_bin_num': range_bin_num,
            'SNR_threshold': SNR_threshold,
            'noise_max': noise_max
        }
    )

    return ds_out


def regrid():
    return 1


