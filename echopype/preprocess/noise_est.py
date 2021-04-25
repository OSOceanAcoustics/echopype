import numpy as np
import xarray as xr
from ..utils import uwa


class NoiseEst:
    """
    Attributes
    ----------
    ds_Sv : xr.Dataset
        dataset containing Sv and range [m]
    ping_num : int
        number of pings to obtain noise estimates
    range_bin_num : int
        number of samples along range to obtain noise estimates
    """

    def __init__(self, ds_Sv, ping_num, range_bin_num):
        self.ds_Sv = ds_Sv
        self.ping_num = ping_num
        self.range_bin_num = range_bin_num
        self.spreading_loss = None
        self.absorption_loss = None
        self.Sv_noise = None

        self._compute_transmission_loss()
        self._compute_power_cal()

    def _compute_transmission_loss(self):
        """Compute transmission loss
        """
        if 'sound_absorption' not in self.ds_Sv:
            sound_absorption = uwa.calc_absorption(
                frequency=self.ds_Sv.frequency,
                temperature=self.ds_Sv['temperature'],
                salinity=self.ds_Sv['salinity'],
                pressure=self.ds_Sv['pressure'],
            )
        else:
            sound_absorption = self.ds_Sv['sound_absorption']

        # Transmission loss
        self.spreading_loss = 20 * np.log10(self.ds_Sv['range'].where(self.ds_Sv['range'] >= 1, other=1))
        self.absorption_loss = 2 * sound_absorption * self.ds_Sv['range']

    def _compute_power_cal(self):
        """Compute calibrated power without TVG, linear domain
        """
        self.power_cal = 10 ** ((self.ds_Sv['Sv'] - self.spreading_loss - self.absorption_loss) / 10)

    def estimate_noise(self, noise_max=None):
        """Estimate noise from a collected of pings

        Parameters
        ----------
        noise_max : Union[int, float]
            the upper limit for background noise expected under the operating conditions
        """
        power_cal_binned_avg = 10 * np.log10(  # binned averages of calibrated power
            self.power_cal.coarsen(
                ping_time=self.ping_num,
                range_bin=self.range_bin_num,
                boundary='pad'
            ).mean()
        )
        noise = power_cal_binned_avg.min(dim='range_bin', skipna=True)

        # align ping_time to first of each ping collection
        noise['ping_time'] = self.power_cal['ping_time'][::self.ping_num]

        if noise_max is not None:
            noise = noise.where(noise < noise_max, noise_max)  # limit max noise level
        self.Sv_noise = (
                noise.reindex({'ping_time': self.power_cal['ping_time']}, method='ffill')  # forward fill empty index
                + self.spreading_loss
                + self.absorption_loss
        )

    def remove_noise(self, noise_max=None, SNR_threshold=3):
        """Remove noise by using estimates of background noise from mean calibrated power of a collection of pings.

        This method adds two data variables to the input ``ds_Sv``:
        - corrected Sv (``Sv_corrected``)
        - noise estimates (``Sv_noise``)

        Reference: De Robertis & Higginbottom. 2007.
        A post-processing technique to estimate the signal-to-noise ratio and remove echosounder background noise.
        ICES Journal of Marine Sciences 64(6): 1282–1291.

        Parameters
        ----------
        noise_max : float
            the upper limit for background noise expected under the operating conditions
        SNR_threshold : float
            acceptable signal-to-noise ratio, default to 3 dB
        """
        # Compute Sv_noise
        self.estimate_noise(noise_max=noise_max)

        # Sv corrected for noise
        fac = 10 ** (self.ds_Sv['Sv'] / 10) - 10 ** (self.Sv_noise / 10)  # linear domain
        Sv_corr = 10 * np.log10(fac.where(fac > 0, other=np.nan))
        Sv_corr = Sv_corr.where(Sv_corr - self.Sv_noise > SNR_threshold, other=np.nan)  # other=-999 (from paper)

        # Assemble output dataset
        self.ds_Sv['Sv_corrected'] = Sv_corr
        self.ds_Sv['Sv_noise'] = self.Sv_noise
        self.ds_Sv = self.ds_Sv.assign_attrs(
            {
                'noise_ping_num': self.ping_num,
                'noise_range_bin_num': self.range_bin_num,
                'SNR_threshold': SNR_threshold,
                'noise_max': noise_max
            }
        )



