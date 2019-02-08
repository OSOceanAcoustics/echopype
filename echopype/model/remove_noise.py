"""
Functions for removing noise from echo data
"""


import numpy as np
import xarray as xr


def get_noise_est(epm, depth_bin, N_ping_bin=10, tvg_correction_factor=2):
    """
    Estimate noise level by getting the minimum value for bins of averaged ping.

    This function is called internally by remove_noise()
    Reference: De Robertis & Higginbottom, 2017, ICES Journal of Marine Sciences

    :param epm: an object of class EchoData
    :param depth_bin: size of each depth bin used for noise estimation [m]
    :param N_ping_bin: number of pings included in ping bin used for noise estimation
    :param tvg_correction_factor: range correction for EK60 power data
    :return noise_est: estimated noise level at each frequency
    """

    # Loop through each frequency to estimate noise
    for f_seq, freq in enumerate(epm.beam.frequency.values):

        # Get range bin size
        c = epm.env.sound_speed_indicative.sel(frequency=freq).values
        t = epm.beam.sample_interval.sel(frequency=freq).values
        bin_size = c*t/2

        # Rough number of range bins for each depth bin
        N_range_bin_tmp = int(np.floor(depth_bin/bin_size))  # rough number of depth bins

        # Average uncompensated power over N_ping_bin pings and N_rang_bin range bins
        # and find minimum value of power for each averaged bin
        # TODO: Let's use xarray functionality for this average

        N_range_bin = int(np.floor((sz[0] - tvg_correction_factor) / N_range_bin_tmp))  # number of depth bins


        # Average uncompensated power over M pings and N depth bins
        # and find minimum value of power for each averaged bin
        noise_est = defaultdict(list)
        for (freq_str, vals) in self.hdf5_handle['power_data'].items():
            sz = vals.shape
            power = vals[:]  # access as a numpy ndarray
            depth_bin_num = int(np.floor((sz[0] - self.tvg_correction_factor) / N))  # number of depth bins
            ping_bin_num = int(np.floor(sz[1] / self.ping_bin))  # number of ping bins
            power_bin = np.empty([depth_bin_num, ping_bin_num])
            for iD in range(depth_bin_num):
                for iP in range(ping_bin_num):
                    depth_idx = np.arange(N) + N * iD + self.tvg_correction_factor  # match the 2-sample offset
                    ping_idx = np.arange(self.ping_bin) + self.ping_bin * iP
                    power_bin[iD, iP] = np.mean(10 ** (power[np.ix_(depth_idx, ping_idx)] / 10))
            noise_est[freq_str] = np.min(power_bin, 0)  # noise = minimum value for each averaged ping
        self.noise_est = noise_est
