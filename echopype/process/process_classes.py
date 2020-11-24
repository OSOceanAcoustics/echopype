from datetime import datetime as dt
import numpy as np
import xarray as xr
import dask
from scipy import signal
import os
from .echodata import EchoDataBase
from ..utils import uwa
from ..utils import io


class ProcessBase:
    """Class for processing sonar data.
    """
    def __init__(self, model=None):
        self.sonar_model = model   # type of echosounder

    def calc_sound_speed(self, ed, env_params=None, src='user'):
        """Base method for calculating sound speed.
        """
        if src == 'file':
            with ed._open_dataset(ed.raw_path, group="Environment") as ds_env:
                if 'sound_speed_indicative' in ds_env:
                    return ds_env.sound_speed_indicative
                else:
                    ValueError("Sound speed not found in file")
        elif src == 'user':
            if env_params is None:
                raise ValueError("`env_params` required for calculating sound speed")
            ss = uwa.calc_sound_speed(salinity=env_params['sea_water_salinity'],
                                      temperature=env_params['sea_water_temperature'],
                                      pressure=env_params['sea_water_pressure'])
            return ss
        else:
            ValueError("Not sure how to update sound speed!")

    def calc_seawater_absorption(self, ed, env_params, src, formula_source):
        """Base method for calculating seawater absorption.
        """
        if src != 'user':
            raise ValueError("'src' can only be 'user'")
        with ed._open_dataset(ed.raw_path, group='Beam') as ds_beam:
            freq = ds_beam.frequency.astype(np.int64)  # should already be in unit [Hz]
        return uwa.calc_seawater_absorption(freq,
                                            temperature=env_params['sea_water_temperature'],
                                            salinity=env_params['sea_water_salinity'],
                                            pressure=env_params['sea_water_pressure'],
                                            formula_source=formula_source)

    def calc_sample_thickness(self, ed):
        """Base method for calculating sample thickness.
        This method is only used for EK echosounders.
        """

    def calc_range(self, ed):
        """Base method for calculating range.

        Parameters
        ----------
        ed : EchoDataBase
        """

    def get_Sv(self, ed, env_params, cal_params, save=True, save_format='zarr'):
        """Base method to be overridden for calculating Sv from raw backscatter data.
        """
        # Issue warning when subclass methods not available
        print('Calibration has not been implemented for this sonar model!')

    def get_TS(self, ed, env_params, cal_params, save=True, save_format='zarr'):
        """Base method to be overridden for calculating TS from raw backscatter data.
        """
        # Issue warning when subclass methods not available
        print('Calibration has not been implemented for this sonar model!')

    def get_MVBS(self):
        """Calculate Mean Volume Backscattering Strength (MVBS).

        The calculation uses class attributes MVBS_ping_size and MVBS_range_bin_size to
        calculate and save MVBS as a new attribute to the calling Process instance.
        MVBS is an xarray DataArray with dimensions ``ping_time`` and ``range_bin``
        that are from the first elements of each tile along the corresponding dimensions
        in the original Sv or Sv_clean DataArray.
        """
        # Issue warning when subclass methods not available
        print('Calibration has not been implemented for this sonar model!')

    def remove_noise(self):
        """Remove noise by using noise estimates obtained from the minimum mean calibrated power level
        along each column of tiles.

        See method noise_estimates() for details of noise estimation.
        Reference: De Robertis & Higginbottom, 2007, ICES Journal of Marine Sciences
        """

    def get_noise_estimates(self):
        """Obtain noise estimates from the minimum mean calibrated power level along each column of tiles.

        The tiles here are defined by class attributes noise_est_range_bin_size and noise_est_ping_size.
        This method contains redundant pieces of code that also appear in method remove_noise(),
        but this method can be used separately to determine the exact tile size for noise removal before
        noise removal is actually performed.
        """


class ProcessAZFP(ProcessBase):
    """
    Class for processing data from ASL Env Sci AZFP echosounder.
    """
    def __init__(self, model='AZFP'):
        super().__init__(model)

    def calc_seawater_absorption(self, ed, env_params, src='user', formula_source='AZFP', param_source=None):
        """Calculate sound absorption using AZFP formula.
        """
        if src != 'user':
            raise ValueError("'src' can only be 'user'")
        try:
            f0 = ed.raw_dataset.frequency_start
            f1 = ed.raw_dataset.frequency_end
            freq = (f0 + f1) / 2
        except AttributeError:
            freq = ed.raw_dataset.frequency
        return uwa.calc_seawater_absorption(freq,
                                            temperature=env_params['sea_water_temperature'],
                                            salinity=env_params['sea_water_salinity'],
                                            pressure=env_params['sea_water_pressure'],
                                            formula_source=formula_source)

    def calc_range(self, ed, env_params, tilt_corrected=False):
        """Calculates range in meters using AZFP formula,
        instead of from sample_interval directly.
        """

        ds_vend = ed._open_dataset(ed.raw_path, group='Vendor')

        # WJ: same as "range_samples_per_bin" used to calculate "sample_interval"
        range_samples = ds_vend.number_of_samples_per_average_bin
        pulse_length = ed.raw.transmit_duration_nominal   # units: seconds
        bins_to_avg = 1   # set to 1 since we want to calculate from raw data
        sound_speed = env_params['speed_of_sound_in_sea_water']
        dig_rate = ds_vend.digitization_rate
        lockout_index = ds_vend.lockout_index

        # Below is from LoadAZFP.m, the output is effectively range_bin+1 when bins_to_avg=1
        range_mod = xr.DataArray(np.arange(1, len(ed.raw.range_bin) - bins_to_avg + 2, bins_to_avg),
                                 coords=[('range_bin', ed.raw.range_bin)])

        # Calculate range using parameters for each freq
        range_meter = (lockout_index / (2 * dig_rate) * sound_speed + sound_speed / 4 *
                       (((2 * range_mod - 1) * range_samples * bins_to_avg - 1) / dig_rate +
                        pulse_length))

        if tilt_corrected:
            range_meter = ed.raw.cos_tilt_mag.mean() * range_meter

        ds_vend.close()

        return range_meter

    def get_Sv(self, ed, env_params, cal_params=None, save=True, save_path=None, save_format='zarr'):
        """Calibrate to get volume backscattering strength (Sv) from AZFP power data.

        The calibration formula used here is documented in eq.(9) on p.85
        of GU-100-AZFP-01-R50 Operator's Manual.
        Note a Sv_offset factor that varies depending on frequency is used
        in the calibration as documented on p.90.
        See calc_Sv_offset() in convert/azfp.py
        """
        if ed.range is None:
            ed.range = self.calc_range(ed, env_params)

        Sv = (ed.raw.EL - 2.5 / ed.raw.DS +
              ed.raw.backscatter_r / (26214 * ed.raw.DS) -
              ed.raw.TVR - 20 * np.log10(ed.raw.VTX) + 20 * np.log10(ed.range) +
              2 * env_params['seawater_absorption'] * ed.range -
              10 * np.log10(0.5 * env_params['speed_of_sound_in_sea_water'] *
                            ed.raw.transmit_duration_nominal *
                            ed.raw.equivalent_beam_angle) + ed.raw.Sv_offset)

        Sv.name = 'Sv'
        Sv = Sv.to_dataset()

        # Attached calculated range into the dataset
        Sv['range'] = (('frequency', 'ping_time', 'range_bin'), ed.range)

        # Save calibrated data into the calling instance and
        #  to a separate .nc file in the same directory as the data filef.Sv = Sv
        if save:
            # Update pointer in EchoData
            Sv_path = io.validate_proc_path(ed, '_Sv', save_path)
            print("{} saving calibrated Sv to {}".format(dt.now().strftime('%H:%M:%S'), Sv_path))
            ed._save_dataset(Sv, Sv_path, mode="w", save_format=save_format)
            ed.Sv_path = Sv_path
        else:
            # TODO Add to docs
            ed.Sv = Sv

    def get_TS(self, ed, env_params, cal_params=None, save=True, save_path=None, save_format='zarr'):
        """Calibrate to get Target Strength (TS) from AZFP power data.
        """
        if ed.range is None:
            ed.range = self.calc_range(ed, env_params)

        TS = (ed.raw.EL - 2.5 / ed.raw.DS +
              ed.raw.backscatter_r /(26214 * ed.raw.DS) -
              ed.raw.TVR - 20 * np.log10(ed.raw.VTX) + 40 * np.log10(ed.range) +
              2 * env_params['seawater_absorption'] * ed.range)

        TS.name = "TS"
        TS = TS.to_dataset()

        # Attached calculated range into the dataset
        TS['range'] = (('frequency', 'ping_time', 'range_bin'), ed.range)

        if save:
            # Update pointer in EchoData
            TS_path = io.validate_proc_path(ed, '_TS', save_path)
            print("{} saving calibrated TS to {}".format(dt.now().strftime('%H:%M:%S'), TS_path))
            ed._save_dataset(TS, TS_path, mode="w", save_format=save_format)
            ed.TS_path = TS_path
        else:
            ed.TS = TS


class ProcessEK(ProcessBase):
    """
    Class for processing data from Simrad EK echosounders.
    """
    def __init__(self, model=None):
        super().__init__(model)

    def calc_sample_thickness(self, ed, env_params):
        """Calculate sample thickness.
        """
        return env_params['speed_of_sound_in_sea_water'] * ed.raw.sample_interval / 2

    def _cal_narrowband(self, ed, env_params, cal_params, cal_type,
                        save=True, save_path=None, save_format='zarr'):
        """Calibrate narrowband data from EK60 and EK80.
        """
        # Derived params
        wavelength = env_params['speed_of_sound_in_sea_water'] / ed.raw.frequency  # wavelength
        if ed.range is None:
            ed.range = self.calc_range(ed, env_params)
        # Get TVG and absorption
        TVG = np.real(20 * np.log10(ed.range.where(ed.range >= 1, other=1)))
        ABS = 2 * env_params['seawater_absorption'] * ed.range
        if cal_type == 'Sv':
            # Print raw data nc file

            # Calc gain
            CSv = 10 * np.log10((cal_params['transmit_power'] * (10 ** (cal_params['gain_correction'] / 10)) ** 2 *
                                wavelength ** 2 * env_params['speed_of_sound_in_sea_water'] *
                                cal_params['transmit_duration_nominal'] *
                                10 ** (cal_params['equivalent_beam_angle'] / 10)) /
                                (32 * np.pi ** 2))

            # Calibration and echo integration
            Sv = ed.raw.backscatter_r + TVG + ABS - CSv - 2 * cal_params['sa_correction']
            Sv.name = 'Sv'
            Sv = Sv.to_dataset()

            # Attach calculated range into data set
            Sv['range'] = (('frequency', 'ping_time', 'range_bin'), ed.range)

            # Save calibrated data into the calling instance and
            #  to a separate .nc file in the same directory as the data filef.Sv = Sv
            if save:
                # Update pointer in EchoData
                Sv_path = io.validate_proc_path(ed, '_Sv', save_path)
                print("{} saving calibrated Sv to {}".format(dt.now().strftime('%H:%M:%S'), Sv_path))
                ed._save_dataset(Sv, Sv_path, mode="w", save_format=save_format)
                ed.Sv_path = Sv_path
            else:
                ed.Sv = Sv
        elif cal_type == 'TS':
            # calculate TS
            # Open data set for Environment and Beam groups

            # Calc gain
            CSp = 10 * np.log10((cal_params['transmit_power'] *
                                (10 ** (cal_params['gain_correction'] / 10)) ** 2 *
                                wavelength ** 2) / (16 * np.pi ** 2))

            # Calibration and echo integration
            TS = ed.raw.backscatter_r + TVG * 2 + ABS - CSp
            TS.name = 'TS'
            TS = TS.to_dataset()

            # Attach calculated range into data set
            TS['range'] = (('frequency', 'ping_time', 'range_bin'), ed.range)

            if save:
                # Update pointer in EchoData
                TS_path = io.validate_proc_path(ed, '_TS', save_path)
                print("{} saving calibrated TS to {}".format(dt.now().strftime('%H:%M:%S'), TS_path))
                ed._save_dataset(TS, TS_path, mode="w", save_format=save_format)
                ed.TS_path = TS_path
            else:
                ed.TS = TS


class ProcessEK60(ProcessEK):
    """
    Class for processing data from Simrad EK60 echosounder.
    """
    def __init__(self, model='EK60'):
        super().__init__(model)

        self.tvg_correction_factor = 2  # EK60 specific parameter

    def get_Sv(self, ed, env_params, cal_params=None, save=True, save_path=None, save_format='zarr'):
        """Calibrate to get volume backscattering strength (Sv) from EK60 data.
        """
        return self._cal_narrowband(ed=ed,
                                    env_params=env_params,
                                    cal_params=cal_params,
                                    cal_type='Sv',
                                    save=save,
                                    save_path=save_path,
                                    save_format=save_format)

    def get_TS(self, ed, env_params, cal_params=None, save=True, save_path=None, save_format='zarr'):
        """Calibrate to get target strength (TS) from EK60 data.
        """
        return self._cal_narrowband(ed=ed,
                                    env_params=env_params,
                                    cal_params=cal_params,
                                    cal_type='TS',
                                    save=save,
                                    save_path=save_path,
                                    save_format=save_format)

    def calc_range(self, ed, env_params):
        """Calculates range in meters.
        """
        st = self.calc_sample_thickness(ed, env_params)
        range_meter = st * ed.raw.range_bin - \
            self.tvg_correction_factor * st  # DataArray [frequency x range_bin]
        range_meter = range_meter.where(range_meter > 0, other=0)
        return range_meter


class ProcessEK80(ProcessEK):
    """
    Class for processing data from Simrad EK80 echosounder.
    """
    def __init__(self, model='EK80'):
        super().__init__(model)

    def get_calibration_params(self, ed):
        pass

    def calc_transmit_signal(self, ed, cal_params):
        """Generate transmit signal as replica for pulse compression.
        """
        def chirp_linear(t, f0, f1, tau):
            beta = (f1 - f0) * (tau ** -1)
            return np.cos(2 * np.pi * (beta / 2 * (t ** 2) + f0 * t))

        # Retrieve filter coefficients
        with ed._open_dataset(ed.raw_path, group="Vendor") as ds_fil:
            # Get various parameters
            Ztrd = 75  # Transducer quadrant nominal impedance [Ohms] (Supplied by Simrad)
            delta = 1 / 1.5e6   # Hard-coded EK80 sample interval
            tau = cal_params['transmit_duration_nominal'].values
            tx_power = cal_params['transmit_power'].values
            slope = cal_params['slope'].values

            # Find indexes with a unique transmitted signal.
            # Get outer product of 3 parameters
            beam_params = np.einsum('i,j,k->ijk', tau[0], slope[0], tx_power[0])
            # Get diagonal of 3D matix
            _, unique_signal_idx, counts = np.unique(beam_params[np.diag_indices(tau.shape[1], ndim=3)],
                                                     return_index=True, return_counts=True)
            # Re order beam parameters because np.unique sorts the output
            sort_idx = np.argsort(unique_signal_idx)
            unique_signal_idx = unique_signal_idx[sort_idx]
            counts = counts[sort_idx]
            tau = tau[:, unique_signal_idx]
            tx_power = tx_power[:, unique_signal_idx]
            slope = slope[:, unique_signal_idx]

            amp = np.sqrt((tx_power / 4) * (2 * Ztrd))
            f0 = ed.raw.frequency_start.values
            f1 = ed.raw.frequency_end.values
            ch_ids = ed.raw.channel_id.values

            ytx = []
            # Loop over each unique transmit signal
            for i in range(tau.shape[1]):
                ytx_ping = []
                # Create transmit signal
                for ch in range(ed.raw.frequency.size):
                    t = np.arange(0, tau[ch][i], delta)
                    nt = len(t)
                    nwtx = (int(2 * np.floor(slope[ch][i] * nt)))
                    wtx_tmp = np.hanning(nwtx)
                    nwtxh = (int(np.round(nwtx / 2)))
                    wtx = np.concatenate([wtx_tmp[0:nwtxh], np.ones((nt - nwtx)), wtx_tmp[nwtxh:]])
                    y_tmp = amp[ch][i] * chirp_linear(t, f0[ch], f1[ch], tau[ch][i]) * wtx
                    # The transmit signal must have a max amplitude of 1
                    y = (y_tmp / np.max(np.abs(y_tmp)))

                    # filter and decimation
                    wbt_fil = ds_fil.attrs[ch_ids[ch] + '_WBT_filter_r'] + 1j * \
                        ds_fil.attrs[ch_ids[ch] + "_WBT_filter_i"]
                    pc_fil = ds_fil.attrs[ch_ids[ch] + '_PC_filter_r'] + 1j * \
                        ds_fil.attrs[ch_ids[ch] + '_PC_filter_i']

                    # Apply WBT filter and downsample
                    ytx_tmp = np.convolve(y, wbt_fil)
                    ytx_tmp = ytx_tmp[0::ds_fil.attrs[ch_ids[ch] + "_WBT_decimation"]]

                    # Apply PC filter and downsample
                    ytx_tmp = np.convolve(ytx_tmp, pc_fil)
                    ytx_tmp = ytx_tmp[0::ds_fil.attrs[ch_ids[ch] + "_PC_decimation"]]
                    ytx_ping.append(ytx_tmp)
                    del nwtx, wtx_tmp, nwtxh, wtx, y_tmp, y, ytx_tmp
                ytx.extend([ytx_ping for n in range(counts[i])])

            return np.array(ytx).T

    def pulse_compression(self, ed, cal_params, tx_signal):
        """Pulse compression using transmit signal as replica.
        """
        backscatter = ed.raw.backscatter_r + ed.raw.backscatter_i * 1j  # Construct complex backscatter
        nfreq, _, npings, _ = backscatter.shape
        backscatter_compressed = []
        tau_constants = []

        def rectangularize(v):
            lens = np.array([len(item) for item in v])
            mask = lens[:, None] > np.arange(lens.max())
            out = np.full(mask.shape, np.nan, dtype='complex128')
            out[mask] = np.concatenate(v)
            return out

        def compress(ping_idx):
            # Convolve tx signal with backscatter. atol=1e-7 between fft and direct convolution
            compressed = np.apply_along_axis(signal.convolve, axis=1,
                                             arr=tmp_b[:, ping_idx, :],
                                             in2=np.flipud(tmp_y[ping_idx])) / \
                np.linalg.norm(tx_signal[ch][ping_idx]) ** 2
            return np.mean(compressed, axis=0)

        def calc_effective_pulse_length(ping_idx):
            ptxa = np.square(np.abs(signal.convolve(tx_signal[ch][ping_idx],
                                    np.flipud(tmp_y[ping_idx]), method='direct') /
                                    np.linalg.norm(tx_signal[ch][ping_idx]) ** 2))
            return np.sum(ptxa) / (np.max(ptxa))

        # Loop over channels
        for ch in range(nfreq):
            # tmp_x = np.fft.fft(backscatter[i].dropna('range_bin'))
            # tmp_y = np.fft.fft(np.flipud(np.conj(ytx[i])))
            # remove quadrants that are nans across all samples
            tmp_b = backscatter[ch].dropna('range_bin', how='all')
            # remove samples that are nans across all quadrants
            tmp_b = tmp_b.dropna('quadrant', how='all')
            # tmp_b = tmp_b[:, 0, :]        # 1 ping
            tmp_y = np.conj(tx_signal[ch])

            # backscatter_compressed.append(compressed)
            # TODO: try out xrft for convolution
            backscatter_lazy = [dask.delayed(compress)(i) for i in range(npings)]
            backscatter_compressed.append(dask.compute(*backscatter_lazy))
            backscatter_compressed[-1] = rectangularize(np.array(backscatter_compressed[-1]))
            backscatter_compressed[-1] = xr.DataArray(
                np.expand_dims(backscatter_compressed[-1], 0),
                coords={
                    'frequency': [backscatter[ch].frequency],
                    'ping_time': backscatter.ping_time,
                    'range_bin': np.arange(backscatter_compressed[-1].shape[1]),
                    'frequency_start': backscatter[ch].frequency_start,
                    'frequency_end': backscatter[ch].frequency_end
                },
                dims=['frequency', 'ping_time', 'range_bin'])

            # Effective pulse length TODO: calculate unique
            ptxa_lazy = [dask.delayed(calc_effective_pulse_length)(i) for i in range(npings)]
            tau_constants.append(dask.compute(*ptxa_lazy))

        tau_effective = tau_constants * cal_params['sample_interval']
        backscatter_compressed = xr.concat(backscatter_compressed, dim='frequency')

        return backscatter_compressed, tau_effective

    def _cal_broadband(self, ed, env_params, cal_params, cal_type,
                       save=True, save_path=None, save_format='zarr'):
        """Calibrate broadband EK80 data.
        """
        Ztrd = 75       # Transducer quadrant nominal impedance [Ohms] (Supplied by Simrad)
        Rwbtrx = 1000   # Wideband transceiver impedance [Ohms] (Supplied by Simrad)
        tx_signal = self.calc_transmit_signal(ed, cal_params)  # Get transmit signal
        backscatter_compressed, tau_effective = self.pulse_compression(ed, cal_params, tx_signal)   # Pulse compress

        c = env_params['speed_of_sound_in_sea_water']
        tx_power = cal_params['transmit_power']
        f_nominal = ed.raw.frequency
        f_center = (ed.raw.frequency_start.data + ed.raw.frequency_end.data) / 2
        psifc = cal_params['equivalent_beam_angle'] + 20 * np.log10(f_nominal / f_center)
        la2 = (c / f_center) ** 2
        Sv = []
        TS = []

        # Take absolute value of complex backscatter
        prx = np.abs(backscatter_compressed)
        prx = prx * prx / 2 * (np.abs(Rwbtrx + Ztrd) / Rwbtrx) ** 2 / np.abs(Ztrd)
        # TODO Gfc should be gain interpolated at the center frequency
        # Only 1 gain value is given provided per channel
        Gfc = cal_params['gain_correction']
        ranges = self.calc_range(ed, env_params, range_bins=prx.shape[2])
        ranges = ranges.where(ranges >= 1, other=1)

        if cal_type == 'Sv':
            # calculate Sv
            Sv = (
                10 * np.log10(prx) + 20 * np.log10(ranges) +
                2 * env_params['seawater_absorption'] * ranges -
                10 * np.log10(tx_power * la2[:, None] * c / (32 * np.pi * np.pi)) -
                2 * Gfc - 10 * np.log10(tau_effective) - psifc
            )
        else:
            # calculate TS
            TS = (
                10 * np.log10(prx) + 40 * np.log10(ranges) +
                2 * env_params['seawater_absorption'] * ranges -
                10 * np.log10(tx_power * la2[:, None] / (16 * np.pi * np.pi)) -
                2 * Gfc
            )

        # Save Sv calibrated data
        if cal_type == 'Sv':
            Sv.name = 'Sv'
            Sv = Sv.to_dataset()
            Sv['range'] = (('frequency', 'ping_time', 'range_bin'), ranges)
            if save:
                # Update pointer in EchoData
                Sv_path = io.validate_proc_path(ed, '_Sv', save_path)
                print("{} saving calibrated Sv to {}".format(dt.now().strftime('%H:%M:%S'), Sv_path))
                ed._save_dataset(Sv, Sv_path, mode="w", save_format=save_format)
                ed.Sv_path = Sv_path
            else:
                ed.Sv = Sv
        # Save TS calibrated data
        elif cal_type == 'TS':
            TS.name = 'TS'
            TS = TS.to_dataset()
            TS['range'] = (('frequency', 'ping_time', 'range_bin'), ranges)
            if save:
                # Update pointer in EchoData
                TS_path = io.validate_proc_path(ed, '_TS', save_path)
                print("{} saving calibrated TS to {}".format(dt.now().strftime('%H:%M:%S'), TS_path))
                ed._save_dataset(TS, TS_path, mode="w", save_format=save_format)
                ed.TS_path = TS_path
            else:
                ed.TS = TS

    def _choose_mode(self, ed):
        """Choose which calibration mode to use.

        Parameters
        ----------
        mm : str
            'BB' indicates broadband calibration
            'CW' indicates narrowband calibration
        """
        if hasattr(ed.raw, 'backscatter_i'):
            return self._cal_broadband
        else:
            return self._cal_narrowband

    def get_Sv(self, ed, env_params, cal_params,
               save=True, save_path=None, save_format='zarr'):
        """Calibrate to get volume backscattering strength (Sv) from EK80 data.
        """
        self._choose_mode(ed)(ed, env_params, cal_params, 'Sv', save, save_path, save_format)

    def get_TS(self, ed, env_params, cal_params,
               save=True, save_path=None, save_format='zarr'):
        """Calibrate to get target strength (TS) from EK80 data.
        """
        self._choose_mode(ed)(ed, env_params, cal_params, 'TS', save, save_path, save_format)

    def calc_range(self, ed, env_params, range_bins=None):
        range_bin = xr.DataArray(np.arange(range_bins), coords=[np.arange(range_bins)], dims=['range_bin']) if \
            range_bins is not None else ed.raw.range_bin
        st = self.calc_sample_thickness(ed, env_params)
        range_meter = st * range_bin - ed.raw.transmit_duration_nominal * \
            env_params['speed_of_sound_in_sea_water'] / 2  # DataArray [frequency x range_bin]
        range_meter = range_meter.where(range_meter > 0, other=0)
        return range_meter
