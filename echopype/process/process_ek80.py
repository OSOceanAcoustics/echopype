from datetime import datetime as dt
import numpy as np
import xarray as xr
import dask
from scipy import signal
from ..utils import io
from .process_base import ProcessEK


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
        # Retrieve filter coefficients
        with xr.open_dataset(ed.raw_path, group="Vendor", engine=ed._file_format) as ds_fil:
            # Get various parameters
            Ztrd = 75  # Transducer quadrant nominal impedance [Ohms] (Supplied by Simrad)
            delta = 1 / 1.5e6   # Hard-coded EK80 sample interval
            tau = cal_params['transmit_duration_nominal'].values
            tx_power = cal_params['transmit_power'].values
            slope = cal_params['slope'].values

            # Find indexes with a unique transmitted signal.
            # Get outer product of 3 parameters
            beam_params = np.einsum('i,j,k->ijk', tau[0], slope[0], tx_power[0])
            # Get diagonal of 3D matrix
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
            for txi in range(tau.shape[1]):
                ytx_ping = []
                # Create transmit signal
                for ch in range(ed.raw.frequency.size):
                    t = np.arange(0, tau[ch][txi], delta)
                    nt = len(t)
                    nwtx = (int(2 * np.floor(slope[ch][txi] * nt)))
                    wtx_tmp = np.hanning(nwtx)
                    nwtxh = (int(np.round(nwtx / 2)))
                    wtx = np.concatenate([wtx_tmp[0:nwtxh], np.ones((nt - nwtx)), wtx_tmp[nwtxh:]])
                    y_tmp = amp[ch][txi] * signal.chirp(t, f0[ch], tau[ch][txi], f1[ch]) * wtx
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
                ytx.extend([ytx_ping for n in range(counts[txi])])

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

        c = env_params['speed_of_sound_in_water']
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
                print(f"{dt.now().strftime('%H:%M:%S')}  saving calibrated Sv to {Sv_path}")
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
                print(f"{dt.now().strftime('%H:%M:%S')}  saving calibrated TS to {TS_path}")
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

    def calc_range(self, ed, env_params, cal_params, range_bins=None):
        range_bin = xr.DataArray(np.arange(range_bins), coords=[np.arange(range_bins)], dims=['range_bin']) if \
            range_bins is not None else ed.raw.range_bin
        st = self.calc_sample_thickness(ed, env_params, cal_params)
        range_meter = st * range_bin - cal_params['transmit_duration_nominal'] * \
            env_params['speed_of_sound_in_water'] / 2  # DataArray [frequency x range_bin]
        range_meter = range_meter.where(range_meter > 0, other=0)
        return range_meter
