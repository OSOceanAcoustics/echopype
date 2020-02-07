"""
echopype data model inherited from based class EchoData for EK80 data.
"""

import datetime as dt
import numpy as np
import xarray as xr
from echopype.utils import uwa
from .modelbase import ModelBase


class ModelEK80(ModelBase):
    """Class for manipulating EK60 echo data that is already converted to netCDF."""

    def __init__(self, file_path=""):
        ModelBase.__init__(self, file_path)
        self._acidity = None
        self._salinity = None
        self._temperature = None
        self._pressure = None
        self._ch_ids = None

    @property
    def ch_ids(self):
        if self._ch_ids is None:
            with xr.open_dataset(self.file_path, group="Beam") as ds_beam:
                self._ch_ids = ds_beam.channel_id.data
        return self._ch_ids

    def get_salinity(self):
        if self._salinity is None:
            with xr.open_dataset(self.file_path, group="Environment") as ds_env:
                self._salinity = ds_env.salinity
        return self._salinity

    def get_temperature(self):
        if self._temperature is None:
            with xr.open_dataset(self.file_path, group="Environment") as ds_env:
                self._temperature = ds_env.temperature
        return self._temperature

    def get_pressure(self):
        if self._pressure is None:
            with xr.open_dataset(self.file_path, group="Environment") as ds_env:
                self._pressure = ds_env.depth
        return self._pressure

    def get_sound_speed(self):
        with xr.open_dataset(self.file_path, group="Environment") as ds_env:
            return ds_env.sound_speed_indicative

    def calc_seawater_absorption(self, src='FG'):
        with xr.open_dataset(self.file_path, group='Beam') as ds_beam:
            f0 = ds_beam.frequency_start
            f1 = ds_beam.frequency_end
            f_center = (f0 + f1) / 2
        sea_abs = uwa.calc_seawater_absorption(f_center,
                                               temperature=self.temperature,
                                               salinity=self.salinity,
                                               pressure=self.pressure,
                                               formula_source='FG')
        return sea_abs

    def calc_sample_thickness(self):
        with xr.open_dataset(self.file_path, group="Beam") as ds_beam:
            sth = self.sound_speed * ds_beam.sample_interval / 2  # sample thickness
            return sth

    def calc_range(self):
        """Calculates range in meters using parameters stored in the .nc file.
        """
        with xr.open_dataset(self.file_path, group="Beam") as ds_beam:
            range_meter = ds_beam.range_bin * self.sample_thickness - \
                ds_beam.transmit_duration_nominal * self.sound_speed / 2  # DataArray [frequency x range_bin]
            range_meter = range_meter.where(range_meter > 0, other=0)
            return range_meter

    def calibrate(self, save=False):
        def calc_sent_signal():
            def chirp_linear(t, f0, f1, tau):
                beta = (f1 - f0) * (tau ** -1)
                return np.cos(2 * np.pi * (beta / 2 * (t ** 2) + f0 * t))
            # Retrieve filter coefficients
            ds_fil = xr.open_dataset(self.file_path, group="Vendor")
            # WBT signal generator
            delta = 1 / 1.5e6   # Hard-coded EK80 sample interval
            Ztrd = 75           # Transducer quadrant nominal impedance [Ohms]
            a = np.sqrt((txpower / 4) * (2 * Ztrd))

            # Create transmit signal
            ytx = []
            for i in range(len(f0)):
                t = np.arange(0, tau[i], delta)
                nt = len(t)
                nwtx = (int(2 * np.floor(slope[i].data * nt)))
                wtx_tmp = np.hanning(nwtx)
                nwtxh = (int(np.round(nwtx / 2)))
                wtx = np.concatenate([wtx_tmp[0:nwtxh], np.ones((nt - nwtx)), wtx_tmp[nwtxh:]])
                y_tmp = a[i] * chirp_linear(t, f0[i], f1[i], tau[i]) * wtx
                # The transmit signal must have a max amplitude of 1
                y = (y_tmp / np.max(np.abs(y_tmp)))

                # filter and decimation
                wbt_fil = ds_fil[self.ch_ids[i] + "_WBT_filter"].data
                pc_fil = ds_fil[self.ch_ids[i] + "_PC_filter"].data
                # if saved as netCDF4, convert compound complex datatype to complex64
                if wbt_fil.ndim == 1:
                    wbt_fil = np.array([complex(n[0], n[1]) for n in wbt_fil], dtype='complex64')
                    pc_fil = np.array([complex(n[0], n[1]) for n in pc_fil], dtype='complex64')
                # Apply WBT filter and downsample
                ytx_tmp = np.convolve(y, wbt_fil)
                ytx_tmp = ytx_tmp[0::ds_fil.attrs[self.ch_ids[i] + "_WBT_decimation"]]
                # Apply PC filter and downsample
                ytx_tmp = np.convolve(ytx_tmp, pc_fil)
                ytx_tmp = ytx_tmp[0::ds_fil.attrs[self.ch_ids[i] + "_PC_decimation"]]
                ytx.append(ytx_tmp)
                del nwtx, wtx_tmp, nwtxh, wtx, y_tmp, y, ytx_tmp
            return np.array(ytx)

        # Hard-coded EK80 values supplied by Simrad
        Rwbtrx = 1000       # Wideband transceiver impedance [Ohms]
        ds_beam = xr.open_dataset(self.file_path, group="Beam")
        tau = ds_beam.transmit_duration_nominal.data
        txpower = ds_beam.transmit_power.data
        f0 = ds_beam.frequency_start[:, 0].data     # Use start frequency of first ping
        f1 = ds_beam.frequency_end[:, 0].data       # Use end frequency of first ping
        slope = ds_beam.slope[:, 0]                 # Use slope frequency of first ping
        sample_interval = ds_beam.sample_interval
        f_center = (f0 + f1) / 2
        f_nominal = ds_beam.frequency
        c = self.sound_speed

        ytx = calc_sent_signal()
        backscatter_r = ds_beam.backscatter_r
        backscatter_i = ds_beam.backscatter_i

        # FM
        if np.all(f1 - f0 > 1):
            if backscatter_r.ndim == 4:
                nq = 4 # Number of quadrants
                compressed = np.flipud(np.conj(ytx)) / np.square(np.norm(ytx))
            else:
                nq = 1
        # CW
        else:
            ptxa = np.square(np.abs(ytx))
            fs_dec = 1 / sample_interval

        # Average accross quadrants and take the absolute value of complex backscatter
        prx = np.sqrt(np.mean(backscatter_r, 1) ** 2 + np.mean(backscatter_i, 1) ** 2)
        prx = prx * prx / 2 * (np.abs(Rwbtrx + Ztrd) / Rwbtrx) ** 2 / np.abs(Ztrd)
        sea_abs = self.seawater_absorption

        la2 = (c / f_center) ** 2
        f = np.moveaxis(np.array((ds_beam.frequency_start, ds_beam.frequency_end)), 0, 2)
        # TODO Gfc should be gain interpolated at the center frequency
        Gfc = np.mean(ds_beam.gain_correction.values)
        ptx = ds_beam.transmit_power
        r = self.range
        r = r.where(r >= 1, other=1)
        pass
