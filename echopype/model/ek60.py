"""
echopype data model inherited from based class EchoData for EK60 data.
"""

import os
import datetime as dt
import numpy as np
import xarray as xr
from .modelbase import ModelBase
from echopype.utils import uwa


class ModelEK60(ModelBase):
    """Class for manipulating EK60 echo data that is already converted to netCDF."""

    def __init__(self, file_path=""):
        ModelBase.__init__(self, file_path)
        self.tvg_correction_factor = 2  # range bin offset factor for calculating time-varying gain in EK60
        self._piece = None   # If range changes during data collection, piece specifies which range_bin to select
        self._salinity = None
        self._temperature = None
        self._pressure = None

    @property
    def piece(self):
        return self._piece

    @piece.setter
    def piece(self, p):
        with xr.open_dataset(self.file_path, group="Beam") as ds_beam:
            pp = list(range(int(ds_beam.pieces)))
            # if len(pp) == 1:
            #     # TODO: Mute this for now, will revise everything about 'piece' later.
            #     print('Your data does not contain pings with different ranges.')
            if p in pp:
                self._piece = p
            else:
                print(f"\'piece\' must be one of: {pp}")

    def get_biggest_piece(self):
        """Get the index of the biggest piece (which piece has the most pings)
        Used if self.piece is not set by the user

        Returns
        -------
        The index of the biggest piece. 0 if range_bin does not vary in time
        """
        with xr.open_dataset(self.file_path, group="Beam") as ds_beam:
            times_list = []
            try:
                for i in list(range(int(ds_beam.pieces))):
                    times_list.append(ds_beam[f'ping_time_{i}'])
            # Exception occurs when there is only one range. It is caught and the only piece is returned
            except KeyError:
                self.piece = 0
            else:
                # Get longest ping_time
                piece = max(times_list, key=len)
                # Get and save index of longest ping_time
                self.piece = [i for i, j in enumerate(times_list) if np.array_equal(j, piece)].pop()
            finally:
                return self.piece

    def get_piece(self, sel):
        """Returns an element from the .nc file

        Parameters
        ----------
        sel : str
            Can be 'backscatter_r', 'ping_time', or 'range_bin'
            Will raise an error if not one of these three

        Returns
        -------
        The selected element specified in sel.
        """
        if self.piece is None:
            self.get_biggest_piece()
        with xr.open_dataset(self.file_path, group="Beam") as ds_beam:
            try:
                return ds_beam[sel]
            except KeyError:
                if sel == 'backscatter_r':
                    # Return backscatter with coordinates without _x on the end
                    return xr.DataArray(ds_beam[f'backscatter_r_{self.piece}'].values,
                                        coords=[('frequency', ds_beam['frequency']),
                                                ('ping_time', ds_beam[f'ping_time_{self.piece}']),
                                                ('range_bin', ds_beam[f'range_bin_{self.piece}'])],
                                        name='backscatter_r')
                else:
                    try:
                        return ds_beam[f'{sel}_{self.piece}'].rename({
                            f'{sel}_{self.piece}': sel})
                    except KeyError:
                        raise(f'{sel} is not a valid input')

    def get_salinity(self):
        return self._salinity

    def get_temperature(self):
        return self._temperature

    def get_pressure(self):
        return self._pressure

    def get_sound_speed(self):
        with xr.open_dataset(self.file_path, group="Environment") as ds_env:
            return ds_env.sound_speed_indicative

    def calc_seawater_absorption(self, src='file'):
        """Returns the seawater absorption values from the .nc file.
        """
        if src == 'file':
            with xr.open_dataset(self.file_path, group="Environment") as ds_env:
                return ds_env.absorption_indicative
        elif src == 'user':
            with xr.open_dataset(self.file_path, group='Beam') as ds_beam:
                freq = ds_beam.frequency.astype(np.int64)  # should already be in unit [Hz]
            sea_abs = uwa.calc_seawater_absorption(freq,
                                                   temperature=self.temperature,
                                                   salinity=self.salinity,
                                                   pressure=self.pressure,
                                                   formula_source='AM')
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
                        self.tvg_correction_factor * self.sample_thickness  # DataArray [frequency x range_bin]
            range_meter = range_meter.where(range_meter > 0, other=0)
            return range_meter

    def calibrate(self, save=False, save_postfix='_Sv'):
        """Perform echo-integration to get volume backscattering strength (Sv) from EK60 power data.

        Parameters
        -----------
        save : bool, optional
            whether to save calibrated Sv output
            default to ``False``
        save_postfix : str
            Filename postfix, default to '_Sv'
        """
        # Open data set for Environment and Beam groups
        ds_beam = xr.open_dataset(self.file_path, group="Beam")

        # Derived params
        wavelength = self.sound_speed / ds_beam.frequency  # wavelength

        # Get backscatter_r and range_bin pieces
        backscatter_r = self.get_piece('backscatter_r')

        # Calc gain
        CSv = 10 * np.log10((ds_beam.transmit_power * (10 ** (ds_beam.gain_correction / 10)) ** 2 *
                             wavelength ** 2 * self.sound_speed * ds_beam.transmit_duration_nominal *
                             10 ** (ds_beam.equivalent_beam_angle / 10)) /
                            (32 * np.pi ** 2))

        # Get TVG and absorption
        range_meter = self.range
        TVG = np.real(20 * np.log10(range_meter.where(range_meter >= 1, other=1)))
        ABS = 2 * self.seawater_absorption * range_meter

        # Calibration and echo integration
        Sv = backscatter_r + TVG + ABS - CSv - 2 * ds_beam.sa_correction
        Sv.name = 'Sv'
        Sv = Sv.to_dataset()

        # Attach calculated range into data set
        Sv['range'] = (('frequency', 'range_bin'), self.range.T)

        # Save calibrated data into the calling instance and
        #  to a separate .nc file in the same directory as the data filef.Sv = Sv
        self.Sv = Sv
        if save:
            if save_postfix is not '_Sv':
                self.Sv_path = os.path.join(os.path.dirname(self.file_path),
                                            os.path.splitext(os.path.basename(self.file_path))[0] +
                                            save_postfix + '.nc')
            print('%s  saving calibrated Sv to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))
            Sv.to_netcdf(path=self.Sv_path, mode="w")

        # Close opened resources
        ds_beam.close()

    # TODO: Need to write a separate method for calculating TS as have been done for AZFP data.
