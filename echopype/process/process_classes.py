from datetime import datetime as dt
import numpy as np
from .echodata import EchoDataBase
from ..utils import uwa


class ProcessBase:
    """Class for processing sonar data.
    """
    def __init__(self, model=None):
        self.sonar_model = model   # type of echosounder
        self._open_dataset = None

    def calc_sound_speed(self, env_params):
        """Base method for calculating sound speed.
        """

    def calc_seawater_absorption(self, env_params, formula_source, param_source):
        """Base method for calculating seawater absorption.
        """

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

    def get_MVBS(self, ed, proc_params, save=True, save_format='zarr'):
        """Calculate Mean Volume Backscattering Strength (MVBS).

        The calculation uses class attributes MVBS_ping_size and MVBS_range_bin_size to
        calculate and save MVBS as a new attribute to the calling Process instance.
        MVBS is an xarray DataArray with dimensions ``ping_time`` and ``range_bin``
        that are from the first elements of each tile along the corresponding dimensions
        in the original Sv or Sv_clean DataArray.
        """

    def remove_noise(self, ed, proc_params, save=True, save_format='zarr'):
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

    def db_diff(self, ed, proc_params, save=True, save_format='zarr'):
        """Perform dB-differencing (frequency-differencing) for specified thresholds.
        """

class ProcessAZFP(ProcessBase):
    """
    Class for processing data from ASL Env Sci AZFP echosounder.
    """
    def __init__(self, model='AZFP'):
        super().__init__(model)

    # TODO: need something to prompt user to use set_environment_parameters()
    #  to put in pressure and salinity before trying to calibrate

    def get_default_env_params(self, ed=None, env_params={}):
        """ Initialize environment parameters with default values if
        none are provided by the user or by the raw file"""
        if not hasattr(env_params, 'sea_water_salinity'):
            env_params['sea_water_salinity'] = 29.6
            print("Initialize using default sea water salinity of 29.6 ppt")
        if not hasattr(env_params, 'sea_water_pressure'):
            env_params['sea_water_pressure'] = 60
            print("Initialize using default sea water salinity of 60 dbars")
        if not hasattr(env_params, 'sea_water_temperature'):
            if ed is not None:
                with ed._open_dataset(ed.raw_path, group='Environment') as ds_env:
                    print("Initialize using average temperature recorded by instrument")
                    env_params['sea_water_temperature'] = np.nanmean(ds_env.temperature)   # temperature in [Celsius]
            else:
                env_params['sea_water_temperature'] = 3.5
                print("Using default sea water temperature of 3.5 C")
        return env_params

    def calc_sound_speed(self, env_params, formula_source='AZFP'):
        """Calculate sound speed using AZFP formula.
        """
        return uwa.calc_sound_speed(temperature=env_params['sea_water_temperature'],
                                    salinity=env_params['sea_water_salinity'],
                                    pressure=env_params['sea_water_pressure'],
                                    formula_source=formula_source)

    def calc_seawater_absorption(self, ed, env_params, formula_source='AZFP', param_source=None):
        """Calculate sound absorption using AZFP formula.
        """
        with ed._open_dataset(ed.raw_path, group='Beam') as ds_beam:
            freq = ds_beam.frequency.astype(np.int64)  # should already be in unit [Hz]
        return uwa.calc_seawater_absorption(freq,
                                            temperature=env_params['sea_water_temperature'],
                                            salinity=env_params['sea_water_salinity'],
                                            pressure=env_params['sea_water_pressure'],
                                            formula_source=formula_source)

    def calc_range(self, ed, env_params, tilt_corrected=False):
        """Calculates range in meters using AZFP formula,
        instead of from sample_interval directly.
        """

        ds_beam = ed._open_dataset(ed.raw_path, group='Beam')
        ds_vend = ed._open_dataset(ed.raw_path, group='Vendor')

        # WJ: same as "range_samples_per_bin" used to calculate "sample_interval"
        range_samples = ds_vend.number_of_samples_per_average_bin
        pulse_length = ds_beam.transmit_duration_nominal   # units: seconds
        bins_to_avg = 1   # set to 1 since we want to calculate from raw data
        sound_speed = env_params['speed_of_sound_in_sea_water']
        dig_rate = ds_vend.digitization_rate
        lockout_index = ds_vend.lockout_index

        # Below is from LoadAZFP.m, the output is effectively range_bin+1 when bins_to_avg=1
        range_mod = xr.DataArray(np.arange(1, len(ds_beam.range_bin) - bins_to_avg + 2, bins_to_avg),
                                 coords=[('range_bin', ds_beam.range_bin)])

        # Calculate range using parameters for each freq
        range_meter = (sound_speed * lockout_index / (2 * dig_rate) + sound_speed / 4 *
                       (((2 * range_mod - 1) * range_samples * bins_to_avg - 1) / dig_rate +
                        pulse_length))

        if tilt_corrected:
            range_meter = ds_beam.cos_tilt_mag.mean() * range_meter

        ds_beam.close()
        ds_vend.close()

        return range_meter

    def get_Sv(self, ed, env_params, cal_params=None, save=True, save_format='zarr'):
        """Calibrate to get volume backscattering strength (Sv) from AZFP power data.
        """
        # TODO: transplant what was in .calibrate() before
        #  this operation should make an xarray Dataset in ed.Sv
        #  if save=True:
        #      - save the results as zarr in self.save_paths['Sv']
        #      - update ed.Sv_path

        # Print raw data nc file
        print('%s  calibrating data in %s' % (dt.now().strftime('%H:%M:%S'), ed.raw_path))

        # Open data set for Environment and Beam groups
        ds_beam = ed._open_dataset(ed.raw_path, group="Beam")
        """
        The calibration formula used here is documented in eq.(9) on p.85
        of GU-100-AZFP-01-R50 Operator's Manual.
        Note a Sv_offset factor that varies depending on frequency is used
        in the calibration as documented on p.90.
        See calc_Sv_offset() in convert/azfp.py
        """
        range_meter = self.calc_range(ed, env_params) if ed.range is None else ed.range

        Sv = (ds_beam.EL - 2.5 / ds_beam.DS + ds_beam.backscatter_r / (26214 * ds_beam.DS) -
              ds_beam.TVR - 20 * np.log10(ds_beam.VTX) + 20 * np.log10(range_meter) +
              2 * self.seawater_absorption * range_meter -
              10 * np.log10(0.5 * self.sound_speed *
                            ds_beam.transmit_duration_nominal *
                            ds_beam.equivalent_beam_angle) + ds_beam.Sv_offset)

        Sv.name = 'Sv'
        Sv = Sv.to_dataset()

        # Attached calculated range into the dataset
        Sv['range'] = (('frequency', 'range_bin'), self.range)

        # Save calibrated data into the calling instance and
        #  to a separate .nc file in the same directory as the data filef.Sv = Sv
        self.Sv = Sv
        if save:
            self.Sv_path = self.validate_path(save_path, save_postfix)
            print("{} saving calibrated Sv to {}".format(dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))
            self._save_dataset(self.Sv, self.Sv_path, mode="w")

        # Close opened resources
        ds_beam.close()


    def get_TS(self, ed, env_params, cal_params=None, save=True, save_format='zarr'):
        """Calibrate to get Target Strength (TS) from AZFP power data.
        """
        # TODO: transplant what was in .calibrate_TS() before,
        #  other requirements are the same as .get_Sv()


class ProcessEK(ProcessBase):
    """
    Class for processing data from Simrad EK echosounders.
    """
    def __init__(self, model=None):
        super().__init__(model)

    def calc_sound_speed(self, env_params, param_source='file'):
        """Calculate sound speed.
        """

    def calc_seawater_absorption(self, env_params, formula_source, param_source='file'):
        """Calculate sound absorption using AZFP formula.
        """

    def calc_sample_thickness(self, ed):
        """Calculate sample thickness.
        """

    def calc_range(self, ed):
        """Calculates range in meter.
        """

    def _cal_narrowband(self, ed, env_params, cal_params, cal_type,
                        save=True, save_format='zarr'):
        """Calibrate narrowband data from EK60 and EK80.
        """
        # TODO: this operation should make an xarray Dataset in ed.Sv
        #  if save=True:
        #      - save the results as zarr in self.save_paths['Sv']
        #      - update ed.Sv_path
        if cal_type == 'Sv':
            # calculate Sv
            print('Computing Sv')  # replace this line with actual code
        else:
            # calculate TS
            print('Computing TS')  # replace this line with actual code


class ProcessEK60(ProcessEK):
    """
    Class for processing data from Simrad EK60 echosounder.
    """
    def __init__(self, model='EK60'):
        super().__init__(model)

    def get_Sv(self, ed, env_params, cal_params, save=True, save_format='zarr'):
        """Calibrate to get volume backscattering strength (Sv) from EK60 data.
        """
        return self._cal_narrowband(ed=ed,
                                    env_params=env_params,
                                    cal_params=cal_params,
                                    cal_type='Sv',
                                    save=True,
                                    save_format='zarr')

    def get_TS(self, ed, env_params, cal_params, save=True, save_format='zarr'):
        """Calibrate to get target strength (TS) from EK60 data.
        """
        return self._cal_narrowband(ed=ed,
                                    env_params=env_params,
                                    cal_params=cal_params,
                                    cal_type='TS',
                                    save=True,
                                    save_format='zarr')


class ProcessEK80(ProcessEK):
    """
    Class for processing data from Simrad EK80 echosounder.
    """
    def __init__(self, model='EK80'):
        super().__init__(model)

    def calc_transmit_signal(self, ed):
        """Generate transmit signal as replica for pulse compression.
        """

    def pulse_compression(self, ed):
        """Pulse compression using transmit signal as replica.
        """

    def _cal_broadband(self, ed, env_params, cal_params, cal_type,
                       save=True, save_format='zarr'):
        """Calibrate broadband EK80 data.
        """
        if cal_type == 'Sv':
            # calculate Sv
            print('Computing Sv')  # replace this line with actual code
        else:
            # calculate TS
            print('Computing TS')  # replace this line with actual code

    def _choose_mode(self, mm):
        """Choose which calibration mode to use.

        Parameters
        ----------
        mm : str
            'BB' indicates broadband calibration
            'CW' indicates narrowband calibration
        """
        if mm == 'BB':
            return self._cal_broadband
        else:
            return self._cal_narrowband

    def get_Sv(self, ed, env_params, cal_params, mode='BB',
               save=True, save_format='zarr'):
        """Calibrate to get volume backscattering strength (Sv) from EK80 data.
        """
        self._choose_mode(mode)(ed, env_params, cal_params, 'Sv', save, save_format)

    def get_TS(self, ed, env_params, cal_params, mode='BB',
               save=True, save_format='zarr'):
        """Calibrate to get target strength (TS) from EK80 data.
        """
        self._choose_mode(mode)(ed, env_params, cal_params, 'TS', save, save_format)
