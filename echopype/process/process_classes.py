from .echodata import EchoDataBase


class ProcessBase:
    """Class for processing sonar data.
    """
    def __init__(self, model=None):
        self.sonar_model = model   # type of echosounder

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

    # TODO: need something to prompt user to use set_environment_parameters()
    #  to put in pressure and salinity before trying to calibrate

    def calc_sound_speed(self, env_params):
        """Calculate sound speed using AZFP formula.
        """

    def calc_seawater_absorption(self, env_params, formula_source='AZFP', param_source=None):
        """Calculate sound absorption using AZFP formula.
        """

    def calc_range(self, ed):
        """Calculates range in meters using AZFP formula,
        instead of from sample_interval directly.
        """

    def get_Sv(self, ed, env_params, cal_params=None, save=True, save_format='zarr'):
        """Calibrate to get volume backscattering strength (Sv) from AZFP power data.
        """
        # TODO: transplant what was in .calibrate() before
        #  this operation should make an xarray Dataset in ed.Sv
        #  if save=True:
        #      - save the results as zarr in self.save_paths['Sv']
        #      - update ed.Sv_path

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
