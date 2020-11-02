"""
Process classes perform computation on EchoData objects.

Some operations are instrument-dependent, such as calibration to obtain Sv.

Some operations are instrument-agnostic, such as obtaining MVBS or detecting bottom.
"""
from ..utils import uwa


class ProcessBase:
    """Class for processing sonar data.
    """
    def __init__(self, model=None, echodata=None):
        self.sonar_model = model   # type of echosounder
        self.echodata = echodata   # EchoData object

        self.env_parameters = {
            'sea_water_salinity': None,            # [psu]
            'sea_water_temperature': None,         # [degC]
            'sea_water_pressure': None,            # [dbars] (~depth in meters)
            'speed_of_sound_in_sea_water': None,   # [m/s]
            'seawater_absorption': None            # [dB/m]
        }
        self.sample_thickness = None  # TODO: this is not used in AZFP , right?

        self.check_model_echodata_match()

    def check_model_echodata_match(self):
        """Check if sonar model corresponds with the type of data in EchoData object.
        """
        # Raise error if they do not match

    def set_environment_parameters(self, var_name, var_val):
        """Allow user to add and overwrite environment parameters from raw data files.

        TODO: finish docstring

        Parameters
        ----------
        var_name
        var_val
        """
        self.env_parameters[var_name] = var_val

    def _calc_sound_speed(self):
        """Base method for calculating sound speed.
        """

    def _calc_seawater_absorption(self):
        """Base method for calculating seawater absorption.
        """

    def _calc_sample_thickness(self):
        """Base method for calculating sample thickness.

        This method is only used for EK echosounders.
        """

    def _calc_range(self):
        """Base method for calculating range.
        """

    def update_env_parameters(self, ss=True, sa=True, st=True, r=True):
        """Recalculates sound speed, seawater absorption, sample thickness, and range using
        updated salinity, temperature, and pressure.

        # TODO: how about we just force this to always re-calculate everything?
        Parameters
        ----------
        ss : bool
            Whether to calculate sound speed. Defaults to `True`.
        sa : bool
            Whether to calculate seawater absorption. Defaults to `True`.
        st : bool
            Whether to calculate sample thickness. Defaults to `True`.
        r : bool
            Whether to calculate range. Defaults to `True`.
        """

    def get_Sv(self):
        """Base method to be overridden for calculating Sv from raw backscatter data.
        """
        # Issue warning when subclass methods not available
        print('Calibration has not been implemented for this sonar model!')

    def get_TS(self):
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
    """Class for processing data from ASL Env Sci AZFP echosounder.
    """
    def __init__(self, model=None, echodata=None):
        super().__init__(model, echodata)

    # TODO: need something to prompt user to use set_environment_parameters()
    #  to put in pressure and salinity before trying to calibrate

    def _calc_sound_speed(self):
        """Calculate sound speed using AZFP formula and save results to self.env_parameters.
        """

    def _calc_seawater_absorption(self):
        """Calculate sound absorption using AZFP formula and save results to self.env_parameters.
        """

    def _calc_range(self):
        """Calculates range in meters using AZFP formula, instead of from sample_interval directly.
        """
        # TODO: required parameters are dictionary items in
        #  self.echodata.env_parameters and
        #  self.echodata.instr_parameters

    def get_Sv(self):
        """Calibrate to get volume backscattering strength (Sv) from AZFP power data.
        """
        # TODO: transplant what was in .calibrate() before

    def get_TS(self):
        """Calibrate to get Target Strength (TS) from AZFP power data.
        """
        # TODO: transplant what was in .calibrate_TS() before


class ProcessEK(ProcessBase):
    """Class for processing data from Simrad EK echosounders.
    """


class ProcessEK60(ProcessEK):
    """Class for processing data from Simrad EK60 echosounder.
    """


class ProcessEK80(ProcessEK):
    """Class for processing data from Simrad EK80 echosounder.
    """
