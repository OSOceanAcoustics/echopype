"""
Process classes perform computation on EchoData objects.

Some operations are instrument-dependent, such as calibration to obtain Sv.

Some operations are instrument-agnostic, such as obtaining MVBS or detecting bottom.
"""


class ProcessBase:
    """Class for processing sonar data.
    """
    def __init__(self):
        self.env_parameters = {
            'sea_water_salinity': None,
            'sea_water_temperature': None,
            'sea_water_pressure': None,
            'speed_of_sound_in_sea_water': None,
            'seawater_absorption': None
        }
        self.sample_thickness = None

    def calc_sound_speed(self, src='file'):
        """Base method for calculating sound speed.
        """

    def calc_seawater_absorption(self, src='file'):
        """Base method for calculating seawater absorption.
        """

    def calc_sample_thickness(self):
        """Base method for calculating sample thickness.
        """

    def calc_range(self):
        """Base method for calculating range depending on the variable.
        """

    def update_env_parameters(self, ss=True, sa=True, st=True, r=True):
        """Recalculates sound speed, seawater absorption, sample thickness, and range using
        salinity, temperature, and pressure.

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


class ProcessEK(ProcessBase):
    """Class for processing data from Simrad EK echosounders.
    """


class ProcessEK60(ProcessEK):
    """Class for processing data from Simrad EK60 echosounder.
    """


class ProcessEK80(ProcessEK):
    """Class for processing data from Simrad EK80 echosounder.
    """
