"""
EchoData classes are data models that keep track of various types of data,
such as those stored in the converted raw data files (raw backscatter,
environmental parameters, instrument settings)
or derived data variables (Sv, Sv_clean, MVBS, TS)

Users use Process objects to perform computation on EchoData objects and
use Plot objects for visualization.
"""

import os
import glob
import xarray as xr
from ..utils import uwa


class EchoDataBase:
    """Echo data model base class.
    """
    def __init__(self, raw_path,
                 Sv_path=None, Sv_clean_path=None,
                 TS_path=None, MVBS_path=None):
        # Paths to files: can be netcdf or zarr
        self.raw_path = raw_path
        self._Sv_path = Sv_path
        self.Sv_clean_path = Sv_clean_path
        self.MVBS_path = MVBS_path
        self.TS_path = TS_path

        # Pointer to data
        self._raw_backscatter = None
        self._Sv = None
        self._Sv_clean = None
        self._MVBS = None
        self._TS = None

        # Data parameters: mostly instrument-dependent
        self.environment_parameters = {}
        self.instrument_parameters = {}

        # Initialize data pointers
        self._init_data_pointer()

    # TODO: Do we want to make raw_backscatter a property? maybe not?

    # TODO: need to make Sv_clean, MVBS, TS all properties
    #  seems that we can use a decorator to avoid too much repetitive code
    @property
    def Sv(self):
        if self._Sv is None:
            print('Data has not been calibrated. '
                  'Call `proc.calibrate(ed)` to calibrate data in `ed`.')
        else:
            return self._Sv

    @Sv.setter
    def Sv(self, val):
        """Point self._Sv to the dataset opened by user.

        Use case:
            ed = EchoData(raw_path=[...])
            ed.Sv = xr.open_mfdataset([some files containing Sv data])
        """
        self._Sv = val

    # TODO: change below to use decorator
    @property
    def Sv_path(self):
        return self._Sv_path

    @Sv_path.setter
    def Sv_path(self, val):
        """Update self._Sv_path and point to the specified dataset.

        Use case:
            ed = EchoData(raw_path=[...])
            ed.Sv_path = [some path of files containing Sv data]
        """
        self._Sv_path = val
        self._update_data_pointer()

    def _update_file_list(self, varname):
        """Update the path specified by user to a list of all files to be opened together.
        """
        # If user passes in a list in self.X_path, use that list directly.
        # If user passes in a path to folder in self.X_path, index all files in the folder.
        # Update self.varname to be the above list, probably by using setattr?

    def _check_key_param_consistency(self):
        """Check if key params in the files are identical so that
        they can be opened together.

        This function is instrument-dependent and should be inherited.
        """
        # TODO: throw an error if parameters are not consistent, which
        #  can be caught in the calling function like self._init_pointer()

    def _get_environment_parameters(self):
        """Retrieve environment parameters from the raw data file.

        This function can be instrument-dependent and may need to be inherited.
        """
        # TODO: the open_mfdataset below should be changed to something like _set_open_dataset
        #  but also include a check for whether it is one file or multiple files
        self.environment_parameters = xr.open_mfdataset(self.raw_path, group='Environment')

    def _get_instrument_parameters(self):
        """Retrieve instrument parameters from the raw data file.

        This function can be instrument-dependent and may need to be inherited.
        """

    def _update_data_pointer(self, varname):
        """Update pointer to data for the specified type and path.
        """
        # TODO: below is written for Sv, make it general and applicable to Sv_clean, MVBS, and TS
        self._update_file_list('Sv')
        try:
            self.Sv = xr.open_mfdataset(self.Sv_path)
        except:  # catch errors thrown from the above
            raise

    def _init_data_pointer(self):
        """Initialize pointer to data if the path exists.
        """
        # Initialize pointer to raw data files
        if self.raw_path is None:
            raise ValueError('Please specify a path to nc or zarr files containing raw data.')
        else:
            # Get paths to files
            self._update_file_list('raw_path')
            try:
                self._check_key_param_consistency()
                self._raw_backscatter = xr.open_mfdataset(self.raw_path, group='Beam')
                self._get_environment_parameters()
                self._get_instrument_parameters()
            except:  # catch errors thrown from any of the above
                raise

        # Initialize Sv, Sv_clean, MVBS, TS  # TODO: use the same procedure as shown below for Sv
        if self.Sv_path is None:
            self._Sv = None
        else:
            self._update_data_pointer('Sv')


class EchoDataAZFP(EchoDataBase):
    """Echo data model for data from AZFP echosounder.
    """
    def __init__(self, raw_path):
        """
        TODO: Where to put docstring for attributes?
        `self.environment_parameters` is a dictionary that stores various environmental parameters.
        For AZFP data, it includes the following keys:
           salinity : [psu], user-defined
           pressure : [psu], user-defined
           temperature : [deg Celsius], unpacked from raw data files
           sound_speed : [m/s], derived using AZFP formula
           seawater_absorption : [db/m], derived using AZFP formula
        """
        super().__init__(raw_path)

        # Get and set instrument-specific parameters
        self.environment_parameters['salinity'] = None  # salinity [psu]
        self.environment_parameters['pressure'] = None  # pressure [dbars] (~depth in meters)
        self._get_environment_parameters()

    def _get_environment_parameters(self):
        """Retrieve environment parameters from the raw data file.
        """
        # TODO: the open_mfdataset below should be changed to something like _set_open_dataset()
        #  but also include a check for whether it is one file or multiple files
        # Read parameters stored in raw data
        # TODO: fix this, now it points to a data set but it is a dict
        self.environment_parameters = xr.open_mfdataset(self.raw_path, group='Environment')

        # Calculate parameters needed for calibration
        self.environment_parameters['sound_speed'] = (
            uwa.calc_sound_speed(temperature=self.environment_parameters['temperature'],
                                 salinity=self.environment_parameters['salinity'],
                                 pressure=self.environment_parameters['pressure'],
                                 formula_source='AZFP')
        )
        self.environment_parameters['seawater_absorption'] = (
            uwa.calc_seawater_absorption(self._raw_backscatter['frequency'],  # make sure freq is in units [Hz]
                                         temperature=self.environment_parameters['temperature'],
                                         salinity=self.environment_parameters['salinity'],
                                         pressure=self.environment_parameters['pressure'],
                                         formula_source='AZFP')
        )
