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
    def __init__(self, raw_path=None,
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
        self.env_parameters = {}
        self.instr_parameters = {}

        # Initialize data pointers
        self._init_data_pointer()

    # TODO: need to make raw_backscatter, Sv_clean, MVBS, TS all properties
    #  seems that we can use a decorator to avoid too much repetitive code
    @property
    def Sv(self):
        if self._Sv is None:
            print('Data has not been calibrated. '
                  'Call `Process.calibrate(EchoData)` to calibrate.')
        else:
            return self._Sv

    @Sv.setter
    def Sv(self, val):
        """Point self._Sv to the dataset opened by user.

        Use case:
            ed = EchoData(raw_path=[...])
            ed.Sv = xr.open_mfdataset([some files containing corresponding Sv data])
        """
        self._Sv = val

    # TODO: change below to use decorator for all paths
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

    def _get_env_parameters_from_raw(self):
        """Retrieve environment parameters from the raw data file.

        This function is instrument-dependent and needs to be inherited.
        """

    def _get_instr_parameters_from_raw(self):
        """Retrieve instrument parameters from the raw data file.

        This function is instrument-dependent and needs to be inherited.
        """

    def _update_file_list(self, varname):
        """Update the path specified by user to a list of all files to be opened together.
        """
        # If user passes in a list in self.X_path, use that list directly.
        # If user passes in a path to folder in self.X_path, index all files in the folder.
        # Update self.varname to be the above list, probably by using setattr?

    def _check_key_param_consistency(self):
        """Check if key params in the files are identical so that
        they can be opened together.

        This function is instrument-dependent and needs to be inherited.
        """
        # TODO: throw an error if parameters are not consistent, which
        #  can be caught in the calling function like self._init_pointer()

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
        if (self.raw_path is None) and (self.Sv_path is None):
            raise ValueError('Please specify a path to nc or zarr files '
                             'containing either raw data or calibrated Sv.')
        else:
            # Point _raw_backscatter to data and read environment and instrument params
            if self.raw_path is not None:
                # Get paths to files
                self._update_file_list('raw_path')
                try:
                    self._check_key_param_consistency()
                    self._raw_backscatter = xr.open_mfdataset(self.raw_path, group='Beam')
                    self._get_env_parameters_from_raw()
                    self._get_instr_parameters_from_raw()
                except:  # TODO: need to specify exception type
                    raise
            else:
                self._raw_backscatter = None

            # Point _Sv to data
            if self.Sv_path is not None:
                # Get paths to files
                self._update_file_list('Sv')
            else:
                self._Sv = None

        # Initialize Sv_clean, MVBS, TS  # TODO: do the same as below for MVBS and TS
        if self.Sv_clean_path is None:
            self._Sv_clean = None
        else:
            self._update_data_pointer('Sv_clean')


class EchoDataAZFP(EchoDataBase):
    """Echo data model for data from AZFP echosounder.
    """
    def __init__(self, raw_path=None, Sv_path=None, Sv_clean_path=None, TS_path=None, MVBS_path=None):
        super().__init__(raw_path, Sv_path, Sv_clean_path, TS_path, MVBS_path)

        # Get and set instrument-specific parameters
        self._get_env_parameters_from_raw()

    def _get_env_parameters_from_raw(self):
        """Retrieve environment parameters from the raw data file.
        """
        # TODO: the open_mfdataset below should be changed to something like _set_open_dataset()
        #  but also include a check for whether it is one file or multiple files
        # Read parameters stored in raw data that are used for calibration
        self._env_params_temp_ds = xr.open_mfdataset(self.raw_path, group='Environment')
        self.env_parameters['temperature'] = self._env_params_temp_ds['temperature']

    def _get_instr_parameters_from_raw(self):
        """Retrieve instrument parameters from the raw data file.
        """
        # TODO: the open_mfdataset below should be changed to something like _set_open_dataset()
        #  but also include a check for whether it is one file or multiple files
        # Read parameters stored in raw data that are used for calibration
        # TODO: find out what parameters ds_vend were returned in any part of calibration
