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
from ..utils import io
import xarray as xr


class EchoData:
    """Echo data model base class.
    """
    def __init__(self, raw_path=None,
                 Sv_path=None, Sv_clean_path=None,
                 TS_path=None, MVBS_path=None):

        # Pointer to data
        # self._raw = None
        # self._Sv = None
        # self._Sv_clean = None
        # self._MVBS = None
        # self._TS = None

        # self._raw_path = None
        # self._Sv_path = None
        # self._Sv_clean_path = None
        # self._TS_path = None
        # self._MVBS_path = None

        # # Paths to files: can be netcdf or zarr
        # self.raw_path = raw_path
        # self.Sv_path = Sv_path
        # self.Sv_clean_path = Sv_clean_path
        # self.TS_path = TS_path
        # self.MVBS_path = MVBS_path

        self._sonar_model = None
        self.range = None

        # Initialize data pointers and paths to files
        attrs = ['raw', 'Sv', 'Sv_clean', 'TS', 'MVBS']    # Data that is handled in Process
        for attr in attrs:
            setattr(self, '_' + attr, None)
            setattr(self, '_' + attr + '_path', None)
            files = eval(attr + '_path')
            files = [files] if isinstance(files, str) else files
            setattr(self, attr + '_path', files)

        self._file_format = None
        self._open_dataset = None

        # Initialize data pointers
        self._set_file_format()
        self._set_dataset_handlers()

    # https://stackoverflow.com/questions/21918561/python-multiple-property-statements-in-class-definition

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

    # Data
    @property
    def Sv_clean(self):
        if self._Sv_clean is None:
            print('Data has not been calibrated. '
                  'Call `Process.calibrate(EchoData)` to calibrate.')
        else:
            return self._Sv_clean

    @Sv_clean.setter
    def Sv_clean(self, val):
        self._Sv_clean = val

    @property
    def TS(self):
        if self._TS is None:
            print('Data has not been calibrated. '
                  'Call `Process.calibrate(EchoData)` to calibrate.')
        else:
            return self._TS

    @TS.setter
    def TS(self, val):
        self._TS = val

    @property
    def MVBS(self):
        if self._MVBS is None:
            print('Data has not been calibrated. '
                  'Call `Process.calibrate(EchoData)` to calibrate.')
        else:
            return self._MVBS

    @MVBS.setter
    def MVBS(self, val):
        self._MVBS = val

    @property
    def raw(self):
        if self._raw is None:
            print('No raw backscatter data availible')
        else:
            return self._raw

    @raw.setter
    def raw(self, val):
        self._raw = val

    # Paths
    @property
    def raw_path(self):
        return self._raw_path

    @raw_path.setter
    def raw_path(self, val):
        self._update_data_pointer(val, arr_type='raw')

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
        # self._Sv_path = val
        self._update_data_pointer(val, arr_type='Sv')

    @property
    def TS_path(self):
        return self._TS_path

    @TS_path.setter
    def TS_path(self, val):
        self._update_data_pointer(val, arr_type='TS')

    @property
    def Sv_clean_path(self):
        return self._Sv_clean_path

    @Sv_clean_path.setter
    def Sv_clean_path(self, val):
        self._update_data_pointer(val, arr_type='Sv_clean')

    @property
    def MVBS_path(self):
        return self._MVBS_path

    @MVBS_path.setter
    def MVBS_path(self, val):
        self._update_data_pointer(val, arr_type='MVBS')

    @property
    def sonar_model(self):
        if self._sonar_model is None:
            with self._open_dataset(self.raw_path[0]) as ds:
                self._sonar_model = ds.keywords
        return self._sonar_model

    @io._check_key_param_consistency(group='Environment')
    def get_env_from_raw(self):
        """Open the Environment group from raw data files.
        """
        if self.sonar_model == 'AZFP':
            return self._open_mfdataset(self.raw_path, group='Environment',
                                        combine='by_coords', data_vars='minimal')
        else:
            return self._open_dataset(self.raw_path[0], group='Environment')

    @io._check_key_param_consistency(group='Vendor')
    def get_vend_from_raw(self):
        """Open the Vendor group from raw data files.
        """
        return self._open_dataset(self.raw_path[0], group='Vendor')

    def _update_file_list(self, path, file_list):
        """Update the path specified by user to a list of all files to be opened together.
        """
        # If user passes in a list in self.X_path, use that list directly.
        # If user passes in a path to folder in self.X_path, index all files in the folder.
        # Update self.varname to be the above list, probably by using setattr?
        if isinstance(path, str):
            ext = os.path.splitext(path)[1]
            # Check for folder to paths
            if ext == '':
                setattr(self, file_list, io.get_files_from_dir(path))
            # Check single file path
            elif ext in ['.nc', '.zarr']:
                setattr(self, file_list, path)
            else:
                raise ValueError("Unsupported file path")
        else:
            # Check for list of paths
            if isinstance(path, list):
                setattr(self, file_list, path)
            else:
                raise ValueError("Unsupported file path")

    def _update_data_pointer(self, path, arr_type):
        """Update pointer to data for the specified type and path.
        """
        attr = f'_{arr_type}'
        attr_path = attr + '_path'
        if path is None:
            setattr(self, attr, None)
            setattr(self, attr_path, None)
        else:
            self._update_file_list(path, attr_path)
            group = 'Beam' if attr == '_raw' else None
            try:
                # Lazy load data into instance variable ie. self.Sv, self.raw, etc
                setattr(self, attr, xr.open_mfdataset(getattr(self, attr_path), group=group))
            except xr.MergeError as e:
                var = str(e).split("'")[1]
                raise ValueError(f"Files cannot be opened due to {var} changing across the files")

    def _set_file_format(self):
        if self.raw_path[0].endswith('.nc'):
            self._file_format = 'netcdf'
        elif self.raw_path.endswith('.zarr'):
            self._file_format = 'zarr'

    def _set_dataset_handlers(self):
        if self._file_format == 'netcdf':
            self._open_dataset = xr.open_dataset
        elif self._file_format == 'zarr':
            self._open_dataset = xr.open_zarr

    def _save_dataset(self, ds, path, mode="w", save_format='zarr'):
        """Save dataset to the appropriate formats.

        A utility method to use the correct function to save the dataset,
        based on the input file format.

        Parameters
        ----------
        ds : xr.Dataset
            xarray dataset object
        path : str
            output file
        """
        if save_format == 'netcdf':
            ds.to_netcdf(path, mode=mode)
        elif save_format == 'zarr':
            ds.to_zarr(path, mode=mode)
        else:
            raise ValueError("Unsupported save format " + save_format)
