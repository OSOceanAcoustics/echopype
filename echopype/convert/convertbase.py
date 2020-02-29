import os
import xarray as xr
import shutil


class ConvertBase:
    # Class for assigning attributes common to all echosounders
    def __init__(self):
        self._platform = {
            'platform_name': '',
            'platform_code_ICES': '',
            'platform_type': ''
        }
        self.out_dir = None
        self.nc_path = None
        self.zarr_path = None
        self.save_path = None

    @property
    def platform_name(self):
        return self._platform['platform_name']

    @platform_name.setter
    def platform_name(self, platform_name):
        self._platform['platform_name'] = platform_name

    @property
    def platform_type(self):
        return self._platform['platform_type']

    @platform_type.setter
    def platform_type(self, platform_type):
        self._platform['platform_type'] = platform_type

    @property
    def platform_code_ICES(self):
        return self._platform['platform_code_ICES']

    @platform_code_ICES.setter
    def platform_code_ICES(self, platform_code_ICES):
        self._platform['platform_code_ICES'] = platform_code_ICES

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, p):
        if isinstance(p, list):
            self._filename = p
        else:
            self._filename = [p]

    def validate_path(self, save_path, file_format, combine_opt):
        """ Takes in either a path for a file or directory for either a .nc or .zarr output file.
        If the directory does not exist, create it. Raises an error if the directory cannot
        be created or if the filename does not match the file_format given.
        If combine_opt is true, then save_path must be a filename. If false then save_path must be a directory.

        Parameters
        ----------
        save_path : str
            Either a directory or a file. If none then the save path is the same as the raw file.
        file_format : str
            .nc or .zarr
        combine_opt : bool
            Whether or not multiple files will be combined into one file.
        """
        n_files = len(self.filename)

        # Raise error if output format is not .nc or .zarr
        if file_format != '.nc' and file_format != '.zarr':
            raise ValueError("File format is not .nc or .zarr")

        # Cannot combine if there is only 1 file. Changes combine_opt to False
        if combine_opt and n_files == 1:
            combine_opt = False

        filenames = self.filename
        if save_path is not None:   # if save_path is specified
            path_ext = os.path.splitext(save_path)[1]
            # Check if save_path is a file or a directory
            if path_ext == '':   # if a directory
                if combine_opt:   # but want to combine multiple files
                    raise ValueError("Please set save_path to path to a file if combine_opt=True.")
                else:   # if not combining multiple files, save_path is the directory to store converted files
                    self.out_dir = save_path
            else:  # if a file
                self.out_dir, filenames = os.path.split(save_path)  # overwrite filenames as output path
                filenames = [filenames]
                ext = os.path.splitext(os.path.basename(save_path))[1]  # get extension of the specified path
                if ext != file_format:
                    raise ValueError(f'The path must have the extension "{file_format}"')
                # Raise error if input path is a file and there are multiple file not being combined
                if not combine_opt and n_files > 1:
                    raise ValueError(f"Output path must be either "
                                     f"a directory when not combining files (combine_opt=False) or "
                                     f"a path to a file when combining multiple files (combine_opt=True).")

            # Use directory of input file is self.out_dir is empty
            if self.out_dir == '':
                self.out_dir = os.path.dirname(self.filename[0])

            # Create folder if save_path does not exist already
            if (self.out_dir is not None) and (not os.path.exists(self.out_dir)):
                try:
                    os.mkdir(self.out_dir)
                # Raise error if save_path is not a folder
                except FileNotFoundError:
                    raise ValueError("A valid save directory was not given.")

        else:  # Save in the same directory as raw file if save_path is not specified
            if combine_opt:
                raise ValueError("Specify a output file path when combining multiple raw files by"
                                 "setting save_path=PATH_TO_COMBINED_FILENAME")
            self.out_dir = os.path.dirname(self.filename[0])

        # Store output filenames
        files = [os.path.splitext(os.path.basename(f))[0] for f in filenames]
        self.save_path = [os.path.join(self.out_dir, f + file_format) for f in files]
        self.nc_path = [os.path.join(self.out_dir, f + '.nc') for f in files]
        self.zarr_path = [os.path.join(self.out_dir, f + '.zarr') for f in files]

        # Convert to sting if only 1 output file
        if len(self.save_path) == 1:
            self.save_path = self.save_path[0]
            self.nc_path = self.nc_path[0]
            self.zarr_path = self.zarr_path[0]

        # If combining, create temp folder for files
        if combine_opt and ext == '.nc':
            orig_files = [os.path.splitext(os.path.basename(f))[0] for f in self.filename]
            self._temp_dir = os.path.join(self.out_dir, '.echopype_tmp')
            if not os.path.exists(self._temp_dir):
                os.mkdir(self._temp_dir)
            self._temp_path = [os.path.join(self._temp_dir, f + file_format) for f in orig_files]

    def combine_files(self, echo_type):
        # Do nothing if combine_opt is true even if there was nothing to combine
        if not hasattr(self, '_temp_path'):
            return
        # Open multiple files as one dataset of each group and save them into a single file
        with xr.open_dataset(self._temp_path[0], group='Environment') as ds_env:
            ds_env.to_netcdf(path=self.save_path, mode='w', group='Environment')
        with xr.open_dataset(self._temp_path[0], group='Provenance') as ds_prov:
            ds_prov.to_netcdf(path=self.save_path, mode='a', group='Provenance')
        with xr.open_dataset(self._temp_path[0], group='Sonar') as ds_sonar:
            ds_sonar.to_netcdf(path=self.save_path, mode='a', group='Sonar')
        with xr.open_mfdataset(self._temp_path, group='Beam', combine='by_coords') as ds_beam:
            ds_beam.to_netcdf(path=self.save_path, mode='a', group='Beam')
        # The platform group for AZFP does not have coordinates, so it must be handled differently from EK60
        try:
            ds_plat = xr.open_mfdataset(self._temp_path, group='Platform', combine='by_coords')
        except ValueError:
            ds_plat = xr.open_dataset(self._temp_path[0], group='Platform')
        ds_plat.to_netcdf(path=self.save_path, mode='a', group='Platform')
        ds_plat.close()
        # AZFP does not record NMEA data
        if echo_type == 'ek60':
            with xr.open_mfdataset(self._temp_path, group='Platform/NMEA',
                                   combine='nested', concat_dim='time') as ds_nmea:
                ds_nmea.to_netcdf(path=self.save_path, mode='a', group='Platform/NMEA')
        # EK60 does not have the "vendor specific" group
        if echo_type == 'azfp':
            with xr.open_mfdataset(self._temp_path, group='Vendor', combine='by_coords') as ds_vend:
                ds_vend.to_netcdf(path=self.save_path, mode='a', group='Vendor')

        # Delete temporary folder:
        shutil.rmtree(self._temp_dir)

    def raw2nc(self, save_path=None, combine_opt=False, overwrite=False, compress=True):
        """Wrapper for saving to netCDF.

        Parameters
        ----------
        save_path : str
            Path to save output to. Must be a directory if converting multiple files.
            Must be a filename if combining multiple files.
            If `False`, outputs in the same location as the input raw file.
        combine_opt : bool
            Whether or not to combine a list of input raw files.
            Raises error if combine_opt is true and there is only one file being converted.
        overwrite : bool
            Whether or not to overwrite the file if the output path already exists.
        compress : bool
            Whether or not to compress backscatter data. Defaults to `True`
        """
        self.save(".nc", save_path, combine_opt, overwrite, compress)

    def raw2zarr(self, save_path=None, combine_opt=False, overwrite=False, compress=True):
        """Wrapper for saving to zarr.

        Parameters
        ----------
        save_path : str
            Path to save output to. Must be a directory if converting multiple files.
            Must be a filename if combining multiple files.
            If `False`, outputs in the same location as the input raw file.
        combine_opt : bool
            Whether or not to combine a list of input raw files.
            Raises error if combine_opt is true and there is only one file being converted.
        overwrite : bool
            Whether or not to overwrite the file if the output path already exists.
        compress : bool
            Whether or not to compress backscatter data. Defaults to `True`
        """
        self.save(".zarr", save_path, combine_opt, overwrite, compress)

    def save(self, param, save_path, combine_opt, overwrite, compress):
        """Wrapper for saving functions.
        """
        pass
