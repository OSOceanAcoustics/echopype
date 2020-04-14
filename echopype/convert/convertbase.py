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
                                           # grouped into files with the same range_bin length
        self.out_dir = None            # folder of output file
        self.nc_path = None            # path to .nc file for reference
        self.zarr_path = None          # path to .zarr file for reference
        self.save_path = None          # path to the file saved
        self.all_files = []            # Used for splitting up a file with multiple range bins
        self._append_zarr = False      # flag to determine if combining raw files into 1 zarr file
        self._temp_dir = None          # path of temporary folder for storing .nc files before combination
        self._temp_path = []           # paths of temporary files for storing .nc files before combination

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

    def reset_vars(self, echo_type):
        if echo_type == 'EK60':
            self.config_datagram = None
            self.ping_data_dict = {}
            self.power_dict = {}
            self.angle_dict = {}
            self.ping_time = []
            self.CON1_datagram = None
            self.range_lengths = None
            self.ping_time_split = {}
            self.power_dict_split = {}
            self.angle_dict_split = {}
            self.tx_sig = {}
            self.ping_slices = []
            self.all_files = []
        elif echo_type == 'EK80':
            pass
        elif echo_type == 'AZFP':
            pass

    def validate_path(self, save_path, file_format, combine_opt):
        """ Takes in either a path for a file or directory for either a .nc or .zarr output file.
        If the directory does not exist, create it. Raises an error if the directory cannot
        be created or if the filename does not match the file_format given.
        If combine_opt is true, then save_path must be a filename. If false then save_path must be a directory.
        Writes to self.save_path, self.nc_path, self.zarr_path, self.out_dir, self._temp_dir, self._temp_path

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
        save_path = self.save_path
        split = os.path.splitext(self.save_path)
        all_temp = os.listdir(self._temp_dir)
        file_groups = [[]]
        start_idx = 0
        i = 0
        # Split the files in the temp directory into range_bin groups
        while i < len(all_temp):
            file_groups[-1].append(os.path.join(self._temp_dir, all_temp[i]))
            if "_part" in all_temp[i]:
                i += 1
                file_groups.append([os.path.join(self._temp_dir, all_temp[i])])
            i += 1

        for n, file_group in enumerate(file_groups):
            if len(file_groups) > 1:
                # Construct a new path with _part[n] if there are multiple range_bin lengths 
                save_path = split[0] + '_part%02d' % (n + 1) + split[1]
            # Open multiple files as one dataset of each group and save them into a single file
            with xr.open_dataset(file_group[0], group='Environment') as ds_env:
                ds_env.to_netcdf(path=save_path, mode='w', group='Environment')
            with xr.open_dataset(file_group[0], group='Provenance') as ds_prov:
                ds_prov.to_netcdf(path=save_path, mode='a', group='Provenance')
            with xr.open_dataset(file_group[0], group='Sonar') as ds_sonar:
                ds_sonar.to_netcdf(path=save_path, mode='a', group='Sonar')
            with xr.open_mfdataset(file_group, group='Beam', combine='by_coords') as ds_beam:
                ds_beam.to_netcdf(path=save_path, mode='a', group='Beam')
            if echo_type == 'EK60' or echo_type == 'EK80':
                with xr.open_mfdataset(file_group, group='Platform', combine='by_coords') as ds_plat:
                    ds_plat.to_netcdf(path=save_path, mode='a', group='Platform')
                with xr.open_mfdataset(file_group, group='Platform/NMEA',
                                    combine='nested', concat_dim='time') as ds_nmea:
                    ds_nmea.to_netcdf(path=save_path, mode='a', group='Platform/NMEA')
            if echo_type == 'AZFP':
            # The platform group for AZFP does not have coordinates, so it must be handled differently from EK60
                with xr.open_dataset(file_group[0], group='Platform') as ds_plat:
                    ds_plat.to_netcdf(path=save_path, mode='a', group='Platform')
            # EK60 does not have the "vendor specific" group
                with xr.open_mfdataset(file_group, group='Vendor', combine='by_coords') as ds_vend:
                    ds_vend.to_netcdf(path=save_path, mode='a', group='Vendor')

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
