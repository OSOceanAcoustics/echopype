import os

class ConvertBase:
    # Class for assigning attributes common to all echosounders
    def __init__(self):
        self._platform = {
            'platform_name': '',
            'platform_code_ICES': '',
            'platform_type': ''
        }
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

        # Raise error if there is only 1 raw file, but combine_opt is True
        if combine_opt and n_files == 1:
            raise ValueError("Cannot combine raw files if there is only one.")

        filenames = self.filename
        if save_path is not None:
            ext = os.path.splitext(save_path)[1]
            # Check if save_path is a file or a directory
            if ext == '':
                if combine_opt:
                    raise ValueError("Save_path must be a valid file path when combining files")
                self.out_dir = save_path
            else:
                self.out_dir, filenames = os.path.split(save_path)
                filenames = [filenames]
                if ext != file_format:
                    raise ValueError(f'The path must have the extension "{file_format}"')
                # Raise error if input path is a file and there are multiple file not being combined
                if not combine_opt and n_files > 1:
                    raise ValueError(f"Output path must be a directory when not combining files")
            # Create folder if save_path does not exist.
            if not os.path.exists(self.out_dir):
                try:
                    os.mkdir(self.out_dir)
                # Raise error if save_path is not a folder.
                except FileNotFoundError:
                    raise ValueError("A valid save directory was not given")
        # Save in the same directory as raw file if save_path is not specified
        else:
            if combine_opt:
                raise ValueError("Specify a save path when combining raw files")
            self.out_dir = os.path.dirname(self.filename[0])

        # Store output filenames
        files = [os.path.splitext(os.path.basename(f))[0] for f in filenames]
        self.save_path = [os.path.join(self.out_dir, f + file_format) for f in files]
        self.nc_path = [os.path.join(self.out_dir, f + '.nc') for f in files]
        self.zarr_path = [os.path.join(self.out_dir, f + '.zarr') for f in files]
        # Convert to string if only 1 output file
        if len(self.save_path) == 1:
            self.save_path = self.save_path[0]
            self.nc_path = self.nc_path[0]
            self.zarr_path = self.zarr_path[0]

    def raw2nc(self, save_path=None, combine_opt=False, overwrite=False, compress=True):
        """Wrapper for save function

        Parameters
        ----------
        file_format : str
            format of output file. ".nc" for netCDF4 or ".zarr" for Zarr
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
        """Wrapper for save function

        Parameters
        ----------
        file_format : str
            format of output file. ".nc" for netCDF4 or ".zarr" for Zarr
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
