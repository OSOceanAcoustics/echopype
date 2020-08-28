"""
UI class for converting raw data from different echosounders to netcdf or zarr.
"""
import os
import shutil
import xarray as xr
from .convertbase_new import ParseEK60, ParseEK80, ParseAZFP
from .utils.setgroups_new import SetGroupsEK60, SetGroupsEK80, SetGroupsAZFP


class Convert:
    """UI class for using convert objects.

    Sample use case:
        ec = echopype.Convert()

        # set source files
        ec.set_source(
            files=[FILE1, FILE2, FILE3],  # file or list of files
            model='EK80',       # echosounder model
            # xml_path='ABC.xml'  # optional, for AZFP only
            )

        # set parameters that may not already in source files
        ec.set_param({
            'platform_name': 'OOI',
            'platform_type': 'mooring'
            })

        # convert to netcdf, do not combine files, save to source path
        ec.to_netcdf()

        # convert to zarr, combine files, save to s3 bucket
        ec.to_netcdf(combine_opt=True, save_path='s3://AB/CDE')

        # get GPS info only (EK60, EK80)
        ec.to_netcdf(data_type='GPS')

        # get configuration XML only (EK80)
        ec.to_netcdf(data_type='CONFIG_XML')

        # get environment XML only (EK80)
        ec.to_netcdf(data_type='ENV_XML')
    """
    def __init__(self):
        # Attributes
        self.sonar_model = None     # type of echosounder
        self.xml_path = ''          # path to xml file (AZFP only)
                                    # users will get an error if try to set this directly for EK60 or EK80 data
        self.source_file = None     # input file path or list of input file paths
        self.output_file = None     # converted file path or list of converted file paths
        self._source_path = None    # for convenience only, the path is included in source_file already;
                                    # user should not interact with this directly
        self._output_path = None    # for convenience only, the path is included in source_file already;
                                    # user should not interact with this directly
        self._conversion_params = {}    # a dictionary of conversion parameters,
                                        # the keys could be different for different echosounders.
                                        # This dictionary is set by the `set_param` method.
        self.data_type = 'all'      # type of data to be converted into netcdf or zarr.
                                # - default to 'all'
                                # - 'GPS' are valid for EK60 and EK80 to indicate only GPS related data
                                #   (lat/lon and roll/heave/pitch) are exported.
                                # - 'XML' is valid for EK80 data only to indicate when only the XML
                                #   condiguration header is exported.
        self.combine = False
        self.compress = True
        self.overwrite = False
        self.timestamp_pattern = ''  # regex pattern for timestamp encoded in filename
        self.nmea_gps_sentence = 'GGA'  # select GPS datagram in _set_platform_dict(), default to 'GGA'
        self.set_param({})      # Initialize parameters with empty strings

    def set_source(self, file, model, xml_path=None):
        """Set source file and echosounder model.
        """
        # Check if specified model is valid
        if model == "AZFP":
            ext = '.01A'
            # Check for the xml file if dealing with an AZFP
            if xml_path:
                if '.XML' in xml_path.upper():
                    if not os.path.isfile(xml_path):
                        raise FileNotFoundError(f"There is no file named {os.path.basename(xml_path)}")
                else:
                    raise ValueError(f"{os.path.basename(xml_path)} is not an XML file")
                self.xml_path = xml_path
            else:
                raise ValueError("XML file is required for AZFP raw data")
        elif model == 'EK60' or model == 'EK80':
            ext = '.raw'
        else:
            raise ValueError(model + " is not a supported echosounder model")

        self.sonar_model = model

        # Check if given files are valid
        if isinstance(file, str):
            file = [file]
        try:
            for p in file:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"There is no file named {os.path.basename(p)}")
                if os.path.splitext(p)[1] != ext:
                    raise ValueError("Not all files are in the same format.")
        except TypeError:
            raise ValueError("file must be string or list-like")

        self.source_file = file

    def set_param(self, param_dict):
        """Allow users to set, ``platform_name``, ``platform_type``, and ``platform_code_ICES``
        to be saved during the conversion.
        """
        self._conversion_params['platform_name'] = param_dict.get('platform_name', '')
        self._conversion_params['platform_code_ICES'] = param_dict.get('platform_code_ICES', '')
        self._conversion_params['platform_type'] = param_dict.get('platform_type', '')
        self._conversion_params['survey_name'] = param_dict.get('survey_name', '')
        self._conversion_params['water_level'] = param_dict.get('water_level', 0)

    def _validate_path(self, file_format, save_path=None):
        """Assemble output file names and path.

        Parameters
        ----------
        save_path : str
            Either a directory or a file. If none then the save path is the same as the raw file.
        file_format : str            .nc or .zarr
        """

        # Raise error if output format is not .nc or .zarr
        if file_format != '.nc' and file_format != '.zarr':
            raise ValueError("File format is not .nc or .zarr")

        filenames = self.source_file

        # Default output directory taken from first input file
        self.out_dir = os.path.dirname(filenames[0])
        if save_path is not None:
            path_ext = os.path.splitext(save_path)[1]
            # Check if save_path is a file or a directory
            if path_ext == '':   # if a directory
                self.out_dir = save_path
            elif (path_ext == '.nc' or path_ext == '.zarr') and len(filenames) == 1:
                self.out_dir = os.path.dirname(save_path)
            else:  # if a file
                raise ValueError("save_path must be a directory")

        # Create folder if save_path does not exist already
        if not os.path.exists(self.out_dir):
            try:
                os.mkdir(self.out_dir)
            # Raise error if save_path is not a folder
            except FileNotFoundError:
                raise ValueError("A valid save directory was not given.")

        # Store output filenames
        files = [os.path.splitext(os.path.basename(f))[0] for f in filenames]
        self.output_file = [os.path.join(self.out_dir, f + file_format) for f in files]
        self.nc_path = [os.path.join(self.out_dir, f + '.nc') for f in files]
        self.zarr_path = [os.path.join(self.out_dir, f + '.zarr') for f in files]

    def _convert_indiv_file(self, file, output_path, save_ext):
        """Convert a single file.
        """
        params = self._conversion_params
        # use echosounder-specific object
        if self.sonar_model == 'EK60':
            c = ParseEK60
            sg = SetGroupsEK60
        elif self.sonar_model == 'EK80':
            c = ParseEK80
            sg = SetGroupsEK80
        elif self.sonar_model == 'AZFP':
            c = ParseAZFP
            sg = SetGroupsAZFP
            params['xml_path'] = self.xml_path
        else:
            raise ValueError("Unknown sonar model", self.sonar_model)


        # Check if file exists
        if os.path.exists(output_path) and self.overwrite:
            # Remove the file if self.overwrite is true
            print("          overwriting: " + output_path)
            self._remove(output_path)
        if os.path.exists(output_path):
            # Otherwise, skip saving
            print(f'          ... this file has already been converted to {save_ext}, conversion not executed.')
        else:
            c = c(file, params)
            c.parse_raw()
            sg = sg(c, input_file=file, output_path=output_path, save_ext=save_ext,
                    compress=self.compress, overwrite=self.overwrite)
            sg.save()

    def _check_param_consistency(self):
        """Check consistency of key params so that xr.open_mfdataset() will work.
        """
        # TODO: need to figure out exactly what parameters to check.
        #  These will be different for each echosounder model.
        #  Can think about using something like
        #  _check_tx_param_uniqueness() or _check_env_param_uniqueness() for EK60/EK80,
        #  and _check_uniqueness() for AZFP.
        # if self.sonar_model == 'EK60':
        #     pass
        # elif self.sonar_model == 'EK80':
        #     pass
        # elif self.sonar_model == 'AZFP':
        #     parser._check_uniqueness()
        return True

    @staticmethod
    def _remove(path):
        fname, ext = os.path.splitext(path)
        if ext == '.zarr':
            shutil.rmtree(path)
        else:
            os.remove(path)

    def combine_files(self, src_files=None, save_path=None, remove_orig=True):
        """Combine output files when self.combine=True.
        """
        if self._check_param_consistency():
            # code to actually combine files
            print('combining files...')
            src_files = self.output_file if src_files is None else src_files
            if save_path is None:
                fname, ext = os.path.splitext(src_files[0])
                save_path = fname + '[combined]' + ext
            elif isinstance(save_path, str):
                fname, ext = os.path.splitext(save_path)
                # If save_path is a directory. (It must exist due to validate_path)
                if ext == '':
                    file = os.path.basename(src_files[0])
                    fname, ext = os.path.splitext(file)
                    save_path = os.path.join(save_path, fname + '[combined]' + ext)
            else:
                raise ValueError("Invalid save path")

            # Open multiple files as one dataset of each group and save them into a single file
            with xr.open_dataset(src_files[0]) as ds_top:
                ds_top.to_netcdf(path=save_path, mode='w')
            with xr.open_dataset(src_files[0], group='Provenance') as ds_prov:
                ds_prov.to_netcdf(path=save_path, mode='a', group='Provenance')
            with xr.open_dataset(src_files[0], group='Sonar') as ds_sonar:
                ds_sonar.to_netcdf(path=save_path, mode='a', group='Sonar')
            with xr.open_mfdataset(src_files, group='Beam', combine='by_coords', data_vars='minimal') as ds_beam:
                ds_beam.to_netcdf(path=save_path, mode='a', group='Beam')
            if self.sonar_model == 'AZFP':
                with xr.open_mfdataset(src_files, group='Environment', combine='by_coords') as ds_env:
                    ds_env.to_netcdf(path=save_path, mode='a', group='Environment')
            else:
                with xr.open_dataset(src_files[0], group='Environment') as ds_env:
                    ds_env.to_netcdf(path=save_path, mode='a', group='Environment')
            # The platform group for AZFP does not have coordinates, so it must be handled differently from EK60
            if self.sonar_model == 'AZFP':
                with xr.open_dataset(src_files[0], group='Platform') as ds_plat:
                    ds_plat.to_netcdf(path=save_path, mode='a', group='Platform')
            else:
                with xr.open_mfdataset(src_files, group='Platform', combine='by_coords') as ds_plat:
                    ds_plat.to_netcdf(path=save_path, mode='a', group='Platform')
            if self.sonar_model == 'AZFP':
                # EK60 does not have the "vendor specific" group
                with xr.open_mfdataset(src_files, group='Vendor', combine='by_coords', data_vars='minimal') as ds_vend:
                    ds_vend.to_netcdf(path=save_path, mode='a', group='Vendor')
            else:
                with xr.open_mfdataset(src_files, group='Platform/NMEA',
                                       combine='nested', concat_dim='time', decode_times=False) as ds_nmea:
                    ds_nmea.to_netcdf(path=save_path, mode='a', group='Platform/NMEA')

            # Delete files after combining
            if remove_orig:
                for f in src_files:
                    self._remove(f)
        else:
            print('cannot combine files...')

    def to_netcdf(self, save_path=None, data_type='all', compress=True, overwrite=True, combine=False, parallel=False):
        """Convert a file or a list of files to netcdf format.
        """
        self.data_type = data_type
        self.compress = compress
        self.combine = combine
        self.overwrite = overwrite

        self._validate_path('.nc', save_path)
        # Sequential or parallel conversion
        if not parallel:
            for i, file in enumerate(self.source_file):
                # convert file one by one into path set by validate_path()
                self._convert_indiv_file(file=file, output_path=self.output_file[i], save_ext='.nc')
        # else:
            # use dask syntax but we'll probably use something else, like multiprocessing?
            # delayed(self._convert_indiv_file(file=file, path=save_path, output_format='netcdf'))

        # combine files if needed
        if self.combine:
            self.combine_files(save_path=save_path, remove_orig=True)

    def to_zarr(self, save_path=None, data_type='all', compress=True, combine=False, overwrite=False, parallel=False):
        """Convert a file or a list of files to zarr format.
        """
        self.data_type = data_type
        self.compress = compress
        self.combine = combine
        self.overwrite = overwrite

        self._validate_path('.zarr', save_path)
        # Sequential or parallel conversion
        if not parallel:
            for i, file in enumerate(self.source_file):
                # convert file one by one into path set by validate_path()
                self._convert_indiv_file(file=file, output_path=self.output_file[i], save_ext='.zarr')
        # else:
            # use dask syntax but we'll probably use something else, like multiprocessing?
            # delayed(self._convert_indiv_file(file=file, path=save_path, output_format='netcdf'))

        # combine files if needed
        if self.combine:
            self.combine_files(save_path=save_path, remomve_orig=True)
