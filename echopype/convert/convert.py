"""
UI class for converting raw data from different echosounders to netcdf or zarr.
"""
import os
import shutil
import xarray as xr
import dask
from collections import MutableMapping
from .parse_azfp import ParseAZFP
from .parse_ek60 import ParseEK60
from .parse_ek80 import ParseEK80
from .set_groups_azfp import SetGroupsAZFP
from .set_groups_ek60 import SetGroupsEK60
from .set_groups_ek80 import SetGroupsEK80


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

        # set parameters that may not already be in source files
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
    def __init__(self, file=None, model=None, xml_path=None):
        # Attributes
        self.sonar_model = None     # type of echosounder
        self.xml_path = ''          # path to xml file (AZFP only)
                                    # users will get an error if try to set this directly for EK60 or EK80 data
        self.source_file = None     # input file path or list of input file paths
        self.output_path = None     # converted file path or list of converted file paths
        self.output_cw_files = []          # additional files created when setting groups (EK80 only)
        self._conversion_params = {}    # a dictionary of conversion parameters,
                                        # the keys could be different for different echosounders.
                                        # This dictionary is set by the `set_param` method.
        self.data_type = 'ALL'      # type of data to be converted into netcdf or zarr.
                                # - default to 'ALL'
                                # - 'GPS' are valid for EK60 and EK80 to indicate only GPS related data
                                #   (lat/lon and roll/heave/pitch) are exported.
                                # - 'XML' is valid for EK80 data only to indicate when only the XML
                                #   configuration header is exported.
        self.combine = False
        self.compress = True
        self.overwrite = False
        self.timestamp_pattern = ''  # regex pattern for timestamp encoded in filename
        self.nmea_gps_sentence = 'GGA'  # select GPS datagram in _set_platform_dict(), default to 'GGA'
        self.set_param({})      # Initialize parameters with empty strings
        self.set_source(file, model, xml_path)

    def __str__(self):
        """Overload the print function to allow user to print basic properties of this object.

        Print out should include: source file name, source folder location, echosounder model.
        """
        if self.sonar_model is None:
            return "empty echopype convert object (call set_source)"
        else:
            return (f"echopype {self.sonar_model} convert object\n" +
                    f"\tsource filename: {[os.path.basename(f) for f in self.source_file]}\n" +
                    f"\tsource directory: {os.path.dirname(self.source_file[0])}")

    def __repr__(self):
        return self.__str__()

    def set_source(self, file, model, xml_path=None):
        """Set source and echosounder model

        Parameters
        ----------
        file : str, list
            A file or list of files to be converted
        model : str
            echosounder model. "AZFP", "EK60", or "EK80"
        xml_path : str
            path to xml file required for AZFP conversion
        """
        if file is None:
            return
        elif file is not None and model is None:
            print("Please specify the echosounder model")
            return

        # TODO: Allow pointing directly a cloud data source
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
        elif model in ['EK60', 'EK80', 'EA640']:
            ext = '.raw'
        else:
            raise ValueError(model + " is not a supported echosounder model")

        self.sonar_model = model

        # Check if given files are valid
        if isinstance(file, str):
            file = [file]
        if not isinstance(file, list):
            raise ValueError("file must be a string or list of strings")
        for p in file:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"There is no file named {os.path.basename(p)}")
            if os.path.splitext(p)[1] != ext:

                raise ValueError(f"Not all files are in the same format. Expecting a {ext} file but got {p}")

        self.source_file = file

    def set_param(self, param_dict):
        """Allow users to set ``platform_name``, ``platform_type``, ``platform_code_ICES``, ``water_level``,
        and ``survey_name`` to be saved during the conversion. Extra values are saved to the top level.
        ``nmea_gps_sentence`` can be specified to save specific messages in a nmea string.
        nmea sentence Defaults to 'GGA'. Originally ['GGA', 'GLL', 'RMC'].
        """
        # Platform
        self._conversion_params['platform_name'] = param_dict.get('platform_name', '')
        self._conversion_params['platform_code_ICES'] = param_dict.get('platform_code_ICES', '')
        self._conversion_params['platform_type'] = param_dict.get('platform_type', '')
        self._conversion_params['water_level'] = param_dict.get('water_level', None)
        self._conversion_params['nmea_gps_sentence'] = param_dict.get('nmea_gps_sentence', 'GGA')
        # Top level
        self._conversion_params['survey_name'] = param_dict.get('survey_name', '')
        for k, v in param_dict.items():
            if k not in self._conversion_params:
                self._conversion_params[k] = v

    def _validate_object_store(self, store):
        fs = store.fs
        root = store.root
        fname, ext = os.path.splitext(root)
        if ext == '':
            files = [root + '/' + os.path.splitext(os.path.basename(f))[0] + '.zarr'
                     for f in self.source_file]
            self.output_path = [fs.get_mapper(f) for f in files]
        elif ext == '.zarr':
            if len(self.source_file) > 1:
                raise ValueError("save_path must be a directory")
            else:
                self.output_path = [store]

    def _validate_path(self, file_format, save_path=None):
        """Assemble output file names and path.

        Parameters
        ----------
        save_path : str
            Either a directory or a file. If none then the save path is the same as the raw file.
        file_format : str {'.nc', '.zarr'}
        """

        filenames = self.source_file

        # Default output directory taken from first input file
        out_dir = os.path.dirname(filenames[0])
        if save_path is not None:
            dirname, fname = os.path.split(save_path)
            basename, path_ext = os.path.splitext(fname)
            if path_ext != file_format and path_ext != '':
                raise ValueError("Saving {file_format} file but save path is to a {path_ext} file")
            if path_ext != '.nc' and path_ext != '.zarr' and path_ext != '.xml' and path_ext != '':
                raise ValueError("File format must be .nc, .zarr, or .xml")
            # Check if save_path is a file or a directory
            if path_ext == '':   # if a directory
                out_dir = save_path
            elif len(filenames) == 1:
                if dirname != '':
                    out_dir = dirname
            else:  # if a file
                raise ValueError("save_path must be a directory")

        # Create folder if save_path does not exist already
        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)
            # Raise error if save_path is not a folder
            except FileNotFoundError:
                raise ValueError("A valid save directory was not given.")

        # Store output filenames
        if save_path is not None and not os.path.isdir(save_path):
            files = [os.path.basename(basename)]
        else:
            files = [os.path.splitext(os.path.basename(f))[0] for f in filenames]
        self.output_path = [os.path.join(out_dir, f + file_format) for f in files]

    def _construct_cw_file_path(self, c, output_path):
        if c.ch_ids['power'] and c.ch_ids['complex']:
            fname, ext = os.path.splitext(output_path)
            new_path = fname + '_cw' + ext
            self.output_cw_files.append(new_path)

    def _convert_indiv_file(self, file, output_path=None, save_ext=None):
        """Convert a single file.
        """
        # Use echosounder-specific object
        if self.sonar_model == 'EK60':
            c = ParseEK60
            sg = SetGroupsEK60
            params = self.data_type
        elif self.sonar_model in ['EK80', 'EA640']:
            c = ParseEK80
            sg = SetGroupsEK80
            params = self.data_type
        elif self.sonar_model == 'AZFP':
            c = ParseAZFP
            sg = SetGroupsAZFP
            params = self.xml_path
        else:
            raise ValueError("Unknown sonar model", self.sonar_model)

        # Handle saving to cloud or local filesystem
        # TODO: @ngkvain: You mean this took long before, what is the latest status?
        if isinstance(output_path, MutableMapping):
            if not self.overwrite:
                if output_path.fs.exists(output_path.root):
                    print(f"          ... this file has already been converted to {save_ext}, " +
                          "conversion not executed.")
                    return
            # output_path.fs.rm(output_path.root, recursive=True)
        else:
            # Check if file exists
            if os.path.exists(output_path) and self.overwrite:
                # Remove the file if self.overwrite is true
                print("          overwriting: " + output_path)
                self._remove_files(output_path)
            if os.path.exists(output_path):
                # Otherwise, skip saving
                print(f"          ... this file has already been converted to {save_ext}, conversion not executed.")
                return

        # Actually parsing and saving file(s)
        c = c(file, params=params)
        c.parse_raw()
        if self.sonar_model in ['EK80', 'EA640']:
            self._construct_cw_file_path(c, output_path)
        sg = sg(c, input_file=file, output_path=output_path, save_ext=save_ext, compress=self.compress,
                overwrite=self.overwrite, params=self._conversion_params, sonar_model=self.sonar_model)
        sg.save()

    @staticmethod
    def _remove_files(path):
        """Used to delete .nc or .zarr files"""
        if isinstance(path, MutableMapping):
            path.fs.rm(path.root, recursive=True)
        else:
            fname, ext = os.path.splitext(path)
            if ext == '.zarr':
                shutil.rmtree(path)
            else:
                os.remove(path)

    @staticmethod
    def _path_list_to_str(path):
        # Takes a list of filepaths and if there is only 1 path, return that path as a string
        if len(path) == 1:
            return path[0]

    def combine_files(self, src_files=None, save_path=None, remove_orig=False):
        """Combine output files when self.combine=True.

        Parameters
        ----------
        src_files : list
            List of NetCDF or Zarr files to combine
        save_path : str
            Either a directory or a file. If none, use the name of the first ``src_file``
        remove_orig : bool
            Whether or not to remove the files in ``src_files``
            Defaults to ``False``

        Returns
        -------
        True or False depending on whether or not the combination was successful
        """
        if len(self.source_file) < 2:
            print("Combination did not occur as there is only 1 source file")
            return False

        def get_filetype(file):
            file = file.root if isinstance(file, MutableMapping) else file
            ext = os.path.splitext(file)[1]
            if ext == '.nc':
                return 'netcdf4'
            elif ext == '.zarr':
                return 'zarr'

        def coerce(ds, group):
            if group == 'Beam':
                if self.sonar_model == 'EK80':
                    ds['transceiver_software_version'] = ds['transceiver_software_version'].astype('<U10')
                    ds['channel_id'] = ds['channel_id'].astype('<U50')
                elif self.sonar_model == 'EK60':
                    ds['gpt_software_version'] = ds['gpt_software_version'].astype('<U10')
                    ds['channel_id'] = ds['channel_id'].astype('<U50')

        # TODO: @ngkvain: factor out path handling to a separate function,
        #  it can be absorbed into _validate_path as well.
        #  Let's discuss this along with whether to have ..io.validate_path
        # Construct save path
        def get_save_path(save_path, source_paths):
            def get_combined_fname(path):
                fname, ext = os.path.splitext(path)
                return fname + '[combined]' + ext
            # Handle saving to cloud storage
            # Flags on whether the source or output is a path to an object store
            cloud_src = isinstance(source_paths[0], MutableMapping)
            cloud_save_path = isinstance(save_path, MutableMapping)
            if cloud_src or cloud_save_path:
                fs = source_paths[0].fs if cloud_src else save_path.fs
                if save_path is None:
                    save_path = fs.get_mapper(get_combined_fname(source_paths[0].root))
                elif cloud_save_path:
                    fname, ext = os.path.splitext(save_path.root)
                    if ext == '':
                        save_path = save_path.root + '/' + get_combined_fname(os.path.basename(source_paths[0].root))
                        save_path = fs.get_mapper(save_path)
                    elif ext != '.zarr':
                        raise ValueError("save_path must be a zarr file")
                else:
                    raise ValueError("save_path must be a MutableMapping to a cloud store")
            # Handle saving to local paths
            else:
                if save_path is None:
                    save_path = get_combined_fname(source_paths[0])
                elif isinstance(save_path, str):
                    fname, ext = os.path.splitext(save_path)
                    # If save_path is a directory. (It must exist due to validate_path)
                    if ext == '':
                        save_path = os.path.join(save_path, get_combined_fname(os.path.basename(source_paths[0])))
                    elif ext != '.nc' and ext != '.zarr':
                        raise ValueError("save_path must be '.nc' or '.zarr'")
                else:
                    raise ValueError("Invalid save_path")
            return save_path

        def perform_combination(input_paths, output_path, engine):
            # Function that opens multiple file groups and writes to a single file
            def save_file(engine, ds, path, mode, group=None):
                # Allows saving both NetCDF and Zarr files from an xarray dataset
                if engine == 'netcdf4':
                    ds.to_netcdf(path=path, mode=mode, group=group)
                elif engine == 'zarr':
                    ds.to_zarr(store=path, mode=mode, group=group)

            print('combining files...')
            # Open multiple files as one dataset of each group and save them into a single file
            # Combine Top-level
            with xr.open_dataset(input_paths[0], engine=engine) as ds_top:
                save_file(engine, ds_top, output_path, 'w')

            # Combine Provenance
            with xr.open_dataset(input_paths[0], group='Provenance', engine=engine) as ds_prov:
                save_file(engine, ds_prov, output_path, 'a', group='Provenance')

            # Combine Sonar
            with xr.open_dataset(input_paths[0], group='Sonar', engine=engine) as ds_sonar:
                save_file(engine, ds_sonar, output_path, 'a', group='Sonar')

            try:
                # Combine Beam
                with xr.open_mfdataset(input_paths, group='Beam', decode_times=False, combine='nested',
                                       concat_dim='ping_time', data_vars='minimal', engine=engine) as ds_beam:
                    coerce(ds_beam, 'Beam')
                    save_file(engine, ds_beam.chunk({'range_bin': 25000, 'ping_time': 100}),
                              output_path, 'a', group='Beam')
                # Combine Environment
                # AZFP environment changes as a function of ping time
                # TODO test for ek60
                with xr.open_mfdataset(input_paths, group='Environment', combine='nested', concat_dim='ping_time',
                                       data_vars='minimal', engine=engine) as ds_env:
                    save_file(engine, ds_env, output_path, 'a', group='Environment')

                # Combine Platform
                # The platform group for AZFP does not have coordinates, so it must be handled differently from EK60
                if self.sonar_model == 'AZFP':
                    with xr.open_mfdataset(input_paths, group='Platform', combine='by_coords',
                                           compat='identical', engine=engine) as ds_plat:
                        save_file(engine, ds_plat, output_path, 'a', group='Platform')
                elif self.sonar_model in ['EK60', 'EK80', 'EA640']:
                    with xr.open_mfdataset(input_paths, group='Platform', decode_times=False, combine='nested',
                                           concat_dim='ping_time', data_vars='minimal', engine=engine) as ds_plat:
                        if self.sonar_model in ['EK80', 'EA640']:
                            save_file(engine, ds_plat.chunk({'location_time': 100, 'mru_time': 100}),
                                      output_path, 'a', group='Platform')
                        else:
                            save_file(engine, ds_plat.chunk({'location_time': 100, 'ping_time': 100}),
                                      output_path, 'a', group='Platform')
                        # AZFP does not record NMEA data
                        # TODO: Look into why decode times = True for beam does not error out
                        with xr.open_mfdataset(input_paths, group='Platform/NMEA',
                                               decode_times=False, data_vars='minimal',
                                               combine='nested', concat_dim='location_time', engine=engine) as ds_nmea:
                            save_file(engine, ds_nmea.chunk({'location_time': 100}).astype('str'),
                                      output_path, 'a', group='Platform/NMEA')

                # Combine sonar model-specific
                if self.sonar_model == 'AZFP':
                    with xr.open_mfdataset(input_paths, group='Vendor', combine='nested',
                                           concat_dim='ping_time', data_vars='minimal', engine=engine) as ds_vend:
                        save_file(engine, ds_vend, output_path, 'a', group='Vendor')
                elif self.sonar_model in ['EK80', 'EA640']:
                    # EK60 does not have the "vendor specific" group
                    with xr.open_mfdataset(input_paths, group='Vendor', combine='nested',
                                           concat_dim='location_time', compat='no_conflicts',
                                           data_vars='minimal', engine=engine) as ds_vend:
                        coerce(ds_vend, 'Vendor')
                        save_file(engine, ds_vend, output_path, 'a', group='Vendor')

            except xr.MergeError as e:
                var = str(e).split("'")[1]
                raise ValueError(f"Files cannot be combined due to {var} changing across the files")

            print("All input files combined into " + output_path)

        src_files = self.output_path if src_files is None else src_files
        save_path = get_save_path(save_path, src_files)
        # Get the correct xarray functions for opening datasets
        filetype = get_filetype(save_path)
        perform_combination(src_files, save_path, filetype)
        # Update output_path to be the combined path name
        self.output_path = [save_path]
        if self.output_cw_files:
            # Append '_cw' to EK80 filepath if combining CW files
            fname, ext = os.path.splitext(save_path)
            save_path_cw = fname + '_cw' + ext
            perform_combination(self.output_cw_files, save_path_cw, filetype)
            self.output_path.append(save_path_cw)

        # Delete individual files after combining
        if remove_orig:
            for f in src_files + self.output_cw_files:
                self._remove_files(f)

    def to_netcdf(self, save_path=None, data_type='ALL', compress=True,
                  combine=False, overwrite=False, parallel=False):
        """Convert a file or a list of files to NetCDF format.

        Parameters
        ----------
        save_path : str
            path that converted .nc file will be saved
        data_type : str {'ALL', 'GPS', 'CONFIG', 'ENV'}
            select specific datagrams to save (EK60 and EK80 only)
            Defaults to ``ALL``
        compress : bool
            whether or not to perform compression on data variables
            Defaults to ``True``
        combine : bool
            whether or not to combine all converted individual files into one file
            Defaults to ``False``
        overwrite : bool
            whether or not to overwrite existing files
            Defaults to ``False``
        parallel : bool
            whether or not to use parallel processing. (Not yet implemented)
        """
        self.data_type = data_type
        self.compress = compress
        self.combine = combine
        self.overwrite = overwrite

        self._validate_path('.nc', save_path)
        # Sequential or parallel conversion
        if not parallel:
            for idx, file in enumerate(self.source_file):
                # convert file one by one into path set by validate_path()
                self._convert_indiv_file(file=file, output_path=self.output_path[idx], save_ext='.nc')
        else:
            # # use dask syntax but we'll probably use something else, like multiprocessing?
            # open_tasks = [dask.delayed(self._convert_indiv_file)(file=file,
            #                                                      output_path=self.output_path[i], save_ext='.nc')
            #               for i, file in enumerate(self.source_file)]
            # datasets = dask.compute(open_tasks)  # get a list of xarray.Datasets
            pass

        # combine files if needed
        if self.combine:
            self.combine_files(save_path=save_path, remove_orig=True)
        self.output_path = self._path_list_to_str(self.output_path)

    def to_zarr(self, save_path=None, data_type='ALL', compress=True, combine=False, overwrite=False, parallel=False):
        """Convert a file or a list of files to zarr format.

        Parameters
        ----------
        save_path : str
            path that converted .zarr file will be saved
        data_type : str {'ALL', 'GPS', 'CONFIG', 'ENV'}
            select specific datagrams to save (EK60 and EK80 only)
            Defaults to ``ALL``
        compress : bool
            whether or not to perform compression on data variables
            Defaults to ``True``
        combine : bool
            whether or not to combine all converted individual files into one file
            Defaults to ``False``
        overwrite : bool
            whether or not to overwrite existing files
            Defaults to ``False``
        parallel : bool
            whether or not to use parallel processing. (Not yet implemented)
        """
        self.data_type = data_type
        self.compress = compress
        self.combine = combine
        self.overwrite = overwrite

        if isinstance(save_path, MutableMapping):
            self._validate_object_store(save_path)
        else:
            self._validate_path('.zarr', save_path)
        # Sequential or parallel conversion
        if not parallel:
            for idx, file in enumerate(self.source_file):
                # convert file one by one into path set by validate_path()
                self._convert_indiv_file(file=file, output_path=self.output_path[idx], save_ext='.zarr')
        # else:
            # use dask syntax but we'll probably use something else, like multiprocessing?
            # delayed(self._convert_indiv_file(file=file, path=save_path, output_format='netcdf'))

        # combine files if needed
        if self.combine:
            self.combine_files(save_path=save_path, remove_orig=True)
        self.output_path = self._path_list_to_str(self.output_path)

    def to_xml(self, save_path=None, data_type='CONFIG'):
        """Save an xml file containing the configuration of the transducer and transceiver (EK80/EA640 only)

        Parameters
        ----------
        save_path : str
            path that converted .xml file will be saved
        type: str
            which XML to export
            either 'CONFIG' or 'ENV'
        """
        # Check export parameters
        if self.sonar_model not in ['EK80', 'EA640']:
            raise ValueError("Exporting to xml is not available for " + self.sonar_model)
        if data_type not in ['CONFIG', 'ENV']:
            raise ValueError(f"data_type must be either 'CONFIG' or 'ENV' not {data_type}")

        # Get export path
        self._validate_path('.xml', save_path)

        # Parse files
        for idx, file in enumerate(self.source_file):
            # convert file one by one into path set by validate_path()
            tmp = ParseEK80(file, params=[data_type, 'EXPORT_XML'])
            tmp.parse_raw()
            with open(self.output_path[idx], 'w') as xml_file:
                # Select between data types
                if data_type == 'CONFIG':
                    data = tmp.config_datagram['xml']
                elif data_type == 'ENV':
                    data = tmp.environment['xml']
                else:
                    raise ValueError("Unknown data type", data_type)
                xml_file.write(data)
        self.output_path = self._path_list_to_str(self.output_path)
