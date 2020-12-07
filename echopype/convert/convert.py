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
    def __init__(self, file=None, model=None, xml_path=None):
        # Attributes
        self.sonar_model = None     # type of echosounder
        self.xml_path = ''          # path to xml file (AZFP only)
                                    # users will get an error if try to set this directly for EK60 or EK80 data
        self.source_file = None     # input file path or list of input file paths
        self.output_path = None     # converted file path or list of converted file paths
        self.cw_files = []          # additional files created when setting groups (EK80 only)
        self._source_path = None    # for convenience only, the path is included in source_file already;
                                    # user should not interact with this directly
        self._output_path = None    # for convenience only, the path is included in source_file already;
                                    # user should not interact with this directly
        self._conversion_params = {}    # a dictionary of conversion parameters,
                                        # the keys could be different for different echosounders.
                                        # This dictionary is set by the `set_param` method.
        self.data_type = 'ALL'      # type of data to be converted into netcdf or zarr.
                                # - default to 'ALL'
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
        try:
            for p in file:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"There is no file named {os.path.basename(p)}")
                if os.path.splitext(p)[1] != ext:
                    # TODO: @ngkavin:
                    #  Change the msg to explicitly say what is expected and what is received.
                    #  for example when I passed in 'ABC.nc' it is not obvious what I did wrong
                    #  with the current msg.
                    #
                    raise ValueError("Not all files are in the same format.")
        except TypeError:
            # TODO: @ngkavin: Change this to use if-else, since you can test the string and list type easily.
            raise ValueError("file must be string or list-like")

        self.source_file = file

    def set_param(self, param_dict):
        """Allow users to set, ``platform_name``, ``platform_type``, ``platform_code_ICES``, ``water_level``,
        and ```survey_name`` to be saved during the conversion. Extra values are saved to the top level.
        ```nmea_gps_sentence``` can be specified to save specific messages in a nmea string.
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
        self.zarr_path = self.output_path.copy()
        self.nc_path = None

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
        self.nc_path = [os.path.join(out_dir, f + '.nc') for f in files]
        self.zarr_path = [os.path.join(out_dir, f + '.zarr') for f in files]

    def _fetch_cw_files(self, c, output_path):
        if c.bb_ch_ids and c.cw_ch_ids:
            fname, ext = os.path.splitext(output_path)
            new_path = fname + '_cw' + ext
            self.cw_files.append(new_path)

    def _convert_indiv_file(self, file, output_path=None, save_ext=None):
        """Convert a single file.
        """
        # use echosounder-specific object
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
                self._remove(output_path)
            if os.path.exists(output_path):
                # Otherwise, skip saving
                print(f"          ... this file has already been converted to {save_ext}, conversion not executed.")
                return

        c = c(file, params=params)
        c.parse_raw()
        if self.sonar_model in ['EK80', 'EA640']:
            self._fetch_cw_files(c, output_path)
        sg = sg(c, input_file=file, output_path=output_path, save_ext=save_ext, compress=self.compress,
                overwrite=self.overwrite, params=self._conversion_params, sonar_model=self.sonar_model)
        sg.save()

    @staticmethod
    def _remove(path):
        """Used to delete .nc or .zarr files"""
        if isinstance(path, MutableMapping):
            path.fs.rm(path.root, recursive=True)
        else:
            fname, ext = os.path.splitext(path)
            if ext == '.zarr':
                shutil.rmtree(path)
            else:
                os.remove(path)

    def _path_list_to_str(self):
        # Convert to string if only 1 output file
        self.output_path = self.output_path[0] if len(self.output_path) == 1 else self.output_path
        self.nc_path = self.nc_path[0] if self.nc_path is not None and len(self.nc_path) == 1 else self.nc_path
        self.zarr_path = self.zarr_path[0] if len(self.zarr_path) == 1 else self.zarr_path[0]

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

        # TODO: WJ: check the now merged zarr engine for open_mfdataset
        def open_mfzarr(files, group, combine='by_coords', data_vars='minimal',
                        concat_dim='time', decode_times=False, compat='no_conflicts'):
            def modify(task, group):
                return task
            # This is basically what open_mfdataset does
            open_kwargs = dict(decode_cf=True, decode_times=decode_times)
            open_tasks = [dask.delayed(xr.open_zarr)(f, group=group, **open_kwargs) for f in files]
            tasks = [dask.delayed(modify)(task, group) for task in open_tasks]
            datasets = dask.compute(tasks)  # get a list of xarray.Datasets
            # Return combined data
            if combine == 'by_coords':
                combined = xr.combine_by_coords(datasets[0], data_vars=data_vars, compat=compat)
            else:
                combined = xr.combine_nested(datasets[0], concat_dim=concat_dim, data_vars=data_vars)
            return combined

        def set_filetype(save_path):
            save_path = save_path.root if isinstance(save_path, MutableMapping) else save_path
            ext = os.path.splitext(save_path)[1]
            if ext == '.nc':
                return 'netcdf4', ext
            elif ext == '.zarr':
                return 'zarr', ext

        def _save(ext, ds, path, mode, group=None):
            # Allows saving both NetCDF and Zarr files from an xarray dataset
            if ext == '.nc':
                ds.to_netcdf(path=path, mode=mode, group=group)
            elif ext == '.zarr':
                ds.to_zarr(store=path, mode=mode, group=group)

        def check_vendor_consistency(files):
            filter_coeffs = []
            for f in files:
                with xr.open_dataset(f, group='Vendor', engine=engine) as ds:
                    filter_coeffs.append(ds)
            # Check to see if filter coefficients change across files. Raise error if it does
            try:
                xr.merge(filter_coeffs, combine_attrs='identical')
            except xr.MergeError:
                raise ValueError("Filter coefficients must be the same across the files being combined")
            del filter_coeffs
            return True

        def split_bb_cw_files(files):
            # Sorts the cw and bb files from EK80 into groups
            if self.sonar_model in ['EK80', 'EA640']:
                file_groups = [[], []]
                for f in files:
                    if '_cw' in f:
                        file_groups[1].append(f)
                    else:
                        file_groups[0].append(f)
            else:
                file_groups = [files]
            return file_groups

        def _coerce(ds, group):
            if self.sonar_model == 'EK80' or self.sonar_model == 'EK60':
                if group == 'Beam':
                    ds['wbt_software_version'] = ds['wbt_software_version'].astype('<U10')
                    ds['channel_id'] = ds['channel_id'].astype('<U50')

        src_files = self.output_path if src_files is None else src_files
        file_groups = [src_files]

        def get_combined_fname(path):
            fname, ext = os.path.splitext(path)
            return fname + '[combined]' + ext

        if self.sonar_model in ['EK80', 'EA640']:
            file_groups = split_bb_cw_files(src_files + self.cw_files)

        # Construct save path
        # Handle saving to cloud storage
        if isinstance(src_files[0], MutableMapping) or isinstance(save_path, MutableMapping):
            fs = src_files[0].fs if isinstance(src_files[0], MutableMapping) else save_path.fs
            if save_path is None:
                save_path = fs.get_mapper(get_combined_fname(src_files[0].root))
            elif isinstance(save_path, MutableMapping):
                fname, ext = os.path.splitext(save_path.root)
                if ext == '':
                    save_path = save_path.root + '/' + get_combined_fname(os.path.basename(src_files[0].root))
                    save_path = fs.get_mapper(save_path)
                elif ext != '.zarr':
                    raise ValueError("save_path must be a zarr file")
            else:
                raise ValueError("save_path must be a MutableMapping to a cloud store")
        # Handle saving to local paths
        else:
            if save_path is None:
                save_path = get_combined_fname(src_files[0])
            elif isinstance(save_path, str):
                fname, ext = os.path.splitext(save_path)
                # If save_path is a directory. (It must exist due to validate_path)
                if ext == '':
                    save_path = os.path.join(save_path, get_combined_fname(os.path.basename(src_files[0])))
                elif ext != '.nc' and ext != '.zarr':
                    raise ValueError("save_path must be '.nc' or '.zarr'")
            else:
                raise ValueError("Invalid save_path")

        # Get the correct xarray functions for opening datasets
        engine, ext = set_filetype(save_path)

        for i, file_group in enumerate(file_groups):
            print('combining files...')
            # Append '_cw' to EK80 filepath if combining CW files
            if i == 1:
                if not file_group:
                    continue
                fname, ext = os.path.splitext(save_path)
                save_path = fname + '_cw' + ext
            # Open multiple files as one dataset of each group and save them into a single file
            # Combine Top-level
            with xr.open_dataset(file_group[0], engine=engine) as ds_top:
                _save(ext, ds_top, save_path, 'w')
            # Combine Provenance
            with xr.open_dataset(file_group[0], group='Provenance', engine=engine) as ds_prov:
                _save(ext, ds_prov, save_path, 'a', group='Provenance')
            # Combine Sonar
            with xr.open_dataset(file_group[0], group='Sonar', engine=engine) as ds_sonar:
                _save(ext, ds_sonar, save_path, 'a', group='Sonar')
            # Combine Beam
            try:
                with xr.open_mfdataset(file_group, group='Beam', decode_times=False, combine='nested',
                                       concat_dim='ping_time', data_vars='minimal', engine=engine) as ds_beam:
                    _coerce(ds_beam, 'Beam')
                    _save(ext, ds_beam.chunk({'range_bin': 25000, 'ping_time': 100}),
                          save_path, 'a', group='Beam')
            except xr.MergeError as e:
                var = str(e).split("'")[1]
                raise ValueError(f"Files cannot be combined due to {var} changing across the files")
            # Combine Environment
            # AZFP environment changes as a function of ping time
            # TODO test for ek80 and 60
            with xr.open_mfdataset(file_group, group='Environment', combine='nested', concat_dim='ping_time',
                                   data_vars='minimal', engine=engine) as ds_env:
                _save(ext, ds_env, save_path, 'a', group='Environment')
            # Combine Platfrom
            # The platform group for AZFP does not have coordinates, so it must be handled differently from EK60
            if self.sonar_model == 'AZFP':
                with xr.open_dataset(file_group[0], group='Platform', engine=engine) as ds_plat:
                    _save(ext, ds_plat, save_path, 'a', group='Platform')
            else:
                with xr.open_mfdataset(file_group, group='Platform', decode_times=False, combine='nested',
                                       concat_dim='ping_time', data_vars='minimal', engine=engine) as ds_plat:
                    if self.sonar_model in ['EK80', 'EA640']:
                        _save(ext, ds_plat.chunk({'location_time': 100, 'mru_time': 100}),
                            save_path, 'a', group='Platform')
                    else:
                        _save(ext, ds_plat.chunk({'location_time': 100, 'ping_time': 100}),
                              save_path, 'a', group='Platform')
            # Combine Sonar-specific
            if self.sonar_model == 'AZFP':
                # EK60 does not have the "vendor specific" group
                with xr.open_mfdataset(file_group, group='Vendor',
                                       combine='by_coords', data_vars='minimal', engine=engine) as ds_vend:
                    _save(ext, ds_vend, save_path, 'a', group='Vendor')
            if self.sonar_model in ['EK80', 'EK60', 'EA640']:
                # AZFP does not record NMEA data
                # TODO: Look into why decode times = True for beam does not error out
                with xr.open_mfdataset(file_group, group='Platform/NMEA', decode_times=False,
                                       combine='nested', concat_dim='time', engine=engine) as ds_nmea:
                    _save(ext, ds_nmea.chunk({'location_time': 100}).astype('str'),
                          save_path, 'a', group='Platform/NMEA')
            if self.sonar_model in ['EK80', 'EA640']:
                if check_vendor_consistency(file_group):
                    # Save filter coefficients in EK80
                    with xr.open_dataset(file_group[0], group='Vendor', engine=engine) as ds_vend:
                        _save(ext, ds_vend, save_path, 'a', group='Vendor')

            print("Files combined into", save_path)

        # Delete files after combining
        if remove_orig:
            for f in src_files + self.cw_files:
                self._remove(f)
        return True

    def to_netcdf(self, save_path=None, data_type='ALL', compress=True,
                  combine=False, overwrite=False, parallel=False):
        """Convert a file or a list of files to NetCDF format.

        Parameters
        ----------
        save_path : str
            path that converted .nc file will be saved
        data_type : str {'ALL', 'GPS', 'CONFIG_XML', 'ENV_XML'}
            select specific datagrams to save (EK60 and EK80 only)
            Defaults to ``ALL``
        compress : bool
            whether or not to preform compression on data variables
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
            for i, file in enumerate(self.source_file):
                # convert file one by one into path set by validate_path()
                self._convert_indiv_file(file=file, output_path=self.output_path[i], save_ext='.nc')
        else:
            # # use dask syntax but we'll probably use something else, like multiprocessing?
            # open_tasks = [dask.delayed(self._convert_indiv_file)(file=file,
            #                                                      output_path=self.output_path[i], save_ext='.nc')
            #               for i, file in enumerate(self.source_file)]
            # datasets = dask.compute(open_tasks)  # get a list of xarray.Datasets
            pass

        self._path_list_to_str()
        # combine files if needed
        if self.combine:
            self.combine_files(save_path=save_path, remove_orig=True)

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
            whether or not to preform compression on data variables
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
            for i, file in enumerate(self.source_file):
                # convert file one by one into path set by validate_path()
                self._convert_indiv_file(file=file, output_path=self.output_path[i], save_ext='.zarr')
        # else:
            # use dask syntax but we'll probably use something else, like multiprocessing?
            # delayed(self._convert_indiv_file(file=file, path=save_path, output_format='netcdf'))

        self._path_list_to_str()
        # combine files if needed
        if self.combine:
            self.combine_files(save_path=save_path, remove_orig=True)

    def to_xml(self, save_path=None, data_type='CONFIG_XML'):
        """Save an xml file containing the configuration of the transducer and transceiver (EK80/EA640 only)

        Parameters
        ----------
        save_path : str
            path that converted .xml file will be saved
        type: str
            which XML to export
            either 'CONFIG' or 'ENV'
        """
        if self.sonar_model not in ['EK80', 'EA640']:
            raise ValueError("Exporting to xml is not availible for " + self.sonar_model)
        if data_type != 'CONFIG_XML' and data_type != 'ENV_XML':
            raise ValueError(f"data_type must be either 'CONFIG_XML' or 'ENV_XML' not {data_type}")
        self._validate_path('.xml', save_path)
        for i, file in enumerate(self.source_file):
            # convert file one by one into path set by validate_path()
            tmp = ParseEK80(file, params=[data_type, 'EXPORT'])
            tmp.parse_raw()
            with open(self.output_path[i], 'w') as xml_file:
                data = tmp.config_datagram['xml'] if data_type == 'CONFIG_XML' else tmp.environment['xml']
                xml_file.write(data)
        self._path_list_to_str()
