"""
UI class for converting raw data from different echosounders to netcdf or zarr.
"""
import os
import warnings
import xarray as xr
import numpy as np
import zarr
from pathlib import Path
from collections.abc import MutableMapping
from datetime import datetime as dt
import fsspec
from fsspec.implementations.local import LocalFileSystem
from .parse_azfp import ParseAZFP
from .parse_ek60 import ParseEK60
from .parse_ek80 import ParseEK80
from .set_groups_azfp import SetGroupsAZFP
from .set_groups_ek60 import SetGroupsEK60
from .set_groups_ek80 import SetGroupsEK80
from ..utils import io
from ..convert import convert_combine as combine_fcn


warnings.simplefilter('always', DeprecationWarning)

MODELS = {
    "AZFP": {
        "ext": ".01A",
        "xml": True,
        "parser": ParseAZFP,
        "set_groups": SetGroupsAZFP,
    },
    "EK60": {
        "ext": ".raw",
        "xml": False,
        "parser": ParseEK60,
        "set_groups": SetGroupsEK60
    },
    "EK80": {
        "ext": ".raw",
        "xml": False,
        "parser": ParseEK80,
        "set_groups": SetGroupsEK80
    },
    "EA640": {
        "ext": ".raw",
        "xml": False,
        "parser": ParseEK80,
        "set_groups": SetGroupsEK80
    }
}

NMEA_SENTENCE_DEFAULT = ['GGA', 'GLL', 'RMC']


# TODO: Used for backwards compatibility. Delete in future versions
def ConvertEK80(_filename=""):
    warnings.warn("`ConvertEK80` is deprecated, use `Convert(file, model='EK80')` instead.",
                  DeprecationWarning, 2)
    return Convert(file=_filename, model='EK80')


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
    def __init__(self, file=None, xml_path=None, model=None, storage_options=None):
        if model is None:
            if xml_path is None:
                model = 'EK60'
                warnings.warn("Current behavior is to default model='EK60' when no XML file is passed in as argument. "
                              "Specifying model='EK60' will be required in the future, "
                              "since .raw extension is used for many Kongsberg/Simrad sonar systems.",
                              DeprecationWarning, 2)
            else:
                xml_path = model
                model = 'AZFP'
                warnings.warn("Current behavior is to set model='AZFP' when an XML file is passed in as argument. "
                              "Specifying model='AZFP' will be required in the future.",
                              DeprecationWarning, 2)

        # TODO: Remove _zarr_path and _nc_path in the future. These are for backwards compatibility.
        # Initialize old path names (replaced by output_path). Only filled if raw2nc/raw2zarr is called
        self._zarr_path = None
        self._nc_path = None

        # Attributes
        self.sonar_model = None  # type of echosounder
        self.xml_path = ''       # path to xml file (AZFP only)
                                 # users will get an error if try to set this directly for EK60 or EK80 data
        self.source_file = None  # input file path or list of input file paths
        self.output_file = None  # converted file path or list of converted file paths
        self._conversion_params = {}  # a dictionary of conversion parameters,
                                      # the keys could be different for different echosounders.
                                      # This dictionary is set by the `set_param` method.
        self.data_type = 'ALL'  # type of data to be converted into netcdf or zarr.
                                # - default to 'ALL'
                                # - 'GPS' are valid for EK60 and EK80 to indicate only GPS related data
                                #   (lat/lon and roll/heave/pitch) are exported.
                                # - 'CONFIG' and 'ENV' are valid for EK80 data only because EK80 provides
                                #    configuration and environment information in the XML format
        self.combine = False
        self.compress = True
        self.overwrite = False
        self.set_param({})      # Initialize parameters with empty strings
        self.storage_options = storage_options if storage_options is not None else {}
        self._output_storage_options = {}
        self.set_source(file, model, xml_path)

    def __str__(self):
        """Overload the print function to allow user to print basic properties of this object.

        Print out should include: source file name, source folder location, echosounder model.
        """
        if self.sonar_model is None:
            return "empty echopype Convert object (call set_source to set sonar model and files to convert)"
        else:
            return (f"echopype {self.sonar_model} convert object\n" +
                    f"\tsource filename: {[os.path.basename(f) for f in self.source_file]}\n" +
                    f"\tsource directory: {os.path.dirname(self.source_file[0])}")

    def __repr__(self):
        return self.__str__()

    def set_source(self, file, model, xml_path=None):
        """Set source file(s) and echosounder model.

        Parameters
        ----------
        file : str, list
            A file or list of files to be converted
        model : str
            echosounder model. "AZFP", "EK60", or "EK80"
        xml_path : str
            path to xml file required for AZFP conversion
        """
        if (model is None) and (file is None):
            print('Please specify paths to raw data files and the sonar model.')
            return

        # Check paths and file types
        if file is not None:
            # Make file always a list
            if not isinstance(file, list):
                file = [file]

            # Check for path type
            if not all(isinstance(elem, str) for elem in file):
                raise ValueError("file must be a string or a list of string")

            # Check file extension and existence
            file_chk, xml_chk = self.check_files(file, model, xml_path, self.storage_options)

            self.source_file = file_chk  # this is always a list
            self.xml_path = xml_chk
        else:
            print('Please specify paths to raw data files.')

        # Set sonar model type
        if model is not None:
            # Uppercased model in case people use lowercase
            model = model.upper()

            # Check models
            if model not in MODELS:
                raise ValueError(f"Unsupported echosounder model: {model}\nMust be one of: {list(MODELS)}")

            self.sonar_model = model
        else:
            # TODO: this currently does not happen because we default to model='EK60'
            #  when not specified to be consistent with previous behavior.
            #  Remember to take this out later.
            # Ask user to provide model
            print('Please specify the sonar model.')

    def set_param(self, param_dict):
        """Allow users to set parameters to be stored in the converted files.

        The default set of parameters include:
        ``platform_name``, ``platform_type``, ``platform_code_ICES``, ``water_level``
        (to be stored in the Platform group),
        and ``survey_name`` (to be stored in the Top-level group).

        ``nmea_gps_sentence`` is used to select specific NMEA sentences,  defaults ['GGA', 'GLL', 'RMC'].

        Other parameters will be saved to the top level.
        """
        # TODO: revise docstring, give examples.
        # TODO: need to check and return valid/invalid params as done for Process
        # Parameters for the Platform group
        self._conversion_params['platform_name'] = param_dict.get('platform_name', '')
        self._conversion_params['platform_code_ICES'] = param_dict.get('platform_code_ICES', '')
        self._conversion_params['platform_type'] = param_dict.get('platform_type', '')
        self._conversion_params['water_level'] = param_dict.get('water_level', None)
        self._conversion_params['nmea_gps_sentence'] = param_dict.get('nmea_gps_sentence', NMEA_SENTENCE_DEFAULT)

        # Parameters for the Top-level group
        self._conversion_params['survey_name'] = param_dict.get('survey_name', '')
        for k, v in param_dict.items():
            if k not in self._conversion_params:
                self._conversion_params[k] = v

    # TODO: combine _validate_path and _validate_object_store
    def _validate_object_store(self, store):
        fs = store.fs
        root = store.root
        fname, ext = os.path.splitext(root)
        if ext == '':
            files = [root + '/' + os.path.splitext(os.path.basename(f))[0] + '.zarr'
                     for f in self.source_file]
            self.output_file = [fs.get_mapper(f) for f in files]
        elif ext == '.zarr':
            if len(self.source_file) > 1:
                raise ValueError("save_path must be a directory")
            else:
                self.output_file = [store]

    def _validate_path(self, file_format, save_path=None):
        """Assemble output file names and path.

        Parameters
        ----------
        save_path : str
            Either a directory or a file. If none then the save path is the same as the raw file.
        file_format : str {'.nc', '.zarr'}
        """
        if save_path is None:
            warnings.warn("save_path is not provided")
            fsmap = fsspec.get_mapper(self.source_file[0], **self.storage_options)
            fs = fsmap.fs

            if not isinstance(fs, LocalFileSystem):
                # Defaults to Echopype directory if source is not localfile system
                current_dir = Path.cwd()
                # Check permission, raise exception if no permission
                io.check_file_permissions(current_dir)
                out_dir = current_dir.joinpath(Path('temp_echopype_output'))
                if not out_dir.exists():
                    out_dir.mkdir(parents=True)
            else:
                # Default output directory taken from first input file
                out_dir = Path(fsmap.root).parent.absolute()

                # Check permission, raise exception if no permission
                io.check_file_permissions(out_dir)

            warnings.warn(f"Resulting converted file(s) will be available at {str(out_dir)}")
            out_path = [str(out_dir.joinpath(Path(os.path.splitext(Path(f).name)[0] + file_format)))
                        for f in self.source_file]

        else:
            fsmap = fsspec.get_mapper(save_path, **self._output_storage_options)
            fs = fsmap.fs
            root = fsmap.root
            fname, ext = os.path.splitext(root)
            if ext == '':  # directory
                out_dir = fname
                out_path = [root + '/' + os.path.splitext(os.path.basename(f))[0] + file_format
                            for f in self.source_file]
            else:  # file
                out_dir = os.path.dirname(root)
                if len(self.source_file) > 1:  # get dirname and assemble path
                    out_path = [out_dir + '/' + os.path.splitext(os.path.basename(f))[0] + file_format
                                for f in self.source_file]
                else:
                    # force file_format to conform
                    out_path = [os.path.join(out_dir, fname + file_format)]

        # Create folder if save_path does not exist already
        fsmap = fsspec.get_mapper(str(out_dir), **self._output_storage_options)
        if file_format == '.nc' and not isinstance(fs, LocalFileSystem):
            raise ValueError("Only local filesystem allowed for NetCDF output.")
        else:
            try:
                # Check permission, raise exception if no permission
                io.check_file_permissions(fsmap)
                fs.mkdir(fsmap.root)
            except FileNotFoundError:
                raise ValueError("Specified save_path is not valid.")

        # Store output path
        self.output_file = out_path  # output_path is always a list

    def _convert_indiv_file(self, file, output_path=None, engine=None):
        """Convert a single file.
        """

        if self.sonar_model not in MODELS:
            raise ValueError(f"Unsupported sonar model: {self.sonar_model}\n"
                             f"Must be one of: {list(MODELS)}")

        # Use echosounder-specific object
        c = MODELS[self.sonar_model]['parser']
        sg = MODELS[self.sonar_model]['set_groups']

        # TODO: the if-else below only works for the AZFP vs EK contrast,
        #  but is brittle since it is abusing params by using it implicitly
        if MODELS[self.sonar_model]['xml']:
            params = self.xml_path
        else:
            params = self.data_type

        # Actually parsing and saving file
        c = c(file, params=params, storage_options=self.storage_options)
        c.parse_raw()
        sg = sg(c, input_file=file, output_path=output_path, engine=engine, compress=self.compress,
                overwrite=self.overwrite, params=self._conversion_params, sonar_model=self.sonar_model)
        sg.save()

    def combine_files(self, indiv_files=None, save_path=None, remove_indiv=True):
        """Combine output files when self.combine=True.

        `combine_files` can be called to combine files that have just be converted
        by the current instance of Convert (those listed in self.output_path)
        or files that were previously converted (specified in `indiv_files`).

        Parameters
        ----------
        indiv_files : list
            List of NetCDF or Zarr files to combine
        save_path : str
            Either a directory or a file. If none, use the name of the first ``src_file``
        remove_indiv : bool
            Whether or not to remove the files in ``indiv_files``
            Defaults to ``True``

        Returns
        -------
        True or False depending on whether or not the combination was successful
        """
        # self.output_path contains individual files to be combined if
        #  they have just been converted using this object
        indiv_files = self.output_file if indiv_files is None else indiv_files

        if isinstance(indiv_files, str):
            indiv_files = [indiv_files]
        if len(indiv_files) < 2:
            print("Combination did not occur as there is only one source file.")
            return True

        # Construct the final combined save path
        if save_path is not None:
            # TODO: we need to check validity/permission of the user-specified save_path here
            combined_save_path = save_path
        else:
            combined_save_path = combine_fcn.get_combined_save_path(indiv_files)

        # Get the correct xarray functions for opening datasets
        engine = io.get_file_format(combined_save_path)
        combine_fcn.perform_combination(self.sonar_model, indiv_files, combined_save_path, engine)

        # Update output_path to be the combined path name
        self.output_file = combined_save_path

        # Delete individual files after combining
        if remove_indiv:
            for f in indiv_files:
                combine_fcn.remove_indiv_files(f)

    def update_platform(self, files=None, extra_platform_data=None):
        """
        Parameters
        ----------
        files : str / list
            path of converted .nc/.zarr files
        extra_platform_data : xarray dataset
            dataset containing platform information along a 'time' dimension
        """
        # self.extra_platform data passed into to_netcdf or from another function
        if extra_platform_data is None:
            return
        files = self.output_file if files is None else files
        if not isinstance(files, list):
            files = [files]
        engine = io.get_file_format(files[0])

        # saildrone specific hack
        if "trajectory" in extra_platform_data:
            extra_platform_data = extra_platform_data.isel(trajectory=0).drop("trajectory")
            extra_platform_data = extra_platform_data.swap_dims({'obs': 'time'})

        # Try to find the time dimension in the extra_platform_data
        possible_time_keys = ['time', 'ping_time', 'location_time']
        time_name = ''
        for k in possible_time_keys:
            if k in extra_platform_data:
                time_name = k
                break
        if not time_name:
            raise ValueError('Time dimension not found')

        for f in files:
            ds_beam = xr.open_dataset(f, group="Beam", engine=engine)
            ds_platform = xr.open_dataset(f, group="Platform", engine=engine)

            # only take data during ping times
            # start_time, end_time = min(ds_beam["ping_time"]), max(ds_beam["ping_time"])
            start_time, end_time = ds_beam.ping_time.min(), ds_beam.ping_time.max()

            extra_platform_data = extra_platform_data.sel({time_name: slice(start_time, end_time)})

            def mapping_get_multiple(mapping, keys, default=None):
                for key in keys:
                    if key in mapping:
                        return mapping[key]
                return default

            if self.sonar_model in ['EK80', 'EA640']:
                ds_platform = ds_platform.reindex({
                    "mru_time": extra_platform_data[time_name].values,
                    "location_time": extra_platform_data[time_name].values,
                })
                # merge extra platform data
                num_obs = len(extra_platform_data[time_name])
                ds_platform = ds_platform.update({
                    "pitch": ("mru_time", mapping_get_multiple(
                        extra_platform_data, ["pitch", "PITCH"], np.full(num_obs, np.nan))),
                    "roll": ("mru_time", mapping_get_multiple(
                        extra_platform_data, ["roll", "ROLL"], np.full(num_obs, np.nan))),
                    "heave": ("mru_time", mapping_get_multiple(
                        extra_platform_data, ["heave", "HEAVE"], np.full(num_obs, np.nan))),
                    "latitude": ("location_time", mapping_get_multiple(
                        extra_platform_data, ["lat", "latitude", "LATITUDE"], default=np.full(num_obs, np.nan))),
                    "longitude": ("location_time", mapping_get_multiple(
                        extra_platform_data, ["lon", "longitude", "LONGITUDE"], default=np.full(num_obs, np.nan))),
                    "water_level": ("location_time", mapping_get_multiple(
                        extra_platform_data, ["water_level", "WATER_LEVEL"], np.ones(num_obs)))
                })
            elif self.sonar_model == 'EK60':
                ds_platform = ds_platform.reindex({
                    "ping_time": extra_platform_data[time_name].values,
                    "location_time": extra_platform_data[time_name].values,
                })
                # merge extra platform data
                num_obs = len(extra_platform_data[time_name])
                ds_platform = ds_platform.update({
                    "pitch": ("ping_time", mapping_get_multiple(
                        extra_platform_data, ["pitch", "PITCH"], np.full(num_obs, np.nan))),
                    "roll": ("ping_time", mapping_get_multiple(
                        extra_platform_data, ["roll", "ROLL"], np.full(num_obs, np.nan))),
                    "heave": ("ping_time", mapping_get_multiple(
                        extra_platform_data, ["heave", "HEAVE"], np.full(num_obs, np.nan))),
                    "latitude": ("location_time", mapping_get_multiple(
                        extra_platform_data, ["lat", "latitude", "LATITUDE"], default=np.full(num_obs, np.nan))),
                    "longitude": ("location_time", mapping_get_multiple(
                        extra_platform_data, ["lon", "longitude", "LONGITUDE"], default=np.full(num_obs, np.nan))),
                    "water_level": ("location_time", mapping_get_multiple(
                        extra_platform_data, ["water_level", "WATER_LEVEL"], np.ones(num_obs)))
                })

            # need to close the file in order to remove it
            # (and need to close the file so to_netcdf can write to it)
            ds_platform.close()
            ds_beam.close()

            if engine == "netcdf4":
                # https://github.com/Unidata/netcdf4-python/issues/65
                # Copy groups over to temporary file
                # TODO: Add in documentation: recommended to use Zarr if using add_platform
                new_dataset_filename = f + ".temp"
                groups = ['Provenance', 'Environment', 'Beam', 'Sonar', 'Vendor']
                with xr.open_dataset(f) as ds_top:
                    ds_top.to_netcdf(new_dataset_filename, mode='w')
                for group in groups:
                    with xr.open_dataset(f, group=group) as ds:
                        ds.to_netcdf(new_dataset_filename, mode='a', group=group)
                ds_platform.to_netcdf(new_dataset_filename, mode="a", group="Platform")
                # Replace original file with temporary file
                os.remove(f)
                os.rename(new_dataset_filename, f)
            elif engine == "zarr":
                # https://github.com/zarr-developers/zarr-python/issues/65
                old_dataset = zarr.open_group(f, mode="a")
                del old_dataset["Platform"]
                ds_platform.to_zarr(f, mode="a", group="Platform")

    def _to_file(self, convert_type, save_path=None, data_type='ALL', compress=True, combine=False,
                 overwrite=False, parallel=False, extra_platform_data=None, storage_options={}, **kwargs):
        """Convert a file or a list of files to netCDF or zarr.

        Parameters
        ----------
        save_path : str
            path that converted .nc file will be saved
        convert_type : str
            type of converted file, '.nc' or '.zarr'
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
        extra_platform_data : Dataset
            The dataset containing the platform information to be added to the output
        storage_options : dict
            Additional keywords to pass to the filesystem class.
        """
        self.data_type = data_type
        self.compress = compress
        self.combine = combine
        self.overwrite = overwrite

        # TODO: probably can combine both the remote and local cases below once
        #  _validate_path and _validate_object_store are combined
        # Assemble output file names and path
        if convert_type == 'netcdf4':
            self._validate_path('.nc', save_path)
        elif convert_type == 'zarr':
            if isinstance(save_path, MutableMapping):
                self._validate_object_store(save_path)
            else:
                self._validate_path('.zarr', save_path)
        else:
            raise ValueError('Unknown type to convert file to!')

        # Get all existing files
        exist_list = []
        fs = fsspec.get_mapper(self.output_file[0]).fs  # get file system
        for out_f in self.output_file:
            if isinstance(out_f, MutableMapping):  # output is remote
                if fs.exists(out_f.root):
                    exist_list.append(out_f)
            else:  # output is local
                if fs.exists(out_f):
                    exist_list.append(out_f)

        # Sequential or parallel conversion
        if not parallel:
            for src_f, out_f in zip(self.source_file, self.output_file):
                if out_f in exist_list and not self.overwrite:
                    print(f"{dt.now().strftime('%H:%M:%S')}  {src_f} has already been converted to {convert_type}. "
                          f"Conversion not executed.")
                    continue
                else:
                    if out_f in exist_list:
                        print(f"{dt.now().strftime('%H:%M:%S')}  overwriting {out_f}")
                    else:
                        print(f"{dt.now().strftime('%H:%M:%S')}  converting {out_f}")
                    self._convert_indiv_file(file=src_f, output_path=out_f, engine=convert_type)
        else:
            # TODO: add parallel conversion
            print('Parallel conversion is not yet implemented. Use parallel=False.')
            return

        # Combine files if needed
        if self.combine:
            self.combine_files(save_path=save_path, remove_indiv=True)

        # Attached platform data
        if extra_platform_data is not None:
            self.update_platform(files=self.output_file, extra_platform_data=extra_platform_data)

        # If only one output file make it a string instead of a list
        if len(self.output_file) == 1:
            self.output_file = self.output_file[0]

    def to_netcdf(self, **kwargs):
        """Convert a file or a list of files to netCDF.

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
        extra_platform_data : Dataset
            The dataset containing the platform information to be added to the output
        storage_options : dict
            Additional keywords to pass to the filesystem class.
        """
        return self._to_file('netcdf4', **kwargs)

    def to_zarr(self, **kwargs):
        """Convert a file or a list of files to zarr.

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
        extra_platform_data : Dataset
            The dataset containing the platform information to be added to the output
        storage_options : dict
            Additional keywords to pass to the filesystem class.
        """
        return self._to_file('zarr', **kwargs)

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
            raise ValueError(f"data_type must be either 'CONFIG' or 'ENV'")

        # Get export path
        # TODO: this currently only works for local path, need to add the cloud part
        self._validate_path('.xml', save_path)

        # Parse files
        for idx, file in enumerate(self.source_file):
            # convert file one by one into path set by validate_path()
            tmp = ParseEK80(file, params=[data_type, 'EXPORT_XML'])
            tmp.parse_raw()
            with open(self.output_file[idx], 'w') as xml_file:
                # Select between data types
                if data_type == 'CONFIG':
                    data = tmp.config_datagram['xml']
                elif data_type == 'ENV':
                    data = tmp.environment['xml']
                else:
                    raise ValueError("Unknown data type", data_type)
                xml_file.write(data)

        # If only one output file make it a string instead of a list
        if len(self.output_file) == 1:
            self.output_file = self.output_file[0]

    @staticmethod
    def check_files(file, model, xml_path=None, storage_options={}):

        if MODELS[model]["xml"]:  # if this sonar model expects an XML file
            if not xml_path:
                raise ValueError(f"XML file is required for {model} raw data")
            elif ".XML" not in os.path.splitext(xml_path)[1].upper():
                raise ValueError(f"{os.path.basename(xml_path)} is not an XML file")

            xmlmap = fsspec.get_mapper(xml_path, **storage_options)
            if not xmlmap.fs.exists(xmlmap.root):
                raise FileNotFoundError(f"There is no file named {os.path.basename(xml_path)}")

            xml = xml_path
        else:
            xml = ''

        # TODO: https://github.com/OSOceanAcoustics/echopype/issues/229
        #  to add compatibility for pathlib.Path objects for local paths
        for f in file:
            fsmap = fsspec.get_mapper(f, **storage_options)
            ext = MODELS[model]["ext"]
            if not fsmap.fs.exists(fsmap.root):
                raise FileNotFoundError(f"There is no file named {os.path.basename(f)}")

            if os.path.splitext(f)[1] != ext:
                raise ValueError(f"Not all files are in the same format. Expecting a {ext} file but got {f}")

        return file, xml

    # TODO: Remove below in future versions. They are for supporting old API calls.
    @property
    def nc_path(self):
        warnings.warn("`nc_path` is deprecated, Use `output_path` instead.", DeprecationWarning, 2)
        path = self._nc_path if self._nc_path is not None else self.output_file
        return path

    @property
    def zarr_path(self):
        warnings.warn("`zarr_path` is deprecated, Use `output_path` instead.", DeprecationWarning, 2)
        path = self._zarr_path if self._zarr_path is not None else self.output_file
        return path

    def raw2nc(self, save_path=None, combine_opt=False, overwrite=False, compress=True):
        warnings.warn("`raw2nc` is deprecated, use `to_netcdf` instead.", DeprecationWarning, 2)
        self.to_netcdf(save_path=save_path, compress=compress, combine=combine_opt,
                       overwrite=overwrite)
        self._nc_path = self.output_file

    def raw2zarr(self, save_path=None, combine_opt=False, overwrite=False, compress=True):
        warnings.warn("`raw2zarr` is deprecated, use `to_zarr` instead.", DeprecationWarning, 2)
        self.to_zarr(save_path=save_path, compress=compress, combine=combine_opt,
                     overwrite=overwrite)
        self._zarr_path = self.output_file
