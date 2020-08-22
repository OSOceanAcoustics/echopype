"""
UI class for converting raw data from different echosounders to netcdf or zarr.
"""

from .convertbase_new import ParseEK60
from .utils.setgroups_new import SetGroupsEK60


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
        self.sonar_model = None    # type of echosounder
        self.xml_file = ''         # path to xml file (AZFP only)
                                   # users will get an error if try to set this directly for EK60 or EK80 data
        self.source_file = None    # input file path or list of input file paths
        self.output_file = None    # converted file path or list of converted file paths
        self._source_path = None   # for convenience only, the path is included in source_file already;
                                   # user should not interact with this directly
        self._output_path = None   # for convenience only, the path is included in source_file already;
                                   # user should not interact with this directly
        self._conversion_params = {}   # a dictionary of conversion parameters,
                                       # the keys could be different for different echosounders.
                                       # This dictionary is set by the `set_param` method.
        self.data_type = 'all'  # type of data to be converted into netcdf or zarr.
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

    def set_source(self, file, model):
        """Set source file and echosounder model.
        """

    def set_param(self, param_dict):
        """Allow users to set specific parameters in the conversion.
        """

    def _validate_path(self):
        """Assemble output file names and path.

        The file names and path will depend on inputs to to_netcdf() and to_zarr().
        """
        # TODO: this was previously in ConvertBase

    def _convert_indiv_file(self, file, path, output_format):
        """Convert a single file.
        """
        # if converting EK60 files:
        c = ParseEK60(file)  # use echosounder-specific object
        c.parse_raw()   # pass data_type to convert here, for EK60 and EK80 only
        sg = SetGroupsEK60(c, output_file=file, output_path=path, output_format='netcdf',
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

    def combine_files(self):
        """Combine output files when self.combine=True.
        """
        if self._check_param_consistency():
            # code to actually combine files
            print('combine files...')
        else:
            print('cannot combine files...')

    def to_netcdf(self, save_path, data_type='all', compress=True, combine=False, parallel=False):
        """Convert a file or a list of files to netcdf format.
        """
        self.data_type = data_type
        self.compress = compress
        self.combine = combine

        # Sequential or parallel conversion
        if not parallel:
            for file in self.source_file:
                # convert file one by one into path set by validate_path()
                self._convert_indiv_file(file=file, path=save_path, output_format='netcdf')
        # else:
            # use dask syntax but we'll probably use something else, like multiprocessing?
            # delayed(self._convert_indiv_file(file=file, path=save_path, output_format='netcdf'))

        # combine files if needed
        if self.combine:
            self.combine_files()

    def to_zarr(self, save_path, data_type='all', compress=True, combine=False):
        """Convert a file or a list of files to zarr format.
    """


