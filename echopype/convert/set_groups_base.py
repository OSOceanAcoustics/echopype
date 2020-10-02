import os
from datetime import datetime as dt
import xarray as xr
import numpy as np
import zarr
import netCDF4
from .._version import get_versions
ECHOPYPE_VERSION = get_versions()['version']
del get_versions

NETCDF_COMPRESSION_SETTINGS = {'zlib': True, 'complevel': 4}
ZARR_COMPRESSION_SETTINGS = {'compressor': zarr.Blosc(cname='zstd', clevel=3, shuffle=2)}


class SetGroupsBase:
    """Base class for saving groups to netcdf or zarr from echosounder data files.
    """
    def __init__(self, convert_obj, input_file, output_path, sonar_model=None,
                 save_ext='.nc', compress=True, overwrite=True, params=None, extra_files=None):
        self.convert_obj = convert_obj   # a convert object ParseEK60/ParseAZFP/etc...
        self.sonar_model = sonar_model   # Used for when a sonar that is not AZFP/EK60/EK80 can still be saved
        self.input_file = input_file
        self.output_path = output_path
        self.save_ext = save_ext
        self.compress = compress
        self.overwrite = overwrite
        self.ui_param = params
        self.extra_files = extra_files
        self.NETCDF_COMPRESSION_SETTINGS = NETCDF_COMPRESSION_SETTINGS
        self.ZARR_COMPRESSION_SETTINGS = ZARR_COMPRESSION_SETTINGS

    def save(self):
        """Actually save groups to file by calling the set methods.
        """

    def set_toplevel(self, sonar_model):
        """Set the top-level group.
        """
        # Capture datetime from filename
        # timestamp_pattern = re.compile(self.convert_obj.timestamp_pattern)
        # raw_date_time = timestamp_pattern.match(os.path.basename(self.input_file))
        # filedate = raw_date_time['date']
        # filetime = raw_date_time['time']
        # date_created = dt.strptime(filedate + '-' + filetime, '%Y%m%d-%H%M%S').isoformat() + 'Z'

        # Collect variables
        date_created = self.convert_obj.ping_time[0]

        tl_dict = {'conventions': 'CF-1.7, SONAR-netCDF4-1.0, ACDD-1.3',
                   'keywords': sonar_model,
                   'sonar_convention_authority': 'ICES',
                   'sonar_convention_name': 'SONAR-netCDF4',
                   'sonar_convention_version': '1.0',
                   'summary': '',
                   'title': '',
                   'date_created': np.datetime_as_string(date_created, 's') + 'Z',
                   'survey_name': self.ui_param['survey_name']}
        # Add any extra user defined values
        for k, v in list(self.ui_param.items())[5:]:
            tl_dict[k] = v

        # Save
        if self.save_ext == '.nc':
            with netCDF4.Dataset(self.output_path, "w", format="NETCDF4") as ncfile:
                [ncfile.setncattr(k, v) for k, v in tl_dict.items()]
        elif self.save_ext == '.zarr':
            zarrfile = zarr.open(self.output_path, mode="w")
            for k, v in tl_dict.items():
                zarrfile.attrs[k] = v
        else:
            raise ValueError("Unsupported file format")

    def set_provenance(self):
        """Set the Provenance group.
        """
        # Collect variables
        prov_dict = {'conversion_software_name': 'echopype',
                     'conversion_software_version': ECHOPYPE_VERSION,
                     'conversion_time': dt.utcnow().isoformat(timespec='seconds') + 'Z',    # use UTC time
                     'src_filenames': self.input_file}
        # Save
        if self.save_ext == '.nc':
            with netCDF4.Dataset(self.output_path, "a", format="NETCDF4") as ncfile:
                prov = ncfile.createGroup("Provenance")
                [prov.setncattr(k, v) for k, v in prov_dict.items()]
        elif self.save_ext == '.zarr':
            zarr_file = zarr.open(self.output_path, mode="a")
            prov = zarr_file.create_group('Provenance')
            for k, v in prov_dict.items():
                prov.attrs[k] = v
        else:
            raise ValueError("Unsupported file format")

    def set_sonar(self, sonar_vals):
        """Set the Sonar group.
        """
        # Collect variables
        sonar_dict = dict(zip(('sonar_manufacturer', 'sonar_model', 'sonar_serial_number',
                               'sonar_software_name', 'sonar_software_version', 'sonar_type'), sonar_vals))

        # Save variables
        if self.save_ext == '.nc':
            with netCDF4.Dataset(self.output_path, "a", format="NETCDF4") as ncfile:
                snr = ncfile.createGroup("Sonar")
                # set group attributes
                [snr.setncattr(k, v) for k, v in sonar_dict.items()]

        elif self.save_ext == '.zarr':
            zarrfile = zarr.open(self.output_path, mode='a')
            snr = zarrfile.create_group('Sonar')

            for k, v in sonar_dict.items():
                snr.attrs[k] = v

    def set_nmea(self):
        """Set the Platform/NMEA group.
        """

        # Convert np.datetime64 numbers to seconds since 1900-01-01
        # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
        time = (self.convert_obj.nmea_data.nmea_times -
                np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')
        ds = xr.Dataset(
            {'NMEA_datagram': (['time'], self.convert_obj.nmea_data.raw_datagrams,
                               {'long_name': 'NMEA datagram'})
             },
            coords={'time': (['time'], time,
                             {'axis': 'T',
                              'calendar': 'gregorian',
                              'long_name': 'Timestamps for NMEA datagrams',
                              'standard_name': 'time',
                              'units': 'seconds since 1900-01-01'})},
            attrs={'description': 'All NMEA sensor datagrams'})

        # save to file
        if self.save_ext == '.nc':
            nc_encoding = {'time': NETCDF_COMPRESSION_SETTINGS} if self.compress else {}
            ds.to_netcdf(path=self.output_path, mode='a', group='Platform/NMEA', encoding=nc_encoding)
        elif self.save_ext == '.zarr':
            zarr_encoding = {'time': ZARR_COMPRESSION_SETTINGS} if self.compress else {}
            ds.to_zarr(store=self.output_path, mode='a', group='Platform/NMEA', encoding=zarr_encoding)
