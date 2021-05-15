import pynmea2
from datetime import datetime as dt
import xarray as xr
import numpy as np
import zarr
from xarray import coding
from _echopype_version import version as ECHOPYPE_VERSION

COMPRESSION_SETTINGS = {
    'netcdf4': {'zlib': True, 'complevel': 4},
    'zarr': {'compressor': zarr.Blosc(cname='zstd', clevel=3, shuffle=2)}
}

DEFAULT_CHUNK_SIZE = {
    'range_bin': 25000,
    'ping_time': 2500
}

DEFAULT_ENCODINGS = {
    'ping_time': {
        'units': 'seconds since 1900-01-01T00:00:00+00:00',
        'calendar': 'gregorian',
        '_FillValue': np.nan,
        'dtype': np.dtype('float64'),
    },
    'mru_time': {
        'units': 'seconds since 1900-01-01T00:00:00+00:00',
        'calendar': 'gregorian',
        '_FillValue': np.nan,
        'dtype': np.dtype('float64'),
    },
    'location_time': {
        'units': 'seconds since 1900-01-01T00:00:00+00:00',
        'calendar': 'gregorian',
        '_FillValue': np.nan,
        'dtype': np.dtype('float64'),
    }
}


def _encode_dataarray(da, dtype):
    """Encodes and decode datetime64 array similar to writing to file"""
    read_encoding = {
        "units": "seconds since 1900-01-01T00:00:00+00:00",
        "calendar": "gregorian",
    }

    if dtype in [np.float64, np.int64]:
        encoded_data = da
    else:
        encoded_data, _, _ = coding.times.encode_cf_datetime(
            da, **read_encoding
        )
    return coding.times.decode_cf_datetime(
        encoded_data, **read_encoding
    )


def set_encodings(ds: xr.Dataset) -> xr.Dataset:
    """
    Set the default encoding for variables.
    """
    new_ds = ds.copy(deep=True)
    for var, encoding in DEFAULT_ENCODINGS.items():
        if var in new_ds:
            da = new_ds[var].copy()
            if "_time" in var:
                new_ds[var] = xr.apply_ufunc(
                    _encode_dataarray,
                    da,
                    keep_attrs=True,
                    kwargs={
                        'dtype': da.dtype
                    }
                )

            new_ds[var].encoding = encoding

    return new_ds


class SetGroupsBase:
    """Base class for saving groups to netcdf or zarr from echosounder data files.
    """
    def __init__(self, parser_obj, input_file, output_path, sonar_model=None,
                 engine='zarr', compress=True, overwrite=True, params=None):
        self.parser_obj = parser_obj   # parser object ParseEK60/ParseAZFP/etc...
        self.sonar_model = sonar_model   # Used for when a sonar that is not AZFP/EK60/EK80 can still be saved
        self.input_file = input_file
        self.output_path = output_path
        self.engine = engine
        self.compress = compress
        self.overwrite = overwrite
        self.ui_param = params

        if not self.compress:
            self.compression_settings = None
        else:
            self.compression_settings = COMPRESSION_SETTINGS[self.engine]

    # TODO: change the set_XXX methods to return a dataset to be saved in the overarching save method
    def set_toplevel(self, sonar_model, date_created=None) -> xr.Dataset:
        """Set the top-level group.
        """
        # Collect variables
        tl_dict = {'conventions': 'CF-1.7, SONAR-netCDF4-1.0, ACDD-1.3',
                   'keywords': sonar_model,
                   'sonar_convention_authority': 'ICES',
                   'sonar_convention_name': 'SONAR-netCDF4',
                   'sonar_convention_version': '1.0',
                   'summary': '',
                   'title': '',
                   'date_created': np.datetime_as_string(date_created, 's') + 'Z',
                   'survey_name': self.ui_param['survey_name']}
        # Save
        ds = xr.Dataset()
        ds = ds.assign_attrs(tl_dict)
        return ds

    def set_provenance(self) -> xr.Dataset:
        """Set the Provenance group.
        """
        # Collect variables
        prov_dict = {'conversion_software_name': 'echopype',
                     'conversion_software_version': ECHOPYPE_VERSION,
                     'conversion_time': dt.utcnow().isoformat(timespec='seconds') + 'Z',    # use UTC time
                     'src_filenames': self.input_file}
        # Save
        ds = xr.Dataset()
        ds = ds.assign_attrs(prov_dict)
        return ds

    def set_env(self) -> xr.Dataset:
        """Set the Environment group.
        """
        pass

    def set_sonar(self) -> xr.Dataset:
        """Set the Sonar group.
        """
        pass

    def set_beam(self) -> xr.Dataset:
        """Set the Beam group.
        """
        pass

    def set_beam_complex(self) -> xr.Dataset:
        """Set the Beam_Complex group
        """
        pass

    def set_platform(self) -> xr.Dataset:
        """Set the Platform group.
        """
        pass

    def set_nmea(self) -> xr.Dataset:
        """Set the Platform/NMEA group.
        """
        # Save nan if nmea data is not encoded in the raw file
        if len(self.parser_obj.nmea['nmea_string']) != 0:
            # Convert np.datetime64 numbers to seconds since 1900-01-01 00:00:00Z
            # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
            time = (self.parser_obj.nmea['timestamp'] -
                    np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')
            raw_nmea = self.parser_obj.nmea['nmea_string']
        else:
            time = [np.nan]
            raw_nmea = [np.nan]

        ds = xr.Dataset(
            {'NMEA_datagram': (['location_time'], raw_nmea,
                               {'long_name': 'NMEA datagram'})
             },
            coords={'location_time': (['location_time'], time,
                                      {'axis': 'T',
                                       'calendar': 'gregorian',
                                       'long_name': 'Timestamps for NMEA datagrams',
                                       'standard_name': 'time',
                                       'units': 'seconds since 1900-01-01 00:00:00Z'})},
            attrs={'description': 'All NMEA sensor datagrams'})

        return ds

    def set_vendor(self) -> xr.Dataset:
        """Set the Vendor group.
        """
        pass

    # TODO: move this to be part of parser as it is not a "set" operation
    def _parse_NMEA(self):
        """Get the lat and lon values from the raw nmea data"""
        messages = [string[3:6] for string in self.parser_obj.nmea['nmea_string']]
        idx_loc = np.argwhere(np.isin(messages,
                                      self.ui_param['nmea_gps_sentence'])).squeeze()
        if idx_loc.size == 1:  # in case of only 1 matching message
            idx_loc = np.expand_dims(idx_loc, axis=0)
        nmea_msg = [pynmea2.parse(self.parser_obj.nmea['nmea_string'][x]) for x in idx_loc]
        lat = np.array([x.latitude if hasattr(x, 'latitude') else np.nan
                        for x in nmea_msg]) if nmea_msg else [np.nan]
        lon = np.array([x.longitude if hasattr(x, 'longitude') else np.nan
                        for x in nmea_msg]) if nmea_msg else [np.nan]
        msg_type = np.array([x.sentence_type if hasattr(x, 'sentence_type') else np.nan
                             for x in nmea_msg]) if nmea_msg else [np.nan]
        location_time = (np.array(self.parser_obj.nmea['timestamp'])[idx_loc] -
                         np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's') if nmea_msg else [np.nan]

        return location_time, msg_type, lat, lon
