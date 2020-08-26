"""
Class to save unpacked echosounder data to appropriate groups in netcdf or zarr.
"""
import os
import shutil
from datetime import datetime as dt
import xarray as xr
import numpy as np
import zarr
import netCDF4
from ..._version import get_versions
ECHOPYPE_VERSION = get_versions()['version']
del get_versions


class SetGroupsBase:
    """Base class for saving groups to netcdf or zarr from echosounder data files.
    """
    def __init__(self, convert_obj, input_file, output_path, save_ext='.nc', compress=True, overwrite=True):
        self.convert_obj = convert_obj   # a convert object ConvertEK60/ConvertAZFP/etc...
        self.input_file = input_file
        self.output_path = output_path
        self.save_ext = save_ext
        self.compress = compress
        self.overwrite = overwrite

    def save(self):
        """Actually save groups to file by calling the set methods.
        """

    def set_toplevel(self, sonar_model):
        """Set the top-level group.
        """
        # Collect variables
        tl_dict = {'conventions': 'CF-1.7, SONAR-netCDF4-1.0, ACDD-1.3',
                   'keywords': sonar_model,
                   'sonar_convention_authority': 'ICES',
                   'sonar_convention_name': 'SONAR-netCDF4',
                   'sonar_convention_version': '1.0',
                   'summary': '',
                   'title': ''}
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
                for k, v in prov_dict.items():
                    prov.setncattr(k, v)
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
    @staticmethod
    def _remove(path):
        fname, ext = os.path.splitext(path)
        if ext == '.zarr':
            shutil.rmtree(path)
        else:
            os.remove(path)


class SetGroupsEK60(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from EK60 data files.
    """
    def save(self):
        """Actually save groups to file by calling the set methods.
        """

    def set_env(self):
        """Set the Environment group.
        """

    def set_platform(self):
        """Set the Platform group.
        """

    def set_beam(self):
        """Set the Beam group.
        """


class SetGroupsEK80(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from EK80 data files.
    """
    def save(self):
        """Actually save groups to file by calling the set methods.
        """

    def set_env(self):
        """Set the Environment group.
        """

    def set_platform(self):
        """Set the Platform group.
        """
        # TODO: let's add ConvertEK80.environment['drop_keel_offset'] to
        #  the Platform group even though it is not in the convention.

    def set_beam(self):
        """Set the Beam group.
        """

    def set_vendor(self):
        """Set the Vendor-specific group.
        """


class SetGroupsAZFP(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from AZFP data files.
    """
    def save(self):
        """Actually save groups to file by calling the set methods.
        """
        # Check if file exists
        if os.path.exists(self.output_path) and self.overwrite:
            # Remove the file if self.overwrite is true
            print("          overwriting: " + self.output_path)
            self._remove(self.output_path)
        if os.path.exists(self.output_path):
            # Otherwise, skip saving
            print(f'          ... this file has already been converted to {self.save_ext}, conversion not executed.')
        else:
            ping_time = self.convert_obj._get_ping_time()
            sonar_values = ('ASL Environmental Sciences', 'Acoustic Zooplankton Fish Profiler',
                            int(self.convert_obj.unpacked_data['serial_number']),
                            'Based on AZFP Matlab Toolbox', '1.4', 'echosounder')
            self.set_toplevel("AZFP")
            self.set_env(ping_time)
            self.set_provenance()
            self.set_platform()
            self.set_sonar(sonar_values)
            self.set_beam(ping_time)
            self.set_vendor(ping_time)

    def set_env(self, ping_time):
        """Set the Environment group.
        """
        # Only save environment group if file_path exists
        if not os.path.exists(self.output_path):
            print('netCDF file does not exist, exiting without saving Environment group...')
        else:
            ds = xr.Dataset({'temperature': (['ping_time'], self.convert_obj.unpacked_data['temperature'])},
                            coords={'ping_time': (['ping_time'], ping_time,
                                    {'axis': 'T',
                                     'calendar': 'gregorian',
                                     'long_name': 'Timestamp of each ping',
                                     'standard_name': 'time',
                                     'units': 'seconds since 1970-01-01'})},
                            attrs={'long_name': "Water temperature",
                                   'units': "C"})

            # save to file
            if self.save_ext == '.nc':
                ds.to_netcdf(path=self.output_path, mode='a', group='Environment')
            elif self.save_ext == '.zarr':
                ds.to_zarr(store=self.output_path, mode='a', group='Environment')

    def set_platform(self):
        """Set the Platform group.
        """
        platform_dict = {'platform_name': self.convert_obj.ui_param['platform_name'],
                         'platform_type': self.convert_obj.ui_param['platform_type'],
                         'platform_code_ICES': self.convert_obj.ui_param['platform_code_ICES']}
        # Only save platform group if file_path exists
        if not os.path.exists(self.output_path):
            print("netCDF file does not exist, exiting without saving Platform group...")
        else:
            if self.save_ext == '.nc':
                with netCDF4.Dataset(self.output_path, 'a', format='NETCDF4') as ncfile:
                    plat = ncfile.createGroup('Platform')
                    [plat.setncattr(k, v) for k, v in platform_dict.items()]
            elif self.save_ext == '.zarr':
                zarrfile = zarr.open(self.output_path, mode='a')
                plat = zarrfile.create_group('Platform')
                for k, v in platform_dict.items():
                    plat.attrs[k] = v

    def set_beam(self, ping_time):
        """Set the Beam group.
        """
        unpacked_data = self.convert_obj.unpacked_data
        parameters = self.convert_obj.parameters
        anc = np.array(unpacked_data['ancillary'])   # convert to np array for easy slicing
        dig_rate = unpacked_data['dig_rate']         # dim: freq
        freq = np.array(unpacked_data['frequency']) * 1000    # Frequency in Hz

        # Build variables in the output xarray Dataset
        N = []   # for storing backscatter_r values for each frequency
        Sv_offset = np.zeros(freq.shape)
        for ich in range(len(freq)):
            Sv_offset[ich] = self.convert_obj._calc_Sv_offset(freq[ich], unpacked_data['pulse_length'][ich])
            N.append(np.array([unpacked_data['counts'][p][ich]
                               for p in range(len(unpacked_data['year']))]))

        tdn = unpacked_data['pulse_length'] / 1e6  # Convert microseconds to seconds
        range_samples_xml = np.array(parameters['range_samples'])         # from xml file
        range_samples_per_bin = unpacked_data['range_samples_per_bin']    # from data header

        # Calculate sample interval in seconds
        if len(dig_rate) == len(range_samples_per_bin):
            sample_int = range_samples_per_bin / dig_rate
        else:
            raise ValueError("dig_rate and range_samples not unique across frequencies")

        # Largest number of counts along the range dimension among the different channels
        longest_range_bin = np.max(unpacked_data['num_bins'])
        range_bin = np.arange(longest_range_bin)
        # TODO: replace the following with an explicit check of length of range across channels
        try:
            np.array(N)
        # Exception occurs when N is not rectangular,
        #  so it must be padded with nan values to make it rectangular
        except ValueError:
            N = [np.pad(n, ((0, 0), (0, longest_range_bin - n.shape[1])),
                 mode='constant', constant_values=np.nan)
                 for n in N]

        ds = xr.Dataset({'backscatter_r': (['frequency', 'ping_time', 'range_bin'], N),
                         'equivalent_beam_angle': (['frequency'], parameters['BP']),
                         'gain_correction': (['frequency'], parameters['gain']),
                         'sample_interval': (['frequency'], sample_int,
                                             {'units': 's'}),
                         'transmit_duration_nominal': (['frequency'], tdn,
                                                       {'long_name': 'Nominal bandwidth of transmitted pulse',
                                                        'units': 's',
                                                        'valid_min': 0.0}),
                         'temperature_counts': (['ping_time'], anc[:, 4]),
                         'tilt_x_count': (['ping_time'], anc[:, 0]),
                         'tilt_y_count': (['ping_time'], anc[:, 1]),
                         'tilt_x': (['ping_time'], unpacked_data['tilt_x']),
                         'tilt_y': (['ping_time'], unpacked_data['tilt_y']),
                         'cos_tilt_mag': (['ping_time'], unpacked_data['cos_tilt_mag']),
                         'DS': (['frequency'], parameters['DS']),
                         'EL': (['frequency'], parameters['EL']),
                         'TVR': (['frequency'], parameters['TVR']),
                         'VTX': (['frequency'], parameters['VTX']),
                         'Sv_offset': (['frequency'], Sv_offset),
                         'number_of_samples_digitized_per_pings': (['frequency'], range_samples_xml),
                         'number_of_digitized_samples_averaged_per_pings': (['frequency'],
                                                                            parameters['range_averaging_samples'])},
                        coords={'frequency': (['frequency'], freq,
                                              {'units': 'Hz',
                                               'valid_min': 0.0}),
                                'ping_time': (['ping_time'], ping_time,
                                              {'axis': 'T',
                                               'calendar': 'gregorian',
                                               'long_name': 'Timestamp of each ping',
                                               'standard_name': 'time',
                                               'units': 'seconds since 1970-01-01'}),
                                'range_bin': (['range_bin'], range_bin)},
                        attrs={'beam_mode': '',
                               'conversion_equation_t': 'type_4',
                               'number_of_frequency': parameters['num_freq'],
                               'number_of_pings_per_burst': parameters['pings_per_burst'],
                               'average_burst_pings_flag': parameters['average_burst_pings'],
                               # Temperature coefficients
                               'temperature_ka': parameters['ka'],
                               'temperature_kb': parameters['kb'],
                               'temperature_kc': parameters['kc'],
                               'temperature_A': parameters['A'],
                               'temperature_B': parameters['B'],
                               'temperature_C': parameters['C'],
                               # Tilt coefficients
                               'tilt_X_a': parameters['X_a'],
                               'tilt_X_b': parameters['X_b'],
                               'tilt_X_c': parameters['X_c'],
                               'tilt_X_d': parameters['X_d'],
                               'tilt_Y_a': parameters['Y_a'],
                               'tilt_Y_b': parameters['Y_b'],
                               'tilt_Y_c': parameters['Y_c'],
                               'tilt_Y_d': parameters['Y_d']})

        if self.save_ext == '.nc':
            comp = {'zlib': True, 'complevel': 4}
            c_settings = {var: comp for var in ds.data_vars} if self.compress else {}
            ds.to_netcdf(path=self.output_path, mode='a', group='Beam', encoding=c_settings)
        elif self.save_ext == '.zarr':
            comp = {'compressor': zarr.Blosc(cname='zstd', clevel=3, shuffle=2)}
            c_settings = {var: comp for var in ds.data_vars} if self.compress else {}
            ds.to_zarr(store=self.output_path, mode='a', group='Beam', encoding=c_settings)

    def set_vendor(self, ping_time):
        """Set the Vendor-specific group.
        """
        unpacked_data = self.convert_obj.unpacked_data
        freq = np.array(unpacked_data['frequency']) * 1000    # Frequency in Hz

        ds = xr.Dataset({
            'digitization_rate': (['frequency'], unpacked_data['dig_rate']),
            'lockout_index': (['frequency'], unpacked_data['lockout_index']),
            'number_of_bins_per_channel': (['frequency'], unpacked_data['num_bins']),
            'number_of_samples_per_average_bin': (['frequency'], unpacked_data['range_samples_per_bin']),
            'board_number': (['frequency'], unpacked_data['board_num']),
            'data_type': (['frequency'], unpacked_data['data_type']),
            'ping_status': (['ping_time'], unpacked_data['ping_status']),
            'number_of_acquired_pings': (['ping_time'], unpacked_data['num_acq_pings']),
            'first_ping': (['ping_time'], unpacked_data['first_ping']),
            'last_ping': (['ping_time'], unpacked_data['last_ping']),
            'data_error': (['ping_time'], unpacked_data['data_error']),
            'sensor_flag': (['ping_time'], unpacked_data['sensor_flag']),
            'ancillary': (['ping_time', 'ancillary_len'], unpacked_data['ancillary']),
            'ad_channels': (['ping_time', 'ad_len'], unpacked_data['ad']),
            'battery_main': (['ping_time'], unpacked_data['battery_main']),
            'battery_tx': (['ping_time'], unpacked_data['battery_tx'])},
            coords={
                'frequency': (['frequency'], freq,
                              {'units': 'Hz',
                               'valid_min': 0.0}),
                'ping_time': (['ping_time'], ping_time,
                              {'axis': 'T',
                               'calendar': 'gregorian',
                               'long_name': 'Timestamp of each ping',
                               'standard_name': 'time',
                               'units': 'seconds since 1970-01-01'}),
                'ancillary_len': (['ancillary_len'], list(range(len(unpacked_data['ancillary'][0])))),
                'ad_len': (['ad_len'], list(range(len(unpacked_data['ad'][0]))))},
            attrs={
                'profile_flag': unpacked_data['profile_flag'],
                'profile_number': unpacked_data['profile_number'],
                'burst_interval': unpacked_data['burst_int'],
                'ping_per_profile': unpacked_data['ping_per_profile'],
                'average_pings_flag': unpacked_data['avg_pings'],
                'spare_channel': unpacked_data['spare_chan'],
                'ping_period': unpacked_data['ping_period'],
                'phase': unpacked_data['phase'],
                'number_of_channels': unpacked_data['num_chan']}
        )

        if self.save_ext == '.nc':
            ds.to_netcdf(path=self.output_path, mode='a', group='Vendor')
        elif self.save_ext == '.zarr':
            ds.to_zarr(store=self.output_path, mode='a', group='Vendor')
