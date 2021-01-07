"""
Class to save unpacked echosounder data to appropriate groups in netcdf or zarr.
"""
import os
import xarray as xr
import numpy as np
import zarr
import netCDF4
from .set_groups_base import SetGroupsBase
from ..utils import io


class SetGroupsAZFP(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from AZFP data files.
    """
    def save(self):
        """Actually save groups to file by calling the set methods.
        """
        sonar_values = ('ASL Environmental Sciences', 'Acoustic Zooplankton Fish Profiler',
                        int(self.parser_obj.unpacked_data['serial_number']),
                        'Based on AZFP Matlab Toolbox', '1.4', 'echosounder')
        self.set_toplevel("AZFP")
        self.set_env()
        self.set_provenance()
        self.set_platform()
        self.set_sonar(sonar_values)
        self.set_beam()
        self.set_vendor()

    def set_env(self, ):
        """Set the Environment group.
        """
        # TODO Look at why this cannot be encoded without the modifications
        ping_time = (self.parser_obj.ping_time - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        # ping_time = self.convert_obj.ping_time
        ds = xr.Dataset({'temperature': (['ping_time'], self.parser_obj.unpacked_data['temperature'])},
                        coords={'ping_time': (['ping_time'], ping_time,
                                {'axis': 'T',
                                 'calendar': 'gregorian',
                                 'long_name': 'Timestamp of each ping',
                                 'standard_name': 'time',
                                 'units': 'seconds since 1970-01-01'})},
                        attrs={'long_name': "Water temperature",
                               'units': "C"})
        # Save to file
        io.save_file(ds, path=self.output_path, mode='a', group='Environment', engine=self.engine)

    def set_platform(self):
        """Set the Platform group.
        """
        platform_dict = {'platform_name': self.ui_param['platform_name'],
                         'platform_type': self.ui_param['platform_type'],
                         'platform_code_ICES': self.ui_param['platform_code_ICES']}
        if self.engine == 'netcdf4':
            with netCDF4.Dataset(self.output_path, 'a', format='NETCDF4') as ncfile:
                plat = ncfile.createGroup('Platform')
                [plat.setncattr(k, v) for k, v in platform_dict.items()]
        elif self.engine == 'zarr':
            zarrfile = zarr.open(self.output_path, mode='a')
            plat = zarrfile.create_group('Platform')
            for k, v in platform_dict.items():
                plat.attrs[k] = v

    def set_beam(self):
        """Set the Beam group.
        """
        unpacked_data = self.parser_obj.unpacked_data
        parameters = self.parser_obj.parameters
        anc = np.array(unpacked_data['ancillary'])   # convert to np array for easy slicing
        dig_rate = unpacked_data['dig_rate']         # dim: freq
        freq = np.array(unpacked_data['frequency']) * 1000    # Frequency in Hz
        ping_time = (self.parser_obj.ping_time - np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')

        # Build variables in the output xarray Dataset
        N = []   # for storing backscatter_r values for each frequency
        Sv_offset = np.zeros(freq.shape)
        for ich in range(len(freq)):
            Sv_offset[ich] = self.parser_obj._calc_Sv_offset(freq[ich], unpacked_data['pulse_length'][ich])
            N.append(np.array([unpacked_data['counts'][p][ich]
                               for p in range(len(unpacked_data['year']))]))

        # Largest number of counts along the range dimension among the different channels
        longest_range_bin = np.max(unpacked_data['num_bins'])
        range_bin = np.arange(longest_range_bin)

        # Pad power data
        if any(unpacked_data['num_bins'] != longest_range_bin):
            N_tmp = np.full((len(N), len(ping_time), longest_range_bin), np.nan)
            for i, n in enumerate(N):
                N_tmp[i, :, :n.shape[1]] = n
            N = N_tmp
            del N_tmp

        tdn = unpacked_data['pulse_length'] / 1e6  # Convert microseconds to seconds
        range_samples_xml = np.array(parameters['range_samples'])         # from xml file
        range_samples_per_bin = unpacked_data['range_samples_per_bin']    # from data header

        # Calculate sample interval in seconds
        if len(dig_rate) == len(range_samples_per_bin):
            sample_int = range_samples_per_bin / dig_rate
        else:
            raise ValueError("dig_rate and range_samples not unique across frequencies")

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
                                               'units': 'seconds since 1900-01-01'}),
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

        # Save to file
        io.save_file(ds.chunk({'range_bin': 25000, 'ping_time': 100}),
                     path=self.output_path, mode='a', engine=self.engine,
                     group='Beam', compression_settings=self.compression_settings)

    def set_vendor(self):
        """Set the Vendor-specific group.
        """
        unpacked_data = self.parser_obj.unpacked_data
        freq = np.array(unpacked_data['frequency']) * 1000    # Frequency in Hz
        ping_time = (self.parser_obj.ping_time - np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')

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

        # Save to file
        io.save_file(ds, path=self.output_path, mode='a', engine=self.engine,
                     group='Vendor', compression_settings=self.compression_settings)
