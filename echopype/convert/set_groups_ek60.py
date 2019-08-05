from .set_groups import SetGroupsBase
import xarray as xr
import os
import numpy as np


class SetGroupsEK60(SetGroupsBase):
    """Class for setting groups in netCDF file for EK60 data.
    """

    def set_beam(self, beam_dict):
        """Set the Beam group in the nc file.

        Parameters
        ----------
        beam_dict
            dictionary containing general beam parameters
        """

        # Only save beam group if file_path exists
        if not os.path.exists(self.file_path):
            print('netCDF file does not exist, exiting without saving Beam group...')
        else:
            # Convert np.datetime64 numbers to seconds since 1900-01-01
            # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
            ping_time = (beam_dict['ping_time'] - np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')

            ds = xr.Dataset(
                {'backscatter_r': (['frequency', 'ping_time', 'range_bin'], beam_dict['backscatter_r'],
                                   {'long_name': 'Backscatter power',
                                    'units': 'dB'}),
                    'beamwidth_receive_major': (['frequency'], beam_dict['beam_width']['beamwidth_receive_major'],
                                                {'long_name': 'Half power one-way receive beam width along '
                                                              'major (horizontal) axis of beam',
                                                 'units': 'arc_degree',
                                                 'valid_range': (0.0, 360.0)}),
                    'beamwidth_receive_minor': (['frequency'], beam_dict['beam_width']['beamwidth_receive_minor'],
                                                {'long_name': 'Half power one-way receive beam width along '
                                                              'minor (vertical) axis of beam',
                                                 'units': 'arc_degree',
                                                 'valid_range': (0.0, 360.0)}),
                    'beamwidth_transmit_major': (['frequency'], beam_dict['beam_width']['beamwidth_transmit_major'],
                                                 {'long_name': 'Half power one-way transmit beam width along '
                                                               'major (horizontal) axis of beam',
                                                  'units': 'arc_degree',
                                                  'valid_range': (0.0, 360.0)}),
                    'beamwidth_transmit_minor': (['frequency'], beam_dict['beam_width']['beamwidth_transmit_minor'],
                                                {'long_name': 'Half power one-way transmit beam width along '
                                                            'minor (vertical) axis of beam',
                                                'units': 'arc_degree',
                                                'valid_range': (0.0, 360.0)}),
                    'beam_direction_x': (['frequency'], beam_dict['beam_direction']['beam_direction_x'],
                                        {'long_name': 'x-component of the vector that gives the pointing '
                                                    'direction of the beam, in sonar beam coordinate '
                                                    'system',
                                        'units': '1',
                                        'valid_range': (-1.0, 1.0)}),
                    'beam_direction_y': (['frequency'], beam_dict['beam_direction']['beam_direction_x'],
                                        {'long_name': 'y-component of the vector that gives the pointing '
                                                    'direction of the beam, in sonar beam coordinate '
                                                    'system',
                                        'units': '1',
                                        'valid_range': (-1.0, 1.0)}),
                    'beam_direction_z': (['frequency'], beam_dict['beam_direction']['beam_direction_x'],
                                        {'long_name': 'z-component of the vector that gives the pointing '
                                                    'direction of the beam, in sonar beam coordinate '
                                                    'system',
                                        'units': '1',
                                        'valid_range': (-1.0, 1.0)}),
                    'equivalent_beam_angle': (['frequency'], beam_dict['equivalent_beam_angle'],
                                            {'long_name': 'Equivalent beam angle',
                                            'units': 'sr',
                                            'valid_range': (0.0, 4 * np.pi)}),
                    'gain_correction': (['frequency'], beam_dict['gain_correction'],
                                        {'long_name': 'Gain correction',
                                        'units': 'dB'}),
                    'non_quantitative_processing': (['frequency'], beam_dict['non_quantitative_processing'],
                                                    {'flag_meanings': 'no_non_quantitative_processing',
                                                    'flag_values': '0',
                                                    'long_name': 'Presence or not of non-quantitative '
                                                                'processing applied to the backscattering '
                                                                'data (sonar specific)'}),
                    'sample_interval': (['frequency'], beam_dict['sample_interval'],
                                        {'long_name': 'Interval between recorded raw data samples',
                                        'units': 's',
                                        'valid_min': 0.0}),
                    'sample_time_offset': (['frequency'], beam_dict['sample_time_offset'],
                                        {'long_name': 'Time offset that is subtracted from the timestamp '
                                                        'of each sample',
                                            'units': 's'}),
                    'transmit_bandwidth': (['frequency'], beam_dict['transmit_signal']['transmit_bandwidth'],
                                        {'long_name': 'Nominal bandwidth of transmitted pulse',
                                            'units': 'Hz',
                                            'valid_min': 0.0}),
                    'transmit_duration_nominal': (['frequency'], beam_dict['transmit_signal']['transmit_duration_nominal'],
                                                {'long_name': 'Nominal bandwidth of transmitted pulse',
                                                'units': 's',
                                                'valid_min': 0.0}),
                    'transmit_power': (['frequency'], beam_dict['transmit_signal']['transmit_power'],
                                    {'long_name': 'Nominal transmit power',
                                        'units': 'W',
                                        'valid_min': 0.0}),
                    'transducer_offset_x': (['frequency'], beam_dict['transducer_position']['transducer_offset_x'],
                                            {'long_name': 'x-axis distance from the platform coordinate system '
                                                        'origin to the sonar transducer',
                                            'units': 'm'}),
                    'transducer_offset_y': (['frequency'], beam_dict['transducer_position']['transducer_offset_y'],
                                            {'long_name': 'y-axis distance from the platform coordinate system '
                                                        'origin to the sonar transducer',
                                            'units': 'm'}),
                    'transducer_offset_z': (['frequency'], beam_dict['transducer_position']['transducer_offset_z'],
                                            {'long_name': 'z-axis distance from the platform coordinate system '
                                                        'origin to the sonar transducer',
                                            'units': 'm'}),
                    },
                coords={'frequency': (['frequency'], beam_dict['frequency'],
                                        {'units': 'Hz',
                                        'valid_min': 0.0}),
                        'ping_time': (['ping_time'], ping_time,
                                        {'axis': 'T',
                                        'calendar': 'gregorian',
                                        'long_name': 'Timestamp of each ping',
                                        'standard_name': 'time',
                                        'units': 'seconds since 1900-01-01'}),
                        'range_bin': (['range_bin'], beam_dict['range_bin'])},
                attrs={'beam_mode': beam_dict['beam_mode'],
                        'conversion_equation_t': beam_dict['conversion_equation_t']})

            # Below are specific to Simrad EK60 .raw files
            if 'channel_id' in beam_dict:
                ds['channel_id'] = ('frequency', beam_dict['channel_id'])
            if 'gpt_software_version' in beam_dict:
                ds['gpt_software_version'] = ('frequency', beam_dict['gpt_software_version'])
            if 'sa_correction' in beam_dict:
                ds['sa_correction'] = ('frequency', beam_dict['sa_correction'])

            # save to file
            ds.to_netcdf(path=self.file_path, mode="a", group="Beam")
