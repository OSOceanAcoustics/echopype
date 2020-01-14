import os
from collections import defaultdict
import numpy as np
from datetime import datetime as dt
import pytz
import pynmea2

from echopype.convert.utils.ek_raw_io import RawSimradFile, SimradEOF
from echopype.convert.utils.nmea_data import NMEAData
from echopype.convert.utils.set_groups import SetGroups
from echopype._version import get_versions
from .convertbase import ConvertBase
ECHOPYPE_VERSION = get_versions()['version']
del get_versions


class ConvertEK80(ConvertBase):
    def __init__(self, _filename=""):
        ConvertBase.__init__(self)
        self.filename = _filename  # path to EK60 .raw filename to be parsed

        # Initialize file parsing storage variables
        self.config_datagram = None
        self.nmea_data = NMEAData()  # object for NMEA data
        self.ping_data_dict = {}   # dictionary to store metadata
        self.power_dict = {}    # dictionary to store power data
        self.angle_dict = {}    # dictionary to store angle data
        self.complex_dict = {}  # dictionary to store complex data
        self.ping_time = []     # list to store ping time
        self.environment = {}   # dictionary to store environment data
        self.parameters = defaultdict(dict)   # Dictionary to hold parameter data
        self.mru_data = defaultdict(list)     # dictionary to store MRU data (heading, pitch, roll, heave)
        self.fil_dict = defaultdict(dict)
        self.complex = True     # Flags whether or not backscatter is complex
        self.ch_ids = []
        self.nc_path = None
        self.zarr_path = None

    def _read_datagrams(self, fid):
        """
        Read various datagrams until the end of a ``.raw`` file.

        Only includes code for storing RAW, NMEA, MRU, and XML datagrams and
        ignoring the TAG datagram.

        Parameters
        ----------
        fid
            a RawSimradFile file object opened in ``self.load_ek60_raw()``
        """

        num_datagrams_parsed = 0
        tmp_num_ch_per_ping_parsed = 0   # number of channels of the same ping parsed
                                         # this is used to control saving only pings
                                         # that have all freq channels present
        tmp_datagram_dict = []  # tmp list of datagrams, only saved to actual output
                                # structure if data from all freq channels are present

        while True:
            try:
                new_datagram = fid.read(1)
            except SimradEOF:
                break

            num_datagrams_parsed += 1

            # Convert the timestamp to a datetime64 object.
            new_datagram['timestamp'] = np.datetime64(new_datagram['timestamp'].replace(tzinfo=None), '[ms]')

            # The first XML datagram contains environment information
            # Subsequent XML datagrams preceed RAW datagrams and give parameter information
            if new_datagram['type'].startswith("XML"):
                if new_datagram['subtype'] == 'environment':
                    self.environment = new_datagram['environment']
                elif new_datagram['subtype'] == 'parameter':
                    current_parameters = new_datagram['parameter']
                    # If frequency_start/end is not found, fill values with frequency
                    if 'frequency_start' not in current_parameters:
                        self.parameters[current_parameters['channel_id']]['frequency_start'].append(
                            int(current_parameters['frequency']))
                        self.parameters[current_parameters['channel_id']]['frequency_end'].append(
                            int(current_parameters['frequency']))
                    else:
                        self.parameters[current_parameters['channel_id']]['frequency_start'].append(
                            int(current_parameters['frequency_start']))
                        self.parameters[current_parameters['channel_id']]['frequency_end'].append(
                            int(current_parameters['frequency_end']))
                    self.parameters[current_parameters['channel_id']]['pulse_duration'].append(
                        current_parameters['pulse_duration'])
                    self.parameters[current_parameters['channel_id']]['pulse_form'].append(
                        current_parameters['pulse_form'])
                    self.parameters[current_parameters['channel_id']]['sample_interval'].append(
                        current_parameters['sample_interval'])
                    self.parameters[current_parameters['channel_id']]['slope'].append(
                        current_parameters['slope'])
                    self.parameters[current_parameters['channel_id']]['transmit_power'].append(
                        current_parameters['transmit_power'])
                    self.parameters[current_parameters['channel_id']]['timestamp'].append(
                        new_datagram['timestamp'])
            # Contains data
            elif new_datagram['type'].startswith("RAW"):
                curr_ch_id = new_datagram['channel_id']
                if current_parameters['channel_id'] != curr_ch_id:
                    raise ValueError("Parameter ID does not match RAW")

                if new_datagram['n_complex'] == 0:
                    self.complex = False

                # Reset counter and storage for parsed number of channels
                # if encountering datagram from the first channel
                if curr_ch_id == self.ch_ids[0]:
                    tmp_num_ch_per_ping_parsed = -1
                    tmp_datagram_dict = []

                # Save datagram temporarily before knowing if all freq channels are present
                tmp_num_ch_per_ping_parsed += 1
                tmp_datagram_dict.append(new_datagram)
                # Actually save datagram when all freq channels are present
                if np.all(np.array([curr_ch_id, self.ch_ids[tmp_num_ch_per_ping_parsed]]) ==
                          self.ch_ids[-1]):

                    # append ping time from first channel
                    self.ping_time.append(tmp_datagram_dict[0]['timestamp'])

                    for i, ch_id in enumerate(self.ch_ids):
                        # self._append_channel_ping_data(ch_id, tmp_datagram_dict[ch_id])  # ping-by-ping metadata
                        self.power_dict[ch_id].append(tmp_datagram_dict[i]['power'])  # append power data
                        self.angle_dict[ch_id].append(tmp_datagram_dict[i]['angle'])  # append angle data
                        self.complex_dict[ch_id].append(tmp_datagram_dict[i]['complex'])  # append complex data

            # NME datagrams store ancillary data as NMEA-0817 style ASCII data.
            elif new_datagram['type'].startswith("NME"):
                # Add the datagram to our nmea_data object.
                self.nmea_data.add_datagram(new_datagram['timestamp'],
                                            new_datagram['nmea_string'])

            # MRU datagrams contain motion data for each ping
            elif new_datagram['type'].startswith("MRU"):
                self.mru_data['heading'].append(new_datagram['heading'])
                self.mru_data['pitch'].append(new_datagram['pitch'])
                self.mru_data['roll'].append(new_datagram['roll'])
                self.mru_data['heave'].append(new_datagram['heave'])
                self.mru_data['timestamp'].append(new_datagram['timestamp'])

            # FIL datagrams contain filters for proccessing bascatter data
            elif new_datagram['type'].startswith("FIL"):
                self.fil_dict[new_datagram['channel_id']][new_datagram['stage']] = new_datagram['coefficients']

    def load_ek80_raw(self):
        print('%s  converting file: %s' % (dt.now().strftime('%H:%M:%S'), os.path.basename(self.filename)))

        with RawSimradFile(self.filename, 'r') as fid:
            self.config_datagram = fid.read(1)
            self.config_datagram['timestamp'] = np.datetime64(self.config_datagram['timestamp'], '[ms]')

            # IDs of the channels found in the dataset
            self.ch_ids = list(self.config_datagram[self.config_datagram['subtype']])

            for ch_id in self.ch_ids:
                self.ping_data_dict[ch_id] = defaultdict(list)
                self.ping_data_dict[ch_id]['frequency'] = \
                    self.config_datagram['configuration'][ch_id]['transducer_frequency']
                self.power_dict[ch_id] = []
                self.angle_dict[ch_id] = []
                self.complex_dict[ch_id] = []

                # Parameters recorded for each frequency for each ping
                self.parameters[ch_id]['frequency_start'] = []
                self.parameters[ch_id]['frequency_end'] = []
                self.parameters[ch_id]['frequency'] = []
                self.parameters[ch_id]['pulse_duration'] = []
                self.parameters[ch_id]['pulse_form'] = []
                self.parameters[ch_id]['sample_interval'] = []
                self.parameters[ch_id]['slope'] = []
                self.parameters[ch_id]['transmit_power'] = []
                self.parameters[ch_id]['timestamp'] = []

            # Read the rest of datagrams
            self._read_datagrams(fid)

    def save(self, file_format):
        """Save data from EK60 `.raw` to netCDF format.
        """

        # Subfunctions to set various dictionaries
        def _set_toplevel_dict():
            out_dict = dict(Conventions='CF-1.7, SONAR-netCDF4, ACDD-1.3',
                            keywords='EK80',
                            sonar_convention_authority='ICES',
                            sonar_convention_name='SONAR-netCDF4',
                            sonar_convention_version='1.7',
                            summary='',
                            title='')
            out_dict['date_created'] = dt.strptime(filedate + '-' + filetime, '%Y%m%d-%H%M%S').isoformat() + 'Z'
            return out_dict

        def _set_env_dict():
            return dict(temperature=self.environment['temperature'],
                        depth=self.environment['depth'],
                        acidity=self.environment['acidity'],
                        salinity=self.environment['salinity'],
                        sound_speed_indicative=self.environment['sound_speed'])

        def _set_prov_dict():
            return dict(conversion_software_name='echopype',
                        conversion_software_version=ECHOPYPE_VERSION,
                        conversion_time=dt.now(tz=pytz.utc).isoformat(timespec='seconds'))  # use UTC time

        def _set_sonar_dict():
            channels = defaultdict(dict)
            for ch_id in self.ch_ids:
                channels[ch_id]['sonar_manufacturer'] = 'Simrad'
                channels[ch_id]['sonar_model'] = self.config_datagram['configuration'][ch_id]['transducer_name']
                channels[ch_id]['sonar_serial_number'] = self.config_datagram['configuration'][ch_id]['serial_number']
                channels[ch_id]['sonar_software_name'] = self.config_datagram['configuration'][ch_id]['application_name']
                channels[ch_id]['sonar_software_version'] = self.config_datagram['configuration'][ch_id]['application_version']
                channels[ch_id]['sonar_type'] = 'echosounder'
            return channels

        def _set_platform_dict():
            out_dict = dict()
            # TODO: Need to reconcile the logic between using the unpacked "survey_name"
            #  and the user-supplied platform_name
            # self.platform_name = self.config_datagram['survey_name']
            out_dict['platform_name'] = self.platform_name
            out_dict['platform_type'] = self.platform_type
            out_dict['platform_code_ICES'] = self.platform_code_ICES

            # Read pitch/roll/heave from ping data
            out_dict['ping_time'] = self.ping_time  # [seconds since 1900-01-01] for xarray.to_netcdf conversion
            out_dict['pitch'] = np.array(self.mru_data['pitch'])
            out_dict['roll'] = np.array(self.mru_data['roll'])
            out_dict['heave'] = np.array(self.mru_data['heave'])
            out_dict['water_level'] = self.environment['water_level_draft']

            # Read lat/long from NMEA datagram
            idx_loc = np.argwhere(np.isin(self.nmea_data.messages, ['GGA', 'GLL', 'RMC'])).squeeze()
            nmea_msg = []
            [nmea_msg.append(pynmea2.parse(self.nmea_data.raw_datagrams[x])) for x in idx_loc]
            out_dict['lat'] = np.array([x.latitude for x in nmea_msg])
            out_dict['lon'] = np.array([x.longitude for x in nmea_msg])
            out_dict['location_time'] = self.nmea_data.nmea_times[idx_loc]
            return out_dict

        def _set_nmea_dict():
            # Assemble dict for saving to groups
            out_dict = dict()
            out_dict['nmea_time'] = self.nmea_data.nmea_times
            out_dict['nmea_datagram'] = self.nmea_data.raw_datagrams
            return out_dict

        def _set_beam_dict():
            beam_dict = dict()
            beam_dict['beam_mode'] = 'vertical'
            beam_dict['conversion_equation_t'] = 'type_3'  # type_3 is EK60 conversion
            beam_dict['ping_time'] = self.ping_time   # [seconds since 1900-01-01] for xarray.to_netcdf conversion
            beam_dict['frequency'] = freq
            # beam_dict['range_lengths'] = self.range_lengths
            # beam_dict['power_dict'] = self.power_dict_split
            # beam_dict['angle_dict'] = self.angle_dict_split

            backscatter_r = []
            backscatter_i = []
            for tx in self.ch_ids:
                if self.complex:
                    reshaped = np.array(self.complex_dict[tx]).reshape((ping_num, -1, 4))
                    backscatter_r.append(np.real(reshaped))
                    backscatter_i.append(np.imag(reshaped))
                else:
                    backscatter_r.append(np.array(self.power_dict[tx], dtype='float32'))

            # Find index of channel with longest range
            largest = max(enumerate(backscatter_r), key=lambda x: x[1].shape[1])[0]

            # Loop through each transducer for channel-specific variables
            bm_width = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
            bm_dir = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
            bm_angle = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
            tx_pos = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
            beam_dict['equivalent_beam_angle'] = np.zeros(shape=(tx_num,), dtype='float32')
            beam_dict['gain_correction'] = np.zeros(shape=(tx_num,), dtype='float32')
            beam_dict['gpt_software_version'] = []
            beam_dict['channel_id'] = []
            beam_dict['frequency_start'] = []
            beam_dict['frequency_end'] = []

            c_seq = 0
            for k, c in self.config_datagram['configuration'].items():
                bm_width['beamwidth_receive_major'][c_seq] = c['beam_width_alongship']
                bm_width['beamwidth_receive_minor'][c_seq] = c['beam_width_athwartship']
                bm_width['beamwidth_transmit_major'][c_seq] = c['beam_width_alongship']
                bm_width['beamwidth_transmit_minor'][c_seq] = c['beam_width_athwartship']
                bm_dir['beam_direction_x'][c_seq] = c['transducer_alpha_x']
                bm_dir['beam_direction_y'][c_seq] = c['transducer_alpha_y']
                bm_dir['beam_direction_z'][c_seq] = c['transducer_alpha_z']
                bm_angle['angle_offset_alongship'][c_seq] = c['angle_offset_alongship']
                bm_angle['angle_offset_athwartship'][c_seq] = c['angle_offset_athwartship']
                bm_angle['angle_sensitivity_alongship'][c_seq] = c['angle_sensitivity_alongship']
                bm_angle['angle_sensitivity_athwartship'][c_seq] = c['angle_sensitivity_athwartship']
                tx_pos['transducer_offset_x'][c_seq] = c['transducer_offset_x']
                tx_pos['transducer_offset_y'][c_seq] = c['transducer_offset_y']
                tx_pos['transducer_offset_z'][c_seq] = c['transducer_offset_z']
                beam_dict['equivalent_beam_angle'][c_seq] = c['equivalent_beam_angle']
                # TODO: gain is 5 values in test dataset
                beam_dict['gain_correction'][c_seq] = c['gain'][c_seq]
                beam_dict['gpt_software_version'].append(c['transceiver_software_version'])
                beam_dict['channel_id'].append(c['channel_id'])
                beam_dict['frequency_start'].append(self.parameters[k]['frequency_start'])
                beam_dict['frequency_end'].append(self.parameters[k]['frequency_end'])

                # Pad each channel with nan so that they can be stacked
                diff = backscatter_r[largest].shape[1] - backscatter_r[c_seq].shape[1]
                if diff > 0:
                    if self.complex:
                        backscatter_r[c_seq] = np.pad(backscatter_r[c_seq], ((0, 0), (0, diff), (0, 0)),
                                                      mode='constant', constant_values=np.nan)
                        backscatter_i[c_seq] = np.pad(backscatter_i[c_seq], ((0, 0), (0, diff), (0, 0)),
                                                      mode='constant', constant_values=np.nan)
                    else:
                        backscatter_r[c_seq] = np.pad(backscatter_r[c_seq], ((0, 0), (0, diff)),
                                                      mode='constant', constant_values=np.nan)
                c_seq += 1

            # Stack channels and order axis as: channel, quadrant, ping, range
            if self.complex:
                beam_dict['backscatter_r'] = np.moveaxis(np.stack(backscatter_r), 3, 1)
                beam_dict['backscatter_i'] = np.moveaxis(np.stack(backscatter_i), 3, 1)
                beam_dict['complex'] = True
            else:
                beam_dict['backscatter_r'] = np.stack(backscatter_r)
                beam_dict['complex'] = False
            beam_dict['range_bin'] = np.arange(backscatter_r[largest].shape[1])

            beam_dict['beam_width'] = bm_width
            beam_dict['beam_direction'] = bm_dir
            beam_dict['beam_angle'] = bm_angle
            beam_dict['transducer_position'] = tx_pos

            # Loop through each transducer for variables that may vary at each ping
            # -- this rarely is the case for EK60 so we check first before saving
            pl_tmp = np.unique(self.parameters[self.ch_ids[0]]['pulse_duration']).size
            pw_tmp = np.unique(self.parameters[self.ch_ids[0]]['transmit_power']).size
            # bw_tmp = np.unique(self.ping_data_dict[1]['bandwidth']).size      # Not in EK80
            si_tmp = np.unique(self.parameters[self.ch_ids[0]]['sample_interval']).size
            if np.all(np.array([pl_tmp, pw_tmp, si_tmp]) == 1):
                tx_sig = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
                beam_dict['sample_interval'] = np.zeros(shape=(tx_num,), dtype='float32')
                for t_seq in range(tx_num):
                    tx_sig['transmit_duration_nominal'][t_seq] = \
                        np.float32(self.parameters[self.ch_ids[t_seq]]['pulse_duration'][0])
                    tx_sig['transmit_power'][t_seq] = \
                        np.float32(self.parameters[self.ch_ids[t_seq]]['transmit_power'][0])
                    # tx_sig['transmit_bandwidth'][t_seq] = \
                    #     np.float32((self.parameters[self.ch_ids[t_seq]]['bandwidth'][0])
                    beam_dict['sample_interval'][t_seq] = \
                        np.float32(self.parameters[self.ch_ids[t_seq]]['sample_interval'][0])
            else:
                tx_sig = defaultdict(lambda: np.zeros(shape=(tx_num, ping_num), dtype='float32'))
                beam_dict['sample_interval'] = np.zeros(shape=(tx_num, ping_num), dtype='float32')
                for t_seq in range(tx_num):
                    tx_sig['transmit_duration_nominal'][t_seq, :] = \
                        np.array(self.parameters[self.ch_ids[t_seq]]['pulse_duration'], dtype='float32')
                    tx_sig['transmit_power'][t_seq, :] = \
                        np.array(self.parameters[self.ch_ids[t_seq]]['transmit_power'], dtype='float32')
                    # tx_sig['transmit_bandwidth'][t_seq, :] = \
                    #     np.array(self.parameters[self.ch_ids[t_seq]]['bandwidth'], dtype='float32')
                    beam_dict['sample_interval'][t_seq, :] = \
                        np.array(self.parameters[self.ch_ids[t_seq]]['sample_interval'], dtype='float32')

            beam_dict['transmit_signal'] = tx_sig
            # Build other parameters
            # beam_dict['non_quantitative_processing'] = np.array([0, ] * freq.size, dtype='int32')
            # -- sample_time_offset is set to 2 for EK60 data, this value is NOT from sample_data['offset']
            # beam_dict['sample_time_offset'] = np.array([2, ] * freq.size, dtype='int32')

            # TODO: Make the following work
            # idx = [np.argwhere(np.isclose(tx_sig['transmit_duration_nominal'][x - 1],
            #                               self.config_datagram['transceivers'][x]['pulse_length_table'])).squeeze()
            #        for x in self.config_datagram['transceivers'].keys()]
            # beam_dict['sa_correction'] = \
            #     np.array([x['sa_correction'][y]
            #               for x, y in zip(self.config_datagram['transceivers'].values(), np.array(idx))])

            return beam_dict

        if not bool(self.power_dict):  # if haven't parsed .raw file
            self.load_ek80_raw()

        filename = os.path.splitext(os.path.basename(self.filename))[0]
        self.save_path = os.path.join(os.path.split(self.filename)[0], filename + file_format)
        self.nc_path = os.path.join(os.path.split(self.filename)[0], filename + '.nc')
        self.zarr_path = os.path.join(os.path.split(self.filename)[0], filename + '.zarr')
        # filename must have "-" as the field separator for the last 2 fields
        filename_tup = filename.split('-')
        filedate = filename_tup[len(filename_tup) - 2].replace("D", "")
        filetime = filename_tup[len(filename_tup) - 1].replace("T", "")

        if os.path.exists(self.save_path):
            print(f'          ... this file has already been converted to {file_format}, conversion not executed.')
        else:
            tx_num = len(self.ch_ids)
            ping_num = len(self.ping_time)
            freq = np.array([self.config_datagram['configuration'][x]['transducer_frequency']
                            for x in self.config_datagram['configuration'].keys()], dtype='float32')

            grp = SetGroups(file_path=self.save_path, echo_type='EK80')
            grp.set_toplevel(_set_toplevel_dict())  # top-level group
            grp.set_env(_set_env_dict())            # environment group
            grp.set_provenance(os.path.basename(self.filename), _set_prov_dict())    # provenance group
            grp.set_platform(_set_platform_dict())  # platform group
            grp.set_nmea(_set_nmea_dict())          # platform/NMEA group
            grp.set_sonar(_set_sonar_dict())        # sonar group
            grp.set_beam(_set_beam_dict())          # beam group
