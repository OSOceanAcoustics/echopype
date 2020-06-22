import os
import shutil
from collections import defaultdict
import numpy as np
import xarray as xr
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
    """Class for converting EK80 ``.raw`` files.
    """
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
        self.mru_data = defaultdict(list)     # Dictionary to store MRU data (heading, pitch, roll, heave)
        self.fil_coeffs = defaultdict(dict)   # Dictionary to store PC and WBT coefficients
        self.fil_df = defaultdict(dict)       # Dictionary to store filter decimation factors
        self.ch_ids = []                      # List of all channel ids
        self.recorded_ch_ids = []

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
                        self.parameters[current_parameters['channel_id']]['frequency'].append(
                            int(current_parameters['frequency']))
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

                # tmp_num_ch_per_ping_parsed += 1
                if curr_ch_id not in self.recorded_ch_ids:
                    self.recorded_ch_ids.append(curr_ch_id)

                # append ping time from first channel
                if curr_ch_id == self.recorded_ch_ids[0]:
                    self.ping_time.append(new_datagram['timestamp'])

                self.power_dict[curr_ch_id].append(new_datagram['power'])  # append power data
                self.angle_dict[curr_ch_id].append(new_datagram['angle'])  # append angle data
                self.complex_dict[curr_ch_id].append(new_datagram['complex'])  # append complex data

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
                self.fil_coeffs[new_datagram['channel_id']][new_datagram['stage']] = new_datagram['coefficients']
                self.fil_df[new_datagram['channel_id']][new_datagram['stage']] = new_datagram['decimation_factor']

    def load_ek80_raw(self, raw):
        """Method to parse the EK80 ``.raw`` data file.

        This method parses the ``.raw`` file and saves the parsed data
        to the ConvertEK80 instance.

        Parameters
        ----------
        raw : str
            raw filename
        """
        print('%s  converting file: %s' % (dt.now().strftime('%H:%M:%S'), os.path.basename(raw)))

        with RawSimradFile(raw, 'r') as fid:
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
            # Remove empty lists
            for ch_id in self.ch_ids:
                if all(x is None for x in self.power_dict[ch_id]):
                    self.power_dict[ch_id] = None
                if all(x is None for x in self.complex_dict[ch_id]):
                    self.complex_dict[ch_id] = None

        if len(self.ch_ids) != len(self.recorded_ch_ids):
            self.ch_ids = self.recorded_ch_ids

    def sort_ch_ids(self):
        """ Sorts the channel ids into broadband and continuous wave channel ids

            Returns
            -------
            2 lists containing the bb channel ids and the cw channel ids
        """
        bb_ch_ids = []
        cw_ch_ids = []
        for k, v in self.complex_dict.items():
            if v is not None:
                bb_ch_ids.append(k)
            else:
                if self.power_dict[k] is not None:
                    cw_ch_ids.append(k)
        return bb_ch_ids, cw_ch_ids

    # Functions to set various dictionaries
    def _set_toplevel_dict(self, raw_file):
        # filename must have "-" as the field separator for the last 2 fields. Uses first file
        filename_tup = os.path.splitext(os.path.basename(raw_file))[0].split("-")
        filedate = filename_tup[len(filename_tup) - 2].replace("D", "")
        filetime = filename_tup[len(filename_tup) - 1].replace("T", "")

        out_dict = dict(Conventions='CF-1.7, SONAR-netCDF4, ACDD-1.3',
                        keywords='EK80',
                        sonar_convention_authority='ICES',
                        sonar_convention_name='SONAR-netCDF4',
                        sonar_convention_version='1.7',
                        summary='',
                        title='')
        out_dict['date_created'] = dt.strptime(filedate + '-' + filetime, '%Y%m%d-%H%M%S').isoformat() + 'Z'
        return out_dict

    def _set_env_dict(self):
        return dict(temperature=self.environment['temperature'],
                    depth=self.environment['depth'],
                    acidity=self.environment['acidity'],
                    salinity=self.environment['salinity'],
                    sound_speed_indicative=self.environment['sound_speed'])

    def _set_prov_dict(self, raw_file, combine_opt):
        out_dict = dict(conversion_software_name='echopype',
                        conversion_software_version=ECHOPYPE_VERSION,
                        conversion_time=dt.now(tz=pytz.utc).isoformat(timespec='seconds'))  # use UTC time
        # Send a list of all filenames if combining raw files. Else, send the one file to be converted
        out_dict['src_filenames'] = self.filename if combine_opt else [raw_file]
        return out_dict

    def _set_sonar_dict(self, ch_ids, path):
        channels = defaultdict(dict)
        channels['path'] = path
        # channels['frequency'] = np.array([self.config_datagram['configuration'][x]['transducer_frequency']
        #                                   for x in self.ch_ids], dtype='float32')
        for ch_id in ch_ids:
            channels[ch_id]['frequency'] = self.config_datagram['configuration'][ch_id]['transducer_frequency']
            channels[ch_id]['sonar_manufacturer'] = 'Simrad'
            channels[ch_id]['sonar_model'] = self.config_datagram['configuration'][ch_id]['transducer_name']
            channels[ch_id]['sonar_serial_number'] = self.config_datagram['configuration'][ch_id]['serial_number']
            channels[ch_id]['sonar_software_name'] = self.config_datagram['configuration'][ch_id]['application_name']
            channels[ch_id]['sonar_software_version'] = self.config_datagram['configuration'][ch_id]['application_version']
            channels[ch_id]['sonar_type'] = 'echosounder'
        return channels

    def _set_platform_dict(self):
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

    def _set_nmea_dict(self):
        # Assemble dict for saving to groups
        out_dict = dict()
        out_dict['nmea_time'] = self.nmea_data.nmea_times
        out_dict['nmea_datagram'] = self.nmea_data.raw_datagrams
        return out_dict

    def _set_beam_dict(self, ch_ids, bb, path):
        """Sets the dictionary used to save the beam group.

        Parameters
        ----------
        ch_ids : list of str
            lists of all channels to be saved. Either all bb or all cw channels
        bb : bool
            flags whether the data is broadband or not
        path : str
            save path

        Returns
        -------
        Dictionary containing data for saving the beam group
        """
        beam_dict = dict()
        beam_dict['path'] = path        # Path to save file to
        beam_dict['beam_mode'] = 'vertical'
        beam_dict['conversion_equation_t'] = 'type_3'  # type_3 is EK60 conversion
        beam_dict['ping_time'] = self.ping_time   # [seconds since 1900-01-01] for xarray.to_netcdf conversion
        beam_dict['frequency'] = np.array([self.config_datagram['configuration'][x]['transducer_frequency']
                                          for x in ch_ids], dtype='float32')
        tx_num = len(ch_ids)
        ping_num = len(self.ping_time)
        b_r_tmp = {}      # Real part of broadband backscatter
        b_i_tmp = {}      # Imaginary part of b 99-6 raodband backscatter

        # Find largest array in order to pad and stack smaller arrays
        max_len = 0
        for tx in ch_ids:
            if bb:
                reshaped = np.array(self.complex_dict[tx]).reshape((ping_num, -1, 4))
                b_r_tmp[tx] = np.real(reshaped)
                b_i_tmp[tx] = np.imag(reshaped)
                max_len = b_r_tmp[tx].shape[1] if b_r_tmp[tx].shape[1] > max_len else max_len
            else:
                b_r_tmp[tx] = np.array(self.power_dict[tx], dtype='float32')
                max_len = b_r_tmp[tx].shape[1] if b_r_tmp[tx].shape[1] > max_len else max_len

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
        beam_dict['slope'] = []
        beam_dict['backscatter_r'] = []
        beam_dict['backscatter_i'] = []
        beam_dict['angle_dict'] = []
        c_seq = 0
        for k, c in self.config_datagram['configuration'].items():
            if k not in ch_ids:
                continue
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
            beam_dict['slope'].append(self.parameters[k]['slope'])

            # Pad each channel with nan so that they can be stacked
            # Broadband
            if bb:
                diff = max_len - b_r_tmp[k].shape[1]
                beam_dict['backscatter_r'].append(np.pad(b_r_tmp[k], ((0, 0), (0, diff), (0, 0)),
                                                  mode='constant', constant_values=np.nan))
                beam_dict['backscatter_i'].append(np.pad(b_i_tmp[k], ((0, 0), (0, diff), (0, 0)),
                                                  mode='constant', constant_values=np.nan))
                beam_dict['frequency_start'].append(self.parameters[k]['frequency_start'])
                beam_dict['frequency_end'].append(self.parameters[k]['frequency_end'])
            # Continuous wave
            else:
                diff = max_len - b_r_tmp[k].shape[1]
                beam_dict['backscatter_r'].append(np.pad(b_r_tmp[k], ((0, 0), (0, diff)),
                                                         mode='constant', constant_values=np.nan))
                beam_dict['angle_dict'].append(np.pad(np.array(self.angle_dict[k], dtype='float32'),
                                                      ((0, 0), (0, diff), (0, 0)),
                                                      mode='constant', constant_values=np.nan))
            c_seq += 1

        # Stack channels and order axis as: channel, quadrant, ping, range
        if bb:
            beam_dict['backscatter_r'] = np.moveaxis(np.stack(beam_dict['backscatter_r']), 3, 1)
            beam_dict['backscatter_i'] = np.moveaxis(np.stack(beam_dict['backscatter_i']), 3, 1)
            beam_dict['frequency_start'] = np.unique(beam_dict['frequency_start'])
            beam_dict['frequency_end'] = np.unique(beam_dict['frequency_end'])
            beam_dict['frequency_center'] = (beam_dict['frequency_start'] + beam_dict['frequency_end']) / 2
        else:
            beam_dict['backscatter_r'] = np.stack(beam_dict['backscatter_r'])
            beam_dict['angle_dict'] = np.stack(beam_dict['angle_dict'])
        beam_dict['range_bin'] = np.arange(max_len)
        beam_dict['beam_width'] = bm_width
        beam_dict['beam_direction'] = bm_dir
        beam_dict['beam_angle'] = bm_angle
        beam_dict['transducer_position'] = tx_pos

        # Loop through each transducer for variables that may vary at each ping
        # -- this rarely is the case for EK60 so we check first before saving
        pl_tmp = np.unique(self.parameters[ch_ids[0]]['pulse_duration']).size
        pw_tmp = np.unique(self.parameters[ch_ids[0]]['transmit_power']).size
        # bw_tmp = np.unique(self.ping_data_dict[1]['bandwidth']).size      # Not in EK80
        si_tmp = np.unique(self.parameters[ch_ids[0]]['sample_interval']).size
        if np.all(np.array([pl_tmp, pw_tmp, si_tmp]) == 1):
            tx_sig = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
            beam_dict['sample_interval'] = np.zeros(shape=(tx_num,), dtype='float32')
            for t_seq in range(tx_num):
                tx_sig['transmit_duration_nominal'][t_seq] = \
                    np.float32(self.parameters[ch_ids[t_seq]]['pulse_duration'][0])
                tx_sig['transmit_power'][t_seq] = \
                    np.float32(self.parameters[ch_ids[t_seq]]['transmit_power'][0])
                # tx_sig['transmit_bandwidth'][t_seq] = \
                #     np.float32((self.parameters[self.ch_ids[t_seq]]['bandwidth'][0])
                beam_dict['sample_interval'][t_seq] = \
                    np.float32(self.parameters[ch_ids[t_seq]]['sample_interval'][0])
        else:
            tx_sig = defaultdict(lambda: np.zeros(shape=(tx_num, ping_num), dtype='float32'))
            beam_dict['sample_interval'] = np.zeros(shape=(tx_num, ping_num), dtype='float32')
            for t_seq in range(tx_num):
                tx_sig['transmit_duration_nominal'][t_seq, :] = \
                    np.array(self.parameters[ch_ids[t_seq]]['pulse_duration'], dtype='float32')
                tx_sig['transmit_power'][t_seq, :] = \
                    np.array(self.parameters[ch_ids[t_seq]]['transmit_power'], dtype='float32')
                # tx_sig['transmit_bandwidth'][t_seq, :] = \
                #     np.array(self.parameters[self.ch_ids[t_seq]]['bandwidth'], dtype='float32')
                beam_dict['sample_interval'][t_seq, :] = \
                    np.array(self.parameters[ch_ids[t_seq]]['sample_interval'], dtype='float32')

        beam_dict['transmit_signal'] = tx_sig
        # Build other parameters
        # beam_dict['non_quantitative_processing'] = np.array([0, ] * freq.size, dtype='int32')
        # -- sample_time_offset is set to 2 for EK60 data, this value is NOT from sample_data['offset']
        # beam_dict['sample_time_offset'] = np.array([2, ] * freq.size, dtype='int32')
        pulse_length = 'pulse_duration_fm' if bb else 'pulse_duration'
        # Gets indices from pulse length table using the transmit_duration_nominal values selected
        idx = [np.argwhere(np.isclose(tx_sig['transmit_duration_nominal'][i],
                                      self.config_datagram['configuration'][ch][pulse_length])).squeeze()
               for i, ch in enumerate(ch_ids)]
        # Use the indices to select sa_correction values from the sa correction table
        beam_dict['sa_correction'] = \
            np.array([x['sa_correction'][y]
                     for x, y in zip(self.config_datagram['configuration'].values(), np.array(idx))])

        return beam_dict

    def _set_vendor_dict(self):
        out_dict = dict()
        out_dict['ch_ids'] = self.ch_ids
        coeffs = dict()
        decimation_factors = dict()
        for ch in self.ch_ids:
            # Coefficients for wide band transceiver
            coeffs[f'{ch}_WBT_filter'] = self.fil_coeffs[ch][1]
            # Coefficients for pulse compression
            coeffs[f'{ch}_PC_filter'] = self.fil_coeffs[ch][2]
            decimation_factors[f'{ch}_WBT_decimation'] = self.fil_df[ch][1]
            decimation_factors[f'{ch}_PC_decimation'] = self.fil_df[ch][2]
        out_dict['filter_coefficients'] = coeffs
        out_dict['decimation_factors'] = decimation_factors

        return out_dict

    def _set_groups(self, raw_file, out_file, save_settings):
        grp = SetGroups(file_path=out_file, echo_type='EK80',
                        compress=save_settings['compress'], append_zarr=self._append_zarr)
        grp.set_toplevel(self._set_toplevel_dict(raw_file))  # top-level group
        grp.set_env(self._set_env_dict())            # environment group
        grp.set_provenance(self._set_prov_dict(raw_file, save_settings['combine_opt']))    # provenance group
        grp.set_platform(self._set_platform_dict())  # platform group
        grp.set_nmea(self._set_nmea_dict())          # platform/NMEA group
        grp.set_vendor(self._set_vendor_dict())      # vendor group
        """Handles saving the beam and sonar group. These groups a frequency dimension
        Splits up broadband and continuous wave data into separate files"""
        bb_ch_ids, cw_ch_ids = self.sort_ch_ids()
        # If there is both bb and cw data
        if bb_ch_ids and cw_ch_ids:
            # Copy the current file into a new file with _cw appended to filename
            split = os.path.splitext(out_file)
            new_path = split[0] + '_cw' + split[1]
            # Do not create _cw file if appendng because it already exists
            if not self._append_zarr:
                if split[1] == '.zarr':
                    shutil.copytree(out_file, new_path)
                elif split[1] == '.nc':
                    shutil.copyfile(out_file, new_path)
            grp.set_beam(self._set_beam_dict(bb_ch_ids, bb=True, path=out_file))
            grp.set_sonar(self._set_sonar_dict(bb_ch_ids, path=out_file))
            grp.set_beam(self._set_beam_dict(cw_ch_ids, bb=False, path=new_path))
            grp.set_sonar(self._set_sonar_dict(cw_ch_ids, path=new_path))
        # If there is only bb data
        elif bb_ch_ids:
            grp.set_beam(self._set_beam_dict(bb_ch_ids, bb=True, path=out_file))
            grp.set_sonar(self._set_sonar_dict(bb_ch_ids, path=out_file))
        # If there is only cw data
        else:
            grp.set_beam(self._set_beam_dict(cw_ch_ids, bb=False, path=out_file))
            grp.set_sonar(self._set_sonar_dict(cw_ch_ids, path=out_file))

    def _export_nc(self, save_settings, file_idx=0):
        if self._temp_path:
            out_file = self._temp_path[file_idx]
        else:
            # If there are multiple files, self.save_path is a list otherwise it is a string
            out_file = self.save_path[file_idx] if type(self.save_path) == list else self.save_path
        raw_file = self.filename[file_idx]

        # Check if nc file already exists and deletes it if overwrite is true
        if os.path.exists(out_file) and save_settings['overwrite']:
            print("          overwriting: " + out_file)
            os.remove(out_file)
        # Remove _cw file if present
        split = os.path.splitext(out_file)
        cw_path = split[0] + '_cw' + split[1]
        if os.path.exists(cw_path) and save_settings['overwrite'] and not self._append_zarr:
            print("          overwriting: " + cw_path)
            os.remove(cw_path)

        if os.path.exists(out_file):
            print(f'          ... this file has already been converted to .nc, conversion not executed.')
        else:
            self._set_groups(raw_file, out_file, save_settings)

    def _export_zarr(self, save_settings, file_idx=0):
        """
        Save parsed raw files to Zarr.
        Zarr files can be appened to so combining multiple raw files into 1 Zarr file can be done
        without creating temporary files
        """
        out_file = self.save_path[file_idx] if type(self.save_path) == list else self.save_path
        raw_file = self.filename[file_idx]

        if os.path.exists(out_file) and save_settings['overwrite'] and not self._append_zarr:
            print("          overwriting: " + out_file)
            shutil.rmtree(out_file)
        # Remove _cw file if present
        split = os.path.splitext(out_file)
        cw_path = split[0] + '_cw' + split[1]
        if os.path.exists(cw_path) and save_settings['overwrite'] and not self._append_zarr:
            print("          overwriting: " + cw_path)
            shutil.rmtree(cw_path)
        # Check if zarr file already exists
        # ... if yes, abort conversion and issue warning
        # ... if not, continue with conversion
        if os.path.exists(out_file) and not self._append_zarr:
            print(f'          ... this file has already been converted to .zarr, conversion not executed.')
        else:
            self._set_groups(raw_file, out_file, save_settings=save_settings)

    def _combine_files(self):
        # Do nothing if combine_opt is true if there is nothing to combine
        if not self._temp_path:
            return
        save_path = self.save_path
        split = os.path.splitext(self.save_path)
        all_temp = os.listdir(self._temp_dir)
        # Group files into cw (index 0) and broadband files (index 1)
        file_groups = [[], []]
        for f in all_temp:
            if "_cw" in f:
                file_groups[0].append(os.path.join(self._temp_dir, f))
            else:
                file_groups[1].append(os.path.join(self._temp_dir, f))

        for n, file_group in enumerate(file_groups):
            if len(file_groups) > 1:
                if not file_groups[n]:
                    # Skip saving either bb or cw if only one or the other is present
                    continue
                save_path = split[0] + '_cw' + split[1] if n == 0 else self.save_path
            # Open multiple files as one dataset of each group and save them into a single file
            with xr.open_dataset(file_group[0], group='Provenance') as ds_prov:
                ds_prov.to_netcdf(path=save_path, mode='w', group='Provenance')
            with xr.open_dataset(file_group[0], group='Sonar') as ds_sonar:
                ds_sonar.to_netcdf(path=save_path, mode='a', group='Sonar')
            with xr.open_mfdataset(file_group, group='Beam', combine='by_coords') as ds_beam:
                ds_beam.to_netcdf(path=save_path, mode='a', group='Beam')
            with xr.open_dataset(file_group[0], group='Environment') as ds_env:
                ds_env.to_netcdf(path=save_path, mode='a', group='Environment')
            with xr.open_mfdataset(file_group, group='Platform', combine='by_coords') as ds_plat:
                ds_plat.to_netcdf(path=save_path, mode='a', group='Platform')
            with xr.open_mfdataset(file_group, group='Platform/NMEA',
                                   combine='nested', concat_dim='time', decode_times=False) as ds_nmea:
                ds_nmea.to_netcdf(path=save_path, mode='a', group='Platform/NMEA')

        # Delete temporary folder:
        shutil.rmtree(self._temp_dir)

    def save(self, file_format, save_path=None, combine_opt=False, overwrite=False, compress=True):
        """Save data from EK60 `.raw` to netCDF format.
        """
        save_settings = dict(combine_opt=combine_opt, overwrite=overwrite, compress=compress)
        self.validate_path(save_path, file_format, combine_opt)
        # Loop over all files being parsed
        for file_idx, file in enumerate(self.filename):
            # Reset instance variables for each raw file. Always reset if there is more than 1 file being parsed
            if file_idx > 0 or len(self.filename) > 1:
                self.reset_vars('EK80')
            # Load data if it has not already been loaded.
            if self.config_datagram is None:
                self.load_ek80_raw(file)
            # multiple raw files are saved differently between the .nc and .zarr formats
            if file_format == '.nc':
                self._export_nc(save_settings, file_idx)
            elif file_format == '.zarr':
                # Sets flag for combining raw files into 1 zarr file
                self._append_zarr = True if file_idx and combine_opt else False
                self._export_zarr(save_settings, file_idx)
        if combine_opt and file_format == '.nc':
            self._combine_files()
