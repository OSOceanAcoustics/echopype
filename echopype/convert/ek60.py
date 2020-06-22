"""
Functions to unpack Simrad EK60 .raw data file and save to .nc.
"""


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


# Create a constant to convert indexed power to power.
INDEX2POWER = (10.0 * np.log10(2.0) / 256.0)

# Create a constant to convert from indexed angles to electrical angles.
INDEX2ELEC = 180.0 / 128.0


class ConvertEK60(ConvertBase):
    """Class for converting EK60 ``.raw`` files.
    """
    def __init__(self, _filename=''):
        ConvertBase.__init__(self)
        self.filename = _filename  # path to EK60 .raw filename to be parsed

        # Initialize file parsing storage variables
        self.config_datagram = None
        self.nmea_data = NMEAData()  # object for NMEA data
        self.ping_data_dict = {}   # dictionary to store metadata
        self.power_dict = {}   # dictionary to store power data
        self.angle_dict = {}   # dictionary to store angle data
        self.ping_time = []    # list to store ping time
        self.CON1_datagram = None    # storage for CON1 datagram for ME70

        # Variables only used in EK60 parsing
        self.range_lengths = None    # number of range_bin groups
        self.ping_time_split = {}    # dictionaries to store variables of each range_bin groups (if there are multiple)
        self.power_dict_split = {}
        self.angle_dict_split = {}
        self.tx_sig = {}   # dictionary to store transmit signal parameters and sample interval
        self.ping_slices = []

    def _append_channel_ping_data(self, ch_num, datagram):
        """ Append ping-by-ping channel metadata extracted from the newly read datagram of type 'RAW'.

        Parameters
        ----------
        ch_num : int
            number of the channel to append metadata to
        datagram : dict
            the newly read datagram of type 'RAW'
        """
        self.ping_data_dict[ch_num]['mode'].append(datagram['mode'])
        self.ping_data_dict[ch_num]['transducer_depth'].append(datagram['transducer_depth'])
        self.ping_data_dict[ch_num]['transmit_power'].append(datagram['transmit_power'])
        self.ping_data_dict[ch_num]['pulse_length'].append(datagram['pulse_length'])
        self.ping_data_dict[ch_num]['bandwidth'].append(datagram['bandwidth'])
        self.ping_data_dict[ch_num]['sample_interval'].append(datagram['sample_interval'])
        self.ping_data_dict[ch_num]['sound_velocity'].append(datagram['sound_velocity'])
        self.ping_data_dict[ch_num]['absorption_coefficient'].append(datagram['absorption_coefficient'])
        self.ping_data_dict[ch_num]['heave'].append(datagram['heave'])
        self.ping_data_dict[ch_num]['roll'].append(datagram['roll'])
        self.ping_data_dict[ch_num]['pitch'].append(datagram['pitch'])
        self.ping_data_dict[ch_num]['temperature'].append(datagram['temperature'])
        self.ping_data_dict[ch_num]['heading'].append(datagram['heading'])

    def _read_datagrams(self, fid):
        """
        Read various datagrams until the end of a ``.raw`` file.

        Only includes code for storing RAW and NMEA datagrams and
        ignoring the TAG, BOT, and DEP datagrams.

        Parameters
        ----------
        fid
            a RawSimradFile file object opened in ``self.load_ek60_raw()``
        """
        num_datagrams_parsed = 0
        tmp_num_ch_per_ping_parsed = 0  # number of channels of the same ping parsed
                                        # this is used to control saving only pings
                                        # that have all freq channels present
        tmp_datagram_dict = []  # tmp list of datagrams, only saved to actual output
                                # structure if data from all freq channels are present

        while True:
            try:
                new_datagram = fid.read(1)
            except SimradEOF:
                break

            # Convert the timestamp to a datetime64 object.
            new_datagram['timestamp'] = np.datetime64(new_datagram['timestamp'].replace(tzinfo=None), '[ms]')

            num_datagrams_parsed += 1

            # RAW datagrams store raw acoustic data for a channel
            if new_datagram['type'].startswith('RAW'):
                curr_ch_num = new_datagram['channel']

                # Reset counter and storage for parsed number of channels
                # if encountering datagram from the first channel
                if curr_ch_num == 1:
                    tmp_num_ch_per_ping_parsed = 0
                    tmp_datagram_dict = []

                # Save datagram temporarily before knowing if all freq channels are present
                tmp_num_ch_per_ping_parsed += 1
                tmp_datagram_dict.append(new_datagram)

                # Actually save datagram when all freq channels are present
                if np.all(np.array([curr_ch_num, tmp_num_ch_per_ping_parsed])
                          == self.config_datagram['transceiver_count']):

                    # append ping time from first channel
                    self.ping_time.append(tmp_datagram_dict[0]['timestamp'])

                    for ch_seq in range(self.config_datagram['transceiver_count']):
                        # If frequency matches for this channel, actually store data
                        # Note all storage structure indices are 1-based since they are indexed by
                        # the channel number as stored in config_datagram['transceivers'].keys()
                        if self.config_datagram['transceivers'][ch_seq+1]['frequency'] \
                                == tmp_datagram_dict[ch_seq]['frequency']:
                            self._append_channel_ping_data(ch_seq+1, tmp_datagram_dict[ch_seq])  # ping-by-ping metadata
                            self.power_dict[ch_seq+1].append(tmp_datagram_dict[ch_seq]['power'])  # append power data
                            self.angle_dict[ch_seq+1].append(tmp_datagram_dict[ch_seq]['angle'])  # append angle data
                        else:
                            # TODO: need error-handling code here
                            print('Frequency mismatch for data from the same channel number!')

            # NME datagrams store ancillary data as NMEA-0817 style ASCII data.
            elif new_datagram['type'].startswith('NME'):
                # Add the datagram to our nmea_data object.
                self.nmea_data.add_datagram(new_datagram['timestamp'],
                                            new_datagram['nmea_string'])

            # TAG datagrams contain time-stamped annotations inserted via the recording software
            elif new_datagram['type'].startswith('TAG'):
                print('TAG datagram encountered.')

            # BOT datagrams contain sounder detected bottom depths from .bot files
            elif new_datagram['type'].startswith('BOT'):
                print('BOT datagram encountered.')

            # DEP datagrams contain sounder detected bottom depths from .out files
            # as well as reflectivity data
            elif new_datagram['type'].startswith('DEP'):
                print('DEP datagram encountered.')

            else:
                print("Unknown datagram type: " + str(new_datagram['type']))

    def split_by_range_group(self):
        """Split ping_time, power_dict, angle_dict, tx_sig by range_group.

        This is to deal with cases when there is a switch of range_bin size in the middle of the file.
        """
        # Find out the number of range_bin groups in power data
        # since there are files with a clear switch of length of range_bin in the middle
        range_bin_lens = [len(l) for l in self.power_dict[1]]
        uni, uni_inv, uni_cnt = np.unique(range_bin_lens, return_inverse=True, return_counts=True)

        # Initialize dictionaries. keys are index for ranges. values are dictionaries with keys for each freq
        uni_cnt_insert = np.cumsum(np.insert(uni_cnt, 0, 0))
        beam_type = np.array([x['beam_type'] for x in self.config_datagram['transceivers'].values()])
        for range_group in range(len(uni)):
            self.ping_time_split[range_group] = np.array(self.ping_time)[uni_cnt_insert[range_group]:
                                                                         uni_cnt_insert[range_group+1]]
            range_bin_freq_lens = np.unique(
                [x_val[uni_cnt_insert[range_group]].shape for x_val in self.power_dict.values()])
            self.angle_dict_split[range_group] = np.empty(
                (len(self.power_dict), uni_cnt_insert[range_group + 1] - uni_cnt_insert[range_group],
                 range_bin_freq_lens.max(), 2))
            self.angle_dict_split[range_group][:] = np.nan
            if len(range_bin_freq_lens) != 1:  # different frequency channels have different range_bin lengths
                tmp_power_pad, tmp_angle_pad = [], []
                for x_p, x_a in zip(self.power_dict.values(), self.angle_dict.values()):  # pad nan to shorter channels
                    tmp_p_data = np.array(x_p[uni_cnt_insert[range_group]:uni_cnt_insert[range_group + 1]])
                    tmp_a_data = np.array(x_a[uni_cnt_insert[range_group]:uni_cnt_insert[range_group + 1]])
                    tmp_power = np.pad(tmp_p_data.astype('float64'),
                                       ((0, 0), (0, range_bin_freq_lens.max()-tmp_p_data.shape[1])),
                                       mode='constant', constant_values=(np.nan,))
                    tmp_angle = np.pad(tmp_a_data.astype('float64'),
                                       ((0, 0), (0, range_bin_freq_lens.max()-tmp_a_data.shape[1]), (0, 0)),
                                       mode='constant', constant_values=(np.nan,))
                    tmp_power_pad.append(tmp_power)
                    tmp_angle_pad.append(tmp_angle)
                self.angle_dict_split[range_group] = np.array(tmp_angle_pad)
                self.power_dict_split[range_group] = np.array(tmp_power_pad) * INDEX2POWER
            else:
                self.power_dict_split[range_group] = np.array(
                    [x[uni_cnt_insert[range_group]:uni_cnt_insert[range_group + 1]]
                     for x_key, x in self.power_dict.items()]) * INDEX2POWER
                for ch in np.argwhere(beam_type == 1):   # if split-beam
                    self.angle_dict_split[range_group][ch, :, :, :] = np.array(
                        self.angle_dict[ch[0]+1][uni_cnt_insert[range_group]:uni_cnt_insert[range_group + 1]])
            self.tx_sig[range_group] = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))

        pulse_length, transmit_power, bandwidth, sample_interval = [], [], [], []
        param = [pulse_length, transmit_power, bandwidth, sample_interval]
        param_name = ['pulse_length', 'transmit_power', 'bandwidth', 'sample_interval']
        param_name_save = ['transmit_duration_nominal', 'transmit_power', 'transmit_bandwidth', 'sample_interval']
        for range_group in range(len(uni)):
            for p, pname in zip(param, param_name):
                p.append(np.array([np.array(
                    self.ping_data_dict[x][pname][uni_cnt_insert[range_group]:uni_cnt_insert[range_group + 1]])
                    for x in self.config_datagram['transceivers'].keys()]))
        tx_num = self.config_datagram['transceiver_count']  # number of transceivers
        for range_group in range(len(uni)):
            for p, pname, pname_save in zip(param, param_name, param_name_save):
                if np.unique(p[range_group], axis=1).size != tx_num:
                    # TODO: right now set_groups_ek60/set_beam doens't deal with this case, need to add
                    ValueError('%s changed in the middle of range_bin group' % pname)
                else:
                    self.tx_sig[range_group][pname_save] = np.unique(p[range_group], axis=1).squeeze(axis=1)

        self.range_lengths = uni  # used in looping when saving files with different range_bin numbers

    def load_ek60_raw(self, raw):
        """Method to parse the EK60 ``.raw`` data file.

        This method parses the ``.raw`` file and saves the parsed data
        to the ConvertEK60 instance.

        Parameters
        ----------
        raw : string
            raw filename
        """
        print('%s  converting file: %s' % (dt.now().strftime('%H:%M:%S'), os.path.basename(raw)))

        with RawSimradFile(raw, 'r') as fid:
            # Read the CON0 configuration datagram. Only keep 1 if multiple files
            if self.config_datagram is None:
                self.config_datagram = fid.read(1)
                self.config_datagram['timestamp'] = np.datetime64(
                    self.config_datagram['timestamp'].replace(tzinfo=None), '[ms]')

                for ch_num in self.config_datagram['transceivers'].keys():
                    self.ping_data_dict[ch_num] = defaultdict(list)
                    self.ping_data_dict[ch_num]['frequency'] = \
                        self.config_datagram['transceivers'][ch_num]['frequency']
                    self.power_dict[ch_num] = []
                    self.angle_dict[ch_num] = []
            else:
                tmp_config = fid.read(1)

            # Check if reading an ME70 file with a CON1 datagram.
            next_datagram = fid.peek()
            if next_datagram == 'CON1':
                self.CON1_datagram = fid.read(1)
            else:
                self.CON1_datagram = None

            # Read the rest of datagrams
            self._read_datagrams(fid)

        # Split data based on range_group (when there is a switch of range_bin in the middle of a file)
        self.split_by_range_group()

        # Trim excess data from NMEA object
        self.nmea_data.trim()

    # Functions to set various dictionaries
    def _set_toplevel_dict(self, raw_file):
        # filename must have "-" as the field separator for the last 2 fields. Uses first file
        filename_tup = os.path.splitext(os.path.basename(raw_file))[0].split("-")
        filedate = filename_tup[len(filename_tup) - 2].replace("D", "")
        filetime = filename_tup[len(filename_tup) - 1].replace("T", "")
        out_dict = dict(Conventions='CF-1.7, SONAR-netCDF4, ACDD-1.3',
                        keywords='EK60',
                        sonar_convention_authority='ICES',
                        sonar_convention_name='SONAR-netCDF4',
                        sonar_convention_version='1.7',
                        summary='',
                        title='')
        out_dict['date_created'] = dt.strptime(filedate + '-' + filetime, '%Y%m%d-%H%M%S').isoformat() + 'Z'
        return out_dict

    def _set_env_dict(self):
        freq = np.array([self.config_datagram['transceivers'][x]['frequency']
                            for x in self.config_datagram['transceivers'].keys()], dtype='float32')
        # Extract absorption and sound speed depending on if the values are identical for all pings
        abs_tmp = np.unique(self.ping_data_dict[1]['absorption_coefficient']).size
        ss_tmp = np.unique(self.ping_data_dict[1]['sound_velocity']).size
        # --- if identical for all pings, save only values from the first ping
        if np.all(np.array([abs_tmp, ss_tmp]) == 1):
            abs_val = np.array([self.ping_data_dict[x]['absorption_coefficient'][0]
                                for x in self.config_datagram['transceivers'].keys()], dtype='float32')
            ss_val = np.array([self.ping_data_dict[x]['sound_velocity'][0]
                                for x in self.config_datagram['transceivers'].keys()], dtype='float32')
        # --- if NOT identical for all pings, save as array of dimension [frequency x ping_time]
        else:  # TODO: right now set_groups_ek60/set_env doens't deal with this case, need to add
            abs_val = np.array([self.ping_data_dict[x]['absorption_coefficient']
                                for x in self.config_datagram['transceivers'].keys()],
                                dtype='float32')
            ss_val = np.array([self.ping_data_dict[x]['sound_velocity']
                                for x in self.config_datagram['transceivers'].keys()],
                                dtype='float32')

        return dict(frequency=freq,
                    absorption_coeff=abs_val,
                    sound_speed=ss_val)

    def _set_prov_dict(self, raw_file, combine_opt):
        out_dict = dict(conversion_software_name='echopype',
                        conversion_software_version=ECHOPYPE_VERSION,
                        conversion_time=dt.now(tz=pytz.utc).isoformat(timespec='seconds'))  # use UTC time
        # Send a list of all filenames if combining raw files. Else, send the one file to be converted
        out_dict['src_filenames'] = self.filename if combine_opt else [raw_file]
        return out_dict

    def _set_sonar_dict(self):
        return dict(sonar_manufacturer='Simrad',
                    sonar_model=self.config_datagram['sounder_name'],
                    sonar_serial_number='',
                    sonar_software_name='',
                    sonar_software_version=self.config_datagram['version'],
                    sonar_type='echosounder')

    def _set_platform_dict(self, out_file=None, piece_seq=0):
        out_dict = dict()
        # TODO: Need to reconcile the logic between using the unpacked "survey_name"
        #  and the user-supplied platform_name
        # self.platform_name = self.config_datagram['survey_name']
        out_dict['platform_name'] = self.platform_name
        out_dict['platform_type'] = self.platform_type
        out_dict['platform_code_ICES'] = self.platform_code_ICES

        # Read pitch/roll/heave from ping data
        # [seconds since 1900-01-01] for xarray.to_netcdf conversion
        out_dict['ping_time'] = self.ping_time
        out_dict['pitch'] = np.array(self.ping_data_dict[1]['pitch'], dtype='float32')
        out_dict['roll'] = np.array(self.ping_data_dict[1]['roll'], dtype='float32')
        out_dict['heave'] = np.array(self.ping_data_dict[1]['heave'], dtype='float32')
        # water_level is set to 0 for EK60 since this is not separately recorded
        # and is part of transducer_depth
        out_dict['water_level'] = np.int32(0)

        # Read lat/long from NMEA datagram
        idx_loc = np.argwhere(np.isin(self.nmea_data.messages, ['GGA', 'GLL', 'RMC'])).squeeze()
        # TODO: use NaN when nmea_msg is empty
        nmea_msg = []
        [nmea_msg.append(pynmea2.parse(self.nmea_data.raw_datagrams[x])) for x in idx_loc]
        out_dict['lat'] = np.array([x.latitude for x in nmea_msg])
        out_dict['lon'] = np.array([x.longitude for x in nmea_msg])
        out_dict['location_time'] = self.nmea_data.nmea_times[idx_loc]

        if len(self.range_lengths) > 1:
            out_dict['path'] = self.all_files[piece_seq]
            out_dict['ping_slice'] = self.ping_time_split[piece_seq]
            out_dict['overwrite_plat'] = True if piece_seq == 1 else False
        else:
            out_dict['path'] = out_file
            out_dict['overwrite_plat'] = False
        return out_dict

    def _set_nmea_dict(self, out_file=None, piece_seq=0):
        # Assemble dict for saving to groups
        out_dict = dict()
        out_dict['ping_time'] = self.ping_time
        out_dict['nmea_time'] = self.nmea_data.nmea_times
        out_dict['nmea_datagram'] = self.nmea_data.raw_datagrams

        if len(self.range_lengths) > 1:
            out_dict['path'] = self.all_files[piece_seq]
            out_dict['ping_slice'] = self.ping_time_split[piece_seq]
            out_dict['overwrite_plat'] = True if piece_seq == 1 else False
        else:
            out_dict['path'] = out_file
            out_dict['overwrite_plat'] = False
        return out_dict

    def _set_beam_dict(self, out_file=None, piece_seq=0):
        beam_dict = dict()
        beam_dict['beam_mode'] = 'vertical'
        beam_dict['conversion_equation_t'] = 'type_3'  # type_3 is EK60 conversion
        beam_dict['ping_time'] = self.ping_time_split[piece_seq]   # [seconds since 1900-01-01] for xarray.to_netcdf conversion
        beam_dict['backscatter_r'] = self.power_dict_split[piece_seq]  # dimension [freq x ping_time x range_bin]
        beam_dict['angle_dict'] = self.angle_dict_split[piece_seq]

        # Additional coordinate variables added by echopype for storing data as a cube with
        # dimensions [frequency x ping_time x range_bin]
        freq = np.array([self.config_datagram['transceivers'][x]['frequency']
                            for x in self.config_datagram['transceivers'].keys()], dtype='float32')
        beam_dict['frequency'] = freq
        beam_dict['range_bin'] = np.arange(self.power_dict_split[piece_seq].shape[2])

        # Loop through each transducer for channel-specific variables
        param_numerical = {"beamwidth_receive_major": "beamwidth_alongship",
                        "beamwidth_receive_minor": "beamwidth_athwartship",
                        "beamwidth_transmit_major": "beamwidth_alongship",
                        "beamwidth_transmit_minor": "beamwidth_athwartship",
                        "beam_direction_x": "dir_x",
                        "beam_direction_y": "dir_y",
                        "beam_direction_z": "dir_z",
                        "angle_offset_alongship": "angle_offset_alongship",
                        "angle_offset_athwartship": "angle_offset_athwartship",
                        "angle_sensitivity_alongship": "angle_sensitivity_alongship",
                        "angle_sensitivity_athwartship": "angle_sensitivity_athwartship",
                        "transducer_offset_x": "pos_x",
                        "transducer_offset_y": "pos_y",
                        "transducer_offset_z": "pos_z",
                        "equivalent_beam_angle": "equivalent_beam_angle",
                        "gain_correction": "gain"}
        param_str = {"gpt_software_version": "gpt_software_version",
                    "channel_id": "channel_id",
                    "beam_type": "beam_type"}

        for encode_name, origin_name in param_numerical.items():
            beam_dict[encode_name] = np.array(
                [val[origin_name] for key, val in self.config_datagram['transceivers'].items()]).astype('float32')
        beam_dict['transducer_offset_z'] += [self.ping_data_dict[x]['transducer_depth'][0]
                                            for x in self.config_datagram['transceivers'].keys()]

        for encode_name, origin_name in param_str.items():
            beam_dict[encode_name] = [val[origin_name]
                                    for key, val in self.config_datagram['transceivers'].items()]

        beam_dict['transmit_signal'] = self.tx_sig[piece_seq]  # only this range_bin group

        # Build other parameters
        beam_dict['non_quantitative_processing'] = np.array([0, ] * freq.size, dtype='int32')
        # -- sample_time_offset is set to 2 for EK60 data, this value is NOT from sample_data['offset']
        beam_dict['sample_time_offset'] = np.array([2, ] * freq.size, dtype='int32')

        if len(self.config_datagram['transceivers']) == 1:   # only 1 channel
            idx = np.argwhere(np.isclose(self.tx_sig[piece_seq]['transmit_duration_nominal'],
                                        self.config_datagram['transceivers'][1]['pulse_length_table'])).squeeze()
            idx = np.expand_dims(np.array(idx), axis=0)
        else:
            idx = [np.argwhere(np.isclose(self.tx_sig[piece_seq]['transmit_duration_nominal'][key - 1],
                                        val['pulse_length_table'])).squeeze()
                for key, val in self.config_datagram['transceivers'].items()]
        beam_dict['sa_correction'] = \
            np.array([x['sa_correction_table'][y]
                    for x, y in zip(self.config_datagram['transceivers'].values(), np.array(idx))])

        # New path created if the power data is broken up due to varying range bins
        if len(self.range_lengths) > 1:
            beam_dict['path'] = self.all_files[piece_seq]
            beam_dict['overwrite_beam'] = True if piece_seq == 1 else False
        else:
            beam_dict['path'] = out_file
            beam_dict['overwrite_beam'] = False

        return beam_dict

    def copyfiles(self, out_file, overwrite=False):
        """
        Creates a duplicate file so that parsed data can be split into multiple files
        Used for when the length of range changes over time
        """
        # Avoid output filenames with _part02_part01 by using original filename
        out_file = self.save_path if self._use_original else out_file
        split = os.path.splitext(out_file)
        print("          splitting into: ")
        # Offsets the _part## depending on the number of parts already created
        n_offset = len(self._zarr_split) if self._zarr_split and self._use_original else 0
        for n in range(len(self.range_lengths)):
            n += n_offset
            new_path = split[0] + '_part%02d' % (n + 1) + split[1]
            self.all_files.append(new_path)     # Resets for every file
            self._zarr_split.append(new_path)   # Persists across files
            if os.path.exists(new_path) and overwrite:
                os.remove(new_path) 
            if n > 0:
                if split[1] == '.zarr':
                    # Handle splitting combined zarr files into more than 2 parts
                    if len(self._zarr_split) > 2 and self._use_original:
                        # Add previous file path to self.all_files in order to complete saving the beam and platform
                        self.all_files.insert(0, self._zarr_split[-2])
                        shutil.copytree(self._zarr_split[-2], new_path)
                        print("                " + new_path)
                        return
                    else:
                        shutil.copytree(out_file, new_path)
                elif split[1] == '.nc':
                    shutil.copyfile(out_file, new_path)
            print("                " + new_path)
        if os.path.exists(self.all_files[0]) and overwrite:
            os.remove(self.all_files[0]) 
        os.rename(out_file, self.all_files[0])

    def _set_groups(self, raw_file, out_file, save_settings):
        # Create SetGroups object
        grp = SetGroups(file_path=out_file, echo_type='EK60',
                        compress=save_settings['compress'], append_zarr=self._append_zarr)
        grp.set_toplevel(self._set_toplevel_dict(raw_file))  # top-level group
        grp.set_env(self._set_env_dict())            # environment group
        grp.set_provenance(self._set_prov_dict(raw_file, save_settings['combine_opt']))    # provenance group
        grp.set_sonar(self._set_sonar_dict())        # sonar group
        if len(self.range_lengths) > 1:
            self.copyfiles(out_file, save_settings['overwrite'])
        for piece in range(len(self.range_lengths)):
            grp.set_beam(self._set_beam_dict(out_file, piece_seq=piece))          # beam group
            grp.set_platform(self._set_platform_dict(out_file, piece_seq=piece))  # platform group
            grp.set_nmea(self._set_nmea_dict(out_file, piece_seq=piece))          # platform/NMEA group

    def _export_nc(self, save_settings, file_idx=0):
        """
        Saves parsed raw files to a NetCDF file.
        NetCDF files created by combining multiple raw files are saved to a temporary folder and
        merged after all raw file conversion has been completed due to limitations to appending
        to a NetCDF file with xarray.
        """
        # self._temp_path exists if combining multiple files into 1 .nc file
        if self._temp_path:
            out_file = self._temp_path[file_idx]
        else:
            # If there are multiple files, self.save_path is a list otherwise it is a string
            out_file = self.save_path[file_idx] if type(self.save_path) == list else self.save_path
        raw_file = self.filename[file_idx]

        # Check if out_file file already exists
        # Deletes it if overwrite is true
        if os.path.exists(out_file) and save_settings['overwrite']:
            print("          overwriting: " + out_file)
            os.remove(out_file)
        # Check if nc file already exists
        # ... if yes, abort conversion and issue warning
        # ... if not, continue with conversion
        if os.path.exists(out_file):
            print(f'          ... this file has already been converted to .nc, conversion not executed.')
        else:
            self._set_groups(raw_file, out_file, save_settings=save_settings)

    def _export_zarr(self, save_settings, file_idx=0):
        """
        Save parsed raw files to Zarr.
        Zarr files can be appened to so combining multiple raw files into 1 Zarr file can be done
        without creating temporary files
        """
        self._use_original = False
        if save_settings['combine_opt'] and self._zarr_split:
            out_file = self._zarr_split[-1]
            self._use_original = True
        elif type(self.save_path) == list:
            out_file = self.save_path[file_idx]
        else:
            out_file = self.save_path
        raw_file = self.filename[file_idx]
        if os.path.exists(out_file) and save_settings['overwrite'] and not self._append_zarr:
            print("          overwriting: " + out_file)
            shutil.rmtree(out_file)
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
        file_groups = [[]]
        # Split the files in the temp directory into range_bin groups
        i = 0
        while i < len(all_temp):
            file_groups[-1].append(os.path.join(self._temp_dir, all_temp[i]))
            if "_part" in all_temp[i]:
                i += 1
                file_groups.append([os.path.join(self._temp_dir, all_temp[i])])
            i += 1
        for n, file_group in enumerate(file_groups):
            if len(file_groups) > 1:
                # Construct a new path with _part[n] if there are multiple range_bin lengths
                save_path = split[0] + '_part%02d' % (n + 1) + split[1]
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
        """Save data from .raw format to a netCDF4 or Zarr file

        Parameters
        ----------
        file_format : str
            format of output file. ".nc" for netCDF4 or ".zarr" for Zarr
        save_path : str
            Path to save output to. Must be a directory if converting multiple files.
            Must be a filename if combining multiple files.
            If `False`, outputs in the same location as the input raw file.
        combine_opt : bool
            Whether or not to combine a list of input raw files.
            Raises error if combine_opt is true and there is only one file being converted.
        overwrite : bool
            Whether or not to overwrite the file if the output path already exists.
        compress : bool
            Whether or not to compress backscatter data. Defaults to `True`
            """
        save_settings = dict(combine_opt=combine_opt, overwrite=overwrite, compress=compress)
        self.validate_path(save_path, file_format, combine_opt)
        # Loop over all files being parsed
        for file_idx, file in enumerate(self.filename):
            # Reset instance variables for each raw file. Always reset if there is more than 1 file being parsed
            if file_idx > 0 or len(self.filename) > 1:
                self.reset_vars('EK60')
            # Load data if it has not already been loaded.
            if not self.power_dict:
                self.load_ek60_raw(file)
            # multiple raw files are saved differently between the .nc and .zarr formats
            if file_format == '.nc':
                self._export_nc(save_settings, file_idx)
            elif file_format == '.zarr':
                # Sets flag for combining raw files into 1 zarr file
                self._append_zarr = True if file_idx and combine_opt else False
                self._export_zarr(save_settings, file_idx)
        if combine_opt and file_format == '.nc':
            self._combine_files()
