import os
from collections import defaultdict
from datetime import datetime as dt
from typing import Any, Dict, Literal, Optional, Tuple

import dask
import dask.array as da
import numpy as np
import zarr
from dask.array.core import auto_chunks

from ..utils.io import create_temp_zarr_store
from ..utils.log import _init_logger
from .utils.ek_raw_io import RawSimradFile, SimradEOF
from .utils.ek_swap import calc_final_shapes

FILENAME_DATETIME_EK60 = (
    "(?P<survey>.+)?-?D(?P<date>\\w{1,8})-T(?P<time>\\w{1,6})-?(?P<postfix>\\w+)?.raw"
)

# Manufacturer-specific power conversion factor
INDEX2POWER = 10.0 * np.log10(2.0) / 256.0

logger = _init_logger(__name__)


class ParseBase:
    """Parent class for all convert classes."""

    def __init__(self, file, storage_options, sonar_model):
        self.source_file = file
        self.timestamp_pattern = None  # regex pattern used to grab datetime embedded in filename
        self.ping_time = []  # list to store ping time
        self.storage_options = storage_options
        self.sonar_model = sonar_model
        self.data_types = ["power", "angle", "complex"]
        self.raw_types = ["receive", "transmit"]

    def _print_status(self):
        """Prints message to console giving information about the raw file being parsed."""


class ParseEK(ParseBase):
    """Class for converting data from Simrad echosounders."""

    def __init__(self, file, bot_file, idx_file, storage_options, sonar_model):
        super().__init__(file, storage_options, sonar_model)
        # Parent class attributes
        #  regex pattern used to grab datetime embedded in filename
        self.timestamp_pattern = FILENAME_DATETIME_EK60

        # Class attributes
        self.bot_file = bot_file
        self.idx_file = idx_file
        self.config_datagram = None
        self.ping_data_dict = defaultdict(lambda: defaultdict(list))  # ping data
        self.ping_data_dict_tx = defaultdict(lambda: defaultdict(list))  # transmit ping data
        self.ping_time = defaultdict(list)  # store ping time according to channel
        self.num_range_sample_groups = None  # number of range_sample groups
        self.ch_ids = defaultdict(
            list
        )  # Stores the channel ids for each data type (power, angle, complex)

        self.nmea = defaultdict(list)  # Dictionary to store NMEA data(timestamp and string)
        self.mru0 = defaultdict(list)  # Dictionary to store MRU0 data (heading, pitch, roll, heave)
        self.mru1 = defaultdict(list)  # Dictionary to store MRU1 data (latitude, longitude)
        self.fil_coeffs = defaultdict(dict)  # Dictionary to store PC and WBT coefficients
        self.fil_df = defaultdict(dict)  # Dictionary to store filter decimation factors
        self.bot = defaultdict(list)  # Dictionary to store bottom depth values
        self.idx = defaultdict(list)  # Dictionary to store index file values

        self.CON1_datagram = None  # Holds the ME70 CON1 datagram

    def _print_status(self):
        time = dt.utcfromtimestamp(self.config_datagram["timestamp"].tolist() / 1e9).strftime(
            "%Y-%b-%d %H:%M:%S"
        )
        logger.info(
            f"parsing file {os.path.basename(self.source_file)}, " f"time of first ping: {time}"
        )

    @property
    def num_transducer_sectors(self) -> Dict[Any, int]:
        """Get the number of transducer sectors for each channel.
        This is for receive raw type only."""
        num_sectors = {}
        if self.ping_data_dict["n_complex"]:
            n_complex = self.ping_data_dict["n_complex"]
            for ch_id, n_complex_list in n_complex.items():
                num_transducer_sectors = np.unique(np.array(n_complex_list))
                if num_transducer_sectors.size > 1:  # this is not supposed to happen
                    raise ValueError("Transducer sector number changes in the middle of the file!")
                else:
                    num_transducer_sectors = num_transducer_sectors[0]

                num_sectors[ch_id] = int(num_transducer_sectors)
        return num_sectors

    def _get_data_shapes(self) -> dict:
        """Get all the expanded data shapes"""
        all_data_shapes = {}
        # Get all data type shapes
        for raw_type in self.raw_types:
            ping_data_dict = (
                self.ping_data_dict_tx if raw_type == "transmit" else self.ping_data_dict
            )
            # data_types: ["power", "angle", "complex"]
            data_type_shapes = calc_final_shapes(self.data_types, ping_data_dict)
            all_data_shapes[raw_type] = data_type_shapes

        return all_data_shapes

    def __should_use_swap(
        self, expanded_data_shapes: Dict[str, Any], mem_mult: float = 0.4
    ) -> bool:
        import sys

        import psutil

        # Calculate expansion and current data sizes
        total_req_mem = 0
        current_data_size = 0
        for raw_type, expanded_shapes in expanded_data_shapes.items():
            ping_data_dict = (
                self.ping_data_dict_tx if raw_type == "transmit" else self.ping_data_dict
            )
            for data_type, shape in expanded_shapes.items():
                if shape:
                    # Get current data size
                    size = sum([sys.getsizeof(val) for val in ping_data_dict[data_type].values()])
                    current_data_size += size

                    # Estimate expansion sizes
                    itemsize = np.dtype("float64").itemsize
                    req_mem = np.prod(shape) * itemsize
                    total_req_mem += req_mem

        # get statistics about system memory usage
        mem = psutil.virtual_memory()
        # approx. the amount of memory that will be used after expansion
        req_mem = mem.used - current_data_size + total_req_mem

        return mem.total * mem_mult < req_mem

    def rectangularize_data(
        self,
        use_swap: "bool | Literal['auto']" = "auto",
        max_chunk_size: str = "100MB",
    ) -> None:
        """
        Rectangularize the power, angle, and complex data.
        Additionally, convert the data to a numpy array
        indexed by channel.
        """
        # Compute the final expansion shapes for each data type
        expanded_data_shapes = self._get_data_shapes()

        # Determine use_swap
        if use_swap == "auto":
            use_swap = self.__should_use_swap(expanded_data_shapes)

        # Perform rectangularization
        zarr_root = None
        if use_swap:
            # Setup temp store
            zarr_store = create_temp_zarr_store()
            # Setup zarr store
            zarr_root = zarr.group(
                store=zarr_store, overwrite=True, synchronizer=zarr.ThreadSynchronizer()
            )

        for raw_type in self.raw_types:
            data_type_shapes = expanded_data_shapes[raw_type]
            for data_type in self.data_types:
                # Parse and pad the datagram
                self._parse_and_pad_datagram(
                    data_type=data_type,
                    data_type_shapes=data_type_shapes,
                    raw_type=raw_type,
                    use_swap=use_swap,
                    zarr_root=zarr_root,
                    max_chunk_size=max_chunk_size,
                )

    @staticmethod
    def _write_to_temp_zarr(
        arr: np.ndarray,
        zarr_root: zarr.Group,
        path: str,
        shape: Tuple[int],
        chunks: Tuple[int],
    ) -> dask.array.Array:
        if shape == arr.shape:
            z_arr = zarr_root.array(
                name=path,
                data=arr,
                fill_value=np.nan,
                chunks=chunks,
                dtype="f8",
                write_empty_chunks=False,
            )
        else:
            # Figure out current data region
            region = tuple([slice(0, i) for i in arr.shape])

            # Create zarr array
            z_arr = zarr_root.full(
                name=path,
                shape=shape,
                chunks=chunks,
                dtype="f8",
                fill_value=np.nan,  # same as float64
                write_empty_chunks=False,
            )

            # Fill zarr array with actual data
            z_arr.set_basic_selection(region, arr)

        # Set dask array from zarr array
        d_arr = da.from_zarr(z_arr)
        return d_arr

    def _parse_and_pad_datagram(
        self,
        data_type,
        data_type_shapes: dict = {},
        raw_type: Literal["transmit", "receive"] = "receive",
        use_swap: bool = False,
        zarr_root: Optional[zarr.Group] = None,
        max_chunk_size: str = "100MB",
    ) -> None:
        ping_data_dict = self.ping_data_dict_tx if raw_type == "transmit" else self.ping_data_dict

        # If there's no data, set and skip
        if data_type_shapes[data_type] is None:
            no_data_dict = {ch_id: None for ch_id in ping_data_dict[data_type].keys()}
            ping_data_dict[data_type] = no_data_dict
            return

        # Set up zarr when using swap
        # by determining the chunk sizes
        if use_swap:
            if zarr_root is None:
                raise ValueError("zarr_root cannot be None when use_swap is True")

            # Get the final data shape
            data_shape = data_type_shapes[data_type]

            # Auto chunk on first dimension
            # since this is ping time
            chunks = ("auto",) + (data_shape[1],)
            chunks = auto_chunks(
                chunks=chunks, shape=data_shape, limit=max_chunk_size, dtype=np.dtype("float64")
            )
            chunks = chunks + data_shape[2:]  # Get the n dimension sizes if > 2D
            chunks = tuple([c[0] if isinstance(c, tuple) else c for c in chunks])

        # Go through data for each channel
        for ch_id, arr_list in ping_data_dict[data_type].items():
            # NO DATA -----------------------------------------------------------
            if all(
                (arr is None) or (arr.size == 0) for arr in arr_list
            ):  # if no data in a particular channel
                # Skip all together if there's no data
                ping_data_dict[data_type][ch_id] = None
                continue
            # -------------------------------------------------------------------

            # If there are data do the following
            # Add channel id to list if there's data
            # for "receive" raw type only
            if raw_type == "receive":
                self.ch_ids[data_type].append(ch_id)

            # Pad shorter ping with NaN
            # do this for each channel
            # this is the first small expansion
            padded_arr = self.pad_shorter_ping(arr_list)

            # POWER and COMPLEX Pre-processing --------
            # data_type="angle" does not require extra manipulation, so below
            # only handles power and complex data

            # Multiply power data by conversion factor
            if data_type == "power":
                padded_arr = padded_arr.astype("float32") * INDEX2POWER

            # Split complex data into real and imaginary components
            if data_type == "complex":
                # Split the complex data into real and imaginary components
                padded_arr = {
                    "real": np.real(padded_arr).astype("float64"),
                    "imag": np.imag(padded_arr).astype("float64"),
                }

                # Take care of 0s in imaginary part data
                imag_arr = padded_arr["imag"]
                padded_arr["imag"] = np.where(imag_arr == 0, np.nan, imag_arr)

            # END POWER and COMPLEX Pre-processing -------

            # NO SWAP -----------------------------------------------------------
            # Directly store the padded array
            # to the existing dictionary for the particular
            # data type and channel when not using swap
            if not use_swap:
                ping_data_dict[data_type][ch_id] = padded_arr
                continue
            # -------------------------------------------------------------------

            # SWAP --------------------------------------------------------------
            if data_shape[0] != len(self.ping_time[ch_id]):
                # Let's ensure that the ping time dimension is the same
                # as the original data shape when written to zarr array
                # since this will become the coordinate dimension
                # of the channel data when set in set_groups operation
                data_shape = (len(self.ping_time[ch_id]),) + data_shape[1:]
            # Write to temp zarr for swap
            # then assign the dask array to the dictionary
            if data_type == "complex":
                # Save real and imaginary components separately
                ping_data_dict[data_type][ch_id] = {}
                # Go through real and imaginary components
                for complex_part, arr in padded_arr.items():
                    d_arr = self._write_to_temp_zarr(
                        arr,
                        zarr_root,
                        os.path.join(raw_type, data_type, str(ch_id), complex_part),
                        data_shape,
                        chunks,
                    )
                    ping_data_dict[data_type][ch_id][complex_part] = d_arr

            else:
                d_arr = self._write_to_temp_zarr(
                    padded_arr,
                    zarr_root,
                    os.path.join(raw_type, data_type, str(ch_id)),
                    data_shape,
                    chunks,
                )
                ping_data_dict[data_type][ch_id] = d_arr
            # -------------------------------------------------------------------

    def parse_raw(self):
        """Parse raw data file from Simrad EK60, EK80, and EA640 echosounders."""
        with RawSimradFile(self.source_file, "r", storage_options=self.storage_options) as fid:
            self.config_datagram = fid.read(1)
            self.config_datagram["timestamp"] = np.datetime64(
                self.config_datagram["timestamp"].replace(tzinfo=None), "[ns]"
            )

            # Only EK80 files have configuration in self.config_datagram
            if "configuration" in self.config_datagram:
                # Remove EC150 (ADCP) from config
                channel_id = list(self.config_datagram["configuration"].keys())
                channel_id_rm = [ch for ch in channel_id if "EC150" in ch]
                for ch in channel_id_rm:
                    _ = self.config_datagram["configuration"].pop(ch)

                for v in self.config_datagram["configuration"].values():
                    if "pulse_duration" not in v and "pulse_length" in v:
                        # it seems like sometimes this field can appear with the name "pulse_length"
                        # and in the form of floats separated by semicolons
                        v["pulse_duration"] = [float(x) for x in v["pulse_length"].split(";")]

            # print the usual converting message
            self._print_status()

            # Check if reading an ME70 file with a CON1 datagram.
            next_datagram = fid.peek()
            if next_datagram == "CON1":
                self.CON1_datagram = fid.read(1)
            else:
                self.CON1_datagram = None

            # IDs of the channels found in the dataset
            # self.ch_ids = list(self.config_datagram['configuration'].keys())

            # Read the rest of datagrams
            self._read_datagrams(fid)

        # Read bottom datagrams if `self.bot_file`` is not empty
        if self.bot_file != "":
            bot_datagrams = RawSimradFile(self.bot_file, "r", storage_options=self.storage_options)
            bot_datagrams.read(1)  # Read everything after the `.CON` config datagram
            self._read_datagrams(bot_datagrams)

        # Read index datagrams if `self.idx_file`` is not empty
        if self.idx_file != "":
            idx_datagrams = RawSimradFile(self.idx_file, "r", storage_options=self.storage_options)
            idx_datagrams.read(1)  # Read everything after the `.CON` config datagram
            self._read_datagrams(idx_datagrams)

        # Convert ping time to 1D numpy array, stored in dict indexed by channel,
        #  this will help merge data from all channels into a cube
        for ch, val in self.ping_time.items():
            self.ping_time[ch] = np.array(val, dtype="datetime64[ns]")

    def _read_datagrams(self, fid):
        """Read all datagrams.

        A sample EK60 RAW0 datagram:
            {'type': 'RAW0',
            'low_date': 71406392,
            'high_date': 30647127,
            'channel': 1,
            'mode': 3,
            'transducer_depth': 9.149999618530273,
            'frequency': 18000.0,
            'transmit_power': 2000.0,
            'pulse_length': 0.0010239999974146485,
            'bandwidth': 1573.66552734375,
            'sample_interval': 0.00025599999935366213,
            'sound_velocity': 1466.0,
            'absorption_coefficient': 0.0030043544247746468,
            'heave': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'temperature': 4.0,
            'heading': 0.0,
            'transmit_mode': 1,
            'spare0': '\x00\x00\x00\x00\x00\x00',
            'offset': 0,
            'count': 1386,
            'timestamp': numpy.datetime64('2018-02-11T16:40:25.276'),
            'bytes_read': 5648,
            'power': array([ -6876,  -8726, -11086, ..., -11913, -12522, -11799], dtype=int16),
            'angle': array([[ 110,   13],
                [   3,   -4],
                [ -54,  -65],
                ...,
                [ -92, -107],
                [-104, -122],
                [  82,   74]], dtype=int8)}

        A sample EK80 XML-parameter datagram:
            {'channel_id': 'WBT 545612-15 ES200-7C',
             'channel_mode': 0,
             'pulse_form': 1,
             'frequency_start': '160000',
             'frequency_end': '260000',
             'pulse_duration': 0.001024,
             'sample_interval': 5.33333333333333e-06,
             'transmit_power': 15.0,
             'slope': 0.01220703125}

        A sample EK80 XML-environment datagram:
            {'type': 'XML0',
             'low_date': 3137819385,
             'high_date': 30616609,
             'timestamp': numpy.datetime64('2017-09-12T23:49:10.723'),
             'bytes_read': 448,
             'subtype': 'environment',
             'environment': {'depth': 240.0,
              'acidity': 8.0,
              'salinity': 33.7,
              'sound_speed': 1486.4,
              'temperature': 6.9,
              'latitude': 45.0,
              'sound_velocity_profile': [1.0, 1486.4, 1000.0, 1486.4],
              'sound_velocity_source': 'Manual',
              'drop_keel_offset': 0.0,
              'drop_keel_offset_is_manual': 0,
              'water_level_draft': 0.0,
              'water_level_draft_is_manual': 0,
              'transducer_name': 'Unknown',
              'transducer_sound_speed': 1490.0},
             'xml': '<?xml version="1.0" encoding="utf-8"?>\r\n<Environment Depth="240" ... />\r\n</Environment>'}
        """  # noqa
        num_datagrams_parsed = 0

        while True:
            try:
                # TODO: @ngkvain: what I need in the code to not PARSE the raw0/3 datagram
                #  when users only want CONFIG or ENV, but the way this is implemented
                #  the raw0/3 datagrams are still parsed, you are just not saving them
                new_datagram = fid.read(1)

            except SimradEOF:
                break

            # Convert the timestamp to a datetime64 object.
            new_datagram["timestamp"] = np.datetime64(
                new_datagram["timestamp"].replace(tzinfo=None), "[ns]"
            )

            # # For debugging EC150 datagrams
            # if new_datagram["type"].startswith("XML") and "subtype" in new_datagram:
            #     print(f"{new_datagram['type']} - {new_datagram['subtype']}")
            # else:
            #     print(new_datagram["type"])

            num_datagrams_parsed += 1

            # XML datagrams store environment or instrument parameters for EK80
            if new_datagram["type"].startswith("XML"):
                # Check that environment datagrams contain more than
                # just drop_keel_offset and drop_keel_offset_is_manual
                # Temporary fix for handling >1 EK80 environment datagrams described in:
                # https://github.com/OSOceanAcoustics/echopype/issues/1386
                if new_datagram["subtype"] == "environment" and set(
                    ["drop_keel_offset", "drop_keel_offset_is_manual"]
                ) != set(new_datagram["environment"].keys()):
                    self.environment = new_datagram["environment"]
                    self.environment["xml"] = new_datagram["xml"]
                    self.environment["timestamp"] = new_datagram["timestamp"]
                elif new_datagram["subtype"] == "parameter":
                    if "EC150" not in new_datagram["parameter"]["channel_id"]:
                        #    print(
                        #        f"{new_datagram['parameter']['channel_id']} from XML-parameter "
                        #        "-- NOT SKIPPING"
                        #    )
                        current_parameters = new_datagram["parameter"]
                # else:
                #     print(f"{new_datagram['parameter']['channel_id']} from XML-parameter")

            # RAW0 datagrams store raw acoustic data for a channel for EK60
            elif new_datagram["type"].startswith("RAW0"):
                # Save channel-specific ping time. The channels are stored as 1-based indices
                self.ping_time[new_datagram["channel"]].append(new_datagram["timestamp"])

                # Append ping by ping data
                self._append_channel_ping_data(new_datagram)

            # EK80 datagram sequence:
            #   - XML0 pingsequence
            #   - XML0 parameter
            #   - RAW4
            #   - RAW3
            # RAW3 datagrams store raw acoustic data for a channel for EK80
            elif new_datagram["type"].startswith("RAW3"):
                if "EC150" not in new_datagram["channel_id"]:
                    # print(f"{new_datagram['channel_id']} from RAW3 -- NOT SKIPPING")
                    curr_ch_id = new_datagram["channel_id"]
                    # Check if the proceeding Parameter XML does not
                    # match with data in this RAW3 datagram
                    if current_parameters["channel_id"] != curr_ch_id:
                        raise ValueError("Parameter ID does not match RAW")

                    # Save channel-specific ping time
                    self.ping_time[curr_ch_id].append(new_datagram["timestamp"])

                    # Append ping by ping data
                    new_datagram.update(current_parameters)
                    self._append_channel_ping_data(new_datagram)
                # else:
                #     print(f"{new_datagram['channel_id']} from RAW3")

            # RAW4 datagrams store raw transmit pulse for a channel for EK80
            elif new_datagram["type"].startswith("RAW4"):
                if "EC150" not in new_datagram["channel_id"]:
                    # print(f"{new_datagram['channel_id']} from RAW4 -- NOT SKIPPING")
                    curr_ch_id = new_datagram["channel_id"]
                    # Check if the proceeding Parameter XML does not
                    # match with data in this RAW4 datagram
                    if current_parameters["channel_id"] != curr_ch_id:
                        raise ValueError("Parameter ID does not match RAW")

                    # Ping time is identical to the immediately following RAW3 datagram
                    # so does not need to be stored separately

                    # Append ping by ping data
                    new_datagram.update(current_parameters)
                    self._append_channel_ping_data(new_datagram, raw_type="transmit")
                # else:
                #     print(f"{new_datagram['channel_id']} from RAW4")

            # NME datagrams store ancillary data as NMEA-0817 style ASCII data.
            elif new_datagram["type"].startswith("NME"):
                self.nmea["timestamp"].append(new_datagram["timestamp"])
                self.nmea["nmea_string"].append(new_datagram["nmea_string"])

            # MRU0 datagrams contain motion data for each ping for EK80
            elif new_datagram["type"].startswith("MRU0"):
                self.mru0["heading"].append(new_datagram["heading"])
                self.mru0["pitch"].append(new_datagram["pitch"])
                self.mru0["roll"].append(new_datagram["roll"])
                self.mru0["heave"].append(new_datagram["heave"])
                self.mru0["timestamp"].append(new_datagram["timestamp"])

            # MRU1 datagrams contain latitude/longitude data for each ping for EK80
            elif new_datagram["type"].startswith("MRU1"):
                # TODO: Process other motion fields in `new_datagram`
                self.mru1["latitude"].append(new_datagram["latitude"])
                self.mru1["longitude"].append(new_datagram["longitude"])
                self.mru1["timestamp"].append(new_datagram["timestamp"])

            # FIL datagrams contain filters for processing bascatter data for EK80
            elif new_datagram["type"].startswith("FIL"):
                if "EC150" not in new_datagram["channel_id"]:
                    # print(f"{new_datagram['channel_id']} from FIL -- NOT SKIPPING")
                    self.fil_coeffs[new_datagram["channel_id"]][new_datagram["stage"]] = (
                        new_datagram["coefficients"]
                    )
                    self.fil_df[new_datagram["channel_id"]][new_datagram["stage"]] = new_datagram[
                        "decimation_factor"
                    ]
                # else:
                #     print(f"{new_datagram['channel_id']} from FIL")

            # TAG datagrams contain time-stamped annotations inserted via the recording software
            elif new_datagram["type"].startswith("TAG"):
                logger.info("TAG datagram encountered.")

            # BOT datagrams contain sounder detected bottom depths from .bot files
            elif new_datagram["type"].startswith("BOT"):
                self.bot["depth"].append(new_datagram["depth"])
                self.bot["timestamp"].append(new_datagram["timestamp"])

            # IDX datagrams contain lat/lon and vessel distance from .idx files
            elif new_datagram["type"].startswith("IDX"):
                self.idx["ping_number"].append(new_datagram["ping_number"])
                self.idx["file_offset"].append(new_datagram["file_offset"])
                self.idx["vessel_distance"].append(new_datagram["distance"])
                self.idx["latitude"].append(new_datagram["latitude"])
                self.idx["longitude"].append(new_datagram["longitude"])
                self.idx["timestamp"].append(new_datagram["timestamp"])

            # DEP datagrams contain sounder detected bottom depths from .out files
            # as well as reflectivity data
            elif new_datagram["type"].startswith("DEP"):
                logger.info("DEP datagram encountered.")
            else:
                logger.info("Unknown datagram type: " + str(new_datagram["type"]))

    def _append_channel_ping_data(
        self, datagram, raw_type: Literal["transmit", "receive"] = "receive"
    ):
        """
        Append ping by ping data.

        Parameters
        ----------
        datagram : dict
            the newly read sample datagram
        raw_type : {"transmit", "receive"}, default "receive"
            The raw type of the datagram. "transmit" is for RAW4 datagrams and
            "receive" is for other datagrams.
        """
        if raw_type not in ["transmit", "receive"]:
            raise ValueError(f"raw_type must be one of 'transmit' or 'receive' not {raw_type}")

        # TODO: do a thorough check with the convention and processing
        # unsaved = ['channel', 'channel_id', 'low_date', 'high_date', # 'offset', 'frequency' ,
        #            'transmit_mode', 'spare0', 'bytes_read', 'type'] #, 'n_complex']
        ch_id = datagram["channel_id"] if "channel_id" in datagram else datagram["channel"]

        for k, v in datagram.items():
            if raw_type == "receive":
                self.ping_data_dict[k][ch_id].append(v)
            else:
                self.ping_data_dict_tx[k][ch_id].append(v)

    @staticmethod
    def pad_shorter_ping(data_list) -> np.ndarray:
        """
        Pad shorter ping with NaN: power, angle, complex samples.

        Parameters
        ----------
        data_list : list
            Power, angle, or complex samples for each channel from RAW3 datagram.
            Each ping is one entry in the list.

        Returns
        -------
        out_array : np.ndarray
            Numpy array containing samplings from all pings.
            The array is NaN-padded if some pings are of different lengths.
        """
        lens = np.array([len(item) for item in data_list])
        if np.unique(lens).size != 1:  # if some pings have different lengths along range
            if data_list[0].ndim == 2:
                # Data may have an extra dimension:
                #  - Angle data have an extra dimension for alongship and athwartship samples
                #  - Complex data have an extra dimension for different transducer sectors
                mask = (
                    lens[:, None, None]
                    > np.array([np.arange(lens.max())] * data_list[0].shape[1]).T
                )

            else:
                mask = lens[:, None] > np.arange(lens.max())

            # Create output array from mask
            out_array = np.full(mask.shape, np.nan)

            # Concatenate short pings
            concat_short_pings = np.concatenate(data_list).reshape(-1)  # reshape in case data > 1D

            # Take care of problem of np.nan being implicitly "real"
            if concat_short_pings.dtype == np.complex64:
                out_array = out_array.astype(np.complex64)

            # Fill in values
            out_array[mask] = concat_short_pings
        else:
            out_array = np.array(data_list)
        return out_array
