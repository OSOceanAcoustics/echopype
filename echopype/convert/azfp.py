import os
import numpy as np
from struct import unpack

path = "D:\\Documents\\Projects\\echopype\\toolbox\\12022316.01A"


class ConvertAZFP:
    """Class for converting raw .01A AZFP files """

    def __init__(self, _filename=""):
        self.file_name = _filename
        self.file_type = 64770
        self.bytes_per_header = 124
        self.format = ">HHHHIHHHHHHHHHHHHHHHHHHHHHHHHHHHHHBBBBHBBBBBBBBHHHHHHHHHHHHHHHHHHHH"

    def parse_raw(self):
        with open(self.file_name, "rb") as raw:
            data_values = []
            ii = 1  # Number of chunks
            while True:
                chunk = raw.read(self.bytes_per_header)
                if chunk:
                    unpacked_chunk = unpack(self.format, chunk)
                    split = self.split_chunk(raw, ii, unpacked_chunk)
                    data_values.append(split)
                    break
                else:
                    # End of file
                    break
                ii += 1

        dtype = self.get_dtypes()
        # Puts data values and data type into a numpy record array
        Data = np.rec.array(data_values, dtype=dtype)
        return Data

    def split_chunk(self, raw, ii, unpacked_chunk):
        # Checks the flag in each header with the correct AZFP flag
        Flag = unpacked_chunk[0]
        if Flag != self.file_type:
            check_eof = raw.read(1)
            if check_eof:
                print("Error: Unknown file type")
            return
        fields = self.get_fields()
        data_values = []

        i = 0
        for field in fields:
            if len(field) == 3:
                arr = []
                for _ in range(field[2]):
                    arr.append(unpacked_chunk[i])
                    i += 1
                data_values.append(arr)
            else:
                data_values.append(unpacked_chunk[i])
                i += 1

        return tuple(data_values)

    def get_dtypes(self):
        '''Returns a set containing the data types of each header field'''
        fields = self.get_fields()
        dtype = []
        for field in fields:
            dtype.append(field)

        return dtype

    @staticmethod
    def get_fields():
        '''Returns the fields contained in each header of the raw file'''
        _fields = (
            ('profile_flag', 'u2'),
            ('profile_number', 'u2'),
            ('serial_number', 'u2'),
            ('ping_status', 'u2'),
            ('burst_int', 'u4'),
            ('year', 'u2'),                 # 012 - Year
            ('month', 'u2'),                # 014 - Month
            ('day', 'u2'),                  # 016 - Day
            ('hour', 'u2'),                 # 018 - Hour
            ('minute', 'u2'),               # 020 - Minute
            ('second', 'u2'),               # 022 - Second
            ('hundredths', 'u2'),           # 024 - Hundreths of a second
            ('dig_rate', 'u2', 4),
            ('lockout_index', 'u2', 4),
            ('num_bins', 'u2', 4),
            ('range_samples', 'u2', 4),
            ('ping_per_profile', 'u2'),
            ('avg_ping', 'u2'),
            ('num_ping_acq', 'u2'),
            ('ping_period', 'u2'),
            ('first_ping', 'u2'),
            ('last_ping', 'u2'),
            ('data_type', "u1", 4),
            ('data_error', 'u2'),
            ('phase', 'u1'),
            ('over_run', 'u1'),
            ('num_chan', 'u1'),
            ('gain', 'u1', 4),
            ('spare_chan', 'u1'),
            ('pulse_length', 'u2', 4),
            ('board_num', 'u2', 4),
            ('frequency', 'u2', 4),
            ('sensor_flag', 'u2'),
            ('tilt_x', 'u2'),                 # 110 - Tilt X (counts)
            ('tilt_y', 'u2'),                 # 112 - Tilt Y (counts)
            ('battery_voltage', 'u2'),        # 114 - Battery voltage (counts)
            ('pressure', 'u2'),               # 116 - Pressure (counts)
            ('temperature', 'u2'),            # 118 - Temperature (counts)
            ('ad_channel_6', 'u2'),           # 120 - AD channel 6
            ('ad_channel_7', 'u2')            # 122 - AD channel 7
        )

        return _fields


file1 = ConvertAZFP(path)
d = file1.parse_raw()
print(d)