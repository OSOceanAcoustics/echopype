
from os import path

from echopype import ek60

from . import constants

def test_unpack_ek60_raw():

    print(dir(ek60))

    input_file_path = constants.ek60_test_file

    if not path.isfile(input_file_path):
        print("Couldn't find the input file \"%s\"" % input_file_path )

    first_ping_metadata, data_times, power_data_dict, frequencies, bin_size, \
        config_header, config_transducer = ek60.load_ek60_raw(input_file_path)
