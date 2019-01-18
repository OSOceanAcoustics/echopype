import os
from echopype.convert import ek60

raw_filename = '../data/DY1801_EK60-D20180211-T164025.raw'

def test_convert_ek60():

first_ping_metadata, data_times, motion, \
    power_data_dict, angle_data_dict, tr_data_dict, \
    config_header, config_transducer = ek60.load_ek60_raw(raw_filename)

ek60.save_raw_to_nc(raw_filename)



# OceanStarr 2 channel EK60
# raw_filename = 'data_zplsc/OceanStarr_2017-D20170725-T004612.raw'

# Dyson 5 channel EK60
# raw_filename = '../data/DY1801_EK60-D20180211-T164025.raw'

# EK80
# raw_filename = 'data_zplsc/D20180206-T000625.raw's

