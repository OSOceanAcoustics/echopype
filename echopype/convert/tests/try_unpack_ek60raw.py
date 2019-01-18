from echopype.convert import ek60

# OOI CE04OSPS EK60
first_ping_metadata, data_times, motion, \
power_data_dict, angle_data_dict, tr_data_dict, \
config_header, config_transducer = \
    ek60.load_ek60_raw('../data/DY1801_EK60-D20180211-T164025.raw')

# OceanStarr 2 channel EK60
# first_ping_metadata, data_times, power_data_dict, frequencies, bin_size, config_header, config_transducer = \
#     ek60.load_ek60_raw('data_zplsc/OceanStarr_2017-D20170725-T004612.raw')

# Dyson 5 channel EK60
# first_ping_metadata, data_times, power_data_dict, angle_data_dict, motion_data_dict, config_header, config_transducer = \
#     ek60.load_ek60_raw('data_zplsc/DY1801_EK60-D20180211-T164025.raw')

# EK80
# first_ping_metadata, data_times, power_data_dict, frequencies, bin_size, config_header, config_transducer = \
#     ek60.load_ek60_raw('data_zplsc/D20180206-T000625.raw')

