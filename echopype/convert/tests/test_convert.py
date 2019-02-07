import os
import numpy as np
import xarray as xr
from echopype.convert import ek60

raw_path = './echopype/data/DY1801_EK60-D20180211-T164025.raw'

# Other data files
# raw_filename = 'data_zplsc/OceanStarr_2017-D20170725-T004612.raw'  # OceanStarr 2 channel EK60
# raw_filename = '../data/DY1801_EK60-D20180211-T164025.raw'  # Dyson 5 channel EK60
# raw_filename = 'data_zplsc/D20180206-T000625.raw   # EK80


def test_convert_ek60():
    """Test converting """
    # Unpacking data
    first_ping_metadata, data_times, motion, \
        power_data_dict, angle_data_dict, tr_data_dict, \
        config_header, config_transducer = ek60.load_ek60_raw(raw_path)

    # Convert to .nc file
    nc_path = ek60.save_raw_to_nc(raw_path)

    # Read .nc file into an xarray DataArray
    ds_beam = xr.open_dataset(nc_path, group='Beam')

    # Check if backscatter data from all channels are identical to those directly unpacked
    for idx in range(config_header['transducer_count']):
        # idx is channel index starting from 0
        assert np.any(power_data_dict[idx + 1] ==
                      ds_beam.backscatter_r.sel(frequency=config_transducer[idx]['frequency']).data)
    os.remove(nc_path)


