import os
import numpy as np
import xarray as xr
from echopype.convert.ek60 import ConvertEK60

raw_path = './echopype/data/DY1801_EK60-D20180211-T164025.raw'

# Other data files
# raw_filename = 'data_zplsc/OceanStarr_2017-D20170725-T004612.raw'  # OceanStarr 2 channel EK60
# raw_filename = '../data/DY1801_EK60-D20180211-T164025.raw'  # Dyson 5 channel EK60
# raw_filename = 'data_zplsc/D20180206-T000625.raw   # EK80


def test_convert_ek60():
    """Test converting """
    # Unpacking data
    tmp = ConvertEK60(raw_path)
    tmp.load_ek60_raw()

    # Convert to .nc file
    tmp.raw2nc()

    # Read .nc file into an xarray DataArray
    ds_beam = xr.open_dataset(tmp.nc_path, group='Beam')

    # Check if backscatter data from all channels are identical to those directly unpacked
    for idx in range(tmp.config_header['transducer_count']):
        # idx is channel index starting from 0
        assert np.any(tmp.power_data_dict[idx + 1] ==
                      ds_beam.backscatter_r.sel(frequency=tmp.config_transducer[idx]['frequency']).data)
    os.remove(tmp.nc_path)
    del tmp


