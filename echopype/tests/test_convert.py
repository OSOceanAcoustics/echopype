import os
import numpy as np
import xarray as xr
from echopype.convert.ek60 import ConvertEK60
from echopype.convert.azfp import ConvertAZFP

raw_path = './echopype/data/DY1801_EK60-D20180211-T164025.raw'
azfp_01a_path = './echopype/data/17082117.01A'
azfp_xml_path = './echopype/data/17041823.XML'
azfp_test_path = './echopype/data/azfp_test.nc'
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


def test_convert_AZFP():
    # Read in the dataset that will be used to confirm working conversions. Generated from MATLAB code
    ds_test = xr.open_dataset(azfp_test_path)

    # Unpacking data
    tmp = ConvertAZFP(azfp_01a_path, azfp_xml_path)
    tmp.parse_raw()

    # Convert to .nc file
    tmp.raw2nc()

    # Test beam group
    with xr.open_dataset(tmp.nc_path, group='Beam') as ds_beam:
        # Test frequency
        assert np.array_equal(ds_test.frequency.values, ds_beam.frequency.values)
        # Test ping time
        assert np.array_equal(ds_test.ping_time.values, ds_beam.ping_time.values)
        # Test tilt x and y
        assert np.array_equal(ds_test.tilt_x.values, ds_beam.tilt_x.values)
        assert np.array_equal(ds_test.tilt_y.values, ds_beam.tilt_y.values)
        # Test range
        assert np.array_equal(ds_test.range.values.squeeze(), ds_beam.range.values)
        # Test backscatter_r
        assert np.array_equal(ds_test.backscatter.values, ds_beam.backscatter_r.values)

    # Test enviroment group
    with xr.open_dataset(tmp.nc_path, group='Environment') as ds_env:
        # Test temperature
        assert np.array_equal(ds_test.temperature.values, ds_env.temperature.values)
        # Test sound speed. 1 value is used because sound speed is the same across frequencies
        assert ds_test.sound_speed == ds_env.sound_speed_indicative.values[0]

    ds_test.close()
    os.remove(tmp.nc_path)
    del tmp


test_convert_AZFP()
