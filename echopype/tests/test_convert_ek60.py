from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
from scipy.io import loadmat
from ..convert import open_raw

ek60_path = Path('./echopype/test_data/ek60/')

csv_paths = ['./echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Power18.csv',
             './echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Power38.csv',
             './echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Power70.csv',
             './echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Power120.csv',
             './echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Power200.csv']
# raw_paths = ['./echopype/test_data/ek60/set1/' + file
            #  for file in os.listdir('./echopype/test_data/ek60/set1')]    # 2 range lengths
# raw_path = ['./echopype/test_data/ek60/set2/' + file
#                  for file in os.listdir('./echopype/test_data/ek60/set2')]    # 3 range lengths
# Other data files
# raw_filename = 'data_zplsc/OceanStarr_2017-D20170725-T004612.raw'  # OceanStarr 2 channel EK60
# raw_filename = '../data/DY1801_EK60-D20180211-T164025.raw'  # Dyson 5 channel EK60
# raw_filename = 'data_zplsc/D20180206-T000625.raw   # EK80


def test_convert_ek60_matlab_raw():
    """Compare parsed Beam group data with Matlab outputs.
    """
    ek60_raw_path = str(ek60_path.joinpath('DY1801_EK60-D20180211-T164025.raw'))
    ek60_matlab_path = str(ek60_path.joinpath('from_matlab/DY1801_EK60-D20180211-T164025_rawData.mat'))

    # Convert file
    echodata = open_raw(file=ek60_raw_path, model='EK60')
    echodata.to_netcdf()

    # Compare with matlab outputs
    with xr.open_dataset(echodata.output_file, group='Beam') as ds_beam:
        ds_matlab = loadmat(ek60_matlab_path)

        # power
        assert np.allclose(
            [ds_matlab['rawData'][0]['pings'][0]['power'][0][fidx] for fidx in range(5)],
            ds_beam.backscatter_r.transpose('frequency', 'range_bin', 'ping_time'),
            rtol=0,
            atol=1.6e-5
        )
        # angle: alongship and athwartship
        for angle in ['alongship', 'athwartship']:
            assert np.alltrue(
                [ds_matlab['rawData'][0]['pings'][0][angle][0][fidx] for fidx in range(5)]
                == ds_beam['angle_' + angle].transpose('frequency', 'range_bin', 'ping_time')
            )

    Path(echodata.output_file).unlink()


def test_convert_power_echoview():
    """Test converting power and compare it to echoview output"""
    tmp = Convert(file=raw_path, model='EK60')
    tmp.to_netcdf()

    channels = []
    for file in csv_paths:
        channels.append(pd.read_csv(file, header=None, skiprows=[0]).iloc[:, 13:])
    test_power = np.stack(channels)
    with xr.open_dataset(tmp.output_file, group='Beam') as ds_beam:
        assert np.allclose(
            test_power,
            ds_beam.backscatter_r.isel(ping_time=slice(None, 10), range_bin=slice(1, None)),
            atol=1e-10)

    os.remove(tmp.output_file)
