from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from echopype import open_raw

ek60_path = Path('./echopype/test_data/ek60/')

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
    echodata = open_raw(raw_file=ek60_raw_path, sonar_model='EK60')

    # Compare with matlab outputs
    ds_matlab = loadmat(ek60_matlab_path)

    # power
    assert np.allclose(
        [ds_matlab['rawData'][0]['pings'][0]['power'][0][fidx] for fidx in range(5)],
        echodata.beam.backscatter_r.transpose('frequency', 'range_bin', 'ping_time'),
        rtol=0,
        atol=1.6e-5
    )
    # angle: alongship and athwartship
    for angle in ['alongship', 'athwartship']:
        assert np.array_equal(
            [ds_matlab['rawData'][0]['pings'][0][angle][0][fidx] for fidx in range(5)],
            echodata.beam['angle_' + angle].transpose('frequency', 'range_bin', 'ping_time')
        )


def test_convert_ek60_echoview_raw():
    """Compare parsed power data (count) with csv exported by EchoView.
    """
    ek60_raw_path = str(ek60_path.joinpath('DY1801_EK60-D20180211-T164025.raw'))
    ek60_csv_path = [
        ek60_path.joinpath('from_echoview/DY1801_EK60-D20180211-T164025-Power%d.csv' % freq)
        for freq in [18, 38, 70, 120, 200]
    ]

    # Read csv files exported by EchoView
    channels = []
    for file in ek60_csv_path:
        channels.append(pd.read_csv(file, header=None, skiprows=[0]).iloc[:, 13:])
    test_power = np.stack(channels)

    # Convert to netCDF and check
    echodata = open_raw(raw_file=ek60_raw_path, sonar_model='EK60')
    for fidx, atol in zip(range(5), [1e-5, 1.1e-5, 1.1e-5, 1e-5, 1e-5]):
        assert np.allclose(
            test_power[fidx, :, :],
            echodata.beam.backscatter_r.isel(frequency=fidx, ping_time=slice(None, 10), range_bin=slice(1, None)),
            atol=9e-6, rtol=atol
        )
