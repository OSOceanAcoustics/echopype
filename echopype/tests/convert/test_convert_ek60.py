import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
import pytest

from echopype import open_raw


@pytest.fixture
def ek60_path(test_path):
    return test_path["EK60"]

@pytest.fixture
def es60_path(test_path):
    return test_path["ES60"]


# raw_paths = ['./echopype/test_data/ek60/set1/' + file
#  for file in os.listdir('./echopype/test_data/ek60/set1')]    # 2 range lengths
# raw_path = ['./echopype/test_data/ek60/set2/' + file
#                  for file in os.listdir('./echopype/test_data/ek60/set2')]    # 3 range lengths
# Other data files
# raw_filename = 'data_zplsc/OceanStarr_2017-D20170725-T004612.raw'  # OceanStarr 2 channel EK60
# raw_filename = '../data/DY1801_EK60-D20180211-T164025.raw'  # Dyson 5 channel EK60
# raw_filename = 'data_zplsc/D20180206-T000625.raw   # EK80


def test_convert_ek60_matlab_raw(ek60_path):
    """Compare parsed Beam group data with Matlab outputs."""
    ek60_raw_path = str(
        ek60_path.joinpath('DY1801_EK60-D20180211-T164025.raw')
    )
    ek60_matlab_path = str(
        ek60_path.joinpath(
            'from_matlab', 'DY1801_EK60-D20180211-T164025_rawData.mat'
        )
    )

    # Convert file
    echodata = open_raw(raw_file=ek60_raw_path, sonar_model='EK60')

    # Compare with matlab outputs
    ds_matlab = loadmat(ek60_matlab_path)

    # check platform
    nan_plat_vars = [
        "MRU_offset_x",
        "MRU_offset_y",
        "MRU_offset_z",
        "MRU_rotation_x",
        "MRU_rotation_y",
        "MRU_rotation_z",
        "position_offset_x",
        "position_offset_y",
        "position_offset_z"
    ]
    for plat_var in nan_plat_vars:
        assert plat_var in echodata["Platform"]
        assert np.isnan(echodata["Platform"][plat_var]).all()
    zero_plat_vars = [
        "transducer_offset_x",
        "transducer_offset_y",
        "transducer_offset_z",
    ]
    for plat_var in zero_plat_vars:
        assert plat_var in echodata["Platform"]
        assert (echodata["Platform"][plat_var] == 0).all()
    # check water_level
    assert np.allclose(echodata["Platform"]["water_level"], 9.14999962, rtol=0)

    # power
    assert np.allclose(
        [
            ds_matlab['rawData'][0]['pings'][0]['power'][0][fidx]
            for fidx in range(5)
        ],
        echodata["Sonar/Beam_group1"].backscatter_r.transpose(
            'channel', 'range_sample', 'ping_time'
        ),
        rtol=0,
        atol=1.6e-5,
    )
    # angle: alongship and athwartship
    for angle in ['alongship', 'athwartship']:
        assert np.array_equal(
            [
                ds_matlab['rawData'][0]['pings'][0][angle][0][fidx]
                for fidx in range(5)
            ],
            echodata["Sonar/Beam_group1"]['angle_' + angle].transpose(
                'channel', 'range_sample', 'ping_time'
            ),
        )


def test_convert_ek60_echoview_raw(ek60_path):
    """Compare parsed power data (count) with csv exported by EchoView."""
    ek60_raw_path = str(
        ek60_path.joinpath('DY1801_EK60-D20180211-T164025.raw')
    )
    ek60_csv_path = [
        ek60_path.joinpath(
            'from_echoview', 'DY1801_EK60-D20180211-T164025-Power%d.csv' % freq
        )
        for freq in [18, 38, 70, 120, 200]
    ]

    # Read csv files exported by EchoView
    channels = []
    for file in ek60_csv_path:
        channels.append(
            pd.read_csv(file, header=None, skiprows=[0]).iloc[:, 13:]
        )
    test_power = np.stack(channels)

    # Convert to netCDF and check
    echodata = open_raw(raw_file=ek60_raw_path, sonar_model='EK60')

    # get indices of sorted frequency_nominal values. This is necessary
    # because the frequency_nominal values are not always in ascending order.
    sorted_freq_ind = np.argsort(echodata["Sonar/Beam_group1"].frequency_nominal)

    for fidx, atol in zip(range(5), [1e-5, 1.1e-5, 1.1e-5, 1e-5, 1e-5]):
        assert np.allclose(
            test_power[fidx, :, :],
            echodata["Sonar/Beam_group1"].backscatter_r.isel(
                channel=sorted_freq_ind[fidx],
                ping_time=slice(None, 10),
                range_sample=slice(1, None)
            ),
            atol=9e-6,
            rtol=atol,
        )

    # check platform
    nan_plat_vars = [
        "MRU_offset_x",
        "MRU_offset_y",
        "MRU_offset_z",
        "MRU_rotation_x",
        "MRU_rotation_y",
        "MRU_rotation_z",
        "position_offset_x",
        "position_offset_y",
        "position_offset_z"
    ]
    for plat_var in nan_plat_vars:
        assert plat_var in echodata["Platform"]
        assert np.isnan(echodata["Platform"][plat_var]).all()
    zero_plat_vars = [
        "transducer_offset_x",
        "transducer_offset_y",
        "transducer_offset_z",
    ]
    for plat_var in zero_plat_vars:
        assert plat_var in echodata["Platform"]
        assert (echodata["Platform"][plat_var] == 0).all()

    # check water_level
    assert np.allclose(echodata["Platform"]["water_level"], 9.14999962, rtol=0)


def test_convert_ek60_duplicate_frequencies(ek60_path):
    """Convert a file with duplicate frequencies"""

    raw_path = (
        ek60_path
        / "DY1002_EK60-D20100318-T023008_rep_freq.raw"
    )
    ed = open_raw(raw_path, "EK60")

    truth_chan_vals = np.array(['GPT  18 kHz 009072034d45 1-1 ES18-11',
                                'GPT  38 kHz 009072033fa2 2-1 ES38B',
                                'GPT  70 kHz 009072058c6c 3-1 ES70-7C',
                                'GPT  70 kHz 009072058c6c 3-2 ES70-7C',
                                'GPT 120 kHz 00907205794e 4-1 ES120-7C',
                                'GPT 200 kHz 0090720346a8 5-1 ES200-7C'], dtype='<U37')

    truth_freq_nom_vals = np.array([18000., 38000., 70000.,
                                    70000., 120000., 200000.], dtype=np.float64)

    assert np.allclose(ed['Sonar/Beam_group1'].frequency_nominal,
                       truth_freq_nom_vals, rtol=1e-05, atol=1e-08)

    assert np.all(ed['Sonar/Beam_group1'].channel.values == truth_chan_vals)


def test_convert_ek60_splitbeam_no_angle(ek60_path):
    """Convert a file from a split-beam setup that does not record angle data."""

    raw_path = (
        ek60_path
        / "NBP_B050N-D20180118-T090228.raw"
    )
    ed = open_raw(raw_path, "EK60")

    assert "angle_athwartship" not in ed["Sonar/Beam_group1"]
    assert "angle_alongship" not in ed["Sonar/Beam_group1"]


def test_convert_es60_no_unicode_error(es60_path):
    """Convert a file should not give unicode error"""

    raw_path = (
        es60_path
        / "L0007-D20191202-T060239-ES60.raw"
    )
    try:
        open_raw(raw_path, sonar_model='EK60')
    except UnicodeDecodeError:
        pytest.fail("UnicodeDecodeError raised")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("file_path"),
    [
        "DY1002_EK60-D20100318-T023008_rep_freq.raw",
        "from_echopy/JR230-D20091215-T121917.raw"
    ]
)
def test_convert_ek60_different_num_channel_mode_values(file_path, ek60_path):
    """
    Check that no runtime warning is called when there are different number of channel mode
    values per channel and check that `channel_mode` is of type `np.float32`.
    """
    # Catch and throw error for any `RuntimeWarning`
    with warnings.catch_warnings():
        warnings.simplefilter(action="error", category=RuntimeWarning)
        ed = open_raw(ek60_path / file_path, sonar_model="EK60")

        # Check dtype
        assert np.issubdtype(
            ed["Sonar/Beam_group1"]["channel_mode"].data.dtype,
            np.float32
        )
