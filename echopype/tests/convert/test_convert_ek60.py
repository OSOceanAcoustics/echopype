import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
import pytest

from echopype import open_raw
from echopype.convert import ParseEK60
from echopype.convert.set_groups_ek60 import SetGroupsEK60


@pytest.fixture
def ek60_path(test_path):
    return test_path["EK60"]

@pytest.fixture
def ek60_missing_channel_power_path(test_path):
    return test_path["EK60_MISSING_CHANNEL_POWER"]

@pytest.fixture
def es60_path(test_path):
    return test_path["ES60"]

@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
def test_convert_ek60_splitbeam_no_angle(ek60_path):
    """Convert a file from a split-beam setup that does not record angle data."""

    raw_path = (
        ek60_path
        / "NBP_B050N-D20180118-T090228.raw"
    )
    ed = open_raw(raw_path, "EK60")

    assert "angle_athwartship" not in ed["Sonar/Beam_group1"]
    assert "angle_alongship" not in ed["Sonar/Beam_group1"]


@pytest.mark.integration
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


@pytest.mark.integration
def test_converting_ek60_raw_with_missing_channel_power(ek60_missing_channel_power_path):
    """
    Tests that we can convert a EK60 RAW file that has missing power data for a
    specific channel.
    """
    # Parse RAW
    ek60_missing_channel_power_raw_path = str(
        ek60_missing_channel_power_path.joinpath("Summer2017-D20170807-T171736.raw")
    )
    ek60_parser = ParseEK60(ek60_missing_channel_power_raw_path)
    ek60_parser.parse_raw()

    # Open RAW
    ed = open_raw(ek60_missing_channel_power_raw_path, sonar_model="EK60")

    # Get channels that have empty `power`
    channels = list(ek60_parser.config_datagram["transceivers"].keys())
    empty_power_chs = {
        ch: ek60_parser.config_datagram["transceivers"][ch]["channel_id"]
        for ch in channels
        if len(ek60_parser.ping_data_dict["power"][ch]) == 0
    } 
    
    # Check that all empty power channels do not exist in the EchoData Beam group
    for _, empty_power_channel_name in empty_power_chs.items():
        assert empty_power_channel_name not in ed["Sonar/Beam_group1"]["channel"]


@pytest.mark.unit
def test_parse_ek60_validate_channels():
    """Validate requested EK60 channel_id values against the config datagram."""
    parser = ParseEK60("dummy.raw", channels=["GPT  18 kHz 009072034d45 1-1 ES18-11"])
    parser.config_datagram = {
        "transceivers": {
            1: {"channel_id": "GPT  18 kHz 009072034d45 1-1 ES18-11"},
            2: {"channel_id": "GPT  38 kHz 009072033fa2 2-1 ES38B"},
        }
    }
    parser._validate_channels()
    assert parser.channels == {"GPT  18 kHz 009072034d45 1-1 ES18-11"}

    parser.channels = ["nonexistent-channel"]
    with pytest.raises(ValueError, match="Requested channel_id"):
        parser._validate_channels()


@pytest.mark.unit
def test_parse_ek60_is_selected():
    """Check EK60 channel selection using integer channel indices."""
    parser = ParseEK60("dummy.raw")
    parser.config_datagram = {
        "transceivers": {
            1: {"channel_id": "GPT  18 kHz 009072034d45 1-1 ES18-11"},
        }
    }
    parser.channels = {"GPT  18 kHz 009072034d45 1-1 ES18-11"}
    assert parser._is_selected(1)
    assert not parser._is_selected(99)


@pytest.mark.unit
def test_set_groups_ek60_filters_sorted_channel():
    """SetGroupsEK60 keeps only selected channel_id values in sorted_channel."""
    parser = ParseEK60("dummy.raw", channels=["GPT  38 kHz 009072033fa2 2-1 ES38B"])
    parser.config_datagram = {
        "transceivers": {
            1: {"channel_id": "GPT  18 kHz 009072034d45 1-1 ES18-11", "frequency": 18000.0},
            2: {"channel_id": "GPT  38 kHz 009072033fa2 2-1 ES38B", "frequency": 38000.0},
        }
    }
    parser.ping_data_dict["power"] = {1: [1], 2: [1]}
    parser.channels = {"GPT  38 kHz 009072033fa2 2-1 ES38B"}

    set_groups = SetGroupsEK60(parser, "dummy.raw", "", None)
    assert list(set_groups.sorted_channel.values()) == ["GPT  38 kHz 009072033fa2 2-1 ES38B"]


@pytest.mark.integration
def test_open_raw_channels_subset_ek60(ek60_path):
    """Parse and store only a single EK60 channel by channel_id."""
    raw_file = str(ek60_path / "DY1801_EK60-D20180211-T164025.raw")
    echodata_all = open_raw(raw_file=raw_file, sonar_model="EK60")
    all_channels = list(echodata_all["Sonar/Beam_group1"].channel.values)
    assert len(all_channels) > 1

    selected = all_channels[0]
    echodata_sub = open_raw(raw_file=raw_file, sonar_model="EK60", channels=[selected])
    assert echodata_sub["Sonar/Beam_group1"].sizes["channel"] == 1
    assert echodata_sub["Sonar/Beam_group1"].channel.item() == selected


@pytest.mark.integration
def test_open_raw_channels_multiple_ek60(ek60_path):
    """Parse and store multiple EK60 channels by channel_id."""
    raw_file = str(ek60_path / "DY1801_EK60-D20180211-T164025.raw")
    echodata_all = open_raw(raw_file=raw_file, sonar_model="EK60")
    all_channels = list(echodata_all["Sonar/Beam_group1"].channel.values)
    assert len(all_channels) >= 2

    selected = all_channels[:2]
    echodata_sub = open_raw(raw_file=raw_file, sonar_model="EK60", channels=selected)
    assert echodata_sub["Sonar/Beam_group1"].sizes["channel"] == 2
    assert set(echodata_sub["Sonar/Beam_group1"].channel.values) == set(selected)


@pytest.mark.integration
def test_open_raw_channels_ek60_data_matches_full_parse(ek60_path):
    """Selected-channel parse produces the same backscatter as indexing the full parse."""
    raw_file = str(ek60_path / "DY1801_EK60-D20180211-T164025.raw")
    echodata_all = open_raw(raw_file=raw_file, sonar_model="EK60")
    selected = echodata_all["Sonar/Beam_group1"].channel.values[0]

    echodata_sub = open_raw(raw_file=raw_file, sonar_model="EK60", channels=[selected])
    full_backscatter = echodata_all["Sonar/Beam_group1"]["backscatter_r"].sel(channel=selected)
    sub_backscatter = echodata_sub["Sonar/Beam_group1"]["backscatter_r"].isel(channel=0)
    np.testing.assert_array_equal(full_backscatter.values, sub_backscatter.values)


@pytest.mark.integration
def test_open_raw_channels_invalid_ek60(ek60_path):
    """Raise when requested EK60 channel_id values are not in the file."""
    raw_file = str(ek60_path / "DY1801_EK60-D20180211-T164025.raw")
    with pytest.raises(ValueError, match="Requested channel_id"):
        open_raw(
            raw_file=raw_file,
            sonar_model="EK60",
            channels=["nonexistent-channel"],
        )


def test_parse_speed_over_ground(ek60_path):
    """Make sure we parse speed over ground from a RAW file."""

    # This raw file has speed in NMEA VTG and RMC messages
    echodata = open_raw(
        raw_file=ek60_path/'NBP_B050N-D20180118-T090228.raw',
        sonar_model='EK60'
    )

    # Check that there are data that are not NaN
    assert (echodata["Platform"]['speed_over_ground'].sizes == {'time11': 584})
    # this .raw file has nan's in the speed over ground data 
    # assert (not np.any(np.isnan(echodata["Platform"]['speed_over_ground'])))


@pytest.mark.unit
def test_parse_NMEA_heading(ek60_path):
    """Make sure we parse NMEA heading from a RAW file when MRU heading is not present."""

    echodata = open_raw(
        raw_file=ek60_path/'NBP_B050N-D20180118-T090228.raw',
        sonar_model='EK60'
    )

    # Check that there are non-NaN data
    assert (echodata["Platform"]['heading'].sizes == {'time10': 584})
    assert (not np.any(np.isnan(echodata["Platform"]['heading'])))
