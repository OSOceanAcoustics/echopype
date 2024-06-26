import pytest
import numpy as np
import pandas as pd
from scipy.io import loadmat
from echopype import open_raw

from echopype.testing import TEST_DATA_FOLDER
from echopype.convert.parse_ek80 import ParseEK80
from echopype.convert.set_groups_ek80 import WIDE_BAND_TRANS, PULSE_COMPRESS, FILTER_IMAG, FILTER_REAL, DECIMATION


@pytest.fixture
def ek80_path(test_path):
    return test_path["EK80"]


def pytest_generate_tests(metafunc):
    ek80_new_path = TEST_DATA_FOLDER / "ek80_new"
    ek80_new_files = ek80_new_path.glob("**/*.raw")
    if "ek80_new_file" in metafunc.fixturenames:
        metafunc.parametrize(
            "ek80_new_file", ek80_new_files, ids=lambda f: str(f.name)
        )


@pytest.fixture
def ek80_new_file(request):
    return request.param


# raw_path_simrad  = ['./echopype/test_data/ek80/simrad/EK80_SimradEcho_WC381_Sequential-D20150513-T090935.raw',
#                     './echopype/test_data/ek80/simrad/EK80_SimradEcho_WC381_Sequential-D20150513-T091004.raw',
#                     './echopype/test_data/ek80/simrad/EK80_SimradEcho_WC381_Sequential-D20150513-T091034.raw',
#                     './echopype/test_data/ek80/simrad/EK80_SimradEcho_WC381_Sequential-D20150513-T091105.raw']
# raw_paths = ['./echopype/test_data/ek80/Summer2018--D20180905-T033113.raw',
#              './echopype/test_data/ek80/Summer2018--D20180905-T033258.raw']  # Multiple files (CW and BB)

def check_env_xml(echodata):
    # check environment xml datagram

    # check env vars
    env_vars = {
        "sound_velocity_source": ["Manual", "Calculated"],
        "transducer_name": ["Unknown"],
    }
    for env_var, expected_env_var_values in env_vars.items():
        assert env_var in echodata["Environment"]
        assert echodata["Environment"][env_var].dims == ("time1",)
        assert all([env_var_value in expected_env_var_values for env_var_value in echodata["Environment"][env_var]])
    assert "transducer_sound_speed" in echodata["Environment"]
    assert echodata["Environment"]["transducer_sound_speed"].dims == ("time1",)
    assert (1480 <= echodata["Environment"]["transducer_sound_speed"]).all() and (echodata["Environment"]["transducer_sound_speed"] <= 1500).all()
    assert "sound_velocity_profile" in echodata["Environment"]
    assert echodata["Environment"]["sound_velocity_profile"].dims == ("time1", "sound_velocity_profile_depth")
    assert (1470 <= echodata["Environment"]["sound_velocity_profile"]).all() and (echodata["Environment"]["sound_velocity_profile"] <= 1500).all()

    # check env dims
    assert "time1" in echodata["Environment"]
    assert "sound_velocity_profile_depth"
    assert np.array_equal(echodata["Environment"]["sound_velocity_profile_depth"], [1, 1000])

    # check a subset of platform variables. plat_vars specifies a list of possible, expected scalar values
    # for each variable. The variables from the EchoData object are tested against this dictionary
    # to verify their presence and their scalar values
    plat_vars = {
        "drop_keel_offset": [np.nan, 0, 7.5],
        "drop_keel_offset_is_manual": [0, 1],
        "water_level": [0],
        "water_level_draft_is_manual": [0, 1]
    }
    for plat_var, expected_plat_var_values in plat_vars.items():
        assert plat_var in echodata["Platform"]
        if np.isnan(expected_plat_var_values).all():
            assert np.isnan(echodata["Platform"][plat_var]).all()
        else:
            assert echodata["Platform"][plat_var] in expected_plat_var_values

    # check plat dims
    assert "time1" in echodata["Platform"]
    assert "time2" in echodata["Platform"]


def test_convert(ek80_new_file, dump_output_dir):
    print("converting", ek80_new_file)
    echodata = open_raw(raw_file=str(ek80_new_file), sonar_model="EK80")
    echodata.to_netcdf(save_path=dump_output_dir, overwrite=True)

    nc_file = (dump_output_dir / ek80_new_file.name).with_suffix('.nc')
    assert nc_file.is_file() is True

    nc_file.unlink()

    check_env_xml(echodata)


def test_convert_ek80_complex_matlab(ek80_path):
    """Compare parsed EK80 CW power/angle data with Matlab parsed data."""
    ek80_raw_path_bb = str(ek80_path.joinpath('D20170912-T234910.raw'))
    ek80_matlab_path_bb = str(
        ek80_path.joinpath('from_matlab', 'D20170912-T234910_data.mat')
    )

    # Convert file
    echodata = open_raw(raw_file=ek80_raw_path_bb, sonar_model='EK80')

    # check water_level
    assert (echodata["Platform"]["water_level"] == 0).all()

    # Test complex parsed data
    ds_matlab = loadmat(ek80_matlab_path_bb)
    assert np.array_equal(
        (
            echodata["Sonar/Beam_group1"].backscatter_r
            .sel(channel='WBT 549762-15 ES70-7C')
            .isel(ping_time=0)
            .dropna('range_sample').squeeze().values[1:, :]  # squeeze remove ping_time dimension
        ),
            np.real(
                ds_matlab['data']['echodata'][0][0][0, 0]['complexsamples']
        ),  # real part
    )
    assert np.array_equal(
        (
            echodata["Sonar/Beam_group1"].backscatter_i
            .sel(channel='WBT 549762-15 ES70-7C')
            .isel(ping_time=0)
            .dropna('range_sample').squeeze().values[1:, :]  # squeeze remove ping_time dimension
        ),
        np.imag(
            ds_matlab['data']['echodata'][0][0][0, 0]['complexsamples']
        ),  # imag part
    )

    check_env_xml(echodata)
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


def test_convert_ek80_cw_power_angle_echoview(ek80_path):
    """Compare parsed EK80 CW power/angle data with csv exported by EchoView."""
    ek80_raw_path_cw = str(
        ek80_path.joinpath('D20190822-T161221.raw')
    )  # Small file (CW)
    freq_list = [18, 38, 70, 120, 200]
    ek80_echoview_power_csv = [
        ek80_path.joinpath(
            'from_echoview', 'D20190822-T161221', '%dkHz.power.csv' % freq
        )
        for freq in freq_list
    ]
    ek80_echoview_angle_csv = [
        ek80_path.joinpath(
            'from_echoview', 'D20190822-T161221', '%dkHz.angles.points.csv' % freq
        )
        for freq in freq_list
    ]

    # Convert file
    echodata = open_raw(ek80_raw_path_cw, sonar_model='EK80')

    # get indices of sorted frequency_nominal values. This is necessary
    # because the frequency_nominal values are not always in ascending order.
    sorted_freq_ind = np.argsort(echodata["Sonar/Beam_group1"].frequency_nominal)

    # get sorted channel list based on frequency_nominal values
    channel_list = echodata["Sonar/Beam_group1"].channel[sorted_freq_ind.values]

    # check water_level
    assert (echodata["Platform"]["water_level"] == 0).all()

    # Test power
    # single point error in original raw data. Read as -2000 by echopype and -999 by EchoView
    echodata["Sonar/Beam_group1"].backscatter_r[sorted_freq_ind.values[3], 4, 13174] = -999
    for file, chan in zip(ek80_echoview_power_csv, channel_list):
        test_power = pd.read_csv(file, delimiter=';').iloc[:, 13:].values
        assert np.allclose(
            test_power,
            echodata["Sonar/Beam_group1"].backscatter_r.sel(channel=chan).dropna('range_sample'),
            rtol=0,
            atol=1.1e-5,
        )

    # Convert from electrical angles to physical angle [deg]
    major = (
        echodata["Sonar/Beam_group1"]['angle_athwartship']
        * 1.40625
        / echodata["Sonar/Beam_group1"]['angle_sensitivity_athwartship']
        - echodata["Sonar/Beam_group1"]['angle_offset_athwartship']
    )
    minor = (
        echodata["Sonar/Beam_group1"]['angle_alongship']
        * 1.40625
        / echodata["Sonar/Beam_group1"]['angle_sensitivity_alongship']
        - echodata["Sonar/Beam_group1"]['angle_offset_alongship']
    )
    for chan, file in zip(channel_list, ek80_echoview_angle_csv):
        df_angle = pd.read_csv(file)
        # NB: EchoView exported data only has 6 pings, but raw data actually has 7 pings.
        #     The first raw ping (ping 0) was removed in EchoView for some reason.
        #     Therefore the comparison will use ping 1-6.
        for ping_idx in df_angle['Ping_index'].value_counts().index:
            assert np.allclose(
                df_angle.loc[df_angle['Ping_index'] == ping_idx, ' Major'],
                major.sel(channel=chan)
                .isel(ping_time=ping_idx)
                .dropna('range_sample'),
                rtol=0,
                atol=5e-5,
            )
            assert np.allclose(
                df_angle.loc[df_angle['Ping_index'] == ping_idx, ' Minor'],
                minor.sel(channel=chan)
                .isel(ping_time=ping_idx)
                .dropna('range_sample'),
                rtol=0,
                atol=5e-5,
            )

    check_env_xml(echodata)
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
    ]
    for plat_var in zero_plat_vars:
        assert plat_var in echodata["Platform"]
        assert (echodata["Platform"][plat_var] == 0).all()
    assert "transducer_offset_z" in echodata["Platform"]
    assert (echodata["Platform"]["transducer_offset_z"] == 9.15).all()


def test_convert_ek80_complex_echoview(ek80_path):
    """Compare parsed EK80 BB data with csv exported by EchoView."""
    ek80_raw_path_bb = ek80_path.joinpath('D20170912-T234910.raw')
    ek80_echoview_bb_power_csv = ek80_path.joinpath(
        'from_echoview', 'D20170912-T234910', '70 kHz raw power.complex.csv'
    )

    # Convert file
    echodata = open_raw(raw_file=ek80_raw_path_bb, sonar_model='EK80')

    # check water_level
    assert (echodata["Platform"]["water_level"] == 0).all()

    # Test complex parsed data
    df_bb = pd.read_csv(
        ek80_echoview_bb_power_csv, header=None, skiprows=[0]
    )  # averaged across beams
    assert np.allclose(
        echodata["Sonar/Beam_group1"].backscatter_r.sel(channel='WBT 549762-15 ES70-7C')
        .dropna('range_sample')
        .mean(dim='beam'),
        df_bb.iloc[::2, 14:],  # real rows
        rtol=0,
        atol=8e-6,
    )
    assert np.allclose(
        echodata["Sonar/Beam_group1"].backscatter_i.sel(channel='WBT 549762-15 ES70-7C')
        .dropna('range_sample')
        .mean(dim='beam'),
        df_bb.iloc[1::2, 14:],  # imag rows
        rtol=0,
        atol=4e-6,
    )

    check_env_xml(echodata)
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


def test_convert_ek80_cw_bb_in_single_file(ek80_path):
    """Make sure can convert a single EK80 file containing both CW and BB mode data."""
    ek80_raw_path_bb_cw = str(
        ek80_path.joinpath('Summer2018--D20180905-T033113.raw')
    )
    echodata = open_raw(raw_file=ek80_raw_path_bb_cw, sonar_model='EK80')

    # Check there are both Sonar/Beam_group1 and /Sonar/Beam_power groups in the converted file
    assert echodata["Sonar/Beam_group2"]
    assert echodata["Sonar/Beam_group1"]

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
    assert (echodata["Platform"]["water_level"] == 0).all()

    check_env_xml(echodata)


def test_convert_ek80_freq_subset(ek80_path):
    """Make sure we can convert EK80 file with multiple frequency channels off."""
    ek80_raw_path_freq_subset = str(
        ek80_path.joinpath('2019118 group2survey-D20191214-T081342.raw')
    )
    echodata = open_raw(raw_file=ek80_raw_path_freq_subset, sonar_model='EK80')

    # Check if converted output has only 2 frequency channels
    assert echodata["Sonar/Beam_group1"].channel.size == 2

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
    assert (echodata["Platform"]["water_level"] == 0).all()

    check_env_xml(echodata)


def test_convert_ek80_raw4(ek80_path):
    """Make sure we can convert EK80 file with RAW4 datagram.."""
    ek80_raw_path_freq_subset = str(
        ek80_path.joinpath('raw4-D20220514-T172704.raw')
    )
    echodata = open_raw(raw_file=ek80_raw_path_freq_subset, sonar_model='EK80')

    # Check if correct data variables exist in Beam_group1
    assert "transmit_sample" in echodata["Sonar/Beam_group1"]
    for var in ["transmit_pulse_r", "transmit_pulse_i"]:
        assert var in echodata["Sonar/Beam_group1"]
        assert echodata["Sonar/Beam_group1"][var].dims == (
            'channel', 'ping_time', 'transmit_sample'
        )


def test_convert_ek80_no_fil_coeff(ek80_path):
    """Make sure we can convert EK80 file with empty filter coefficients."""
    echodata = open_raw(raw_file=ek80_path.joinpath('D20210330-T123857.raw'), sonar_model='EK80')

    vendor_spec_ds = echodata["Vendor_specific"]

    for t in [WIDE_BAND_TRANS, PULSE_COMPRESS]:
        for p in [FILTER_REAL, FILTER_IMAG, DECIMATION]:
            assert f"{t}_{p}" not in vendor_spec_ds


def test_convert_ek80_mru1(ek80_path):
    """Make sure we can convert EK80 file with MRU1 datagram."""
    ek80_mru1_path = str(ek80_path.joinpath('20231016_Cal_-D20231016-T220322.raw'))
    echodata = open_raw(raw_file=ek80_mru1_path, sonar_model='EK80')
    parser = ParseEK80(str(ek80_mru1_path))
    parser.parse_raw()

    np.all(echodata["Platform"]["pitch"].data == np.array(parser.mru["pitch"]))
    np.all(echodata["Platform"]["roll"].data == np.array(parser.mru["roll"]))
    np.all(echodata["Platform"]["vertical_offset"].data == np.array(parser.mru["heave"]))
    np.all(echodata["Platform"]["heading"].data == np.array(parser.mru["heading"]))
