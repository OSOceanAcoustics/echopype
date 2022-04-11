import pytest
import numpy as np
import pandas as pd
from scipy.io import loadmat
from echopype import open_raw

from echopype.testing import TEST_DATA_FOLDER


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

def test_convert(ek80_new_file, dump_output_dir):
    print("converting", ek80_new_file)
    echodata = open_raw(raw_file=str(ek80_new_file), sonar_model="EK80")
    echodata.to_netcdf(save_path=dump_output_dir, overwrite=True)

    nc_file = (dump_output_dir / ek80_new_file.name).with_suffix('.nc')
    assert nc_file.is_file() is True

    nc_file.unlink()


def test_convert_ek80_complex_matlab(ek80_path):
    """Compare parsed EK80 CW power/angle data with Matlab parsed data."""
    ek80_raw_path_bb = str(ek80_path.joinpath('D20170912-T234910.raw'))
    ek80_matlab_path_bb = str(
        ek80_path.joinpath('from_matlab/D20170912-T234910_data.mat')
    )

    # Convert file
    echodata = open_raw(raw_file=ek80_raw_path_bb, sonar_model='EK80')

    # Test complex parsed data
    ds_matlab = loadmat(ek80_matlab_path_bb)
    assert np.array_equal(
        echodata.beam.backscatter_r.isel(frequency=0, ping_time=0)
        .dropna('range_sample')
        .values[1:, :],
        np.real(
            ds_matlab['data']['echodata'][0][0][0, 0]['complexsamples']
        ),  # real part
    )
    assert np.array_equal(
        echodata.beam.backscatter_i.isel(frequency=0, ping_time=0)
        .dropna('range_sample')
        .values[1:, :],
        np.imag(
            ds_matlab['data']['echodata'][0][0][0, 0]['complexsamples']
        ),  # imag part
    )


def test_convert_ek80_cw_power_angle_echoview(ek80_path):
    """Compare parsed EK80 CW power/angle data with csv exported by EchoView."""
    ek80_raw_path_cw = str(
        ek80_path.joinpath('D20190822-T161221.raw')
    )  # Small file (CW)
    freq_list = [18, 38, 70, 120, 200]
    ek80_echoview_power_csv = [
        ek80_path.joinpath(
            'from_echoview/D20190822-T161221/%dkHz.power.csv' % freq
        )
        for freq in freq_list
    ]
    ek80_echoview_angle_csv = [
        ek80_path.joinpath(
            'from_echoview/D20190822-T161221/%dkHz.angles.points.csv' % freq
        )
        for freq in freq_list
    ]

    # Convert file
    echodata = open_raw(ek80_raw_path_cw, sonar_model='EK80')

    # Test power
    # single point error in original raw data. Read as -2000 by echopype and -999 by EchoView
    echodata.beam.backscatter_r[3, 4, 13174] = -999
    for file, freq in zip(ek80_echoview_power_csv, freq_list):
        test_power = pd.read_csv(file, delimiter=';').iloc[:, 13:].values
        assert np.allclose(
            test_power,
            echodata.beam.backscatter_r.sel(frequency=freq * 1e3).dropna(
                'range_sample'
            ),
            rtol=0,
            atol=1.1e-5,
        )

    # Convert from electrical angles to physical angle [deg]
    major = (
        echodata.beam['angle_athwartship']
        * 1.40625
        / echodata.beam['angle_sensitivity_athwartship']
        - echodata.beam['angle_offset_athwartship']
    )
    minor = (
        echodata.beam['angle_alongship']
        * 1.40625
        / echodata.beam['angle_sensitivity_alongship']
        - echodata.beam['angle_offset_alongship']
    )
    for freq, file in zip(freq_list, ek80_echoview_angle_csv):
        df_angle = pd.read_csv(file)
        # NB: EchoView exported data only has 6 pings, but raw data actually has 7 pings.
        #     The first raw ping (ping 0) was removed in EchoView for some reason.
        #     Therefore the comparison will use ping 1-6.
        for ping_idx in df_angle['Ping_index'].value_counts().index:
            assert np.allclose(
                df_angle.loc[df_angle['Ping_index'] == ping_idx, ' Major'],
                major.sel(frequency=freq * 1e3)
                .isel(ping_time=ping_idx)
                .dropna('range_sample'),
                rtol=0,
                atol=5e-5,
            )
            assert np.allclose(
                df_angle.loc[df_angle['Ping_index'] == ping_idx, ' Minor'],
                minor.sel(frequency=freq * 1e3)
                .isel(ping_time=ping_idx)
                .dropna('range_sample'),
                rtol=0,
                atol=5e-5,
            )


def test_convert_ek80_complex_echoview(ek80_path):
    """Compare parsed EK80 BB data with csv exported by EchoView."""
    ek80_raw_path_bb = ek80_path.joinpath('D20170912-T234910.raw')
    ek80_echoview_bb_power_csv = ek80_path.joinpath(
        'from_echoview/D20170912-T234910/70 kHz raw power.complex.csv'
    )

    # Convert file
    echodata = open_raw(raw_file=ek80_raw_path_bb, sonar_model='EK80')

    # Test complex parsed data
    df_bb = pd.read_csv(
        ek80_echoview_bb_power_csv, header=None, skiprows=[0]
    )  # averaged across beams
    assert np.allclose(
        echodata.beam.backscatter_r.sel(frequency=70e3)
        .dropna('range_sample')
        .mean(dim='beam'),
        df_bb.iloc[::2, 14:],  # real rows
        rtol=0,
        atol=8e-6,
    )
    assert np.allclose(
        echodata.beam.backscatter_i.sel(frequency=70e3)
        .dropna('range_sample')
        .mean(dim='beam'),
        df_bb.iloc[1::2, 14:],  # imag rows
        rtol=0,
        atol=4e-6,
    )


def test_convert_ek80_cw_bb_in_single_file(ek80_path):
    """Make sure can convert a single EK80 file containing both CW and BB mode data."""
    ek80_raw_path_bb_cw = str(
        ek80_path.joinpath('Summer2018--D20180905-T033113.raw')
    )
    echodata = open_raw(raw_file=ek80_raw_path_bb_cw, sonar_model='EK80')

    # Check there are both Sonar/Beam_group1 and /Sonar/Beam_power groups in the converted file
    assert echodata.beam_power is not None
    assert echodata.beam is not None


def test_convert_ek80_freq_subset(ek80_path):
    """Make sure can convert EK80 file with multiple frequency channels off."""
    ek80_raw_path_freq_subset = str(
        ek80_path.joinpath('2019118 group2survey-D20191214-T081342.raw')
    )
    echodata = open_raw(raw_file=ek80_raw_path_freq_subset, sonar_model='EK80')

    # Check if converted output has only 2 frequency channels
    assert echodata.beam.frequency.size == 2
