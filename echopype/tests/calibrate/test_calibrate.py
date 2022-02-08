import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat
import echopype as ep
from echopype.calibrate.calibrate_ek import CalibrateEK80
from echopype.calibrate.calibrate_base import EnvParams
import xarray as xr


@pytest.fixture
def azfp_path(test_path):
    return test_path['AZFP']


@pytest.fixture
def ek60_path(test_path):
    return test_path['EK60']


@pytest.fixture
def ek80_path(test_path):
    return test_path['EK80']


@pytest.fixture
def ek80_cal_path(test_path):
    return test_path['EK80_CAL']


def test_compute_Sv_ek60_echoview(ek60_path):
    # constant range_bin
    ek60_raw_path = str(
        ek60_path.joinpath('DY1801_EK60-D20180211-T164025.raw')
    )
    ek60_echoview_path = ek60_path.joinpath('from_echoview')

    # Convert file
    echodata = ep.open_raw(ek60_raw_path, sonar_model='EK60')

    # Calibrate to get Sv
    ds_Sv = ep.calibrate.compute_Sv(echodata)

    # Compare with EchoView outputs
    channels = []
    for freq in [18, 38, 70, 120, 200]:
        fname = str(
            ek60_echoview_path.joinpath(
                'DY1801_EK60-D20180211-T164025-Sv%d.csv' % freq
            )
        )
        channels.append(
            pd.read_csv(fname, header=None, skiprows=[0]).iloc[:, 13:]
        )
    test_Sv = np.stack(channels)

    # Echoview data is shifted by 1 sample along range (missing the first sample)
    assert np.allclose(
        test_Sv[:, :, 7:],
        ds_Sv.Sv.isel(ping_time=slice(None, 10), range_bin=slice(8, None)),
        atol=1e-8,
    )


def test_compute_Sv_ek60_matlab(ek60_path):
    ek60_raw_path = str(
        ek60_path.joinpath('DY1801_EK60-D20180211-T164025.raw')
    )
    ek60_matlab_path = str(
        ek60_path.joinpath('from_matlab/DY1801_EK60-D20180211-T164025.mat')
    )

    # Convert file
    echodata = ep.open_raw(ek60_raw_path, sonar_model='EK60')

    # Calibrate to get Sv
    ds_Sv = ep.calibrate.compute_Sv(echodata)
    ds_Sp = ep.calibrate.compute_Sp(echodata)

    # Load matlab outputs and test

    # matlab outputs were saved using
    #   save('from_matlab/DY1801_EK60-D20180211-T164025.mat', 'data')
    ds_base = loadmat(ek60_matlab_path)

    def check_output(ds_cmp, cal_type):
        for fidx in range(5):  # loop through all freq
            assert np.allclose(
                ds_cmp[cal_type].isel(frequency=0).T.values,
                ds_base['data']['pings'][0][0][cal_type][0, 0],
                atol=4e-5,
                rtol=0,
            )  # difference due to use of Single in matlab code

    # Check Sv
    check_output(ds_Sv, 'Sv')

    # Check Sp
    check_output(ds_Sp, 'Sp')


def test_compute_Sv_azfp(azfp_path):
    azfp_01a_path = str(azfp_path.joinpath('17082117.01A'))
    azfp_xml_path = str(azfp_path.joinpath('17041823.XML'))
    azfp_matlab_Sv_path = str(
        azfp_path.joinpath('from_matlab/17082117_matlab_Output_Sv.mat')
    )
    azfp_matlab_Sp_path = str(
        azfp_path.joinpath('from_matlab/17082117_matlab_Output_TS.mat')
    )

    # Convert to .nc file
    echodata = ep.open_raw(
        raw_file=azfp_01a_path, sonar_model='AZFP', xml_path=azfp_xml_path
    )

    # Calibrate using identical env params as in Matlab ParametersAZFP.m
    # AZFP Matlab code uses average temperature
    avg_temperature = (
        echodata.environment['temperature'].mean('ping_time').values
    )
    env_params = {
        'temperature': avg_temperature,
        'salinity': 27.9,
        'pressure': 59,
    }

    ds_Sv = ep.calibrate.compute_Sv(echodata=echodata, env_params=env_params)
    ds_Sp = ep.calibrate.compute_Sp(echodata=echodata, env_params=env_params)

    # Load matlab outputs and test
    # matlab outputs were saved using
    #   save('from_matlab/17082117_matlab_Output.mat', 'Output')  # data variables
    #   save('from_matlab/17082117_matlab_Par.mat', 'Par')  # parameters

    def check_output(base_path, ds_cmp, cal_type):
        ds_base = loadmat(base_path)
        cal_type_in_ds_cmp = {
            'Sv': 'Sv',
            'TS': 'Sp',  # Sp here is TS in matlab outputs
        }
        for fidx in range(4):  # loop through all freq
            assert np.alltrue(
                ds_cmp.range.isel(frequency=fidx).values
                == ds_base['Output'][0]['Range'][fidx]
            )
            assert np.allclose(
                ds_cmp[cal_type_in_ds_cmp[cal_type]]
                .isel(frequency=fidx)
                .values,
                ds_base['Output'][0][cal_type][fidx],
                atol=1e-13,
                rtol=0,
            )

    # Check Sv
    check_output(base_path=azfp_matlab_Sv_path, ds_cmp=ds_Sv, cal_type='Sv')

    # Check Sp
    check_output(base_path=azfp_matlab_Sp_path, ds_cmp=ds_Sp, cal_type='TS')


def test_compute_Sv_ek80_matlab(ek80_path):
    """Compare pulse compressed outputs from echopype and Matlab outputs.

    Unresolved: there is a discrepancy between the range vector due to minRange=0.02 m set in Matlab.
    """
    ek80_raw_path = str(ek80_path.joinpath('D20170912-T234910.raw'))
    ek80_matlab_path = str(
        ek80_path.joinpath('from_matlab/D20170912-T234910_data.mat')
    )

    echodata = ep.open_raw(ek80_raw_path, sonar_model='EK80')
    ds_Sv = ep.calibrate.compute_Sv(
        echodata, waveform_mode='BB', encode_mode='complex'
    )

    # TODO: resolve discrepancy in range between echopype and Matlab code
    ds_matlab = loadmat(ek80_matlab_path)
    Sv_70k = ds_Sv.Sv.isel(frequency=0, ping_time=0).dropna('range_bin').values


def test_compute_Sv_ek80_pc_echoview(ek80_path):
    """Compare pulse compressed outputs from echopype and csv exported from EchoView.

    Unresolved: the difference is large and it is not clear why.
    """
    ek80_raw_path = str(ek80_path.joinpath('D20170912-T234910.raw'))
    ek80_bb_pc_test_path = str(
        ek80_path.joinpath(
            'from_echoview/70 kHz pulse-compressed power.complex.csv'
        )
    )

    echodata = ep.open_raw(ek80_raw_path, sonar_model='EK80')

    # Create a CalibrateEK80 object to perform pulse compression
    cal_obj = CalibrateEK80(
        echodata,
        env_params=None,
        cal_params=None,
        waveform_mode="BB",
        encode_mode="complex",
    )
    cal_obj.compute_range_meter(
        waveform_mode="BB", encode_mode="complex"
    )  # compute range [m]
    chirp, _, tau_effective = cal_obj.get_transmit_chirp(waveform_mode="BB")
    freq_center = (
        echodata.beam["frequency_start"] + echodata.beam["frequency_end"]
    ).dropna(
        dim="frequency"
    ) / 2  # drop those that contain CW samples (nan in freq start/end)
    pc = cal_obj.compress_pulse(chirp, freq_BB=freq_center.frequency)
    pc_mean = (
        pc.pulse_compressed_output.isel(frequency=0)
        .mean(dim='quadrant')
        .dropna('range_bin')
    )

    # Read EchoView pc raw power output
    df = pd.read_csv(ek80_bb_pc_test_path, header=None, skiprows=[0])
    df_header = pd.read_csv(
        ek80_bb_pc_test_path, header=0, usecols=range(14), nrows=0
    )
    df = df.rename(
        columns={
            cc: vv for cc, vv in zip(df.columns, df_header.columns.values)
        }
    )
    df.columns = df.columns.str.strip()
    df_real = df.loc[df['Component'] == ' Real', :].iloc[:, 14:]

    # Compare only values for range > 0: difference is surprisingly large
    range_meter = cal_obj.range_meter.isel(frequency=0, ping_time=0).values
    first_nonzero_range = np.argwhere(range_meter == 0).squeeze().max()
    assert np.allclose(
        df_real.values[:, first_nonzero_range : pc_mean.values.shape[1]],
        pc_mean.values.real[:, first_nonzero_range:],
        rtol=0,
        atol=1.03e-3,
    )


def test_compute_Sv_ek80_CW_complex(ek80_path):
    """Test calibrate CW mode data encoded as complex samples."""
    ek80_raw_path = str(
        ek80_path.joinpath('ar2.0-D20201210-T000409.raw')
    )  # CW complex
    echodata = ep.open_raw(ek80_raw_path, sonar_model='EK80')
    ds_Sv = ep.calibrate.compute_Sv(
        echodata, waveform_mode='CW', encode_mode='complex'
    )
    assert isinstance(ds_Sv, xr.Dataset) is True
    ds_Sp = ep.calibrate.compute_Sp(
        echodata, waveform_mode='CW', encode_mode='complex'
    )
    assert isinstance(ds_Sp, xr.Dataset) is True


def test_compute_Sv_ek80_BB_complex(ek80_path):
    """Test calibrate BB mode data encoded as complex samples."""
    ek80_raw_path = str(
        ek80_path.joinpath('ar2.0-D20201209-T235955.raw')
    )  # CW complex
    echodata = ep.open_raw(ek80_raw_path, sonar_model='EK80')
    ds_Sv = ep.calibrate.compute_Sv(
        echodata, waveform_mode='BB', encode_mode='complex'
    )
    assert isinstance(ds_Sv, xr.Dataset) is True
    ds_Sp = ep.calibrate.compute_Sp(
        echodata, waveform_mode='BB', encode_mode='complex'
    )
    assert isinstance(ds_Sp, xr.Dataset) is True


def test_compute_Sv_ek80_CW_power_BB_complex(ek80_path):
    """
    Tests calibration in CW mode data encoded as power samples
    and calibration in BB mode data encoded as complex seamples,
    while the file contains both CW power and BB complex samples.
    """
    ek80_raw_path = ek80_path / "Summer2018--D20180905-T033113.raw"
    ed = ep.open_raw(ek80_raw_path, sonar_model="EK80")
    ds_Sv = ep.calibrate.compute_Sv(
        ed, waveform_mode="CW", encode_mode="power"
    )
    assert isinstance(ds_Sv, xr.Dataset)
    ds_Sv = ep.calibrate.compute_Sv(
        ed, waveform_mode="BB", encode_mode="complex"
    )
    assert isinstance(ds_Sv, xr.Dataset)


def test_compute_Sv_ek80_CW_complex_BB_complex(ek80_cal_path):
    """
    Tests calibration for file containing both BB and CW mode data
    with both encoded as complex samples.
    """
    ek80_raw_path = ek80_cal_path / "2018115-D20181213-T094600.raw"
    ed = ep.open_raw(ek80_raw_path, sonar_model="EK80")
    ds_Sv = ep.calibrate.compute_Sv(
        ed, waveform_mode="CW", encode_mode="complex"
    )
    assert isinstance(ds_Sv, xr.Dataset)
    ds_Sv = ep.calibrate.compute_Sv(
        ed, waveform_mode="BB", encode_mode="complex"
    )
    assert isinstance(ds_Sv, xr.Dataset)

def test_env_params():
    """
    Tests EnvParams interpolation
    """

    ed = ep.open_raw(ek60_path / "ncei-wcsd/Summer2017-D20170620-T011027.raw", "EK60")

    # stationary
    # since the raw ping_times go up to 1:43 but the env_params ping_time stops at 1:25,
    # values after 1:25 will be extrapolated
    env_params_data = xr.Dataset(
        data_vars={
            "pressure": ("ping_time", np.arange(50)),
            "salinity": ("ping_time", np.arange(50)),
            "temperature": ("ping_time", np.arange(50)),
        },
        coords={
            "ping_time": np.arange("2017-06-20T01:00", "2017-06-20T01:25", np.timedelta64(30, "s"), dtype="datetime64[ns]")
        }
    )
    env_params = EnvParams(env_params_data, "stationary")
    converted_env_params = env_params._apply(ed)
    for var in converted_env_params.values():
        assert np.all(np.diff(var) > 0)
        assert np.all(0 <= var)
        assert np.all(var < 100)
    known_values = {
        "temperature": {
            "2017-06-20T01:10:27.136999936": 20.904566664533334,
            "2017-06-20T01:10:28.149000192": 20.9383000064,
            "2017-06-20T01:10:29.160999936": 20.9720333312,
            "2017-06-20T01:10:30.174000128": 21.005800004266668,
            "2017-06-20T01:10:31.184999936": 21.039499997866667,
            "2017-06-20T01:42:56.995999744": 85.89986665813333,
            "2017-06-20T01:42:58.008999936": 85.9336333312,
            "2017-06-20T01:42:59.020000256": 85.96733334186666,
            "2017-06-20T01:43:00.032000000": 86.00106666666667,
            "2017-06-20T01:43:01.045000192": 86.03483333973334,
        },
        "salinity": {
            "2017-06-20T01:10:27.136999936": 20.904566664533334,
            "2017-06-20T01:10:28.149000192": 20.9383000064,
            "2017-06-20T01:10:29.160999936": 20.9720333312,
            "2017-06-20T01:10:30.174000128": 21.005800004266668,
            "2017-06-20T01:10:31.184999936": 21.039499997866667,
            "2017-06-20T01:42:56.995999744": 85.89986665813333,
            "2017-06-20T01:42:58.008999936": 85.9336333312,
            "2017-06-20T01:42:59.020000256": 85.96733334186666,
            "2017-06-20T01:43:00.032000000": 86.00106666666667,
            "2017-06-20T01:43:01.045000192": 86.0348333397333,
        },
        "pressure": {
            "2017-06-20T01:10:27.136999936": 20.904566664533334,
            "2017-06-20T01:10:28.149000192": 20.9383000064,
            "2017-06-20T01:10:29.160999936": 20.9720333312,
            "2017-06-20T01:10:30.174000128": 21.005800004266668,
            "2017-06-20T01:10:31.184999936": 21.039499997866667,
            "2017-06-20T01:42:56.995999744": 85.89986665813333,
            "2017-06-20T01:42:58.008999936": 85.9336333312,
            "2017-06-20T01:42:59.020000256": 85.96733334186666,
            "2017-06-20T01:43:00.032000000": 86.00106666666667,
            "2017-06-20T01:43:01.045000192": 86.03483333973334,
        }
    }
    for var, values in known_values.items():
        for time, value in values.items():
            assert np.isclose(converted_env_params[var].sel(ping_time=time), value)

    # mobile
    rng = np.random.default_rng(0)
    env_params_data = xr.Dataset(
        data_vars={
            "pressure": ("time", np.arange(100)),
            "salinity": ("time", np.arange(100)),
            "temperature": ("time", np.arange(100)),
        },
        coords={
            "latitude": ("time", rng.random(size=100) + 44),
            "longitude": ("time", rng.random(size=100) - 125),
        }
    )
    env_params = EnvParams(env_params_data, "mobile")
    converted_env_params = env_params._apply(ed)
    for var in converted_env_params.values():
        assert np.all(0 <= var[~np.isnan(var)])
        assert np.all(var[~np.isnan(var)] < 100)
    known_values = {
        "temperature": {
            "2017-06-20T01:10:27.136999936":  np.nan,
            "2017-06-20T01:10:28.149000192":  72.57071056437047,
            "2017-06-20T01:10:29.160999936":  72.56164311204404,
            "2017-06-20T01:10:30.174000128":  72.5641609908268,
            "2017-06-20T01:10:31.184999936":  72.5540675620769,
            "2017-06-20T01:42:56.995999744":  64.78639664394186,
            "2017-06-20T01:42:58.008999936":  64.76543272189699,
            "2017-06-20T01:42:59.020000256":  64.77890258158483,
            "2017-06-20T01:43:00.032000000":  64.76186093048929,
            "2017-06-20T01:43:01.045000192":  64.76763007606817,
        },
        "salinity": {
            "2017-06-20T01:10:27.136999936":  np.nan,
            "2017-06-20T01:10:28.149000192":  72.57071056437047,
            "2017-06-20T01:10:29.160999936":  72.56164311204404,
            "2017-06-20T01:10:30.174000128":  72.5641609908268,
            "2017-06-20T01:10:31.184999936":  72.5540675620769,
            "2017-06-20T01:42:56.995999744":  64.78639664394186,
            "2017-06-20T01:42:58.008999936":  64.76543272189699,
            "2017-06-20T01:42:59.020000256":  64.77890258158483,
            "2017-06-20T01:43:00.032000000":  64.76186093048929,
            "2017-06-20T01:43:01.045000192":  64.76763007606817,
        },
        "pressure": {
            "2017-06-20T01:10:27.136999936": np.nan,
            "2017-06-20T01:10:28.149000192": 72.57071056437047,
            "2017-06-20T01:10:29.160999936": 72.56164311204404,
            "2017-06-20T01:10:30.174000128": 72.5641609908268,
            "2017-06-20T01:10:31.184999936": 72.5540675620769,
            "2017-06-20T01:42:56.995999744": 64.78639664394186,
            "2017-06-20T01:42:58.008999936": 64.76543272189699,
            "2017-06-20T01:42:59.020000256": 64.77890258158483,
            "2017-06-20T01:43:00.032000000": 64.76186093048929,
            "2017-06-20T01:43:01.045000192": 64.76763007606817,
        },
    }
    for var, values in known_values.items():
        for time, value in values.items():
            print(var, time, value)
            assert np.isnan(value) or np.isclose(converted_env_params[var].sel(ping_time=time), value)
