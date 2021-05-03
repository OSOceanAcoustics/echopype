from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
import echopype as ep
from echopype.calibrate.calibrate_ek import CalibrateEK80

azfp_path = Path('./echopype/test_data/azfp')
ek60_path = Path('./echopype/test_data/ek60')
ek80_path = Path('./echopype/test_data/ek80')


def test_compute_Sv_ek60_echoview():
    ek60_raw_path = str(ek60_path.joinpath('DY1801_EK60-D20180211-T164025.raw'))  # constant range_bin
    ek60_echoview_path = ek60_path.joinpath('from_echoview')

    # Convert file
    echodata = ep.open_raw(ek60_raw_path, sonar_model='EK60')

    # Calibrate to get Sv
    ds_Sv = ep.calibrate.compute_Sv(echodata)

    # Compare with EchoView outputs
    channels = []
    for freq in [18, 38, 70, 120, 200]:
        fname = str(ek60_echoview_path.joinpath('DY1801_EK60-D20180211-T164025-Sv%d.csv' % freq))
        channels.append(pd.read_csv(fname, header=None, skiprows=[0]).iloc[:, 13:])
    test_Sv = np.stack(channels)

    # Echoview data is shifted by 1 sample along range (missing the first sample)
    assert np.allclose(test_Sv[:, :, 7:],
                       ds_Sv.Sv.isel(ping_time=slice(None, 10), range_bin=slice(8, None)), atol=1e-8)


def test_compute_Sv_ek60_matlab():
    ek60_raw_path = str(ek60_path.joinpath('DY1801_EK60-D20180211-T164025.raw'))
    ek60_matlab_path = str(ek60_path.joinpath('from_matlab/DY1801_EK60-D20180211-T164025.mat'))

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
            assert np.allclose(ds_cmp[cal_type].isel(frequency=0).T.values,
                               ds_base['data']['pings'][0][0][cal_type][0, 0],
                               atol=4e-5, rtol=0)  # difference due to use of Single in matlab code
    # Check Sv
    check_output(ds_Sv, 'Sv')

    # Check Sp
    check_output(ds_Sp, 'Sp')


def test_compute_Sv_azfp():
    azfp_01a_path = str(azfp_path.joinpath('17082117.01A'))
    azfp_xml_path = str(azfp_path.joinpath('17041823.XML'))
    azfp_matlab_Sv_path = str(azfp_path.joinpath('from_matlab/17082117_matlab_Output_Sv.mat'))
    azfp_matlab_Sp_path = str(azfp_path.joinpath('from_matlab/17082117_matlab_Output_TS.mat'))

    # Convert to .nc file
    echodata = ep.open_raw(raw_file=azfp_01a_path, sonar_model='AZFP', xml_path=azfp_xml_path)

    # Calibrate using identical env params as in Matlab ParametersAZFP.m
    # AZFP Matlab code uses average temperature
    avg_temperature = echodata.environment['temperature'].mean('ping_time').values
    env_params = {'temperature': avg_temperature, 'salinity': 27.9, 'pressure': 59}

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
                ds_cmp.range.isel(frequency=fidx).values == ds_base['Output'][0]['Range'][fidx]
            )
            assert np.allclose(
                ds_cmp[cal_type_in_ds_cmp[cal_type]].isel(frequency=fidx).values,
                ds_base['Output'][0][cal_type][fidx],
                atol=1e-13, rtol=0
            )
    # Check Sv
    check_output(base_path=azfp_matlab_Sv_path, ds_cmp=ds_Sv, cal_type='Sv')

    # Check Sp
    check_output(base_path=azfp_matlab_Sp_path, ds_cmp=ds_Sp, cal_type='TS')


def test_compute_Sv_ek80_matlab():
    """Compare pulse compressed outputs from echopype and Matlab outputs.

    Unresolved: there is a discrepancy between the range vector due to minRange=0.02 m set in Matlab.
    """
    ek80_raw_path = str(ek80_path.joinpath('D20170912-T234910.raw'))
    ek80_matlab_path = str(ek80_path.joinpath('from_matlab/D20170912-T234910_data.mat'))

    echodata = ep.open_raw(ek80_raw_path, sonar_model='EK80')
    ds_Sv = ep.calibrate.compute_Sv(echodata, waveform_mode='BB', encode_mode='complex')

    # TODO: resolve discrepancy in range between echopype and Matlab code
    ds_matlab = loadmat(ek80_matlab_path)
    Sv_70k = ds_Sv.Sv.isel(frequency=0, ping_time=0).dropna('range_bin').values


def test_compute_Sv_ek80_pc_echoview():
    """Compare pulse compressed outputs from echopype and csv exported from EchoView.

    Unresolved: the difference is large and it is not clear why.
    """
    ek80_raw_path = str(ek80_path.joinpath('D20170912-T234910.raw'))
    ek80_bb_pc_test_path = str(ek80_path.joinpath('from_echoview/70 kHz pulse-compressed power.complex.csv'))

    echodata = ep.open_raw(ek80_raw_path, sonar_model='EK80')

    # Create a CalibrateEK80 object to perform pulse compression
    waveform_mode = 'BB'
    cal_obj = CalibrateEK80(echodata, env_params=None, cal_params=None, waveform_mode=waveform_mode)
    cal_obj.compute_range_meter(waveform_mode=waveform_mode, tvg_correction_factor=0)  # compute range [m]
    chirp, _, tau_effective = cal_obj.get_transmit_chirp(waveform_mode=waveform_mode)
    pc = cal_obj.compress_pulse(chirp)
    pc_mean = pc.pulse_compressed_output.isel(frequency=0).mean(dim='quadrant').dropna('range_bin')

    # Read EchoView pc raw power output
    df = pd.read_csv(ek80_bb_pc_test_path, header=None, skiprows=[0])
    df_header = pd.read_csv(ek80_bb_pc_test_path, header=0, usecols=range(14), nrows=0)
    df = df.rename(columns={cc: vv for cc, vv in zip(df.columns, df_header.columns.values)})
    df.columns = df.columns.str.strip()
    df_real = df.loc[df['Component'] == ' Real', :].iloc[:, 14:]

    # Compare only values for range > 0: difference is surprisingly large
    range_meter = cal_obj.range_meter.isel(frequency=0, ping_time=0).values
    first_nonzero_range = np.argwhere(range_meter == 0).squeeze().max()
    assert np.allclose(
        df_real.values[:, first_nonzero_range:pc_mean.values.shape[1]],
        pc_mean.values.real[:, first_nonzero_range:],
        rtol=0,
        atol=1.03e-3
    )


def test_compute_Sv_ek80_CW_complex():
    """Test calibrate CW mode data encoded as complex sam[les.
    """
    ek80_raw_path = str(ek80_path.joinpath('ar2.0-D20201210-T000409.raw'))  # CW complex
    echodata = ep.open_raw(ek80_raw_path, sonar_model='EK80')
    assert ep.calibrate.compute_Sv(echodata, waveform_mode='CW', encode_mode='complex')


def test_compute_Sv_ek80_BB_complex():
    """Test calibrate BB mode data encoded as complex sam[les.
    """
    ek80_raw_path = str(ek80_path.joinpath('ar2.0-D20201209-T235955.raw'))  # CW complex
    echodata = ep.open_raw(ek80_raw_path, sonar_model='EK80')
    assert ep.calibrate.compute_Sv(echodata, waveform_mode='BB', encode_mode='complex')
