import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat
import echopype as ep
from echopype.calibrate.env_params_old import EnvParams
import xarray as xr
import dask.array as da


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


@pytest.fixture
def ek80_ext_path(test_path):
    return test_path['EK80_EXT']


def test_compute_Sv_returns_water_level(ek60_path):

    # get EchoData object that has the water_level variable under platform and compute Sv of it
    ed = ep.open_raw(ek60_path / "ncei-wcsd" / "Summer2017-D20170620-T011027.raw", "EK60")
    ds_Sv = ep.calibrate.compute_Sv(ed)

    # make sure the returned Dataset has water_level and throw an assertion error if the
    # EchoData object does not have water_level (just in case we remove it from the file
    # used in the future)
    assert 'water_level' in ed["Platform"].data_vars.keys()
    assert 'water_level' in ds_Sv.data_vars


@pytest.mark.integration
def test_compute_Sv_ek60_echoview(ek60_path):
    # constant range_sample
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
    # TODO: resolve: pydevd warning: Computing repr of channels (list) was slow (took 0.29s)
    assert np.allclose(
        test_Sv[:, :, 7:],
        ds_Sv.Sv.isel(ping_time=slice(None, 10), range_sample=slice(8, None)),
        atol=1e-8
    )


@pytest.mark.integration
def test_compute_Sv_ek60_matlab(ek60_path):
    ek60_raw_path = str(
        ek60_path.joinpath('DY1801_EK60-D20180211-T164025.raw')
    )
    ek60_matlab_path = str(
        ek60_path.joinpath('from_matlab', 'DY1801_EK60-D20180211-T164025.mat')
    )

    # Convert file
    echodata = ep.open_raw(ek60_raw_path, sonar_model='EK60')

    # Calibrate to get Sv
    ds_Sv = ep.calibrate.compute_Sv(echodata)
    ds_TS = ep.calibrate.compute_TS(echodata)

    # Load matlab outputs and test

    # matlab outputs were saved using
    #   save('from_matlab/DY1801_EK60-D20180211-T164025.mat', 'data')
    ds_base = loadmat(ek60_matlab_path)

    def check_output(da_cmp, cal_type):
        # ds_base["data"]["pings"][0][0]["Sv"].shape = (1, 5)  [5 channels]
        for seq, ch in enumerate(ds_base["data"]["config"][0][0]["channelid"][0]):
            ep_vals = da_cmp.sel(channel=ch).squeeze().data[:, 8:]  # ignore the first 8 samples
            pyel_vals = ds_base['data']['pings'][0][0][cal_type][0, seq].T[:, 8:]
            assert np.allclose(pyel_vals, ep_vals)

    # Check Sv
    check_output(ds_Sv['Sv'], 'Sv')

    # Check TS
    check_output(ds_TS['TS'], 'Sp')


def test_compute_Sv_ek60_duplicated_freq(ek60_path):

    # TODO: add comparison of actual values in this test

    ek60_raw_path = str(
        ek60_path.joinpath('DY1002_EK60-D20100318-T023008_rep_freq.raw')
    )

    # Convert file
    echodata = ep.open_raw(ek60_raw_path, sonar_model='EK60')

    # Calibrate to get Sv
    ds_Sv = ep.calibrate.compute_Sv(echodata)
    ds_TS = ep.calibrate.compute_TS(echodata)

    assert isinstance(ds_Sv, xr.Dataset)
    assert isinstance(ds_TS, xr.Dataset)


def test_compute_Sv_azfp(azfp_path):
    azfp_01a_path = str(azfp_path.joinpath('17082117.01A'))
    azfp_xml_path = str(azfp_path.joinpath('17041823.XML'))
    azfp_matlab_Sv_path = str(
        azfp_path.joinpath('from_matlab', '17082117_matlab_Output_Sv.mat')
    )
    azfp_matlab_TS_path = str(
        azfp_path.joinpath('from_matlab', '17082117_matlab_Output_TS.mat')
    )

    # Convert to .nc file
    echodata = ep.open_raw(
        raw_file=azfp_01a_path, sonar_model='AZFP', xml_path=azfp_xml_path
    )

    # Calibrate using identical env params as in Matlab ParametersAZFP.m
    # AZFP Matlab code uses average temperature
    avg_temperature = echodata["Environment"]['temperature'].values.mean()

    env_params = {
        'temperature': avg_temperature,
        'salinity': 27.9,
        'pressure': 59,
    }

    ds_Sv = ep.calibrate.compute_Sv(echodata=echodata, env_params=env_params)
    ds_TS = ep.calibrate.compute_TS(echodata=echodata, env_params=env_params)

    # Load matlab outputs and test
    # matlab outputs were saved using
    #   save('from_matlab/17082117_matlab_Output.mat', 'Output')  # data variables
    #   save('from_matlab/17082117_matlab_Par.mat', 'Par')  # parameters

    def check_output(base_path, ds_cmp, cal_type):
        ds_base = loadmat(base_path)
        # print(f"ds_base = {ds_base}")
        cal_type_in_ds_cmp = {
            'Sv': 'Sv',
            'TS': 'TS',  # TS here is TS in matlab outputs
        }
        for fidx in range(4):  # loop through all freq
            assert np.alltrue(
                ds_cmp.echo_range.isel(channel=fidx, ping_time=0).values[None, :]
                == ds_base['Output'][0]['Range'][fidx]
            )
            assert np.allclose(
                ds_cmp[cal_type_in_ds_cmp[cal_type]].isel(channel=fidx).values,
                ds_base['Output'][0][cal_type][fidx],
                atol=1e-13,
                rtol=0,
            )

    # Check Sv
    check_output(base_path=azfp_matlab_Sv_path, ds_cmp=ds_Sv, cal_type='Sv')

    # Check TS
    check_output(base_path=azfp_matlab_TS_path, ds_cmp=ds_TS, cal_type='TS')


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
    ds_TS = ep.calibrate.compute_TS(
        echodata, waveform_mode='CW', encode_mode='complex'
    )
    assert isinstance(ds_TS, xr.Dataset) is True


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
    ds_TS = ep.calibrate.compute_TS(
        echodata, waveform_mode='BB', encode_mode='complex'
    )
    assert isinstance(ds_TS, xr.Dataset) is True


def test_compute_Sv_ek80_CW_power_BB_complex(ek80_path):
    """
    Tests calibration in CW mode data encoded as power samples
    and calibration in BB mode data encoded as complex samples,
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


def test_compute_Sv_ek80_CW_complex_BB_complex(ek80_cal_path, ek80_path):
    """
    Tests calibration for file containing both BB and CW mode data
    with both encoded as complex samples.
    """
    ek80_raw_path = ek80_cal_path / "2018115-D20181213-T094600.raw"  # rx impedance / rx fs / tcvr type
    # ek80_raw_path = ek80_path / "D20170912-T234910.raw"  # rx impedance / rx fs / tcvr type
    # ek80_raw_path = ek80_path / "Summer2018--D20180905-T033113.raw"  # BB only, rx impedance / rx fs / tcvr type
    # ek80_raw_path = ek80_path / "ar2.0-D20201210-T000409.raw"  # CW only, rx impedance / rx fs / tcvr type
    # ek80_raw_path = ek80_path / "saildrone/SD2019_WCS_v05-Phase0-D20190617-T125959-0.raw"  # rx impedance / tcvr type
    # ek80_raw_path = ek80_path / "D20200528-T125932.raw"  # CW only,  WBT MINI, rx impedance / rx fs / tcvr type
    ed = ep.open_raw(ek80_raw_path, sonar_model="EK80")
    # ds_Sv = ep.calibrate.compute_Sv(
    #     ed, waveform_mode="CW", encode_mode="complex"
    # )
    # assert isinstance(ds_Sv, xr.Dataset)
    ds_Sv = ep.calibrate.compute_Sv(
        ed, waveform_mode="BB", encode_mode="complex"
    )
    assert isinstance(ds_Sv, xr.Dataset)


@pytest.mark.integration
def test_compute_Sv_combined_ed_ping_time_extend_past_time1():
    """
    Test computing combined Echodata object when ping time dimension in Beam group
    extends past time1 dimension in Environment group.
    The output Sv dataset should not have any NaN values within any of the time1
    variables derived from the Environment group. Additionally, the output Sv dataset
    should not contain the time1 dimension.
    """
    # Parse RAW files and combine Echodata objects
    raw_list = [
        "echopype/test_data/ek80/pifsc_saildrone/SD_TPOS2023_v03-Phase0-D20230530-T001150-0.raw",
        "echopype/test_data/ek80/pifsc_saildrone/SD_TPOS2023_v03-Phase0-D20230530-T002350-0.raw",
    ]
    ed_list = []
    for raw_file in raw_list:
        ed = ep.open_raw(raw_file, sonar_model="EK80")
        # Modify environment variables so that they are non-uniform across Echodata objects
        ed["Environment"]["acidity"].values = [np.random.uniform(low=7.9, high=8.1)]
        ed["Environment"]["salinity"].values = [np.random.uniform(low=34.0, high=35.0)]
        ed["Environment"]["temperature"].values = [np.random.uniform(low=25.0, high=26.0)]
        ed_list.append(ed)
    ed_combined = ep.combine_echodata(ed_list)

    # Compute Sv
    ds_Sv = ep.calibrate.compute_Sv(
        ed_combined,
        waveform_mode="CW",
        encode_mode="complex"
    )

    # Check that Sv doesn't have time1 coordinate
    assert "time1" not in ds_Sv.coords

    # Define environment related variables
    environment_related_variable_names = [
        "sound_absorption",
        "temperature",
        "salinity",
        "pH",
    ]

    # Iterate through vars
    for env_var_name in environment_related_variable_names:
        env_var = ds_Sv[env_var_name]
        # Check that no NaNs exist
        assert not np.any(np.isnan(env_var.data))

                
@pytest.mark.parametrize(
    "raw_path, sonar_model, xml_path, waveform_mode, encode_mode",
    [
        ("azfp/17031001.01A", "AZFP", "azfp/17030815.XML", None, None),
        ("ek60/DY1801_EK60-D20180211-T164025.raw", "EK60", None, None, None),
        ("ek80/D20170912-T234910.raw", "EK80", None, "BB", "complex"),
        ("ek80/D20230804-T083032.raw", "EK80", None, "CW", "complex"),
        ("ek80/Summer2018--D20180905-T033113.raw", "EK80", None, "CW", "power")
    ]
)
def test_check_echodata_backscatter_size(
    raw_path,
    sonar_model,
    xml_path,
    waveform_mode,
    encode_mode,
    caplog
):
    """Tests for _check_echodata_backscatter_size warning."""
    # Parse Echodata Object
    ed = ep.open_raw(
        raw_file=f"echopype/test_data/{raw_path}",
        sonar_model=sonar_model,
        xml_path=f"echopype/test_data/{xml_path}",
    )

    # Compute environment parameters if AZFP
    env_params = None
    if sonar_model == "AZFP":
        avg_temperature = ed["Environment"]['temperature'].values.mean()
        env_params = {
            'temperature': avg_temperature,
            'salinity': 27.9,
            'pressure': 59,
        }

    # Create calibration object
    cal_obj = ep.calibrate.api.CALIBRATOR[ed.sonar_model](
        ed,
        env_params=env_params,
        cal_params=None,
        ecs_file=None,
        waveform_mode=waveform_mode,
        encode_mode=encode_mode,
    )

    # Replace Beam Group 1 with a mock Dataset with a large (more than 2 GB)
    # backscatter_r array
    if sonar_model == "EK80" and encode_mode == "complex":
        cal_obj.echodata[cal_obj.ed_beam_group] = xr.Dataset(
            {
                "backscatter_r": (
                    ("channel", "beam", "ping_time", "range_sample"),
                    da.random.random((3, 4, 100000, 1000)),
                ),
                "backscatter_i": (
                    ("channel", "beam", "ping_time", "range_sample"),
                    da.random.random((3, 4, 100000, 1000)).astype(np.complex128),
                )
            }
        )
    elif sonar_model == "EK80" and encode_mode == "power":
        cal_obj.echodata[cal_obj.ed_beam_group] = xr.Dataset(
            {
                "backscatter_r": (
                    ("channel", "ping_time", "range_sample"),
                    da.random.random((3, 100000, 1000))
                )
            }
        )  
    elif sonar_model in ["EK60", "AZFP"]:
        cal_obj.echodata["Sonar/Beam_group1"] = xr.Dataset(
            {
                "backscatter_r": (
                    ("channel", "ping_time", "range_sample"),
                    da.random.random((3, 100000, 1000))
                )
            }
        )

    # Turn on logger verbosity
    ep.utils.log.verbose(override=False)

    # Run Backscatter Size check
    cal_obj._check_echodata_backscatter_size()

    # Check that warning message is called
    warning_message = (
        "The Echodata Backscatter Variables are large and can cause memory issues. "
        "Consider modifying compute_Sv workflow: "
        "Prior to `compute_Sv` run `echodata.chunk(CHUNK_DICTIONARY) "
        "and after `compute_Sv` run `ds_Sv.to_zarr(ZARR_STORE, compute=True)`. "
        "This will ensure that the computation is lazily evaluated, "
        "with the results stored directly in a Zarr store on disk, rather then in memory."
    )
    assert warning_message == caplog.records[0].message
    
    # Turn off logger verbosity
    ep.utils.log.verbose(override=True)


@pytest.mark.integration
def test_fm_equals_bb():
    """Check that waveform_mode='BB' and waveform_mode='FM' result in the same Sv/TS."""
    # Open Raw and Compute both Sv and both TS
    ed = ep.open_raw("echopype/test_data/ek80/D20170912-T234910.raw", sonar_model = "EK80")
    ds_Sv_bb = ep.calibrate.compute_Sv(ed, waveform_mode="BB", encode_mode="complex")
    ds_Sv_fm = ep.calibrate.compute_Sv(ed, waveform_mode="FM", encode_mode="complex")
    ds_TS_bb = ep.calibrate.compute_TS(ed, waveform_mode="BB", encode_mode="complex")
    ds_TS_fm = ep.calibrate.compute_TS(ed, waveform_mode="FM", encode_mode="complex")

    # Check that they are equal
    assert ds_Sv_bb.equals(ds_Sv_fm)
    assert ds_TS_bb.equals(ds_TS_fm)
