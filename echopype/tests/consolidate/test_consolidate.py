import pathlib

import pytest

import numpy as np
import pandas as pd
import xarray as xr
import scipy.io as io
import echopype as ep
from typing import List
from pathlib import Path


@pytest.fixture(
    params=[
        (
            ("EK60", "DY1002_EK60-D20100318-T023008_rep_freq.raw"),
            "EK60",
            None,
            {},
        ),
        (
            ("EK80_NEW", "D20211004-T233354.raw"),
            "EK80",
            None,
            {'waveform_mode': 'CW', 'encode_mode': 'power'},
        ),
        (
            ("AZFP", "17082117.01A"),
            "AZFP",
            ("AZFP", "17041823.XML"),
            {},
        ),
    ],
    ids=[
        "ek60_dup_freq",
        "ek80_cw_power",
        "azfp",
    ],
)
def test_data_samples(request, test_path):
    (
        filepath,
        sonar_model,
        azfp_xml_path,
        range_kwargs,
    ) = request.param
    path_model, *paths = filepath
    filepath = test_path[path_model].joinpath(*paths)

    if azfp_xml_path is not None:
        path_model, *paths = azfp_xml_path
        azfp_xml_path = test_path[path_model].joinpath(*paths)

    return (
        filepath,
        sonar_model,
        azfp_xml_path,
        range_kwargs,
    )


def _check_swap(ds, ds_swap):
    assert "channel" in ds.dims
    assert "frequency_nominal" not in ds.dims
    assert "frequency_nominal" in ds_swap.dims
    assert "channel" not in ds_swap.dims


def test_swap_dims_channel_frequency(test_data_samples):
    """
    Test swapping dimension/coordinate from channel to frequency_nominal.
    """
    (
        filepath,
        sonar_model,
        azfp_xml_path,
        range_kwargs,
    ) = test_data_samples
    ed = ep.open_raw(filepath, sonar_model, azfp_xml_path)
    if ed.sonar_model.lower() == 'azfp':
        avg_temperature = (
            ed['Environment']['temperature'].mean('time1').values
        )
        env_params = {
            'temperature': avg_temperature,
            'salinity': 27.9,
            'pressure': 59,
        }
        range_kwargs['env_params'] = env_params
        if 'azfp_cal_type' in range_kwargs:
            range_kwargs.pop('azfp_cal_type')

    dup_freq_valueerror = (
        "Duplicated transducer nominal frequencies exist in the file. "
        "Operation is not valid."
    )

    Sv = ep.calibrate.compute_Sv(ed, **range_kwargs)
    try:
        Sv_swapped = ep.consolidate.swap_dims_channel_frequency(Sv)
        _check_swap(Sv, Sv_swapped)
    except Exception as e:
        assert isinstance(e, ValueError) is True
        assert str(e) == dup_freq_valueerror

    MVBS = ep.preprocess.compute_MVBS(Sv)
    try:
        MVBS_swapped = ep.consolidate.swap_dims_channel_frequency(MVBS)
        _check_swap(Sv, MVBS_swapped)
    except Exception as e:
        assert isinstance(e, ValueError) is True
        assert str(e) == dup_freq_valueerror


def _build_ds_Sv(channel, range_sample, ping_time, sample_interval):
    return xr.Dataset(
        data_vars={
            "Sv": ( 
                ("channel", "range_sample", "ping_time"),
                np.random.random((len(channel), range_sample.size, ping_time.size)),
            ),
            "echo_range": (
                ("channel", "range_sample", "ping_time"),
                (
                    np.swapaxes(np.tile(range_sample, (len(channel), ping_time.size, 1)), 1, 2)
                    * sample_interval
                ),
            ),
        },
        coords={
            "channel": channel,
            "range_sample": range_sample,
            "ping_time": ping_time,
        },
    )


def test_add_depth():
    # Build test Sv dataset
    channel = ["channel_0", "channel_1", "channel_2"]
    range_sample = np.arange(100)
    ping_time = pd.date_range(start="2022-08-10T10:00:00", end="2022-08-10T12:00:00", periods=121)
    sample_interval = 0.01
    ds_Sv = _build_ds_Sv(channel, range_sample, ping_time, sample_interval)

    # # no water_level in ds
    # try:
    #     ds_Sv_depth = ep.consolidate.add_depth(ds_Sv)
    # except ValueError:
    #     ...

    # user input water_level
    water_level = 10
    ds_Sv_depth = ep.consolidate.add_depth(ds_Sv, depth_offset=water_level)
    assert ds_Sv_depth["depth"].equals(ds_Sv["echo_range"] + water_level)

    # user input water_level and tilt
    tilt = 15
    ds_Sv_depth = ep.consolidate.add_depth(ds_Sv, depth_offset=water_level, tilt=tilt)
    assert ds_Sv_depth["depth"].equals(ds_Sv["echo_range"] * np.cos(tilt / 180 * np.pi) + water_level)

    # inverted echosounder
    ds_Sv_depth = ep.consolidate.add_depth(ds_Sv, depth_offset=water_level, tilt=tilt, downward=False)
    assert ds_Sv_depth["depth"].equals(-1 * ds_Sv["echo_range"] * np.cos(tilt / 180 * np.pi) + water_level)

    # check attributes
    assert ds_Sv_depth["depth"].attrs == {"long_name": "Depth", "standard_name": "depth"}


def test_add_location(test_path):
    ed = ep.open_raw(
        test_path["EK60"] / "Winter2017-D20170115-T150122.raw",
        sonar_model="EK60"
    )
    ds = ep.calibrate.compute_Sv(ed)

    def _check_var(ds_test):
        assert "latitude" in ds_test
        assert "longitude" in ds_test
        assert "time1" not in ds_test

    ds_all = ep.consolidate.add_location(ds=ds, echodata=ed)
    _check_var(ds_all)

    ds_sel = ep.consolidate.add_location(ds=ds, echodata=ed, nmea_sentence="GGA")
    _check_var(ds_sel)


def _create_array_list_from_echoview_mats(paths_to_echoview_mat: List[pathlib.Path]) -> List[np.ndarray]:
    """
    Opens each mat file in ``paths_to_echoview_mat``, selects the first ``ping_time``,
    and then stores the array in a list.

    Parameters
    ----------
    paths_to_echoview_mat: list of pathlib.Path
        A list of paths corresponding to mat files, where each mat file contains the
        echoview generated angle alongship and athwartship data for a channel

    Returns
    -------
    list of np.ndarray
        A list of numpy arrays generated by choosing the appropriate data from the mat files.
        This list will have the same length as ``paths_to_echoview_mat``
    """

    list_of_mat_arrays = []
    for mat_file in paths_to_echoview_mat:

        # open mat file and grab appropriate data
        list_of_mat_arrays.append(io.loadmat(file_name=mat_file)["P0"]["Data_values"][0][0])

    return list_of_mat_arrays


@pytest.mark.parametrize(
    ("sonar_model", "test_path_key", "raw_file_name", "paths_to_echoview_mat",
     "waveform_mode", "encode_mode", "pulse_compression"),
    [
        (
            "EK60", "EK60", "DY1801_EK60-D20180211-T164025.raw",
            [
                '/Users/brandonreyes/UW_work/Echopype_work/code_playing_around/scripts/wu_jung_split_beam_code/sample_data/DY1801_EK60-D20180211-T164025_echoview_files/DY1801_EK60-D20180211-T164025_angles_T1.mat',
                '/Users/brandonreyes/UW_work/Echopype_work/code_playing_around/scripts/wu_jung_split_beam_code/sample_data/DY1801_EK60-D20180211-T164025_echoview_files/DY1801_EK60-D20180211-T164025_angles_T2.mat',
                '/Users/brandonreyes/UW_work/Echopype_work/code_playing_around/scripts/wu_jung_split_beam_code/sample_data/DY1801_EK60-D20180211-T164025_echoview_files/DY1801_EK60-D20180211-T164025_angles_T3.mat',
                '/Users/brandonreyes/UW_work/Echopype_work/code_playing_around/scripts/wu_jung_split_beam_code/sample_data/DY1801_EK60-D20180211-T164025_echoview_files/DY1801_EK60-D20180211-T164025_angles_T4.mat',
                '/Users/brandonreyes/UW_work/Echopype_work/code_playing_around/scripts/wu_jung_split_beam_code/sample_data/DY1801_EK60-D20180211-T164025_echoview_files/DY1801_EK60-D20180211-T164025_angles_T5.mat'
            ],
            "CW", "power", False
        ),
        (
            "EK80", "EK80_CAL", "2018115-D20181213-T094600.raw",
            [
                '/Users/brandonreyes/UW_work/Echopype_work/code_playing_around/scripts/wu_jung_split_beam_code/sample_data/2018115-D20181213-T094600_echoview_files/2018115-D20181213-T094600_angles_T1.mat',
                '/Users/brandonreyes/UW_work/Echopype_work/code_playing_around/scripts/wu_jung_split_beam_code/sample_data/2018115-D20181213-T094600_echoview_files/2018115-D20181213-T094600_angles_T4.mat',
                '/Users/brandonreyes/UW_work/Echopype_work/code_playing_around/scripts/wu_jung_split_beam_code/sample_data/2018115-D20181213-T094600_echoview_files/2018115-D20181213-T094600_angles_T6.mat',
                '/Users/brandonreyes/UW_work/Echopype_work/code_playing_around/scripts/wu_jung_split_beam_code/sample_data/2018115-D20181213-T094600_echoview_files/2018115-D20181213-T094600_angles_T5.mat'
            ],
            "CW", "complex", False
        ),
        pytest.param(
            "EK80", "EK80_CAL", "2018115-D20181213-T094600.raw",
            [
                '/Users/brandonreyes/UW_work/Echopype_work/code_playing_around/scripts/wu_jung_split_beam_code/sample_data/2018115-D20181213-T094600_echoview_files/2018115-D20181213-T094600_angles_T3_nopc.mat',
                '/Users/brandonreyes/UW_work/Echopype_work/code_playing_around/scripts/wu_jung_split_beam_code/sample_data/2018115-D20181213-T094600_echoview_files/2018115-D20181213-T094600_angles_T2_nopc.mat'
            ],
            "BB", "complex", False,
            marks=pytest.mark.xfail(strict=True, reason="We need to investigate why the echoview data is not matching")
        ),
    ],
    ids=["ek60_CW_power", "ek80_CW_complex", "ek80_BB_complex_no_pulse"]
)
def test_add_splitbeam_angle(sonar_model, test_path_key, raw_file_name, test_path,
                             paths_to_echoview_mat, waveform_mode, encode_mode, pulse_compression):

    # TODO: make sure paths_to_echoview_mat is pathlib so we can get around OS dependency

    # obtain the EchoData object with the data needed for the calculation
    ed = ep.open_raw(test_path[test_path_key] / raw_file_name, sonar_model=sonar_model)

    # compute Sv as it is required for the split-beam angle calculation
    ds_Sv = ep.calibrate.compute_Sv(ed, waveform_mode=waveform_mode, encode_mode=encode_mode)

    # add the split-beam angles to an empty Dataset
    ds_Sv = ep.consolidate.add_splitbeam_angle(ds=ds_Sv, echodata=ed,
                                               waveform_mode=waveform_mode,
                                               encode_mode=encode_mode,
                                               pulse_compression=pulse_compression)

    # obtain corresponding echoview output
    echoview_arr_list = _create_array_list_from_echoview_mats(paths_to_echoview_mat)

    # compare echoview output against computed output for all channels
    for chan_ind in range(len(echoview_arr_list)):

        # grabs the appropriate ds data to compare against
        ds_reduced = ds_Sv.isel(channel=chan_ind, ping_time=0).dropna("range_sample")

        # for some files the echoview data is shifted by one index, here we account for that
        if ds_reduced.angle_alongship.shape == (echoview_arr_list[chan_ind].shape[1], ):
            start = 0
        else:
            start = 1

        # check the computed angle_alongship values against the echoview output
        assert np.allclose(ds_reduced.angle_alongship.values[start:],
                           echoview_arr_list[chan_ind][0, :], rtol=1e-2, atol=1e-3)

        # check the computed angle_alongship values against the echoview output
        assert np.allclose(ds_reduced.angle_athwartship.values[start:],
                           echoview_arr_list[chan_ind][1, :], rtol=1e-2, atol=1e-3)






