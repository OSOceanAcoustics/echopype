import echopype
import echopype.visualize
from echopype.testing import TEST_DATA_FOLDER
from echopype.calibrate.calibrate_ek import CalibrateEK60

import pytest
from xarray.plot.facetgrid import FacetGrid
from matplotlib.collections import QuadMesh
import xarray as xr
import numpy as np

ek60_path = TEST_DATA_FOLDER / "ek60"
ek80_path = TEST_DATA_FOLDER / "ek80_new"
azfp_path = TEST_DATA_FOLDER / "azfp"
ad2cp_path = TEST_DATA_FOLDER / "ad2cp"

param_args = ("filepath", "sonar_model", "azfp_xml_path", "range_kwargs")
param_testdata = [
    (
        ek60_path / "ncei-wcsd" / "Summer2017-D20170719-T211347.raw",
        "EK60",
        None,
        {},
    ),
    (
        ek60_path / "DY1002_EK60-D20100318-T023008_rep_freq.raw",
        "EK60",
        None,
        {},
    ),
    (
        ek80_path / "echopype-test-D20211004-T235930.raw",
        "EK80",
        None,
        {'waveform_mode': 'BB', 'encode_mode': 'complex'},
    ),
    (
        ek80_path / "D20211004-T233354.raw",
        "EK80",
        None,
        {'waveform_mode': 'CW', 'encode_mode': 'power'},
    ),
    (
        ek80_path / "D20211004-T233115.raw",
        "EK80",
        None,
        {'waveform_mode': 'CW', 'encode_mode': 'complex'},
    ),
    (
        azfp_path / "17082117.01A",
        "AZFP",
        azfp_path / "17041823.XML",
        {},
    ),  # Will always need env variables
    pytest.param(
        ad2cp_path / "raw" / "090" / "rawtest.090.00001.ad2cp",
        "AD2CP",
        None,
        {},
        marks=pytest.mark.xfail(
            run=False,
            reason="Not supported at the moment",
        ),
    ),
]


@pytest.mark.parametrize(param_args, param_testdata)
def test_plot_multi(
    filepath,
    sonar_model,
    azfp_xml_path,
    range_kwargs,
):
    # TODO: Need to figure out how to compare the actual rendered plots
    ed = echopype.open_raw(filepath, sonar_model, azfp_xml_path)
    plots = echopype.visualize.create_echogram(ed)
    assert isinstance(plots, list) is True
    assert all(isinstance(plot, FacetGrid) for plot in plots) is True


@pytest.mark.parametrize(param_args, param_testdata)
def test_plot_single(
    filepath,
    sonar_model,
    azfp_xml_path,
    range_kwargs,
):
    # TODO: Need to figure out how to compare the actual rendered plots
    ed = echopype.open_raw(filepath, sonar_model, azfp_xml_path)
    plots = echopype.visualize.create_echogram(
        ed, channel=ed["Sonar/Beam_group1"].channel[0].values
    )
    assert isinstance(plots, list) is True
    if (
        sonar_model.lower() == 'ek80'
        and range_kwargs['encode_mode'] == 'complex'
    ):
        assert all(isinstance(plot, FacetGrid) for plot in plots) is True
    else:
        assert all(isinstance(plot, QuadMesh) for plot in plots) is True


@pytest.mark.parametrize(param_args, param_testdata)
def test_plot_multi_get_range(
    filepath,
    sonar_model,
    azfp_xml_path,
    range_kwargs,
):
    # TODO: Need to figure out how to compare the actual rendered plots
    ed = echopype.open_raw(filepath, sonar_model, azfp_xml_path)
    if ed.sonar_model.lower() == 'azfp':
        avg_temperature = ed["Environment"]['temperature'].values.mean()
        env_params = {
            'temperature': avg_temperature,
            'salinity': 27.9,
            'pressure': 59,
        }
        range_kwargs['env_params'] = env_params
    plots = echopype.visualize.create_echogram(
        ed, get_range=True, range_kwargs=range_kwargs
    )
    assert isinstance(plots, list) is True
    assert all(isinstance(plot, FacetGrid) for plot in plots) is True

    # Beam shape check
    if (
        sonar_model.lower() == 'ek80'
        and range_kwargs['encode_mode'] == 'complex'
    ):
        assert plots[0].axes.shape[-1] > 1
    else:
        assert plots[0].axes.shape[-1] == 1

    # Channel shape check
    assert ed["Sonar/Beam_group1"].channel.shape[0] == len(plots)


@pytest.mark.parametrize(param_args, param_testdata)
def test_plot_Sv(
    filepath,
    sonar_model,
    azfp_xml_path,
    range_kwargs,
):
    # TODO: Need to figure out how to compare the actual rendered plots
    ed = echopype.open_raw(filepath, sonar_model, azfp_xml_path)
    if ed.sonar_model.lower() == 'azfp':
        avg_temperature = ed["Environment"]['temperature'].values.mean()
        env_params = {
            'temperature': avg_temperature,
            'salinity': 27.9,
            'pressure': 59,
        }
        range_kwargs['env_params'] = env_params
        if 'azfp_cal_type' in range_kwargs:
            range_kwargs.pop('azfp_cal_type')
    Sv = echopype.calibrate.compute_Sv(ed, **range_kwargs)
    plots = echopype.visualize.create_echogram(Sv)
    assert isinstance(plots, list) is True
    assert all(isinstance(plot, FacetGrid) for plot in plots) is True


@pytest.mark.parametrize(param_args, param_testdata)
def test_plot_mvbs(
    filepath,
    sonar_model,
    azfp_xml_path,
    range_kwargs,
):

    # TODO: Need to figure out how to compare the actual rendered plots
    ed = echopype.open_raw(filepath, sonar_model, azfp_xml_path)
    if ed.sonar_model.lower() == 'azfp':
        avg_temperature = ed["Environment"]['temperature'].values.mean()
        env_params = {
            'temperature': avg_temperature,
            'salinity': 27.9,
            'pressure': 59,
        }
        range_kwargs['env_params'] = env_params
        if 'azfp_cal_type' in range_kwargs:
            range_kwargs.pop('azfp_cal_type')
    Sv = echopype.calibrate.compute_Sv(ed, **range_kwargs)
    mvbs = echopype.commongrid.compute_MVBS(Sv, ping_time_bin='10S')

    plots = []
    try:
        plots = echopype.visualize.create_echogram(mvbs)
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "Ping time must have a length that is greater or equal to 2"  # noqa

    if len(plots) > 0:
        assert all(isinstance(plot, FacetGrid) for plot in plots) is True


@pytest.mark.parametrize(
    ("vertical_offset", "expect_warning"),
    [
        (True, False),
        ([True], True),
        (False, True),
        (xr.DataArray(np.array(50.0)), False),
        ([10, 30.5], False),
        (10, False),
        (30.5, False),
    ],
)
def test_vertical_offset_echodata(vertical_offset, expect_warning, caplog):
    from echopype.echodata import EchoData
    from echopype.visualize.api import _add_vertical_offset
    echopype.verbose()

    filepath = ek60_path / "ncei-wcsd" / "Summer2017-D20170719-T211347.raw"
    sonar_model = "EK60"
    range_kwargs = {}

    echodata = echopype.open_raw(
        sonar_model=sonar_model, raw_file=filepath, xml_path=None
    )

    cal_obj = CalibrateEK60(
        echodata=echodata,
        env_params=range_kwargs.get("env_params", {}),
        cal_params=None,
        ecs_file=None,
    )
    range_in_meter = cal_obj.range_meter

    single_array = range_in_meter.sel(channel='GPT  18 kHz 009072058c8d 1-1 ES18-11').isel(ping_time=0).values
    no_input_vertical_offset = False

    if isinstance(vertical_offset, list):
        vertical_offset = vertical_offset[0]
        echodata["Platform"] = echodata["Platform"].drop_vars('vertical_offset')
        no_input_vertical_offset = True

    if isinstance(vertical_offset, xr.DataArray):
        original_array = single_array + vertical_offset.values
    elif isinstance(vertical_offset, bool) and vertical_offset is True:
        if not no_input_vertical_offset:
            original_array = (
                single_array
                + echodata["Platform"].vertical_offset.isel(time2=0).values
            )
        else:
            original_array = single_array
    elif vertical_offset is not False and isinstance(vertical_offset, (int, float)):
        original_array = single_array + vertical_offset
    else:
        original_array = single_array

    results = None
    try:
        results = _add_vertical_offset(
            range_in_meter=range_in_meter,
            vertical_offset=vertical_offset,
            data_type=EchoData,
            platform_data=echodata["Platform"],
        )
        if expect_warning:
            assert 'WARNING' in caplog.text
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == 'vertical_offset must have any of these dimensions: ping_time, range_sample'  # noqa

    if isinstance(results, xr.DataArray):
        final_array = results.sel(channel='GPT  18 kHz 009072058c8d 1-1 ES18-11').isel(ping_time=0).values
        print(f"original_array = {original_array}")
        print(f"results = {results}")
        assert np.array_equal(original_array, final_array)


@pytest.mark.parametrize(
    ("vertical_offset", "expect_warning"),
    [
        (True, True),
        (False, True),
        (xr.DataArray(np.array(50.0)), False),
        (10, False),
        (30.5, False),
    ],
)
def test_vertical_offset_Sv_dataset(vertical_offset, expect_warning, caplog):
    from echopype.visualize.api import _add_vertical_offset
    echopype.verbose()

    filepath = ek60_path / "ncei-wcsd" / "Summer2017-D20170719-T211347.raw"
    sonar_model = "EK60"
    range_kwargs = {}

    echodata = echopype.open_raw(
        sonar_model=sonar_model, raw_file=filepath, xml_path=None
    )
    Sv = echopype.calibrate.compute_Sv(echodata, **range_kwargs)
    ds = Sv.set_coords('echo_range')
    range_in_meter = ds.echo_range

    single_array = range_in_meter.sel(channel='GPT  18 kHz 009072058c8d 1-1 ES18-11').isel(ping_time=0).values

    if isinstance(vertical_offset, xr.DataArray):
        original_array = single_array + vertical_offset.values
    elif not isinstance(vertical_offset, bool) and isinstance(vertical_offset, (int, float)):
        original_array = single_array + vertical_offset
    else:
        original_array = single_array

    results = None
    try:
        results = _add_vertical_offset(
            range_in_meter=range_in_meter,
            vertical_offset=vertical_offset,
            data_type=xr.Dataset,
        )

        if expect_warning:
            assert 'WARNING' in caplog.text
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == 'vertical_offset must have any of these dimensions: ping_time, range_sample'  # noqa

    if isinstance(results, xr.DataArray):
        final_array = results.sel(channel='GPT  18 kHz 009072058c8d 1-1 ES18-11').isel(ping_time=0).values

        assert np.array_equal(original_array, final_array)
