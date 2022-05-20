import echopype
import echopype.visualize
from echopype.testing import TEST_DATA_FOLDER

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
    # pytest.param(
    (
        ek60_path / "DY1002_EK60-D20100318-T023008_rep_freq.raw",
        "EK60",
        None,
        {},
        # marks=pytest.mark.xfail(
        #     run=False,
        #     reason="Repeated frequencies not supported at the moment.",
        # ),
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
        ed, channel=ed.beam.channel[0].values
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
        avg_temperature = (
            ed.environment['temperature'].mean('ping_time').values
        )
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
    assert ed.beam.channel.shape[0] == len(plots)


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
        avg_temperature = (
            ed.environment['temperature'].mean('ping_time').values
        )
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
        avg_temperature = (
            ed.environment['temperature'].mean('ping_time').values
        )
        env_params = {
            'temperature': avg_temperature,
            'salinity': 27.9,
            'pressure': 59,
        }
        range_kwargs['env_params'] = env_params
        if 'azfp_cal_type' in range_kwargs:
            range_kwargs.pop('azfp_cal_type')
    Sv = echopype.calibrate.compute_Sv(ed, **range_kwargs)
    mvbs = echopype.preprocess.compute_MVBS(Sv, ping_time_bin='10S')

    plots = []
    try:
        plots = echopype.visualize.create_echogram(mvbs)
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "Ping time must have a length that is greater or equal to 2"  # noqa

    if len(plots) > 0:
        assert all(isinstance(plot, FacetGrid) for plot in plots) is True


@pytest.mark.parametrize(
    ("water_level", "expect_warning"),
    [
        (True, False),
        ([True], True),
        (False, True),
        (xr.DataArray(np.array(50.0)).expand_dims({'channel': 3}), False),
        (xr.DataArray(np.array(50.0)), False),
        (10, False),
        (30.5, False),
    ],
)
def test_water_level_echodata(water_level, expect_warning):
    from echopype.echodata import EchoData
    from echopype.visualize.api import _add_water_level

    filepath = ek60_path / "ncei-wcsd" / "Summer2017-D20170719-T211347.raw"
    sonar_model = "EK60"
    range_kwargs = {}

    echodata = echopype.open_raw(
        sonar_model=sonar_model, raw_file=filepath, xml_path=None
    )

    range_in_meter = echodata.compute_range(
        env_params=range_kwargs.get('env_params', {}),
        azfp_cal_type=range_kwargs.get('azfp_cal_type', None),
        ek_waveform_mode=range_kwargs.get('waveform_mode', 'CW'),
        ek_encode_mode=range_kwargs.get('encode_mode', 'power'),
    )

    single_array = range_in_meter.sel(channel='GPT  18 kHz 009072058c8d 1-1 ES18-11',
                                      ping_time='2017-07-19T21:13:47.984999936').values
    no_input_water_level = False

    if isinstance(water_level, list):
        water_level = water_level[0]
        echodata.platform = echodata.platform.drop_vars('water_level')
        no_input_water_level = True

    if isinstance(water_level, xr.DataArray):
        if 'channel' in water_level.dims:
            original_array = single_array + water_level.isel(channel=0).values
    elif isinstance(water_level, bool) and water_level is True:
        if no_input_water_level is False:
            original_array = (
                single_array
                + echodata.platform.water_level.sel(channel='GPT  18 kHz 009072058c8d 1-1 ES18-11',
                                                    ping_time='2017-07-19T21:13:47.984999936').values
            )
        else:
            original_array = single_array
    elif water_level is not False and isinstance(water_level, (int, float)):
        original_array = single_array + water_level
    else:
        original_array = single_array

    results = None
    try:
        if expect_warning:
            with pytest.warns(UserWarning):
                results = _add_water_level(
                    range_in_meter=range_in_meter,
                    water_level=water_level,
                    data_type=EchoData,
                    platform_data=echodata.platform,
                )
        else:
            results = _add_water_level(
                range_in_meter=range_in_meter,
                water_level=water_level,
                data_type=EchoData,
                platform_data=echodata.platform,
            )
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == 'Water level must have any of these dimensions: channel, ping_time, range_sample'  # noqa

    if isinstance(results, xr.DataArray):
        final_array = results.sel(channel='GPT  18 kHz 009072058c8d 1-1 ES18-11',
                                  ping_time='2017-07-19T21:13:47.984999936').values

        assert np.array_equal(original_array, final_array)


@pytest.mark.parametrize(
    ("water_level", "expect_warning"),
    [
        (True, True),
        (False, True),
        (xr.DataArray(np.array(50.0)).expand_dims({'channel': 3}), False),
        (xr.DataArray(np.array(50.0)), False),
        (10, False),
        (30.5, False),
    ],
)
def test_water_level_Sv_dataset(water_level, expect_warning):
    from echopype.visualize.api import _add_water_level

    filepath = ek60_path / "ncei-wcsd" / "Summer2017-D20170719-T211347.raw"
    sonar_model = "EK60"
    range_kwargs = {}

    echodata = echopype.open_raw(
        sonar_model=sonar_model, raw_file=filepath, xml_path=None
    )
    Sv = echopype.calibrate.compute_Sv(echodata, **range_kwargs)
    ds = Sv.set_coords('echo_range')
    range_in_meter = ds.echo_range

    single_array = range_in_meter.sel(channel='GPT  18 kHz 009072058c8d 1-1 ES18-11',
                                      ping_time='2017-07-19T21:13:47.984999936').values

    if isinstance(water_level, xr.DataArray):
        if 'channel' in water_level.dims:
            original_array = single_array + water_level.isel(channel=0).values
    elif not isinstance(water_level, bool) and isinstance(water_level, (int, float)):
        original_array = single_array + water_level
    else:
        original_array = single_array

    results = None
    try:
        if expect_warning:
            with pytest.warns(UserWarning):
                results = _add_water_level(
                    range_in_meter=range_in_meter,
                    water_level=water_level,
                    data_type=xr.Dataset,
                )
        else:
            results = _add_water_level(
                range_in_meter=range_in_meter,
                water_level=water_level,
                data_type=xr.Dataset,
            )
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == 'Water level must have any of these dimensions: channel, ping_time, range_sample'  # noqa

    if isinstance(results, xr.DataArray):
        final_array = results.sel(channel='GPT  18 kHz 009072058c8d 1-1 ES18-11',
                                  ping_time='2017-07-19T21:13:47.984999936').values

        assert np.array_equal(original_array, final_array)
