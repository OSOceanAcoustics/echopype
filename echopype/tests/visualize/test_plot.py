import echopype
from echopype.testing import TEST_DATA_FOLDER

import pytest
from xarray.plot.facetgrid import FacetGrid
from matplotlib.collections import QuadMesh

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
    plot = echopype.visualize.create_echogram(ed)
    assert isinstance(plot, FacetGrid) is True


@pytest.mark.parametrize(param_args, param_testdata)
def test_plot_single(
    filepath,
    sonar_model,
    azfp_xml_path,
    range_kwargs,
):
    # TODO: Need to figure out how to compare the actual rendered plots
    ed = echopype.open_raw(filepath, sonar_model, azfp_xml_path)
    plot = echopype.visualize.create_echogram(
        ed, frequency=ed.beam.frequency[0].values
    )
    if (
        sonar_model.lower() == 'ek80'
        and range_kwargs['encode_mode'] == 'complex'
    ):
        assert isinstance(plot, FacetGrid) is True
    else:
        assert isinstance(plot, QuadMesh) is True


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
    plot = echopype.visualize.create_echogram(
        ed, get_range=True, range_kwargs=range_kwargs
    )
    assert isinstance(plot, FacetGrid) is True

    if (
        sonar_model.lower() == 'ek80'
        and range_kwargs['encode_mode'] == 'complex'
    ):
        assert plot.axes.shape[-1] > 1
    else:
        assert plot.axes.shape[-1] == 1

    assert ed.beam.frequency.shape[0] == plot.axes.shape[0]


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
    plot = echopype.visualize.create_echogram(Sv)
    assert isinstance(plot, FacetGrid) is True
