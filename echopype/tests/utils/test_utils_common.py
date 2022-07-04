import echopype as ep
import pytest


# TODO: consolidate this first section with the same section
# in tests/preprocess/test_preprocess.py where the below came from
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
        Sv_swapped = ep.utils.swap_dims_channel_frequency(Sv)
        _check_swap(Sv, Sv_swapped)
    except Exception as e:
        assert isinstance(e, ValueError) is True
        assert str(e) == dup_freq_valueerror

    MVBS = ep.preprocess.compute_MVBS(Sv)
    try:
        MVBS_swapped = ep.utils.swap_dims_channel_frequency(MVBS)
        _check_swap(Sv, MVBS_swapped)
    except Exception as e:
        assert isinstance(e, ValueError) is True
        assert str(e) == dup_freq_valueerror


def _check_swap(ds, ds_swap):
    assert "channel" in ds.dims
    assert "frequency_nominal" not in ds.dims
    assert "frequency_nominal" in ds_swap.dims
    assert "channel" not in ds_swap.dims
