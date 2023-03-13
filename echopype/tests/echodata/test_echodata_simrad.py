"""
Tests functions contained within echodata/simrad.py
"""
import pytest
from echopype.echodata.simrad import retrieve_correct_beam_group, check_input_args_combination


@pytest.mark.parametrize(
    ("waveform_mode", "encode_mode", "pulse_compression"),
    [
        pytest.param("CW", "comp_power", None,
                     marks=pytest.mark.xfail(strict=True,
                                             reason='This test should fail since comp_power '
                                                    'is not an acceptable choice for encode_mode.')),
        pytest.param("CB", None, None,
                     marks=pytest.mark.xfail(strict=True,
                                             reason='This test should fail since CB is not an '
                                                    'acceptable choice for waveform_mode.')),
        pytest.param("BB", "power", None,
                     marks=pytest.mark.xfail(strict=True,
                                             reason='This test should fail since BB and power is '
                                                    'not an acceptable combination.')),
        pytest.param("BB", "power", True,
                     marks=pytest.mark.xfail(strict=True,
                                             reason='This test should fail since BB and complex '
                                                    'must be used if pulse_compression is True.')),
        pytest.param("CW", "complex", True,
                     marks=pytest.mark.xfail(strict=True,
                                             reason='This test should fail since BB and complex '
                                                    'must be used if pulse_compression is True.')),
        pytest.param("CW", "power", True,
                     marks=pytest.mark.xfail(strict=True,
                                             reason='This test should fail since BB and complex '
                                                    'must be used if pulse_compression is True.')),
        ("CW", "complex", False),
        ("CW", "power", False),
        ("BB", "complex", False),
        ("BB", "complex", True),

    ],
    ids=["incorrect_encode_mode", "incorrect_waveform_mode", "BB_power_combo",
         "BB_power_pc_True", "CW_complex_pc_True", "CW_power_pc_True", "CW_complex_pc_False",
         "CW_power_pc_False", "BB_complex_pc_False", "BB_complex_pc_True"]
)
def test_check_input_args_combination(waveform_mode: str, encode_mode: str,
                                      pulse_compression: bool):
    """
    Ensures that ``check_input_args_combination`` functions correctly when
    provided various combinations of the input parameters.

    Parameters
    ----------
    waveform_mode: str
        Type of transmit waveform
    encode_mode: str
        Type of encoded return echo data
    pulse_compression: bool
        States whether pulse compression should be used
    """

    check_input_args_combination(waveform_mode, encode_mode, pulse_compression)


def test_retrieve_correct_beam_group():

    # TODO: create this test once we are happy with the form of retrieve_correct_beam_group

    pytest.skip("We need to add tests for retrieve_correct_beam_group!")
