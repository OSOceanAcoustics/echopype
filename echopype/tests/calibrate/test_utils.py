import pytest

from echopype.calibrate.utils import check_input_args_combination


@pytest.mark.parametrize(
    ("waveform_mode", "encode_mode"),
    [("CW", "power"), ("CW", "complex"), ("BB", "complex")],
)
def test_check_input_args_combination_accepts_valid_modes(waveform_mode, encode_mode):
    check_input_args_combination(waveform_mode, encode_mode)


@pytest.mark.parametrize(
    ("waveform_mode", "encode_mode"),
    [("FM", "complex"), ("CW", "raw"), ("BB", "power")],
)
def test_check_input_args_combination_rejects_invalid_modes(waveform_mode, encode_mode):
    with pytest.raises(ValueError):
        check_input_args_combination(waveform_mode, encode_mode)


def test_check_input_args_combination_rejects_invalid_pulse_compression():
    with pytest.raises(RuntimeError, match="Pulse compression"):
        check_input_args_combination("CW", "complex", pulse_compression=True)
