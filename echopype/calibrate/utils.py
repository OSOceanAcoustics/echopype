"""Utilities shared by calibration workflows."""


def check_input_args_combination(
    waveform_mode: str, encode_mode: str, pulse_compression: bool = None
) -> None:
    """Validate waveform, encoding, and pulse-compression options."""
    if waveform_mode not in ["CW", "BB"]:
        raise ValueError("The input waveform_mode must be either 'CW' or 'BB'!")

    if encode_mode not in ["complex", "power"]:
        raise ValueError("The input encode_mode must be either 'complex' or 'power'!")

    if waveform_mode == "BB" and encode_mode == "power":
        raise ValueError(
            "Data from broadband ('BB') transmission must be recorded as complex samples"
        )

    if pulse_compression is not None:
        if pulse_compression and ((waveform_mode != "BB") or (encode_mode != "complex")):
            raise RuntimeError(
                "Pulse compression can only be used with "
                "waveform_mode='BB' and encode_mode='complex'"
            )
