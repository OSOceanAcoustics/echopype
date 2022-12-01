"""
Contains functions that are specific to Simrad echo sounders
"""
from .echodata import EchoData


def _check_waveform_encode_mode(echodata: EchoData, waveform_mode: str, encode_mode: str):
    """
    A function to make sure that the user has provided the correct
    ``waveform_mode`` and ``encode_mode`` inputs based off of the
    supplied ``echodata`` object.

    """

    # checks logic of mode inputs without referencing data
    #########################################################################################

    if waveform_mode not in ["CW", "BB"]:
        raise RuntimeError("The input waveform_mode must be either 'CW' or 'BB'!")

    if encode_mode not in ["complex", "power"]:
        raise RuntimeError("The input encode_mode must be either 'complex' or 'power'!")

    # BB has complex data only, but CW can have complex or power data
    if (waveform_mode == "BB") and (encode_mode == "power"):
        raise ValueError("encode_mode='power' not allowed when waveform_mode='BB'!")

    #########################################################################################

    power_data_location = None
    complex_data_location = None

    # checks mode inputs against data
    #########################################################################################

    # EK60-like sensors must have 'power' and 'CW' modes
    if echodata.sonar_model in ["EK60", "ES70"]:

        if waveform_mode != "CW":
            raise RuntimeError("Incorrect waveform_mode input provided!")

        if encode_mode != "power":
            raise RuntimeError("Incorrect encode_mode input provided!")

        # ensure that no complex data exists (this should never be triggered)
        # TODO: is this necessary? I just wanted to provide a check against the data ...
        if "backscatter_i" in echodata["Sonar/Beam_group1"].variables:
            raise RuntimeError(
                "Provided echodata object does not correspond to an EK60-like "
                "sensor, but is labeled as data from an EK60-like sensor!"
            )
        else:
            power_data_location = "Sonar/Beam_group1"

    # EK80-like sensors
    if encode_mode.sonar_model in ["EK80", "ES80", "EA640"]:

        if waveform_mode == "BB":

            # check BB waveform_mode, BB must always have complex data, only 1 beam group,
            # and frequency_start variable in the beam group
            if waveform_mode == "BB" and "frequency_start" not in echodata["Sonar/Beam_group1"]:
                raise ValueError("waveform_mode='BB', but broadband data not found!")
            elif "backscatter_i" not in echodata["Sonar/Beam_group1"].variables:
                raise RuntimeError("waveform_mode='BB', but complex data does not exist!")
            elif echodata["Sonar/Beam_group2"] is not None:
                raise RuntimeError("Sonar/Beam_group2 should not exist for waveform_mode='BB'!")
            else:
                complex_data_location = "Sonar/Beam_group1"

        else:

            # CW can have complex or power data, so we just need to make sure that
            # 1) complex samples always exist in Sonar/Beam_group1
            # 2) power samples are in Sonar/Beam_group1 if only one beam group exists
            # 3) power samples are in Sonar/Beam_group2 if two beam groups exist
            if echodata["Sonar/Beam_group2"] is None:

                if encode_mode == "power":
                    # power samples must be in Sonar/Beam_group1 (thus no complex samples)
                    if "backscatter_i" in echodata["Sonar/Beam_group1"].variables:
                        raise RuntimeError(
                            "Data provided does not correspond to encode_mode='power'!"
                        )
                    else:
                        power_data_location = "Sonar/Beam_group1"
                elif encode_mode == "complex":
                    # complex samples must be in Sonar/Beam_group1
                    if "backscatter_i" not in echodata["Sonar/Beam_group1"].variables:
                        raise RuntimeError(
                            "Data provided does not correspond to encode_mode='complex'!"
                        )
                    else:
                        complex_data_location = "Sonar/Beam_group1"

            else:

                # complex should be in Sonar/Beam_group1 and power in Sonar/Beam_group2
                if "backscatter_i" not in echodata["Sonar/Beam_group1"].variables:
                    raise RuntimeError(
                        "Complex data does not exist in Sonar/Beam_group1, "
                        "input echodata object must be incorrectly constructed!"
                    )
                elif "backscatter_r" not in echodata["Sonar/Beam_group2"].variables:
                    raise RuntimeError(
                        "Power data does not exist in Sonar/Beam_group2, "
                        "input echodata object must be incorrectly constructed!"
                    )
                else:
                    complex_data_location = "Sonar/Beam_group1"
                    power_data_location = "Sonar/Beam_group2"
    #########################################################################################

    print(complex_data_location)
    print(power_data_location)
