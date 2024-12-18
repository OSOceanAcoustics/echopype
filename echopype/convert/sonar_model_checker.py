import numpy as np

from .utils.ek_raw_io import RawSimradFile


def is_EK80(raw_file, storage_options):
    """Check if a raw data file is from Simrad EK80 echosounder."""
    with RawSimradFile(raw_file, "r", storage_options=storage_options) as fid:
        config_datagram = fid.read(1)
        config_datagram["timestamp"] = np.datetime64(
            config_datagram["timestamp"].replace(tzinfo=None), "[ns]"
        )

        # Return True if "configuration" exists in config_datagram
        return "configuration" in config_datagram


def is_EK60(raw_file, storage_options):
    """Check if a raw data file is from Simrad EK60 echosounder."""
    with RawSimradFile(raw_file, "r", storage_options=storage_options) as fid:
        config_datagram = fid.read(1)
        config_datagram["timestamp"] = np.datetime64(
            config_datagram["timestamp"].replace(tzinfo=None), "[ns]"
        )

        try:
            # Return True if the sounder name matches "EK60"
            return config_datagram["sounder_name"] in {"ER60", "EK60"}
        except KeyError:
            return False


def is_AZFP6(raw_file):
    """
    Check if the provided file has a .azfp extension.

    Parameters:
    raw_file (str): The name of the file to check.

    Returns:
    bool: True if the file has a .azfp extension, False otherwise.
    """

    # Check if the input is a string
    if not isinstance(raw_file, str):
        return False  # Return False if the input is not a string

    # Use the str.lower() method to check for the .azfp extension
    has_azfp_extension = raw_file.lower().endswith(".azfp")

    # Return the result of the check
    return has_azfp_extension


def is_AZFP(raw_file):
    """
    Check if the specified XML file contains an <InstrumentType> with string="AZFP".

    Parameters:
    raw_file (str): The base name of the XML file (with or without extension).

    Returns:
    bool: True if <InstrumentType> with string="AZFP" is found, False otherwise.
    """

    # Check if the filename ends with .xml or .XML, and strip the extension if it does
    base_filename = raw_file.rstrip(".xml").rstrip(".XML")

    # Create a list of possible filenames with both extensions
    possible_files = [f"{base_filename}.xml", f"{base_filename}.XML"]

    for full_filename in possible_files:
        if os.path.isfile(full_filename):
            try:
                # Parse the XML file
                tree = ET.parse(full_filename)
                root = tree.getroot()

                # Check for <InstrumentType> elements
                for instrument in root.findall(".//InstrumentType"):
                    if instrument.get("string") == "AZFP":
                        return True
            except ET.ParseError:
                print(f"Error parsing the XML file: {full_filename}.")

    return False


def is_AD2CP(raw_file):
    """
    Check if the provided file has a .ad2cp extension.

    Parameters:
    raw_file (str): The name of the file to check.

    Returns:
    bool: True if the file has a .ad2cp extension, False otherwise.
    """

    # Check if the input is a string
    if not isinstance(raw_file, str):
        return False  # Return False if the input is not a string

    # Use the str.lower() method to check for the .ad2cp extension
    has_ad2cp_extension = raw_file.lower().endswith(".ad2cp")

    # Return the result of the check
    return has_ad2cp_extension


def is_ER60(raw_file, storage_options):
    """Check if a raw data file is from Simrad EK60 echosounder."""
    with RawSimradFile(raw_file, "r", storage_options=storage_options) as fid:
        config_datagram = fid.read(1)
        config_datagram["timestamp"] = np.datetime64(
            config_datagram["timestamp"].replace(tzinfo=None), "[ns]"
        )
        # Return True if the sounder name matches "ER60"
        try:
            return config_datagram["sounder_name"] in {"ER60", "EK60"}
        except KeyError:
            return False
