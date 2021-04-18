from .echodata import EchoData
from ..convert.convert import MODELS, CONVERT_PARAMS, Convert


def open_raw(file=None, model=None, xml_path=None, convert_params=None, storage_options=None):
    """Create an EchoData object containing parsed data from a single raw data file.

    The EchoData object can be used for adding metadata and ancillary data
    as well as to serialize the parsed data to zarr or netcdf.

    Parameters
    ----------
    file : str or list
        path to raw data file(s)
    model : str
        model of the sonar instrument
    xml_path : str
        path to XML config file used by AZFP
    convert_params : dict
        parameters (metadata) that may not exist in the raw file
        and need to be added to the converted file
    storage_options : dict
        options for cloud storage
    """
    # Check inputs
    storage_options = storage_options if storage_options is not None else {}
    if model not in MODELS:
        raise ValueError(
            f"Unsupported sonar model: {model}\n" f"Must be one of: {list(MODELS)}"
        )
    # TODO: the if-else below only works for the AZFP vs EK contrast,
    #  but is brittle since it is abusing params by using it implicitly
    if MODELS[model]["xml"]:
        params = xml_path
    else:
        params = "ALL"  # reserved to control if only wants to parse a certain type of datagram

    # Set up echodata object
    echodata = EchoData(
        convert_obj=Convert(
            file=file,
            xml_path=xml_path,
            model=model,
            storage_options=storage_options
        )
    )

    # Parse raw file and organize data into groups
    parser = MODELS[model]["parser"](file, params=params, storage_options=storage_options)
    parser.parse_raw()
    setgrouper = MODELS[model]["set_groups"](
        parser,
        input_file=file,
        output_path=None,
        sonar_model=model,
        params=dict.fromkeys(CONVERT_PARAMS) if convert_params is None else convert_params
    )
    # Top-level date_created varies depending on sonar model
    if model in ["EK60", "EK80"]:
        echodata.top = setgrouper.set_toplevel(
            sonar_model=model,
            date_created=parser.config_datagram['timestamp']
        )
    else:
        echodata.top = setgrouper.set_toplevel(
            sonar_model=model,
            date_created=parser.ping_time[0]
        )
    echodata.environment = setgrouper.set_env()
    echodata.platform = setgrouper.set_platform()
    echodata.provenance = setgrouper.set_provenance()
    echodata.sonar = setgrouper.set_sonar()
    # Beam_power group only exist if EK80 has both complex and power/angle data
    if model == "EK80":
        echodata.beam, echodata.beam_power = setgrouper.set_beam()
    else:
        echodata.beam = setgrouper.set_beam()
    echodata.vendor = setgrouper.set_vendor()

    return echodata


def open_converted(converted_raw_path, storage_options=None):
    """Create an EchoData object from a single converted zarr/nc files.
    """
    # TODO: combine multiple files when opening
    return EchoData(converted_raw_path=converted_raw_path, storage_options=storage_options)
