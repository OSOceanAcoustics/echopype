from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import fsspec
from datatree import DataTree

# fmt: off
# black and isort have conflicting ideas about how this should be formatted
from ..core import SONAR_MODELS
from .parsed_to_zarr import Parsed2Zarr

if TYPE_CHECKING:
    from ..core import EngineHint, PathHint, SonarModelsHint
# fmt: on
from ..echodata.echodata import XARRAY_ENGINE_MAP, EchoData
from ..utils import io
from ..utils.coding import COMPRESSION_SETTINGS
from ..utils.log import _init_logger

BEAM_SUBGROUP_DEFAULT = "Beam_group1"

# Logging setup
logger = _init_logger(__name__)


def to_file(
    echodata: EchoData,
    engine: "EngineHint",
    save_path: Optional["PathHint"] = None,
    compress: bool = True,
    overwrite: bool = False,
    parallel: bool = False,
    output_storage_options: Dict[str, str] = {},
    **kwargs,
):
    """Save content of EchoData to netCDF or zarr.

    Parameters
    ----------
    engine : str {'netcdf4', 'zarr'}
        type of converted file
    save_path : str
        path that converted .nc file will be saved
    compress : bool
        whether or not to perform compression on data variables
        Defaults to ``True``
    overwrite : bool
        whether or not to overwrite existing files
        Defaults to ``False``
    parallel : bool
        whether or not to use parallel processing. (Not yet implemented)
    output_storage_options : dict
        Additional keywords to pass to the filesystem class.
    **kwargs : dict, optional
        Extra arguments to either `xr.Dataset.to_netcdf`
        or `xr.Dataset.to_zarr`: refer to each method documentation
        for a list of all possible arguments.

    """
    if parallel:
        raise NotImplementedError("Parallel conversion is not yet implemented.")
    if engine not in XARRAY_ENGINE_MAP.values():
        raise ValueError("Unknown type to convert file to!")

    # Assemble output file names and path
    output_file = io.validate_output_path(
        source_file=echodata.source_file,
        engine=engine,
        save_path=save_path,
        output_storage_options=output_storage_options,
    )

    # Get all existing files
    fs = fsspec.get_mapper(output_file, **output_storage_options).fs  # get file system
    exists = True if fs.exists(output_file) else False

    # Sequential or parallel conversion
    if exists and not overwrite:
        logger.info(
            f"{echodata.source_file} has already been converted to {engine}. "  # noqa
            f"File saving not executed."
        )
    else:
        if exists:
            logger.info(f"overwriting {output_file}")
        else:
            logger.info(f"saving {output_file}")
        _save_groups_to_file(
            echodata,
            output_path=io.sanitize_file_path(
                file_path=output_file, storage_options=output_storage_options
            ),
            engine=engine,
            compress=compress,
            **kwargs,
        )

    # Link path to saved file with attribute as if from open_converted
    echodata.converted_raw_path = output_file


def _save_groups_to_file(echodata, output_path, engine, compress=True, **kwargs):
    """Serialize all groups to file."""
    # TODO: in terms of chunking, would using rechunker at the end be faster and more convenient?
    # TODO: investigate chunking before we save Dataset to a file

    # Top-level group
    io.save_file(
        echodata["Top-level"],
        path=output_path,
        mode="w",
        engine=engine,
        compression_settings=COMPRESSION_SETTINGS[engine] if compress else None,
        **kwargs,
    )

    # Environment group
    io.save_file(
        echodata["Environment"],  # TODO: chunking necessary?
        path=output_path,
        mode="a",
        engine=engine,
        group="Environment",
        compression_settings=COMPRESSION_SETTINGS[engine] if compress else None,
        **kwargs,
    )

    # Platform group
    io.save_file(
        echodata["Platform"],  # TODO: chunking necessary? time1 and time2 (EK80) only
        path=output_path,
        mode="a",
        engine=engine,
        group="Platform",
        compression_settings=COMPRESSION_SETTINGS[engine] if compress else None,
        **kwargs,
    )

    # Platform/NMEA group: some sonar model does not produce NMEA data
    if echodata["Platform/NMEA"] is not None:
        io.save_file(
            echodata["Platform/NMEA"],  # TODO: chunking necessary?
            path=output_path,
            mode="a",
            engine=engine,
            group="Platform/NMEA",
            compression_settings=COMPRESSION_SETTINGS[engine] if compress else None,
            **kwargs,
        )

    # Provenance group
    io.save_file(
        echodata["Provenance"],
        path=output_path,
        group="Provenance",
        mode="a",
        engine=engine,
        compression_settings=COMPRESSION_SETTINGS[engine] if compress else None,
        **kwargs,
    )

    # Sonar group
    io.save_file(
        echodata["Sonar"],
        path=output_path,
        group="Sonar",
        mode="a",
        engine=engine,
        compression_settings=COMPRESSION_SETTINGS[engine] if compress else None,
        **kwargs,
    )

    # /Sonar/Beam_groupX group
    if echodata.sonar_model == "AD2CP":
        for i in range(1, len(echodata["Sonar"]["beam_group"]) + 1):
            io.save_file(
                echodata[f"Sonar/Beam_group{i}"],
                path=output_path,
                mode="a",
                engine=engine,
                group=f"Sonar/Beam_group{i}",
                compression_settings=COMPRESSION_SETTINGS[engine] if compress else None,
                **kwargs,
            )
    else:
        io.save_file(
            echodata[f"Sonar/{BEAM_SUBGROUP_DEFAULT}"],
            path=output_path,
            mode="a",
            engine=engine,
            group=f"Sonar/{BEAM_SUBGROUP_DEFAULT}",
            compression_settings=COMPRESSION_SETTINGS[engine] if compress else None,
            **kwargs,
        )
        if echodata["Sonar/Beam_group2"] is not None:
            # some sonar model does not produce Sonar/Beam_group2
            io.save_file(
                echodata["Sonar/Beam_group2"],
                path=output_path,
                mode="a",
                engine=engine,
                group="Sonar/Beam_group2",
                compression_settings=COMPRESSION_SETTINGS[engine] if compress else None,
                **kwargs,
            )

    # Vendor_specific group
    io.save_file(
        echodata["Vendor_specific"],  # TODO: chunking necessary?
        path=output_path,
        mode="a",
        engine=engine,
        group="Vendor_specific",
        compression_settings=COMPRESSION_SETTINGS[engine] if compress else None,
        **kwargs,
    )


def _set_convert_params(param_dict: Dict[str, str]) -> Dict[str, str]:
    """Set parameters (metadata) that may not exist in the raw files.

    The default set of parameters include:
    - Platform group: ``platform_name``, ``platform_type``, ``platform_code_ICES``, ``water_level``
    - Top-level group: ``survey_name``

    Other parameters will be saved to the top level.

    # TODO: revise docstring, give examples.
    Examples
    --------
    # set parameters that may not already be in source files
    echodata.set_param({
        'platform_name': 'OOI',
        'platform_type': 'mooring'
    })
    """
    out_params = dict()

    # Parameters for the Platform group
    out_params["platform_name"] = param_dict.get("platform_name", "")
    out_params["platform_code_ICES"] = param_dict.get("platform_code_ICES", "")
    out_params["platform_type"] = param_dict.get("platform_type", "")
    out_params["water_level"] = param_dict.get("water_level", None)

    # Parameters for the Top-level group
    out_params["survey_name"] = param_dict.get("survey_name", "")
    for k, v in param_dict.items():
        if k not in out_params:
            out_params[k] = v

    return out_params


def _check_file(
    raw_file,
    sonar_model: "SonarModelsHint",
    xml_path: Optional["PathHint"] = None,
    storage_options: Dict[str, str] = {},
) -> Tuple[str, str]:
    """Checks whether the file and/or xml file exists and
    whether they have the correct extensions.

    Parameters
    ----------
    raw_file : str
        path to raw data file
    sonar_model : str
        model of the sonar instrument
    xml_path : str
        path to XML config file used by AZFP
    storage_options : dict
        options for cloud storage

    Returns
    -------
    file : str
        path to existing raw data file
    xml : str
        path to existing xml file
        empty string if no xml file is required for the specified model
    """
    if SONAR_MODELS[sonar_model]["xml"]:  # if this sonar model expects an XML file
        if not xml_path:
            raise ValueError(f"XML file is required for {sonar_model} raw data")
        else:
            if ".XML" not in Path(xml_path).suffix.upper():
                raise ValueError(f"{Path(xml_path).name} is not an XML file")

        xmlmap = fsspec.get_mapper(str(xml_path), **storage_options)
        if not xmlmap.fs.exists(xmlmap.root):
            raise FileNotFoundError(f"There is no file named {Path(xml_path).name}")

        xml = xml_path
    else:
        xml = ""

    # TODO: https://github.com/OSOceanAcoustics/echopype/issues/229
    #  to add compatibility for pathlib.Path objects for local paths
    fsmap = fsspec.get_mapper(raw_file, **storage_options)
    validate_ext = SONAR_MODELS[sonar_model]["validate_ext"]
    if not fsmap.fs.exists(fsmap.root):
        raise FileNotFoundError(f"There is no file named {Path(raw_file).name}")

    validate_ext(Path(raw_file).suffix.upper())

    return str(raw_file), str(xml)


def open_raw(
    raw_file: "PathHint",
    sonar_model: "SonarModelsHint",
    xml_path: Optional["PathHint"] = None,
    convert_params: Optional[Dict[str, str]] = None,
    storage_options: Optional[Dict[str, str]] = None,
    use_swap: bool = False,
    max_mb: int = 100,
) -> Optional[EchoData]:
    """Create an EchoData object containing parsed data from a single raw data file.

    The EchoData object can be used for adding metadata and ancillary data
    as well as to serialize the parsed data to zarr or netcdf.

    Parameters
    ----------
    raw_file : str
        path to raw data file
    sonar_model : str
        model of the sonar instrument

        - ``EK60``: Kongsberg Simrad EK60 echosounder
        - ``ES70``: Kongsberg Simrad ES70 echosounder
        - ``EK80``: Kongsberg Simrad EK80 echosounder
        - ``EA640``: Kongsberg EA640 echosounder
        - ``AZFP``: ASL Environmental Sciences AZFP echosounder
        - ``AD2CP``: Nortek Signature series ADCP
          (tested with Signature 500 and Signature 1000)

    xml_path : str
        path to XML config file used by AZFP
    convert_params : dict
        parameters (metadata) that may not exist in the raw file
        and need to be added to the converted file
    storage_options : dict
        options for cloud storage
    use_swap: bool
        If True, variables with a large memory footprint will be
        written to a temporary zarr store at ``~/.echopype/temp_output/parsed2zarr_temp_files``
    max_mb : int
        The maximum data chunk size in Megabytes (MB), when offloading
        variables with a large memory footprint to a temporary zarr store


    Returns
    -------
    EchoData object

    Raises
    ------
    ValueError
        If ``sonar_model`` is ``None`` or ``sonar_model``
        given is unsupported.
    FileNotFoundError
        If ``raw_file`` is ``None``.
    TypeError
        If ``raw_file`` input is neither ``str`` or
        ``pathlib.Path`` type.

    Notes
    -----
    ``use_swap=True`` is only available for the following
    echosounders: EK60, ES70, EK80, ES80, EA640. Additionally, this feature
    is currently in beta.
    """
    if raw_file is None:
        raise FileNotFoundError("The path to the raw data file must be specified.")

    # Check for path type
    if isinstance(raw_file, Path):
        raw_file = str(raw_file)
    if not isinstance(raw_file, str):
        raise TypeError("File path must be a string or Path")

    if sonar_model is None:
        raise ValueError("Sonar model must be specified.")

    # Check inputs
    if convert_params is None:
        convert_params = {}
    storage_options = storage_options if storage_options is not None else {}

    # Uppercased model in case people use lowercase
    sonar_model = sonar_model.upper()  # type: ignore

    # Check models
    if sonar_model not in SONAR_MODELS:
        raise ValueError(
            f"Unsupported echosounder model: {sonar_model}\nMust be one of: {list(SONAR_MODELS)}"  # noqa
        )

    # Check file extension and existence
    file_chk, xml_chk = _check_file(raw_file, sonar_model, xml_path, storage_options)

    # TODO: remove once 'auto' option is added
    if not isinstance(use_swap, bool):
        raise ValueError("use_swap must be of type bool.")

    # Ensure use_swap is 'auto', if it is a string
    # TODO: use the following when we allow for 'auto' option
    # if isinstance(use_swap, str) and use_swap != "auto":
    #     raise ValueError("use_swap must be a bool or equal to 'auto'.")

    # TODO: the if-else below only works for the AZFP vs EK contrast,
    #  but is brittle since it is abusing params by using it implicitly
    if SONAR_MODELS[sonar_model]["xml"]:
        params = xml_chk
    else:
        params = "ALL"  # reserved to control if only wants to parse a certain type of datagram

    # obtain dict associated with directly writing to zarr
    dgram_zarr_vars = SONAR_MODELS[sonar_model]["dgram_zarr_vars"]

    # Parse raw file and organize data into groups
    parser = SONAR_MODELS[sonar_model]["parser"](
        file_chk, params=params, storage_options=storage_options, dgram_zarr_vars=dgram_zarr_vars
    )

    parser.parse_raw()

    # Direct offload to zarr and rectangularization only available for some sonar models
    if sonar_model in ["EK60", "ES70", "EK80", "ES80", "EA640"]:
        # Create sonar_model-specific p2z object
        p2z = SONAR_MODELS[sonar_model]["parsed2zarr"](parser)

        # Determines if writing to zarr is necessary and writes to zarr
        p2z_flag = use_swap is True or (
            use_swap == "auto" and p2z.whether_write_to_zarr(mem_mult=0.4)
        )

        if p2z_flag:
            p2z.datagram_to_zarr(max_mb=max_mb)
            # Rectangularize the transmit data
            parser.rectangularize_transmit_ping_data(data_type="complex")
        else:
            del p2z
            # Create general p2z object
            p2z = Parsed2Zarr(parser)
            parser.rectangularize_data()

    else:
        # No rectangularization for other sonar models
        p2z = Parsed2Zarr(parser)  # Create general p2z object

    setgrouper = SONAR_MODELS[sonar_model]["set_groups"](
        parser,
        input_file=file_chk,
        xml_path=xml_chk,
        output_path=None,
        sonar_model=sonar_model,
        params=_set_convert_params(convert_params),
        parsed2zarr_obj=p2z,
    )

    # Setup tree dictionary
    tree_dict = {}

    # Top-level date_created varies depending on sonar model
    # Top-level is called "root" within tree
    if sonar_model in ["EK60", "ES70", "EK80", "ES80", "EA640"]:
        tree_dict["/"] = setgrouper.set_toplevel(
            sonar_model=sonar_model,
            date_created=parser.config_datagram["timestamp"],
        )
    else:
        tree_dict["/"] = setgrouper.set_toplevel(
            sonar_model=sonar_model, date_created=parser.ping_time[0]
        )
    tree_dict["Environment"] = setgrouper.set_env()
    tree_dict["Platform"] = setgrouper.set_platform()
    if sonar_model in ["EK60", "ES70", "EK80", "ES80", "EA640"]:
        tree_dict["Platform/NMEA"] = setgrouper.set_nmea()
    tree_dict["Provenance"] = setgrouper.set_provenance()
    # Allocate a tree_dict entry for Sonar? Otherwise, a DataTree error occurs
    tree_dict["Sonar"] = None

    # Set multi beam groups
    beam_groups = setgrouper.set_beam()

    beam_group_type = []
    for idx, beam_group in enumerate(beam_groups, start=1):
        if beam_group is not None:
            # fill in beam_group_type (only necessary for EK80, ES80, EA640)
            if idx == 1:
                # choose the appropriate description key for Beam_group1
                beam_group_type.append("complex" if "backscatter_i" in beam_group else "power")
            else:
                # provide None for all other beam groups (since the description does not have a key)
                beam_group_type.append(None)

            tree_dict[f"Sonar/Beam_group{idx}"] = beam_group

    if sonar_model in ["EK80", "ES80", "EA640"]:
        tree_dict["Sonar"] = setgrouper.set_sonar(beam_group_type=beam_group_type)
    else:
        tree_dict["Sonar"] = setgrouper.set_sonar()

    tree_dict["Vendor_specific"] = setgrouper.set_vendor()

    # Create tree and echodata
    # TODO: make the creation of tree dynamically generated from yaml
    tree = DataTree.from_dict(tree_dict, name="root")
    echodata = EchoData(
        source_file=file_chk, xml_path=xml_chk, sonar_model=sonar_model, parsed2zarr_obj=p2z
    )
    echodata._set_tree(tree)
    echodata._load_tree()

    return echodata
