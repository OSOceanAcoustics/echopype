import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import fsspec
import zarr
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
from ..utils.log import _init_logger

COMPRESSION_SETTINGS = {
    "netcdf4": {"zlib": True, "complevel": 4},
    "zarr": {"compressor": zarr.Blosc(cname="zstd", clevel=3, shuffle=2)},
}

DEFAULT_CHUNK_SIZE = {"range_sample": 25000, "ping_time": 2500}

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
        )

    # Link path to saved file with attribute as if from open_converted
    echodata.converted_raw_path = output_file


def _save_groups_to_file(echodata, output_path, engine, compress=True):
    """Serialize all groups to file."""
    # TODO: in terms of chunking, would using rechunker at the end be faster and more convenient?

    # Top-level group
    io.save_file(echodata["Top-level"], path=output_path, mode="w", engine=engine)

    # Environment group
    if "time1" in echodata["Environment"]:
        io.save_file(
            # echodata["Environment"].chunk(
            #     {"time1": DEFAULT_CHUNK_SIZE["ping_time"]}
            # ),  # TODO: chunking necessary?
            echodata["Environment"],
            path=output_path,
            mode="a",
            engine=engine,
            group="Environment",
        )
    else:
        io.save_file(
            echodata["Environment"],
            path=output_path,
            mode="a",
            engine=engine,
            group="Environment",
        )

    # Platform group
    io.save_file(
        echodata["Platform"],  # TODO: chunking necessary? time1 and time2 (EK80) only
        path=output_path,
        mode="a",
        engine=engine,
        group="Platform",
        compression_settings=COMPRESSION_SETTINGS[engine] if compress else None,
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
        )

    # Provenance group
    io.save_file(
        echodata["Provenance"],
        path=output_path,
        group="Provenance",
        mode="a",
        engine=engine,
    )

    # Sonar group
    io.save_file(
        echodata["Sonar"],
        path=output_path,
        group="Sonar",
        mode="a",
        engine=engine,
    )

    # /Sonar/Beam_groupX group
    if echodata.sonar_model == "AD2CP":
        for i in range(1, len(echodata["Sonar"]["beam_group"]) + 1):
            io.save_file(
                # echodata[f"Sonar/Beam_group{i}"].chunk(
                #     {
                #         "ping_time": DEFAULT_CHUNK_SIZE["ping_time"],
                #     }
                # ),
                echodata[f"Sonar/Beam_group{i}"],
                path=output_path,
                mode="a",
                engine=engine,
                group=f"Sonar/Beam_group{i}",
                compression_settings=COMPRESSION_SETTINGS[engine] if compress else None,
            )
    else:
        io.save_file(
            # echodata[f"Sonar/{BEAM_SUBGROUP_DEFAULT}"].chunk(
            #     {
            #         "range_sample": DEFAULT_CHUNK_SIZE["range_sample"],
            #         "ping_time": DEFAULT_CHUNK_SIZE["ping_time"],
            #     }
            # ),
            echodata[f"Sonar/{BEAM_SUBGROUP_DEFAULT}"],
            path=output_path,
            mode="a",
            engine=engine,
            group=f"Sonar/{BEAM_SUBGROUP_DEFAULT}",
            compression_settings=COMPRESSION_SETTINGS[engine] if compress else None,
        )
        if echodata["Sonar/Beam_group2"] is not None:
            # some sonar model does not produce Sonar/Beam_group2
            io.save_file(
                # echodata["Sonar/Beam_group2"].chunk(
                #     {
                #         "range_sample": DEFAULT_CHUNK_SIZE["range_sample"],
                #         "ping_time": DEFAULT_CHUNK_SIZE["ping_time"],
                #     }
                # ),
                echodata["Sonar/Beam_group2"],
                path=output_path,
                mode="a",
                engine=engine,
                group="Sonar/Beam_group2",
                compression_settings=COMPRESSION_SETTINGS[engine] if compress else None,
            )

    # Vendor_specific group
    if "ping_time" in echodata["Vendor_specific"]:
        io.save_file(
            # echodata["Vendor_specific"].chunk(
            #     {"ping_time": DEFAULT_CHUNK_SIZE["ping_time"]}
            # ),  # TODO: chunking necessary?
            echodata["Vendor_specific"],
            path=output_path,
            mode="a",
            engine=engine,
            group="Vendor_specific",
            compression_settings=COMPRESSION_SETTINGS[engine] if compress else None,
        )
    else:
        io.save_file(
            echodata["Vendor_specific"],  # TODO: chunking necessary?
            path=output_path,
            mode="a",
            engine=engine,
            group="Vendor_specific",
            compression_settings=COMPRESSION_SETTINGS[engine] if compress else None,
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
    raw_file: Optional["PathHint"] = None,
    sonar_model: Optional["SonarModelsHint"] = None,
    xml_path: Optional["PathHint"] = None,
    convert_params: Optional[Dict[str, str]] = None,
    storage_options: Optional[Dict[str, str]] = None,
    offload_to_zarr: bool = False,
    max_zarr_mb: int = 100,
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
    offload_to_zarr: bool
        If True, variables with a large memory footprint will be
        written to a temporary zarr store.
    max_zarr_mb : int
        maximum MB that each zarr chunk should hold, when offloading
        variables with a large memory footprint to a temporary zarr store

    Returns
    -------
    EchoData object

    Notes
    -----
    ``offload_to_zarr=True`` is only available for the following
    echosounders: EK60, ES70, EK80, ES80, EA640. Additionally, this feature
    is currently in beta.
    """
    if (sonar_model is None) and (raw_file is None):
        logger.warning("Please specify the path to the raw data file and the sonar model.")
        return

    # Check inputs
    if convert_params is None:
        convert_params = {}
    storage_options = storage_options if storage_options is not None else {}

    if sonar_model is None:
        logger.warning("Please specify the sonar model.")

        if xml_path is None:
            sonar_model = "EK60"
            warnings.warn(
                "Current behavior is to default sonar_model='EK60' when no XML file is passed in as argument. "  # noqa
                "Specifying sonar_model='EK60' will be required in the future, "
                "since .raw extension is used for many Kongsberg/Simrad sonar systems.",
                DeprecationWarning,
                2,
            )
        else:
            sonar_model = "AZFP"
            warnings.warn(
                "Current behavior is to set sonar_model='AZFP' when an XML file is passed in as argument. "  # noqa
                "Specifying sonar_model='AZFP' will be required in the future.",
                DeprecationWarning,
                2,
            )
    else:
        # Uppercased model in case people use lowercase
        sonar_model = sonar_model.upper()  # type: ignore

        # Check models
        if sonar_model not in SONAR_MODELS:
            raise ValueError(
                f"Unsupported echosounder model: {sonar_model}\nMust be one of: {list(SONAR_MODELS)}"  # noqa
            )

    # Check paths and file types
    if raw_file is None:
        raise FileNotFoundError("Please specify the path to the raw data file.")

    # Check for path type
    if isinstance(raw_file, Path):
        raw_file = str(raw_file)
    if not isinstance(raw_file, str):
        raise TypeError("file must be a string or Path")

    assert sonar_model is not None

    # Check file extension and existence
    file_chk, xml_chk = _check_file(raw_file, sonar_model, xml_path, storage_options)

    # TODO: remove once 'auto' option is added
    if not isinstance(offload_to_zarr, bool):
        raise ValueError("offload_to_zarr must be of type bool.")

    # Ensure offload_to_zarr is 'auto', if it is a string
    # TODO: use the following when we allow for 'auto' option
    # if isinstance(offload_to_zarr, str) and offload_to_zarr != "auto":
    #     raise ValueError("offload_to_zarr must be a bool or equal to 'auto'.")

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

    # code block corresponding to directly writing parsed data to zarr
    if offload_to_zarr and (sonar_model in ["EK60", "ES70", "EK80", "ES80", "EA640"]):

        # Determines if writing to zarr is necessary and writes to zarr
        p2z = SONAR_MODELS[sonar_model]["parsed2zarr"](parser)

        # TODO: perform more robust tests for the 'auto' heuristic value
        if offload_to_zarr == "auto" and p2z.write_to_zarr(mem_mult=0.4):
            p2z.datagram_to_zarr(max_mb=max_zarr_mb)
        elif offload_to_zarr is True:
            p2z.datagram_to_zarr(max_mb=max_zarr_mb)
        else:
            del p2z
            p2z = Parsed2Zarr(parser)
            if "ALL" in parser.data_type:
                parser.rectangularize_data()

    else:
        p2z = Parsed2Zarr(parser)
        if (sonar_model in ["EK60", "ES70", "EK80", "ES80", "EA640"]) and (
            "ALL" in parser.data_type
        ):
            parser.rectangularize_data()

    setgrouper = SONAR_MODELS[sonar_model]["set_groups"](
        parser,
        input_file=file_chk,
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

    valid_beam_groups_count = 0
    for idx, beam_group in enumerate(beam_groups, start=1):
        if beam_group is not None:
            valid_beam_groups_count += 1
            tree_dict[f"Sonar/Beam_group{idx}"] = beam_group

    if sonar_model in ["EK80", "ES80", "EA640"]:
        tree_dict["Sonar"] = setgrouper.set_sonar(beam_group_count=valid_beam_groups_count)
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
