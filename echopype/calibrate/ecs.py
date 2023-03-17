import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import xarray as xr

from ..utils.log import _init_logger

logger = _init_logger(__name__)


# String matcher for parser
SEPARATOR = re.compile(r"#=+#\n")
STATUS_CRUDE = re.compile(r"#\s+(?P<status>(.+))\s+#\n")  # noqa
STATUS_FINE = re.compile(r"#\s+(?P<status>\w+) SETTINGS\s*#\n")  # noqa
ECS_HEADER = re.compile(
    r"#\s+ECHOVIEW CALIBRATION SUPPLEMENT \(.ECS\) FILE \((?P<data_type>\w+)\)\s+#\n"  # noqa
)
ECS_TIME = re.compile(
    r"#\s+(?P<date>\d{1,2}\/\d{1,2}\/\d{4}) (?P<time>\d{1,2}\:\d{1,2}\:\d{1,2})(.\d+)?\s+#\n"  # noqa
)
ECS_VERSION = re.compile(r"Version (?P<version>\d+\.\d+)\s*\n")  # noqa
PARAM_MATCHER = re.compile(
    r"\s*(?P<skip>#?)\s*(?P<param>\w+)\s*=\s*(?P<val>((-?\d+(?:\.\d+))|\w+)?)?\s*#?(.*)\n"  # noqa
)
CAL = re.compile(r"(SourceCal|LocalCal) (?P<source>\w+)\s*\n", re.I)  # ignore case  # noqa


# Convert dict from ECS to echopype format
EV_EP_MAP = {
    # from Echoview-generated ECS template (unless noted otherwise) : EchoData variable name
    # Ex60 / Ex70 / EK15
    "EK60": {
        "AbsorptionCoefficient": "sound_absorption",
        "Frequency": "frequency_nominal",  # will use for checking channel and freq match
        "MajorAxis3dbBeamAngle": "beamwidth_athwartship",
        "MajorAxisAngleOffset": "angle_offset_athwartship",
        "MajorAxisAngleSensitivity": "angle_sensitivity_athwartship",
        "MinorAxis3dbBeamAngle": "beamwidth_alongship",
        "MinorAxisAngleOffset": "angle_offset_alongship",
        "MinorAxisAngleSensitivity": "angle_sensitivity_alongship",
        "PulseDuration": "transmit_duration_nominal",
        "SaCorrectionFactor": "sa_correction",
        "SoundSpeed": "sound_speed",
        "EK60SaCorrection": "sa_correction",  # from NWFSC template
        "TransducerGain": "gain_correction",
        "Ek60TransducerGain": "gain_correction",  # from NWFSC template
        "TransmittedPower": "transmit_power",
        "TvgRangeCorrection": "tvg_range_correction",  # not in EchoData
        "TvgRangeCorrectionOffset": "tvg_range_correction_offset",  # not in EchoData
        "TwoWayBeamAngle": "equivalent_beam_angle",
    },
    # Additional on EK80, ES80, WBAT, EA640
    # Note these should be concat after the EK60 dict
    "EK80": {
        "AbsorptionDepth": "sound_absorption",
        "Acidity": "acidity",
        "EffectivePulseDuration": "transmit_duration_nominal",
        "Salinity": "salinity",
        "SamplingFrequency": "sampling_frequency",  # does not exist in echopype.EchoData
        "Temperature": "temperature",
        "TransceiverImpedance": "impedance_transceiver",
        "TransceiverSamplingFrequency": "receiver_sampling_frequency",
        "TransducerModeActive": "transducer_mode",  # TODO: CHECK IN ECHODATA
        "FrequencyTableWideband": "cal_frequency",  # frequency axis for broadband cal params
        "GainTableWideband": "gain",  # frequency-dependent gain
        "MajorAxisAngleOffsetTableWideband": "angle_offset_athwartship",  # TODO: Vendor-specific
        "MajorAxisBeamWidthTableWideband": "beamwidth_athwartship",  # TODO: Vendor-specific
        "MinorAxisAngleOffsetTableWideband": "angle_offset_alongship",  # TODO: Vendor-specific
        "MinorAxisBeamWidthTableWideband": "beamwidth_alongship",  # TODO: Vendor-specific
        "NumberOfTransducerSegments": "n_sector",  # TODO: CHECK IN ECHODATA
        "PulseCompressedEffectivePulseDuration": "tau_effective",  # TODO: not in EchoData
    },
    # AZFP-specific
    # Note: not sure why it doesn't contain salinity and pressure required for computing absorption
    # "AZFP": {
    #     AzfpDetectionSlope = 0.023400 # [0.000000..1.000000]
    #     AzfpEchoLevelMax = 142.8 # (decibels) [0.0..9999.0]
    #     AzfpTransmitVoltage = 53.0 # [0.0..999.0]
    #     AzfpTransmitVoltageResponse = 170.9 # (decibels) [0.0..999.0]
    #     Frequency = 38.00 # (kilohertz) [0.01..10000.00]
    #     PulseDuration = 1.000 # (milliseconds) [0.001..200.000]
    #     SoundSpeed = 1450.50 # (meters per second) [1400.00..1700.00]
    #     TwoWayBeamAngle = -16.550186 # (decibels re 1 steradian) [-99.000000..11.000000]
    #     TvgRangeCorrection = # [None, BySamples, SimradEx500, SimradEx60, BioSonics, Kaijo, PulseLength, Ex500Forced, SimradEK80, Standard]
    #     TvgRangeCorrectionOffset = # (samples) [-10000.00..10000.00]
    # },
}
ENV_PARAMS = ["AbsorptionCoefficient", "SoundSpeed"]
CAL_PARAMS = set(EV_EP_MAP.keys()).difference(set(ENV_PARAMS))


class ECSParser:
    """
    Class for parsing Echoview calibration supplement (ECS) files.
    """

    TvgRangeCorrection_allowed_str = (
        "None",
        "BySamples",
        "SimradEx500",
        "SimradEx60",
        "BioSonics",
        "Kaijo",
        "PulseLength",
        "Ex500Forced",
    )

    def __init__(self, input_file=None):
        self.input_file = input_file
        self.data_type = None
        self.version = None
        self.file_creation_time: Optional[datetime] = None
        self.parsed_params: Optional[dict] = None

    def _parse_header(self, fid) -> bool:
        """
        Parse header block.
        """
        tmp = ECS_TIME.match(fid.readline())
        self.file_creation_time = datetime.strptime(
            tmp["date"] + " " + tmp["time"], "%m/%d/%Y %H:%M:%S"
        )
        if SEPARATOR.match(fid.readline()) is None:  # line 4: separator
            raise ValueError("Unexpected line in ECS file!")
        # line 5-10: skip
        [fid.readline() for ff in range(6)]
        if SEPARATOR.match(fid.readline()) is None:  # line 11: separator
            raise ValueError("Unexpected line in ECS file!")
        # read lines until seeing version number
        line = "\n"
        while line == "\n":
            line = fid.readline()
        self.version = ECS_VERSION.match(line)["version"]
        return True

    def _parse_block(self, fid, status) -> dict:
        """
        Parse the FileSet, SourceCal or LocalCal block.

        Parameters
        ----------
        fid : File Object
        status : str {"sourcecal", "localcal"}
        """
        param_val = dict()
        if SEPARATOR.match(fid.readline()) is None:  # skip 1 separator line
            raise ValueError("Unexpected line in ECS file!")
        source = None
        cont = True
        while cont:
            curr_pos = fid.tell()  # current position
            line = fid.readline()
            if SEPARATOR.match(line) is not None:
                # reverse to previous position and jump out
                fid.seek(curr_pos)
                cont = False
            elif line == "":  # EOF
                break
            else:
                if status == "fileset" and source is None:
                    source = "fileset"  # force this for easy organization
                    param_val[source] = dict()
                elif status in line.lower():  # {"sourcecal", "localcal"}
                    source = CAL.match(line)["source"]
                    param_val[source] = dict()
                else:
                    if line != "\n" and source is not None:
                        tmp = PARAM_MATCHER.match(line)
                        if tmp["skip"] == "" or tmp["param"] == "Frequency":  # not skipping
                            param_val[source][tmp["param"]] = tmp["val"]
        return param_val

    def _convert_param_type(self):
        """
        Convert data type for all parameters.
        """

        def convert_type(input_dict):
            for k, v in input_dict.items():
                if k == "TvgRangeCorrection":
                    if v not in self.TvgRangeCorrection_allowed_str:
                        raise ValueError("TvgRangeCorrection contains unexpected setting!")
                else:
                    input_dict[k] = float(v)

        for status, status_settings in self.parsed_params.items():
            if status == "fileset":  # fileset only has 1 layer of dict
                convert_type(status_settings)
            else:  # sourcecal or localcal has another layer of dict
                for src_k, src_v in status_settings.items():
                    for k, v in src_v.items():
                        convert_type(src_v)

    def parse(self):
        """
        Parse the entire ECS file.
        """
        fid = open(self.input_file, encoding="utf-8-sig")
        line = fid.readline()

        parsed_params = dict()
        status = None  # status = {"ecs", "fileset", "sourcecal", "localcal"}
        while line != "":  # EOF: line=""
            if line != "\n":  # skip empty line
                if SEPARATOR.match(line) is not None:
                    if status is not None:  # entering another block
                        status = None
                elif status is None:  # going into a block
                    status_str = STATUS_CRUDE.match(line)["status"].lower()
                    if "ecs" in status_str:
                        status = "ecs"
                        self.data_type = ECS_HEADER.match(line)["data_type"]  # get data type
                        self._parse_header(fid)
                    elif (
                        "fileset" in status_str
                        or "sourcecal" in status_str
                        or "localcal" in status_str
                    ):
                        status = STATUS_FINE.match(line)["status"].lower()
                        parsed_params[status] = self._parse_block(fid, status)
                    else:
                        raise ValueError("Expecting a new block but got something else!")
            line = fid.readline()  # read next line

        # Make FileSet settings dict less awkward
        parsed_params["fileset"] = parsed_params["fileset"]["fileset"]

        # Store params
        self.parsed_params = parsed_params

        # Convert parameter type to float
        self._convert_param_type()

    def get_cal_params(self, localcal_name=None) -> dict():
        """
        Get a consolidated set of calibration parameters that is applied to data by Echoview.

        The calibration settings in Echoview have an overwriting hierarchy as documented
        `here <https://support.echoview.com/WebHelp/Reference/File_formats/Echoview_calibration_supplement_files.html>`_.  # noqa

        Parameters
        ----------
        localcal_name : str or None
            Name of the LocalCal settings selected in Echoview.
            Default is the first one read in the ECS file.

        Returns
        -------
        A dictionary containing calibration parameters as interpreted by Echoview.
        """
        # Create template based on sources
        sources = self.parsed_params["sourcecal"].keys()
        ev_cal_params = dict().fromkeys(sources)

        # FileSet settings: apply to all sources
        for src in sources:
            ev_cal_params[src] = self.parsed_params["fileset"].copy()

        # SourceCal settings: overwrite FileSet settings for each source
        for src in sources:
            for k, v in self.parsed_params["sourcecal"][src].items():
                ev_cal_params[src][k] = v

        # LocalCal settings: overwrite the above settings for all sources
        if self.parsed_params["localcal"] != {}:
            if localcal_name is None:  # use the first LocalCal setting by default
                localcal_name = list(self.parsed_params["localcal"].keys())[0]
            for k, v in self.parsed_params["localcal"][localcal_name].items():
                for src in sources:
                    ev_cal_params[src][k] = v

        return ev_cal_params


def ev2ep(
    ev_dict: Dict[str, Union[int, float, str]], channel: List[str] = None
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Convert dictionary from consolidated ECS form to xr.DataArray expected by echopype.

    Parameters
    ----------
    ev_dict : dict
        A dictionary of the format parsed by the ECS parser
    channel : Option, list
        A list containing channel id for all transducers
        in the order of sources listed in the ECS file (T1, T2, etc.).
        If ``channel`` is not provided, the generated data array will have
        an arbitrary coordinate, which will be replaced in ``sanitize_source_channel_order``.

    Returns
    -------
    xr.Dataset
        An xr.Dataset containing calibration parameters
    xr.Dataset
        An xr.Dataset containing environmental parameters
    """
    # Gather cal and env params
    env_dict = defaultdict(list)
    cal_dict = defaultdict(list)
    # loop through all transducers (sources)
    for source, source_dict in ev_dict.items():
        # loop through all params and append to list
        for p_name, p_val in source_dict.items():
            if p_name in ENV_PARAMS:
                env_dict[EV_EP_MAP[p_name]].append((p_val))
            elif p_name in CAL_PARAMS:
                cal_dict[EV_EP_MAP[p_name]].append((p_val))
            else:
                logger.warning(
                    f"{source}: {p_name} is not an allowable calibration "
                    "or environmental parameter."
                )

    # Add dimension to dict
    env_dict = {k: (["channel"], v) for k, v in env_dict.items()}
    cal_dict = {k: (["channel"], v) for k, v in cal_dict.items()}
    env_dict["frequency_nominal"] = cal_dict["frequency_nominal"]  # used for checking later

    # Assemble xr.DataArray
    ds_env = xr.Dataset(data_vars=env_dict)
    ds_cal = xr.Dataset(data_vars=cal_dict)
    ds_env["frequency_nominal"] = ds_env["frequency_nominal"] * 1000  # convert from kHz to Hz
    ds_cal["frequency_nominal"] = ds_cal["frequency_nominal"] * 1000  # convert from kHz to Hz

    return ds_cal, ds_env


def check_source_channel_order(ds_in: xr.Dataset, freq_ref: xr.DataArray) -> xr.Dataset:
    """
    Check the sequence of channels against a set of reference channels and reorder if necessary.

    Parameters
    ----------
    ds_in : xr.Dataset
        An xr.Dataset generated by ``ev2ep_dict``.
        It must contain 'frequency_nominal' as a data variable and 'channel' as a dimension.
    freq_ref : xr.DataArray
        An xr.DataArray containing the nominal frequency in the order to be conformed with.
        It must contain 'channel' as a coordinate.

    Returns
    -------
    xr.Dataset
        An xr.Dataset with channel in the same order as the set of reference frequencies
    """
    if not isinstance(freq_ref, xr.DataArray):
        raise ValueError("'freq_ref' has to be an xr.DataArray!")
    else:
        if "channel" not in freq_ref.coords:
            raise ValueError("'channel' has to be a coordinate 'freq_ref'!")

    # Set both datasets to align with frequency
    freq_ref.name = "frequency_nominal"
    freq_ref = (
        freq_ref.to_dataset()
        .set_coords("frequency_nominal")
        .swap_dims({"channel": "frequency_nominal"})
    )
    ds_in = ds_in.set_coords("frequency_nominal").swap_dims({"channel": "frequency_nominal"})

    # Reorder according to the frequency dimension
    ds_in = ds_in.reindex_like(freq_ref)
    ds_in["channel"] = freq_ref["channel"]

    return ds_in.swap_dims({"frequency_nominal": "channel"}).drop("frequency_nominal")
