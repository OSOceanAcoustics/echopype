import re
from datetime import datetime
from typing import Optional

import numpy as np

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
                        if tmp["skip"] == "":  # not skipping
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
                    input_dict[k] = np.float(v)

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
        fileset_dict = self.parsed_params["fileset"].copy()
        for src in sources:
            ev_cal_params[src] = fileset_dict

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
