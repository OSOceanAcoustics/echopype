import re
from datetime import datetime


SEPARATOR = re.compile("#=+#\n")
STATUS = re.compile("#\s+(?P<status>(.+))\s+#\n")  # noqa  # TODO: removing trailing 0s
ECS_HEADER = re.compile("#\s+ECHOVIEW CALIBRATION SUPPLEMENT \(.ECS\) FILE \((?P<data_type>\w+)\)\s+#\n")  # noqa
ECS_TIME = re.compile("#\s+(?P<date>\d{1,2}\/\d{1,2}\/\d{4}) (?P<time>\d{1,2}\:\d{1,2}\:\d{1,2})(.\d+)?\s+#\n")  # noqa
ECS_VERSION = re.compile("Version (?P<version>\d+\.\d+)\s*\n")  # noqa
PARAM_MATCHER = re.compile("\s*(?P<skip>#?)\s*(?P<param>\w+)\s*=\s*(?P<val>-?\d+(?:\.\d+)?)?\s*#?(.*)\n")  # noqa
CAL = re.compile("(SourceCal|LocalCal) (?P<source>\w+)\s*\n", re.I)  # ignore case  # noqa


class ECSParser():
    """
    Class for parsing Echoview calibration supplement (ECS) files.
    """

    def __init__(self, input_file=None):
        self.input_file = input_file
        self.data_type = None
        self.version = None
        self.file_creation_time = None

    def _parse_header(self, fid) -> bool:
        """
        Parse header block.
        """
        tmp = ECS_TIME.match(fid.readline())
        self.file_creation_time = datetime.strptime(
            tmp["date"] + ' ' + tmp["time"], "%m/%d/%Y %H:%M:%S"
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
        Parse the SourceCal or LocalCal block.

        Parameters
        ----------
        fid : File Object
        status : str {"sourcecal", "localcal"}
        """
        param_val = dict()
        if status == "fileset":  # go straight into parsing params
            if SEPARATOR.match(fid.readline()) is None:  # skip 1 separator line
                raise ValueError("Unexpected line in ECS file!")
            cont = True
            while cont:
                curr_pos = fid.tell()  # current position
                line = fid.readline()
                if SEPARATOR.match(line) is not None:
                    # reverse to previous position and jump out
                    fid.seek(curr_pos)
                    cont = False
                else:
                    if line != "\n":
                        tmp = PARAM_MATCHER.match(line)
                        if tmp["skip"] == "":  # not skipping
                            param_val[tmp["param"]] = tmp["val"]
        else:
            param_val_src = None
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
                elif line == "":
                    break
                else:
                    if status in line.lower():  # {"sourcecal", "localcal"}
                        if source is not None:
                            param_val[source] = param_val_src  # save previous source params
                        source = CAL.match(line)["source"]
                        param_val_src = dict()
                    else:
                        if line != "\n" and source is not None:
                            tmp = PARAM_MATCHER.match(line)
                            if tmp["skip"] == "":  # not skipping
                                param_val_src[tmp["param"]] = tmp["val"]
        return param_val

    def parse(self):

        fid = open(self.input_file, encoding="utf-8-sig")
        line = fid.readline()

        status = None  # status = {"ecs", "fileset", "sourcecal", "localcal"}
        while line != "":  # EOF: line=""
            if line != "\n":  # skip empty line
                if SEPARATOR.match(line) is not None:
                    if status is not None:  # entering another block
                        status = None
                elif status is None:  # going into a block
                    status_str = STATUS.match(line)["status"].lower()
                    if "ecs" in status_str:
                        status = "ecs"
                        self.data_type = ECS_HEADER.match(line)["data_type"]  # get data type
                        self._parse_header(fid)
                    elif "fileset" in status_str:
                        status = "fileset"
                        self._parse_block(fid, status)
                    elif "sourcecal" in status_str:
                        status = "sourcecal"
                        self._parse_block(fid, status)
                    elif "localcal" in status_str:
                        status = "localcal"
                        self._parse_block(fid, status)
                    else:
                        raise ValueError("Expecting a new block but got something else!")
            line = fid.readline()  # read next line
