"""
Code originally developed for pyEcholab
(https://github.com/CI-CMG/pyEcholab)
by Rick Towler <rick.towler@noaa.gov> at NOAA AFSC.

The code has been modified to handle split-beam data and
channel-transducer structure from different EK80 setups.
"""

import logging
import re
import struct
import sys
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

from .ek_date_conversion import nt_to_unix

TCVR_CH_NUM_MATCHER = re.compile(r"\d{6}-\w{1,2}")

__all__ = [
    "SimradNMEAParser",
    "SimradDepthParser",
    "SimradBottomParser",
    "SimradAnnotationParser",
    "SimradConfigParser",
    "SimradRawParser",
]

log = logging.getLogger(__name__)


class _SimradDatagramParser(object):
    """"""

    def __init__(self, header_type, header_formats):
        self._id = header_type
        self._headers = header_formats
        self._versions = list(header_formats.keys())

    def header_fmt(self, version=0):
        return "=" + "".join([x[1] for x in self._headers[version]])

    def header_size(self, version=0):
        return struct.calcsize(self.header_fmt(version))

    def header_fields(self, version=0):
        return [x[0] for x in self._headers[version]]

    def header(self, version=0):
        return self._headers[version][:]

    def validate_data_header(self, data):

        if isinstance(data, dict):
            type_ = data["type"][:3]
            version = int(data["type"][3])

        elif isinstance(data, str):
            type_ = data[:3]
            version = int(data[3])

        else:
            raise TypeError("Expected a dict or str")

        if type_ != self._id:
            raise ValueError("Expected data of type %s, not %s" % (self._id, type_))

        if version not in self._versions:
            raise ValueError(
                "No parser available for type %s version %d" % (self._id, version)
            )

        return type_, version

    def from_string(self, raw_string, bytes_read):

        header = raw_string[:4]
        if sys.version_info.major > 2:
            header = header.decode()
        id_, version = self.validate_data_header(header)
        return self._unpack_contents(raw_string, bytes_read, version=version)

    def to_string(self, data={}):

        id_, version = self.validate_data_header(data)
        datagram_content_str = self._pack_contents(data, version=version)
        return self.finalize_datagram(datagram_content_str)

    def _unpack_contents(self, raw_string="", version=0):
        raise NotImplementedError

    def _pack_contents(self, data={}, version=0):
        raise NotImplementedError

    @classmethod
    def finalize_datagram(cls, datagram_content_str):
        datagram_size = len(datagram_content_str)
        final_fmt = "=l%dsl" % (datagram_size)
        return struct.pack(
            final_fmt, datagram_size, datagram_content_str, datagram_size
        )


class SimradDepthParser(_SimradDatagramParser):
    """
    ER60 Depth Detection datagram (from .bot files) contain the following keys:

        type:         string == 'DEP0'
        low_date:     long uint representing LSBytes of 64bit NT date
        high_date:    long uint representing MSBytes of 64bit NT date
        timestamp:    datetime.datetime object of NT date, assumed to be UTC
        transceiver_count:  [long uint] with number of tranceivers

        depth:        [float], one value for each active channel
        reflectivity: [float], one value for each active channel
        unused:       [float], unused value for each active channel

    The following methods are defined:

        from_string(str):    parse a raw ER60 Depth datagram
                             (with leading/trailing datagram size stripped)

        to_string():         Returns the datagram as a raw string
                             (including leading/trailing size fields)
                             ready for writing to disk

    """

    def __init__(self):
        headers = {
            0: [
                ("type", "4s"),
                ("low_date", "L"),
                ("high_date", "L"),
                ("transceiver_count", "L"),
            ]
        }
        _SimradDatagramParser.__init__(self, "DEP", headers)

    def _unpack_contents(self, raw_string, bytes_read, version):
        """"""

        header_values = struct.unpack(
            self.header_fmt(version), raw_string[: self.header_size(version)]
        )
        data = {}

        for indx, field in enumerate(self.header_fields(version)):
            data[field] = header_values[indx]
            if isinstance(data[field], bytes):
                data[field] = data[field].decode()

        data["timestamp"] = nt_to_unix((data["low_date"], data["high_date"]))
        data["bytes_read"] = bytes_read

        if version == 0:
            data_fmt = "=3f"
            data_size = struct.calcsize(data_fmt)

            data["depth"] = np.zeros((data["transceiver_count"],))
            data["reflectivity"] = np.zeros((data["transceiver_count"],))
            data["unused"] = np.zeros((data["transceiver_count"],))

            buf_indx = self.header_size(version)
            for indx in range(data["transceiver_count"]):
                d, r, u = struct.unpack(
                    data_fmt, raw_string[buf_indx : buf_indx + data_size]  # noqa
                )
                data["depth"][indx] = d
                data["reflectivity"][indx] = r
                data["unused"][indx] = u

                buf_indx += data_size

        return data

    def _pack_contents(self, data, version):

        datagram_fmt = self.header_fmt(version)
        datagram_contents = []

        if version == 0:

            lengths = [
                len(data["depth"]),
                len(data["reflectivity"]),
                len(data["unused"]),
                data["transceiver_count"],
            ]

            if len(set(lengths)) != 1:
                min_indx = min(lengths)
                log.warning(
                    "Data lengths mismatched:  d:%d, r:%d, u:%d, t:%d", *lengths
                )
                log.warning("  Using minimum value:  %d", min_indx)
                data["transceiver_count"] = min_indx

            else:
                min_indx = data["transceiver_count"]

            for field in self.header_fields(version):
                datagram_contents.append(data[field])

            datagram_fmt += "%df" % (3 * data["transceiver_count"])

            for indx in range(data["transceiver_count"]):
                datagram_contents.extend(
                    [
                        data["depth"][indx],
                        data["reflectivity"][indx],
                        data["unused"][indx],
                    ]
                )

        return struct.pack(datagram_fmt, *datagram_contents)


class SimradBottomParser(_SimradDatagramParser):
    """
    Bottom Detection datagram contains the following keys:

        type:         string == 'BOT0'
        low_date:     long uint representing LSBytes of 64bit NT date
        high_date:    long uint representing MSBytes of 64bit NT date
        datetime:     datetime.datetime object of NT date converted to UTC
        transceiver_count:  long uint with number of tranceivers
        depth:        [float], one value for each active channel

    The following methods are defined:

        from_string(str):    parse a raw ER60 Bottom datagram
                            (with leading/trailing datagram size stripped)

        to_string():         Returns the datagram as a raw string
                             (including leading/trailing size fields)
                             ready for writing to disk
    """

    def __init__(self):
        headers = {
            0: [
                ("type", "4s"),
                ("low_date", "L"),
                ("high_date", "L"),
                ("transceiver_count", "L"),
            ]
        }
        _SimradDatagramParser.__init__(self, "BOT", headers)

    def _unpack_contents(self, raw_string, bytes_read, version):
        """"""

        header_values = struct.unpack(
            self.header_fmt(version), raw_string[: self.header_size(version)]
        )
        data = {}

        for indx, field in enumerate(self.header_fields(version)):
            data[field] = header_values[indx]
            if isinstance(data[field], bytes):
                data[field] = data[field].decode()

        data["timestamp"] = nt_to_unix((data["low_date"], data["high_date"]))
        data["bytes_read"] = bytes_read

        if version == 0:
            depth_fmt = "=%dd" % (data["transceiver_count"],)
            depth_size = struct.calcsize(depth_fmt)
            buf_indx = self.header_size(version)
            data["depth"] = np.fromiter(
                struct.unpack(
                    depth_fmt, raw_string[buf_indx : buf_indx + depth_size]
                ),  # noqa
                "float",
            )

        return data

    def _pack_contents(self, data, version):

        datagram_fmt = self.header_fmt(version)
        datagram_contents = []

        if version == 0:

            if len(data["depth"]) != data["transceiver_count"]:
                log.warning(
                    "# of depth values %d does not match transceiver count %d",
                    len(data["depth"]),
                    data["transceiver_count"],
                )

                data["transceiver_count"] = len(data["depth"])

            for field in self.header_fields(version):
                datagram_contents.append(data[field])

            datagram_fmt += "%dd" % (data["transceiver_count"])
            datagram_contents.extend(data["depth"])

        return struct.pack(datagram_fmt, *datagram_contents)


class SimradAnnotationParser(_SimradDatagramParser):
    """
    ER60 Annotation datagram contains the following keys:


        type:         string == 'TAG0'
        low_date:     long uint representing LSBytes of 64bit NT date
        high_date:    long uint representing MSBytes of 64bit NT date
        timestamp:     datetime.datetime object of NT date, assumed to be UTC

        text:         Annotation

    The following methods are defined:

        from_string(str):    parse a raw ER60 Annotation datagram
                            (with leading/trailing datagram size stripped)

        to_string():         Returns the datagram as a raw string
                             (including leading/trailing size fields)
                             ready for writing to disk
    """

    def __init__(self):
        headers = {0: [("type", "4s"), ("low_date", "L"), ("high_date", "L")]}

        _SimradDatagramParser.__init__(self, "TAG", headers)

    def _unpack_contents(self, raw_string, bytes_read, version):
        """"""

        header_values = struct.unpack(
            self.header_fmt(version), raw_string[: self.header_size(version)]
        )
        data = {}

        for indx, field in enumerate(self.header_fields(version)):
            data[field] = header_values[indx]
            if isinstance(data[field], bytes):
                data[field] = data[field].decode()

        data["timestamp"] = nt_to_unix((data["low_date"], data["high_date"]))
        data["bytes_read"] = bytes_read

        #        if version == 0:
        #            data['text'] = raw_string[self.header_size(version):].strip('\x00')
        #            if isinstance(data['text'], bytes):
        #                data['text'] = data['text'].decode()

        if version == 0:
            if sys.version_info.major > 2:
                data["text"] = str(
                    raw_string[self.header_size(version) :].strip(b"\x00"),
                    "ascii",
                    errors="replace",
                )
            else:
                data["text"] = unicode(  # noqa
                    raw_string[self.header_size(version) :].strip("\x00"),
                    "ascii",
                    errors="replace",
                )

        return data

    def _pack_contents(self, data, version):

        datagram_fmt = self.header_fmt(version)
        datagram_contents = []

        if version == 0:

            for field in self.header_fields(version):
                datagram_contents.append(data[field])

            if data["text"][-1] != "\x00":
                tmp_string = data["text"] + "\x00"
            else:
                tmp_string = data["text"]

            # Pad with more nulls to 4-byte word boundry if necessary
            if len(tmp_string) % 4:
                tmp_string += "\x00" * (4 - (len(tmp_string) % 4))

            datagram_fmt += "%ds" % (len(tmp_string))
            datagram_contents.append(tmp_string)

        return struct.pack(datagram_fmt, *datagram_contents)


class SimradNMEAParser(_SimradDatagramParser):
    """
    ER60 NMEA datagram contains the following keys:


        type:         string == 'NME0'
        low_date:     long uint representing LSBytes of 64bit NT date
        high_date:    long uint representing MSBytes of 64bit NT date
        timestamp:     datetime.datetime object of NT date, assumed to be UTC

        nmea_string:  full (original) NMEA string

    The following methods are defined:

        from_string(str):    parse a raw ER60 NMEA datagram
                            (with leading/trailing datagram size stripped)

        to_string():         Returns the datagram as a raw string
                             (including leading/trailing size fields)
                             ready for writing to disk
    """

    nmea_head_re = re.compile("\$[A-Za-z]{5},")  # noqa

    def __init__(self):
        headers = {
            0: [("type", "4s"), ("low_date", "L"), ("high_date", "L")],
            1: [("type", "4s"), ("low_date", "L"), ("high_date", "L"), ("port", "32s")],
        }

        _SimradDatagramParser.__init__(self, "NME", headers)

    def _unpack_contents(self, raw_string, bytes_read, version):
        """
        Parses the NMEA string provided in raw_string

        :param raw_string:  Raw NMEA strin (i.e. '$GPZDA,160012.71,11,03,2004,-1,00*7D')
        :type raw_string: str

        :returns: None
        """

        header_values = struct.unpack(
            self.header_fmt(version), raw_string[: self.header_size(version)]
        )
        data = {}

        for indx, field in enumerate(self.header_fields(version)):
            data[field] = header_values[indx]
            if isinstance(data[field], bytes):
                data[field] = data[field].decode()

        data["timestamp"] = nt_to_unix((data["low_date"], data["high_date"]))
        data["bytes_read"] = bytes_read

        # Remove trailing \x00 from the PORT field for NME1, rest of the datagram identical to NME0
        if version == 1:
            data["port"] = data["port"].strip("\x00")

        if version == 0 or version == 1:
            if sys.version_info.major > 2:
                data["nmea_string"] = str(
                    raw_string[self.header_size(version) :].strip(b"\x00"),
                    "ascii",
                    errors="replace",
                )
            else:
                data["nmea_string"] = unicode(  # noqa
                    raw_string[self.header_size(version) :].strip("\x00"),
                    "ascii",
                    errors="replace",
                )

            if self.nmea_head_re.match(data["nmea_string"][:7]) is not None:
                data["nmea_talker"] = data["nmea_string"][1:3]
                data["nmea_type"] = data["nmea_string"][3:6]
            else:
                data["nmea_talker"] = ""
                data["nmea_type"] = "UNKNOWN"

        return data

    def _pack_contents(self, data, version):

        datagram_fmt = self.header_fmt(version)
        datagram_contents = []

        if version == 0:

            for field in self.header_fields(version):
                datagram_contents.append(data[field])

            if data["nmea_string"][-1] != "\x00":
                tmp_string = data["nmea_string"] + "\x00"
            else:
                tmp_string = data["nmea_string"]

            # Pad with more nulls to 4-byte word boundry if necessary
            if len(tmp_string) % 4:
                tmp_string += "\x00" * (4 - (len(tmp_string) % 4))

            datagram_fmt += "%ds" % (len(tmp_string))

            # Convert to python string if needed
            if isinstance(tmp_string, str):
                tmp_string = tmp_string.encode("ascii", errors="replace")

            datagram_contents.append(tmp_string)

        return struct.pack(datagram_fmt, *datagram_contents)


class SimradMRUParser(_SimradDatagramParser):
    """
    EK80 MRU datagram contains the following keys:


        type:         string == 'MRU0'
        low_date:     long uint representing LSBytes of 64bit NT date
        high_date:    long uint representing MSBytes of 64bit NT date
        timestamp:    datetime.datetime object of NT date, assumed to be UTC
        heave:        float
        roll :        float
        pitch:        float
        heading:      float

    The following methods are defined:

        from_string(str):    parse a raw ER60 NMEA datagram
                            (with leading/trailing datagram size stripped)

        to_string():         Returns the datagram as a raw string
                             (including leading/trailing size fields)
                             ready for writing to disk
    """

    def __init__(self):
        headers = {
            0: [
                ("type", "4s"),
                ("low_date", "L"),
                ("high_date", "L"),
                ("heave", "f"),
                ("roll", "f"),
                ("pitch", "f"),
                ("heading", "f"),
            ]
        }

        _SimradDatagramParser.__init__(self, "MRU", headers)

    def _unpack_contents(self, raw_string, bytes_read, version):
        """
        Unpacks the data in raw_string into dictionary containing MRU data

        :param raw_string:
        :type raw_string: str

        :returns: None
        """

        header_values = struct.unpack(
            self.header_fmt(version), raw_string[: self.header_size(version)]
        )
        data = {}

        for indx, field in enumerate(self.header_fields(version)):
            data[field] = header_values[indx]
            if isinstance(data[field], bytes):
                data[field] = data[field].decode()

        data["timestamp"] = nt_to_unix((data["low_date"], data["high_date"]))
        data["bytes_read"] = bytes_read

        return data

    def _pack_contents(self, data, version):

        datagram_fmt = self.header_fmt(version)
        datagram_contents = []

        if version == 0:

            for field in self.header_fields(version):
                datagram_contents.append(data[field])

            if data["nmea_string"][-1] != "\x00":
                tmp_string = data["nmea_string"] + "\x00"
            else:
                tmp_string = data["nmea_string"]

            # Pad with more nulls to 4-byte word boundry if necessary
            if len(tmp_string) % 4:
                tmp_string += "\x00" * (4 - (len(tmp_string) % 4))

            datagram_fmt += "%ds" % (len(tmp_string))

            # Convert to python string if needed
            if isinstance(tmp_string, str):
                tmp_string = tmp_string.encode("ascii", errors="replace")

            datagram_contents.append(tmp_string)

        return struct.pack(datagram_fmt, *datagram_contents)


class SimradXMLParser(_SimradDatagramParser):
    """
    EK80 XML datagram contains the following keys:


        type:         string == 'XML0'
        low_date:     long uint representing LSBytes of 64bit NT date
        high_date:    long uint representing MSBytes of 64bit NT date
        timestamp:    datetime.datetime object of NT date, assumed to be UTC
        subtype:      string representing Simrad XML datagram type:
                      configuration, environment, or parameter

        [subtype]:    dict containing the data specific to the XML subtype.

    The following methods are defined:

        from_string(str):    parse a raw EK80 XML datagram
                            (with leading/trailing datagram size stripped)

        to_string():         Returns the datagram as a raw string
                             (including leading/trailing size fields)
                             ready for writing to disk
    """

    #  define the XML parsing options - here we define dictionaries for various xml datagram
    #  types. When parsing that xml datagram, these dictionaries are used to inform the parser about
    #  type conversion, name wrangling, and delimiter. If a field is missing, the parser
    #  assumes no conversion: type will be string, default mangling, and that there is only 1
    #  element.
    #
    #  the dicts are in the form:
    #       'XMLParamName':[converted type,'fieldname', 'parse char']
    #
    #  For example: 'PulseDurationFM':[float,'pulse_duration_fm',';']
    #
    #  will result in a return dictionary field named 'pulse_duration_fm' that contains a list
    #  of float values parsed from a string that uses ';' to separate values. Empty strings
    #  for fieldname and/or parse char result in the default action for those parsing steps.

    channel_parsing_options = {
        "MaxTxPowerTransceiver": [int, "", ""],
        "PulseDuration": [float, "", ";"],
        "PulseDurationFM": [float, "pulse_duration_fm", ";"],
        "SampleInterval": [float, "", ";"],
        "ChannelID": [str, "channel_id", ""],
        "HWChannelConfiguration": [str, "hw_channel_configuration", ""],
    }

    transceiver_parsing_options = {
        "TransceiverNumber": [int, "", ""],
        "Version": [str, "transceiver_version", ""],
        "IPAddress": [str, "ip_address", ""],
        "Impedance": [int, "", ""],
    }

    transducer_parsing_options = {
        "SerialNumber": [str, "transducer_serial_number", ""],
        "Frequency": [float, "transducer_frequency", ""],
        "FrequencyMinimum": [float, "transducer_frequency_minimum", ""],
        "FrequencyMaximum": [float, "transducer_frequency_maximum", ""],
        "BeamType": [int, "transducer_beam_type", ""],
        "Gain": [float, "", ";"],
        "SaCorrection": [float, "", ";"],
        "MaxTxPowerTransducer": [float, "", ""],
        "EquivalentBeamAngle": [float, "", ""],
        "BeamWidthAlongship": [float, "", ""],
        "BeamWidthAthwartship": [float, "", ""],
        "AngleSensitivityAlongship": [float, "", ""],
        "AngleSensitivityAthwartship": [float, "", ""],
        "AngleOffsetAlongship": [float, "", ""],
        "AngleOffsetAthwartship": [float, "", ""],
        "DirectivityDropAt2XBeamWidth": [
            float,
            "directivity_drop_at_2x_beam_width",
            "",
        ],
        "TransducerOffsetX": [float, "", ""],
        "TransducerOffsetY": [float, "", ""],
        "TransducerOffsetZ": [float, "", ""],
        "TransducerAlphaX": [float, "", ""],
        "TransducerAlphaY": [float, "", ""],
        "TransducerAlphaZ": [float, "", ""],
    }

    header_parsing_options = {"Version": [str, "application_version", ""]}

    envxdcr_parsing_options = {"SoundSpeed": [float, "transducer_sound_speed", ""]}

    environment_parsing_options = {
        "Depth": [float, "", ""],
        "Acidity": [float, "", ""],
        "Salinity": [float, "", ""],
        "SoundSpeed": [float, "", ""],
        "Temperature": [float, "", ""],
        "Latitude": [float, "", ""],
        "SoundVelocityProfile": [float, "", ";"],
        "DropKeelOffset": [float, "", ""],
        "DropKeelOffsetIsManual": [int, "", ""],
        "WaterLevelDraft": [float, "", ""],
        "WaterLevelDraftIsManual": [int, "", ""],
    }

    parameter_parsing_options = {
        "ChannelID": [str, "channel_id", ""],
        "ChannelMode": [int, "", ""],
        "PulseForm": [int, "", ""],
        "Frequency": [float, "", ""],
        "PulseDuration": [float, "", ""],
        "SampleInterval": [float, "", ""],
        "TransmitPower": [float, "", ""],
        "Slope": [float, "", ""],
    }

    def __init__(self):
        headers = {0: [("type", "4s"), ("low_date", "L"), ("high_date", "L")]}
        _SimradDatagramParser.__init__(self, "XML", headers)

    def _unpack_contents(self, raw_string, bytes_read, version):
        """
        Parses the NMEA string provided in raw_string

        :param raw_string:  Raw NMEA strin (i.e. '$GPZDA,160012.71,11,03,2004,-1,00*7D')
        :type raw_string: str

        :returns: None
        """

        def from_CamelCase(xml_param):
            """
            convert name from CamelCase to fit with existing naming convention by
            inserting an underscore before each capital and then lowering the caps
            e.g. CamelCase becomes camel_case.
            """
            idx = list(reversed([i for i, c in enumerate(xml_param) if c.isupper()]))
            param_len = len(xml_param)
            for i in idx:
                #  check if we should insert an underscore
                if i > 0 and i < param_len:
                    xml_param = xml_param[:i] + "_" + xml_param[i:]
            xml_param = xml_param.lower()

            return xml_param

        def dict_to_dict(xml_dict, data_dict, parse_opts):
            """
            dict_to_dict appends the ETree xml value dicts to a provided dictionary
            and along the way converts the key name to conform to the project's
            naming convention and optionally parses and or converts values as
            specified in the parse_opts dictionary.
            """

            for k in xml_dict:
                #  check if we're parsing this key/value
                if k in parse_opts:
                    #  try to parse the string
                    if parse_opts[k][2]:
                        try:
                            data = xml_dict[k].split(parse_opts[k][2])
                        except:
                            #  bad or empty parse chararacter(s) provided
                            data = xml_dict[k]
                    else:
                        #  no parse char provided - nothing to parse
                        data = xml_dict[k]

                    #  try to convert to specified type
                    if isinstance(data, list):
                        for i in range(len(data)):
                            try:
                                data[i] = parse_opts[k][0](data[i])
                            except:
                                pass
                    else:
                        data = parse_opts[k][0](data)

                    #  and add the value to the provided dict
                    if parse_opts[k][1]:
                        #  add using the specified key name
                        data_dict[parse_opts[k][1]] = data
                    else:
                        #  add using the default key name wrangling
                        data_dict[from_CamelCase(k)] = data
                else:
                    #  nothing to do with the value string
                    data = xml_dict[k]

                    #  add the parameter to the provided dictionary
                    data_dict[from_CamelCase(k)] = data

        header_values = struct.unpack(
            self.header_fmt(version), raw_string[: self.header_size(version)]
        )
        data = {}

        for indx, field in enumerate(self.header_fields(version)):
            data[field] = header_values[indx]
            if isinstance(data[field], bytes):
                data[field] = data[field].decode()

        data["timestamp"] = nt_to_unix((data["low_date"], data["high_date"]))
        data["bytes_read"] = bytes_read

        if version == 0:
            if sys.version_info.major > 2:
                xml_string = str(
                    raw_string[self.header_size(version) :].strip(b"\x00"),
                    "ascii",
                    errors="replace",
                )
            else:
                xml_string = unicode(  # noqa
                    raw_string[self.header_size(version) :].strip("\x00"),
                    "ascii",
                    errors="replace",
                )

            #  get the ElementTree element
            root = ET.fromstring(xml_string)

            #  get the XML message type
            data["subtype"] = root.tag.lower()

            #  create the dictionary that contains the message data
            data[data["subtype"]] = {}

            #  parse it
            if data["subtype"] == "configuration":

                #  parse the Transceiver section
                for tcvr in root.iter("Transceiver"):
                    #  parse the Transceiver section
                    tcvr_xml = tcvr.attrib

                    #  parse the Channel section -- this works with multiple channels
                    #  under 1 transceiver
                    for tcvr_ch in tcvr.iter("Channel"):
                        tcvr_ch_xml = tcvr_ch.attrib
                        channel_id = tcvr_ch_xml["ChannelID"]

                        #  create the configuration dict for this channel
                        data["configuration"][channel_id] = {}

                        #  add the transceiver data to the config dict (this is
                        #  replicated for all channels)
                        dict_to_dict(
                            tcvr_xml,
                            data["configuration"][channel_id],
                            self.transceiver_parsing_options,
                        )

                        #  add the general channel data to the config dict
                        dict_to_dict(
                            tcvr_ch_xml,
                            data["configuration"][channel_id],
                            self.channel_parsing_options,
                        )

                        #  check if there are >1 transducer under a single transceiver channel
                        if len(list(tcvr_ch)) > 1:
                            ValueError(
                                "Found >1 transducer under a single transceiver channel!"
                            )
                        else:  # should only have 1 transducer
                            tcvr_ch_xducer = tcvr_ch.find(
                                "Transducer"
                            )  # get Element of this xducer
                            f_par = tcvr_ch_xducer.findall("FrequencyPar")
                            # Save calibration parameters
                            if f_par:
                                cal_par = {
                                    "frequency": np.array(
                                        [int(f.attrib["Frequency"]) for f in f_par]
                                    ),
                                    "gain": np.array(
                                        [float(f.attrib["Gain"]) for f in f_par]
                                    ),
                                    "impedance": np.array(
                                        [int(f.attrib["Impedance"]) for f in f_par]
                                    ),
                                    "phase": np.array(
                                        [float(f.attrib["Phase"]) for f in f_par]
                                    ),
                                    "beamwidth_alongship": np.array(
                                        [
                                            float(f.attrib["BeamWidthAlongship"])
                                            for f in f_par
                                        ]
                                    ),
                                    "beamwidth_athwartship": np.array(
                                        [
                                            float(f.attrib["BeamWidthAthwartship"])
                                            for f in f_par
                                        ]
                                    ),
                                    "angle_offset_alongship": np.array(
                                        [
                                            float(f.attrib["AngleOffsetAlongship"])
                                            for f in f_par
                                        ]
                                    ),
                                    "angle_offset_athwartship": np.array(
                                        [
                                            float(f.attrib["AngleOffsetAthwartship"])
                                            for f in f_par
                                        ]
                                    ),
                                }
                                data["configuration"][channel_id][
                                    "calibration"
                                ] = cal_par
                            #  add the transducer data to the config dict
                            dict_to_dict(
                                tcvr_ch_xducer.attrib,
                                data["configuration"][channel_id],
                                self.transducer_parsing_options,
                            )

                        # get unique transceiver channel number stored in channel_id
                        tcvr_ch_num = TCVR_CH_NUM_MATCHER.search(channel_id)[0]

                        # parse the Transducers section from the root
                        # TODO Remove Transducers if doesnt exist
                        xducer = root.find("Transducers")
                        if xducer is not None:
                            # built occurrence lookup table for transducer name
                            xducer_name_list = []
                            for xducer_ch in xducer.iter("Transducer"):
                                xducer_name_list.append(
                                    xducer_ch.attrib["TransducerName"]
                                )

                            # find matching transducer for this channel_id
                            match_found = False
                            for xducer_ch in xducer.iter("Transducer"):
                                if not match_found:
                                    xducer_ch_xml = xducer_ch.attrib
                                    match_name = (
                                        xducer_ch.attrib["TransducerName"]
                                        == tcvr_ch_xducer.attrib["TransducerName"]
                                    )
                                    if xducer_ch.attrib["TransducerSerialNumber"] == "":
                                        match_sn = False
                                    else:
                                        match_sn = (
                                            xducer_ch.attrib["TransducerSerialNumber"]
                                            == tcvr_ch_xducer.attrib["SerialNumber"]
                                        )
                                    match_tcvr = (
                                        tcvr_ch_num
                                        in xducer_ch.attrib["TransducerCustomName"]
                                    )

                                    # if find match add the transducer mounting details
                                    if (
                                        Counter(xducer_name_list)[
                                            xducer_ch.attrib["TransducerName"]
                                        ]
                                        > 1
                                    ):
                                        # if more than one transducer has the same name
                                        # only check sn and transceiver unique number
                                        match_found = match_sn or match_tcvr
                                    else:
                                        match_found = (
                                            match_name or match_sn or match_tcvr
                                        )

                                    # add transducer mounting details
                                    if match_found:
                                        dict_to_dict(
                                            xducer_ch_xml,
                                            data["configuration"][channel_id],
                                            self.transducer_parsing_options,
                                        )

                        #  add the header data to the config dict
                        h = root.find("Header")
                        dict_to_dict(
                            h.attrib,
                            data["configuration"][channel_id],
                            self.header_parsing_options,
                        )

            elif data["subtype"] == "parameter":

                #  parse the parameter XML datagram
                for h in root.iter("Channel"):
                    parm_xml = h.attrib
                    #  add the data to the environment dict
                    dict_to_dict(
                        parm_xml, data["parameter"], self.parameter_parsing_options
                    )

            elif data["subtype"] == "environment":

                #  parse the environment XML datagram
                for h in root.iter("Environment"):
                    env_xml = h.attrib
                    #  add the data to the environment dict
                    dict_to_dict(
                        env_xml, data["environment"], self.environment_parsing_options
                    )

                for h in root.iter("Transducer"):
                    transducer_xml = h.attrib
                    #  add the data to the environment dict
                    dict_to_dict(
                        transducer_xml,
                        data["environment"],
                        self.envxdcr_parsing_options,
                    )

        data["xml"] = xml_string
        return data

    def _pack_contents(self, data, version):
        def to_CamelCase(xml_param):
            """
            convert name from project's convention to CamelCase for converting back to
            XML to in Kongsberg's convention.
            """
            idx = list(reversed([i for i, c in enumerate(xml_param) if c.isupper()]))
            param_len = len(xml_param)
            for i in idx:
                #  check if we should insert an underscore
                if idx > 0 and idx < param_len - 1:
                    xml_param = xml_param[:idx] + "_" + xml_param[idx:]
            xml_param = xml_param.lower()

            return xml_param

        datagram_fmt = self.header_fmt(version)
        datagram_contents = []

        if version == 0:

            for field in self.header_fields(version):
                datagram_contents.append(data[field])

            if data["nmea_string"][-1] != "\x00":
                tmp_string = data["nmea_string"] + "\x00"
            else:
                tmp_string = data["nmea_string"]

            # Pad with more nulls to 4-byte word boundry if necessary
            if len(tmp_string) % 4:
                tmp_string += "\x00" * (4 - (len(tmp_string) % 4))

            datagram_fmt += "%ds" % (len(tmp_string))

            # Convert to python string if needed
            if isinstance(tmp_string, str):
                tmp_string = tmp_string.encode("ascii", errors="replace")

            datagram_contents.append(tmp_string)

        return struct.pack(datagram_fmt, *datagram_contents)


class SimradFILParser(_SimradDatagramParser):
    """
    EK80 FIL datagram contains the following keys:


        type:               string == 'FIL1'
        low_date:           long uint representing LSBytes of 64bit NT date
        high_date:          long uint representing MSBytes of 64bit NT date
        timestamp:          datetime.datetime object of NT date, assumed to be UTC
        stage:              int
        channel_id:         string
        n_coefficients:     int
        decimation_factor:  int
        coefficients:       np.complex64

    The following methods are defined:

        from_string(str):    parse a raw EK80 FIL datagram
                            (with leading/trailing datagram size stripped)

        to_string():         Returns the datagram as a raw string
                            (including leading/trailing size fields)
                             ready for writing to disk
    """

    def __init__(self):
        headers = {
            1: [
                ("type", "4s"),
                ("low_date", "L"),
                ("high_date", "L"),
                ("stage", "h"),
                ("spare", "2s"),
                ("channel_id", "128s"),
                ("n_coefficients", "h"),
                ("decimation_factor", "h"),
            ]
        }

        _SimradDatagramParser.__init__(self, "FIL", headers)

    def _unpack_contents(self, raw_string, bytes_read, version):

        data = {}
        header_values = struct.unpack(
            self.header_fmt(version), raw_string[: self.header_size(version)]
        )

        for indx, field in enumerate(self.header_fields(version)):
            data[field] = header_values[indx]

            #  handle Python 3 strings
            if (sys.version_info.major > 2) and isinstance(data[field], bytes):
                data[field] = data[field].decode("latin_1")

        data["timestamp"] = nt_to_unix((data["low_date"], data["high_date"]))
        data["bytes_read"] = bytes_read

        if version == 1:
            #  clean up the channel ID
            data["channel_id"] = data["channel_id"].strip("\x00")

            #  unpack the coefficients
            indx = self.header_size(version)
            block_size = data["n_coefficients"] * 8
            data["coefficients"] = np.frombuffer(
                raw_string[indx : indx + block_size], dtype="complex64"  # noqa
            )

        return data

    def _pack_contents(self, data, version):

        datagram_fmt = self.header_fmt(version)
        datagram_contents = []

        if version == 0:

            pass

        elif version == 1:
            for field in self.header_fields(version):
                datagram_contents.append(data[field])

            datagram_fmt += "%ds" % (len(data["beam_config"]))
            datagram_contents.append(data["beam_config"])

        return struct.pack(datagram_fmt, *datagram_contents)


class SimradConfigParser(_SimradDatagramParser):
    """
    Simrad Configuration Datagram parser operates on dictonaries with the following keys:

        type:         string == 'CON0'
        low_date:     long uint representing LSBytes of 64bit NT date
        high_date:    long uint representing MSBytes of 64bit NT date
        timestamp:    datetime.datetime object of NT date, assumed to be UTC

        survey_name                     [str]
        transect_name                   [str]
        sounder_name                    [str]
        version                         [str]
        spare0                          [str]
        transceiver_count               [long]
        transceivers                    [list] List of dicts representing Transducer Configs:

        ME70 Data contains the following additional values (data contained w/in first 14
            bytes of the spare0 field)

        multiplexing                    [short]  Always 0
        time_bias                       [long] difference between UTC and local time in min.
        sound_velocity_avg              [float] [m/s]
        sound_velocity_transducer       [float] [m/s]
        beam_config                     [str] Raw XML string containing beam config. info


    Transducer Config Keys (ER60/ES60 sounders):
        channel_id                      [str]   channel ident string
        beam_type                       [long]  Type of channel (0 = Single, 1 = Split)
        frequency                       [float] channel frequency
        equivalent_beam_angle           [float] dB
        beamwidth_alongship             [float]
        beamwidth_athwartship           [float]
        angle_sensitivity_alongship     [float]
        angle_sensitivity_athwartship   [float]
        angle_offset_alongship          [float]
        angle_offset_athwartship        [float]
        pos_x                           [float]
        pos_y                           [float]
        pos_z                           [float]
        dir_x                           [float]
        dir_y                           [float]
        dir_z                           [float]
        pulse_length_table              [float[5]]
        spare1                          [str]
        gain_table                      [float[5]]
        spare2                          [str]
        sa_correction_table             [float[5]]
        spare3                          [str]
        gpt_software_version            [str]
        spare4                          [str]

    Transducer Config Keys (ME70 sounders):
        channel_id                      [str]   channel ident string
        beam_type                       [long]  Type of channel (0 = Single, 1 = Split)
        reserved1                       [float] channel frequency
        equivalent_beam_angle           [float] dB
        beamwidth_alongship             [float]
        beamwidth_athwartship           [float]
        angle_sensitivity_alongship     [float]
        angle_sensitivity_athwartship   [float]
        angle_offset_alongship          [float]
        angle_offset_athwartship        [float]
        pos_x                           [float]
        pos_y                           [float]
        pos_z                           [float]
        beam_steering_angle_alongship   [float]
        beam_steering_angle_athwartship [float]
        beam_steering_angle_unused      [float]
        pulse_length                    [float]
        reserved2                       [float]
        spare1                          [str]
        gain                            [float]
        reserved3                       [float]
        spare2                          [str]
        sa_correction                   [float]
        reserved4                       [float]
        spare3                          [str]
        gpt_software_version            [str]
        spare4                          [str]

    from_string(str):   parse a raw config datagram
                        (with leading/trailing datagram size stripped)

    to_string(dict):    Returns raw string (including leading/trailing size fields)
                        ready for writing to disk
    """

    def __init__(self):
        headers = {
            0: [
                ("type", "4s"),
                ("low_date", "L"),
                ("high_date", "L"),
                ("survey_name", "128s"),
                ("transect_name", "128s"),
                ("sounder_name", "128s"),
                ("version", "30s"),
                ("spare0", "98s"),
                ("transceiver_count", "l"),
            ],
            1: [("type", "4s"), ("low_date", "L"), ("high_date", "L")],
        }

        _SimradDatagramParser.__init__(self, "CON", headers)

        self._transducer_headers = {
            "ER60": [
                ("channel_id", "128s"),
                ("beam_type", "l"),
                ("frequency", "f"),
                ("gain", "f"),
                ("equivalent_beam_angle", "f"),
                ("beamwidth_alongship", "f"),
                ("beamwidth_athwartship", "f"),
                ("angle_sensitivity_alongship", "f"),
                ("angle_sensitivity_athwartship", "f"),
                ("angle_offset_alongship", "f"),
                ("angle_offset_athwartship", "f"),
                ("pos_x", "f"),
                ("pos_y", "f"),
                ("pos_z", "f"),
                ("dir_x", "f"),
                ("dir_y", "f"),
                ("dir_z", "f"),
                ("pulse_length_table", "5f"),
                ("spare1", "8s"),
                ("gain_table", "5f"),
                ("spare2", "8s"),
                ("sa_correction_table", "5f"),
                ("spare3", "8s"),
                ("gpt_software_version", "16s"),
                ("spare4", "28s"),
            ],
            "ES60": [
                ("channel_id", "128s"),
                ("beam_type", "l"),
                ("frequency", "f"),
                ("gain", "f"),
                ("equivalent_beam_angle", "f"),
                ("beamwidth_alongship", "f"),
                ("beamwidth_athwartship", "f"),
                ("angle_sensitivity_alongship", "f"),
                ("angle_sensitivity_athwartship", "f"),
                ("angle_offset_alongship", "f"),
                ("angle_offset_athwartship", "f"),
                ("pos_x", "f"),
                ("pos_y", "f"),
                ("pos_z", "f"),
                ("dir_x", "f"),
                ("dir_y", "f"),
                ("dir_z", "f"),
                ("pulse_length_table", "5f"),
                ("spare1", "8s"),
                ("gain_table", "5f"),
                ("spare2", "8s"),
                ("sa_correction_table", "5f"),
                ("spare3", "8s"),
                ("gpt_software_version", "16s"),
                ("spare4", "28s"),
            ],
            "MBES": [
                ("channel_id", "128s"),
                ("beam_type", "l"),
                ("frequency", "f"),
                ("reserved1", "f"),
                ("equivalent_beam_angle", "f"),
                ("beamwidth_alongship", "f"),
                ("beamwidth_athwartship", "f"),
                ("angle_sensitivity_alongship", "f"),
                ("angle_sensitivity_athwartship", "f"),
                ("angle_offset_alongship", "f"),
                ("angle_offset_athwartship", "f"),
                ("pos_x", "f"),
                ("pos_y", "f"),
                ("pos_z", "f"),
                ("beam_steering_angle_alongship", "f"),
                ("beam_steering_angle_athwartship", "f"),
                ("beam_steering_angle_unused", "f"),
                ("pulse_length", "f"),
                ("reserved2", "f"),
                ("spare1", "20s"),
                ("gain", "f"),
                ("reserved3", "f"),
                ("spare2", "20s"),
                ("sa_correction", "f"),
                ("reserved4", "f"),
                ("spare3", "20s"),
                ("gpt_software_version", "16s"),
                ("spare4", "28s"),
            ],
        }

    def _unpack_contents(self, raw_string, bytes_read, version):

        data = {}
        round6 = lambda x: round(x, ndigits=6)  # noqa
        header_values = struct.unpack(
            self.header_fmt(version), raw_string[: self.header_size(version)]
        )

        for indx, field in enumerate(self.header_fields(version)):
            data[field] = header_values[indx]

            #  handle Python 3 strings
            if (sys.version_info.major > 2) and isinstance(data[field], bytes):
                data[field] = data[field].decode("latin_1")

        data["timestamp"] = nt_to_unix((data["low_date"], data["high_date"]))
        data["bytes_read"] = bytes_read

        if version == 0:

            data["transceivers"] = {}

            for field in ["transect_name", "version", "survey_name", "sounder_name"]:
                data[field] = data[field].strip("\x00")

            sounder_name = data["sounder_name"]
            if sounder_name == "MBES":
                _me70_extra_values = struct.unpack("=hLff", data["spare0"][:14])
                data["multiplexing"] = _me70_extra_values[0]
                data["time_bias"] = _me70_extra_values[1]
                data["sound_velocity_avg"] = _me70_extra_values[2]
                data["sound_velocity_transducer"] = _me70_extra_values[3]
                data["spare0"] = data["spare0"][:14] + data["spare0"][14:].strip("\x00")

            else:
                data["spare0"] = data["spare0"].strip("\x00")

            buf_indx = self.header_size(version)

            try:
                transducer_header = self._transducer_headers[sounder_name]
                _sounder_name_used = sounder_name
            except KeyError:
                log.warning(
                    "Unknown sounder_name:  %s, (no one of %s)",
                    sounder_name,
                    list(self._transducer_headers.keys()),
                )
                log.warning("Will use ER60 transducer config fields as default")

                transducer_header = self._transducer_headers["ER60"]
                _sounder_name_used = "ER60"

            txcvr_header_fields = [x[0] for x in transducer_header]
            txcvr_header_fmt = "=" + "".join([x[1] for x in transducer_header])
            txcvr_header_size = struct.calcsize(txcvr_header_fmt)

            for txcvr_indx in range(1, data["transceiver_count"] + 1):
                txcvr_header_values_encoded = struct.unpack(
                    txcvr_header_fmt,
                    raw_string[buf_indx : buf_indx + txcvr_header_size],  # noqa
                )
                txcvr_header_values = list(txcvr_header_values_encoded)
                for tx_idx, tx_val in enumerate(txcvr_header_values_encoded):
                    if isinstance(tx_val, bytes):
                        txcvr_header_values[tx_idx] = tx_val.decode()

                txcvr = data["transceivers"].setdefault(txcvr_indx, {})

                if _sounder_name_used in ["ER60", "ES60"]:
                    for txcvr_field_indx, field in enumerate(txcvr_header_fields[:17]):
                        txcvr[field] = txcvr_header_values[txcvr_field_indx]

                    txcvr["pulse_length_table"] = np.fromiter(
                        list(map(round6, txcvr_header_values[17:22])), "float"
                    )
                    txcvr["spare1"] = txcvr_header_values[22]
                    txcvr["gain_table"] = np.fromiter(
                        list(map(round6, txcvr_header_values[23:28])), "float"
                    )
                    txcvr["spare2"] = txcvr_header_values[28]
                    txcvr["sa_correction_table"] = np.fromiter(
                        list(map(round6, txcvr_header_values[29:34])), "float"
                    )
                    txcvr["spare3"] = txcvr_header_values[34]
                    txcvr["gpt_software_version"] = txcvr_header_values[35]
                    txcvr["spare4"] = txcvr_header_values[36]

                elif _sounder_name_used == "MBES":
                    for txcvr_field_indx, field in enumerate(txcvr_header_fields):
                        txcvr[field] = txcvr_header_values[txcvr_field_indx]

                else:
                    raise RuntimeError(
                        "Unknown _sounder_name_used (Should not happen, this is a bug!)"
                    )

                txcvr["channel_id"] = txcvr["channel_id"].strip("\x00")
                txcvr["spare1"] = txcvr["spare1"].strip("\x00")
                txcvr["spare2"] = txcvr["spare2"].strip("\x00")
                txcvr["spare3"] = txcvr["spare3"].strip("\x00")
                txcvr["spare4"] = txcvr["spare4"].strip("\x00")
                txcvr["gpt_software_version"] = txcvr["gpt_software_version"].strip(
                    "\x00"
                )

                buf_indx += txcvr_header_size

        elif version == 1:
            # CON1 only has a single data field:  beam_config, holding an xml string
            data["beam_config"] = raw_string[self.header_size(version) :].strip("\x00")

        return data

    def _pack_contents(self, data, version):

        datagram_fmt = self.header_fmt(version)
        datagram_contents = []

        if version == 0:

            if data["transceiver_count"] != len(data["transceivers"]):
                log.warning(
                    "Mismatch between 'transceiver_count' and actual # of transceivers"
                )
                data["transceiver_count"] = len(data["transceivers"])

            sounder_name = data["sounder_name"]
            if sounder_name == "MBES":
                _packed_me70_values = struct.pack(
                    "=hLff",
                    data["multiplexing"],
                    data["time_bias"],
                    data["sound_velocity_avg"],
                    data["sound_velocity_transducer"],
                )
                data["spare0"] = _packed_me70_values + data["spare0"][14:]

            for field in self.header_fields(version):
                datagram_contents.append(data[field])

            try:
                transducer_header = self._transducer_headers[sounder_name]
                _sounder_name_used = sounder_name
            except KeyError:
                log.warning(
                    "Unknown sounder_name:  %s, (no one of %s)",
                    sounder_name,
                    list(self._transducer_headers.keys()),
                )
                log.warning("Will use ER60 transducer config fields as default")

                transducer_header = self._transducer_headers["ER60"]
                _sounder_name_used = "ER60"

            txcvr_header_fields = [x[0] for x in transducer_header]
            txcvr_header_fmt = "=" + "".join([x[1] for x in transducer_header])
            txcvr_header_size = struct.calcsize(txcvr_header_fmt)  # noqa

            for txcvr_indx, txcvr in list(data["transceivers"].items()):
                txcvr_contents = []

                if _sounder_name_used in ["ER60", "ES60"]:
                    for field in txcvr_header_fields[:17]:
                        txcvr_contents.append(txcvr[field])

                    txcvr_contents.extend(txcvr["pulse_length_table"])
                    txcvr_contents.append(txcvr["spare1"])

                    txcvr_contents.extend(txcvr["gain_table"])
                    txcvr_contents.append(txcvr["spare2"])

                    txcvr_contents.extend(txcvr["sa_correction_table"])
                    txcvr_contents.append(txcvr["spare3"])

                    txcvr_contents.extend(
                        [txcvr["gpt_software_version"], txcvr["spare4"]]
                    )

                    txcvr_contents_str = struct.pack(txcvr_header_fmt, *txcvr_contents)

                elif _sounder_name_used == "MBES":
                    for field in txcvr_header_fields:
                        txcvr_contents.append(txcvr[field])

                    txcvr_contents_str = struct.pack(txcvr_header_fmt, *txcvr_contents)

                else:
                    raise RuntimeError(
                        "Unknown _sounder_name_used (Should not happen, this is a bug!)"
                    )

                datagram_fmt += "%ds" % (len(txcvr_contents_str))
                datagram_contents.append(txcvr_contents_str)

        elif version == 1:
            for field in self.header_fields(version):
                datagram_contents.append(data[field])

            datagram_fmt += "%ds" % (len(data["beam_config"]))
            datagram_contents.append(data["beam_config"])

        return struct.pack(datagram_fmt, *datagram_contents)


class SimradRawParser(_SimradDatagramParser):
    """
    Sample Data Datagram parser operates on dictonaries with the following keys:

        type:         string == 'RAW0'
        low_date:     long uint representing LSBytes of 64bit NT date
        high_date:    long uint representing MSBytes of 64bit NT date
        timestamp:    datetime.datetime object of NT date, assumed to be UTC

        channel                         [short] Channel number
        mode                            [short] 1 = Power only, 2 = Angle only 3 = Power & Angle
        transducer_depth                [float]
        frequency                       [float]
        transmit_power                  [float]
        pulse_length                    [float]
        bandwidth                       [float]
        sample_interval                 [float]
        sound_velocity                  [float]
        absorption_coefficient          [float]
        heave                           [float]
        roll                            [float]
        pitch                           [float]
        temperature                     [float]
        heading                         [float]
        transmit_mode                   [short] 0 = Active, 1 = Passive, 2 = Test, -1 = Unknown
        spare0                          [str]
        offset                          [long]
        count                           [long]

        power                           [numpy array] Unconverted power values (if present)
        angle                           [numpy array] Unconverted angle values (if present)

    from_string(str):   parse a raw sample datagram
                        (with leading/trailing datagram size stripped)

    to_string(dict):    Returns raw string (including leading/trailing size fields)
                        ready for writing to disk
    """

    def __init__(self):
        headers = {
            0: [
                ("type", "4s"),
                ("low_date", "L"),
                ("high_date", "L"),
                ("channel", "h"),
                ("mode", "h"),
                ("transducer_depth", "f"),
                ("frequency", "f"),
                ("transmit_power", "f"),
                ("pulse_length", "f"),
                ("bandwidth", "f"),
                ("sample_interval", "f"),
                ("sound_velocity", "f"),
                ("absorption_coefficient", "f"),
                ("heave", "f"),
                ("roll", "f"),
                ("pitch", "f"),
                ("temperature", "f"),
                ("heading", "f"),
                ("transmit_mode", "h"),
                ("spare0", "6s"),
                ("offset", "l"),
                ("count", "l"),
            ],
            3: [
                ("type", "4s"),
                ("low_date", "L"),
                ("high_date", "L"),
                ("channel_id", "128s"),
                ("data_type", "h"),
                ("spare", "2s"),
                ("offset", "l"),
                ("count", "l"),
            ],
        }
        _SimradDatagramParser.__init__(self, "RAW", headers)

    def _unpack_contents(self, raw_string, bytes_read, version):

        header_values = struct.unpack(
            self.header_fmt(version), raw_string[: self.header_size(version)]
        )

        data = {}

        for indx, field in enumerate(self.header_fields(version)):
            data[field] = header_values[indx]
            if isinstance(data[field], bytes):
                data[field] = data[field].decode()

        data["timestamp"] = nt_to_unix((data["low_date"], data["high_date"]))
        data["bytes_read"] = bytes_read

        if version == 0:

            if data["count"] > 0:
                block_size = data["count"] * 2
                indx = self.header_size(version)

                if int(data["mode"]) & 0x1:
                    data["power"] = np.frombuffer(
                        raw_string[indx : indx + block_size], dtype="int16"  # noqa
                    )
                    indx += block_size
                else:
                    data["power"] = None

                if int(data["mode"]) & 0x2:
                    data["angle"] = np.frombuffer(
                        raw_string[indx : indx + block_size], dtype="int8"  # noqa
                    )
                    data["angle"] = data["angle"].reshape((-1, 2))
                else:
                    data["angle"] = None

            else:
                data["power"] = np.empty((0,), dtype="int16")
                data["angle"] = np.empty((0,), dtype="int8")

        elif version == 3:

            # result = 1j*Data[...,1]; result += Data[...,0]

            #  clean up the channel ID
            data["channel_id"] = data["channel_id"].strip("\x00")

            if data["count"] > 0:

                #  set the initial block size and indx value.
                block_size = data["count"] * 2
                indx = self.header_size(version)

                if data["data_type"] & 0b1:
                    data["power"] = np.frombuffer(
                        raw_string[indx : indx + block_size], dtype="int16"  # noqa
                    )
                    indx += block_size
                else:
                    data["power"] = None

                if data["data_type"] & 0b10:
                    data["angle"] = np.frombuffer(
                        raw_string[indx : indx + block_size], dtype="int8"  # noqa
                    )
                    data["angle"] = data["angle"].reshape((-1, 2))
                    indx += block_size
                else:
                    data["angle"] = None

                #  determine the complex sample data type - this is contained in bits 2 and 3
                #  of the datatype <short> value. I'm assuming the types are exclusive...
                data["complex_dtype"] = np.float16
                type_bytes = 2
                if data["data_type"] & 0b1000:
                    data["complex_dtype"] = np.float32
                    type_bytes = 8

                #  determine the number of complex samples
                data["n_complex"] = data["data_type"] >> 8

                #  unpack the complex samples
                if data["n_complex"] > 0:
                    #  determine the block size
                    block_size = data["count"] * data["n_complex"] * type_bytes

                    data["complex"] = np.frombuffer(
                        raw_string[indx : indx + block_size],  # noqa
                        dtype=data["complex_dtype"],
                    )
                    data["complex"].dtype = np.complex64
                else:
                    data["complex"] = None

            else:
                data["power"] = np.empty((0,), dtype="int16")
                data["angle"] = np.empty((0,), dtype="int8")
                data["complex"] = np.empty((0,), dtype="complex64")
                data["n_complex"] = 0

        return data

    def _pack_contents(self, data, version):

        datagram_fmt = self.header_fmt(version)

        datagram_contents = []

        if version == 0:

            if data["count"] > 0:
                if (int(data["mode"]) & 0x1) and (
                    len(data.get("power", [])) != data["count"]
                ):
                    log.warning(
                        "Data 'count' = %d, but contains %d power samples.  Ignoring power."
                    )
                    data["mode"] &= ~(1 << 0)

                if (int(data["mode"]) & 0x2) and (
                    len(data.get("angle", [])) != data["count"]
                ):
                    log.warning(
                        "Data 'count' = %d, but contains %d angle samples.  Ignoring angle."
                    )
                    data["mode"] &= ~(1 << 1)

                if data["mode"] == 0:
                    log.warning(
                        "Data 'count' = %d, but mode == 0.  Setting count to 0",
                        data["count"],
                    )
                    data["count"] = 0

            for field in self.header_fields(version):
                datagram_contents.append(data[field])

            if data["count"] > 0:

                if int(data["mode"]) & 0x1:
                    datagram_fmt += "%dh" % (data["count"])
                    datagram_contents.extend(data["power"])

                if int(data["mode"]) & 0x2:
                    datagram_fmt += "%dH" % (data["count"])
                    datagram_contents.extend(data["angle"])

        return struct.pack(datagram_fmt, *datagram_contents)
