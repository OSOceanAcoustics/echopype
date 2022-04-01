"""
The purpose of this dictionary is to define the structure of each sensor
across echopype versions. The dictionary was developed to ease
the transition from echopype version 0.5.x to version 0.6.x. This dict
primarily focuses on names each EchoData group and their coordinates.
The structure is currently based on the SONAR-netCDF4 convention
file 1.0.yml.

PLEASE CAREFULLY REVIEW THE RULES BEFORE APPLYING CHANGES.

Rules of the structure:

1. For each sensor, the keys of each group should correspond to the groups in 1.0.yml
2. The ep_group should correspond to the group that would be used to access the DataTree.
2. coords should contain all POSSIBLE unique coordinates.
3. If you add a version structure, please add this version to all sensors.
4. For each coords value carefully compare and modify all corresponding
coords of the other groups. For example, if you were to add 'beam' to coords, giving you
'coords': ['channel', 'ping_time', 'range_sample', 'beam'], then you should add
None to any coords that do not have that coordinate (in that sensor). For exampple,
'coords': ['frequency', 'ping_time', 'range_bin', None]
5. If the group does not apply to the sensor, you should replace the dict with None
"""

SENSOR_EP_VERSION_MAPPINGS = {
    "AD2CP": {
        "v0.5.x": {
            "top": {
                "ep_group": None,
                "coords": [],
            },
            "environment": {
                "ep_group": "Environment",
                "coords": [
                    "ping_time",
                    "ping_time_burst",
                    "ping_time_average",
                    "ping_time_echosounder",
                ],
            },
            "platform": {
                "ep_group": "Platform",
                "coords": [
                    "ping_time",
                    "ping_time_burst",
                    "ping_time_average",
                    "ping_time_echosounder",
                    "beam",
                    "range_bin_burst",
                    "range_bin_average",
                    "range_bin_echosounder",
                ],
            },
            "provenance": {
                "ep_group": "Provenance",
                "coords": [],
            },
            "sonar": {
                "ep_group": "Sonar",
                "coords": [],
            },
            "vendor": {
                "ep_group": "Vendor",
                "coords": [
                    "ping_time",
                    "ping_time_burst",
                    "ping_time_average",
                    "ping_time_echosounder",
                    "ping_time_echosounder_raw",
                    "ping_time_echosounder_raw_transmit",
                    "sample",
                    "sample_transmit",
                    "beam",
                    "range_bin_average",
                    "range_bin_burst",
                    "range_bin_echosounder",
                ],
            },
            "beam": {
                "ep_group": "Beam",
                "coords": [
                    "ping_time",
                    "ping_time_burst",
                    "ping_time_average",
                    "ping_time_echosounder",
                    "beam",
                    "range_bin_burst",
                    "range_bin_average",
                    "range_bin_echosounder",
                    "altimeter_sample_bin",
                ],
            },
            "beam_power": None,
            "nmea": {
                "ep_group": "Platform/NMEA",
                "coords": ["location_time"],
            },
        },
        "v0.6.x": {
            "top": {
                "ep_group": None,
                "coords": [],
            },
            "environment": {
                "ep_group": "Environment",
                "coords": [
                    "ping_time",
                    "ping_time_burst",
                    "ping_time_average",
                    "ping_time_echosounder",
                ],
            },
            "platform": {
                "ep_group": "Platform",
                "coords": [
                    "ping_time",
                    "ping_time_burst",
                    "ping_time_average",
                    "ping_time_echosounder",
                    "beam",
                    "range_sample_burst",
                    "range_sample_average",
                    "range_sample_echosounder",
                ],
            },
            "provenance": {
                "ep_group": "Provenance",
                "coords": [],
            },
            "sonar": {
                "ep_group": "Sonar",
                "coords": [],
            },
            "vendor": {
                "ep_group": "Vendor",
                "coords": [
                    "ping_time",
                    "ping_time_burst",
                    "ping_time_average",
                    "ping_time_echosounder",
                    "ping_time_echosounder_raw",
                    "ping_time_echosounder_raw_transmit",
                    "sample",
                    "sample_transmit",
                    "beam",
                    "range_sample_average",
                    "range_sample_burst",
                    "range_sample_echosounder",
                ],
            },
            "beam": {
                "ep_group": "Sonar/Beam_group1",
                "coords": [
                    "ping_time",
                    "ping_time_burst",
                    "ping_time_average",
                    "ping_time_echosounder",
                    "beam",
                    "range_sample_burst",
                    "range_sample_average",
                    "range_sample_echosounder",
                    "altimeter_sample_bin",
                ],
                "group_descr": "contains velocity, correlation, and backscatter power "
                + "(uncalibrated) data and other data derived from acoustic data",
            },
            "beam_power": None,
            "nmea": {
                "ep_group": "Platform/NMEA",
                "coords": ["location_time"],
            },
        },
    },
    "AZFP": {
        "v0.5.x": {
            "top": {
                "ep_group": None,
                "coords": [],
            },
            "environment": {
                "ep_group": "Environment",
                "coords": ["ping_time"],
            },
            "platform": {
                "ep_group": "Platform",
                "coords": [],
            },
            "provenance": {
                "ep_group": "Provenance",
                "coords": [],
            },
            "sonar": {
                "ep_group": "Sonar",
                "coords": [None],
            },
            "vendor": {
                "ep_group": "Vendor",
                "coords": ["frequency", "ping_time", "ancillary_len", "ad_len"],
            },
            "beam": {
                "ep_group": "Beam",
                "coords": ["frequency", "ping_time", "range_bin"],
            },
            "beam_power": None,
            "nmea": {
                "ep_group": "Platform/NMEA",
                "coords": ["location_time"],
            },
        },
        "v0.6.x": {
            "top": {
                "ep_group": None,
                "coords": [],
            },
            "environment": {
                "ep_group": "Environment",
                "coords": ["ping_time"],
            },
            "platform": {
                "ep_group": "Platform",
                "coords": [],
            },
            "provenance": {
                "ep_group": "Provenance",
                "coords": [],
            },
            "sonar": {
                "ep_group": "Sonar",
                "coords": ["beam"],
            },
            "vendor": {
                "ep_group": "Vendor",
                "coords": ["channel", "ping_time", "ancillary_len", "ad_len"],
            },
            "beam": {
                "ep_group": "Sonar/Beam_group1",
                "coords": ["channel", "ping_time", "range_sample"],
                "group_descr": "contains backscatter power (uncalibrated) and other beam"
                + " or channel-specific data.",
            },
            "beam_power": None,
            "nmea": {
                "ep_group": "Platform/NMEA",
                "coords": ["location_time"],
            },
        },
    },
    "EK60": {
        "v0.5.x": {
            "top": {
                "ep_group": None,
                "coords": [],
            },
            "environment": {
                "ep_group": "Environment",
                "coords": ["frequency", "ping_time"],
            },
            "platform": {
                "ep_group": "Platform",
                "coords": ["location_time", "ping_time", "frequency"],
            },
            "provenance": {
                "ep_group": "Provenance",
                "coords": [],
            },
            "sonar": {
                "ep_group": "Sonar",
                "coords": [],
            },
            "vendor": {
                "ep_group": "Vendor",
                "coords": ["frequency", "pulse_length_bin"],
            },
            "beam": {
                "ep_group": "Beam",
                "coords": ["frequency", "ping_time", "range_bin", None],
            },
            "beam_power": None,
            "nmea": {
                "ep_group": "Platform/NMEA",
                "coords": ["location_time"],
            },
        },
        "v0.6.x": {
            "top": {
                "ep_group": None,
                "coords": [],
            },
            "environment": {
                "ep_group": "Environment",
                "coords": ["channel", "ping_time"],
            },
            "platform": {
                "ep_group": "Platform",
                "coords": ["location_time", "ping_time", "channel"],
            },
            "provenance": {
                "ep_group": "Provenance",
                "coords": [],
            },
            "sonar": {
                "ep_group": "Sonar",
                "coords": ["beam_group"],
            },
            "vendor": {
                "ep_group": "Vendor",
                "coords": ["channel", "pulse_length_bin"],
            },
            "beam": {
                "ep_group": "Sonar/Beam_group1",
                "group_descr": "contains backscatter power (uncalibrated) and other beam or"
                + " channel-specific data, including split-beam angle data when they exist.",
                "coords": ["channel", "ping_time", "range_sample", "beam"],
            },
            "beam_power": None,
            "nmea": {
                "ep_group": "Platform/NMEA",
                "coords": ["location_time"],
            },
        },
    },
    "EK80": {
        "v0.5.x": {
            "top": {
                "ep_group": None,
                "coords": [],
            },
            "environment": {
                "ep_group": "Environment",
                "coords": ["ping_time"],
            },
            "platform": {
                "ep_group": "Platform",
                "coords": ["mru_time", "location_time"],
            },
            "provenance": {
                "ep_group": "Provenance",
                "coords": [],
            },
            "sonar": {
                "ep_group": "Sonar",
                "coords": [None, "frequency"],
            },
            "vendor": {
                "ep_group": "Vendor",
                "coords": ["frequency", "pulse_length_bin", "cal_frequency", "cal_channel_id"],
            },
            "beam": {
                "ep_group": "Beam",
                "coords": ["frequency", "ping_time", "range_bin", "quadrant"],
            },
            "beam_power": {
                "ep_group": "Beam_power",
                "coords": ["frequency", "ping_time", "range_bin"],
            },
            "nmea": {
                "ep_group": "Platform/NMEA",
                "coords": ["location_time"],
            },
        },
        "v0.6.x": {
            "top": {
                "ep_group": None,
                "coords": [],
            },
            "environment": {
                "ep_group": "Environment",
                "coords": ["ping_time"],
            },
            "platform": {
                "ep_group": "Platform",
                "coords": ["mru_time", "location_time"],
            },
            "provenance": {
                "ep_group": "Provenance",
                "coords": [],
            },
            "sonar": {
                "ep_group": "Sonar",
                "coords": ["beam", "channel"],
            },
            "vendor": {
                "ep_group": "Vendor",
                "coords": ["channel", "pulse_length_bin", "cal_frequency", "cal_channel_id"],
            },
            "beam": {
                "ep_group": "Sonar/Beam_group1",
                "coords": ["channel", "ping_time", "range_sample", "quadrant"],
                "group_descr": "contains complex backscatter data and other beam"
                + " or channel-specific data.",
            },
            "beam_power": {
                "ep_group": "Sonar/Beam_group2",
                "coords": ["channel", "ping_time", "range_sample"],
                "group_descr": "contains backscatter power (uncalibrated) and other beam"
                + " or channel-specific data,"
                + " including split-beam angle data when they exist.",
            },
            "nmea": {
                "ep_group": "Platform/NMEA",
                "coords": ["location_time"],
            },
        },
    },
}
