"""
Define convention-based global, coordinate and variable attributes
in one place for consistent reuse
"""

DEFAULT_BEAM_COORD_ATTRS = {
    "frequency": {
        "long_name": "Transducer frequency",
        "standard_name": "sound_frequency",
        "units": "Hz",
        "valid_min": 0.0,
    },
    "ping_time": {
        "long_name": "Timestamp of each ping",
        "standard_name": "time",
        "axis": "T",
    },
    "range_bin": {"long_name": "Along-range bin (sample) number, base 0"},
}

DEFAULT_PLATFORM_COORD_ATTRS = {
    "location_time": {
        "axis": "T",
        "long_name": "Timestamps for NMEA datagrams",
        "standard_name": "time",
    }
}

DEFAULT_PLATFORM_VAR_ATTRS = {
    "latitude": {
        "long_name": "Platform latitude",
        "standard_name": "latitude",
        "units": "degrees_north",
        "valid_range": (-90.0, 90.0),
    },
    "longitude": {
        "long_name": "Platform longitude",
        "standard_name": "longitude",
        "units": "degrees_east",
        "valid_range": (-180.0, 180.0),
    },
    "pitch": {
        "long_name": "Platform pitch",
        "standard_name": "platform_pitch_angle",
        "units": "arc_degree",
        "valid_range": (-90.0, 90.0),
    },
    "roll": {
        "long_name": "Platform roll",
        "standard_name": "platform_roll_angle",
        "units": "arc_degree",
        "valid_range": (-90.0, 90.0),
    },
    "heave": {
        "long_name": "Platform heave",
        "standard_name": "platform_heave_angle",
        "units": "arc_degree",
        "valid_range": (-90.0, 90.0),
    },
    "water_level": {
        "long_name": "z-axis distance from the platform coordinate system "
        "origin to the sonar transducer",
        "units": "m",
    },
}
