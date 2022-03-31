
#######################################################################################################################
# PLEASE CAREFULLY REVIEW THE RULES BEFORE APPLYING CHANGES.
#
# Rules of the structure:
#
# 1. For each sensor, the keys of each group should correspond to the groups in 1.0.yml
# 2. The ep_group should correspond to the group that would be used to access the DataTree.
# 2. coords should contain all POSSIBLE unique coordinates.
# 3. If you add a version structure, please add this version to all sensors.
# 4. For each coords value carefully compare and modify all corresponding
# coords of the other groups. For example, if you were to add 'beam' to coords, giving you
# "coords": ['channel', 'ping_time', 'range_sample', 'beam'], then you should add
# None to any coords that do not have that coordinate (in that sensor). For exampple,
# "coords": ['frequency', 'ping_time', 'range_bin', None]
# 5. If the group does not apply to the sensor, you should replace the dict with None
#######################################################################################################################

SONAR_STRUCTURES = {
    'AD2CP': {
        'v0.5.x': {
            'top': {'ep_group': None, 'coords': [],},
            'environment': {'ep_group': None, 'coords': [],},
            'platform': {'ep_group': None, 'coords': [],},
            'provenance': {'ep_group': None, 'coords': [],},
            'sonar': {'ep_group': None, 'coords': [],},
            'vendor': {'ep_group': None, 'coords': [],},
            'beam': {'ep_group': None, 'coords': [],},
            'beam_power': {'ep_group': None, 'coords': [],},
            'nmea': {'ep_group': None, 'coords': [],},
        },
        'v0.6.x': {
            'top': {'ep_group': None, 'coords': [],},
            'environment': {'ep_group': None, 'coords': [],},
            'platform': {'ep_group': None, 'coords': [],},
            'provenance': {'ep_group': None, 'coords': [],},
            'sonar': {'ep_group': None, 'coords': [],},
            'vendor': {'ep_group': None, 'coords': [],},
            'beam': {'ep_group': None, 'coords': [],},
            'beam_power': {'ep_group': None, 'coords': [],},
            'nmea': {'ep_group': None, 'coords': [],},
        }
    },
    'AZFP': {
        'v0.5.x': {
            'top': {'ep_group': None, 'coords': [],},
            'environment': {'ep_group': None, 'coords': [],},
            'platform': {'ep_group': None, 'coords': [],},
            'provenance': {'ep_group': None, 'coords': [],},
            'sonar': {'ep_group': None, 'coords': [],},
            'vendor': {'ep_group': None, 'coords': [],},
            'beam': {'ep_group': None, 'coords': [],},
            'beam_power': {'ep_group': None, 'coords': [],},
            'nmea': {'ep_group': None, 'coords': [],},
        },
        'v0.6.x': {
            'top': {'ep_group': None, 'coords': [],},
            'environment': {'ep_group': None, 'coords': [],},
            'platform': {'ep_group': None, 'coords': [],},
            'provenance': {'ep_group': None, 'coords': [],},
            'sonar': {'ep_group': None, 'coords': [],},
            'vendor': {'ep_group': None, 'coords': [],},
            'beam': {'ep_group': None, 'coords': [],},
            'beam_power': {'ep_group': None, 'coords': [],},
            'nmea': {'ep_group': None, 'coords': [],},
        }
    },
    'EK60': {
        'v0.5.x': {
            'top': {
                'ep_group': None,
                'coords': [],
            },
            'environment': {
                'ep_group': 'Environment',
                'coords': ['frequency', 'ping_time'],
            },
            'platform': {
                'ep_group': 'Platform',
                'coords': ['location_time', 'ping_time', 'frequency'],
            },
            'provenance': {
                'ep_group': 'Provenance',
                'coords': [],
            },
            'sonar': {
                'ep_group': 'Sonar',
                'coords': [],
            },
            'vendor': {
                'ep_group': 'Vendor',
                'coords': ['frequency', 'pulse_length_bin'],
            },
            'beam': {
                'ep_group': 'Beam',
                'coords': ['frequency', 'ping_time', 'range_bin', None],
            },
            'beam_power': None,
            'nmea': {
                'ep_group': 'Platform/NMEA',
                'coords': ['location_time'],
            },
        },
        'v0.6.x': {
            'top': {
                'ep_group': None,
                'coords': [],
            },
            'environment': {
                'ep_group': 'Environment',
                'coords': ['channel', 'ping_time'],
            },
            'platform': {
                'ep_group': 'Platform',
                'coords': ['location_time', 'ping_time', 'channel'],
            },
            'provenance': {
                'ep_group': 'Provenance',
                'coords': [],
            },
            'sonar': {
                'ep_group': 'Sonar',
                'coords': ['beam_group'],
            },
            'vendor': {
                'ep_group': 'Vendor',
                'coords': ['channel', 'pulse_length_bin'],
            },
            'beam': {
                'ep_group': 'Sonar/Beam_group1',
                'group_descr': "contains backscatter power (uncalibrated) and other beam or" +
                               " channel-specific data, including split-beam angle data when they exist.",
                'coords': ['channel', 'ping_time', 'range_sample', 'beam'],
            },
            'beam_power': None,
            'nmea': {
                'ep_group': 'Platform/NMEA',
                'coords': ['location_time'],
            },
        }
    },
    'EK80': {
        'v0.5.x': {
            'top': {'ep_group': None, 'coords': [],},
            'environment': {'ep_group': None, 'coords': [],},
            'platform': {'ep_group': None, 'coords': [],},
            'provenance': {'ep_group': None, 'coords': [],},
            'sonar': {'ep_group': None, 'coords': [],},
            'vendor': {'ep_group': None, 'coords': [],},
            'beam': {'ep_group': None, 'coords': [],},
            'beam_power': {'ep_group': None, 'coords': [],},
            'nmea': {'ep_group': None, 'coords': [],},
        },
        'v0.6.x': {
            'top': {'ep_group': None, 'coords': [],},
            'environment': {'ep_group': None, 'coords': [],},
            'platform': {'ep_group': None, 'coords': [],},
            'provenance': {'ep_group': None, 'coords': [],},
            'sonar': {'ep_group': None, 'coords': [],},
            'vendor': {'ep_group': None, 'coords': [],},
            'beam': {'ep_group': None, 'coords': [],},
            'beam_power': {'ep_group': None, 'coords': [],},
            'nmea': {'ep_group': None, 'coords': [],},
        }
    },
}
