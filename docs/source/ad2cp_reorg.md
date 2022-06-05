## Questions/comments:
- why check the version again?
    - https://github.com/OSOceanAcoustics/echopype/blob/16a574dda792a61f6f7ae0583b70b4b87d787f12/echopype/convert/parse_ad2cp.py#L589-L593
    - https://github.com/OSOceanAcoustics/echopype/blob/16a574dda792a61f6f7ae0583b70b4b87d787f12/echopype/convert/parse_ad2cp.py#L630-L634
- Move `AHRS_COORDS` def to `parse_ad2cp`?
    - https://github.com/OSOceanAcoustics/echopype/blob/16a574dda792a61f6f7ae0583b70b4b87d787f12/echopype/convert/set_groups_ad2cp.py#L10-L14
- the `beam` coordinate is missing in `rawtest.090.00015.nc`?
- `echosounder_raw_beam`, `echosounder_raw_echogram`: not sure what these correspond to
- `status`, `status0`: need additional parsing






# The original `Beam` group

## Which mode goes to which group
Proposal/question:
- use specific group name for different mode
- this means that then some groups may not exist all together, i.e., there won't always be `Beam_group1` in the converted data file
- this conflicts with what we have for EK80, where there will always be `Beam_group1` but the content changes depending on what are in the file (power/angle samples only, complex samples only, or both)
- if we go with this approach, we can use the following:
    - `Beam_group1`: average
    - `Beam_group2`: burst
    - `Beam_group3`: echosounder
    - `Beam_group4`: echosounder raw samples
        - the first ping is the raw transmit signal
        - the rest of pings are the receive raw echoes

## Variables common to all modes
Coordinates:
- `ping_time`: use the original `ping_time_average`, `ping_time_burst` and `ping_time_echosounder` in each of their own `Beam_groupX` but with the same name `ping_time`
- `range_sample`: use the original `range_sample_average`, `range_sample_burst` and `range_sample_echosounder` in each of their own `Beam_groupX` but with the same name `range_sample`
- `beam`:
    - the actual physical beam activated in the setting (1, 2, 3, 4, 5)
    - parsed from the data variable `data_set_description` using `parse_ad2cp._postprocess_beams` for the following packets: `BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT`, `BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT`, `BOTTOM_TRACK_DATA_RECORD_FORMAT`
    - this is missing in `rawtest.090.00008.nc`??

Data variables:
- `number_of_beams`
- `coordinate_system`
- `number_of_cells`
- `blanking`
- `cell_size`
- `velocity_range`
- `echosounder_frequency`
- `ambiguity_velocity`
- `data_set_description`
- `transmit_energy`
- `velocity_scaling`
- `velocity`: separate `velocity_burst` and `velocity_average` into different `Beam_groupX` and use the name `velocity`
- `amplitude`: separate `amplitude_burst`, `amplitude_average` and `amplitude_echosounder` into different `Beam_groupX` and use the name `amplitude`
- `correlation`: separate `correlation_burst`, `correlation_average` and `correlation_echosounder` into different `Beam_groupX` and use the name `correlation`

Attributes:
- `pulse_compressed`
    - this probably should be a variable?
    - should go to `Beam_group3` that stores the echosounder mode data
    - can have a max length of 3 since there can be 3 echograms in the echosounder mode (aligned with `beam`): is this supported currently?


## Move from `Beam` to `Platform` group
- `figure_of_merit`: for bottom tracking
- `altimeter_distance`
- `altimeter_quality`
- `altimeter_spare`
- `altimeter_raw_data_num_samples`
- `altimeter_raw_data_sample_distance`
- `altimeter_raw_data_samples`
- `ast_distance`
- `ast_quality`
- `ast_offset_100us`
- `ast_pressure`







# `Vendor_specific` group
## Remove
- `pulse_compressed`: this already is/will be in `Beam_group3` (echosounder mode data)

## Move from `Vendor_specific` to `Platform` group
- `ahrs_rotation_matrix_mij`
- `ahrs_quaternions_wxyz`
- `ahrs_gyro_xyz`
- `std_dev_pitch`
- `std_dev_roll`
- `std_dev_heading`
- `std_dev_pressure`
- `compass_sensor_valid`
- `tilt_sensor_valid`

## Move from `Vendor_specific` to `Environment` group
Coordinates:
- `time1`: this will be the combined ping_time from all modes

Data variables:
- `temperature_of_pressure_sensor`: rename to "temperature_pressure_sensor"
- `magnetometer_temperature`: rename to "temperature_magnetometer"
- `real_ping_time_clock_temperature`: rename to "temperature_real_ping_time_clock"
- `pressure_sensor_valid`
- `temperature_sensor_valid`

## Move from `Vendor_specific` to each of the `Beam_groupX` groups
Coordinates: `ping_time` for that group (not the combined one)

Data variables:
- `data_record_version`
- `error`
- `status`: need additional parsing
- `status0`: need additional parsing
- `power_level`
- `nominal_correlation`
- `percentage_good_data`
- `battery_voltage`
- `ensemble_counter`

## Move from `Vendor_specific` to `Beam_group4` group
Coordinates:
- `ping_time` for that group (not the combined one)
- `sample`: rename to `range_sample`; this is sample number along range for raw echosounder data
- `sample_transmit`: rename to `transmit_sample` (to be consistent with what's used for Simrad RAW4)

Data variables:
- `echosounder_raw_samples_i`
- `echosounder_raw_samples_q`
- `echosounder_raw_transmit_samples_i`
- `echosounder_raw_transmit_samples_q`
- `echosounder_raw_beam`: not sure what this corresponds to, but couldn't find a variable that correpsonds to `beam` which is the physical beam used in transmission
- `echosounder_raw_echogram`: not sure what this corresponds to







# The original `Environment` group
Coordinates:
- `time1`: this will be the combined ping_time from all modes

Data variables:
- `sound_speed_indicative`
- `temperature`
- `pressure`







# The original `Platform` group
Coordinates
- `time1`: this will be the combined ping_time from all modes

Data variables:
- `heading`
- `pitch`
- `roll`

## Move from `Platform` group to `Vendor_specific` group
Coordinates:
- `time1`: this will be the combined ping_time from all modes
- `xyz`: coordinate for `magnetometer_raw`

Data variable:
- `magnetometer_raw`
