## Questions/comments for @imranmaj:
- why check the version again?
    - https://github.com/OSOceanAcoustics/echopype/blob/16a574dda792a61f6f7ae0583b70b4b87d787f12/echopype/convert/parse_ad2cp.py#L589-L593
    - https://github.com/OSOceanAcoustics/echopype/blob/16a574dda792a61f6f7ae0583b70b4b87d787f12/echopype/convert/parse_ad2cp.py#L630-L634
- Move `AHRS_COORDS` def to `parse_ad2cp`?
    - https://github.com/OSOceanAcoustics/echopype/blob/16a574dda792a61f6f7ae0583b70b4b87d787f12/echopype/convert/set_groups_ad2cp.py#L10-L14
- the `beam` coordinate is missing in `rawtest.090.00015.nc`?



## Overarching action items:
- Double check all raw to actual units conversion
- Move the sequence of variable handling to following the sequence of the data specification, for easier reference
- For all variables, put
    - `long_name`: Field in spec sheet
    - `comment`: Description in spec sheet
    - `units`: following spec sheet and the units convention in [CF convention](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html)
    - for those that are already converted to the actual units, put "but already converted to actual units here."
    - for example: for the field `Temperature`, put
        ```
        {
            "long_name": "Field "Temperature" in the data specification. "
            "comment": (
                "Reading from the temperature sensor. "
                "Raw data given as 0.01 °C "
                "but already converted to actual units here."
            )
            "units": "degree_C"
        }
        ```



# The original `Beam` group

## Which mode goes to which group
- following SONAR-netCDF4 and our implementation of EK80 groups, we will use the following sequence when determining where data from each mode goes in `Beam_groupX`:
    1. average:
        ```
        "descr": (
            "contains echo intensity, velocity and correlation data "
            "as well as other configuration parameters from the Average mode."
        )
        ```
    2. burst
        ```
        "descr": (
            "contains echo intensity, velocity and correlation data "
            "as well as other configuration parameters from the Burst mode."
        )
        ```
    3. echosounder
        ```
        "descr": (
            "contains backscatter echo intensity and other configuration parameters from the Echosounder mode. "
            "Data can be pulse compressed or raw intensity."
        )
        ```
    4. echosounder raw samples
        ```
        "descr": (
            "contains complex backscatter raw samples and other configuration parameters from the Echosounder mode, "
            "including complex data from the transmit pulse."
        )
        ```
    - we need a trickier implementation than what's used for EK80, since for EK80 the max number group is 2, and depending on if complex or power data exist, where they show up may be different:
        - if complex data exists, it is always stored in `Beam_group1`
        - if ONLY power data exists, it is stored in `Beam_group1`
        - if BOTH complex and power data exist, complex data in `Beam_group1` and power data in `Beam_group2`
- examples:
    - example 1: file with average and burst mode:
        - `Beam_group1`: average
        - `Beam_group2`: burst
    - example 2: files with burst and echosounder mode:
        - `Beam_group1`: burst
        - `Beam_group2`: echosounder
    - example 3: files with echosounder mode and raw echosounder data:
        - `Beam_group1`: echosounder
        - `Beam_group2`: echosounder raw samples
- for echosounder raw samples
    - the first ping is the raw transmit signal
    - the rest of pings are the receive raw echoes

## Variables common to all modes
Coordinates:
- `ping_time`
    - use the original `ping_time_average`, `ping_time_burst` and `ping_time_echosounder` in each of their own `Beam_groupX` but with the same name `ping_time`
- `range_sample`
    - use the original `range_sample_average`, `range_sample_burst` and `range_sample_echosounder` in each of their own `Beam_groupX` but with the same name `range_sample`
- `beam`:
    - the actual physical beam activated in the setting (1, 2, 3, 4, 5): @imranmaj please double check this
    - parsed from the data variable `data_set_description` using `parse_ad2cp._postprocess_beams` for the following packets:
      - `BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT`
      - `BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT`
      - `BOTTOM_TRACK_DATA_RECORD_FORMAT`

Data variables:
- `number_of_beams`
- `coordinate_system`
- `number_of_cells`
- `blanking`
- `cell_size`
    - this is conceptually equivalent to `sample_interval` for the other echosounders, just that `sample_interval` is defined in time (second) and `cell_size` is defined in space (meter)
- `velocity_range`
- `echosounder_frequency`
    - the parsed values seem wrong: right now it shows as either 0 (these are probably for pings from other modes and not echosounder -- please verify) or 10000, but should be 1000000 (1 MHz)  @imranmaj
- `ambiguity_velocity`
- `data_set_description`
- `transmit_energy`
- `velocity_scaling`
- `velocity`
    - separate `velocity_burst` and `velocity_average` into different `Beam_groupX` and use the name `velocity`
- `amplitude`
    - separate `amplitude_burst`, `amplitude_average` and `amplitude_echosounder` into different `Beam_groupX` and use the name `backscatter_r` (since this "amplitude" is equivalent to what we have from EK60, AZFP and EK80 power data)
- `correlation`
    - separate `correlation_burst`, `correlation_average` and `correlation_echosounder` into different `Beam_groupX` and use the name `correlation`

Attributes:
- `pulse_compressed`
    - this should be a variable in the `Beam_groupX` that stores echosounder mode data
    - can have a max length of 3 since there can be 3 echograms in the echosounder mode (aligned with `beam`)
    - @imranmaj: now sure what the current status of this is?


## Move from `Beam` to `Vendor_specific` group
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

## Keep in `Vendor_specific` group
- Questions:
  - flag for whether these data will be collected?
  - when collected, what their expected dimensions are?
- `ahrs_rotation_matrix_mij`
- `ahrs_quaternions_wxyz`
- `ahrs_gyro_xyz`
- `std_dev_pitch`
- `std_dev_roll`
- `std_dev_heading`
- `std_dev_pressure`
- `compass_sensor_valid`
- `tilt_sensor_valid`

### Data collected for every ping in all modes
- Coordinate `time1`:
  - this will be the combined `ping_time` from all modes
- `temperature_of_pressure_sensor`: rename to `temperature_pressure_sensor`
- `magnetometer_temperature`: rename to `temperature_magnetometer`
- `real_ping_time_clock_temperature`: rename to `temperature_real_ping_time_clock`
- `pressure_sensor_valid`
- `temperature_sensor_valid`
- `data_record_version`
- `error`
- `status`: need additional parsing @imranmaj
- `status0`: need additional parsing @imranmaj
- `power_level`
- `nominal_correlation`
- `percentage_good_data`
- `battery_voltage`
- `ensemble_counter`

## Move from `Vendor_specific` to `Beam_groupX` group
Coordinates:
- `ping_time` for that group (not the combined one)
- `sample`: rename to `range_sample`; this is sample number along range for raw echosounder data
- `sample_transmit`: rename to `transmit_sample` (following the proposed new variable in PR#714)

Data variables:
- `echosounder_raw_samples_i`: rename to `backscatter_r`
- `echosounder_raw_samples_q`: rename to `backscatter_i`
- `echosounder_raw_transmit_samples_i`: rename to `transmit_pulse_r` (following the proposed new variable in PR#714)
- `echosounder_raw_transmit_samples_q`: rename to `transmit_pulse_i` (following the proposed new variable in PR#714)
- `echosounder_raw_beam`: not sure what this corresponds to, but couldn't find a variable that correpsonds to `beam` which is the physical beam used in transmission @imranmaj
- `echosounder_raw_echogram`: not sure what this corresponds to @imranmaj







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
- `tilt`: is there such a field defined in the spec sheet? I didn't see any in the example files  @imranmaj

## Move from `Platform` group to `Vendor_specific` group
Coordinates:
- `time1`: this will be the combined ping_time from all modes
- `xyz`: coordinate for `magnetometer_raw`

Data variable:
- `magnetometer_raw`
