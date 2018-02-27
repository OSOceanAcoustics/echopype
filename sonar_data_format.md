# A common sonar data NetCDF file format

## Vision
Providing a common file format for storing active sonar data. **Active sonar** here refers to instruments that send out sounds into water and receive returning echoes. These include ADCP, scientific echosounders, and potentially multi-beam sonar in the future - need to find someone who's experienced with that. Scientific echosounders include Simrad EK60, EK80, ASL AZFP.

## Draft format
### dimensions --> all common for ADCP and sonar
- lat
- lon
- time [UTC time]
- depth [m]

### variables
- echo: lat, lon, time, depth  --> variable attribute: **Unit**
- angle_alongship: lat, lon, time, depth
- angle_athwardship: lat, lon, time, depth
- heave: 1 x time
- roll: 1 x time
- pitch: 1 x time

### global attributes --> each file has one set of these
- sensor_manusfacturer
- sensor_model
- deployment_info
- firmware_info
- software_info
- metadata
- first_ping_timestamp  --> ADCP: time coverage start, EK: pulled from filename/time of first ping


### group attributes --> each group/transducer has one set of these

#### Stuff related to user setup for each transducer
- frequency
- bin_size
- pulse_length
- sample_interval
- bandwidth
- channel_id    --> ADCP: beam, ASL/EK: transceiver number

#### Stuff related to environment
- absorption_coeff
- sound_velocity
- temperature             --> ADCP has it, EK: internal or external temperature

#### Transducer properties --> for cal
- transmit_power
- angle_sensitivity_alongship
- angle_sensitivity_athwartship
- angle_offset_alongship
- angle_offset_athwart
- beam_type               --> categorical: split0beam, single-beam
- beam_width_alongship
- beam_width_athwartship
- equiv_beam_angle
- gain
- sa_correction

#### Transducer placement
- transducer_depth
- pos_x, pos_y, pos_z     --> ADCP and EK: relative position of each transducer
- dir_x, dir_y, dir_z     --> EK: relative direction of each transducer
- heading                 --> ADCP: heading

#### Stuff that are not necessary
Those from whole file header
- sounder_name     --> sensor model
- survey_name      --> deployment info
- transducer_count --> not needed
- transect_name    --> X
- version          --> software/firmware

Those from transducer header
- channel       --> X
- gpt_software_version    --> X software info
- gain_table              --> remove, already has gain
- pulse_length_table      --> remove, already has pulse length
- sa_correction_table     --> substitute with sa_correction
