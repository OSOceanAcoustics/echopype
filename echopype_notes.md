## 2018/02/03-04
- Things to be changed in zplsc_b.py
    - make a new generic filename matcher to get file date and time
    - use default logging module to replace call to `log`
    - need to deal with `InstrumentDataException` part -- learn how to write error-exception stuff
- Start reading `zplsc_b.py` and `zplsc_echogram.py` to see waht can be stripped offset
- function `read_header` gives `config_header` and `config_transducer`
- don't need `ZplscBParticleKey.ECHOGRAM_PATH`, can also take out anything with `relpath`
- need to change `from datetime import datetime` to `from datetime import datetime as dt` and replace all `datetime` to `dt` in the code
- function `process_sample` uses `build_windows_time` and `windows_to_ntp`
- the following block in function `process_sample` should be stored and passed when `sample_data['mode'][0] = 1`
  ```
  # Read the athwartship and alongship angle measurements
  if sample_data['mode'][0] > 1:
      numpy.fromfile(input_file, dtype=angle_dtype, count=count)

  ```
  - can do the following to get angle data out:
    ```
    angle_data = np.fromfile(input_file, dtype=angle_dtype, count=count)
    angle_data_nparray = np.vstack(angle_data)
    ```
    here `angle_data` has the same length as `power_data`
    then the output would look like:
    ```
    array([[(  1,   0)],
       [(  1,   0)],
       [(  2,   0)],
       ...,
       [(104, 107)],
       [(-26, 121)],
       [(111, 127)]], dtype=[('athwart', 'i1'), ('along', 'i1')])
    ```
- in `read_config_header`: in python 3, need to change `xrange` to `range`, and add byte declaration when using `strip` function for byte object (in `values[i] = values[i].strip(b'\x00')`)
- toward end of `parse_echogram_file` probably don't need to append `next_time` to `particle_data`
  ```
  particle_data = first_ping_metadata, next_time
  ```
- Need to check if temperature is a parameter set by user when initiating echosounder recording, or a feed-in parameters measured in real-time when recording data. If the latter, will have to return a separate variable for all temperature samples from `next_sample` in `parse_echogram_file`


## 2018/03/03-04
- In the middle of changing everything to conform with the netCDF file format discussed with Robert L
- Now unpacking heave/pitch/roll for each ping and save them in different channels separately, in case the heave/pitch/roll data are only fed in to one channel??? --> check sample data from Robert L --> all heave/pitch/roll data are 0.0
- Cleaned up `first_ping_metadata` and get rid of the need of BaseEnum class from mi-instruments
- `load_ek60_raw` now returns `angle_data_dict` and `motion_data_dict`
  - Alongship and athwarthship are stored in `angle_data_dict[channel#]['along']` and `angle_data_dict[channel#]['athwart']`
  - Heave, pitch, and roll are stored in `motion_data_dict[channel#]['heave']`, `motion_data_dict[channel#]['pitch']`, and `motion_data_dict[channel#]['roll']`

## Next steps
- change `raw2hdf5` functions to `raw2netcdf` functions
- Save temperature??
- Need to revive error catch code
- Consider converting electronic angle to actual angle directly in the unpacked format?? --> maybe make it a method in data manipulation class?


## Echolab metadata format --> common sonar netCDF format
This keeps tracks of which parameters in the common sonar netCDF format are from where in the original Echolab unpacked structure.

Below 'X' indicates not available in the netCDF file.

### `config_header`
These data are stored in `config_header` and unpacked once per file.

Echolab           | Common netCDF                          | Example/Description
----------------- | -------------------------------------- | -----------------------
sunder_name       | sensor_model                           | ER60
survey_name       | deployment_info                        | DY1801_EK60
transect_name     | X                                      |
version           | firmware_info                          | 2.4.3
transducer_count  | X                                      |

### `config_transducer`
These data are stored in `config_transducer` and unpacked once per file.

Echolab                        | Common netCDF                      | Example/Description
------------------------------ | ---------------------------------- | ---------------------------
channel_id                     | sensor_model                       | GPT  18 kHz 009072034d45 1-1 ES18-11
beam_type                      | beam_type                          | 1
frequency                      | frequency                          | 18000.0
gain                           | gain                               |
equiv_beam_angle               | equiv_beam_angle                   |
beam_width_alongship           | beam_width_alongship               |
beam_width_athwartship         | beam_width_athwartship             |
angle_sensitivity_alongship    | angle_sensitivity_alongship        |
angle_sensitivity_athwartship  | angle_sensitivity_athwartship      |
angle_offset_alongship         | angle_offset_alongship             |
angle_offset_athwart           | angle_offset_athwartship           |
pos_x                          | pos_x                              |
pos_y                          | pos_y                              |
pos_z                          | pos_z                              |
dir_x                          | dir_x                              |
dir_y                          | dir_y                              |
dir_z                          | dir_z                              |
pulse_length_table             | X --> only keep pulse_length       |
gain_table                     | X --> only keep gain               |
gpt_software_version           | software_info                      |
sa_correction_table            | X --> only keep sa_correction      |

### Other metadata from `sample_data`
These data are stored in `sample_data` and unpacked **per ping**.

Echolab                        | Common netCDF                      | Example/Description
------------------------------ | ---------------------------------- | ---------------------------
transducer_depth               | transducer_depth                   |
transmit_power                 | transmit_power                     |
pulse_length                   | pulse_length                       |
bandwidth                      | bandwidth                          |
sample_interval                | sample_interval                    |
sound_velocity                 | sound_velocity                     |
absorption_coeff               | absorption_coeff                   |
temperature                    | temperature                        |
channel_number                 | channel_number                     | numerical sequence of channel
