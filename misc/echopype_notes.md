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


****************************************************
## 2018/03/03-04
- In the middle of changing everything to conform with the netCDF file format discussed with Robert L
- Now unpacking heave/pitch/roll for each ping and save them in different channels separately, in case the heave/pitch/roll data are only fed in to one channel??? --> check sample data from Robert L --> all heave/pitch/roll data are 0.0
- Cleaned up `first_ping_metadata` and get rid of the need of BaseEnum class from mi-instruments
- `load_ek60_raw` now returns `angle_data_dict` and `motion_data_dict`
  - Alongship and athwarthship are stored in `angle_data_dict[channel#]['along']` and `angle_data_dict[channel#]['athwart']`
  - Heave, pitch, and roll are stored in `motion_data_dict[channel#]['heave']`, `motion_data_dict[channel#]['pitch']`, and `motion_data_dict[channel#]['roll']`


****************************************************
## 2018/04/21-22
- First try implement SONAR-netCDF4 format --> goal is to make `raw2nc` functions.
- Have a first set of questions sent to Gavin.


****************************************************
## 2018/04/24
- Add additional variables to the SONAR-netCDF4 convention:
  - `/Sonar/Beam_group1`
    - `beamwidth_alongship`, `beamwidth_athwartship`
    - `gain_correction` is used to store `gain` unpaked from EK60 files
    - `angle_sensitivity_alongship`, `angle_sensitivity_athwartship`
    - `angle_offset_alongship`, `angle_offset_athwartship`
- Move `ping_time` to top-level group and directly use it for those variables that need this dimension under `Sonar/Beam_group1`.


****************************************************
## 2018/08/21
- Transfer ownership to oceanhackweek organization for group project.


****************************************************
## 2018/10/23
Reading the code `unpack_ek60.py` again:
- `config_header` is a `dict` with the following field:
  ```python
  In[27]: config_header
  Out[27]:
  {'sounder_name': b'ER60',
   'survey_name': b'DY1801_EK60',
   'transducer_count': 5,
   'transect_name': b'',
   'version': b'2.4.3'}
  ```
- `config_transducer` is a list of `dict`. Each `dict` looks like:
  ```python
  In[29]: config_transducer[0]
  Out[29]:
  {'angle_offset_alongship': 0.10000000149011612,
   'angle_offset_athwartship': 0.10000000149011612,
   'angle_sensitivity_alongship': 15.289999961853027,
   'angle_sensitivity_athwartship': 16.06999969482422,
   'beam_type': 1,
   'beam_width_alongship': 9.8100004196167,
   'beam_width_athwartship': 9.449999809265137,
   'channel_id': b'GPT  18 kHz 009072034d45 1-1 ES18-11',
   'dir_x': 0.0,
   'dir_y': 0.0,
   'dir_z': 0.0,
   'equiv_beam_angle': -17.469999313354492,
   'frequency': 18000.0,
   'gain': 22.889999389648438,
   'gain_table': array([ 21.82999992,  22.88999939,  22.89999962,  23.        ,  23.        ]),
   'gpt_software_version': b'070413',
   'pos_x': 0.0,
   'pos_y': 0.0,
   'pos_z': 0.0,
   'pulse_length_table': array([ 0.000512,  0.001024,  0.002048,  0.004096,  0.008192]),
   'sa_correction_table': array([-0.44999999, -0.50999999,  0.        ,  0.        ,  0.        ])}
  ```
- `append_metadata` only append the following parameters in `sample_data`:
  ```python
  metadata['channel'].append(channel)
  metadata['transducer_depth'].append(sample_data['transducer_depth'][0])          # [meters]
  metadata['frequency'].append(sample_data['frequency'][0])                        # [Hz]
  metadata['transmit_power'].append(sample_data['transmit_power'][0])              # [Watts]
  metadata['pulse_length'].append(sample_data['pulse_length'][0])                  # [seconds]
  metadata['bandwidth'].append(sample_data['bandwidth'][0])                        # [Hz]
  metadata['sample_interval'].append(sample_data['sample_interval'][0])            # [seconds]
  metadata['sound_velocity'].append(sample_data['sound_velocity'][0])              # [m/s]
  metadata['absorption_coeff'].append(sample_data['absorption_coefficient'][0])    # [dB/m]
  metadata['temperature'].append(sample_data['temperature'][0])                    # [degC]
  metadata['depth_bin_size'].append(sample_data['sound_velocity'][0] *
                                    sample_data['sample_interval'][0] / 2)         # [meters]
  ```
Code revision:
  - `motion_data` is actually contained in `sample_data` so don't need an extra variable --> done
  - revised ping-by-ping saving to `angle_data_dict` and `motion_data_dict` as well as their conversion to np.array
TODO:
- add more parameters to `append_metadata` `sample_data` so that all metadata can be saved to nc file
Cautions and questions:
- many variables in `sample_data` are fixed for OOI data but may vary for shipboard data? in SONAR-netCDF4 convention they all have dimension `ping_time`
  - `sample_interval`
  - `bandwidth` (`transmit_bandwidth` in convention)
  - `pulse_length` (`transmit_duration_nominal` in convention)
  - `transmit_power`
  - `frequency`: this is fixed for EK60 but may change for EK80? (**check with Chu/Robert**)
- current not dealing with:
  - `trawl_upper_depth_valid`
  - `trawl_opening_valid`
  - `trawl_upper_depth`
  - `trawl_opening`
- **not sure what `sample_data['offset']` is**

****************************************************
## Next steps
- change `raw2hdf5` functions to `raw2netcdf` functions
- Save temperature??
- Need to revive error catch code
- write a method in the model class to convert electronic angle to mechanical angle



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
