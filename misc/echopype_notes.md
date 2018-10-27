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
## 2018/10/23 to 2018/10/26
Reading the code `unpack_ek60.py` again, below are from the original version:
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
  - `first_ping_metadata` now contains additional variables that were not unpacked from before
  - `motion` contains pitch, roll, heave and unpacks what was stored in `sample_data`
  - many variables in `sample_data` are fixed for OOI data but may vary for shipboard data? in SONAR-netCDF4 convention they all have dimension `ping_time`. Now storing the following to `tr_data_dict`:
    - `bandwidth` (`transmit_bandwidth` in convention)
    - `pulse_length` (`transmit_duration_nominal` in convention)
    - `transmit_power`
    - `frequency`: this is fixed for EK60 but may change for EK80? (**check with Chu/Robert**)
    - `sample_interval`
    When saved to nc file these variables have dimension `(frequency, ping_time)`

Under Sonar group:
  - haven't implemented `beam_stabilisation_t`, `samplet_t` etc. under Sonar group

Under Beam group:
  - adding:
    - dimension `range_bin` which is a bin count from 0
    - dimension `frequency` for multi-freq echosounder files with attributes `units` and `valid_min`
    - `channel_id`
    - `sa_correction`
    - `gpt_software_version`
  - beamwidth-related variables:
    - `beamwidth_receive_major`, `beamwidth_receive_minor`, `beamwidth_transmit_major`, `beamwidth_transmit_minor` are stored as scalar
    - `*_major`-->`alongship` and `*_minor`-->`athwartship`
    - `beamwidth_receive_major` = `beamwidth_transmit_major`
    - `beamwidth_receive_minor` = `beamwidth_transmit_minor`
  - haven't implemented the following:
    - dimension/coordinate `beam`
    - `beam_stabilisation`
    - `beam_type`
    - `receiver_sensitivity` (**not sure what this is**)
    - `time_varied_gain`
    - `transmit_duration_equivalent`
    - `transmit_frequency_start`
    - `transmit_frequency_stop`
    - `transmit_source_level`
    - `transmit_type`
    - `transducer_gain` --> use only `gain_correction` to store values from `config_transducer['gain']`
    - `time_varied_gain` --> do not save this since this is part of the unpacked data, will calculate when data is used, **need to review this!**
  - backscatter measurements:
    - `backscatter_i` doesn't exist for EK60
    - use `backscatter_r` to store `power_data` unpacked from EK60
  - `non_quantitative_processing` is set to 0
  - `sample_data['offset']` is 0 --> set `beam_dict['sample_time_offset']` = 2 for EK60 data
  - `transducer_gain` is used to stored `sample_data['gain']`
  - define `conversion_equation_t` = 'type_3' for EK60 calibration

Under Platform group:
  - haven't implemented the following:
    - `distance`
    - `heading`
    - `latitude`
    - `longitude`
    - `MRU_offset_x/y/z`
    - `MRU_rotation_x/y/z`
    - `position_offset_x/y/z`
    - `speed_ground`
    - `speed_relative`
    - `vertical_offset`
    - `water_level`
  - adding:
    - dimension `frequency` for multi-freq echosounder files with attributes `units` and `valid_min`
  - `transducer_offset_z` is a combination of `config_transducer['pos_z'] + first_ping_metadata['trans']`

Cautions and questions:
- current not storing the following in `sample_data`:
  - `trawl_upper_depth_valid`
  - `trawl_opening_valid`
  - `trawl_upper_depth`
  - `trawl_opening`
- **not sure what `sample_data['offset']` is** --> currently not stored in nc file, `beam_dict['sample_time_offset']` set to 2 based on TVG calculation from example in Echolab
- **CHECK** `transducer_depth` is currently unpacked into `first_ping_metadata` and is only saved once for each channel, need to figure out if it's changing ping-by-ping
- consider moving `channel_id`, `sa_correction`, `gpt_software_version` to the Vendor-specific group, currently they are in the Beam group
- variables that need to think about where it should be:
  - `config_header['sounder_name']` --> current not stored under Sonar group as `sonar_model` (e.g., ER60)
  - `config_header['survey_name']` --> current stored under Platform group as `platform_name` (e.g., DY1801_EK60)
  - `config_transducer['channel_id']` --> currently stored under Beam group as `channel_id`
  - `config_transducer['sa_correction_table']` --> currently stored under Beam group as `sa_correction` after sorting out the match using `pulse_length`
  - `config_transducer['gpt_software_version']` --> currently stored under Beam group as `gpt_software_version`
- [Solved] figure out how to code the time properly in nc files, the SONAR-netCDF4 convention uses "nanoseconds since .." doesn't seem to be allowed under CF convention
  - To take advantage of the serialization supported by xarray, the `ping_time` units is changed to 'seconds since 1900-01-01'. The recommended 'nanoseconds since 1601-01-01' results in errors when using `xr.opendataset` unless a flag `decode_time=False` is used.
  - use UTC time in the Provenance group for file conversion time `dt.now(tz=pytz.utc).isoformat(timespec='seconds')`, note the use of `pytz.utc` here

TODO:
- change all trailing 'Z' in the saved times to actual timezone
- move `transducer_offset_x/y/z` from Platform to Beam group
- consider combining beam group with sonar group
- revive error catching code
- revise data manipulation (TVG compensation, noise removal, db-differencing, etc) classes using nc file as input
- write a method in the model class to convert electronic angle to mechanical angle
