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
