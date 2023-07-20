Data processing
===============


Functionality
-------------

- EK60 and AZFP narrowband echosounders:

  - Calibration and echo-integration to obtain
    volume backscattering strength (Sv) from power data.
  - Simple noise removal by removing data points (set to ``NaN``) below
    an adaptively estimated noise floor [1]_.
  - Binning and averaging to obtain mean volume backscattering strength (MVBS)
    from the calibrated data.
  - Compute mean volume backscattering strength (MVBS) based
    on either the number of pings and sample intervals
    (the ``range_sample`` dimension in the dataset) or a
    specified ping time interval and range interval in
    physics units (seconds and meters, respectively).

- EK80 and EA640 broadband echosounders:

  - Calibration based on pulse compression output in the
    form of average over frequency.
  - The same noise removal and MVBS computation functionality available
    to the narrowband echosounders.


The steps for performing these analyses are summarized below:

- Calibration:

   .. code-block:: python

      import echopype as ep
      nc_path = './converted_files/file.nc'     # path to a converted nc file
      echodata = ep.open_converted(nc_path)     # create an EchoData object
      ds_Sv = ep.calibrate.compute_Sv(echodata)    # obtain a dataset containing Sv, echo_range, and
                                                   # the calibration and environmental parameters

- Reduce data by computing MVBS:

   .. code-block:: python

      # Reduce data based on physical units
      ds_MVBS = ep.preprocess.compute_MVBS(
           ds_Sv,               # calibrated Sv dataset
           range_meter_bin=20,  # bin size to average along echo_range in meters
           ping_time_bin='20S'  # bin size to average along ping_time in seconds
       )

      # Reduce data based on sample number
      ds_MVBS = ep.preprocess.compute_MVBS_index_binning(
           ds_Sv,             # calibrated Sv dataset
           range_sample_num=30,  # number of sample bins to average along the range_sample dimensionm
           ping_num=5         # number of pings to average
       )

- Noise removal:

   .. code-block:: python

      # Remove noise
      ds_Sv_clean = ep.preprocess.remove_noise(    # obtain a denoised Sv dataset
         ds_Sv,             # calibrated Sv dataset
         range_sample_num=30,  # number of samples along the range_sample dimension for estimating noise
         ping_num=5,        # number of pings for estimating noise
      )

The functions in the ``calibrate`` subpackage take in an ``EchoData`` object,
which is essentially a container for multiple xarray ``Dataset`` instances,
and return a single xarray ``Dataset`` containing the calibrated backscatter
quantities and the samples' corresponding range in meters.
The input and output of all functions in the ``preprocess``
subpackage are xarray ``Dataset`` instances, with the input being a ``Dataset``
containing ``Sv`` and ``echo_range`` generated from calibration.

The ``calibrate`` and ``preprocess`` functions do not save the calculation results to disk,
but the returned xarray ``Dataset`` can be saved using native xarray methods
such as ``to_netcdf`` and ``to_zarr``.

For example, to save the Sv and MVBS results to disk:

.. code-block:: python

   ds_Sv.to_netcdf('file_Sv.nc')
   ds_MVBS.to_netcdf('file_MVBS.nc')


.. note:: Echopype's data processing functionality is being developed actively.
   Be sure to check back here often!


Environmental parameters
------------------------

Environmental parameters, including temperature, salinity and pressure, are
critical in biological interpretation of ocean sonar data. They influence:

- Transducer calibration, through seawater absorption. This influence is
  frequency-dependent, and the higher the frequency the more sensitive the
  calibration is to the environmental parameters.

- Sound speed, which impacts the conversion from temporal resolution
  (of each data sample) to spatial resolution, i.e. the sonar observation
  range changes with sound speed.

By default, echopype uses the following for calibration:

- EK60 and EK80: Environmental parameters saved with the raw data files.
  For EK60, instrument operators may enter temperature and salinity values into the
  `Simrad EK60 software's Environment dialog
  <https://www.simrad.online/ek60/ref_english/default.htm?startat=/ek60/ref_english/xxx_para_environment.html>`_
  and the Simrad software will calculate sound speed and sound absorption;
  alternatively, sound speed may be entered directly.
  Only sound speed and sound absorption are saved into the raw file.

- AZFP: Salinity and pressure provided by the user,
  and temperature recorded at the instrument.

Seawater sound absorption and sound speed may be recalculated with echopype if
more accurate in-situ environmental parameters are available.
To update these values, pass the environmental parameters
as a dictionary while calling ``ep.calibrate.compute_Sv()``:

.. code-block:: python

   env_params = {
       'temperature': 8,   # temperature in degree Celsius
       'salinity': 30,     # salinity in PSU
       'pressure': 50,     # pressure in dbar
   }
   ds_Sv = ep.calibrate.compute_Sv(echodata, env_params=env_params)

These values will be used in calculating sound speed,
sound absorption, and the thickness of each sonar sample,
which is used in calculating the range (``echo_range``).
The updated values can be retrieved with:

.. code-block:: python

   ds_Sv['sound_absorption']   # absorption in [dB/m]
   ds_Sv['sound_speed']        # sound speed in [m/s]
   ds_Sv['echo_range']              # echo_range for each sonar sample in [m]


For EK60 and EK80 data, echopype updates
the sound speed using the formula from Mackenzie (1981) [2]_  and
seawater absorption using the formula from Ainslie and McColm (1981) [3]_.

For AZFP data, echopype updates the sound speed and seawater absorption
using the formulae provided by the manufacturer ASL Environmental Sciences.


Calibration parameters
----------------------

*Calibration* here refers to the calibration of transducers on an
echosounder, which finds the mapping between the voltage signal
recorded by the echosounder and the actual (physical) acoustic pressure
received at the transducer. This mapping is critical in deriving biological
quantities from acoustic measurements, such as estimating biomass.
More detail about the calibration procedure can be found in [4]_.

Echopype by default uses calibration parameters stored in the converted
files along with the backscatter measurements and other metadata parsed
from the raw data file.
However, since careful calibration is often done separately from the
data collection phase of the field work, accurate calibration parameters
are often supplied in the post-processing stage.
Currently echopype allows users to overwrite the following calibration parameters:

- EK60 and EK80: ``sa_correction``, ``gain_correction``, and ``equivalent_beam_angle``

- AZFP: ``EL``, ``DS``, ``TVR``, ``VTX``, ``Sv_offset``, and ``equivalent_beam_angle``


As an example, to reset the equivalent beam angle for all frequencies,
specify ``cal_params`` while calling the calibration functions:

.. code-block:: python

   import xarray as xr
   equivalent_beam_angle = xr.DataArray(     # set all channels at once
       [-17.47, -20.77, -21.13, -20.4, -30],
       dims=['frequency'],
       coords=[echodata.beam.frequency]
   )
   cal_params = {
       'equivalent_beam_angle': equivalent_beam_angle
   }
   ds_Sv = ep.calibrate.compute_Sv(echodata, cal_params=cal_params)

To reset the equivalent beam angle for 18 kHz only, one can do:

.. code-block:: python

   # set value for 18 kHz only
   echodata.beam.equivalent_beam_angle.loc[dict(frequency=18000)] = 18.02


References
----------

.. [1] De Robertis A, Higginbottoms I. (2007) A post-processing technique to
   estimate the signal-to-noise ratio and remove echosounder background noise.
   `ICES J. Mar. Sci. 64(6): 1282–1291. <https://academic.oup.com/icesjms/article/64/6/1282/616894>`_

.. [2] Mackenzie K. (1981) Nine‐term equation for sound speed in the oceans.
   `J. Acoust. Soc. Am. 70(3): 806-812 <https://asa.scitation.org/doi/10.1121/1.386920>`_

.. [3] Ainslie MA, McColm JG. (1998) A simplified formula for viscous and
   chemical absorption in sea water.
   `J. Acoust. Soc. Am. 103(3): 1671-1672 <https://asa.scitation.org/doi/10.1121/1.421258>`_

.. [4] Demer DA, Berger L, Bernasconi M, Bethke E, Boswell K, Chu D, Domokos R,
   et al. (2015) Calibration of acoustic instruments. `ICES Cooperative Research Report No.
   1.         133 pp. <https://doi.org/10.17895/ices.pub.5494>`_


.. TODO: Need to specify the changes we made from AZFP Matlab code to here:
   In the Matlab code, users set temperature/salinity parameters in
   AZFP_parameters.m and run that script first before doing unpacking.
   Here we require users to unpack raw data first into netCDF, and then
   set temperature/salinity in the process subpackage if they want to perform
   calibration. This is cleaner and less error prone, because the param
   setting step is separated from the raw data unpacking, so user-defined
   params are not in the unpacked files.
