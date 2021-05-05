Data processing
===============

.. warning::
   Starting with version 0.5.0, the data processing interface ``Process``
   is deprecated. Attempts to use ``Process`` will still
   work at the moment but will no longer be available in the future.


Functionality
-------------

- EK60 and AZFP narrowband echosounders:

  - Calibration and echo-integration to obtain
    volume backscattering strength (Sv) from power data.
  - Simple noise removal by removing data points (set to ``NaN``) below
    an adaptively estimated noise floor [1]_.
  - Binning and averaging to obtain mean volume backscattering strength (MVBS)
    from the calibrated data.

- EK80 and EA640 broadband echosounders:

  - Calibration based on pulse compression output in the
    form of average over frequency.

  - The same noise removal and MVBS functionality available
    to the narrowband echosounders.


The steps for performing these analyses are summarized below.

.. code-block:: python

   import echopype as ep
   nc_path = './converted_files/file.nc'           # path to a converted nc file
   echodata = ep.open_converted(nc_path)           # create a processing object
   Sv = ep.calibrate.compute_Sv(echodata)          # obtain Sv
   Sv_clean = ep.preprocess.remove_noise(Sv, ...)  # obtain a denoised Sv Dataset
   MVBS = ep.preprocess.get_MVBS(Sv_clean)         # obtain MVBS from denoised Sv

The functions in the ``calibrate`` subpackage take in an ``EchoData`` object,
which is essentially a container for multiple xarray ``Dataset`` groups,
and return a single xarray ``Dataset`` (either Sv or Sp).
The Sv ``Dataset`` can then be passed into the functions provided by the
``preprocess`` subpackage which all return a single processed ``Dataset``.

These ``calibrate`` and ``preprocess`` methods do not save the calculation results to disk,
but the returned xarray Dataset have methods like ``to_netcdf`` and ``to_zarr`` that
can be used to save.

For example, to save the Sv and MVBS results to disk:

.. code-block:: python

   Sv.to_netcdf('file_Sv.nc')
   MVBS.to_netcdf('file_MVBS.nc')


.. note:: Echopype's data processing functionality is being developed actively.
   Be sure to check back here often!


Environmental parameters
------------------------

Environmental parameters, including temperature, salinity and pressure, are
critical in biological interpretation of ocean sonar data. They influence:

- Transducer calibration, through seawater absorption. This influence is
  frequency-dependent, and the higher the frequency the more sensitive the
  calibration is to the environmental parameters.

- Sound speed, which impacts the conversion from temporal resolution of
  (of each data sample) to spatial resolution, i.e. the sonar observation
  range would change.

By default, echopype uses the following for calibration:

- EK60: Environmental parameters saved with the data files

- EK80: Environmental parameters saved with the data files

- AZFP: Salinity and pressure provided by the user,
  and temperature recorded at the instrument

These parameters should be overwritten when they differ from the actual
environmental condition during data collection.
To update these parameters, simply pass in the environmental parameters
as a dictionary while calling ``ep.calibrate.compute_Sv()`` like so:

.. code-block:: python

   environment = {
       'temperature': 8,   # temperature in degree Celsius
       'salinity': 30,      # salinity in PSU
       'pressure': 50,     # pressure in dbar
   }
   Sv = ep.calibrate.compute_Sv(echodata, env_params=environment)

These value will be used in calculating sound speed,
seawater absorption, thickness of each sonar sample, and range.

For EK60 and EK80 data, echopype updates the sound speed and seawater absorption
using the formulae from Mackenzie (1981) [2]_ and
Ainslie and McColm (1981) [3]_, respectively.

For AZFP data, echopype updates the sound speed and seawater absorption
using the formulae provided by the manufacturer ASL Environmental Sci.


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
specify ``cal_params`` while calling the calibration functions like so:

.. code-block:: python

   # set all channels at once
   calibration = {
       'equivalent_beam_angle': [-17.47, -20.77, -21.13, -20.4 , -30]
   }
   Sv = ep.calibrate.compute_Sv(echodata, cal_params=calibration)

To reset the equivalent beam angle for 18 kHz only, one can do:

.. code-block:: python

   echodata.beam.equivalent_beam_angle.loc[dict(frequency=18000)] = 18.02  # set value for 18 kHz only

Make sure you use ``echodata.beam.equivalent_beam_angle.frequency`` to check
the sequence of the frequency channels first, and always double
check after setting these parameters!


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
   326. 133 pp. <https://doi.org/10.17895/ices.pub.5494>`_


.. TODO: Need to specify the changes we made from AZFP Matlab code to here:
   In the Matlab code, users set temperature/salinity parameters in
   AZFP_parameters.m and run that script first before doing unpacking.
   Here we require users to unpack raw data first into netCDF, and then
   set temperature/salinity in the process subpackage if they want to perform
   calibration. This is cleaner and less error prone, because the param
   setting step is separated from the raw data unpacking, so user-defined
   params are not in the unpacked files.
