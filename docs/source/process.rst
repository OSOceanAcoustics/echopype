Data processing
===============

.. warning::
   Starting with version 0.5.0, the ``model`` subpackage and the data processing 
   interface ``EchoData`` have been renamed to ``process`` and ``Process``, respectively.
   Attempts to import ``echopype.model`` and use ``EchoData`` will still
   work at the moment but will be deprecated in the future.


Functionality
-------------

- EK60 and AZFP narrowband echosounders:

  - calibration and echo-integration to obtain
    volume backscattering strength (Sv) from power data.
  - Simple noise removal by removing data points (set to ``NaN``) below
    an adaptively estimated noise floor [1]_.
  - Binning and averaging to obtain mean volume backscattering strength (MVBS)
    from the calibrated data.

- EK80 and EA640 broadband echosounders:

  - calibration based on pulse compression output in the
    form of average over frequency.


The steps of performing these analysis for EK60 and AZFP echosounders
are summarized below.
Additional information will be added for broadband EK80 and EA640 echosounders as
additional functionality is developed.

.. code-block:: python

   from echopype import Process
   nc_path = './converted_files/convertedfile.nc'  # path to a converted nc file
   ed = Process(nc_path)    # create a processing object
   ed.calibrate()           # Sv
   ed.remove_noise()        # denoise
   ed.get_MVBS()            # calculate MVBS

By default, these methods do not save the calculation results to disk.
The computation results can be accessed from ``ed.Sv``, ``ed.Sv_clean`` and
``ed.MVBS`` as xarray Datasets with proper dimension labels.

To save results to disk:

.. code-block:: python

   ed.calibrate(save=True)     # output: convertedfile_Sv.nc
   ed.remove_noise(save=True)  # output: convertedfile_Sv_clean.nc
   ed.get_MVBS(save=True)      # output: convertedfile_MVBS.nc


There are various options to save the results:

.. code-block:: python

   # Overwrite the output postfix from _Sv to_Cal: convertedfile_Cal.nc
   ed.calibrate(save=True, save_postfix='_Cal')

   # Save output to another directory: ./cal_results/convertedfile_Sv.nc
   ed.calibrate(save=True, save_path='./cal_results')

   # Save output to another directory with an arbitrary name
   ed.calibrate(save=True, save_path='./cal_results/somethingnew.nc')

By default, for noise removal and MVBS calculation, echopype tries to load Sv
already stored in memory (``ed.Sv``), or tries to calibrate the raw data to
obtain Sv. If ``ed.Sv`` is empty (i.e., whe calibration operation has not been
performed on the object), echopype will try to load Sv from ``*_Sv.nc`` from
the directory containing the converted ``.nc`` file or from the user-specified
path. For example:

1. Try to do MVBS calculation without having previously calibrated data

   .. code-block:: python

      from echopype import Process
      nc_path = './converted_files/convertedfile.nc'  # path to a converted nc file
      ed = Process(nc_path)   # create a processing object
      ed.get_MVBS()  # echopype will call .calibrate() automatically

2. Try to do MVBS calculation with _Sv_clean.nc file previously created in
   folder 'another_directory'

   .. code-block:: python

      from echopype import Process
      nc_path = './converted_files/convertedfile.nc'  # path to a converted nc file
      ed = Process(nc_path)   # create a data processing object
      ed.get_MVBS(source_path='another_directory', source_postfix='_Sv_clean')


.. note:: Echopype's data processing functionality is being developed actively.
   Be sure to check back here often!


Environmental parameters
------------------------

Environmental parameters, including temperature, salinity and pressure, are
critical in biological interpretation of ocean sonar data. They influence

- Transducer calibration, through seawater absorption. This influence is
  frequency-dependent, and the higher the frequency the more sensitive the
  calibration is to the environmental parameters.

- Sound speed, which impacts the conversion from temporal resolution of
  (of each data sample) to spatial resolution, i.e. the sonar observation
  range would change.

By default, echopype uses the following for calibration:

- EK60: Environmental parameters saved with the data files

- AZFP: salinity = 29.6 PSU, pressure = 60 dbar,
  and temperature recorded at the instrument

These parameters should be overwritten when they differ from the actual
environmental condition during data collection.
To update these parameters, simply do the following *before*
calling ``ed.calibrate()``:

.. code-block:: python

   ed.temperature = 8   # temperature in degree Celsius
   ed.salinity = 30     # salinity in PSU
   ed.pressure = 50     # pressure in dbar
   ed.recalculate_environment()  # recalculate related parameters

This will trigger recalculation of all related parameters,
including sound speed, seawater absorption, thickness of each sonar
sample, and range. The updated values can be retrieved with:

.. code-block:: python

   ed.seawater_absorption  # absorption in [dB/m]
   ed.sound_speed          # sound speed in [m/s]
   ed.sample_thickness     # sample spatial resolution in [m]
   ed.range                # range for each sonar sample in [m]

For EK60 data, echopype updates the sound speed and seawater absorption
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
Currently echopypy allows users to overwrite calibration parameters for
EK60 data, including ``sa_correction``, ``equivalent_beam_angle``,
and ``gain_correction``.

As an example, to reset the equivalent beam angle for 18 kHz only,
one can do:

.. code-block:: python

   ed.equivalent_beam_angle.loc[dict(frequency=18000)] = -18.02  # set value for 18 kHz only

To set the equivalent beam angle for all channels at once, do:

.. code-block:: python

   ed.equivalent_beam_angle = [-17.47, -20.77, -21.13, -20.4 , -30]  # set all channels at once

Make sure you use ``ed.equivalent_beam_angle.frequency`` to check
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
