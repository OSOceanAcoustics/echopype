Using echopype
==============


Installation
------------

echopype is pip installable. You can install echopype by doing the following
in your terminal:

.. code-block:: console

    $ pip install echopype



File conversion
---------------
EK60
~~~~

echopype supports batch conversion of ``.raw`` files to netCDF ``.nc``
format in the terminal:

.. code-block:: console

    $ echopype_converter -s ek60 data/*.raw

This will generate corresponding ``.nc`` files with the same leading
filename as the original ``.raw`` files in the same directory.
See :ref:`data-format` for details about the converted file format.

To use the EK60 data converter in an interactive Python session,
you can do:

.. code-block:: python

    # import as part of a submodule
    from echopype.convert import ConvertEK60
    data_tmp = ConvertEK60('FILENAME.raw')
    data_tmp.raw2nc()

Or:

.. code-block:: python

    # import the full module
    import echopype as ep
    data_tmp = ep.convert.ConvertEK60('FILENAME.raw')
    data_tmp.raw2nc()

The same as in the command line case, this will generate a ``FILENAME.nc``
in the same directory as ``FILENAME.raw``.

The ``ConvertEK60`` instance contains all the data unpacked from the
.raw file, so it is a good idea to clear it from memory once done with
conversion.

AZFP
~~~~
AZFP conversion requires an ``.XML`` file along with the raw ``.01A`` file to convert into an ``.nc`` file. To do the conversion, simply use the ``convert.Convert`` method in the in an interactive session as follows:

.. code-block:: python

    import echopype as ep
    data_tmp = ep.convert.Convert('FILENAME.01A', 'XMLFILE.xml')

However, before calling ``data_tmp.raw2nc()`` in order to create your netCDF4 file, you should first set ``platform_name``, ``platform_type``, and ``patform_code_ICES`` as these values are not recorded in the raw files but are used in the netCDF4 convention. Not setting these parameters will save them as empty strings, and you may set them thusly:

.. code-block:: python

    data_tmp.platform_name = 'Wilson'
    data_tmp.platform_type = 'subsurface mooring'
    data_tmp.platform_code_ICES = '3164'

Then simply do the following to save  a ``.nc`` file to the same directory as the ``.01A`` file.

.. code-block:: python

    data_tmp.raw2nc()

Data analysis
-------------

The data analysis functionalities of echopype is being developed actively.
Be sure to check back here often!

echopype currently supports:

- calibration and echo-integration to obtain volume backscattering strength (Sv)
  from the power data collected by EK60 and AZFP.

- simple noise removal by suppressing data points below an adaptively estimated
  noise floor [1]_.

- binning and averaging to obtain mean volume backscattering strength (MVBS)
  from the calibrated data.

The steps of performing these analysis for each echosounder are summarized below:

EK60
~~~~

.. code-block:: python

    from echopype.model import EchoData
    data = EchoData('FILENAME.nc')
    data.calibrate()  # Calibration and echo-integration
    data.remove_noise(save=True)  # Save denoised Sv to FILENAME_Sv_clean.nc
    data.get_MVBS(save=True)  # Save MVBS to FILENAME_MVBS.nc

Note that by default, method ``calibrate`` save the calibrated volume
backscattering strength (Sv) to ``FILENAME_Sv.nc``, while method ``remove_noise``
and ``get_MVBS`` by default do not generate new files. The computation results
from these two methods can be accessed from ``data.Sv_clean`` and ``data.MVBS``
as xarray DataSets. The outputs of these methods are are xarray DataSets with
proper dimension labels.

AZFP
~~~~
You can initialize the functions for AZFP data analysis in exactly the same way
as with EK60.

.. code-block:: python

    from echopype.model import EchoData
    data = EchoData('FILENAME.nc')


Before calibration, the salinity and pressure of the water should be adjusted
if the default values of 29.6 PSU, and 60 dbars do not apply to the environment
where data collection took place. For example:

.. code-block:: python

   data.salinity = 30     # Salinity in PSU
   data.pressure = 50     # Pressure in dbars (~ equal to depth in meters)


These values are used in calculating the sea absorption coefficients for each
frequency as well as the sound speed in the water.
The sound speed is used to calculate the range.
These values can be retrieved with:

.. code-block:: python

    data.sea_abs
    data.sound_speed
    data.range

Get Sv, Target Strength (TS), and MVBS by calling

.. code-block:: python

    data.calibrate()
    data.calibrateTS()
    data.get_MVBS(save=True)


---------------

.. [1] De Robertis and Higginbottoms (2007) A post-processing technique to
   estimate the signal-to-noise ratio and remove echosounder background noise.
   `ICES J. Mar. Sci. 64(6): 1282–1291. <https://academic.oup.com/icesjms/article/64/6/1282/616894>`_

.. TODO: Need to specify the changes we made from AZFP Matlab code to here:
   In the Matlab code, users set temperature/salinity parameters in
   AZFP_parameters.m and run that script first before doing unpacking.
   Here we require users to unpack raw data first into netCDF, and then
   set temperature/salinity in the model module if they want to perform
   calibration. This is cleaner and less error prone, because the param
   setting step is separated from the raw data unpacking, so user-defined
   params are not in the unpacked files.
