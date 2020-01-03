Using echopype
==============


Installation
------------

Echopype can be installed from PyPI or through conda:

.. code-block:: console

    # PyPI
    $ pip install echopype

    # conda
    $ conda install -c conda-forge echopype


File conversion
---------------

Echopype currently supports conversion of data files produced by
the Simrad EK60 echosounder (``.raw`` files) and the ASL Environmental Sciences
AZFP echosounder (``.01A`` files).
The conversion can be conducted using a single interface provided by
the ``Convert`` object.
The only difference is that data files from the AZFP echosounder require an
``.XML`` file that contains associated settings for unpacking the binary data.


Interactive python session
~~~~~~~~~~~~~~~~~~~~~~~~~~

For data files from the EK60 echosounder, you can do
the following in an interactive Python session:

.. code-block:: python

    from echopype.convert import Convert
    data_tmp = Convert('FILENAME.raw')
    data_tmp.raw2nc()

This will generate a  ``FILENAME.nc`` file in the same directory as
the original ``FILENAME.raw`` file.

For data files from the AZFP echosounder, the conversion requires an
``.XML`` file along with the ``.01A`` data file.
This can be done by:

.. code-block:: python

    from echopype.convert import Convert
    data_tmp = Convert('FILENAME.01A', 'XMLFILENAME.xml')
    data_tmp.raw2nc()

However, note that before calling ``raw2nc()`` to create netCDF4 files,
you should first set ``platform_name``, ``platform_type``, and
``patform_code_ICES``, as these values are not recorded in the raw data
files but need to be specified according to the netCDF4 convention.
These parameters will be saved as empty strings unless you specify
them following the example below:

.. code-block:: python

    data_tmp.platform_name = 'OOI'
    data_tmp.platform_type = 'subsurface mooring'
    data_tmp.platform_code_ICES = '3164'   # Platform code for Moorings

The ``platform_code_ICES`` attribute can be chosen by referencing
the platform code from the
`ICES SHIPC vocabulary <https://vocab.ices.dk/?ref=315>`_.

The ``Convert`` instance contains all the data unpacked from the
.raw file, so it is a good idea to clear it from memory once done with
conversion.


Command line tools
~~~~~~~~~~~~~~~~~~~~~~~~~~

Echopype also supports batch conversion of binary data files to netCDF
files (``.nc``) in the terminal. As with before, an ``.XML`` file is
needed to convert the data files from AZFP echosounder.

For converting ``.raw`` files from EK60:

.. code-block:: console

   $ echopype_converter -s some_path/*.raw

For converting ``.01A`` files from AZFP:

.. code-block:: console

   $ echopype_converter -s azfp -x some_path/deployment.xml some_path/*.01A

These will generate corresponding ``.nc`` files with the same leading
filename as the original ``.raw`` files in the same directory.
See :ref:`data-format` for details about the converted file format.

.. note::  Currently the ``.nc`` files generated using the command line
   tool will have the fields
   ``platform_name``, ``platform_type``, and ``patform_code_ICES``
   in the `Platform` group all set to empty strings.


Routine data processing
-----------------------

*The data processing functionalities of echopype is being developed actively.
Be sure to check back here often!*

Echopype currently supports:

- calibration and echo-integration to obtain volume backscattering strength (Sv)
  from the power data collected by EK60 and AZFP.

- simple noise removal by suppressing data points below an adaptively estimated
  noise floor [1]_.

- binning and averaging to obtain mean volume backscattering strength (MVBS)
  from the calibrated data.

The steps of performing these analysis for each echosounder are summarized below:

.. code-block:: python

    from echopype.model import EchoData
    data = EchoData('FILENAME.nc')
    data.calibrate()     # Calibration and echo-integration to obtain Sv
    data.remove_noise()  # denoised Sv
    data.get_MVBS()      # calculate MVBS

By default, these methods do not save the calculation results to disk.
The computation results can be accessed from ``data.Sv``, ``data.Sv_clean`` and
``data.MVBS`` as xarray DataSets with proper dimension labels.

To save the results to disk, pass an optional flag as in:

.. code-block:: python

    data.calibrate(save=True)     # Save Sv to disk
    data.remove_noise(save=True)  # Save Sv_clean to disk
    data.get_MVBS(save=True)      # Save MVBS to disk

The results will be saved into different files with postfixes ``_Sv.nc``,
``_Sv_clean.nc``, ``_MVBS.nc``.

Note that this default choice may be changed in the near future as
we move on to parallelize these operations.

AZFP specifics
~~~~~~~~~~~~~~
Here again there are some additional steps when performing these operations
on AZFP data.
Before calibration, the salinity and pressure values should be adjusted
if the default values of 29.6 PSU, and 60 dbars do not apply to the environment
where data collection took place. For example:

.. code-block:: python

   data.salinity = 30     # Salinity in PSU
   data.pressure = 50     # Pressure in dbars

These values are used in calculating the sea absorption coefficients
for data at each frequency and the sound speed in the water.
The sound speed is used to calculate the range.
These values can be retrieved with:

.. code-block:: python

    data.seawater_absorption
    data.sound_speed
    data.range


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
