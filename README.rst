echopype
===========

The goal of echopype is to improve the interoperability and scalability in
processing water column sonar data. These data are commonly used for inferring
information about mid-trophic organisms in the ocean, such as fish and krill.


Functionality
----------------
echopype include tools for:

- Converting different manufacturer-specifc data files into an interoperable
  netCDF format.

- Processing large volumes of sonar data in local or cloud storage by leveraging
  Python distributed computing libraries.

The current version supports file conversion for the ``.raw`` data files
collected by the SimRad EK60 echosounder. Conversion for other types of data
files, including the ``.01A`` files from AZFP echosounder, the ``.raw`` files
from the SimRad broadband EK80 echosounder, and the *raw beam* data from ADCP
(Acoustic Doppler Current Profiler) will be added in future releases.


Installation
--------------

To install echopype, do the following in your terminal:

.. code-block:: console

    $ pip install echopype


Using echopype
-------------------

File conversion
+++++++++++++++++++

To batch convert ``.raw`` files to the interoperable netCDF format in the
terminal, do:

.. code-block:: console

    $ echopype_converter -s ek60 data/*.raw

This will generate corresponding ``.nc`` files with the same leading
filename as the original ``.raw`` files in the same directory.

To use the EK60 data converter in a Python session, you can do:

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

Same as in the command line case, this will generate a ``FILENAME.nc``
in the same directory as ``FILENAME.raw``.

The ``ConvertEK60`` instance contains all the data unpacked from the
.raw file, so it is a good idea to clear it from memory once done with
conversion.


Data analysis
+++++++++++++++++++

The data analysis functionalites of echopype is being developed actively.
echopype currently supports:

- calibration and echo-integration to obtain volume backscattering strength (Sv)
  from the power data collected by EK60.

- simple noise removal by suppressing data points below an adaptively estimated
  noise floor [1]_.

- binning and averaging to obtain mean volume backscattering strength (MVBS)
  from the calibrated data.

The steps of performing these analysis are summarized below:

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
as xarray DataSets. All outputs are xarray DataSets with proper dimension
labeling.


License
----------

echopype is licensed under the open source Apache 2.0 license.

This project is lead by `Wu-Jung Lee <http://leewujung.github.io>`_ (@leewujung).
Other contributors include:

- `Valentina Staneva <https://escience.washington.edu/people/valentina-staneva/>`_
  (@valentina-s)
- `Marian Peña <https://www.researchgate.net/profile/Marian_Pena2>`_
  (@marianpena)
- `Mark Langhirt <https://www.linkedin.com/in/mark-langhirt-7b33ba80>`_ (@bnwkeys)
- `Erin Labrecque <https://www.linkedin.com/in/erin-labrecque/>`_
  (@erinann)
- `Emma Ozanich <https://www.linkedin.com/in/emma-reeves-ozanich-b8671938/>`_
  (@emma-ozanich)
- `Aaron Marburg <http://apl.uw.edu/people/profile.php?last_name=Marburg&first_name=Aaron>`_
  (@amarburg)


References
------------
.. [1] De Robertis and Higginbottoms (2007) A post-processing technique to estimate
        the signal-to-noise ratio and remove echosounder background noise.
        `ICES J. Mar. Sci. 64(6): 1282–1291. <https://academic.oup
        .com/icesjms/article/64/6/1282/616894>`_


---------------

Copyright (c) 2018--, Wu-Jung Lee, Applied Physics Laboratory, University of Washington.
