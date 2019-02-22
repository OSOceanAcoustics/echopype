Using echopype
========================


Installation
--------------

echopype is pip installable. You can install echopype by doing the following
in your terminal:

.. code-block:: console

    $ pip install echopype



File conversion
-----------------

echopype supports batch conversion of ``.raw`` files to netCDF ``.nc``
format in the terminal:

.. code-block:: console

    $ echopype_converter -s ek60 data/*.raw

This will generate corresponding ``.nc`` files with the same leading
filename as the original ``.raw`` files in the same directory.
See `Data Format <data-format>`_ for details about the converted file format.

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



Data analysis
---------------

The data analysis functionalites of echopype is being developed actively.
Be sure to check back here often!

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
as xarray DataSets. The outputs of these methods are are xarray DataSets with
proper dimension labels.



---------------

.. [1] De Robertis and Higginbottoms (2007) A post-processing technique to
   estimate the signal-to-noise ratio and remove echosounder background noise.
   `ICES J. Mar. Sci. 64(6): 1282–1291. <https://academic.oup.com/icesjms/article/64/6/1282/616894>`_