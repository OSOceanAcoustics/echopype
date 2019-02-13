echopype
===========

The goal of ``echopype`` is to improve the interoperability and scalability in processing water column sonar data. These data are commonly used for inferring information about mid-trophic organisms in the ocean, such as fish and krill.


Functionality
----------------
``echopype`` include tools for:

- Converting different manufacturer-specifc data files into an interoperable netCDF format.

- Processing large volumes of sonar data in local or cloud storage by leveraging Python distributed computing libraries.

The current version supports file conversion for the ``.raw`` data files collected by the SimRad EK60 echosounder. Conversion for other types of data files, including the ``.01A`` files from AZFP echosounder, the ``.raw`` files from the SimRad broadband EK80 echosounder, and the *raw beam* data from ADCP (Acoustic Doppler Current Profiler) will be added in future releases.


Installation
--------------

To install ``echopype``, do the following in your terminal:

.. code-block:: console

    $ pip install echopype


Using ``echopype``
-------------------

To batch convert whole bunch of ``.raw`` files in the terminal:

.. code-block:: console

    $ echopype_converter -s ek60 data/*.raw

This will generate whole bunch of ``.nc`` files with the same leading filename in the same directory as the ``.raw`` files.

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

Same as in the command line case, this will generate a ``FILENAME.nc`` in the same directory as ``FILENAME.raw``.

The ``ConvertEK60`` instance contains all the data unpacked from the .raw file, so it is a good idea to clear it from memory once done with conversion.


License
----------

``echopype`` is licensed under the open source Apache 2.0 license.
his project is lead by `Wu-Jung Lee
<http://leewujung.github.io>`_ (@leewujung) and contributers include:

- `Valentina Staneva <https://escience.washington.edu/people/valentina-staneva/>`_ (@valentina-s)
- `Marian Peña <https://www.researchgate.net/profile/Marian_Pena2>`_ (@marianpena)
- `Mark Langhirt <https://www.linkedin.com/in/mark-langhirt-7b33ba80>`_ (@bnwkeys)
- `Emma Ozanich <https://www.linkedin.com/in/emma-reeves-ozanich-b8671938/>`_ (@emma-ozanich)
- `Erin Labrecque <https://www.linkedin.com/in/erin-labrecque/>`_ (@erinann)
- `Aaron Marburg <http://apl.uw.edu/people/profile.php?last_name=Marburg&first_name=Aaron>`_ (@amarburg)


Copyright (c) 2018--, Wu-Jung Lee, Applied Physics Laboratory, University of Washington.
