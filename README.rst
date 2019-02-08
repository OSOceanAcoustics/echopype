echopype
===========

``echopype`` is an open-source toolkit for analyzing active sonar data for biological information in the ocean.

The goal of ``echopype`` is to improve the interoperability and scalability in processing water column sonar data.

Functionality
----------------

- Convert different manufacturer-specifc data files into an interoperable netCDF format.

- Leverage Python distributed computing libraries in processing large volumes of sonar data in local or cloud storage.

The current version supports converting ``.raw`` data files collected by the SimRad EK60 echosounder. Conversion for other types of data files, including the ``.01A`` files from AZFP echosounder and the ``.raw`` files from the SimRad broadband echosounder, will be added in future releases.

Details of the above can be found `here <https://github.com/OSOceanAcoustics/echopype>`_.




Installation
--------------

   .. code-block:: console

       $ pip install echopype


Using ``echopype``
-------------------

To batch convert whole bunch of *.raw files in the terminal, do:
   .. code-block:: console

      $ echopype_converter -s ek60 data/*.raw

To use the EK60 data converter in a Python session, you can do:
   .. code-block:: python

       # import as part of a submodule
       from echopype.convert import ConvertEK60
       data_tmp = ConvertEK60('FILENAME.raw')
       data_tmp.raw2nc()

Or do
   .. code-block: python

       # import the full module
       import echopype as ep
       data_tmp = ep.convert.ConvertEK60('FILENAME.raw')
       data_tmp.raw2nc()

The ``ConvertEK60`` instance contains all the data unpacked from the .raw file, so it is a good idea to clear it from memory once done with conversion.
