Open converted files
====================


Open a converted netCDF or Zarr dataset
---------------------------------------

Converted netCDF files can be opened with the ``open_converted`` function
that returns a lazy loaded ``EchoData`` object (only metadata are read during opening):

.. code-block:: python

   import echopype as ep
   file_path = "./converted_files/file.nc"      # path to a converted nc file
   ed = ep.open_converted(file_path)            # create an EchoData object

Likewise, specify the path to open a Zarr dataset. To open such a dataset from 
cloud storage, use the same ``storage_options`` parameter as with 
`open_raw <convert.html#aws-s3-access>`_. For example:

.. code-block:: python

   s3_path = "s3://s3bucketname/directory_path/dataset.zarr"     # S3 dataset path
   ed = ep.open_converted(s3_path, storage_options={"anon": True})

Combine EchoData objects
------------------------

Converted data found in multiple files corresponding to the same instrument deployment can be 
combined into a single ``EchoData`` object. First assemble a list of ``EchoData`` objects from the 
converted files (netCDF or Zarr). Then apply ``combine_echodata`` on this list to combine all
the data into a single ``EchoData`` object in memory:

.. code-block:: python

   ed_list = []
   for converted_file in ["convertedfile1.nc", "convertedfile2.nc"]:
      ed_list.append(ep.open_converted(converted_file))

   combined_ed = ep.combine_echodata(ed_list)


.. TODO: Mention ep.qc.exist_reversed_time and coerce rev time ..


EchoData object
---------------

``EchoData`` is an object that conveniently handles raw converted data from either raw 
instrument files (via ``open_raw``) or previously converted and standardized raw files 
(via ``open_converted``). It is essentially a container for multiple ``xarray`` ``Dataset`` 
objects, where each such object corresponds to one of the netCDF4 groups specified in the 
SONAR-netCDF4 convention followed by echopype. ``EchoData`` objects are used for conveniently 
accessing and exploring the echosounder data, for calibration and other processing, and for 
`serializing into netCDF4 or Zarr file formats <convert.html#file-export>`_. 

A sample ``EchoData`` object is presented below using the ``Dataset`` HTML browser generated
by ``xarray``, collected into SONAR-netCDF4 groups. Select each group and drill down to variables 
and attributes to examine the structure and representative content of an ``EchoData`` object.

.. raw:: html

    <iframe src="_static/echodata_sample.html" width="100%" height="400" style="border: none;"></iframe>
