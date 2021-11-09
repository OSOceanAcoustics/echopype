Open converted files
====================


Open a converted netCDF or Zarr dataset
---------------------------------------

Converted netCDF and Zarr files can be opened with the ``open_converted`` function
that returns a lazy loaded ``EchoData`` object (only metadata are read during opening):

.. code-block:: python

   import echopype as ep
   nc_path = './converted_files/file.nc'        # path to a converted nc file
   ed = ep.open_converted(nc_path)              # create an EchoData object


.. TODO: Demo opening from Zarr on S3


EchoData object
---------------

.. TODO: Examine EchoData objects. Include the echopype echodata HTML repr, 
   and a sample of the xarray HTML repr for one of the groups

``EchoData`` is an object for conveniently handling raw converted data from either 
raw instrument files (via ``open_raw``) or previously converted and standardized raw files
(via ``open_converted``). It is essentially a container for multiple xarray ``Dataset`` objects,
where each such object corresponds to one of the netCDF4 groups specified in the SONAR-netCDF4
convention followed by echopype. It is used for conveniently accessing and exploring the 
echosounder data and for calibration and other processing.


.. TODO: Add section on combine_echodata, including the sample code from echopype_tour nb
   ed_list = []
   for converted_file in sorted(glob.glob(str(converted_dpath / "*.nc"))):
      ed_list.append(ep.open_converted(converted_file))
   combined_ed = ep.combine_echodata(ed_list)

.. TODO: Mention ep.qc.exist_reversed_time and coerce rev time ..
