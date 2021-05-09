Open converted files
====================


Open a converted netCDF or Zarr dataset
---------------------------------------

- Open converted nc and zarr files. Open single vs multiple files?


EchoData object
---------------

- Examine EchoData objects
- EchoData is an object that handles interfacing raw converted data. It is used for calibration and other processing.
- Echo data model class for handling raw converted data, including multiple files associated with the same data set.
- essentially a container for multiple xarray ``Dataset`` instances
