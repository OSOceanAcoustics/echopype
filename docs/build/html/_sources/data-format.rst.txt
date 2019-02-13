Data Format
===============

The lack of interoperability among data collected by different sonar systems is currently a major obstacle in integrative analysis of sonar data at large scales.

echopype addresses this problem by creating tools for converting data from manufacturer-specific formats into a common, interoperable [netCDF](https://www.unidata.ucar.edu/software/netcdf/docs/netcdf_introduction.html) file format. NetCDF is the [current defacto standard](https://clouds.eos.ubc.ca/~phil/courses/parallel_python/02_xarray_zarr.html) in climate research and is supported by many powerful Python packages for efficient computation.

### Common file format
Currently echopype use a modified form of the [ICES SONAR-netCDF4 convention](http://www.ices.dk/sites/pub/Publication%20Reports/Cooperative%20Research%20Report%20(CRR)/CRR341/CRR341.pdf) as the common data format to which all data can be converted to. This convention is actually pretty new (released in mid 2018) and we think modification is needed to leverage functionalities of many powerful Python libraries. Detail of the modification is discussed [below](#modification-to-the-SONAR-netCDF4-convention). We consider switching to using [zarr](https://zarr.readthedocs.io/en/stable/) for cloud-optimized data storage and access in the future.

### Supported file types
In the first stage of development (until spring 2019), we plan to support data conversion for three sonar systems commonly found on research vessels:
- `.raw` files from Simrad EK60 narrowband echosounder
- `.raw` files from Simrad EK80 broadband echosounder
- `.01A` files from ASL Environmental Sciences AZFP echosounder

We plan to support conversion of *raw beam* data from common Acoustic Doppler Current Profilers (ADCPs) and echo data from multibeam sonar in the next stage of development.

### Modification to the SONAR-netCDF4 convention
We modified how backscatter data (the sonar echoes) are stored in a netCDF file. In the SONAR-netCDF4 convention, backscatter data from each sonar beam are stored in separate `Beam` subgroups under the `Sonar` group. This was designed to accommodate potential differences in the recorded echo lengths across different channels. Specifically, the number of bins and bin size along range (which corresponds to depth in most cases) may differ. However, this form of storage is not the most efficient if the number of bins and bin size are the same. If these parameters are identical, it is much more convenient to store and access data as a multi-dimensional array indexed by time, range, *and* frequency or beam direction (for multibeam data).

echopype handles this by adaptively switching the storage format depending on the range bin parameters. This is not a perfect solution, since this means that all subsequent computation needs to be adaptive as well. However this is an acceptable solution, since many popular analysis routines operate on top of heavily averaged and interpolated echo data, instead of the *raw* data discussed here. At that stage, data across frequency or channel are required to have the dimensions and coordinates.