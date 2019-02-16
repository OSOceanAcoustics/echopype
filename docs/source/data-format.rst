Data Format
===============

The lack of interoperability among data collected by different sonar
systems is currently a major obstacle in integrative analysis of sonar
data at large scales.

echopype addresses this problem by creating tools for converting data
from manufacturer-specific formats into a common, interoperable
`netCDF <https://www.unidata.ucar.edu/software/netcdf/docs/
netcdf_introduction.html>`_ file format.
NetCDF is the `current defacto standard <https://clouds.eos.ubc.ca/
~phil/courses/parallel_python/02_xarray_zarr.html>`_ in climate
research and is supported by many powerful Python packages for
efficient computation.


Interoperable netCDF file
--------------------------

echopype use a modified form of the `ICES SONAR-netCDF4 convention
<http://www.ices.dk/sites/pub/Publication%20Reports/
Cooperative%20Research%20Report%20(CRR)/CRR341/CRR341.pdf>`_ as the
interoperable data format to which all data can be converted to.
This convention is very recently proposed (mid 2018) and we made
critical modifications to the data storage structure so that
the underlying computation can take full advantage of the power of
xarray in manipulating labelled multi-dimensional arrays. Detail of
the modification is discussed a separate section below. We also
consider switching to use `zarr <https://zarr.readthedocs.io/en/stable/>`_
in the future for cloud-optimized data storage and access.


Supported file types
----------------------

In the first stage of development (until spring 2019), we plan to
support data conversion for three sonar systems commonly found on
research vessels and moorings:
- `.raw` files from Simrad EK60 narrowband echosounder
- `.raw` files from Simrad EK80 broadband echosounder
- `.01A` files from ASL Environmental Sciences AZFP echosounder

We plan to support conversion of *raw beam* data from common Acoustic
Doppler Current Profilers (ADCPs) and echo data from multibeam sonar
in the next stage of development.

Modifications to SONAR-netCDF4
----------------------------------------------
We modified how backscatter data (the sonar echoes) are stored in a
netCDF file. In the SONAR-netCDF4 convention, backscatter data from
each sonar beam are stored in separate ``Beam`` subgroups under the
``Sonar`` group. This was designed to accommodate potential
differences in the recorded echo lengths across different channels.
Specifically, the number of bins and bin size along range
(which corresponds to depth in most cases) may differ. However, this
form of storage is not the most efficient if the number of bins and
bin size are the same. If these parameters are identical, it is much
more convenient to store and access data as a multi-dimensional array
indexed by time, range, *and* frequency or beam direction (for
multibeam data).

echopype handles this by adaptively switching the storage format
depending on the range bin parameters. This is not a perfect solution,
since this means that all subsequent computation needs to be adaptive
as well. However this is an acceptable solution, since many popular
analysis routines operate on top of heavily averaged and interpolated
echo data, instead of the *raw* data discussed here. At that stage,
data across frequency or channel are required to have the dimensions
and coordinates.