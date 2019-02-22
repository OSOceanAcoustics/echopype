Why echopype?
=================

Over the last decade, advancements in instrumentation have produced
a deluge of water column sonar data from ships, moorings and autonomous
vehicles.
These data provide unprecedented opportunities to study marine ecosystems
at diverse spatial and temporal scales.
However, to date, these data remain significantly under‑utilized.
The root cause of this problem is the lack of interoperable data format
and scalable analysis workflows that adapt well with increasing data volume.

echopype aims at addressing the above issues by creating an interoperable
`netCDF`_ sonar data format, and supporting scalable computations through
`xarray`_ and `dask`_ based on these files. This approach makes it possible
to build sonar data analysis pipelines that are scalable and agnostic of
the instrument origin of the data.
In addition to using analysis modules provided by echopype, the converted data
in netCDF format can be easily combined and analyzed flexibly with
other climate and oceanographic data sets, facilitating the integration of
ocean sonar data in interdisciplinary oceanographic research.

.. _netCDF:
   https://www.unidata.ucar.edu/software/netcdf/docs/netcdf_introduction.html
.. _xarray: http://xarray.pydata.org/
.. _dask: http://dask.pydata.org/
.. _pandas: https://pandas.pydata.org/


Interoperability
------------------
echopype contains tools for converting data from various manufacturer-specific
formats into a standardized `netCDF`_ files.
This is useful, because netCDF is the `current defacto standard`_ in climate
research and is supported by many powerful Python packages for efficient
computation.
This means that once data are converted, researchers can then pick and choose
what they need directly from the data and are not restricted by specialized
and often proprietary data analysis software.

.. _current defacto standard:
   https://clouds.eos.ubc.ca/~phil/courses/parallel_python/02_xarray_zarr.html


Scalability
--------------
echopype is written to use as much as functionality of `xarray`_ as possible.
xarray is a powerful module that let you work with multi-dimensional labeled data
set the same way as in `pandas`_.
Under the hood, xarray uses `dask`_ to support parallel computations and
streaming computation on data sets that don't fit into memory.
Building on xarray and dask, the goal of echopype to make sonar data analysis
efficient and scalable through distributed computing through either local clusters
or on the cloud.
