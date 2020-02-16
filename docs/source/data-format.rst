.. _data-format:

Data Format
===========

The lack of interoperability among data collected by different sonar
systems is currently a major obstacle toward integrative analysis of
sonar data at large scales.
echopype aims at addressing this problem by providing tools for
converting data from manufacturer-specific formats into a standardized
`netCDF <https://www.unidata.ucar.edu/software/netcdf/docs/
netcdf_introduction.html>`_ file format.
NetCDF is the `current defacto standard <https://clouds.eos.ubc.ca/
~phil/courses/parallel_python/02_xarray_zarr.html>`_ in climate
research and is supported by many powerful Python packages for
efficient computation, of which echopype take advantage in its
data analysis modules.


Interoperable netCDF files
---------------------------

Echopype follows the `ICES SONAR-netCDF4 convention`_ when possible
to create an interoperable data format to which all data are converted to.
We made modifications to the file structure in the convention so that
the computation can take full advantage of the power of
xarray in manipulating labelled multi-dimensional arrays.
See `Modifications to SONAR-netCDF4`_ for details of this modification.

We are also experimenting with using `zarr <https://zarr.readthedocs.io/en/stable/>`_
for cloud-optimized data storage and access.
It is possible to convert raw data files to .zarr files using the current
version of echopype, but computational support for this format has not
been thoroughly tested.

.. _ICES SONAR-netCDF4 convention:
   http://www.ices.dk/sites/pub/Publication%20Reports/Cooperative%20Research%20Report%20(CRR)/CRR341.pdf


Modifications to SONAR-netCDF4
------------------------------
Echopype is designed to handle multi-dimensional labelled data sets
efficiently, using ``xarray`` under the hood.
Therefore, we store backscatter data (the echoes) from
different frequency channels in a multi-dimensional array under a
single ``Beam`` group within a netCDF file.
Because of this change, all frequency-dependent parameters,
such as absorption coefficients, sample intervals, etc.,
are stored as an array with a frequency coordinate.

.. For example:
    .. code-block:: python

        import xarray as xr
        fname = 'some-path/some-file.nc'
        ds_beam = xr.open_dataset(fname, group='Beam')  # open the Beam group as an xarray DataSet
        ds_env = xr.open_dataset(fname, group='Environment')  # open the Environment group as an xarray DataSet
        In[1]: ds_env.absorption_coefficient_indicative
        Out[1]:
        In[2]: ds_beam.backscatter_r
        Out[2]:

This is different from the SONAR-netCDF4 convention, in which data
and parameters from different frequency channels are stored in different
beam groups under the ``Sonar`` group.
In the convention this was designed to accommodate potential differences
in the number of bins along range, or when there is a change of the
temporal length of data collection in the middle of a file.

However, it is more convenient to store and slice data directly by the
time, range, and frequency/beam direction coordinates (see ``pandas``
and ``xarray`` documentation for more info about coordinates and
dimensions) when the data are stored in a cubic form.
To accommodate this change, in the above two cases, echopype

- handles the uneven number of data samples along range by filling in
  ``NaN`` for the shorter channels, and
- splits the raw data file into multiple files when there is a change of
  the temporal length of data collection along range in the middle of a file.

In addition to computational efficiency, another advantage of
echopype's approach in restructuring the netCDF format is to enhance
the code readability and make data analysis computations more
tractable. For example, to extract data from a particular frequency,
users can simply do the following without worrying about the numerical
sequence of the index of the selected frequency:

.. code-block:: python

    import xarray as xr
    fname = 'some-path/some-file.nc'
    ds = xr.open_dataset(fname, group='Beam')  # open file as an xarray DataSet
    data_120k = ds.backscatter_r.sel(frequency=120000)  # explicit indexing for frequency


