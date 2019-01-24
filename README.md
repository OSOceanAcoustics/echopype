[![Build Status](https://travis-ci.org/leewujung/echopype.svg?branch=master)](https://travis-ci.org/leewujung/echopype)

# echopype

echopype is an open-source tool for converting and processing active sonar data for biological information in the ocean. The goal is to create a toolkit that can leverage the rapidly-developing Python distributed processing libraries and interface with both local and cloud storage.

This is an ongoing project and the first stage (current) development goals are:
1. Conversion of data from 3 sonar manufacturer-specific file formats into a [common file format](#common-file-format) to enhance sonar data interoperability.
2. Basic [data processing functionalities](#sonar-data-processing) that operate on the common data format.

Details of the above can be found below.

This project is lead by [Wu-Jung Lee](http://leewujung.github.io) (@leewujung) and was contributed by multiple participants during [Oceanhackweek 2018](https://oceanhackweek.github.io/).

**Status update**: ongoing work:
- reorganization of project folder structure to facilitate modularization and testing
- modification of the ICES SONAR-netCDF4 convention to allow efficient indexing
- completion of file conversion functionality from Simrad EK60 `.raw` and ASL AZFP `.01A` formats to `.nc` files


## Sonar data format conversion
The lack of interoperability among data collected by different sonar systems is a major obstacle in integrative analysis of sonar data at large scales. echopype addresses this problem by creating tools for converting data from manufacturer-specific formats into a common, interoperable [netCDF](https://www.unidata.ucar.edu/software/netcdf/docs/netcdf_introduction.html) file format. NetCDF is the [current defacto standard](https://clouds.eos.ubc.ca/~phil/courses/parallel_python/02_xarray_zarr.html) in climate research and is supported by many powerful Python packages for efficient computation.

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

## Sonar data processing
In the first stage of development (until spring 2019), we plan to support basic sonar data processing routines, including calibration, noise removal, echo integration, and frequency-differencing for classification of echo sources. These computations are straightforward but require repetitive operations over the entire data set. They are therefore ideal for demonstrating the power of using file formats that support efficient, non-sequential access of large volumes of data.

In the future we plan to extend the package to support other more sophisticated processing, such as bottom detection, single target detection and tracking. We also plan to develop interactive visualization tools for multi-frequency and broadband sonar data.

## Other resources
Below is a list of existing software resources for processing sonar data:
- [EchoView](https://www.echoview.com/): GUI-based commercial software, current go-to tool for echosounder data analysis
  - [EchoView R](https://github.com/AustralianAntarcticDivision/EchoviewR): interfacing R and EchoView
- [Echogram](https://CRAN.R-project.org/package=echogram): an R package for reading data in the [HAC](http://www.ices.dk/sites/pub/Publication%20Reports/Cooperative%20Research%20Report%20%28CRR%29/crr278/crr278.pdf) format
- [ESP3](https://bitbucket.org/echoanalysis/esp3/overview): a Matlab toolbox
- [LSSS](https://cmr.no/projects/10396/lsss/) (Large Scale Survey System)
- [MATECHO](https://usermanual.wiki/Pdf/MatechoUserManual18052017.963673607.pdf): requires MATLAB and Movies3D
  - [Movies3D](http://flotte.ifremer.fr/content/download/6032/129677/file/MOVIES3D_general.pdf): reads and writes HAC files and is capable of joining EK60 and ME70 data
- [PyEchoLab](https://drive.google.com/drive/folders/1q2ddkBx1cathE-6V_bIjqLBQj4yX7csm?usp=sharing): a Python package based solely on the NumPy library
- [Sonar4 and Sonar5-Pro](http://folk.uio.no/hbalk/sonar4_5/index.htm)
