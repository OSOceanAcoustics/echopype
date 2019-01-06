[![Build Status](https://travis-ci.org/leewujung/echopype.svg?branch=master)](https://travis-ci.org/leewujung/echopype)

# echopype

echopype is an open-source tool for converting and processing active sonar data for biological information in the ocean. It is an ongoing project and the planned functionalities will be implemented by spring 2019. The first stage of development goals include:
1. Conversion of data from 3 sonar manufacturer-specific file formats into a common format to enhance sonar data interoperability.
2. Basic data processing functionalities on the common data format. These include calibration, noise removal, and frequency-differencing.

This project is lead by [Wu-Jung Lee](http://leewujung.github.io) ([@leewujung](https://github.com/leewujung)) with contributions from multiple participants of [Oceanhackweek 2018](https://oceanhackweek.github.io/).

**Status update**: ongoing work:
- modification of the ICES SONAR-netCDF4 convention to facilitate later use of Python distributed processing libraries
- reorganization of project folder structure to facilitate modularization and testing.
- file conversion functionality from Simrad EK60 `.raw` files to `.nc` files nearly complete

## Sonar data format conversion
The lack of interoperability among data collected by different sonar systems is currently a major obstacle in integrative analysis of sonar data across large temporal and spatial regions. echopype aims to address this problem by creating convenient tools for converting data from manufacturer-specific formats into [netCDF files](https://www.unidata.ucar.edu/software/netcdf/docs/netcdf_introduction.html) of a common, interoperable format. This file format is widely used in geosciences and climate research, supported by many powerful software packages (e.g.,[xarray](http://xarray.pydata.org)) and allow random data access and out-of-memory operations.

### Common file format
Currently echopype use a modified form of the [ICES SONAR-netCDF4 convention](http://www.ices.dk/sites/pub/Publication%20Reports/Cooperative%20Research%20Report%20(CRR)/CRR341/CRR341.pdf) as the interoperable common data format to which all data will be converted to. [Here](https://github.com/ices-eg/wg_WGFAST/tree/master/SONAR-netCDF4) is a GitHub folder with details of this convention in convenient tables. Detail of the modification is discussed in a separate section below.

### Supported file types
In the first stage of development, we plan to support data conversion for data collected by three sonar systems commonly found on research vessels:
1. `.raw` files from Simrad EK60 narrowband echosounder
- `.raw` files from Simrad EK80 broadband echosounder
- `.01A` files from ASL Environmental Sciences AZFP echosounder

We plan to support conversion from *raw beam* data of common Acoustic Doppler Current Profilers (ADCPs) models in the next stage of development.

As of January 2019, implementation of EK60 `.raw` file conversion in nearly complete, and ASL AZFP `.01A` conversion is partially done.

### Modification to the SONAR-netCDF4 convention
echopype's modification to the SONAR-netCDF4 convention is on how the echo data are stored within the `Sonar` and `Beam` groups. Under the convention, data from different beams are stored in separate `Beam` groups to accommodate potential differences in the recorded lengths of echo data across different frequency channels. Specifically, the number of *bins* and bin size may differ. However, this form of storage is not the most efficient if the number of bins and bin size are the same. In this scenario, it is much more convenient to store (and access) data across frequency as a multi-dimensional array indexed by time, range, *and* frequency.

echopype will handle this by adaptively switching the storage format depending on these parameters. This is not a perfect solution, since this means that all subsequent computation needs to be adaptive as well. However this is an acceptable solution, since many popular echo analysis methods operate on top of heavily averaged and interpolated echo data, instead of the *raw* data discussed here. Once at that stage, data across frequency *will* be of the dimensions and coordinates.

## Sonar data processing
In the first stage of development, we plan to support basic sonar data processing routines, including calibration, noise removal, echo integration, and frequency-differencing for classification of echo sources. These computations are straightforward but largely repetitive, and therefore are ideal for demonstrating the power of using file formats that support efficient, non-sequential access of large volumes of data.

We plan to support more sophisticated sonar data processing routines, such as bottom detection, single target detection,  aggregation detection, and target tracking in the next stage of development. We also plan to develop interactive visualization tools for multi-frequency and broadband sonar data, and demonstrate the functionality using data set across a large spatial scale.

---------------------------------
Below still needs update

## Other software resources
Below is a list of other existing software resources for processing sonar data:
- [EchoView](https://www.echoview.com/): GUI-based commercial software, current go-to tool for echosounder data analysis
  - [EchoView R](https://github.com/AustralianAntarcticDivision/EchoviewR): interfacing R and EchoView
- [ESP3](https://bitbucket.org/echoanalysis/esp3/overview): open-source Matlab toolbox
- [Echogram](https://github.com/hvillalo/echogram)
  - (https://CRAN.R-project.org/package=echogram): R package that reads data in HAC format
- [Movies](http://flotte.ifremer.fr/fleet/Presentation-of-the-fleet/Logiciels-embarques/MOVIES): ifremer, read HAC format
  - [Movies3D](http://flotte.ifremer.fr/content/download/6032/129677/file/MOVIES3D_general.pdf): join EK60 and ME70 data, create HAC
- [PyEchoLab](https://drive.google.com/drive/folders/1q2ddkBx1cathE-6V_bIjqLBQj4yX7csm?usp=sharing)
- [sonar5](http://folk.uio.no/hbalk/sonar4_5/index.htm)
- [MATECHO](https://usermanual.wiki/Pdf/MatechoUserManual18052017.963673607.pdf): UMR LEMAR
- LSSS (IMR, Norway)
- MSS? from DFO, cannot find links


## Echosounder fun fact
Echosounders are high-frequency sonar systems commonly used to observe biological and non-biological objects as well as physical processes in the marine environment. Since World War II, the use of echosounders has allowed scientists to locate and visualize the distributions, abundance, and behavior of a variety of marine organisms, including fish and zooplankton, and to describe physical processes. Utilizing acoustic scattering techniques provides a unique remote sensing tool to sample aquatic environments at finer temporal and spatial scales than many other sampling techniques, such as nets and video cameras.

## Existing resources for echosounder data analysis


## Resources and sample data for echopype development
- Existing code (in a [Google Drive folder](https://drive.google.com/drive/folders/1q2ddkBx1cathE-6V_bIjqLBQj4yX7csm?usp=sharing))
  - Matlab package to read EK80 data
  - Matlab package to read AZFP data
  - PyEchoLab2
- Current echopype (which uses HDF5 and only unpacks EK60)
  - can read data from EK60 and save into HDF5 file
  - can clean up noise, do frequency differencing and plot echogram
- OOI CI parsers:
  - for [AZFP data](https://github.com/oceanobservatories/mi-instrument/blob/master/mi/dataset/parser/zplsc_c.py)
  - for [EK60 data](https://github.com/oceanobservatories/mi-instrument/tree/master/mi/instrument/kut/ek60/ooicore)

## Developement details
### Overall
- unpacking: unpacking data should be a stand-alone function
  - read EK60, EK80, and AZFP data
- model: classes to read in and manipulate data as Python objects
- view: operations on data (adapt from current echopype)
### Use cases (concrete steps that a user wants to do, help guide the tasks)
- scanning data and decide what files you want to analyze
- get rid of bottom, get rid of noise, and then db-diff
### Specific tasks
- set up continuous integration (WJ)
  - write tests
- figure out how to save things into ICES recommended format --> let's look at what MATECHO did (JC)
- clean up current echopype for unpacking EK60, make attribute names match ICES naming convention (WJ, EL)
- unpack EK80 --> EK80 uses XML for metadata, create class attributes that match ICES naming convention (EO, EF)
- unpack AFZP --> AFZP uses XML for metadata, create class attributes that match ICES naming convention (MP, EO, ML)
- echo summary view for where was the data from, are there are things in the water column, time, etc.
- how to speed up scrolling when viewing echograms?
- manipulate data by combining current echopype and pyecholab
  - provide bottom as a mask
  - remove noise: may be different for ship-based and moored data
  - freq-differencing
  - broadband single target detection
  - narrowband single target detection
  - multi-freq indicator functions
  - narrowband calibration re. Demer et al. (JC)
  - broadband calibration re. Stanton/Chu+Jech/Lavery (WJ)
