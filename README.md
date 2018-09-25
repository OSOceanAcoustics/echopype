[![Build Status](https://travis-ci.org/leewujung/echopype.svg?branch=master)](https://travis-ci.org/leewujung/echopype)

# echopype
Open source tool for unpacking echosounder data.

Echopype is an open source Python package that converts data from various manufacturer proprietary format into netCDF4 files. The netCDF4 file follows the [ICES SONAR-netCDF4 convention](http://www.ices.dk/sites/pub/Publication%20Reports/Cooperative%20Research%20Report%20(CRR)/CRR341/CRR341.pdf). [Here](https://github.com/ices-eg/wg_WGFAST/tree/master/SONAR-netCDF4) is a GitHub folder with details of this convention in convenient tables. Echopype supports reading:
- .raw files from Simrad EK60 narrowband echosounder
- .raw files from Simrad EK80 broadband echosounder
- .01A files from ASL Environmental Sciences AZFP echosounder

Echopype also provides echo analysis (e.g. noise removal, dB differencing, etc.) and visualization functionality. We plan to develop a separate more advanced echo analysis package alongside echopype.

## Echosounder fun fact
Echosounders are high-frequency sonar systems commonly used to observe biological and non-biological objects as well as physical processes in the marine environment. Echosounders are now commonplace and continues to rapidly evolve. Since World War II, the use of echosounders has allowed scientists to locate and visualize the distributions, abundance, and behavior of a variety of marine organisms, including fish and zooplankton, and to describe physical processes. Utilizing acoustic scattering techniques provides a unique remote sensing tool to sample aquatic environments at finer temporal and spatial scales than many other sampling techniques, such as nets and video cameras.

## Existing resources for echosounder data analysis
Below is a list of what people use right now to read/process echosounder data:
- [EchoView](https://www.echoview.com/) (commercial software); [EchoView R](https://github.com/AustralianAntarcticDivision/EchoviewR)
- LSSS (IMR, Norway)
- [ESP3](https://bitbucket.org/echoanalysis/esp3/overview) (NZ, open source in Matlab, yoann.ladroit@niwa.co.nz)
- [Echogram](https://cran.r-project.org/web/packages/echogram/index.html)(read HAC format)
- [Movies](http://flotte.ifremer.fr/fleet/Presentation-of-the-fleet/Logiciels-embarques/MOVIES): ifremer, read HAC format
- [Movies3D](http://flotte.ifremer.fr/content/download/6032/129677/file/MOVIES3D_general.pdf): join EK60 and ME70 data, create HAC
- [PyEchoLab](https://drive.google.com/drive/folders/1q2ddkBx1cathE-6V_bIjqLBQj4yX7csm?usp=sharing)
- [sonar5](http://folk.uio.no/hbalk/sonar4_5/index.htm)
- [MATECHO](https://usermanual.wiki/Pdf/MatechoUserManual18052017.963673607.pdf): UMR LEMAR
- MSS? from DFO, cannot find links

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

