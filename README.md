[![Build Status](https://travis-ci.org/oceanhackweek/ohw18_echopype.svg?branch=master)](https://travis-ci.org/oceanhackweek/ohw18_echopype)

# echopype
Open source tools for unpacking and analyzing active sonar data.

## What?
What do people use right now to read/process echosounder data?
- EchoView (commercial); [EchoView R](https://github.com/AustralianAntarcticDivision/EchoviewR)
- LSSS (Norway)
- [ESP3](https://bitbucket.org/echoanalysis/esp3/overview) (NZ, open source in Matlab, yoann.ladroit@niwa.co.nz)
- [Echogram](https://cran.r-project.org/web/packages/echogram/index.html)(read HAC format)
- [Movies](http://flotte.ifremer.fr/fleet/Presentation-of-the-fleet/Logiciels-embarques/MOVIES): ifremer, read HAC format
- [Movies3D](http://flotte.ifremer.fr/content/download/6032/129677/file/MOVIES3D_general.pdf): join EK60 and ME70 data, create HAC
- MSS? from DFO, cannot find links
- [PyEchoLab](https://drive.google.com/drive/folders/1q2ddkBx1cathE-6V_bIjqLBQj4yX7csm?usp=sharing)
- [sonar5](http://folk.uio.no/hbalk/sonar4_5/index.htm)
- [MATECHO](https://usermanual.wiki/Pdf/MatechoUserManual18052017.963673607.pdf): UMR LEMAR

## So What?
Ultimate goal: combine acoustic data with environmental data to work on science questions.
1. Be able to read data from the OOI arrays (AZFP and EK60)
2. Plot an echogram from real data (OOI and ONC, and glider or AUV data if we can get our hands on some of them)

Misc:
- Explore ways to view and process data using Python
  - example: [echo metrics](https://github.com/ElOceanografo/EchoMetrics) this needs an echogram as input, we can create a pipeline for that
- Convert manufacturer proprietary formats to netCDF files following [ICES sonar-netCDF4 convention](http://www.ices.dk/sites/pub/Publication%20Reports/Cooperative%20Research%20Report%20(CRR)/CRR341/CRR341.pdf)
- [Here](https://github.com/ices-eg/wg_WGFAST/tree/master/SONAR-netCDF4) is a GitHub folder with the details in nice tables
- testing it 

## Now What?
- [Existing code in Google folder](https://drive.google.com/drive/folders/1q2ddkBx1cathE-6V_bIjqLBQj4yX7csm?usp=sharing)
  - Matlab package to read EK80 data
  - Matlab package to read AZFP data
  - PyEchoLab2
- Current echopype (which uses HDF5 and only unpacks EK60)
  - can read data from EK60 and save into HDF5 file
  - can clean up noise, do frequency differencing and plot echogram
- [OOI CI parser](https://github.com/oceanobservatories/mi-instrument/blob/efc9bd67406f996c2448f893541e165ed7bbfb6d/mi/dataset/parser/zplsc_c.py) that parses AZFP data in Python
- Goals
  - model: unpacking data should be a stand-alone function
    - read from AZFP and EK80
  - view: operations on data (adapt from current echopype)
- Use cases (concrete steps that a user wants to do, help guide the tasks)
  - scanning data and decide what files you want to analyze
  - get rid of bottom, get rid of noise, and then db-diff
- Specific tasks
  - set up continuous integration (VS)
    - write tests
  - figure out how to save things into ICES recommended format --> let's look at what MATECHO did (JC)
  - clean up current echopype for unpacking EK60, make attribute names match ICES naming convention (WJ, EL)
  - unpack EK80 --> EK80 uses XML for metadata, create class attributes that match ICES naming convention (EO, EF)
  - unpack AFZP --> AFZP uses XML for metadata, create class attributes that match ICES naming convention (MP, EO, ML)
  - manipulate data by combining current echopype and pyecholab
    - provide bottom as a mask
    - remove noise: may be different for ship-based and moored data
    - freq-differencing
    - broadband single target detection
    - narrowband single target detection
    - multi-freq indicator functions
    - narrowband calibration re. Demer et al. (JC)
    - broadband calibration re. Stanton/Chu+Jech/Lavery (WJ)
  - echo summary view for where was the data from, are there are things in the water column, time, etc.
  - how to speed up scrolling?

