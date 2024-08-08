<div align="center">
  <img src="https://raw.githubusercontent.com/OSOceanAcoustics/echopype/master/docs/source/_static/echopype_logo_banner.png" width="400">
</div>

# Echopype

<div>
  <a href="https://doi.org/10.5281/zenodo.3906999">
    <img src="https://img.shields.io/badge/DOI-10.5281/zenodo.3906999-blue" alt="DOI">
  </a>

  <a href="https://raw.githubusercontent.com/OSOceanAcoustics/echopype/master/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/OSOceanAcoustics/echopype">
  </a>
</div>

<div>
  <a href="https://github.com/OSOceanAcoustics/echopype/actions/workflows/build.yaml">
    <img src="https://github.com/OSOceanAcoustics/echopype/actions/workflows/build.yaml/badge.svg"/>
  </a>

  <a href="https://results.pre-commit.ci/latest/github/OSOceanAcoustics/echopype/master">
    <img src="https://results.pre-commit.ci/badge/github/OSOceanAcoustics/echopype/master.svg"/>
  </a>

  <a href="https://echopype.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/echopype/badge/?version=latest"/>
  </a>

  <a href="https://codecov.io/gh/OSOceanAcoustics/echopype">
    <img src="https://codecov.io/gh/OSOceanAcoustics/echopype/branch/master/graph/badge.svg?token=GT98F919XR"/>
  </a>
</div>

<div>
  <a href="https://pypi.org/project/echopype/">
    <img src="https://img.shields.io/pypi/v/echopype.svg"/>
  </a>

  <a href="https://anaconda.org/conda-forge/echopype">
    <img src="https://img.shields.io/conda/vn/conda-forge/echopype.svg"/>
  </a>
</div>

[![ssec](https://img.shields.io/badge/SSEC-Project-purple?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAQAAABedl5ZAAAACXBIWXMAAAHKAAABygHMtnUxAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAMNJREFUGBltwcEqwwEcAOAfc1F2sNsOTqSlNUopSv5jW1YzHHYY/6YtLa1Jy4mbl3Bz8QIeyKM4fMaUxr4vZnEpjWnmLMSYCysxTcddhF25+EvJia5hhCudULAePyRalvUteXIfBgYxJufRuaKuprKsbDjVUrUj40FNQ11PTzEmrCmrevPhRcVQai8m1PRVvOPZgX2JttWYsGhD3atbHWcyUqX4oqDtJkJiJHUYv+R1JbaNHJmP/+Q1HLu2GbNoSm3Ft0+Y1YMdPSTSwQAAAABJRU5ErkJggg==&style=plastic)](https://escience.washington.edu/echopype/)

Echopype is a package built to enable interoperability and scalability in ocean sonar data processing. These data are widely used for obtaining information about the distribution and abundance of marine animals, such as fish and krill. Our ability to collect large volumes of sonar data from a variety of ocean platforms has grown significantly in the last decade. However, most of the new data remain under-utilized. echopype aims to address the root cause of this problem - the lack of interoperable data format and scalable analysis workflows that adapt well with increasing data volume - by providing open-source tools as entry points for scientists to make discovery using these new data.

Watch the [echopype talk](https://www.youtube.com/watch?v=qboH7MyHrpU)
at SciPy 2019 for background, discussions and a quick demo!


## Documentation

Learn more about echopype in the official documentation at https://echopype.readthedocs.io. Check out executable examples in the companion repository https://github.com/OSOceanAcoustics/echopype-examples.


## Contributing

You can find information about how to contribute to echopype at our [Contributing Page](https://echopype.readthedocs.io/en/latest/contributing.html).

<!-- ## <img src="docs/source/images/GSoC-logo-horizontal.svg" alt="Google Summer of Code logo" width="300" style="padding-right: 50px; vertical-align: middle">

In collaboration with the [Integrated Ocean Observing System (IOOS)](https://ioos.noaa.gov/), the Echopype team aims to recruit talented [Google Summer of Code (GSoC)](https://summerofcode.withgoogle.com/)
participants to help us upgrade the robustness and scalability of the Echopype package.

If you are a GSoC 2024 contributor, please head over to [GSoC contributor's guide](gsoc_contrib_guide.md) to get more information specific to the program. -->



## Echopype doesn't run on your data?

Please report any bugs by [creating issues on GitHub](https://medium.com/nyc-planning-digital/writing-a-proper-github-issue-97427d62a20f).

[Pull requests](https://jarednielsen.com/learn-git-fork-pull-request/) are always welcome!


## Contributors

[![Contributors](https://contrib.rocks/image?repo=OSOceanAcoustics/echopype)](https://github.com/OSOceanAcoustics/echopype/graphs/contributors)

Wu-Jung Lee ([@leewujung](https://github.com/leewujung))
founded the echopype project in 2018 and continue to be the primary contributor
together with Caesar Tuguinay([@ctuguinay](https://github.com/ctuguinay)).
Emilio Mayorga ([@emiliom](https://github.com/emiliom)),
Landung "Don" Setiawan ([@lsetiawan](https://github.com/lsetiawan)),
Praneeth Ratna([@praneethratna](https://github.com/praneethratna)),
Brandon Reyes ([@b-reyes](https://github.com/b-reyes)),
Kavin Nguyen ([@ngkavin](https://github.com/ngkavin))
and Imran Majeed ([@imranmaj](https://github.com/imranmaj))
have contributed significantly to the code.
Valentina Staneva ([@valentina-s](https://github.com/valentina-s)) is also part of the development team.

A complete list of direct contributors is on our [GitHub Contributors Page](https://github.com/OSOceanAcoustics/echopype/graphs/contributors).


## Acknowledgement

We thank all previous and current contributors to Echopype,
including those whose contributions do not include code.
We thank Dave Billenness of ASL Environmental Sciences for
providing the AZFP Matlab Toolbox as reference for developing
support for the AZFP echosounder,
Rick Towler ([@rhtowler](https://github.com/rhtowler))
of the NOAA Alaska Fisheries Science Center
for providing low-level file parsing routines for
Simrad EK60 and EK80 echosounders,
and Alejandro Ariza ([@alejandro-ariza](https://github.com/alejandro-ariza))
for developing NumPy implementation of
acoustic analysis functions via Echopy, which
we referenced for several Echopype functions.

We also thank funding support from
the National Science Foundation,
NOAA Ocean Exploration,
NOAA Fisheries,
and software engineering support from
the University of Washington Scientific Software Engineering Center (SSEC),
as part of the Schmidt Futures Virtual Institute for Scientific Software (VISS) in 2023.

<div>
  <a href="https://oceanexplorer.noaa.gov/news/oer-updates/2021/fy21-ffo-schedule.html">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/79/NOAA_logo.svg/936px-NOAA_logo.svg.png" alt="NOAA_logo" width="120">
  </a>

  <a href="https://www.nsf.gov/awardsearch/showAward?AWD_ID=1849930&HistoricalAwards=false">
    <img src="https://upload.wikimedia.org/wikipedia/commons/7/7e/NSF_logo.png" alt="NSF_logo" width="120">
  </a>

  <a href="https://escience.washington.edu/software-engineering/ssec/">
    <img src="https://avatars.githubusercontent.com/u/122321194?s=200&v=4" alt="SSEC_logo" width="120">
  </a>
</div>





## License

Echopype is licensed under the open source [Apache 2.0 license](https://opensource.org/licenses/Apache-2.0).

---------------

Copyright (c) 2018-2024, Echopype Developers.
