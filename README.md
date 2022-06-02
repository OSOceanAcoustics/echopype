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
  <a href="https://github.com/OSOceanAcoustics/echopype/actions/workflows/ci.yaml">
    <img src="https://github.com/OSOceanAcoustics/echopype/actions/workflows/ci.yaml/badge.svg"/>
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

Echopype is a package built to enable interoperability and scalability in ocean sonar data processing. These data are widely used for obtaining information about the distribution and abundance of marine animals, such as fish and krill. Our ability to collect large volumes of sonar data from a variety of ocean platforms has grown significantly in the last decade. However, most of the new data remain under-utilized. echopype aims to address the root cause of this problem - the lack of interoperable data format and scalable analysis workflows that adapt well with increasing data volume - by providing open-source tools as entry points for scientists to make discovery using these new data.

Watch the [echopype talk](https://www.youtube.com/watch?v=qboH7MyHrpU)
at SciPy 2019 for background, discussions and a quick demo!

## Documentation

Learn more about echopype in the official documentation at https://echopype.readthedocs.io. Check out executable examples in the companion repository https://github.com/OSOceanAcoustics/echopype-examples.


## Contributing

You can find information about how to contribute to echopype at our [Contributing Page](https://echopype.readthedocs.io/en/latest/contributing.html).


## Echopype doesn't run on your data?

Please report any bugs by [creating issues on GitHub](https://medium.com/nyc-planning-digital/writing-a-proper-github-issue-97427d62a20f).

[Pull requests](https://jarednielsen.com/learn-git-fork-pull-request/) are always welcome!


Contributors
------------

Wu-Jung Lee ([@leewujung](https://github.com/leewujung)) founded the echopype project in 2018. It is currently led by Wu-Jung Lee and Emilio Mayorga ([@emiliom](https://github.com/emiliom)), who are primary developers together with Brandon Reyes ([@b-reyes](https://github.com/b-reyes)), Landung "Don" Setiawan ([@lsetiawan](https://github.com/lsetiawan)), and previously Kavin Nguyen ([@ngkavin](https://github.com/ngkavin)) and Imran Majeed ([@imranmaj](https://github.com/imranmaj)). Valentina Staneva ([@valentina-s](https://github.com/valentina-s)) is also part of the development team.

Other contributors are listed in [echopype documentation](https://echopype.readthedocs.io).

We thank Dave Billenness of ASL Environmental Sciences for
providing the AZFP Matlab Toolbox as reference for our
development of AZFP support in echopype.
We also thank Rick Towler ([@rhtowler](https://github.com/rhtowler))
of the NOAA Alaska Fisheries Science Center
for providing low-level file parsing routines for
Simrad EK60 and EK80 echosounders.


License
-------

Echopype is licensed under the open source [Apache 2.0 license](https://opensource.org/licenses/Apache-2.0).


---------------

Copyright (c) 2018-2022, echopype Developers.
