# Welcome to echopype!

**Echopype** is a package built to enable interoperability and scalability
in ocean sonar data processing.
These data are widely used for obtaining information about the distribution and
abundance of marine animals, such as fish and krill.
Our ability to collect large volumes of sonar data from a variety of
ocean platforms has grown significantly in the last decade.
However, most of the new data remain under-utilized.
echopype aims to address the root cause of this problem - the lack of
interoperable data format and scalable analysis workflows that adapt well
with increasing data volume - by providing open-source tools as entry points for
scientists to make discovery using these new data.

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


<!-- ```{image} images/GSoC-logo-horizontal.svg
:alt: Google Summer of Code logo
:width: 300px
```

```{attention}
In collaboration with the [Integrated Ocean Observing System (IOOS)](https://ioos.noaa.gov/), the Echopype team aims to recruit talented [Google Summer of Code (GSoC)](https://summerofcode.withgoogle.com/)
participants to help us upgrade the robustness and scalability of the Echopype package.

If you are a GSoC 2024 contributor, please head over to [GSoC contributor's guide](https://github.com/OSOceanAcoustics/echopype/gsoc_contrib_guide.md) to get more information specific to the program.
``` -->

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



## Citing echopype

Please cite echopype as follows:

Lee, W., Mayorga, E., Setiawan, L., Majeed, I., Nguyen, K., & Staneva, V. (2021). Echopype: A Python library for interoperable and scalable processing of water column sonar data for biological information. arXiv preprint arXiv:2111.00187

Citation information and project metadata are stored in `CITATION.cff`, which uses the [Citation File Format](https://citation-file-format.github.io/).

## License

Echopype is licensed under the open source
[Apache 2.0 license](https://opensource.org/licenses/Apache-2.0).
