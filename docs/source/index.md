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
Valentina Staneva ([@valentina-s](https://github.com/valentina-s)) is also part of the development team. Lloyd Izard ([@LOCEANlloydizard](https://github.com/LOCEANlloydizard)) joined the project in 2025 and is an active contributor.

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

We thank funding support from the National Science Foundation,
NOAA Ocean Exploration, NOAA Fisheries,
and the VOTO Foundation.
We also acknowledge software engineering support from
the University of Washington Scientific Software Engineering Center (SSEC),
as part of the Schmidt Futures Virtual Institute for Scientific Software (VISS) in 2023.

<table align="center" border="0" cellpadding="12" cellspacing="0" style="border-collapse: collapse;">
  <tr>
    <td align="center" style="border: 0;">
      <a href="https://oceanexplorer.noaa.gov/news/oer-updates/2021/fy21-ffo-schedule.html" style="text-decoration:none;">
        <img src="assets/logos/noaa.png" alt="NOAA" width="120">
      </a>
    </td>

    <td align="center" style="border: 0;">
      <a href="https://www.nsf.gov/awardsearch/showAward?AWD_ID=1849930&HistoricalAwards=false" style="text-decoration:none;">
        <img src="assets/logos/nsf.png" alt="NSF" width="120">
      </a>
    </td>

    <td align="center" style="border: 0;">
      <a href="https://escience.washington.edu/software-engineering/ssec/" style="text-decoration:none;">
        <img src="assets/logos/ssec.png" alt="SSEC" width="120">
      </a>
    </td>

    <td align="center" style="border: 0;">
      <a href="https://voiceoftheocean.org/" style="text-decoration:none;">
        <img src="assets/logos/voto.png" alt="VOTO Foundation" width="120">
      </a>
    </td>
  </tr>
</table>


## Citing echopype

Please cite echopype as follows:

Wu-Jung Lee, Landung Setiawan, Caesar Tuguinay, Emilio Mayorga, Valentina Staneva, Interoperable and scalable echosounder data processing with Echopype, ICES Journal of Marine Science, Volume 81, Issue 10, December 2024, Pages 1941–1951, https://doi.org/10.1093/icesjms/fsae133

Citation information and project metadata are stored in `CITATION.cff`, which uses the [Citation File Format](https://citation-file-format.github.io/).

## License

Echopype is licensed under the open source
[Apache 2.0 license](https://opensource.org/licenses/Apache-2.0).
