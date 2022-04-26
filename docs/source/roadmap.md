.. _roadmap:

# Development Roadmap

## Scope and goals

Echopype is a library aimed at enabling interoperability and scalability in processing ocean sonar data. The focus and scope right now are on scientific echosounders that are widely used in fishery and marine ecological surveys. We envision that echopype will provide "building blocks" that can be strung together to construct data processing pipelines to bring raw data files collected by these instruments to "analysis-ready" data products that can be used by a wider community of researchers in the ocean science domain. We also plan for echopype to be flexible in accommodating both local and cloud computing environments, such that data processing pipelines can be prototyped locally and scaled up for larger scale processing on the cloud.


## Data standardization

At the core of echopype is a data conversion and standardization module that returns an `EchoData` object that allows easy and intuitive access and understanding of ocean sonar data in a netCDF data model. Currently echopype converts raw data files from the instruments into `.nc/.zarr` files following the convention SONAR-netCDF4 ver.1, except for a small number of exceptions that we feel improve the data coherence and efficiency of data access. This conversion step ensures that downstream processing can be developed and executed in an instrument-agnostic manner, which is critical for tackling the currently tedious and labor-intensive data wrangling operations associated with integrative analysis of data from heterogeneous instrument sources.

:::{important}
The upcoming release v0.6.0 will include breaking changes to the structure of the core netCDF data model, which impacts both content of the `EchoData` object and the converted .nc/.zarr files. These changes were necessary to enhance the compliance of echopype-generated data with SONAR-netCDF4 ver.1. Data converted using echopype v0.5.x continue to be supported beyond v0.6.0 and can be opened via `open_converted` into the new format.
:::

FIX THIS AND LINK: See :ref:`convert` and :ref:`data-format` for more detail.



## Data processing levels

We plan to put together a set of clearly defined progression of data processing levels for active ocean sonar data, spanning the continuum from raw data records to acoustically derived  biological information. The development will leverage the accumulated experience from remote sensing and large-scale, long-term environmental observing communities (e.g., [NASA Data Processing Levels](https://earthdata.nasa.gov/collaborate/open-data-services-and-software/data-information-policy/data-levels/)). We will embed the processing provenance specifications in the echopype computational pipeline.

See PAGE X for our current draft of data processing levels. [WHERE SHOULD THIS GO?]

Below is a draft of data processing levels for a representative sonar data processing pipeline:

| Level       | Description | Implementation | Required ancillary data |
| ----------- | ----------- | -------------- | ----------------------- |
| 0 | Raw data in vendor sensor format | Instrument-dependent raw data records |  |
| 1 | Standardized raw data packaged with ancillary information | Zarr encoded data in SONAR-netCDF4 format | Instrument and other deployment metadata |
| 2 | Calibrated acoustic quantities at raw data resolution | Volume backscattering strength (Sv) | Sound speed and absorption computed from associated temperature, conductivity and depth (CTD) measurements |
| 3* | Averaged and/or regridded calibrated acoustic quantities | Mean volume backscattering strength (MVBS) over uniform local depth and time grids with seafloor echoes removed | Geographical location and depth of the sonar hosting platform; platform attitude (pitch, roll, heave) |
| 4** | Acoustically derived biological features | Multi-frequency classification results in the form of nautical area backscattering coefficients (NASC or s<sub>A</sub>) | Biological data from net trawls or underwater camera images; empirical or physics-based acoustic scattering models |

*Can be expanded to include removal of other noise sources, such as electronic or acoustic interference.

**Can be expanded to include outputs from other common analyses of lower-level data, such as acoustically detected animal aggregations, target strength (TS) of individual animals, summary statistics of echogram features, biomass estimation, as well as species-level data labels useful for supervised machine learning developments.

<!-- [40] K. Heaney, B. Martin, J. Miksis-Olds, T. Moore, J. Warren, and M. Ainslie, “ADEON data processing specification. Version 1.0,” Technical report by Applied Ocean Sciences for Prime Contract No. M16PC00003, Apr. 2020. [Online]. Available: https://adeon.unh.edu/sites/default/files/user-uploads/ADEON_Data%20Processing_Specification_FINAL.pdf.
[41] IFREMER, “Processing Levels - Oceanographic Data,” Nov. 01, 2019. http://en.data.ifremer.fr/All-about-data/Data-management/Processing-levels.
[42] NEON, “Data Processing & Publication - Open Data to Understand our Ecosystems.” https://www.neonscience.org/data/about-data/data-processing-publication.
[43] OOI, “Ocean Observatories Initiative: Glossary - Data Terminology,” Ocean Observatories Initiative, Sep. 29, 2015. https://oceanobservatories.org/glossary/#DATATERMS. -->

## Accessibility and scalability in computation

Computational accessibility and scalability are core goals of echopype development. We aim to provide data processing capability for researchers both on their own personal computers and in computing clusters, such as on the cloud. Our data conversion tools currently provide direct read/write interface with both local filesystems and cloud storage. The upcoming development focuses are:
- Memory handling in data conversion and combination in memory-limited scenarios, either due to unreasonably large raw data files or limits in computational resources
- Processing functions to carry raw converted data to higher data processing levels, with support for distributed computing
- Example data processing pipelines and associated container-based configuration for convenient deployment by researchers


## Community engagement

By providing data conversion and processing tools in a fully open-source form, we envision that the echopype code repository can also serve as a forum for community discussions of computing details and questions about data/data handling. To date most questions we encountered from echopype users (via [GitHub issues](https://github.com/OSOceanAcoustics/echopype/issues) or private emails) are about data conversion problems, which are due to flexibility of instrument configuration that produce diverse forms of raw data, particularly for the EK80 echosounder. As we develop further down the data processing chain, we aim to engage users in discussing computational needs and implementation details in a similar manner.


## Companion developments

Echopype is a tool for data conversion and processing based on a standardized data model (see DATA FORMAT (NEED TO ADD LINK)). The developer team is taking a modularized approach to develop companion libraries to cover other needs in integrative analysis of ocean sonar data. Below are planned efforts:
- [Echoshader](https://github.com/OSOceanAcoustics/echoshader): provide interactive visualization widgets, leveraging the [HoloViz](https://holoviz.org/) suite of tools
- [Echopydantic](https://github.com/OSOceanAcoustics/echopydantic): provide convention-related functionalities, such as definitions and compliance check
