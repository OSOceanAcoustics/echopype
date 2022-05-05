# Development Roadmap

I SUGGEST WE COMBINE ELEMENTS OF THE TWO SECTIONS BELOW, AND USE IT AS AN INTRODUCTION TO THIS PAGE.
THE RESULTING TEXT SHOULD BE SHORTER THAN THE COMBINATION OF THOSE TWO SECTIONS.

Mention and point to our use of Milestones in GitHub, https://github.com/OSOceanAcoustics/echopype/milestones, for an idea of what features we're working on for upcoming releases

## Scope and goals

Echopype is a library aimed at enabling interoperability and scalability in processing ocean sonar data. The focus and scope right now are on scientific echosounders that are widely used in fishery and marine ecological surveys. We envision that echopype will provide "building blocks" that can be strung together to construct data processing pipelines to bring raw data files collected by these instruments to "analysis-ready" data products that can be used by a wider community of researchers in the ocean science domain. We also plan for echopype to be flexible in accommodating both local and cloud computing environments, such that data processing pipelines can be prototyped locally and scaled up for larger scale processing on the cloud.

## Community engagement

By providing data conversion and processing tools in a fully open-source form, we envision that the echopype code repository can also serve as a forum for community discussions of computing details and questions about data/data handling. To date most questions we encountered from echopype users (via [GitHub issues](https://github.com/OSOceanAcoustics/echopype/issues) or private emails) are about data conversion problems, which are due to flexibility of instrument configuration that produce diverse forms of raw data, particularly for the EK80 echosounder. As we develop further down the data processing chain, we aim to engage users in discussing computational needs and implementation details in a similar manner.



## Data processing levels and provenance

A core goal of `echopype` is to enable sonar data analysis pipelines that are agnostic of the instrument origin of the data and span fthe continuum from raw data files to acoustically derived  biological information. Converting raw data to a standardized form facilitates this capability. This scope and our attention to the processing "chain" expose the need for well-articulated definitions of ["data processing levels"](https://earthdata.nasa.gov/collaborate/open-data-services-and-software/data-information-policy/data-levels/) can lead directly to broad and highly productive use of the data. However, no such community agreement exists for active acoustic data. `echopype` developers are [working towards such definitions](https://github.com/uw-echospace/data-processing-levels/blob/main/from_proposal.md), which will span raw data in vendor formats ("level 0"), standardized raw data ("level 1"), calibrated acoustic quantities at raw data resolution ("level 2"), and more processed products such as mean volume backscattering strength. `echopype` will implement such data level categorizations as standardized metadata in all its outputs. In addition to clear data level definitions and metadata, we are identifying [data provenance](https://eos.org/opinions/the-importance-of-data-set-provenance-for-science) and processing information that should be preserved in a standardized form and carried out from one data level to the next. Initial steps already implemented include parameters passed to processing functions and information about the software and functions used. [Other near-term improvements have been identified](https://github.com/OSOceanAcoustics/echopype/issues?q=is%3Aissue+is%3Aopen+provenance), but a more comprehensive approach will require substantial additional developments.

## Standardized raw data

I HAVEN'T REWORKED THIS YET. ONE THING TO CONSIDER IS ADDING MENTION OF SUPPORT FOR METADATA STANDARDS, ESPECIALLY AcMeta

At the core of echopype is a data conversion and standardization module that returns an `EchoData` object that allows easy and intuitive access and understanding of ocean sonar data in a netCDF data model. Currently echopype converts raw data files from the instruments into `.nc/.zarr` files following the convention [SONAR-netCDF4](https://github.com/ices-publications/SONAR-netCDF4/) ver.1, except for a small number of exceptions that we feel improve the data coherence and efficiency of data access. This conversion step ensures that downstream processing can be developed and executed in an instrument-agnostic manner, which is critical for tackling the currently tedious and labor-intensive data wrangling operations associated with integrative analysis of data from heterogeneous instrument sources.

:::{important}
The upcoming release v0.6.0 will include breaking changes to the structure of the core netCDF data model, which impacts both content of the `EchoData` object and the converted .nc/.zarr files. These changes were necessary to enhance the compliance of echopype-generated data with SONAR-netCDF4 ver.1. Data converted using echopype v0.5.x continue to be supported beyond v0.6.0 and can be opened via `open_converted` into the new format.
:::

FIX THIS AND LINK: See :ref:`convert` and :ref:`data-format` for more detail.

## Derived, higher-level products

`echopype` [already includes functions to generate](process) higher-level products, including calibrated (`Sv`, `TS`), noise and reduced-binned (`MVBS`) products. We are currently reviewing and enhancing these products ([#591](https://github.com/OSOceanAcoustics/echopype/issues/591)), and many of these improvements will be included in the upcoming version 0.6.0. Others are in planning for near-term implementation ([#662](https://github.com/OSOceanAcoustics/echopype/issues/662)).
In addition, other needed functionality will be added in the future, including integration of platform location information into `Sv` and seafloor detection and removal.

## Scalability in computation

Computational scalability is a core goal of `echopype` development. We aim to provide scalable data processing capability for researchers both on their own personal computers and in computing clusters, such as on the cloud. Our data conversion tools currently provide direct read/write interface with both local filesystems and cloud storage. However, some computational and memory challenges remain:

- Memory handling in data conversion and combination in memory-limited scenarios, either due to unusually large raw data files or limits in computational resources ([#407](https://github.com/OSOceanAcoustics/echopype/issues/407))
- Use of regular (dense) arrays in cases where the multi-frequency backscatter data are highly sparse ([#489](https://github.com/OSOceanAcoustics/echopype/issues/489))
- Limited support for distributed computing throughout the data processing chain using Dask delayed and other approaches ([#408](https://github.com/OSOceanAcoustics/echopype/issues/408))

## Companion developments

Echopype is a tool for data conversion and processing based on standardized data models, particularly [SONAR-netCDF4](https://github.com/ices-publications/SONAR-netCDF4/) and the broader [netCDF data model](https://www.unidata.ucar.edu/software/netcdf/). The developer team is taking a modularized approach to develop companion libraries to cover other needs in integrative analysis of ocean sonar data. Below are planned efforts:

- [Echoshader](https://github.com/OSOceanAcoustics/echoshader): provide interactive visualization widgets, leveraging the [HoloViz](https://holoviz.org/) suite of tools
- [Echopydantic](https://github.com/OSOceanAcoustics/echopydantic): provide convention-related functionalities, such as definitions and compliance checking
