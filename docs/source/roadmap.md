# Development Roadmap

This roadmap describes the major areas of `echopype` development currently envisioned, planned or under active work. For information about upcoming releases, especially the next release, see the [Milestones page.](https://github.com/OSOceanAcoustics/echopype/milestones)

`Echopype` is a library aimed at enabling interoperability and scalability in processing ocean sonar data. The current focus and scope are on scientific echosounders that are widely used in fisheries and marine ecological surveys. We envision that `echopype` will provide "building blocks" that can be strung together to construct data processing pipelines to bring raw data files collected by these instruments to "analysis-ready" data products that can be used by a wider community of researchers in the ocean science domain. We also plan for `echopype` to be flexible in accommodating both local and cloud computing environments, such that data processing pipelines can be prototyped locally and scaled up for larger scale processing on the cloud.

## Data processing levels and provenance

A core goal of `echopype` is to enable sonar data analysis pipelines that are agnostic of the instrument that generated the data and span the continuum from raw data files to acoustically derived  biological information. Converting raw data to a standardized form facilitates this goal. This scope and our attention to the processing "chain" expose the need for well-articulated definitions of ["data processing levels"](https://earthdata.nasa.gov/collaborate/open-data-services-and-software/data-information-policy/data-levels/) that can lead directly to broad and highly productive use of the data. However, no such community agreement exists for active acoustic data.

`Echopype` developers are [working towards such definitions](https://github.com/uw-echospace/data-processing-levels/blob/main/from_proposal.md), which will span raw data in vendor formats ("level 0"), standardized (converted) raw data ("level 1"), calibrated acoustic quantities at raw data resolution ("level 2"), and more processed products such as mean volume backscattering strength (MVBS). `Echopype` will implement such data level categorizations as standardized metadata in all its outputs.

In addition to clear data level definitions and metadata, we are identifying [data provenance](https://eos.org/opinions/the-importance-of-data-set-provenance-for-science) and processing information that should be preserved in a standardized form and carried out from one data level to the next. Initial steps already implemented include parameters passed to processing functions and information about the software and functions used. [Other near-term improvements have been identified](https://github.com/OSOceanAcoustics/echopype/issues?q=is%3Aissue+is%3Aopen+provenance), but a more comprehensive approach will require substantial additional developments.

### Standardized raw data

At the core of `echopype` is a data conversion and standardization module that returns an `EchoData` object that allows easy and intuitive access and understanding of ocean sonar data in a netCDF data model. Currently echopype converts raw instrument data files into netCDF4 or Zarr files following the [ICES SONAR-netCDF4 convention](https://github.com/ices-publications/SONAR-netCDF4/) version 1, except for a small number of exceptions that we feel improve the data coherence and efficiency of data access. This conversion step ensures that downstream processing can be developed and executed in an instrument-agnostic manner, which is critical for tackling the currently tedious and labor-intensive data wrangling operations associated with integrative analysis of data from heterogeneous instrument sources.

See [Open converted files](convert) and [Data format](data-format) for more details.

We also would like to explore support for metadata-oriented standards such as [ICES AcMeta](https://github.com/ices-publications/AcMeta).

:::{important}
Release v0.6.0 include breaking changes to the structure of the core SONAR-netCDF4 data model, which impacts both content of the `EchoData` object and the converted .nc/.zarr files. These changes were necessary to enhance the compliance of echopype-generated data with SONAR-netCDF4 ver.1. Data converted using echopype v0.5.x continue to be supported beyond v0.6.0 and can be opened via `open_converted` into the new format.
:::

### Derived, higher-level products

`Echopype` already incorporates [functions to generate higher-level products](process), including calibrated (S<sub>v</sub>, TS), noise and reduced-binned (MVBS) products. Other functionalities are in planning for near-term implementation, including enhancement of processed data products (e.g., [#662](https://github.com/OSOceanAcoustics/echopype/issues/662)), integration of platform location information into S<sub>V</sub>, and seafloor detection and removal, etc.


## Scalability in computation

Computational scalability is a core goal of `echopype` development. We aim to provide scalable data processing capability for researchers both on their own personal computers and in computing clusters, such as on the cloud. Our data conversion tools currently provide direct read/write interface with both local filesystems and cloud storage. However, some computational and memory challenges remain:

- Memory handling in data conversion and combination in memory-limited scenarios, either due to unusually large raw data files or limits in computational resources ([#407](https://github.com/OSOceanAcoustics/echopype/issues/407))
- Use of regular (dense) arrays in cases where the multi-frequency backscatter data are highly sparse ([#489](https://github.com/OSOceanAcoustics/echopype/issues/489))
- Limited support for distributed computing throughout the data processing chain using Dask delayed and other approaches ([#408](https://github.com/OSOceanAcoustics/echopype/issues/408))

## Companion developments

Echopype is a tool for data conversion and processing based on standardized data models, particularly [SONAR-netCDF4](https://github.com/ices-publications/SONAR-netCDF4/) and the broader [netCDF data model](https://www.unidata.ucar.edu/software/netcdf/). The developer team is taking a modularized approach to develop companion libraries to cover other needs in integrative analysis of ocean sonar data. Below are planned efforts:

- [Echoshader](https://github.com/OSOceanAcoustics/echoshader): provide interactive visualization widgets, leveraging the [HoloViz](https://holoviz.org/) suite of tools
- [Echopydantic](https://github.com/OSOceanAcoustics/echopydantic): provide convention-related functionalities, such as definitions and compliance checking
