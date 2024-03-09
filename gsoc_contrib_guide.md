<img src="docs/source/images/GSoC-logo-horizontal.svg" alt="Google Summer of Code logo" width="450" style="padding-right: 50px; vertical-align: middle">

# GSoC 2024 Contributor's Guide

In collaboration with the [US Integrated Ocean Observing System (IOOS)](https://ioos.noaa.gov/), the Echopype team aims to recruit talented [Google Summer of Code (GSoC)](https://summerofcode.withgoogle.com/)
participants to help us upgrade the robustness and scalability of the Echopype package. The project work will focus on making the Echopype testing suite more robust by overhauling its Continuous Integration (CI) mechanisms and tackling distributed computing bottlenecks in processing irregularly spaced echosounder data across computing agents.


## Background

Echosounders, or high-frequency ocean sonar systems, are [the workhorse to study life in the ocean](https://storymaps.arcgis.com/stories/e245977def474bdba60952f30576908f). They provide continuous observations of fish and zooplankton by transmitting sounds and analyzing the echoes bounced off these animals, just like how medical ultrasound images the interior of the human body. In recent years echosounders are widely deployed on ships, autonomous vehicles, or moorings, bringing in significant volumes of data that allow scientists to study the rapidly changing marine ecosystems.

The water column sonar data we work with in Echopype come from different echosounder instruments and are stored in different instrument- or manufacturer-specific binary formats. These binary formats are difficult to work with directly and does not allow for efficient processing, especially when on cloud computing platforms.

Echopype addresses these challenges by:

- converting the raw data into a standardized structure following an interoperable netCDF data model, which can be serialized into
[netCDF](https://www.unidata.ucar.edu/software/netcdf/) or [Zarr](https://zarr.readthedocs.io/en/stable/) formats
- provide other downstream functions that calibrate and process these water column sonar datasets

See the [echopype documentation](https://echopype.readthedocs.io/en/stable/why.html) and our [preprint on arXiv](https://arxiv.org/abs/2111.00187) for more detail.



## Goals

Our goals for GSoC 2024 are:
1. Utilize GitHub release assets for hosting test files to make the Echopype continuous integration mechanisms more robust
2. Increase test coverage for foundational data conversion functions
3. Improve distributed computing performance for major processing functions on large (100s of GB) data sets


## Getting started

To get started on the project, be sure to check out our [contributor's guide](https://echopype.readthedocs.io/en/stable/contributing.html) and make sure:
1. You have the dev environment ready to go
2. You can run the notebooks in the [echopype-examples repo](https://github.com/OSOceanAcoustics/echopype-examples) which we use to host example notebooks of using the echopype package. Note the OOI notebook likely will not run on Binder because of memory limitation, but it should run on your local machine.


## Brainstorm with us

Existing relevant issues are labeled with https://github.com/github/docs/labels/GSoC24. Feel free to comment on the issues directly if you have questions or ideas you would like to discuss.

We encourage you as a GSoC 2024 participant to propose your own original project ideas by [creating a new issue](https://github.com/OSOceanAcoustics/echopype/issues/new?assignees=&labels=GSoC24&projects=&template=gsoc24.yml&title=%5BGSoC24%5D+...) in this repo.




## GSoC proposal

### Preparation
Before you submit your proposal, we'd love to see you exchange your ideas or ask questions on GitHub, either directly on each of the https://github.com/github/docs/labels/GSoC24 label or via the [issue template](https://github.com/OSOceanAcoustics/echopype/issues/new?assignees=&labels=GSoC24&projects=&template=gsoc24.yml&title=%5BGSoC24%5D+...).

### Qualifications
The qualifications we are looking for include:
- Experience with Python and object-oriented programming
- Experience or demonstrated passion to learn Python libraries such as [Xarray](https://docs.xarray.dev/en/stable/), [Dask](https://www.dask.org/), [Zarr](https://zarr.readthedocs.io/en/stable/), [fsspec](https://filesystem-spec.readthedocs.io/en/latest/), [pooch](https://www.fatiando.org/pooch/latest/), all in the scientific Python software ecosystem
- Interests in working with oceanographic, acoustic or geospatial data

### Proposal template

Please [sign up as a GSoC participant](https://summerofcode.withgoogle.com/get-started/) and build your proposal based on the [proposal template](https://github.com/ioos/gsoc/blob/main/proposal-template.md) provided by IOOS (the Integrated Ocean Observing Systems).

We are happy to review your proposal draft. Feel free to share something to us (@leewujung and @valentina-s) via email. Find our emails via websites on our GitHub profile.

### Tips for a successful proposal
- Submit PR(s) that show rough prototypes of what you plan to do to go with your proposal. The prototypes do not have to be completely functional, but we would like to see that you have thought about the goals and propose solutions that are on the right track.
- Reference specific issues under the https://github.com/github/docs/labels/GSoC24 label and explain how you plan to address them with the prototype you have created. Examples include (but not limited to) benchmarking report or potential solutions you find on the internet related to existing issues.
- Demonstrate that either you are already familiar with the Python libraries mentioned above, or you have taken concrete steps toward ensuring that you will be able to learn and use them during the GSoC 2024 period.


## Questions?

For project-related question, feel free to [raise an issue](https://github.com/OSOceanAcoustics/echopype/issues/new?assignees=&labels=GSoC24&projects=&template=gsoc24.yml&title=%5BGSoC24%5D+...).

Having more questions about being a GSoC mentor or participant? Check out the [GSoC mentor & participant guides](https://google.github.io/gsocguides/).


## Mentors
The GSoC 2024 mentor team includes Wu-Jung Lee (@leewujung) and Valentina Staneva (@valentina-s), who co-lead the [Echospace research group](https://uw-echospace.github.io/) at the University of Washington, Seattle.
