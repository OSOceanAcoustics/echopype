(contrib:roadmap)=
# Development roadmap



(contrib:roadmap_dependency)=
## Package dependency
Echopype depends on many libraries in the scientific Python ecosystem, and therefore need to keep up with their updates. The two big ticket items that we hope to resolve soon are:
- Upgrade to use Numpy version 2
- Upgrade to use Zarr version 3

See the [`requirements.txt`](https://github.com/OSOceanAcoustics/echopype/blob/main/requirements.txt) file for the current pinned versions. We aim to remove the specification of maximum version (e.g., zarr<3) whenever possible.



(contrib:roadmap_convert)=
## Data conversion and standardization
Echopype currently support converting files from [a few echosounder models](convert-sonar_types) to netCDF or Zarr files following [a modified version of the ICES SONAR-netCDF4 convention](data-format:sonarnetcdf4-adaptation). As the core data representation stabilizes, the next steps are to:
- Enhance adherence to community conventions of metadata and processed data, such as:
  - [ICES AcMeta](https://github.com/ices-publications/AcMeta)
  - The new Gridded group introduced in SONAR-netCDF4 v2.0
  - [Australia IMOS SOOP-BA conventions](https://imos.org.au/fileadmin/user_upload/shared/SOOP/BASOOP/SOOP-BA_NetCDF_Conventions_Version_2.2.pdf)
- Add support for data from other echosounder models, including:
  - Simrad EK/BI500 ([unfinished PR](https://github.com/OSOceanAcoustics/echopype/pull/1252))
  - Biosonics [DT4](https://www.biosonicsinc.com/download/dt4-file-format-specification/) files



(contrib:roadmap_algorithms)=
## Rule-based algorithms
We plan to add more common rule-based (i.e. non-ML) echosounder data analysis algorithms into Echopype. The high priority items are:
- Full support for broadband processing (in the `calibrate` subpackage)
  - currently `calibrate.compute_Sv` supports generating band-averaged Sv for broadband data
- Noise removal (in the `clean` subpackage)
  - currently `clean` contains a handful of noise removal functions from {cite:t}`Ryan2015`.
  - there are many others that can be useful, including [a more efficient algorithm for detecting transient noise](https://github.com/open-ocean-sounding/echopy/blob/96bb25f83490529a5373aeb3b423f03c9605f7a6/echopy/processing/mask_transient.py#L87C5-L87C13)
- Regridding (in the `commongrid` subpackage)
  - currently `commongrid` contains functions produce MVBS and NASC
  - need a function to [regrid Sv with integrated output preserved](https://github.com/OSOceanAcoustics/echopype/issues/726)
  - need a function to [regrid a mask to a different grid](https://support.echoview.com/WebHelp/Reference/Algorithms/Operators/#match_geometry_)
- Bottom detection (in the `mask` subpackage)
- Swarm or school detection (in the `mask` subpackage)
- Single target detection (in the `mask` subpackage)
- Calibration and other utility functions
  - standard target calibration
  - updated estimates of sound speed and absorption coefficients
  - water column profile-based (rather than water column average-based) Sv and TS computation

:::{note}
Echopype is designed to be used as a programmatic API and not for manual editing. For interactive visualization, check out [Echoshader](https://github.com/OSOceanAcoustics/echoshader).
:::




## Data processing levels

Parallel to Echopype, the Echostack team is also working on defining a set of "data processing levels" for echosounder data: see the [proposal in Echolevels](https://echolevels.readthedocs.io/en/latest/levels_proposed.html). Clearly defined data processing levels are conducive to broader data usage. Many Echopype functions generate prototype [data provenance](https://eos.org/opinions/the-importance-of-data-set-provenance-for-science) and processing level information as data attributes. **However, both the proposed data processing levels and the implementation require revision. Please [chime in with any input or questions via GitHub issues](https://github.com/uw-echospace/data-processing-levels/issues/new)!**





## Echostack: Companion developments

Echopype focuses on data standardization, aggregations, and processing for building efficient and scalable data workflow. To address other needs in integrative analysis of echosounder data, check out the following companion libraries in the Echostack:

- [Echoregions](https://github.com/OSOceanAcoustics/echoregions): Interface with echogram interpretation masks from physics-based or data-driven methods
- [Echoshader](https://github.com/OSOceanAcoustics/echoshader): Interactive visualization widgets leveraging the [HoloViz](https://holoviz.org/) suite of tools
- [Echopop](https://github.com/OSOceanAcoustics/echopop): Incorporate trawl biological data and scattering models for biomass estimation, currently focused on Pacific hake
- [Echodataflow](https://github.com/OSOceanAcoustics/echodataflow): Orchestrate workflow on the cloud or local platforms
- [Echolevels](https://github.com/OSOceanAcoustics/echolevels): Proposed specifications of echosounder data processing levels
<!-- - [Echopydantic](https://github.com/OSOceanAcoustics/echopydantic): provide convention-related functionalities, such as definitions and compliance checking -->






<!-- ## Computational scalability --- SIMPLIFY THIS!!!!

Computational scalability is a core goal of Echopype development. We aim to provide scalable data processing capability for researchers both on their own personal computer and on computing clusters. The Echopype data conversion tools provide direct read/write interface with both local filesystems and cloud storage, and all downstream data processing functions also natively interface with both local and cloud resources through the combination of the Zarr, Xarray, Dask, and related libraries. However, we have found that the often irregular spacing and structure of echosounder data in time and space can impose substantial computational bottleneck and require custom optimization beyond stock Xarray functions to parallelize efficiently across computing agents. With a few important memory issues during data conversion resolved (see [v0.8.0 release notes](https://echopype.readthedocs.io/en/stable/whats-new.html#v0-8-0-2023-august-27)), going forward we plan to:
- Benchmark data processing functions against diverse datasets of different volume (100s of GB to TB) and spatiotemporal features that can cause unintended memory expansion during computation
- Leverage Dask delayed approaches and experiment with different Zarr chunking schemes to resolve computational bottlenecks -->
