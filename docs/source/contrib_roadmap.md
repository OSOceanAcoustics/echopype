# Development roadmap


## Overview
Echopype is a library aimed at enabling interoperability and scalability in processing water column sonar data.
The current focus and scope are on scientific echosounders widely used in fisheries and marine ecological surveys. 
We aim for Echopype to provide "building blocks" that can be strung together to construct data processing pipelines to bring raw instrument-generated files to "analysis-ready" data products that can be easily leveraged by further downstream processing routines.
To this end, the Echopype workflow focuses first on standardizing data to the widely supported netCDF data model, and based on the standardized data build computational routines by leveraging open-source scientific Python libraries, especially those in [the Pandata stack](https://github.com/panstacks/pandata?tab=readme-ov-file)

See the paper [Interoperable and scalable echosounder data processing with Echopype](https://doi.org/10.1093/icesjms/fsae133) for more details on the design philosophy.

![workflow](./images/workflow_v2.png)




## Development priorities

### Dependency resolution
Echopype depends on many libraries in the scientific Python ecosystem, and therefore need to keep up with their updates. The two big ticket items that we hope to resolve soon are:
- upgrade to use Numpy version 2
- upgrade to use Zarr version 3

See the [`requirements.txt`](https://github.com/OSOceanAcoustics/echopype/blob/main/requirements.txt) file for the current pinned versions. Ideally we would like to unpin the maximum version when possible.


### Data conversion and standardization



### Rule-based algorithms
We plan to add more rule-based algorithms that are commonly used to analyze echosounder data into Echopype. The high priority categories are:
- full support for broadband processing
  - currently `calibrate.compute_Sv` supports generating band-averaged Sv for broadband data
- noise removal
  - currently the [`clean`](https://echopype.readthedocs.io/en/stable/api.html#module-echopype.clean) subpackage supports a handful of noise removal functions
- bottom detection
- swarm or school detection
- single target detection


:::{note}
Echopype is designed to be used as a programmatic API and _not_ for manual editing. For interactive visualization, check out [Echoshader](https://github.com/OSOceanAcoustics/echoshader.
:::