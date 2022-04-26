.. _roadmap:

# Development Roadmap

## Scope and goals

Echopype is a library aimed at enabling interoperability and scalability in processing
ocean sonar data.
The focus and scope right now are on scientific echosounders that are widely used in fishery
and marine ecological surveys.
We envision that echopype will provide ''building blocks'' that can be strung together
to construct data processing pipelines to bring raw data files collected by these
instruments to ''analysis-ready'' data products that can be used by a wider community
of researchers in the ocean science domain.
We also plan for echopype to be flexible in accommodating both local and cloud computing
environments, such that data processing pipelines can be prototyped locally and scaled up
for larger scale processing on the cloud.


## Data standardization

At the core of echopype is a data conversion and standardization module that
returns an `EchoData` object that allows easy and intuitive access and understanding
of ocean sonar data following the netCDF data model.
This conversion step ensures that downstream processing can be developed and executed
in an instrument-agnostic manner, which is critical for tackling the currently tedious
and labor-intensive data wrangling operations associated with integrative analysis of
data from heterogeneous instrument sources.

FIX THIS: See :ref:`convert` and :ref:`data-format` for more detail.


## Data processing levels

- put forth the idea that we will define data processing levels for carrying data
  through the pipeline
- a simple table giving a taste of how these data processing levels look like.


## Computing scalability

- bring out that we want our computation to be scalable to handle the big data
- mention cloud-native pipeline and configuration


## Community engagement

- goal is to provide something that current do not exist in the community in a
  fully open-source form
- repo can serve as a forum for discussion of computing details and questions
  about data


## Companion developments

- echopype is only for processing pipeline
- echoshader: visualization
- echopydantic: convention check
- echoregions: parsing labels for ML developments
