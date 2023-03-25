# Proposed Echosounder Data Proccesing Levels (DRAFT)

The decades-long experience from the satellite remote sensing community has shown that a set of robust and well-articulated definitions of "data processing levels" [1]–[5] can lead directly to broad and highly productive use of data. Processing level designations also provide important context for data interpretation [6]. However, no such community agreement exists for active acoustic data. The ambiguity associated with the interoperability and inter-comparability of processed sonar data products has hindered efficient collaboration and integrative use of the rapidly growing data archive across research institutions and agencies.

The `echopype` team is developing a clearly defined progression of data processing levels for active ocean sonar data. The development leverages the collective experience from remote sensing and large-scale, long-term ocean and ecological observatories [7]–[10]. Data processing functions in `echopype` are clearly associated with specific Processing Level inputs and outputs, and when appropriate, will generate a `processing_level` dataset global attribute with entries such as "Level 1A", "Level 2B", "Level 4", etc.


## Processing Levels and Sub-levels

### Level 0 (L0)

**Description:**  Raw data in vendor sensor format.

- raw binary files. Associated metadata may be found in external files.

### Level 1 (L1)

**Description:** Raw data packaged with ancillary information and converted and standardized to an open convention and standard data formats. May be distributed in the following two forms: 
  - as sets of individual converted files as originally segmented into arbitrary time ranges during sensor file creation, or
  - compiled into larger granules corresponding to logical deployment intervals.

- **L1A**: Raw L0 data converted to a standardized, open format with geographic coordinates (latitude & longitude) processed and included. Includes other ancillary information extracted from sensor-generated L0 data or other external sources. May include environmental information such as temperature, salinity and pressure. Use of the SONAR-netcDF4 v1 convention is strongly recommended.
- **L1B**: L1A data with quality-control steps applied, such as time-coordinate corrections that enforce strictly increasing, non-duplicate timestamps.

### Level 2 (L2)

**Description:** Calibrated acoustic quantities at raw data resolution, with spatial coordinates included (latitude, longitude and depth)

- **L2A**: Volume backscattering strength (`Sv`) with interpolated latitude, longitude and depth coordinates. May incorporate addition information, such as  split beam angle
- **L2B**: `Sv` L2A data with noise removal or other data filtering applied, including seafloor bottom removal.

### Level 3 (L3)

**Description:** Calibrated acoustic quantities regridded or aggregated to a common grid across channels. May include noise removal or other filtering.

- `Sv` resampled to a common, high-resolution grid across channels
- Mean Volume Backscattering Strength (`MVBS`)
- `Sv` frequency difference across two channels resampled to a common grid
- **L3A**: The above variables computed on L2A data
- **L3B**: The above variables computed on L2B (filtered) data

### Level 4 (L4)

**Description:** Acoustically derived biological features, involving further processing of L3 data that may include data reduction or incorporation of external sources of data.

- Nautical Area Backscattering Coefficients (`NASC`)
- Summary statistics of echogram features (center_of_mass, dispersion, etc)
- Taxon or species-level data labels (classification). May originate from a variety of methods, including frequency difference thresholds.
- Biomass estimation


## References

- [1] Parkinson, C. L., A. Ward, and M. D. King (eds.). 2006. Earth science reference handbook: A guide to NASA’s Earth Science Program and Earth Observing Satellite Missions. National Aeronautics and Space Administration. https://atrain.nasa.gov/publications/2006ReferenceHandbook.pdf
- [2] NASA. 2021. Data Processing Levels | Earthdata. Last viewed Mar. 24, 2023. https://earthdata.nasa.gov/collaborate/open-data-services-and-software/data-information-policy/data-levels/
- [3] Ramapriyan, H. K., and P. J. T. Leonard. 2021. Data Product Development Guide (DPDG) for Data Producers version1.1. NASA Earth Science Data and Information System Standards Office, 21 October 2021. https://doi.org/10.5067/DOC/ESO/RFC-041VERSION1
- [4] Robinson, I. 2006. Satellite Measurements for Operational Ocean Models, pp. 147-189. In: Chassignet, E.P. and Verron, J. (eds). Ocean Weather Forecasting: An Integrated View of Oceanography. Springer, New York, NY. https://doi.org/10.1007/1-4020-4028-8_6
- [5] Weaver, R. 2014. Processing Levels, pp. 517-520. In: Njoku, E.G. (ed). Encyclopedia of Remote  Sensing. Encyclopedia of Earth Sciences Series. Springer, New York, NY.  https://doi.org/10.1007/978-0-387-36699-9_36
- [6] Hills, D. J., R. R. Downs, R. Duerr, J. C. Goldstein, M. A. Parsons, and H. K. Ramapriyan. 2015. The importance of data set provenance for science. Eos, 96, Published on 4 December 2015. https://doi.org/10.1029/2015EO040557
- [7] Heaney, K., B. Martin, J. Miksis-Olds, T. Moore, J. Warren, and M. Ainslie. 2020. ADEON data processing specification. Version 1.0. Technical report by Applied Ocean Sciences for Prime Contract No. M16PC00003, Apr. 2020. https://adeon.unh.edu/sites/default/files/user-uploads/ADEON_Data%20Processing_Specification_FINAL.pdf
- [8] IFREMER. 2019. Processing Levels - Oceanographic Data. Last viewed Mar. 24, 2023. http://en.data.ifremer.fr/All-about-data/Data-management/Processing-levels
- [9] NEON. 2023 Data Processing | NSF NEON | Open Data to Understand our Ecosystems. Last viewed Mar. 24, 2023. https://www.neonscience.org/data-samples/data-management/data-processing
- [10] OOI. 2023. Ocean Observatories Initiative: Glossary - Data Terminology. Ocean Observatories Initiative. Last viewed Mar. 24, 2023. https://oceanobservatories.org/glossary/#DATATERMS
