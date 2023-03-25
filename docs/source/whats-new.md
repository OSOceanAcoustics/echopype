What's new
==========

See [GitHub releases page](https://github.com/OSOceanAcoustics/echopype/releases) for the complete history.


# v0.7.0 (2023 March 25)

## Overview

This release includes new features to interface with Echoview ECS files for computing Sv, reorganization of computing functions into new subpackages, addition of data processing level attributes to data products, and other improvements and bug fixes.

## New features and major changes
- Allow using ECS for calibrating Simrad echosounders (#996, #1004)
  - This functionality is in a **beta** testing stage
  - Details of implementation may change and bugs are possible
  - Expand ECS parser to accept frequency-dependent values in EK80 ECS files
- Overhaul `env_params` to ensure correct intake for calibration (#985)
  - Now allows using `env_params` entries that are xr.DataArrays
- Move functions previously in `preprocess` subpackage to new subpackages (#993). Calling these functions from `preprocess` is deprecated and will be removed in v0.7.1.
  - `clean`: `remove_noise`, `estimate_noise`
  - `commongrid`: `compute_MVBS`, `compute_MVBS_index_binning`
- Add `commongrid.compute_NASC` (#1005)
  - The current implementation uses brute force looping for mean Sv computation, this will be refactored and optimized together with other functions requiring the same pattern in an upcoming release
- Add global attributes for data processing levels (#1001).
  - This functionality is in a **beta** testing stage
  - See [data processing level specifications](https://echopype.readthedocs.io/en/stable/processing-levels.html) for functions and conditions under which such attributes are added
- Expand `mask.apply_mask` to handle multi-channel Sv datasets (#1010)
- Standardize sonar metadata for EK80 data (#992)
    - `sonar_serial_number` is now an empty global attribute, no longer a variable, as in the EK60 case
    - `transducer_name`, `transducer_serial_number`, `transceiver_serial_number` based on parser parameter `transducer_name`, `transducer_serial_number`, and `serial_number`, respectively

## Bug fixes
- Fix scaling bug for `beamwidth_alongship` and `beamwidth_athwartship` from CW-based parameters to values corresponding to center frequency of broadband transmit signals (#998)

## Tests
- Add more comprehensive tests for `add_location` (#1000)
- Add test for splitbeam angle `ek80_CW_power` case (#994)
- Add unit and integration tests for `env_params` intake for calibration (#985)





# v0.6.4.1 (2023 March 15)

## Overview

This is a minor release that includes a small bug fix and an enhancement, which removes excessive warnings due to changes in v0.6.4.

## Bug fix
- Fix a bug that prevented passing in `env_params` for EK60 calibration (#987)

## Enhancement
- Handling non-positive values that causes log10 warnings on EK calibration (#986)





# v0.6.4 (2023 March 13)

## Overview

This is a release that includes important performance enhancements that allow user-provided calibration parameters for broadband and narrowband calibration, new functionalities to compute and attach split-beam angles to calibrated Sv dataset, perform frequency-differencing and masking, as well as a number of bug fixes and other improvements.

## New features
- Allow passing in `cal_params` as a dictionary for narrowband and broadband calibration (#955)
- Add default chunk encoding for zarr output (#939)
- Add `add_splitbeam_angle` function to the `consolidate` subpackage (#916, #971)
- Add a new `mask` subpackage
  - Add `apply_mask` function to the `mask` subpackage (#912)
  - Add `frequency-differencing` function to the `mask` subpackage (#901)
- Allow selection of a subset of channels when combining multiple echodata objects (#892)
  - This is done via the added `channel_selection` input argument to `combine_echodata`
- Add default consolidated flag for `echodata.to_zarr` (#855)
- Generalize and improve efficiency of `compute_MVBS` (#878)
  - Allow `echo_range` that vary with `ping_time`
  - Allow `Sv` data that are dask arrays
- Add `.nbytes` to obtain that data size of an `echodata` object (#874)
- Add new data variables from the raw data files to facilitate EK80 calibration (#944)
  - transceiver type, transmit impedance, receive impedance, receiver sampling frequency

## Enhancements and other changes
- Unify the order of dimensions of `echo_range` across all sonar models (#968)
- Create and use the default echopype home directory (#896, #954)
  - This default directory is at `~/.echopype`
- Refactor the `calibrate` subpackage (#904)
  - routines for `env_params` and `cal_params` intake and range computation are now in different modules rather than methods in each classes
  - Tidy up `cal_params` related routines (#953)
  - Revise `env_params` related routines (#952)
- Improving pulse compressed broadband Sv computation and echo range computation (#944)
  - The resulting values are tested against pyEcholab and Echoview outputs
  - Note there is unresolved discrepancy between pyEcholab/Echopype outputs with Echoview outputs for the first section of Sv values
- Change input argument names for `open_raw` (#962)
  - Change `offload_to_zarr` to `use_swap`
  - Change `max_zarr_mb` to `max_mb`

## Bug fixes
- Handling of provenance attributes in apply-mask and add-depth, especially for testing (#930)
- Fix bugs in combining multiple echodata objects (the `combine_echodata` function)
  - Fix meta_source_filenames bug and enable (meta)source_filenames appending of path and list (#913)
- Fix bugs in `env_params` intake for calibration (#952)

## Infrastructure
- Update docker build for arm64 mac silicon chips (#964)
- Fix windows github actions workflow for utils module (#947)
- Remove mamba dependency (#946)
- Fix pre-commit ci and update RTD config for jupyter-book (#934)





# v0.6.3 (2022 October 15)

## Overview

This is a minor release that includes an important performance enhancement for combining large volumes of data residing in individual files into a single entity, a number of bug fixes, and other smaller improvements.

## New features
- Overhaul `combine_echodata` function
  - Allow combine a large number of `EchoData` objects exceeding memory limits (#808, #824, #830)
  - Remove reversed time check from combine_echodata (#835)
  - Add minimal ZarrCombine test (#826)
  - Order the channel coordinate in file conversion to ensure consistent combination across files (#818)
  - Revise outdated data combination behavior (#797, #799)
  - Clean up all coordinate and attribute details under `combine_echodata` function (#848, #849)
- Track provanance for filenames of raw data files and auxiliary files
  - Propagate `xml_path` as `meta_source_filenames` to combined echodata (#814, #852)
  - Standardize handling of `source_filenames` in individual and combined echodata as well as downstream datasets (#804, #806)

## Under the hood enhancements
- Clean up functions for setting encoding in the converted files (#851)
- Add `requests` and `aiohttp` to dependency (#844)
- Pin `netcdf4` to be <1.6 for pypi package due to ongoing `netcdf-c` problem (#843)
- Write `Parsed2Zarr` generated files to `temp_echopype_output/parsed2zarr_temp_files` (#832)
- Change `isel` to `sel` to fix `compute_Sv` to allow working with dask array (#828)
- Add `open_raw(offload_to_zarr=True)` integration tests (#794)

## Bug fixes
- Fix regression bug with interpolating environmental variables to `ping_time` grid (#837, #856)
- Fix WindowsPath error with compute_Sv when run on Windows (#829)
- Fix logic problem in `open_raw(offload_to_zarr=True)` that sometimes cause problems (#794, #853)

## Infrastructure
- Update CI to use micromamba (#805)
- Fix version string on CI (#804, #820)
- Rename `ci.yaml` to `build.yaml` for clarity (#807)



# v0.6.2 (2022 August 13)

## Overview

This is a minor release that includes a few new features and memory efficiency-related changes that make echopype better.

## New features
- Add a new subpackage `consolidate` that contains functions to consolidate data across the calibrated Sv dataset and the corresponding raw-converted file
- Add function `consolidate.add_location` to interpolate location to calibrated dataset (#749)
- Add function `consolidate.add_depth` to convert `range_meter` to `depth` with information on transducer tilt and depth (#738)
- Move function that swaps the channel with frequency dimension (now as `consolidate.swap_dims_channel_frequency`) (#738)
- Add new functionality to allow control of logging outputs (#772)

## Under the hood enhancements
- Improve memory usage while converting files that require significant NaN-padding and previously would incur very large memory expansion (#774)
  - This is achieved by directly writing variables that may incur a large memory expansion into a temporary zarr store
  - Beta function that will benefit from user feedback
- Overhaul access pattern for EchoData (#762)
  - Remove previous access pattern for different groups in the raw-converted file
  - Starting from this release all groups are accessed with `echodata["GROUP_PATH"]`, e.g., `echodata["Platform"]`, `echodata["Sonar/Beam_group1"]`, etc.
- Make long_name in ds_power for EK80 consistent with other sonar model (#771)
- Modify `set_beam()` so it returns a list (#780)
- Change the order in `_save_groups_to_file` so the order of groups is preserved when opening a converted netcdf file (#779)
- Remove the user option to select NMEA sentences in `open_raw` to ensure raw data is preserved (#778)

## Infrastructure
- Update CI set up to avoid exceeding GitHub actions memory limitation (#761)
  - decrease the number of workers from 4 to 2







# v0.6.1 (2022 July 7)

## Overview

This is a minor release that includes important bug fixes, a number of new features, and some leftover data format changes specific to parsed AD2CP data format.

## Bug fixes
- We use `datatree` under the hood for the `EchoData` object, but `datatree` v0.0.4 had a bug in accessing subgroups in netCDF data model in Windows OS. This was fixed in `datatree` v0.0.6, and hence we updated our dependency and made other associated changes to support cross platform users (#732, #748)
- Fix a bug in `compute_MVBS` in selecting `echo_range` for specific frequency. This is from the process of converting the data to be aligned with dimension `frequency` to `channel` in v0.6.0 (#736)
- Allow parsing data from EK60 with split-beam transducers but without phase/angle data (#491, #718)
- Fix invalid timestamp issue in AD2CP data conversion (#733)
- Check filter coeffs existence in `SetGroupsEK80.set_vendor` before saving (#720, #724)
- Fix empty Sv problem related to renaming time coordinate associated with environmental parameters used for calibration (#755)
- Fix the check in `compute_MVBS` for handling different variations of NaN entries in the Sv dataset (#753)

## New features
- Enhance `update_platform` to support a new use case (location data from fixed location) and add more consistency (#741)
- Ability to parse and store RAW4 datagram for EK80 data (#714)
- Add utility function for swapping `channel` coordinate with `frequency_nominal` (#710)
- Add ES70, ES80, EA640 to allowed data type for calibration (#759)

## Changes of netCDF data model
- Reorganize AD2CP data variables into different `Sonar/Beam_groupX`s and different first-level groups in a form consistent with v0.6.0 changes for all other sonar models (#731); some variables remain to be discussed and may change in future releases (#719)

## Enhancements
- Refactor AD2CP conversion to improve speed and memory usage through removal of xr.merge (#505)
- Update Python requirements in docs to >=3.8 (#744)

## Infrastructure
- Update PR action to use PR title [all tests ci] to run the entire suite of tests and [skip ci] to skip all tests (#721)




# v0.6.0 (2022 May 26)

## Overview
This is a major release that contains changes that enhances the compliance of echopype data model (and hence generated file structure) to the [SONAR-netCDF4 convention Version 1.0 ](http://www.ices.dk/sites/pub/Publication%20Reports/Cooperative%20Research%20Report%20(CRR)/CRR341.pdf). In addition, some variables were renamed to improve intuitive understanding of sonar data, provenance and standardized attributes are added to the processed dataset (e.g. Sv), the deprecated old API (<0.5.0) was removed, and some bugs were fixed.

## Changes of netCDF data model

- Move and rename the original `Beam` and `Beam_power` group to be subgroups under the `Sonar` group, in order to comply with the structure defined in the convention  (#567, #574, #605, #606, #611)
  - `Beam` --> `Sonar/Beam_group1`: contains either raw power or power/angle data for all sonar models other than EK80. For EK80, if only complex or power/angle data exist, all data are in this group; if _both_ complex and power/angle data exist, the complex data are in this group.
  - `Beam_power` --> `Sonar/Beam_group2`: contains power/angle when complex data occupies `Sonar/Beam_group1`; only exists for EK80 data when _both_ power/angle data and complex data bothexist in the file
- Rename the coordinate `range_bin` to `range_sample` to make it obvious that this coordinate indicates the digitization sample number for the sound waveform or intensity time series, and hence it takes a form of sequential integers 0, 1, 2, 3, ... (#595)
- Rename the data variable `range` in the calibrated Sv or TS dataset to `echo_range`, so that it is not confused with the python built-in function (#590)
- Rename the coordinate `quadrant` for EK80 data to `beam` (#619)
- Add coordinate `beam` with length 1 for all sonar models, except for AD2CP (#638, #646)
- Rename the data variable `Sp` to `TS` since "point backscattering strength" is a Simrad terminology and target strength (TS) is clearly defined and widely used. (#615)
- Rename time dimensions in the `Platform` group (`location_time`: `time1`, `mru_time`: `time2`) (#518, #631, #647)
- Rename the coordinate `frequency` to `channel` for all groups, to be more flexible (can accommodate channels with identical frequencies #490) and reasonable (since for broadband data the channel frequency is only nominal #566) (#657)
- Rename the data variable `heave` to `vertical_offset` in the Platform group (#592, #623)
- Change `src_filenames` string attribute to `source_filenames` list-of-strings variable (#620, #621)
- Bring consistency to the names of the time coordinates for the `Platform` and `Environment groups` (#656, #672)

## Changes of `EchoData` group access pattern

- The groups can now be accessed via a path in the form `echodata["Sonar/Platform"]`, `echodata["Sonar/Platform"]`, `echodata["Sonar/Beam_groupX"]`, etc. using [DataTree v0.0.4](https://github.com/xarray-contrib/datatree/tree/0.0.4) functionalities. (#611)
- The previous access pattern `echodata.platform`, `echodata.sonar`, `echodata.beam`, etc. is deprecated and will be removed in v0.6.1.

## Addition of attributes and variables in raw-converted and processed data

- Add indexing info for `Beam_groupX` as data variable under the `Sonar` group (#658)
- Add missing coordinate and variable attributes in the processed datasets Sv, MVBS, TS(#594)
- Add `water_level` to processed datasets (Sv, MVBS, TS) for convenient downstream conversion to depth (#259, #583, #615)
- Add additional environment variables for EK80 data (#616)
- Add missing platform data variables (#592, #631, #649, #654)

## New features and other enhancements

- Add parser for Echoview ECS file (#510)
- Add provenance to raw-converted and processed datasets (#621)
- Consolidate convention specs into a single yml file for pre-loading when creating `EchoData` objects (#565)
- Extra visualization module can now handle both `frequency` and `channel` filtering, as well as files with duplicated frequencies (#660)
- Improve selection of Vendor-specific calibration parameters for narrowband EK data (#697)

## CI improvements

- Upgrade python minimum to 3.8 and tidy up packaging (#604, #608, #609)
- Upgrade echopype development status to Beta (#610)
- Update `setup-services.py` to include images & volumes subtleties (#651)

## Other changes

- Remove the deprecated old (<0.5.0) API (#506, #601)
- Update README in the `echopype/test_data` folder (#584)
- Add documentation for visualization (#655)
- Add development roadmap to documentation (#636, #688)
- Restructure and expand data format section (#635)



# v0.5.6 (2022 Feb 10)


## Overview

This is a minor release that contains an experimental new feature and a number of enhancements, clean-up and bug fixes, which pave the way for the next major release.

## New feature

- (beta) Allow interpolating CTD data in calibration (#464)

  - Interpolation currently allowed along the ``ping_time`` dimension (the ``"stationary"`` case) and across ``latitude`` and ``longitude`` (the ``"mobile"`` case).
  - This mechanism is enabled via a new ``EnvParams`` class at input of calibration functions.

## Enhancements

- Make visualize module fully optional with ``matplotlib``, ``cmocean`` being optional dependency (#526, #559)
- Set range entries with no backscatter data to NaN in output of ``echodata.compute_range()`` (#547) and still allows quick visualization (#555)
- Add ``codespell`` GitHub action to ensure correct spellings of words (#557)
- Allow ``sonar_model="EA640"`` for ``open_raw`` (before it had to be "EK80") (#539)

## Bug fixes

- Allow using ``sonar_model="EA640"`` (#538, #539)
- Allow flexible and empty environment variables in EA640/EK80 files (#537)
- Docstring overhaul and fix bugs in ``utils.uwa`` (#525)

## Documentation

- Upgrade echopype docs to use jupyter book (#543)
- Change the RTD ``latest`` to point to the ``dev`` branch (#467)

## Testing

- Update convert tests to enable parallel testing (#556)
- Overhaul tests (#523, #498)

  - use ``pytest.fixture`` for testing
  - add ES70/ES80/EA640 test files
  - add new EK80 small test files with parameter combinations
  - reduce size for a subset of large EK80 test data files

- Add packaging testing for the ``dev`` branch (#554)


# v0.5.5 (2021 Dec 10)

## Overview

This is a minor release that includes new features, enhancements, bug fixes, and linking to an echopype preprint.

## New features

- Allow converting ES60/70/80 files and handle  various datagram anomaly (#409)
- Add simple echogram plotting functionality (beta) (#436)

## Enhancements

- ``update_platform`` method for ``EchoData`` now include proper variable attributes and correctly selects time range of platform data variables corresponding to those of the acoustic data (#476, #492, #493, #488)
- Improve testing for ``preprocess.compute_MVBS`` by running through real data for all supported sonar models (#454)
- Generalize handling of Beam group coordinate attributes and a subset of variable attributes (#480, #493)
- Allow optional kwargs when loading ``EchoData`` groups to enable delaying operations (#456)

## Bug fixes

- The gain factor for band-integrated Sv is now computed from broadband calibration data stored in the Vendor group (when available) or use nominal narrowband values (#446, #477)
- Fix time variable encoding for ``combine_echodata`` (#486)
- Fix missing ``ping_time`` dimension in AZFP Sv dataset to enable MVBS computation (#453)
- Fix bugs re path when writing to cloud (#462)

## Documentation

- Improvements to the "Contributing to echopype" page: Elaborate on the git branch workflow. Add description of PR squash and merge vs merge commit. Add instructions for running only a subset of tests locally (#482)
- Add documentation about ``output_storage_options`` for writing to cloud storage (#482)
- Add documentation and docstring for ``sonar_model`` in ``open_raw`` (#475)
- Improve documentation of EchoData object by adding a sample of the xarray Dataset HTML browser (#503)

## Others

- Zenodo badge update (#469)
- Add github citation file (#496), linking to [echopype preprint on arXiv](https://arxiv.org/abs/2111.00187)


# v0.5.4 (2021 Sep 27)

## Overview

This is a minor release that contains a few bug fixes and new functionalities.
The repo has migrated to use ``main`` instead of ``master`` after this release.

## New features

- Adding external platform-related data (e.g., latitude, longitude) to the ``EchoData`` object via the ``update_platform`` method (#434)
- Allow converting and storing data with duplicated ping times (#433)
- Add simple functions to compute summary statistics under the ``metrics`` subpackage (#444)

## Bug fixes

- Allow string info in AD2CP data packet header (#438)
- Re-attach ``sonar_model`` attribute to outputs of ``combine_echodata`` (#437)
- Handle exception in ``open_converted`` due to potentially empty ``beam_power`` group in Zarr files (#447)

## Others

- Warn users of removal of old API in the next release (#443)


# v0.5.3 (2021 Aug 20)

## Overview

This is a minor release that adds a few new functionalities, in particular a method to combine multiple ``EchoData`` objects, addresses a few bugs, improves packaging by removing pinning for dependencies, and improving the testing framework.

## New features

- Add a new method to combine multiple EchoData objects (#383, #414, #422, #425 )

  - Potential time reversal problems in time coordinates (e.g., ``ping_time``, ``location_time``) are checked and corrected as part of the combine function
  - The original timestamps are stored in the ``Provenance`` group

- Add a new method ``compute_range`` for ``EchoData`` object (#400)
- Allow flexible extensions for AZFP files in the form ".XXY" where XX is a number and Y is a letter (#428)

## Bug fixes

- Fix the bug/logic problems that prevented calibrating data in EK80 files that contains coexisting BB and CW data (#400)
- Fix the bug that prevented using the latest version of ``fsspec``  (#401)
- Fix the bug that placed ``echosounder_raw_transmit_samples_i/q`` as the first ping in ``echosounder_raw_samples_i/q`` as they should be separate variables (#427)

## Improvements

- Consolidate functions that handle local/remote paths and checking file existence (#401)
- Unpin all dependencies (#401)
- Improve test coverage accuracy (#411)
- Improve testing structure to match with subpackage structure (#401, #416, #429 )

## Documentation

- Expand ``Contributing to echopype`` page, including development workflow and testing strategy (#417, #420, #423)


# v0.5.2 (2021 Jul 18)

## Overview

This is a minor release that addresses issues related to time encoding for data variables related to platform locations and data conversion/encoding for AD2CP data files.

## Bug fixes and improvements

- Fixed the ``location_time`` encoding in the ``Platform`` group for latitude and longitude data variables (#393)
- Fixed the ``location_time`` encoding in the ``Platform/NMEA`` group (#395)
- Updated ``EchoData`` repr to show ``Platform/NMEA`` (#395, #396)
- Improved AD2CP data parsing and conversion (#388)

   - Cleaned up organization of data from different sampling modes and their corresponding time coordinates
   - Fixed parsing issues that generated spikes in parsed echosounder mode amplitude data
   - Removed the ``Beam_complex`` group and put raw IQ samples in the ``Vendor`` group per convention requirements
   - Populated the ``Sonar`` group with AD2CP information


# v0.5.1 (2021 Jun 16)

## Overview

This is a minor release that addresses a couple of issues from the last major version (0.5.0)
and improves code maintenance and testing procedures.


## New features

- Added experimental functions to detect and correct ``ping_time`` reversals.
  See `qc` subpackage (#297)


## Updates and bug fixes

- Fixed ADCP encoding issues (#361)
- Updated ``SetGroupsBase`` to use
  [ABC (Abstract Base Classes) Interface](https://docs.python.org/3/library/abc.html) (#366)
- Whole code-base linted for pep8 (#317)
- Removed old test data from the repository (#369)
- Updated package dependencies (#365)
- Simplified requirements for setting up local test environment (#375)


## CI improvements

- Added code coverage checking (#317)
- Added version check for echopype install (#367, #370)


#v0.5.0 (2021 May 17)

## Overview

This major release includes:

- major API updates to provide a more coherent data access pattern
- restructuring of subpackages and classes to allow better maintenance and future expansion
- reorganization of documentation, which also documents the API changes
- overhaul and improvements of CI, including removing the use of Git LFS to store test data
- new features
- bug fixes


## API updates

The existing API for converting files from raw instrument formats to a standardized format, and for calibrating data and performing operations such as binned averages and noise removal has been updated.

The new API uses a new ``EchoData`` object to encapsulate all data and metadata related to/parsed from a raw instrument data file. Beyond the calibration of backscatter quantities, other processing functions follow a consistent form to take an xarray Dataset as input argument and returns another xarray Dataset as output.

The major changes include:

- change from an object-oriented method calls to functional calls for file conversion (using the new ``convert`` subpackage), and deprecate the previous ``Convert`` class for handling file parsing and conversion
- deprecate the previous ``Process`` class, which use object-oriented method calls for performing both calibration and data processing
- separate out calibration functions to a new ``calibrate`` subpackage
- separate out noise removal and data reduction functions to a new ``preprocess`` subpackage
- create a new ``EchoData`` object class that encapsulates all raw data and metadata from instrument data files, regardless of whether the data is being parsed directly from the raw binary instrument files (returned by the new function ``open_raw``) or being read from an already converted file (returned by the new function ``open_converted``)


## Subpackage and class restructuring

The subpackages and classes were restructured to improve modularity that will help will future expansion and maintenance. The major restructuring includes:
("SONAR" below is used to indicate the sonar model, such as EK60, EK80 or AZFP)

- consolidate overlapping EK60/EK80 components, deprecate the previous ``Convert`` classes that handled file parsing and serialization, and revise new ``ParseSONAR`` and ``SetGroupsSONAR`` classes for file parsing and serialization
- consolidate all calibration-related components to a new ``calibrate`` submodule, which uses ``CalibrateSONAR`` classes under the hood
- consolidate all preprocessing functions into a a new ``preprocess`` submodule, which will be later expanded to include other functions with similar use in a workflow


## CI overhaul and improvements

- Added github workflows for testing, building test docker images, and publishing directly to PyPI
- Deprecated usage of Travis CI
- Test run is now selective on Github, to run tests only on changed/added files. Or run all locally with ``run-test.py`` script. (#280, #302)


## Documentation reorganization and updates

- Re-organization of pages with better grouping
- Added "What's New" page
- Added "Contributing to echopype" page
- Overhaul "API reference" page


## New features

- Add interfacing capability to read from and write to cloud object storage directly. (#216, #240)
- Allow environmental and calibration parameters to be optionally used in calibration in place of the values stored in data file
- Mean volume backscattering strength (MVBS) can now be computed based on actual time interval (specified in seconds) and range (specified in meters) (#54)
- Add NMEA message type as a data variable in the ``Platform`` group (#232), which allows users to freely select the suitable ones depending on use
- Add support to convert ``.ad2cp`` files generated by Nortek's Signature series ADCP (#326)


## Bug fixes

- Fix EK80 config XML parsing problem for files containing either ``PulseDuration`` or ``PulseLength`` (#305)
- Fix time encoding discrepancy in AZFP conversion (#328)
- Fix problematic automatic encoding of AZFP frequency (previously as ``int``) to ``float64`` (#309)
- Overhaul EK80 pulse compressed calibration (current implementation remaining in beta, see #308)


# v0.4.1 (2020 Oct 20)

Patches and enhancements to file conversion

This minor release includes the following changes:

## Bug fixes

- Fix bug in top level .nc output when combining multiple AZPF `.01A` files
- Correct time stamp for `.raw` MRU data to be from the MRU datagram, instead of those from the RAW3 datagrams (although they are identical from the test files we have).
- Remove unused parameter `sa_correction` from broadband `.raw` files
- Make sure import statement works on Google colab

## Enhancements

- Parse Simrad EK80 config XML correctly for data generated by WBAT and WBT Mini, and those involving the 2-in-1 "combi" transducer
- Parse Simrad `.raw` files with `NME1` datagram, such as files generated by the Simrad EA640 echosounder
- Handle missing or partially valid GPS data in `.raw` files by padding with NaN
- Handle missing MRU data in `.raw` files by padding with NaN
- Parse `.raw` filename with postfix beyond HHMMSS
- Allow export EK80 XML configuration datagram as a separate XML file

## Notes

To increase maintenance efficiency and code readability we are refactoring the `convert` and `process` modules. Some usage of these modules will change in the next major release.


# v0.4.0 (2020 Jun 24)

Add EK80 conversion, rename subpackage model to process

## New features

- Add EK80 support:

  - File conversion from EK80 `.raw` files to netCDF and zarr formats
  - "Simple" calibration to frequency-average Sv based on pulse compression output is implemented but needs to be thoroughly tested.

- Rename subpackage `echopype.model` to `echopype.process`

  - The new name better describes the subpackage's function to process data for further analysis
  - Also rename class `EchoData` to `Process` to mirror the structure in `Convert` better.
  - Importing using the old names will be deprecated in the next release.

- Overhaul converting multiple files with `combine_opt=True`

  - If target format is netCDF, temporary files will be created and finally combined to a single netCDF. This is due to current restriction that xarray does not allow simply appending new data to an existing file.
  - If target format is zarr, data in each file are unpacked and appended to the same output file.

- Allow reading Zarr into `Process` in addition to netCDF: thanks @lsetiawan!

- Add a logo!

## Bug fixes

Fix bugs in slicing NMEA group data based on the same time base when `range_bin` is changed
