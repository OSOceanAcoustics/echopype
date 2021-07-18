What's new
==========

See `GitHub releases page <https://github.com/OSOceanAcoustics/echopype/releases>`_ for the complete history.


v0.5.2 (2021 Jul 18)
--------------------

Overview
~~~~~~~~

This is a minor release that addresses issues related to time encoding for data variables related to platform locations and data conversion/encoding for AD2CP data files.

Bug fixes and improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Fixed the ``location_time`` encoding in the ``Platform`` group for latitude and longitude data variables (#393)
- Fixed the ``location_time`` encoding in the ``Platform/NMEA`` group (#395)
- Updated ``EchoData`` repr to show ``Platform/NMEA`` (#395, #396)
- Improved AD2CP data parsing and conversion (#388)

   - Cleaned up organization of data from different sampling modes and their corresponding time coordinates
   - Fixed parsing issues that generated spikes in parsed echosounder mode amplitude data
   - Removed the ``Beam_complex`` group and put raw IQ samples in the ``Vendor`` group per convention requirements
   - Populated the ``Sonar`` group with AD2CP information



v0.5.1 (2021 Jun 16)
--------------------

Overview
~~~~~~~~

This is a minor release that addresses a couple of issues from the last major version (0.5.0)
and improves code maintenance and testing procedures.


New features
~~~~~~~~~~~~

- Added experimental functions to detect and correct ``ping_time`` reversals.
  See `qc` subpackage (#297)


Updates and bug fixes
~~~~~~~~~~~~~~~~~~~~~

- Fixed ADCP encoding issues (#361)
- Updated ``SetGroupsBase`` to use 
  `ABC (Abstract Base Classes) Interface <https://docs.python.org/3/library/abc.html>`_ (#366)
- Whole code-base linted for pep8 (#317)
- Removed old test data from the repository (#369)
- Updated package dependencies (#365)
- Simplified requirements for setting up local test environment (#375)


CI improvements
~~~~~~~~~~~~~~~

- Added code coverage checking (#317)
- Added version check for echopype install (#367, #370)


v0.5.0 (2021 May 17)
--------------------

Overview
~~~~~~~~

This major release includes:

- major API updates to provide a more coherent data access pattern
- restructuring of subpackages and classes to allow better maintenance and future expansion
- reorganization of documentation, which also documents the API changes
- overhaul and improvements of CI, including removing the use of Git LFS to store test data
- new features
- bug fixes


API updates
~~~~~~~~~~~

The existing API for converting files from raw instrument formats to a standardized format, and for calibrating data and performing operations such as binned averages and noise removal has been updated. 

The new API uses a new ``EchoData`` object to encapsulate all data and metadata related to/parsed from a raw instrument data file. Beyond the calibration of backscatter quantities, other processing functions follow a consistent form to take an xarray Dataset as input argument and returns another xarray Dataset as output.

The major changes include:

- change from an object-oriented method calls to functional calls for file conversion (using the new ``convert`` subpackage), and deprecate the previous ``Convert`` class for handling file parsing and conversion
- deprecate the previous ``Process`` class, which use object-oriented method calls for performing both calibration and data processing
- separate out calibration functions to a new ``calibrate`` subpackage
- separate out noise removal and data reduction functions to a new ``preprocess`` subpackage
- create a new ``EchoData`` object class that encapsulates all raw data and metadata from instrument data files, regardless of whether the data is being parsed directly from the raw binary instrument files (returned by the new function ``open_raw``) or being read from an already converted file (returned by the new function ``open_converted``)


Subpackage and class restructuring 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The subpackages and classes were restructured to improve modularity that will help will future expansion and maintenance. The major restructuring includes:
("SONAR" below is used to indicate the sonar model, such as EK60, EK80 or AZFP)

- consolidate overlapping EK60/EK80 components, deprecate the previous ``Convert`` classes that handled file parsing and serialization, and revise new ``ParseSONAR`` and ``SetGroupsSONAR`` classes for file parsing and serialization
- consolidate all calibration-related components to a new ``calibrate`` submodule, which uses ``CalibrateSONAR`` classes under the hood
- consolidate all preprocessing functions into a a new ``preprocess`` submodule, which will be later expanded to include other functions with similar use in a workflow


CI overhaul and improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added github workflows for testing, building test docker images, and publishing directly to PyPI
- Deprecated usage of Travis CI
- Test run is now selective on Github, to run tests only on changed/added files. Or run all locally with ``run-test.py`` script. (#280, #302)


Documentation reorganization and updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Re-organization of pages with better grouping
- Added "What's New" page
- Added "Contributing to echopype" page
- Overhaul "API reference" page


New features
~~~~~~~~~~~~

- Add interfacing capability to read from and write to cloud object storage directly. (#216, #240)
- Allow environmental and calibration parameters to be optionally used in calibration in place of the values stored in data file
- Mean volume backscattering strength (MVBS) can now be computed based on actual time interval (specified in seconds) and range (specified in meters) (#54)
- Add NMEA message type as a data variable in the ``Platform`` group (#232), which allows users to freely select the suitable ones depending on use
- Add support to convert ``.ad2cp`` files generated by Nortek's Signature series ADCP (#326)


Bug fixes
~~~~~~~~~

- Fix EK80 config XML parsing problem for files containing either ``PulseDuration`` or ``PulseLength`` (#305)
- Fix time encoding discrepancy in AZFP conversion (#328)
- Fix problematic automatic encoding of AZFP frequency (previously as ``int``) to ``float64`` (#309)
- Overhaul EK80 pulse compressed calibration (current implementation remaining in beta, see #308)



v0.4.1 (2020 Oct 20)
--------------------

Patches and enhancements to file conversion

This minor release includes the following changes:

Bug fixes
~~~~~~~~~

- Fix bug in top level .nc output when combining multiple AZPF `.01A` files
- Correct time stamp for `.raw` MRU data to be from the MRU datagram, instead of those from the RAW3 datagrams (although they are identical from the test files we have).
- Remove unused parameter `sa_correction` from broadband `.raw` files
- Make sure import statement works on Google colab

Enhancements
~~~~~~~~~~~~

- Parse Simrad EK80 config XML correctly for data generated by WBAT and WBT Mini, and those involving the 2-in-1 "combi" transducer
- Parse Simrad `.raw` files with `NME1` datagram, such as files generated by the Simrad EA640 echosounder
- Handle missing or partially valid GPS data in `.raw` files by padding with NaN
- Handle missing MRU data in `.raw` files by padding with NaN
- Parse `.raw` filename with postfix beyond HHMMSS
- Allow export EK80 XML configuration datagram as a separate XML file

Notes
~~~~~

To increase maintenance efficiency and code readability we are refactoring the `convert` and `process` modules. Some usage of these modules will change in the next major release.


v0.4.0 (2020 Jun 24)
--------------------

Add EK80 conversion, rename subpackage model to process

New features
~~~~~~~~~~~~

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

Bug fixes
~~~~~~~~~

Fix bugs in slicing NMEA group data based on the same time base when `range_bin` is changed
