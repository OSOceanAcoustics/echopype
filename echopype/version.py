from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: Apache 2.0 License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "echopype: an open-source tool for unpacking echosounder data"
# Long description will go up on the pypi page
long_description = """

echopype
========
Echopype is an open source Python package that converts data from various manufacturer proprietary format into netCDF4 files. The netCDF4 file follows the [ICES SONAR-netCDF4 convention](http://www.ices.dk/sites/pub/Publication%20Reports/Cooperative%20Research%20Report%20(CRR)/CRR341/CRR341.pdf). [Here](https://github.com/ices-eg/wg_WGFAST/tree/master/SONAR-netCDF4) is a GitHub folder with details of this convention in convenient tables. Echopype supports reading:
- .raw files from Simrad EK60 narrowband echosounder
- .raw files from Simrad EK80 broadband echosounder
- .01A files from ASL Environmental Sciences AZFP echosounder

.. _README: https://github.com/leewujung/echopype/blob/master/README.md

License
=======
``echopype`` is licensed under the terms of the Apache 2.0 license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2018--, Wu-Jung Lee, Applied Physics Laboratory, University of Washington.
"""

NAME = "echopype"
MAINTAINER = "Wu-Jung Lee"
MAINTAINER_EMAIL = "leewujung@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/OSOceanAcoustics/echopype"
DOWNLOAD_URL = ""
LICENSE = "Apache 2.0"
AUTHOR = "Wu-Jung Lee"
AUTHOR_EMAIL = "leewujung@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'echopype': [pjoin('data', '*')]}
REQUIRES = ["numpy", "pandas"]
