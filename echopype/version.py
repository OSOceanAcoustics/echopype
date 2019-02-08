from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = 0  # use '' for first of series, number for 1 and above
_version_extra = ''
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
               "License :: OSI Approved :: Apache Software License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "echopype is an open-source toolkit for analyzing active sonar data for biological information in the ocean."
# Long description will go up on the pypi page
with open('README.rst') as file:
    long_description = file.read()

NAME = "echopype"
MAINTAINER = "Wu-Jung Lee"
MAINTAINER_EMAIL = "leewujung@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/OSOceanAcoustics/echopype"
DOWNLOAD_URL = ""
LICENSE = 'Apache License, Version 2.0'
AUTHOR = "Wu-Jung Lee"
AUTHOR_EMAIL = "leewujung@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'echopype': [pjoin('data', '*')]}
REQUIRES = ["numpy", "pandas", "xarray"]
TEST_REQUIRES = ["tox"]
INSTALL_REQUIRES = ["click"]
SCRIPTS = ['echopype/convert/echopype_converter']
