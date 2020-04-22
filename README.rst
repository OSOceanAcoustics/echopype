.. image:: https://travis-ci.org/OSOceanAcoustics/echopype.svg?branch=master
    :target: https://travis-ci.org/OSOceanAcoustics/echopype
.. image:: https://readthedocs.org/projects/echopype/badge/?version=latest
    :target: https://echopype.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/OSOceanAcoustics/echopype/master

Echopype
========

Echopype is a package built for enhancing the interoperability and scalability
in ocean sonar data processing.
These data are widely used for obtaining information about the distribution and
abundance of marine animals, such as fish and krill.
Our ability to collect large volumes of sonar data from a variety of
ocean platforms has grown significantly in the last decade.
However, most of the new data remain under-utilized.
echopype aims to address the root cause of this problem - the lack of
interoperable data format and scalable analysis workflows that adapt well
with increasing data volume - by providing open-source tools as entry points for
scientists to make discovery using these new data.


Installation
------------

Echopype currently supports file conversion and computation of data produced by:

- Simrad EK60 echosounder (``.raw`` files)
- ASL Environmental Sciences AZFP echosounders (``.01A`` files)

Support for ``.raw`` files from the broadband Simrad EK80 echosounder is currently
in the development branch
`combine-refactor <https://github.com/OSOceanAcoustics/echopype/tree/convert-refactor>`_
and we will merge it to the master branch once it's ready for alpha testing.

The file conversion functionality converts data from manufacturer-specific
binary formats into a standardized netCDF files, based on which all subsequent
computations are performed.
The data processing routines include calibration (instrument-specific), noise
removal, and mean volume backscattering strength (MVBS) calculation.

Echopype can be installed from PyPI:

.. code-block:: console

   $ pip install echopype


or through conda:

.. code-block:: console

   $ conda install -c conda-forge echopype


When creating an conda environment to work with echopype,
use the supplied ``environment.yml`` or do

.. code-block:: console

   $ conda create -c conda-forge -n echopype python=3.8 --file requirements.txt


Usage
-----

Check out the `echopype documentation`_ for more details on installation and usage.

Watch the `echopype talk`_  at SciPy 2019 for background, discussions and a quick demo!

.. _echopype documentation: https://echopype.readthedocs.io
.. _echopype talk: https://www.youtube.com/watch?v=qboH7MyHrpU


Contributors
------------

`Wu-Jung Lee <http://leewujung.github.io>`_ (@leewujung) leads this project
and along with `Kavin Nguyen <https://github.com/ngkavin>`_ (@ngkavin)
are primary developers of this package.
`Valentina Staneva <https://escience.washington.edu/people/valentina-staneva/>`_ (@valentina-s)
and `Emilio Mayorga <https://www.apl.washington.edu/people/profile.php?last_name=Mayorga&first_name=Emilio>`_ (@emiliom)
provide consultation and also contribute to the development.
Other contributors are listed `here <echopype documentation>`_.

We thank Dave Billenness of ASL Environmental Sciences for
providing the AZFP Matlab Toolbox as reference for our
development of AZFP support in echopype.
We also thank `Rick Towler <https://github.com/rhtowler>`_
of the Alaska Fisheries Science Center
for providing low-level file parsing routines for
Simrad EK60 and EK80 echosounders.


License
-------

Echopype is licensed under the open source Apache 2.0 license.


---------------

Copyright (c) 2018--, echopype Developers.
