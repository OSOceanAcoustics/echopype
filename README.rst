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


Installation and Usage
----------------------

Echopype currently supports file conversion and computation of data produced by:

- Simrad EK60 echosounder (``.raw`` files)
- ASL Environmental Sciences AZFP echosounders (``.01A`` files)

The file conversion functionality converts data from manufacturer-specific
binary formats into a standardized netCDF files, based on which all subsequent
computations are performed.
The data processing routines include calibration (instrument-specific), noise
removal, and mean volume backscattering strength (MVBS) calculation.

Echopype can be installed from PyPI or through conda:

.. code-block:: console

    # PyPI
    $ pip install echopype

    # conda
    $ conda install -c conda-forge echopype

Check out the `echopype documentation`_ for more details on installation and usage!

Watch the `echopype talk`_  at SciPy 2019 for background, discussions and a quick demo!

.. _echopype documentation: https://echopype.readthedocs.io
.. _echopype talk: https://www.youtube.com/watch?v=qboH7MyHrpU


Contributors
------------

`Wu-Jung Lee <http://leewujung.github.io>`_ (@leewujung),
`Kavin Nguyen <https://github.com/ngkavin>`_ (@ngkavin) and
`Paul Robinson <https://github.com/prarobinson/>`_ (@prarobinson)
are primary developers of this project.
`Valentina Staneva <https://escience.washington.edu/people/valentina-staneva/>`_ (@valentina-s)
provides consultation and also contributes to development.

Other contributors include:
`Frederic Cyr <https://github.com/cyrf0006>`_ (@cyrf0006),
`Sven Gastauer <https://www.researchgate.net/profile/Sven_Gastauer>`_ (@SvenGastauer),
`Marian Peña <https://www.researchgate.net/profile/Marian_Pena2>`_ (@marianpena),
`Mark Langhirt <https://www.linkedin.com/in/mark-langhirt-7b33ba80>`_ (@bnwkeys),
`Erin LaBrecque <https://www.linkedin.com/in/erin-labrecque/>`_ (@erinann),
`Emma Ozanich <https://www.linkedin.com/in/emma-reeves-ozanich-b8671938/>`_ (@emma-ozanich),
`Aaron Marburg <http://apl.uw.edu/people/profile.php?last_name=Marburg&first_name=Aaron>`_ (@amarburg)

We thank Dave Billenness of ASL Environmental Sciences for
providing the AZFP Matlab Toolbox as reference for our
development of AZFP support in echopype.


License
-------

Echopype is licensed under the open source Apache 2.0 license.


---------------

Copyright (c) 2018--, echopype Developers.
