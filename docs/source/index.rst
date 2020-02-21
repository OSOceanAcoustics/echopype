.. echopype documentation master file, created by
   sphinx-quickstart on Wed Feb 13 15:33:27 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to echopype!
====================

**Echopype** is a package built for enhancing the interoperability and scalability
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


.. toctree::
   :maxdepth: 2

   why
   usage
   data-format
   resources
   api


Contributors
------------

`Wu-Jung Lee <http://leewujung.github.io>`_ (@leewujung) and
`Kavin Nguyen <https://github.com/ngkavin>`_ (@ngkavin)
are primary developers of this project.
`Valentina Staneva <https://escience.washington.edu/people/valentina-staneva/>`_ (@valentina-s)
provides consultation and also contributes to development.

Other contributors include:
`Frederic Cyr <https://github.com/cyrf0006>`_ (@cyrf0006),
`Paul Robinson <https://github.com/prarobinson/>`_ (@prarobinson),
`Sven Gastauer <https://www.researchgate.net/profile/Sven_Gastauer>`_ (@SvenGastauer),
`Marian Peña <https://www.researchgate.net/profile/Marian_Pena2>`_ (@marianpena),
`Mark Langhirt <https://www.linkedin.com/in/mark-langhirt-7b33ba80>`_ (@bnwkeys),
`Erin LaBrecque <https://www.linkedin.com/in/erin-labrecque/>`_ (@erinann),
`Emma Ozanich <https://www.linkedin.com/in/emma-reeves-ozanich-b8671938/>`_ (@emma-ozanich),
`Aaron Marburg <http://apl.uw.edu/people/profile.php?last_name=Marburg&first_name=Aaron>`_ (@amarburg)

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

