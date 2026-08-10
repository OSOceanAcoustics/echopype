API reference
==============

API components that most users will interact with.

.. attention::

   In echopype versions prior to 0.5.0, the API in this page focused
   on the ``convert`` and ``process`` subpackages. See the
   `0.4.1 API page <https://echopype.readthedocs.io/en/v0.4.1/api.html>`_
   if you're using a previous release. That workflow is now removed.

EchoData class
--------------

.. automodule:: echopype.echodata
   :members:

Open raw and converted files
----------------------------

.. _api-open_raw:

.. automodule:: echopype
   :members: open_raw, open_converted, combine_echodata

Data processing subpackages
---------------------------

calibrate
^^^^^^^^^

.. automodule:: echopype.calibrate
   :members:

clean
^^^^^

.. automodule:: echopype.clean
   :members:

colormap
^^^^^^^^

.. automodule:: echopype.colormap
   :members:

commongrid
^^^^^^^^^^

.. automodule:: echopype.commongrid
   :members:

consolidate
^^^^^^^^^^^

.. automodule:: echopype.consolidate
   :members:

mask
^^^^

.. automodule:: echopype.mask
   :members:

metrics
^^^^^^^

.. automodule:: echopype.metrics
   :members:

qc
^^^

.. automodule:: echopype.qc
   :members:


Utilities
---------

.. automodule:: echopype.utils.uwa
   :members:
