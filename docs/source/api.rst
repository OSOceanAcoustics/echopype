API reference
==============

API components that most users will interact with.

.. attention::

   In echopype versions prior to 0.5.0, the API in this page focused 
   on the ``convert`` and ``process`` subpackages. See the 
   `0.4.1 API page <https://echopype.readthedocs.io/en/v0.4.1/api.html>`_
   if you're using a previous release. That workflow is being deprecated.

**Content**

* `EchoData class`_
* `Open raw and converted files`_
* `Combine EchoData objects`_
* `Data processing subpackages`_
* `Utilities`_


EchoData class
--------------

.. didn't yield expected results (no better than automodapi) AND hid the open_ functions!
   .. autoclass:: echopype.echodata
      :members:

.. automodapi:: echopype.echodata
   :no-inheritance-diagram:
   :no-heading:

Open raw and converted files
----------------------------

.. automodule:: echopype
   :members: open_raw

.. automodule:: echopype
   :members: open_converted

Combine EchoData objects
------------------------

.. automodule:: echopype
   :members: combine_echodata

Data processing subpackages
---------------------------

calibrate
^^^^^^^^^

.. automodapi:: echopype.calibrate
   :no-inheritance-diagram:
   :no-heading:

preprocess
^^^^^^^^^^

.. automodapi:: echopype.preprocess
   :no-inheritance-diagram:
   :no-heading:

qc
^^^

.. automodapi:: echopype.qc
   :no-inheritance-diagram:
   :no-heading:

Utilities
---------

.. automodapi:: echopype.utils.uwa
   :no-inheritance-diagram:
   :no-heading:
   :members: calc_absorption, calc_sound_speed
