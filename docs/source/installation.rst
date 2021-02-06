Installation
============

Echopype can be installed from `PyPI <https://pypi.org/project/echopype/>`_:

.. code-block:: console

   $ pip install echopype


or through conda from the `conda-forge Anaconda channel <https://anaconda.org/conda-forge/echopype>`_:

.. code-block:: console

   $ conda install -c conda-forge echopype


When creating a conda environment to work with echopype, do

.. code-block:: console

   $ conda create -c conda-forge --name echopype python=3.8 --file requirements.txt --file requirements-dev.txt

Echopype works for python>=3.7.


Test files
----------

Echopype uses Git Large File Storage `(Git LFS) <https://git-lfs.github.com/>`_
to store the binary data and test files used. Git LFS enables the Github
repository to remain small while still being able to access
the large test files needed for testing.
These files are only needed if you plan to work on the code and run the
tests locally.

To access the test files, first
`install Git LFS. <https://help.github.com/en/github/managing-large-files/installing-git-large-file-storage>`_

Cloning echopype after installing Git LFS will automatically pull the test data, but
if echopype was cloned first, then pull the files from Git LFS by running:

.. code-block:: console

   $ git lfs fetch

If no files are fetched, specify the names of the target remote and branch. 
For example, for the ``upstream`` remote and ``mybranch`` branch:

.. code-block:: console

   $ git lfs fetch -all upstream mybranch

.. note::

   Echopype has recently migrated to using Git LFS which required removing the large
   datasets from the history. It is recommended that those who have previously forked
   echopype delete their fork and fork a new one. Otherwise, pulling form the original
   repository will result in twice the number of commits due to the re-written history.
