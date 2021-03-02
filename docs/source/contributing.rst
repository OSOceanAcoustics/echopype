Contributing to echopype
========================

See the installation instructions (**TODO: Add link to that section**), 
and add information about installing for development (EM: see my notes)


Contributing on GitHub
----------------------

- `Creating issues on GitHub <https://medium.com/nyc-planning-digital/writing-a-proper-github-issue-97427d62a20f>`_ 
- `Pull requests <https://jarednielsen.com/learn-git-fork-pull-request/>`_


Running the tests
-----------------

**TODO: I got the Google credentials JSON file from Don, available on Google Drive.** 
Add some information here about the need for that file and how to obtain it.

1. Set up conda env with echopype in dev mode (``pip install -e .``). See instructions, above.
2. Go to the root directory of repo.
3. Set environmental variables

.. code-block:: bash

    $ export GOOGLE_SERVICE_JSON=$(cat /path/to/google-creds.json)
    $ export TEST_DATA_FOLDER_ID=1gnXbrNCT3BO6QM2Ro09kCyXztSe99uzc

4. Build the docker compose images

.. code-block:: bash

    $ docker-compose -f .ci_helpers/docker/docker-compose.yaml build

5. Bring up all the docker compose services

.. code-block:: bash

    $ docker-compose -f .ci_helpers/docker/docker-compose.yaml up

6. Copy ``test_data`` into http server

.. code-block:: bash

    $ docker cp -L ./echopype/test_data docker_httpserver_1:/usr/local/apache2/htdocs/data

7. Run the tests

.. code-block:: bash

    $ python -m pytest --log-cli-level=WARNING --verbose echopype/tests/test_convert.py


Test files
----------

**TODO:** Are we deprecating the use of Git LFS? 
This section should be updated or removed altogether.

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
