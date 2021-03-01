Contributing to echopype
========================

See the installation instructions (**TODO: Add link to that section**), 
and add information about installing for development (EM: see my notes)

Running the tests
-----------------

**TODO: I got the Google credentials JSON file from Don, available on Google Drive.** 
Add some information here about the need for that file and how to obtain it.

1. Set up conda env with echopype in dev mode (``pip install -e .``). See instructions, above.
2. Go to the root directory of repo.
3. Set environmental variables

.. code-block:: bash

    export GOOGLE_SERVICE_JSON=$(cat /path/to/google-creds.json)
    export TEST_DATA_FOLDER_ID=1gnXbrNCT3BO6QM2Ro09kCyXztSe99uzc

4. Build the docker compose images

.. code-block:: bash

    docker-compose -f .ci_helpers/docker/docker-compose.yaml build

5. Bring up all the docker compose services

.. code-block:: bash

    docker-compose -f .ci_helpers/docker/docker-compose.yaml up

6. Copy ``test_data`` into http server

.. code-block:: bash

    docker cp -L ./echopype/test_data docker_httpserver_1:/usr/local/apache2/htdocs/data

7. Run the tests

.. code-block:: bash

    python -m pytest --log-cli-level=WARNING --verbose echopype/tests/test_convert.py


Contributing on GitHub
----------------------

- `Creating issues on GitHub <https://medium.com/nyc-planning-digital/writing-a-proper-github-issue-97427d62a20f>`_ 
- `Pull requests <https://jarednielsen.com/learn-git-fork-pull-request/>`_
