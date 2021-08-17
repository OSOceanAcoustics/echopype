Contributing to echopype
========================

We welcome your contributions, large or small!


Contributing with Git and GitHub
--------------------------------

Please submit questions or report problems via GitHub issues. If you're new to GitHub, 
see these tips for submitting issues: 
`"Creating issues on GitHub" <https://medium.com/nyc-planning-digital/writing-a-proper-github-issue-97427d62a20f>`_.

For echopype development we use the **gitflow workflow** with forking. All development
changes are merged into the ``dev`` development branch. First create your own fork of the 
source GitHub repository 
`https://github.com/OSOceanAcoustics/echopype/ <https://github.com/OSOceanAcoustics/echopype/>`_ 
(``upstream``), then clone your fork; your fork will be the ``origin`` remote. See 
`this excellent tutorial <https://www.dataschool.io/how-to-contribute-on-github/>`_ for 
guidance on forking and opening pull requests, but replace references to the ``master`` 
branch with the ``dev`` development branch. See 
`this description of the gitflow workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_. 
The complete workflow we use is depicted in the diagram below, which includes
components involving documentation updates (see `Documentation development`_ below)
and preparation of releases.

.. mermaid::

    graph LR
        classDef patch fill:#f2ece4
        master --> stable
        master --> dev
        p1([doc patch]):::patch -.-> stable
        p2([code patch]):::patch -.-> dev
        stable --> |docs merge| rel[release/0.x.y]
        dev --> |dev merge| rel
        rel --> master


Installation for echopype development
-------------------------------------

To access and test the latest, unreleased development version of echopype, clone the ``master`` branch from the source repository:

.. code-block:: bash

    git clone https://github.com/OSOceanAcoustics/echopype.git

To set up your local environment to contribute to echopype, 
please follow the git forking workflow as described above. 
After forking the source repository, clone your fork, 
then set the source repository as the ``upstream`` git remote:

.. code-block:: bash

    git clone https://github.com/myusername/echopype.git
    cd echopype
    git remote add upstream https://github.com/OSOceanAcoustics/echopype.git

Create a `conda <https://docs.conda.io>`_ environment for echopype development
(replace the Python version with your preferred version):

.. code-block:: bash

    conda create -c conda-forge -n echopype --yes python=3.8 --file requirements.txt --file requirements-dev.txt
    conda activate echopype
    # ipykernel is recommended, in order to use with JupyterLab and IPython
    # to aid with development. We recommend you install JupyterLab separately
    conda install -c conda-forge ipykernel
    pip install -e .

See the :doc:`installation` page to simply install the latest echopype release from conda or PyPI.


Test data files
---------------

.. attention::

    Echopype previously used Git LFS for managing and accessing large test data files. 
    We have deprecated its use starting with echopype version 0.5.0. The files
    in https://github.com/OSOceanAcoustics/echopype/tree/master/echopype/test_data
    are also being deprecated.

Test echosounder data files are managed in a private Google Drive folder and 
made available via the `cormorack/http <https://hub.docker.com/r/cormorack/http>`_
Docker image on Docker hub; the image is rebuilt daily when new test data are added
on Google Drive. See the `Running the tests`_ section below for 
details.


Running the tests
-----------------

To run the echopype unit tests in ``echopype/tests``, 
`Docker <https://docs.docker.com/get-docker/>`_ 
will need to be installed if not already present 
(`docker-compose <https://docs.docker.com/compose/>`_ is also used, but it's installed
in the conda environment for echopype development). To run the tests:

.. code-block:: bash

    # Install and/or deploy the echopype docker containers for testing
    python .ci_helpers/docker/setup-services.py --deploy

    # Run the tests
    python .ci_helpers/run-test.py --local --pytest-args="-vv"

    # When done, "tear down" the docker containers
    python .ci_helpers/docker/setup-services.py --tear-down

The tests include reading and writing from locally set up (via docker) http 
and `S3 object-storage <https://en.wikipedia.org/wiki/Amazon_S3>`_ sources, 
the latter via `minio <https://minio.io>`_.


pre-commit hooks
----------------

The echopype development conda environment includes `pre-commit <https://pre-commit.com>`_,
and useful pre-commit "hooks" have been configured in the 
`.pre-commit-config.yaml file <https://github.com/OSOceanAcoustics/echopype/blob/master/.pre-commit-config.yaml>`_. 
Current hooks include file formatting (linting) checks (trailing spaces, trailing lines,
JSON and YAML format checks, etc) and Python style autoformatters (PEP8 / flake8, ``black`` and ``isort``).

To run pre-commit hooks locally, run `pre-commit install` before running the 
docker setup-service deploy statement described above. The hooks will run automatically 
during ``git commit`` and will give you options as needed before committing your changes.
You can also run ``pre-commit`` before actually doing ``git commit``, as you edit the code, by running ``pre-commit run --all-files``. See the `pre-commit usage documentation <https://pre-commit.com/#usage>`_ for details.


Continuous integration GitHub Actions
-------------------------------------

echopype makes extensive use of GitHub Actions for continuous integration (CI)
of unit tests and other code quality controls. Every pull request triggers the CI.
See `echopype/.github/workflows <https://github.com/OSOceanAcoustics/echopype/tree/master/.github/workflows>`_.

The CI tests can be a bit slow, taking up to 20-30 minutes.
Under special circumstances, when the submitted changes have a 
very limited scope (such as contributions to the documentation)
or you know exactly what you're doing 
(you're a seasoned echopype contributor), the CI can be skipped.
This is done by including the string "[skip ci]" in your last commit's message.


Documentation development
-------------------------

Echopype documentation (`<https://echopype.readthedocs.io>`_) is based on 
`Sphinx <https://www.sphinx-doc.org>`_ and is hosted at 
`Read The Docs <https://readthedocs.org>`_. The sphinx files are found
in the ``docs`` directory, and the source documentation files, written in 
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
(``.rst``) format, are in the ``docs/source`` directory. The echopype development
conda environment will install all required Sphinx dependencies.
To run Sphinx locally:

.. code-block:: bash

    cd docs
    sphinx-build -b html -d _build/doctrees source _build/html

To view the generated HTML files generated by Sphinx, open the 
``docs/_build/html/index.html`` in your browser.

Updates to the documentation that are based on the current echopype release (that is,
not involving echopype API changes) should be merged into the GitHub ``stable`` branch.
These updates will then become available immediately on the default ReadTheDocs version.
Examples of such updates include fixing spelling mistakes, expanding an explanation, and adding a new section that documents a previously undocumented feature.

Function and object doc strings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For inline strings documenting functions and objects ("doc strings"), we use the
`numpydoc style (Numpy docstring format) <https://numpydoc.readthedocs.io/en/latest/format.html>`_.


Documentation versions
~~~~~~~~~~~~~~~~~~~~~~

`<https://echopype.readthedocs.io>`_ redirects to the documentation ``stable`` version, 
`<https://echopype.readthedocs.io/en/stable/>`_, which is built from the `stable` branch 
on the ``echopype`` GitHub repository. In addition, the ``latest`` version 
(`<https://echopype.readthedocs.io/en/latest/>`_) is built from the `master` branch, 
while the hidden `dev` version (`<https://echopype.readthedocs.io/en/dev/>`_) is built 
from the ``dev`` branch. Finally, each new echopype release is built as a new release version 
on ReadTheDocs. Merging pull requests into any of these three branches or issuing a 
new tagged release will automatically result in a new ReadTheDocs build for the 
corresponding version.

We also maintain a test version of the documentation at `<https://doc-test-echopype.readthedocs.io/>`_
for viewing and debugging larger, more experimental changes, typically from a separate fork.
