Contributing to echopype
========================

We welcome your contributions, large or small!


Contributing with Git and GitHub
--------------------------------

Please submit questions or report problems via GitHub issues. If you're new to GitHub,
see these tips for submitting issues:
`"Creating issues on GitHub" <https://medium.com/nyc-planning-digital/writing-a-proper-github-issue-97427d62a20f>`_.

For echopype development we use the **gitflow workflow** with forking. All development
changes (except for documentation) are merged into the ``dev`` development branch.
First create your own fork of the source GitHub repository
`https://github.com/OSOceanAcoustics/echopype <https://github.com/OSOceanAcoustics/echopype/fork>`_
(``upstream``), then clone your fork; your fork will be the ``origin`` remote. See
`this excellent tutorial <https://www.dataschool.io/how-to-contribute-on-github/>`_ for
guidance on forking and opening pull requests (PRs), but replace references to the ``main``
branch with the ``dev`` development branch. See
`this description of the gitflow workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_.
This diagram depicts the complete workflow we use in the source GitHub repository:

.. mermaid::

    graph LR
        classDef patch fill:#f2ece4
        main --> stable
        main --> dev
        p1([doc patch]):::patch -.-> stable
        p2([code patch]):::patch -.-> dev
        stable --> |docs merge| rel[release/0.x.y]
        dev --> |dev merge| rel
        rel --> main

- ``doc patch``: Updates to the documentation that refer to the current ``echopype``
  release can be pushed out immediately to the
  `echopype documentation site <https://echopype.readthedocs.io>`_
  by contibuting patches (PRs) to the ``stable`` branch. See `Documentation development`_
  below for more details.
- ``code patch``: Code development is carried out as patches (PRs) to the ``dev``
  branch; changes in the documentation corresponding to changes in the code can be
  carried out in this branch as well.
- New releases are prepared in a new release branch that merges changes in ``dev`` and ``stable``.

To maintain a clean and readable commit history, use "Merge pull request > Squash and merge"
when merging an individual PR to `dev` or a documentation-only PR to `stable`. This will
highlight the specific feature(s) contributed by the PR. When merging an ``echopype``
release PR to `main`, use "Merge pull request > Create a merge commit" in order to
retain all the squashed PR commits in the commit history.


Installation for echopype development
-------------------------------------

To access and test the latest, unreleased development version of echopype,
clone the ``main`` branch from the source repository:

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

    # create conda environment using the supplied requirements files
    # note the last one docs/requirements.txt is only required for building docs
    conda create -c conda-forge -n echopype --yes python=3.9 --file requirements.txt --file requirements-dev.txt --file docs/requirements.txt

    # switch to the newly built environment
    conda activate echopype

    # ipykernel is recommended, in order to use with JupyterLab and IPython
    # to aid with development. We recommend you install JupyterLab separately
    conda install -c conda-forge ipykernel

    # install echopype in editable mode (setuptools "develop mode")
    # plot is an extra set of requirements that can be used for plotting.
    # the command will install all the dependencies along with plotting dependencies.
    pip install -e .[plot]

See the :doc:`installation` page to simply install the latest echopype release from conda or PyPI.


Tests and test infrastructure
-----------------------------

Test data files
~~~~~~~~~~~~~~~

.. attention::

    Echopype previously used Git LFS for managing and accessing large test data files.
    We have deprecated its use starting with echopype version 0.5.0. The files
    in https://github.com/OSOceanAcoustics/echopype/tree/main/echopype/test_data
    are also being deprecated.

Test echosounder data files are managed in a private Google Drive folder and
made available via the `cormorack/http <https://hub.docker.com/r/cormorack/http>`_
Docker image on Docker hub; the image is rebuilt daily when new test data are added
on Google Drive. See the `Running the tests`_ section below for details.

Running the tests
~~~~~~~~~~~~~~~~~

To run the echopype unit tests found in ``echopype/tests``,
`Docker <https://docs.docker.com/get-docker/>`_
will need to be installed if not already present
(`docker-compose <https://docs.docker.com/compose/>`_ is also used,
but it's installed in the conda environment for echopype development). Then:

.. code-block:: bash

    # Install and/or deploy the echopype docker containers for testing.
    # Test data files will be downloaded
    python .ci_helpers/docker/setup-services.py --deploy

    # Run all the tests. But first make sure the
    # echopype development conda environment is activated
    python .ci_helpers/run-test.py --local --pytest-args="-vv"

    # When done, "tear down" the docker containers
    python .ci_helpers/docker/setup-services.py --tear-down

The tests include reading and writing from locally set up (via docker) http
and `S3 object-storage <https://en.wikipedia.org/wiki/Amazon_S3>`_ sources,
the latter via `minio <https://minio.io>`_.

`.ci_helpers/run-test.py <https://github.com/OSOceanAcoustics/echopype/blob/main/.ci_helpers/run-test.py>`_
will execute all tests. The entire test suite can be a bit slow, taking up to 40 minutes
or more. If your changes impact only some of the subpackages (``convert``, ``calibrate``,
``preprocess``, etc), you can run ``run-test.py`` with only a subset of tests by passing
as an argument a comma-separated list of the modules that have changed. For example:

.. code-block:: bash

    python .ci_helpers/run-test.py --local --pytest-args="-vv" echopype/calibrate/calibrate_ek.py,echopype/preprocess/noise_est.py

will run only tests associated with the ``calibrate`` and ``preprocess`` subpackages.
For ``run-test.py`` usage information, use the ``-h`` argument:
``python .ci_helpers/run-test.py -h``

pre-commit hooks
~~~~~~~~~~~~~~~~

The echopype development conda environment includes `pre-commit <https://pre-commit.com>`_,
and useful pre-commit "hooks" have been configured in the
`.pre-commit-config.yaml file <https://github.com/OSOceanAcoustics/echopype/blob/main/.pre-commit-config.yaml>`_.
Current hooks include file formatting (linting) checks (trailing spaces, trailing lines,
JSON and YAML format checks, etc) and Python style autoformatters (PEP8 / flake8, ``black`` and ``isort``).

To run pre-commit hooks locally, run ``pre-commit install`` before running the
docker setup-service deploy statement described above. The hooks will run automatically
during ``git commit`` and will give you options as needed before committing your changes.
You can also run ``pre-commit`` before actually doing ``git commit``, as you edit the code,
by running ``pre-commit run --all-files``. See the `pre-commit usage documentation <https://pre-commit.com/#usage>`_ for details.

Continuous integration GitHub Actions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

echopype makes extensive use of GitHub Actions for continuous integration (CI)
of unit tests and other code quality controls. Every pull request (PR) triggers the CI.
See `echopype/.github/workflows <https://github.com/OSOceanAcoustics/echopype/tree/main/.github/workflows>`_,
especially `pr.yaml <https://github.com/OSOceanAcoustics/echopype/blob/main/.github/workflows/pr.yaml>`_.

The entire test suite can be a bit slow, taking up to 40 minutes or more.
To mitigate this, the CI default is to run tests only for subpackages that
were modified in the PR; this is done via ``.ci_helpers/run-test.py``
(see the `Running the tests`_ section). To have the CI execute the
entire test suite, add the GitHub label ``Needs Complete Testing`` to the
PR before submitting it.

Under special circumstances, when the submitted changes have a
very limited scope (such as contributions to the documentation)
or you know exactly what you're doing
(you're a seasoned echopype contributor), the CI can be skipped.
This is done by including the string "[skip ci]" in your last commit's message.


Documentation development
-------------------------

Function and object doc strings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For inline strings documenting functions and objects ("doc strings"), we use the
`numpydoc style (Numpy docstring format) <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

Jupter Book ReadTheDocs documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Echopype documentation (`<https://echopype.readthedocs.io>`_) is based on `Jupyter Book <https://jupyterbook.org/en/stable/intro.html>`_
which are rendered under the hood with `Sphinx <https://www.sphinx-doc.org>`_. The documentation is hosted at
`Read The Docs <https://readthedocs.org>`_. The documentation package dependencies are found
in the ``docs/requirements.txt`` file, and the source documentation files are in the ``docs/source`` directory. The echopype development
conda environment will install all required dependencies.

Our documentation are currently a mixture of the following file formats:

- `CommonMark <https://commonmark.org/>`_ and `MySt <https://jupyterbook.org/en/stable/content/myst.html>`_ Markdown
- `Jupyter Notebook <https://jupyter-notebook.readthedocs.io/en/latest/notebook.html>`_
- `reStructuredText <https://docutils.sourceforge.io/rst.html>`_

To run Jupyter Book locally:

.. code-block:: bash

    jupyter-book build docs/source --path-output docs

To view the generated HTML files generated by Jupyter Book, open the
``docs/_build/html/index.html`` in your browser.

Jupyter Book `configurations <https://jupyterbook.org/en/stable/customize/config.html>`_ can be found in the `docs/source/_config.yml` file.
The `table of contents <https://jupyterbook.org/en/stable/structure/toc.html>`_ arragements for the sidebar can be found in ``docs/source/_toc.yml`` file.

When ready to commit your changes, please pull request your changes to the `stable` branch. Once the PR is submitted, the `pre-commit` CI will run for basic spelling and formatting check (See the `pre-commit hooks section <contributing.html#pre-commit-hooks>`_ for more details). Any changes from the `pre-commit` check have to be pulled to your branch (via `git pull`) before your push further commits. You will also be able to view the newly built doc in the PR via the "docs/readthedocs.org:echopype" entry shown below.

.. image:: https://user-images.githubusercontent.com/15334215/165646718-ebfd4041-b110-4b54-a5b9-54a7a08bc982.png

Updates to the documentation that are based on the current echopype release (that is,
not involving echopype API changes) should be merged into the GitHub ``stable`` branch.
These updates will then become available immediately on the default ReadTheDocs version.
Examples of such updates include fixing spelling mistakes, expanding an explanation,
and adding a new section that documents a previously undocumented feature.

Documentation versions
~~~~~~~~~~~~~~~~~~~~~~

`<https://echopype.readthedocs.io>`_ redirects to the documentation ``stable`` version,
`<https://echopype.readthedocs.io/en/stable/>`_, which is built from the ``stable`` branch
on the ``echopype`` GitHub repository. In addition, the ``latest`` version
(`<https://echopype.readthedocs.io/en/latest/>`_) is built from the ``dev`` branch and
therefore it reflects the bleeding edge development code (which may occasionally break
the documentation build). Finally, each new echopype release is built as a new release version
on ReadTheDocs. Merging pull requests into ``stable`` or ``dev`` or issuing a new
tagged release will automatically result in a new ReadTheDocs build for the
corresponding version.

We also maintain a test version of the documentation at `<https://doc-test-echopype.readthedocs.io/>`_
for viewing and debugging larger, more experimental changes, typically from a separate fork.
This version is used to test one-off, major breaking changes.
