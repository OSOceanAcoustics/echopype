# Contributing to echopype
========================


:::{note}
We welcome your contributions, large or small! 
- Contributions to both code and documentation via Pull Requests are highly appreciated
- Bug reports, feature requests, and questions via Issues are also welcome
:::


## Contributing with Git and GitHub

### Bug reports, feature requests, questions
Please ask questions, report bugs, or request new features via GitHub issues.
If you're new to GitHub, checking out these tips for submitting issues:
`"Creating issues on GitHub" <https://medium.com/nyc-planning-digital/writing-a-proper-github-issue-97427d62a20f>`_.

### Contributions to code and documentation
We use the fork-branch-pull request (PR) workflow to add new code into Echopype. 
If you are new to this workflow, check out this [tutorial](https://medium.com/swlh/forks-and-pull-requests-how-to-contribute-to-github-repos-8843fac34ce8).

We have recently moved away from Gitflow development to [trunk-based development](https://www.atlassian.com/continuous-delivery/continuous-integration/trunk-based-development) to streamline the process and reduce repo management overhead.
The main thing to keep in mind is to set the PR target to the `main` branch in the `upstream` repository (the one sitting under the OSOceanAcoustics GitHub organization).
We will no longer use a `dev` branch.

We encourage all code contributions to be accompanied by tests and documentations when appropriate.
We may ask for these when reviewing the PRs.
If you have added new tests but the continuous integration (CI) workflows is not triggered to run, ping @leewujung to get them started.



## Installation for echopype development
-------------------------------------

To create an environment for developing Echopype, we recommend the following steps:

1. Fork the Echopype repository following the guide above, clone your fork, then in `git remote` set your fork as the `origin` and the OSOceanAcoustics repository as `upstream`
    ```shell
    # Clone your fork
    git clone https://github.com/YOUR_GITHUB_USERNAME/echopype.git

    # Go into the cloned repo folder
    cd echopype

    # Add the OSOceanAcoustics repository as upstream
    git remote add upstream https://github.com/OSOceanAcoustics/echopype.git
    ```

2. Create a conda environment using `mamba`, and follow the steps below:
    ```shell
    # Create a conda environment using the supplied requirements files
    # Note the last one docs/requirements.txt is only required for building docs
    conda create -c conda-forge -n echopype --yes python=3.9 --file requirements.txt --file requirements-dev.txt --file docs/requirements.txt

    # Switch to the newly built environment
    conda activate echopype

    # ipykernel is recommended, in order to use with JupyterLab and IPython for development
    # We recommend you install JupyterLab separately
    conda install -c conda-forge ipykernel

    # install echopype in editable mode (setuptools "develop mode")
    # plot is an extra set of requirements that can be used for plotting.
    # the command will install all the dependencies along with plotting dependencies.
    pip install -e ".[plot]"
    ```

:::{note}
It's common to encounter the situation that installing packages using Conda is slow or fails,
because Conda is unable to resolve dependencies.
We suggest using Mamba to get around this.
See [Mamba's documentation](https://mamba.readthedocs.io/en/latest/) for installation and usage.
:::



## Tests and test infrastructure

### Test data files

Currently, test data are stored in a private Google Drive folder and
made available via the [`cormorack/http`](https://hub.docker.com/r/cormorack/http)
Docker image on Docker hub.
The image is rebuilt daily when new test data are added.
If you tests require adding new test data, ping @leewujung or @ctuguinay
to get them added to the the Google Drive.

In the near future we plan to migrate all test data to GitHub Release Assets,
to keep test data versioned and directly assocaited with the repo.


### Running the tests

To run the echopype unit tests found in `echopype/tests`, 
[`Docker`](https://docs.docker.com/get-docker/) needs to be installed. 
[`docker-compose`](https://docs.docker.com/compose/) is also needed, 
but it should already be installed in the development environment created above.

To run the tests:
```shell
# Install and/or deploy the echopype docker containers for testing.
# Test data files will be downloaded
python .ci_helpers/docker/setup-services.py --deploy

# Run all the tests. But first make sure the
# echopype development conda environment is activated
python .ci_helpers/run-test.py --local --pytest-args="-vv"

# When done, "tear down" the docker containers
python .ci_helpers/docker/setup-services.py --tear-down
```

The tests include reading and writing from locally set up (via docker)
http and [S3 object-storage](https://en.wikipedia.org/wiki/Amazon_S3) sources,
the latter via [minio](https://minio.io).

[`.ci_helpers/run-test.py`](https://github.com/OSOceanAcoustics/echopype/blob/main/.ci_helpers/run-test.py)
will execute all tests.
The entire test suite can take a few minutes to run.
You can use `run-test.py` to run only tests for specific subpackages
(`convert`, `calibrate`, etc) by passing a comma-separated list:
```shell
# Run only tests associated with the calibrate and mask subpackages
python .ci_helpers/run-test.py --local --pytest-args="-vv" echopype/calibrate/calibrate_ek.py,echopype/mask/api.py
```
or specific test files by passing a comma-separated list:
```shell
# Run only tests in the test_convert_azfp.py and test_noise.py files
python .ci_helpers/run-test.py --local --pytest-args="-vv"  echopype/tests/convert/test_convert_azfp.py,echopype/tests/clean/test_noise.py
```

For `run-test.py` usage information, use the ``-h`` argument:
```shell
`python .ci_helpers/run-test.py -h`
```


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

### Continuous integration GitHub Actions


echopype makes extensive use of GitHub Actions for continuous integration (CI)
of unit tests and other code quality controls. Every pull request (PR) triggers the CI.
See `echopype/.github/workflows <https://github.com/OSOceanAcoustics/echopype/tree/main/.github/workflows>`_,
especially `pr.yaml <https://github.com/OSOceanAcoustics/echopype/blob/main/.github/workflows/pr.yaml>`_.

The entire test suite can be a bit slow, taking up to 40 minutes or more.
To mitigate this, the CI default is to run tests only for subpackages that
were modified in the PR; this is done via ``.ci_helpers/run-test.py``
(see the `Running the tests`_ section). To have the CI execute the
entire test suite, add the string "[all tests ci]" to the PR title.
Under special circumstances, when the submitted changes have a
very limited scope (such as contributions to the documentation)
or you know exactly what you're doing
(you're a seasoned echopype contributor), the CI can be skipped.
This is done by adding the string "[skip ci]" to the PR title.


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

To view the HTML files generated by Jupyter Book, open the
``docs/_build/html/index.html`` in your browser.

Jupyter Book `configurations <https://jupyterbook.org/en/stable/customize/config.html>`_ can be found in the ``docs/source/_config.yml`` file.
The `table of contents <https://jupyterbook.org/en/stable/structure/toc.html>`_ arrangements for the sidebar can be found in ``docs/source/_toc.yml`` file.

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
