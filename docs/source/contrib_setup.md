(contrib:setup)=
# Development setup

Thank you for your interest in contributing to Echopype! In this page you will find information on the development workflow, setting up a development environment, and details about testing and documentation.



## Trunk-based development
We have recently moved to follow [trunk-based development](https://www.atlassian.com/continuous-delivery/continuous-integration/trunk-based-development) to streamline the process and reduce repo management overhead.
The main thing to keep in mind is to set the PR target to the `main` branch in the `upstream` repository (the one sitting under the echostack-org GitHub organization).
We will no longer use a `dev` branch.



## Development environment

To create an environment for developing Echopype, we recommend the following steps:

1. Fork the Echopype repository, clone your fork to your machine, then in `git remote` set your fork as the `origin` and the echostack-org repository as `upstream`:

    ```shell
    # Clone your fork
    git clone https://github.com/YOUR_GITHUB_USERNAME/echopype.git

    # Go into the cloned repo folder
    cd echopype

    # Add the echostack-org repository as upstream
    git remote add upstream https://github.com/echostack-org/echopype.git
    ```

2. Create a virtual Python environment and build an editable version of the echopype package. We
suggest doing this with `conda` or `uv`.

```{tab} Conda

  ```shell
  # Create and activate the development environment
  conda create -c conda-forge -n echopype-dev --yes python=3.12
  conda activate echopype-dev

  # Upgrade pip to support dependency groups
  python -m pip install --upgrade pip

  # Install echopype in editable mode with development and testing dependencies
  python -m pip install -e . --group dev --group test

  # Optional plotting dependencies
  # python -m pip install -e ".[plot]"
  ```

```{tab} Uv

  - Install `uv` (instructions [here](https://docs.astral.sh/uv/getting-started/installation/)).
  - From the echopype repository directory, run:

  ```shell
  # Create .venv and install echopype in editable mode
  # with the default development and testing dependencies.
  # The ~/.python-version file in the repository sets which version of Python is installed (3.12 at the moment).
  uv sync
  ```

:::{tip}
If using conda, we recommend using Mamba to get around Conda's sometimes slow or stuck behavior when solving dependencies.
See [Mamba's documentation](https://mamba.readthedocs.io/en/latest/) for installation and usage.
The easiest way to get a minimal installation is through [Miniforge](https://conda-forge.org/download/).
One can replace `conda` with `mamba` in the above commands when creating the environment and installing additional packages.
:::

## Testing infrastructure

### Test data files

Test data are managed using [Pooch](https://www.fatiando.org/pooch/latest/)
and downloaded to a local cache when first needed.

Some integration tests access these data through local HTTP and S3-compatible
services. The Docker setup described below copies the Pooch-managed test data
into these services.

If your contribution requires new test data, contact the maintainers
(@leewujung, @ctuguinay, @LOCEANlloydizard) to have them added to the test-data collection.

### Running the tests

To run all Echopype tests:

```shell
uv run pytest -vv
```

To run tests for specific modules, provide their paths separated by spaces:

```shell
uv run pytest -vv echopype/calibrate/calibrate_ek.py echopype/mask/api.py
```

To run specific test files:

```shell
uv run pytest -vv echopype/tests/convert/test_convert_azfp.py echopype/tests/clean/test_noise.py
```

Most tests use data directly from the local Pooch cache. Some integration tests
also require local HTTP and S3-compatible services provided through Docker.

On Linux/macOS, start the services with:

```shell
uv run python .ci_helpers/docker/setup-services.py --deploy
```

When finished, stop the services with:

```shell
uv run python .ci_helpers/docker/setup-services.py --tear-down
```

On Windows PowerShell, start the services with:

```powershell
uv run python .ci_helpers/setup-services-windows.py start
```

When finished, stop the services with:

```powershell
uv run python .ci_helpers/setup-services-windows.py stop
```

## pre-commit hooks

The echopype development environment includes [pre-commit](https://pre-commit.com),
and useful pre-commit "hooks" have been configured in the
[.pre-commit-config.yaml file](https://github.com/echostack-org/echopype/blob/main/.pre-commit-config.yaml).
Current hooks include file formatting (linting) checks
(trailing spaces, trailing lines, JSON and YAML format checks, etc)
and Python style autoformatters (`Ruff`, `black` and `isort`).

To run pre-commit hooks locally, run `pre-commit install` before running the
docker setup-service deploy statement described above.
The hooks will run automatically during `git commit` and will give you
options as needed before committing your changes.
You can also run `pre-commit` before actually doing `git commit` as you edit the code,
by running `pre-commit run --all-files`.
See the [pre-commit usage documentation](https://pre-commit.com/#usage) for details.

## Documentation

### Function and object docstrings

For inline strings documenting functions and objects ("docstrings"),
we use the [numpydoc style](https://numpydoc.readthedocs.io/en/latest/format.html) (Numpy docstring format).

### General setup

Echopype documentation (https://echopype.readthedocs.io) is based on [Jupyter Book](https://jupyterbook.org/en/stable/intro.html),
which are rendered under the hood with [Sphinx](https://www.sphinx-doc.org).
The documentation is hosted on [Read The Docs](https://readthedocs.org).

To build the documentation locally, run:
```{tab} Conda

  ```shell
  # Install documentation dependencies
  python -m pip install --group docs

  # Build the documentation
  jupyter-book build docs/source --path-output docs
  ```

```{tab} Uv

  ```shell
  uv run --group docs sphinx-build -b html ./docs/source ./docs/_build
  ```

To view the generated HTML files open `docs/_build/html/index.html` in your browser.

For some quick orientation of where things are:
- Documentation dependencies are defined in the `docs` dependency group in `pyproject.toml`
- The documentation source files are in the `docs/source` directory
- The Jupyter Book [configurations](https://jupyterbook.org/en/stable/customize/config.html)
  is in `docs/source/_config.yml`
- The [table of contents](https://jupyterbook.org/en/stable/structure/toc.html) for the sidebar
  is in `docs/source/_toc.yml`

### Versions

ReadTheDocs defaults to having its `stable` version tracking the most recent release and the `main` version tracking the latest changes in the `main` branch of the repository. We follow this pattern for our documentation. See [RTD Versions](https://docs.readthedocs.io/en/stable/versions.html) for more information.



(contrib:setup_CI)=
## GitHub Actions for continuous integration (CI)
When a PR is created, the CI will run through all tests, basic spelling and formatting checks (via pre-commit), and build the documentation.
You can check the test results in a section at the bottom of the PR like below:
![ci_runs](./images/CI_checks.png)

To see the newly built documentation, click  "Details" to the right of the
`docs/readthedocs.org:echopype` entry shown above.
