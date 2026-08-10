# Installation and examples

## Installation

Echopype is available and tested for Python 3.12–3.14. The latest release can be installed through Conda (or Mamba, see below) via the [conda-forge channel](https://anaconda.org/conda-forge/echopype):

```shell
# Install via conda-forge
conda install -c conda-forge echopype
```

It is also available via [PyPI](https://pypi.org/project/echopype):

```shell
# Install via pip
pip install echopype
```

:::{tip}
We recommend using Mamba to avoid Conda's sometimes slow or stuck behavior when solving dependencies.
See [Mamba's documentation](https://mamba.readthedocs.io/en/latest/) for installation and usage.
The easiest way to get a minimal installation is through [Miniforge](https://conda-forge.org/download/).
You can replace `conda` with `mamba` in the commands above when creating environments and installing additional packages.
:::

Previous releases are also available on Conda and PyPI.

For instructions on installing a development version of Echopype,
see the [](contrib:setup) page.

## Example notebooks

The [echopype-examples](https://github.com/echostack-org/echopype-examples) repository contains multiple Jupyter notebook examples illustrating the Echopype workflow. See the [rendered pages](https://echopype-examples.readthedocs.io/) before trying them out!
