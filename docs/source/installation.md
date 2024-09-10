# Installation and examples

## Installation

Echopype is available and tested for Python 3.9-3.11. The latest release can be installed through conda (or mamba, see below) via the [conda-forge channel](https://anaconda.org/conda-forge/echopype):
```shell
# Install via conda-forge
$ conda install -c conda-forge echopype
```

It is available via [PyPI](https://pypi.org/project/echopype):
```shell
# Install via pip
$ pip install echopype
```

:::{note}
We are working on adding support for Python 3.12 soon!
:::

:::{attention}
It's common to encounter the situation that installing packages using Conda is slow or fails,
because Conda is unable to resolve dependencies.
We suggest using Mamba to get around this.
See [Mamba's documentation](https://mamba.readthedocs.io/en/latest/) for installation and usage.
One can replace `conda` with `mamba` in the above commands when creating the environment and installing additional packages.
:::

Previous releases are also available on conda and PyPI.

For instructions on installing a development version of echopype,
see the [](contributing) page.


## Example notebooks

The [echopype-examples](https://github.com/OSOceanAcoustics/echopype-examples) repository contains multiple Jupyter notebook examples illustrating the Echopype workflow. See the [rendered pages](https://echopype-examples.readthedocs.io/) before trying them out!
