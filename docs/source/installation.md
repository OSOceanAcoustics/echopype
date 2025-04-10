# Installation and examples

## Installation

Echopype is available and tested for Python 3.10-3.12. The latest release can be installed through conda (or mamba, see below) via the [conda-forge channel](https://anaconda.org/conda-forge/echopype):
```shell
# Install via conda-forge
$ conda install -c conda-forge echopype
```

It is available via [PyPI](https://pypi.org/project/echopype):
```shell
# Install via pip
$ pip install echopype
```

:::{tip}
We recommend using Mamba to get around Conda's somtimes slow or stuck behavior when solving dependencies.
See [Mamba's documentation](https://mamba.readthedocs.io/en/latest/) for installation and usage.
The easiest way to get a minimal installation is through [Miniforge](https://conda-forge.org/download/).
One can replace `conda` with `mamba` in the above commands when creating the environment and installing additional packages.
:::

Previous releases are also available on conda and PyPI.

For instructions on installing a development version of echopype,
see the [](contrib_setup) page.


## Example notebooks

The [echopype-examples](https://github.com/OSOceanAcoustics/echopype-examples) repository contains multiple Jupyter notebook examples illustrating the Echopype workflow. See the [rendered pages](https://echopype-examples.readthedocs.io/) before trying them out!
