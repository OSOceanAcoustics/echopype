"""
Visualization module to quickly plot raw, Sv, and MVBS dataset.

**NOTE: To use this subpackage. `Matplotlib` and `cmocean` package must be installed.**
"""
import warnings
from .api import create_echogram
from . import cm

__all__ = ["create_echogram", "cm"]

warnings.warn(
    (
        "`echopype.visualization` is deprecated and will be removed in an upcoming release. \n"
        "Use Echoshader for visualization functions instead. \n"
        "Repository: https://github.com/OSOceanAcoustics/echoshader \n"
        "Echogram plotting: https://echoshader.readthedocs.io/en/latest/version_0.1.0/echogram_examples.html"  # noqa
    ),
    DeprecationWarning,
    2,
)
