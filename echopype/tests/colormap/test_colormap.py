import echopype.colormap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import pytest


@pytest.mark.parametrize(
    "cmap_name",
    [
        ("ep.ek500"),
    ],
    ids=["ek500"],
)
def test_colormap(cmap_name):
    fig = plt.imshow(np.random.rand(10,10), cmap=cmap_name)
    assert isinstance(fig, mpl.image.AxesImage) is True
