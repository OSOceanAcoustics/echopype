import matplotlib as mpl
import numpy as np

__cmap_colors = {
    "ek500": {
        "rgb": (
            np.array(
                [
                    [159, 159, 159],  # light grey
                    [95, 95, 95],  # grey
                    [0, 0, 255],  # dark blue
                    [0, 0, 127],  # blue
                    [0, 191, 0],  # green
                    [0, 127, 0],  # dark green
                    [255, 255, 0],  # yellow
                    [255, 127, 0],  # orange
                    [255, 0, 191],  # pink
                    [255, 0, 0],  # red
                    [166, 83, 60],  # light brown
                ]
            )
            / 255
        ),
        "under": "1",  # white
        "over": np.array([120, 60, 40]) / 255,  # dark brown
    }
}


def _create_cmap(rgb, under=None, over=None):
    # When echopype supports only matplotlib>=3.11, the under and over can be in
    # the constructor
    return mpl.colors.ListedColormap(rgb).with_extremes(under=under, over=over)


cmap_d = {}
cmapnames = ["ek500"]

# add colormaps and reversed to dictionary
for cmapname in cmapnames:
    colors_d = __cmap_colors[cmapname]
    rgb = colors_d["rgb"]
    cmap_d[cmapname] = _create_cmap(
        rgb, under=colors_d.get("under", None), over=colors_d.get("over", None)
    )
    cmap_d[cmapname].name = cmapname
    cmap_d[cmapname + "_r"] = _create_cmap(
        rgb[::-1, :],
        under=colors_d.get("over", None),
        over=colors_d.get("under", None),
    )
    cmap_d[cmapname + "_r"].name = cmapname + "_r"

    # Register the cmap with matplotlib
    rgb_with_alpha = np.zeros((rgb.shape[0], 4))
    rgb_with_alpha[:, :3] = rgb
    rgb_with_alpha[:, 3] = 1.0  # set alpha channel to 1
    reg_map = mpl.colors.ListedColormap(rgb_with_alpha, "ep." + cmapname)
    reg_map = reg_map.with_extremes(
        under=colors_d.get("under", None), over=colors_d.get("over", None)
    )
    mpl.colormaps.register(cmap=reg_map)

    # Register the reversed map
    reg_map_r = mpl.colors.ListedColormap(rgb_with_alpha[::-1, :], "ep." + cmapname + "_r")
    reg_map_r = reg_map_r.with_extremes(
        over=colors_d.get("under", None),
        under=colors_d.get("over", None),
    )
    mpl.colormaps.register(cmap=reg_map_r)

# make colormaps available to call
locals().update(cmap_d)
