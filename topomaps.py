import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from palettable import colorbrewer

LAND = colorbrewer.get_map("Greens", "sequential", 9)
OCEAN = colorbrewer.get_map("Blues", "sequential", 9, reverse=True)
GREY = colorbrewer.get_map("Greys", "sequential", 9, reverse=True)
#LAND_OCEAN = np.array(OCEAN.mpl_colors + LAND.mpl_colors)
LAND_GREY = np.array(GREY.mpl_colors + LAND.mpl_colors)


def make_map(extent, figsize=(12, 12), projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": projection}
    )
    ax.set_extent(extent)
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.ylines = gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.coastlines(resolution="50m")
    return fig, ax


def add_etopo2(extent, ax, levels=None):
    import iris
    url = (
        "http://gamone.whoi.edu/thredds/dodsC/usgs/data0/bathy/ETOPO2v2c_f4.nc"
    )
    cube = iris.load_cube(url)
    lon = iris.Constraint(x=lambda cell: extent[0] <= cell <= extent[1])
    lat = iris.Constraint(y=lambda cell: extent[2] <= cell <= extent[3])
    cube = cube.extract(lon & lat)
    lons = cube.coord("x").points
    lats = cube.coord("y").points
    if not levels:
        levels = sorted(
            set(
                np.r_[
                    np.linspace(cube.data.min(), -500, 5),  # bathy
                    [-400, -145, -10],  # coast
                    np.linspace(100, cube.data.max(), 5),  # topo
                ].astype(int)
            )
        )

    ax.contourf(
        lons,
        lats,
        cube.data,
        levels=levels,
        colors=LAND_GREY,
        zorder=0,
        alpha=0.5,
    )
