#!/usr/bin/env python3

"""
10 day ship track plot
       
e.g., python plot_hake_10days.py /media/paulr/Elements/ncei_data/shimada/sh1701/
"""


import os
import sys
from datetime import datetime as dt
from glob import glob
import xarray as xr
import csv
import logging
from echopype.model.ek60 import ModelEK60
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from topomaps import add_etopo2
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import gc
import numpy as np


allncs = sorted(glob(os.path.join(sys.argv[1], "ek60_nc/*[0-9].nc")))

alldates = []

for i in range(0, len(allncs)):
    alldates.append(allncs[i].split('-')[-2])

uniqueDates = sorted(set(alldates))


dx = dy = 0.25

for i in range(0, len(uniqueDates), 10):
   ### just someplace from which to start
   extent = np.array([-124.74383666666667, -123.798, 43.99783333333333, 44.8765])
   tenDates = uniqueDates[i:i+10]
   print(tenDates)
   for date in tenDates:
      try:
         nc_files = [j for j in allncs if date in j]
         nc_plat = xr.open_mfdataset(nc_files, group='Platform', concat_dim='location_time')
         extent_new = np.array((nc_plat.longitude.values.min() - dx, nc_plat.longitude.values.max() + dx, nc_plat.latitude.values.min() - dy, nc_plat.latitude.values.max() + dy))
         nc_plat.close()
         print(extent_new)
         for i in [0,2]:
            if extent_new[i] <= extent[i]:
               extent[i] = extent_new[i]
         for i in [1,3]:
            if extent_new[i] >= extent[i]:
               extent[i] = extent_new[i]

      except Exception as e:
         print('An error occurred:' + str(e))
     

   plt.figure(figsize=[11, 8.5])
   ax = plt.axes(projection=ccrs.PlateCarree())
   ax.set_extent(extent)
   ax.coastlines(resolution='50m')

   add_etopo2(extent, ax)

   gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.2, linestyle='-', draw_labels=True)
   gl.xlabels_top = False
   gl.ylabels_right= False
   gl.xlabel_style = {'rotation': 45}

   ### different color every day
   color_list = [
         "#A6CEE3",
         "#1F78B4",
         "#B2DF8A",
         "#33A02C",
         "#FB9A99",
         "#E31A1C",
         "#FDBF6F",
         "#FF7F00",
         "#CAB2D6",
         "#6A3D9A"
   ]

   leglist = []
   i =0
  
   for date in tenDates:
      try:
         nc_files = [i for i in allncs if date in i]
         ncs = xr.open_mfdataset(nc_files, group='Platform', concat_dim='location_time')
         ax.plot(ncs.longitude.values, ncs.latitude.values, linewidth=3, color=color_list[i])
         leglist.append(date)
         i = i+1

      except Exception as e:
         print('An error occurred:' + str(e))
     
      

   ax.legend(leglist, bbox_to_anchor=(1.05, 1), loc='upper left')
   pngname = os.path.join(sys.argv[1], 'ship_track_10day/') + tenDates[0] + "-" + tenDates[-1] + '-shiptrack.png'
   print("Saving " + pngname)
   plt.savefig(pngname, dpi=120)
   plt.clf() 
   plt.close()
   gc.collect()

