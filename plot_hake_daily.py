#!/usr/bin/env python3

"""
Command line tool for generating "daily" plots of Hake survey data.

NOTE: THIS SCRIPT ASSUMES THE FOLLOWING SUBDIRECTORIES:
   .../shimada/[cruise]/ek60_raw/
                       /ek60_nc/
                       /echogram/
                       /ship_track_01day/
                       /ship_track_10day/

example usage: python plot_hake_daily.py [/path/to/data/basedir] D[YearMonthDay] 
example usage: python plot_hake_daily.py /media/paulr/Elements/ncei_data/shimada/sh1701/ D20170720

This script will determine if there are more than 10 files for a day and create a plot per 10, or all if fewer.
               
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

def main():
   basedir = sys.argv[1]
   path_to_files = os.path.join(basedir, 'ek60_nc')
   files_date = sys.argv[2]

   ###### Echograms
   ### must calibrate
   nc_files = sorted(glob(os.path.join(path_to_files, '*' + files_date + '*[0-9].nc')))
   for nc_file in nc_files:
      if os.path.exists(nc_file.split('.')[0] + '_Sv.nc'):
         print("Calibration already completed for" + nc_file)
      else:
         nc = ModelEK60(nc_file)
         nc.calibrate(save=True)
 
   Sv_files = sorted(glob(os.path.join(path_to_files, '*' + files_date + '*Sv.nc')))
   print("Plotting echograms...")
   for i in range(0, len(Sv_files), 10):
      Sv_chunk = Sv_files[i:i+10]
      lastfile = os.path.basename(Sv_chunk[-1]).split('-')[2].split('_')[0]
      pngname = os.path.join(basedir, 'echogram', os.path.basename(Sv_files[i]).split('_')[0] + '-' + lastfile + '-echo.png')
      Sv = xr.open_mfdataset(Sv_files[i:i+10], combine='by_coords')
      plt.figure(figsize=[11, 8.5])
      plt.subplot(3, 1, 1)
      Sv.Sv.sel(frequency=18000).plot(vmax=-40, vmin=-100, cmap='Spectral_r', x='ping_time')
      plt.gca().invert_yaxis()
      plt.title(files_date + '  (frequency=18000)')
      plt.subplot(3, 1, 2)
      Sv.Sv.sel(frequency=38000).plot(vmax=-40, vmin=-100, cmap='Spectral_r', x='ping_time')
      plt.gca().invert_yaxis()
      plt.title(files_date + '  (frequency=38000)')
      plt.subplot(3, 1, 3)
      Sv.Sv.sel(frequency=120000).plot(vmax=-40, vmin=-100, cmap='Spectral_r', x='ping_time')
      plt.gca().invert_yaxis()
      plt.title(files_date + '  (frequency=120000)')
      plt.tight_layout() 
      plt.savefig(pngname, dpi=120)
      plt.clf()
      plt.close()
      gc.collect()
 

   ##### Ship tracks
   print("Plotting ship tracks...")
   for i in range(0, len(nc_files), 10):
      nc_plat = xr.open_mfdataset(nc_files, group='Platform', concat_dim='location_time')
      dx = dy = 0.25
      extent = (nc_plat.longitude.values.min() - dx, nc_plat.longitude.values.max() + dx, nc_plat.latitude.values.min() - dy, nc_plat.latitude.values.max() + dy)
      nc_plat.close() 
      plt.figure(figsize=[8.5, 11])
      ax = plt.axes(projection=ccrs.PlateCarree())
      ax.set_extent(extent)
      ax.coastlines(resolution='50m')
      #ax.add_feature(cartopy.feature.OCEAN)
      add_etopo2(extent, ax)

      gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.2, linestyle='-', draw_labels=True)
      gl.xlabels_top = False
      gl.ylabels_right= False
      gl.xlabel_style = {'rotation': 45}

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

      ncs_chunk = nc_files[i:i+10]
      leglist = []
      for i in range(0, len(ncs_chunk)):
         ncs = xr.open_dataset(ncs_chunk[i], group='Platform')
         ax.plot(ncs.longitude.values, ncs.latitude.values, linewidth=3, color=color_list[i])
         leglist.append(os.path.basename(ncs_chunk[i]))
      
      ax.legend(leglist, bbox_to_anchor=(.22, -.2), loc='upper left')
      lastfile = os.path.basename(ncs_chunk[-1]).split('-')[2].split('.')[0]
      pngname = os.path.join(basedir, 'ship_track_01day', os.path.basename(nc_files[i]).split('.')[0] + '-' + lastfile + '-shiptrack.png')
      plt.savefig(pngname, dpi=120)
      plt.clf() 
      plt.close()
      gc.collect()

if __name__ == '__main__':
    main()
