#!/usr/bin/env python3

"""
Command line tool for setting up directory structure, converting .RAW files to netCDF .nc, plotting ping intervals, and generating  a .csvfile of Hake survey data.

example: python survey_hake.py /media/paulr/Elements/ncei_data/shimada/ sh1707

"""


import os
import sys
from datetime import datetime as dt
import xarray as xr
import matplotlib.pyplot as plt
import csv
import logging
from glob import glob
import numpy as np
import echopype

def main():  
   ### Organize files
   basedir = sys.argv[1]
   cruisename = sys.argv[2]
   dirs = ['echogram', 'ek60_convert_error', 'ek60_nc', 'ping_interval', 'ship_track_01day', 'ship_track_10day']
   for subdir in dirs:
      try:
         os.mkdir(os.path.join(basedir, cruisename, subdir))
      except OSError as error:
         print(error)

   rawfiles = sorted(glob(os.path.join(basedir, cruisename, 'ek60_raw', '*raw')))
   ncsdir = os.path.join(basedir, cruisename, 'ek60_nc')
   plotsdir = os.path.join(basedir, cruisename, 'ping_interval')

   ### Check if a data survey csv file exists in the folder already, and, if not, create one.
   csvfile = os.path.join(basedir, cruisename, cruisename + '_summary.csv')
   if os.path.exists(csvfile) == False:
      with open(csvfile, 'w', newline='') as csvfiletowrite:
         csvwriter = csv.writer(csvfiletowrite, lineterminator='\n', delimiter=',')
         csvwriter.writerow(['File_name','Start_ping_time', 'End_ping_time', 'Pinging_interval', 'Sonar_freq', 'Sample_Interval', 'Transmit_duration', 'Transmit_power', 'Sound_speed', 'Absorption'])

   ### Iterate over all the .RAW files and convert them.
   for rfile in rawfiles:
      try:
         if not os.path.exists(os.path.join(ncsdir, os.path.basename(rfile).split('.')[0] + '.nc')):
            tmp = echopype.convert.ConvertEK60(rfile)
            tmp.raw2nc()
            nc_created = os.path.basename(rfile).split('.')[0] + '.nc'
            os.rename(os.path.join(os.path.dirname(rfile), nc_created), os.path.join(ncsdir, nc_created))
            del tmp
         else:
            print(rfile + " has been converted already. Skipping...")       

      except Exception as e:
         for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

         logfilename = os.path.basename(rfile).split('.')[0] + '-error-log.txt'
         logging.basicConfig(filename = os.path.join(basedir, cruisename, 'ek60_convert_error', logfilename), filemode='w', level=logging.WARNING)
         print('An error occurred:' + str(e))
         logging.exception(rfile + "  Conversion Error")  



   ### Iterate over all the nc files, extract info for the survey, write to csv, and logany errorsalong the way
   ncfiles = sorted(glob(os.path.join(ncsdir, '*[0-9].nc')))
   for ncfile in ncfiles:
      try:
         print("Working on " + ncfile)
         filebase = os.path.basename(ncfile).split('.')[0]
         ncfile_beam = xr.open_dataset(ncfile, group = 'Beam')
         start_ping_time = ncfile_beam.ping_time.min().values
         end_ping_time = ncfile_beam.ping_time.max().values
         ### calc and plot ping intervals. 
         pings = ncfile_beam.ping_time.values
         ### First, let's just look at the intervals for anomalies
         ping_ints = np.diff(pings)
         plt.plot(ping_ints)
         plt.ylabel('Ping Intervals (s)')
         plt.xlabel('Interval ID')
         plt.savefig(os.path.join(plotsdir, (filebase + '_ping_diffs.png')), dpi=120)
         plt.close()
         ### If there are any significant 2nd order differences record their indicies
         pings_2nd = np.diff(np.diff(pings))
         pinging_interval = np.where(pings_2nd > np.timedelta64(100,'ms'))[0] + 2
         
         sonar_freq = ncfile_beam.frequency.values
         sample_interval = ncfile_beam.sample_interval.values
         transmit_duration = ncfile_beam.transmit_duration_nominal.values
         transmit_power = ncfile_beam.transmit_power.values

         ncfile_env = xr.open_dataset(ncfile, group = 'Environment')
         sound_speed = ncfile_env.sound_speed_indicative.values
         absorption = ncfile_env.absorption_indicative.values
      
         #TODO csvwriter puts carriage returns in the pinging_interval array. Suppress it!
         with open(csvfile, 'a', newline='') as csvfiletowrite:
            csvwriter = csv.writer(csvfiletowrite, lineterminator='\n', delimiter=',')
            csvwriter.writerow([ncfile.split('ncei_data')[1], start_ping_time, end_ping_time, pinging_interval, sonar_freq, sample_interval, transmit_duration, transmit_power, sound_speed, absorption])
         
      except Exception as e:
         logfilename = os.path.basename(ncfile).split('.')[0] + '-error-log.txt'
         logging.basicConfig(filename = os.path.join(basedir, cruisename,'ek60_convert_error', logfilename), filemode='w', level=logging.WARNING)
         print('An error occurred:' + str(e))
         logging.exception(str(e))
      

   ### Append rows for files which didn't convert
   errorFiles = os.listdir(os.path.join(basedir, cruisename, 'ek60_convert_error'))
   for efile in errorFiles:
      efilename = os.path.join(basedir, cruisename, 'ek60_convert_error', efile).split('ncei_data')[1]
      with open(csvfile, 'a', newline='') as csvfiletowrite:
            csvwriter = csv.writer(csvfiletowrite, lineterminator='\n', delimiter=',')
            csvwriter.writerow([efilename, 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])





if __name__ == '__main__':
    main()
