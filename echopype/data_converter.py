#!/usr/bin/env python

"""
Command line tool for converting sonar data into common netCDF format.
Currently in this file is the previous incarnation using HDF5 as the storage format.
This will be changed to use the functions under /convert and support multiple source data formats.
"""

import argparse
import os.path
from echopype.unpack_ek60 import raw2hdf5_initiate, raw2hdf5_concat
from datetime import datetime as dt


def main():
    parser = argparse.ArgumentParser(description='Convert EK60 *.raw to HDF5')

    parser.add_argument('-o','--output-file',action='store', nargs=1, help='output HDF5 filename')
    parser.add_argument('-i','--input-file',action='store', nargs='+', help='input *.raw filename')

    args = parser.parse_args()

    for cnt,f in zip(range(len(args.input_file)),args.input_file):
        if not(os.path.isfile(args.output_file[0])) and cnt==0:
            raw2hdf5_initiate(f,args.output_file[0])
            print('%s  creating and saving data to : %s' % (dt.now().strftime('%H:%M:%S'), args.output_file[0]))
        else:
            raw2hdf5_concat(f,args.output_file[0])
            print('%s  appending data to: %s' % (dt.now().strftime('%H:%M:%S'), args.output_file[0]))

if __name__ == '__main__':
    main()
