#!/usr/bin/env python

"""
Command line tool for converting sonar data into common netCDF format.
Currently in this file is the previous incarnation using HDF5 as the storage format.
This will be changed to use the functions under /convert and support multiple source data formats.
"""

import argparse
import os.path
from echopype.convert.ek60 import ConvertEK60


def main():
    parser = argparse.ArgumentParser(description='Convert EK60 .raw to a netCDF (.nc) file')

    parser.add_argument('-i', '--input-file', action='store', nargs='+', help='input *.raw filename')

    args = parser.parse_args()

    for p in args.input_file:
        ff = os.path.basename(p)
        f, ext = os.path.splitext(ff)
        if not os.path.isfile(f+'.nc'):  # if file has not been converted
            tmp = ConvertEK60(p)  # create an instance of EK60 converter
            tmp.raw2nc()   # convert .raw to .nc
        else:
            print('This .raw file has been converted before. Delete if you want to convert again.')


if __name__ == '__main__':
    main()
