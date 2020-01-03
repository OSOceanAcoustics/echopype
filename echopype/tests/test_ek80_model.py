import os
import numpy as np
import xarray as xr
from echopype.convert import Convert
from echopype.model import EchoData

# ek80_raw_path = './echopype/test_data/ek80/D20170912-T234910.raw'   # Large dataset
ek80_raw_path = './echopype/test_data/ek80/D20190822-T161221.raw'     # Small dataset
ek80_test_path = './echopype/test_data/ek60/from_matlab/DY1801_EK60-D20180211-T164025_Sv_TS.nc'
nc_path = os.path.join(os.path.dirname(ek80_raw_path),
                       os.path.splitext(os.path.basename(ek80_raw_path))[0] + '.nc')
Sv_path = os.path.join(os.path.dirname(ek80_raw_path),
                       os.path.splitext(os.path.basename(ek80_raw_path))[0] + '_Sv.nc')


def test_noise_estimates_removal():
    """Check noise estimation and noise removal using xarray and brute force using numpy.
    """

    # Noise estimation via EchoData method =========
    # Unpack data and convert to .nc file
    tmp = Convert(ek80_raw_path, model="EK80")
    tmp.raw2nc()

    # Read .nc file into an EchoData object and calibrate
    e_data = EchoData(tmp.nc_path)
    e_data.calibrate()

    del e_data
    # os.remove(nc_path)

test_noise_estimates_removal()