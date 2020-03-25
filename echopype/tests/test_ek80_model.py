import os
from echopype.convert import Convert
from echopype.model import EchoData

# ek80_raw_path = './echopype/test_data/ek80/D20170912-T234910.raw'   # Large dataset
ek80_raw_path = './echopype/test_data/ek80/D20190822-T161221.raw'     # Small dataset


def test_noise_estimates_removal():
    """Check noise estimation and noise removal using xarray and brute force using numpy.
    """

    # Noise estimation via EchoData method =========
    # Unpack data and convert to .nc file
    tmp = Convert(ek80_raw_path, model="EK80")
    tmp.raw2nc()

    # Read .nc file into an EchoData object and calibrate
    e_data = EchoData(tmp.nc_path)
    e_data.calibrate(save=True)

    os.remove(tmp.nc_path)
    os.remove(e_data.Sv_path)
    del tmp
    del e_data
