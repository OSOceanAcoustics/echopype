import os
import shutil
import pytest
from ..convert import Convert
from ..process import Process, EchoData

ek80_bb_path = './echopype/test_data/ek80/D20170912-T234910.raw'   # Large dataset (BB)
ek80_cw_path = './echopype/test_data/ek80/D20190822-T161221.raw'     # Small dataset (CW)
# ek80_bb_cw_path = './echopype/test_data/ek80/Summer2018--D20180905-T033113.raw'   # BB and CW

@pytest.mark.skip(reason='Broadband calibration is still under development')
def test_broadband_calibration():
    """Check noise estimation and noise removal using xarray and brute force using numpy.
    """

    # Noise estimation via Process method =========
    # Unpack data and convert to .nc file
    tmp = Convert(ek80_bb_path, model="EK80")
    tmp.to_netcdf()

    # Read .nc file into an Process object and calibrate
    ed = EchoData(raw_path=tmp.output_file)
    proc = Process(model='EK80', ed=ed)
    proc.calibrate(ed, save=True, save_format='netcdf4')

    ed.close()
    os.remove(tmp.output_file)
    os.remove(ed.Sv_path)


def test_narrowband_calibration():
    """Test EK80 narrowband calibration with zarr files"""
    tmp = Convert(file=ek80_cw_path, model='EK80')
    tmp.to_netcdf()

    ed = EchoData(raw_path=tmp.output_file)
    proc = Process(model='EK80', ed=ed)
    proc.get_Sv(ed, save=True)
    proc.remove_noise(ed)

    ed.close()
    shutil.rmtree(ed.Sv_path)
    os.remove(tmp.output_file)
