import os
from echopype.convert import Convert
from echopype.model import EchoData

adcp_path = './echopype/test_data/adcp/Sig1000_IMU.ad2cp'

def test_model_ADCP():
    tmp_convert = Convert(adcp_path)
    tmp_convert.raw2nc()

    tmp_echo = EchoData(tmp_convert.nc_path)

    os.remove(tmp_convert.nc_path)

test_model_ADCP()