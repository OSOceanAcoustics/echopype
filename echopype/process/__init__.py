"""
Include methods to manipulate echo data that is already converted to netCDF.

EK60 and AZFP narrowband echosounders:

- calibration and echo-integration to obtain
  volume backscattering strength (Sv) from power data.
- Simple noise removal by removing data points (set to ``NaN``) below
  an adaptively estimated noise floor.
- Binning and averaging to obtain mean volume backscattering strength (MVBS)
  from the calibrated data.

EK80 broadband echosounder:

- calibration based on pulse compression output in the
  form of average over frequency.

"""
from .process import Process
from .process_base import ProcessBase, ProcessEK
from .process_ek60 import ProcessEK60
from .process_azfp import ProcessAZFP
from .process_ek80 import ProcessEK80
from .echodata import EchoData
from .echodata_new import EchoDataNew