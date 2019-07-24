from echopype.model.azfp import EchoDataAZFP
from echopype.model.ek60 import EchoDataEK60


class Model:
    def __new__(self, convert):
        if convert.type == "EK60":
            return EchoDataEK60(convert.nc_path)
        elif convert.type == "AZFP":
            return EchoDataAZFP(convert.nc_path)
        else:
            raise ValueError("Unsupported file type")
