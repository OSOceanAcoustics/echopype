import os
from echopype.convert.azfp import ConvertAZFP
from echopype.convert.ek60 import ConvertEK60


class Convert:
    def __init__(self, _path='', _xml_path=''):
        if _path:
            if os.path.isfile(_path):
                self.path = _path
                self.type = self._get_type(os.path.basename(self.path))
            else:
                raise FileNotFoundError(f"There is no file named {os.path.basename(_path)}")
        else:
            raise ValueError("Convert requires the path to a raw file")

        if self.type == "AZFP":
            if _xml_path:
                if '.XML' in _xml_path.upper():
                    if os.path.isfile(_xml_path):
                        self.xml_path = _xml_path
                    else:
                        raise FileNotFoundError(f"There is no file named {os.path.basename(_xml_path)}")
                else:
                    raise ValueError(f"{os.path.basename(_xml_path)} is not an XML file")
            else:
                raise ValueError("XML file is rquired for AZFP raw data")

    def _get_type(self, file_name):
        """Gets the type of echosounder the raw file was generated with using the extension of the file

        Parameters
        ----------
        file_name : str
            the file that will be converted. Currently only .raw and .01A files are supported
            for the Simrad EK60 and ASL AZFP echosounders respectively
        Returns
        -------
            The name of the echosounder
        """
        ext = os.path.splitext(file_name)[1]
        if ext == '.raw':
            return 'EK60'
        elif ext == '.01A':
            return 'AZFP'
        else:
            raise ValueError("'{}' is not a supported file type".format(ext))

    def raw2nc(self):
        """
        Converts the raw file into a netCDF4 file following the SONAR-netCDF4 convention
        """
        if self.type == "EK60":
            tmp = ConvertEK60(self.path)
            tmp.raw2nc()
        elif self.type == "AZFP":
            tmp = ConvertAZFP(self.path, self.xml_path)
            tmp.raw2nc()
        self.__dict__.update(tmp.__dict__)
