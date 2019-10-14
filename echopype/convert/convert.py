"""
This file provides a wrapper for the convert objects and functions.
Users will not need to know the names of the specific objects they need to create.
"""
import os
from echopype.convert.azfp import ConvertAZFP
from echopype.convert.ek60 import ConvertEK60


class Convert:
    def __new__(cls, path='', xml_path=''):
        """
        Gets the type of echosounder the raw file was generated with using the filename extension.

        Parameters
        ----------
        path : str
            the file that will be converted. Currently only `.raw` and `.01A` files are supported
            for the Simrad EK60 and ASL AZFP echosounders respectively
        xml_path : str, optional
            If AZFP echo data is used, the XML file that accompanies the raw file is required for conversion.

        Returns
        -------
            Specialized convert object that will be used to produce a .nc file
        """

        # Gets the type of echosounder from the extension of the raw file
        file_name = os.path.basename(path)
        ext = os.path.splitext(file_name)[1]

        if ext == '.raw':
            echosounder_type = 'EK60'  # TODO: EK80 also produced .raw files so need something else later
        elif ext == '.01A':
            echosounder_type = 'AZFP'
        else:
            raise ValueError("'{}' is not a supported file type".format(ext))

        if path:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"There is no file named {os.path.basename(path)}")
        else:
            raise ValueError("Convert requires the path to a raw file")

        # return a Convert object depending on the type of echosounder used to create the raw file
        if echosounder_type == 'AZFP':
            if xml_path:
                if '.XML' in xml_path.upper():
                    if not os.path.isfile(xml_path):
                        raise FileNotFoundError(f"There is no file named {os.path.basename(xml_path)}")
                else:
                    raise ValueError(f"{os.path.basename(xml_path)} is not an XML file")
            else:
                raise ValueError("XML file is required for AZFP raw data")

            return ConvertAZFP(path, xml_path)
        elif echosounder_type == 'EK60':
            return ConvertEK60(path)
        else:
            raise ValueError("Unknown file extension")
