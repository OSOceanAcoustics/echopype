"""
This file provides a wrapper for the convert objects and functions.
Users will not need to know the names of the specific objects they need to create.
"""
import os
from echopype.convert.azfp import ConvertAZFP
from echopype.convert.ek60 import ConvertEK60
from echopype.convert.ek80 import ConvertEK80


def Convert(path='', xml_path='', model='EK60'):
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

    file_name = os.path.basename(path)

    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"There is no file named {os.path.basename(path)}")

        # Gets the type of echosounder from the extension of the raw file
        # return a Convert object depending on the type of echosounder used to create the raw file
        ext = os.path.splitext(file_name)[1]
        if ext == '.raw':
            # TODO: Find something better
            if model == 'EK60':
                return ConvertEK60(path)
            elif model == 'EK80':
                return ConvertEK80(path)
        elif ext == '.01A':
            if xml_path:
                if '.XML' in xml_path.upper():
                    if not os.path.isfile(xml_path):
                        raise FileNotFoundError(f"There is no file named {os.path.basename(xml_path)}")
                    return ConvertAZFP(path, xml_path)
                else:
                    raise ValueError(f"{os.path.basename(xml_path)} is not an XML file")
            else:
                raise ValueError("XML file is required for AZFP raw data")
        else:
            raise ValueError(f"'{ext}' is not a supported file type")
    else:
        raise ValueError("Convert requires the path to a raw file")
