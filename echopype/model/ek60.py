""" Temporary file used for transitioning from using 'model' to 'process' due to rename
    ModelEK60 changed to ProcessEK60
"""
from echopype.process import ProcessEK60
from warnings import warn


def ModelEK60(nc_path):
    warn("ModelEK60 has been renamed to ProcessEK60 and will no longer be supported in the future.",
         DeprecationWarning, 2)
    return ProcessEK60(nc_path)
