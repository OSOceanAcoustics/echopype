""" Temporary file used for transitioning from using 'model' to 'process' due to rename
    ModelAZFP changed to ProcessAZFP
"""
from echopype.process import ProcessAZFP
from warnings import warn


def ModelAZFP(nc_path):
    warn("ModelAZFP has been renamed to ProcessAZFP and will no longer be supported in the future.",
         DeprecationWarning, 2)
    return ProcessAZFP(nc_path)
