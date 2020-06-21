""" Temporary file used for transitioning from using 'model' to 'process' due to rename
    EchoData changed to Process
"""
from echopype.process import Process
from warnings import warn


def EchoData(nc_path):
    warn("Echodata has been renamed to Process and will no longer be supported in the future.",
         DeprecationWarning, 2)
    return Process(nc_path)
