import os

import xarray as xr

from ..convert import Convert

GROUPS = [
    "Environment",
    "Platform",
    "Beam",
    "Vendor"
]

FILES = [
    # "./echopype/test_data/ad2cp/average_only.366.00000.ad2cp",
    "./echopype/test_data/ad2cp/avg_bur_echo.366.00000.ad2cp",
    "./echopype/test_data/ad2cp/burst_echosoun.366.00000.ad2cp",
    "./echopype/test_data/ad2cp/burst_only.366.00000.ad2cp",
]

BASE_FILES = [
    # "./echopype/test_data/ad2cp/average_only.366.00000.base.nc",
    "./echopype/test_data/ad2cp/avg_bur_echo.366.00000.base.nc",
    "./echopype/test_data/ad2cp/burst_echosoun.366.00000.base.nc",
    "./echopype/test_data/ad2cp/burst_only.366.00000.base.nc",
]

def test_process():
    for file, base_file in zip(FILES, BASE_FILES):
        tmp = Convert(file=file, model="AD2CP")
        tmp_file = f"{file}.test.nc"
        tmp.to_netcdf(save_path=tmp_file, overwrite=True)

        for group in GROUPS:
            testing_group = xr.open_dataset(tmp_file, group=group)
            base_group = xr.open_dataset(base_file, group=group)
            assert testing_group.equals(base_group), f"process ad2cp failed on {file} with group {group}"

        os.remove(tmp_file)
