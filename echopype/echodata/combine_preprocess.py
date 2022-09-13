from pathlib import Path
from typing import List

import numpy as np
import xarray as xr


class PreprocessCallable:
    """
    Class that has all preprocessing functions and is callable.
    """

    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths
        self.ed_group = None

    def __call__(self, ds):

        if self.ed_group == "provenance":
            self._assign_file_index(ds)

        self._store_attrs(ds)

        ds = self.re_chunk(ds)

        # TODO: add method to check and correct reversed times

        return ds

    def update_ed_group(self, group: str):
        self.ed_group = group

    def re_chunk(self, ds):

        # chunk_dict = {'time2': 1000, 'time3': 1000}
        # chunk_dict = {'ping_time': 100, 'range_sample': 100}

        # ds = ds.chunk(chunk_dict)

        for drop_var in ["backscatter_r", "angle_athwartship", "angle_alongship"]:

            if drop_var in ds:
                ds = ds.drop_vars(drop_var)

        return ds

    def _assign_file_index(self, ds):

        ind_file = self.file_paths.index(ds.encoding["source"])
        ds["filenames"] = (["filenames"], np.array([ind_file]))

    # TODO: add method to check and correct reversed times

    def _store_attrs(self, ds):

        file_name = Path(ds.encoding["source"]).name

        grp_key_name = self.ed_group + "_attr_key"
        grp_attr_names = np.array(list(ds.attrs.keys()))

        attrs_var = xr.DataArray(
            data=np.array([list(ds.attrs.values())]),
            coords={
                "echodata_filename": (["echodata_filename"], np.array([file_name])),
                grp_key_name: ([grp_key_name], grp_attr_names),
            },
        )

        ds[self.ed_group + "_attrs"] = attrs_var
