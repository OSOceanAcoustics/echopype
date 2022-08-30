import numpy as np
from pathlib import Path
import xarray as xr
from typing import List


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

        # TODO: add method to check and correct reversed times

        return ds

    def update_ed_group(self, group: str):
        self.ed_group = group

    def _assign_file_index(self, ds):

        ind_file = self.file_paths.index(ds.encoding["source"])
        ds['filenames'] = (['filenames'], np.array([ind_file]))

    # TODO: add method to check and correct reversed times

    def _store_attrs(self, ds):

        file_name = Path(ds.encoding["source"]).name

        grp_key_name = self.ed_group + '_attr_key'
        grp_attr_names = np.array(list(ds.attrs.keys()))

        attrs_var = xr.DataArray(data=np.array([list(ds.attrs.values())]),
                                 coords={'echodata_filename': (['echodata_filename'], np.array([file_name])),
                                         grp_key_name: ([grp_key_name], grp_attr_names)})

        ds[self.ed_group + '_attrs'] = attrs_var
