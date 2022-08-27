import numpy as np
from pathlib import Path
import xarray as xr


class ProvenancePreprocess:
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __call__(self, ds):
        self.assign_file_index(ds)
        self.store_attrs(ds)

        return ds

    def assign_file_index(self, ds):

        ind_file = self.file_paths.index(ds.encoding["source"])
        ds['filenames'] = (['filenames'], np.array([ind_file]))

    def store_attrs(self, ds):

        file_name = Path(ds.encoding["source"]).name

        attrs_var = xr.DataArray(data=np.array([list(ds.attrs.values())]),
                                 coords={'echodata_filename': (['echodata_filename'], np.array([file_name])),
                                         'provenance_attr_key': (['provenance_attr_key'],
                                                                 np.array(['conversion_software_name',
                                                                           'conversion_software_version',
                                                                           'conversion_time',
                                                                           'duplicate_ping_times']))})

        ds['provenance_attrs'] = attrs_var






