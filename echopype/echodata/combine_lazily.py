from .combine_preprocess import ProvenancePreprocess
from echopype.echodata import EchoData
from datatree import DataTree
import xarray as xr

group_preprocess = {'provenance': ProvenancePreprocess}


# desired_raw_file_paths = fs.glob('OOI_zarrs_ep_ex/temp/*.zarr')



# initial strucuture for lazy combine
# tree_dict = {}
# result = EchoData()
#
# # for group, value in EchoData.group_map.items()[:2]:
# for group, value in list(EchoData.group_map.items())[:3]:
#
#     print(value["ep_group"])
#
#     obj = ProvenancePreprocess(desired_raw_file_paths)
#
#     combined_group = xr.open_mfdataset(desired_raw_file_paths,
#                                        engine='zarr', coords='minimal', preprocess=obj,
#                                        combine="nested", group=value["ep_group"], concat_dim=None)
#
#     if value["ep_group"] is None:
#         tree_dict["/"] = combined_group
#     else:
#         tree_dict[value["ep_group"]] = combined_group
#
# # Set tree into echodata object
# result._set_tree(tree=DataTree.from_dict(tree_dict, name="root"))
# result._load_tree()



# How to construct  Provenance Group
# obj = ProvenancePreprocess(desired_raw_file_paths)
#
# out = xr.open_mfdataset(desired_raw_file_paths[:2],
#                         engine='zarr', coords='minimal',
#                         combine="nested", group='Provenance',
#                         preprocess=obj, concat_dim=None)
# TODO: to be identical to in-memory combine remove filenames as coordinate (keep as dim)