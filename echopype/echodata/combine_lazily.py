from .combine_preprocess import PreprocessCallable
from echopype.echodata import EchoData
from datatree import DataTree
import xarray as xr

# desired_raw_file_paths = fs.glob('OOI_zarrs_ep_ex/temp/*.zarr')


def reassign_attrs(ed_comb: EchoData):
    """
    Reassigns stored group attributes to the Provenance group.
    """

    for group, value in EchoData.group_map.items():

        if value["ep_group"] not in ['Sonar/Beam_group2', 'Sonar/Beam_group3', 'Sonar/Beam_group4']:

            if value["ep_group"] != "Provenance":

                attr_var_name = group + '_attrs'
                attr_coord_name = group + '_attr_key'

                if value["ep_group"]:
                    ed_grp = value["ep_group"]
                else:
                    ed_grp = "Top-level"

                # move attribute variable to Provenance
                ed_comb["Provenance"][attr_var_name] = ed_comb[ed_grp][attr_var_name]

                # remove attribute variable and coords from group
                ed_comb[ed_grp] = ed_comb[ed_grp].drop_vars([attr_var_name, attr_coord_name,
                                                             'echodata_filename'])


def lazy_combine(desired_raw_file_paths):

    # initial strucuture for lazy combine
    tree_dict = {}
    result = EchoData()

    # grab object that does pre-processing
    preprocess_obj = PreprocessCallable(desired_raw_file_paths)

    for group, value in EchoData.group_map.items():

        print(value["ep_group"])

        if value["ep_group"] not in ['Sonar/Beam_group2', 'Sonar/Beam_group3', 'Sonar/Beam_group4']:

            preprocess_obj.update_ed_group(group)

            combined_group = xr.open_mfdataset(desired_raw_file_paths,
                                               engine='zarr', coords='minimal', preprocess=preprocess_obj,
                                               combine="nested", group=value["ep_group"], concat_dim=None)

            if value["ep_group"] is None:
                tree_dict["/"] = combined_group
            else:
                tree_dict[value["ep_group"]] = combined_group

    # Set tree into echodata object
    result._set_tree(tree=DataTree.from_dict(tree_dict, name="root"))
    result._load_tree()

    # reassign stored group attributes to the provenance group
    reassign_attrs(result)

    # TODO: modify Provenance conversion_time attribute
    #   dt.utcnow().isoformat(timespec="seconds") + "Z",  # use UTC time

    return result



# How to construct  Provenance Group
# obj = ProvenancePreprocess(desired_raw_file_paths)
#
# out = xr.open_mfdataset(desired_raw_file_paths[:2],
#                         engine='zarr', coords='minimal',
#                         combine="nested", group='Provenance',
#                         preprocess=obj, concat_dim=None)
# TODO: to be identical to in-memory combine remove filenames as coordinate (keep as dim)