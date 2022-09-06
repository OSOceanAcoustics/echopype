import xarray as xr
from datatree import DataTree
from fsspec.implementations.local import LocalFileSystem

from echopype.echodata import EchoData

from .combine_preprocess import PreprocessCallable

# desired_raw_file_paths = fs.glob('OOI_zarrs_ep_ex/temp/*.zarr')


def get_ed_path_from_str(zarr_path: str, path: str):
    """

    Parameters
    ----------
    zarr_path: str
        Full path to zarr file
    path: str
        Full path to ``.zgroup``
    """

    # the names of the groups that are needed to get to path
    all_grp_names = [
        elm for elm in path.split("/") if (elm not in zarr_path.split("/")) and (elm != ".zgroup")
    ]

    return "/".join(all_grp_names)


def get_zarr_grp_names(path: str, fs: LocalFileSystem) -> set:
    """
    Identifies the zarr group names using the path
    """

    # grab all paths that have .zgroup
    info = fs.glob(path + "/**.zgroup")

    # infer the group name based on the path
    ed_grp_name = {get_ed_path_from_str(path, entry) for entry in info}

    # remove the zarr file name and replace it with Top-level
    if "" in ed_grp_name:
        ed_grp_name.remove("")
        ed_grp_name.add(None)

    return ed_grp_name


def reassign_attrs(ed_comb: EchoData, common_grps: set):
    """
    Reassigns stored group attributes to the Provenance group.
    """

    for group, value in EchoData.group_map.items():

        if (value["ep_group"] != "Provenance") and (value["ep_group"] in common_grps):

            attr_var_name = group + "_attrs"
            attr_coord_name = group + "_attr_key"

            if value["ep_group"]:
                ed_grp = value["ep_group"]
            else:
                ed_grp = "Top-level"

            # move attribute variable to Provenance
            ed_comb["Provenance"][attr_var_name] = ed_comb[ed_grp][attr_var_name]

            # remove attribute variable and coords from group
            ed_comb[ed_grp] = ed_comb[ed_grp].drop_vars(
                [attr_var_name, attr_coord_name, "echodata_filename"]
            )


def lazy_combine(desired_raw_file_paths, fs):

    # TODO: test code when we have to do an expansion in range_sample

    # initial structure for lazy combine
    tree_dict = {}
    result = EchoData()

    # grab object that does pre-processing
    preprocess_obj = PreprocessCallable(desired_raw_file_paths)

    # TODO: the subsequent line is zarr specific!! Account for nc in the future
    # determine each zarr's group names
    file_grps = [get_zarr_grp_names(path, fs) for path in desired_raw_file_paths]

    # get the group names that all files share
    common_grps = set.intersection(*file_grps)

    # check that all zarrs have the same groups
    if any([common_grps.symmetric_difference(s) for s in file_grps]):
        raise RuntimeError("All input files must have the same groups!")

    for group, value in EchoData.group_map.items():

        if value["ep_group"] in common_grps:

            print(f"ed group = {value['ep_group']}")

            preprocess_obj.update_ed_group(group)

            combined_group = xr.open_mfdataset(
                desired_raw_file_paths,
                engine="zarr",
                coords="minimal",
                preprocess=preprocess_obj,
                combine="nested",
                group=value["ep_group"],
                concat_dim=None,
            )

            if value["ep_group"] is None:
                tree_dict["/"] = combined_group
            else:
                tree_dict[value["ep_group"]] = combined_group

    # Set tree into echodata object
    result._set_tree(tree=DataTree.from_dict(tree_dict, name="root"))
    result._load_tree()

    # reassign stored group attributes to the provenance group
    reassign_attrs(result, common_grps)

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
