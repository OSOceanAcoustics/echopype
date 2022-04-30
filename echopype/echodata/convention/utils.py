from ..convention import sonarnetcdf_1


def _get_sonar_groups():
    """Utility to reorder convention file by the paths"""
    group_mapping = sonarnetcdf_1.yaml_dict["groups"]
    sonar_groups = {}
    for k, v in group_mapping.items():
        group_path = v.get("ep_group")
        if any(group_path == p for p in [None, "/"]):
            group_path = "Top-level"
        sonar_groups.setdefault(
            group_path,
            {"description": v.get("description"), "name": v.get("name")},
        )
    return sonar_groups
