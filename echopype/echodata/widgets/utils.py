import uuid
from hashlib import md5

from datatree import DataTree
from datatree.render import RenderTree

from ..convention.utils import _get_sonar_groups

SONAR_GROUPS = _get_sonar_groups()


def html_repr(value) -> str:
    return value._repr_html_()


def hash_value(value: str) -> str:
    byte_string = value.encode("utf-8")
    hashed = md5(byte_string)
    return hashed.hexdigest()


def make_key(value: str) -> str:
    return value + str(uuid.uuid4())


def _single_node_repr(node: DataTree) -> str:
    """
    Obtains the string repr for a single node in a
    ``RenderTree`` or ``DataTree``.

    Parameters
    ----------
    node: DataTree
        A single node obtained from a ``RenderTree`` or ``DataTree``

    Returns
    -------
    node_info: str
        string representation of repr for the input ``node``
    """

    # initialize node_pathstr
    node_pathstr = "Top-level"

    # obtain the appropriate group name and get its descriptions from the yaml
    if node.name != "root":
        node_pathstr = node.path[1:]
    sonar_group = SONAR_GROUPS[node_pathstr]

    if "Beam_group" in sonar_group["name"]:
        # get description of Beam_group directly from the Sonar group
        group_descr = str(
            node.parent["/Sonar"].ds.beam_group_descr.sel(beam_group=sonar_group["name"]).values
        )
    else:
        # get description of group from yaml file
        group_descr = sonar_group["description"]

    # construct the final node information string for repr
    node_info = f"{sonar_group['name']}: {group_descr}"

    return node_info


def tree_repr(tree: DataTree) -> str:
    renderer = RenderTree(tree)
    lines = []
    for pre, _, node in renderer:
        if node.has_data or node.has_attrs:
            node_repr = _single_node_repr(node)

            node_line = f"{pre}{node_repr.splitlines()[0]}"
            lines.append(node_line)
    return "\n".join(lines)
