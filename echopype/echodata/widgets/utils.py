import uuid
from hashlib import md5

import anytree
from datatree import DataTree

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


def _single_node_repr(node):
    root_path = "root"
    node_pathstr = "Top-level"
    if node.name != root_path:
        node_pathstr = node.pathstr.replace("root/", "")
    sonar_group = SONAR_GROUPS[node_pathstr]
    node_info = f"{sonar_group['name']}: {sonar_group['description']}"
    return node_info


def tree_repr(tree: DataTree) -> str:
    renderer = anytree.RenderTree(tree)
    lines = []
    for pre, _, node in renderer:
        if node.has_data or node.has_attrs:
            node_repr = _single_node_repr(node)

            node_line = f"{pre}{node_repr.splitlines()[0]}"
            lines.append(node_line)
    return "\n".join(lines)
