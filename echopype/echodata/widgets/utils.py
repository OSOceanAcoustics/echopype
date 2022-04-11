import anytree
from datatree import DataTree

from ..convention.utils import _get_sonar_groups

SONAR_GROUPS = _get_sonar_groups()


def html_repr(value) -> str:
    return value._repr_html_()


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
        node_repr = _single_node_repr(node)

        node_line = f"{pre}{node_repr.splitlines()[0]}"
        lines.append(node_line)
    return "\n".join(lines)
