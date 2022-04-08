from functools import lru_cache

import anytree
import pkg_resources
from datatree import DataTree

from echopype.echodata.convention.utils import _get_sonar_groups

STATIC_FILES = ("static/html/icons-svg-inline.html", "static/css/style.css")
SONAR_GROUPS = _get_sonar_groups()


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


@lru_cache(None)
def _load_static_files():
    """Lazily load the resource files into memory the first time they are needed.
    Clone from xarray.core.formatted_html_template.
    """
    return [
        pkg_resources.resource_string("echopype", fname).decode("utf8") for fname in STATIC_FILES
    ]


class HtmlTemplate:
    """Contain html templates for InferenceData repr."""

    html_template = """
            <div>
              <div class='xr-header'>
                <div class="xr-obj-type">EchoData: standardized raw data from {file_path}</div>
              </div>
              <ul class="xr-sections group-sections">
              {}
              </ul>
            </div>
            """  # noqa
    element_template = """
            <li class = "xr-section-item">
                  <input id="idata_{group_id}" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_{group_id}" class = "xr-section-summary">{group}: ({group_name}) {group_description}</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;">{xr_data}<br></div>
                      </ul>
                  </div>
            </li>
            """  # noqa
    _, css_style = _load_static_files()  # noqa
    specific_style = ".xr-wrap{width:700px!important;}"
    css_template = f"<style> {css_style}{specific_style} </style>"
