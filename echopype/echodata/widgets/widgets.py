import datetime
import html
from functools import lru_cache
from pathlib import Path

import pkg_resources
from jinja2 import Environment, FileSystemLoader, Template
from jinja2.exceptions import TemplateNotFound

from .utils import _single_node_repr, hash_value, html_repr, make_key

FILTERS = {
    "datetime_from_timestamp": datetime.datetime.fromtimestamp,
    "html_escape": html.escape,
    "type": type,
    "repr": repr,
    "html_repr": html_repr,
    "hash_value": hash_value,
    "make_key": make_key,
    "node_repr": _single_node_repr,
}

HERE = Path(__file__).parent
STATIC_DIR = HERE / "static"

TEMPLATE_PATHS = [HERE / "templates"]

STATIC_FILES = (
    "static/html/icons-svg-inline.html",
    "static/css/style.css",
)


@lru_cache(None)
def _load_static_files():
    """Lazily load the resource files into memory the first time they are needed.
    Clone from xarray.core.formatted_html_template.
    """
    return [pkg_resources.resource_string(__name__, fname).decode("utf8") for fname in STATIC_FILES]


def get_environment() -> Environment:
    loader = FileSystemLoader(TEMPLATE_PATHS)
    environment = Environment(loader=loader)
    environment.filters.update(FILTERS)

    return environment


def get_template(name: str) -> Template:
    try:
        return get_environment().get_template(name)
    except TemplateNotFound as e:
        raise TemplateNotFound(
            f"Unable to find {name} in echopype.echodata.widgets. TEMPLATE_PATHS {TEMPLATE_PATHS}"
        ) from e
