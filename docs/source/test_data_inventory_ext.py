"""Render the test data inventory page from YAML metadata."""

from __future__ import annotations

from pathlib import Path

import yaml


def _value(value):
    if value in (None, "", []):
        return "NA"
    if isinstance(value, list):
        return "<br>".join(str(v) for v in value)
    return str(value).replace("\n", " ")


def _render_inventory(app):
    inventory_path = Path(app.srcdir) / "test_data_inventory.yml"
    inventory = yaml.safe_load(inventory_path.read_text(encoding="utf-8"))

    lines = [
        "(test-data-inventory)=\n",
        "# Test data inventory\n",
        "This page is generated automatically from `docs/source/test_data_inventory.yml`.\n",
    ]

    for bundle, metadata in sorted(inventory.items()):
        lines.append(f"## `{bundle}`\n")
        lines.append(f"**Instrument:** {_value(metadata.get('instrument'))}  \n")
        lines.append(f"**Description:** {_value(metadata.get('description'))}  \n")
        lines.append(f"**Source:** {_value(metadata.get('source'))}  \n")
        lines.append(f"**Contributor:** {_value(metadata.get('contributor'))}  \n")
        lines.append(f"**Notes:** {_value(metadata.get('notes'))}  \n")

        references = metadata.get("references", [])
        if references:
            lines.append("\n**References:**\n")
            for ref in references:
                lines.append(f"- {ref}\n")
        else:
            lines.append("\n**References:** NA\n")

        files = metadata.get("files", [])
        lines.append("\n### Files\n")

        if files:
            for filename in sorted(files):
                lines.append(f"- `{filename}`\n")
        else:
            lines.append("NA\n")

        lines.append("\n")

    output_path = Path(app.srcdir) / "test_data_inventory.md"
    output_path.write_text("".join(lines), encoding="utf-8")


def setup(app):
    app.connect("builder-inited", _render_inventory)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
