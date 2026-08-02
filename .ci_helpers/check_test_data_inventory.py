#!/usr/bin/env python3
"""Validate the test data inventory against the Pooch test-data bundles."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

CONFTST = ROOT / "echopype" / "tests" / "conftest.py"
INVENTORY = ROOT / "docs" / "source" / "test_data_inventory.yml"

REQUIRED_BUNDLE_KEYS = {
    "documented_checksum",
    "instrument",
    "description",
    "source",
    "contributor",
    "references",
    "notes",
    "files",
}


def load_conftest_bundles_and_registry():
    source = CONFTST.read_text(encoding="utf-8")
    tree = ast.parse(source)

    bundles = None
    registry = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "bundles":
                bundles = ast.literal_eval(node.value)
            elif isinstance(target, ast.Name) and target.id == "registry":
                registry = ast.literal_eval(node.value)

    if bundles is None:
        raise RuntimeError(f"Could not find 'bundles' in {CONFTST}")

    if registry is None:
        raise RuntimeError(f"Could not find 'registry' in {CONFTST}")

    return bundles, registry


def load_inventory():
    with INVENTORY.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    bundles, registry = load_conftest_bundles_and_registry()
    inventory = load_inventory()

    errors = []

    conftest_bundles = set(bundles)
    inventory_bundles = set(inventory)

    missing = sorted(conftest_bundles - inventory_bundles)
    extra = sorted(inventory_bundles - conftest_bundles)

    if missing:
        errors.append(
            "Bundles registered in echopype/tests/conftest.py but missing from "
            "docs/source/test_data_inventory.yml:\n  - "
            + "\n  - ".join(missing)
            + "\n\nAdd one inventory entry for each bundle, following the template "
            "at the top of docs/source/test_data_inventory.yml."
        )

    if extra:
        errors.append(
            "Bundles listed in docs/source/test_data_inventory.yml but not "
            "registered in echopype/tests/conftest.py:\n  - "
            + "\n  - ".join(extra)
            + "\n\nRemove obsolete entries or add the corresponding bundle to the "
            "Pooch registry in echopype/tests/conftest.py."
        )

    for bundle in sorted(conftest_bundles & inventory_bundles):
        metadata = inventory[bundle]

        if not isinstance(metadata, dict):
            errors.append(f"{bundle}: inventory entry must be a mapping")
            continue

        missing_keys = REQUIRED_BUNDLE_KEYS - set(metadata)
        if missing_keys:
            errors.append(
                f"{bundle}: missing required bundle keys: " + ", ".join(sorted(missing_keys))
            )

        checksum = metadata.get("documented_checksum")
        if checksum is not None and checksum != registry[bundle]:
            errors.append(
                f"{bundle}: documented_checksum does not match the Pooch registry.\n"
                f"  inventory: {checksum}\n"
                f"  registry:  {registry[bundle]}\n\n"
                "Review the bundle metadata and file list, then update "
                "documented_checksum to the registry value."
            )

        files = metadata.get("files", [])

        if not isinstance(files, list):
            errors.append(f"{bundle}: files must be a list")
        elif not all(isinstance(filename, str) for filename in files):
            errors.append(f"{bundle}: every entry in files must be a string")
        elif len(files) != len(set(files)):
            errors.append(f"{bundle}: files contains duplicate filenames")

    if errors:
        print("\nTEST DATA INVENTORY CHECK FAILED\n")
        print("\n\n".join(errors))
        sys.exit(1)

    print("Test data inventory is valid.")


if __name__ == "__main__":
    main()
