from pathlib import Path

import yaml

CONVENTION_PATH = Path(__file__).parent.absolute()


def _get_convention(version="1.0") -> dict:
    """Retrieves convention metadata as a dictionary"""
    convention_file = CONVENTION_PATH / f"{version}.yml"
    return yaml.load(convention_file.open(), Loader=yaml.SafeLoader)


__all__ = [_get_convention]
