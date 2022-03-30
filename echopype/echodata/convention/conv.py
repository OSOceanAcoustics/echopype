from importlib import resources
from typing import Optional

import yaml

from .. import convention


class _Convention:
    def __init__(self, version: Optional[str]):
        """Prepare to read the convention yaml file"""
        self._yaml_dict = {}
        # Hardwired to 1.0, for now
        self.version = "1.0"
        if version:
            self.version = version

    @property
    def yaml_dict(self):
        """Read data from disk"""
        if self._yaml_dict:  # Data has already been read, return it directly
            return self._yaml_dict

        with resources.open_text(package=convention, resource=f"{self.version}.yml") as fid:
            convention_yaml = yaml.load(fid, Loader=yaml.SafeLoader)

        self._yaml_dict = convention_yaml

        return self._yaml_dict
