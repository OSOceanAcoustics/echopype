from importlib import resources

import yaml

from .. import convention


class _Convention:
    def __init__(self):
        """Prepare to read the convention yaml file"""
        self._yaml_dict = {}
        # Hardwired to 1.0, for now
        self.version = "1.0"

    @property
    def yaml_dict(self):
        """Read data from disk"""
        if self._yaml_dict:  # Data has already been read, return it directly
            return self._yaml_dict

        with resources.open_text(
            package=convention, resource=f"{self.version}.yml"
        ) as fid:
            convention_yaml = yaml.load(fid, Loader=yaml.SafeLoader)

        self._yaml_dict = convention_yaml

        return self._yaml_dict


# Instantiate the singleton
conv = _Convention()
