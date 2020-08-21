"""
Class to save unpacked echosounder data to appropriate groups in netcdf or zarr.
"""


class SetGroupsBase:
    """Base class for saving groups to netcdf or zarr from echosounder data files.
    """
    def __init__(self, convert_obj, output_file, output_path, output_format='netcdf', compress=True, overwrite=True):
        self.convert_obj = convert_obj   # a convert object ConvertEK60/ConvertAZFP/etc...
        self.output_file = output_file
        self.output_path = output_path
        self.output_format = output_format
        self.compress = compress
        self.overwrite = overwrite

    def save(self):
        """Actually save groups to file by calling the set methods.
        """

    def set_toplevel(self):
        """Set the top-level group.
        """

    def set_provenance(self):
        """Set the Provenance group.
        """

    def set_sonar(self):
        """Set the Sonar group.
        """

    def set_nmea(self):
        """Set the Platform/NMEA group.
        """


class SetGroupsEK60(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from EK60 data files.
    """
    def save(self):
        """Actually save groups to file by calling the set methods.
        """

    def set_env(self):
        """Set the Environment group.
        """

    def set_platform(self):
        """Set the Platform group.
        """

    def set_beam(self):
        """Set the Beam group.
        """


class SetGroupsEK80(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from EK80 data files.
    """
    def save(self):
        """Actually save groups to file by calling the set methods.
        """

    def set_env(self):
        """Set the Environment group.
        """

    def set_platform(self):
        """Set the Platform group.
        """

    def set_beam(self):
        """Set the Beam group.
        """

    def set_vendor(self):
        """Set the Vendor-specific group.
        """


class SetGroupsAZFP(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from AZFP data files.
    """
    def save(self):
        """Actually save groups to file by calling the set methods.
        """

    def set_env(self):
        """Set the Environment group.
        """

    def set_platform(self):
        """Set the Platform group.
        """

    def set_beam(self):
        """Set the Beam group.
        """

    def set_vendor(self):
        """Set the Vendor-specific group.
        """
