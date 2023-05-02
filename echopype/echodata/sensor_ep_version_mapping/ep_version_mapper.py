from _echopype_version import version as __version__  # noqa

from .v05x_to_v06x import convert_v05x_to_v06x


def map_ep_version(echodata_obj):
    """
    Function that coordinates the conversion between echopype versions

    Parameters
    ----------
    echodata_obj : EchoData
        EchoData object that may need to be converted

    Notes
    -----
    The function directly modifies the input EchoData object.
    """

    if (0, 5, 0) <= echodata_obj.version_info < (0, 6, 0):
        convert_v05x_to_v06x(echodata_obj)
    elif (0, 6, 0) <= echodata_obj.version_info < (0, 8, 0):
        pass
    else:
        str_version = ".".join(map(str, echodata_obj.version_info))
        raise NotImplementedError(
            f"Conversion of data from echopype v{str_version} format to"
            + f" v{__version__} format is not available. Please use open_raw"
            + f" to convert data to version {__version__} format."
        )
