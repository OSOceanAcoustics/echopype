import warnings

from ..commongrid import api as commongrid_api
from ..filter import api as filter_api


def _warning_msg(new_subpkg_name, fn_name):
    return (
        f"Calling {fn_name} as preprocess.{fn_name} is deprecated and "
        "will be removed in version 0.7.1. Please call the function as "
        f"{new_subpkg_name}.{fn_name}"
    )


def estimate_noise(*args, **kwargs):
    warnings.warn(_warning_msg("filter", "estimate_noise"), DeprecationWarning, 2)
    return filter_api.estimate_noise(*args, **kwargs)


def remove_noise(*args, **kwargs):
    warnings.warn(_warning_msg("filter", "remove_noise"), DeprecationWarning, 2)
    return filter_api.remove_noise(*args, **kwargs)


def compute_MVBS(*args, **kwargs):
    warnings.warn(_warning_msg("commongrid", "compute_MVBS"), DeprecationWarning, 2)
    return commongrid_api.compute_MVBS(*args, **kwargs)


def compute_MVBS_index_binning(*args, **kwargs):
    warnings.warn(_warning_msg("commongrid", "compute_MVBS_index_binning"), DeprecationWarning, 2)
    return commongrid_api.compute_MVBS_index_binning(*args, **kwargs)
