import numpy as np


def _extvar_properties(ds, name):
    """Test the external variable for presence and all-nan values,
    and extract its time dimension name.
    Returns <presence>, <valid values>, <time dim name>
    """
    if name in ds:
        # Assumes the only dimension in the variable is a time dimension
        time_dim_name = ds[name].dims[0] if len(ds[name].dims) > 0 else "scalar"
        if not ds[name].isnull().all():
            return True, True, time_dim_name
        else:
            return True, False, time_dim_name
    else:
        return False, False, None


def _clip_by_time_dim(external_ds, ext_time_dim_name, ping_time):
    """
    Clip incoming time to 1 less than min of EchoData["Sonar/Beam_group1"]["ping_time"]
    and 1 greater than max of EchoData["Sonar/Beam_group1"]["ping_time"].
    Accounts for unsorted external time by checking whether each time value is between
    min and max ping_time instead of finding the 2 external times corresponding to the
    min and max ping_time and taking all the times between those indices.
    """

    sorted_external_time = external_ds[ext_time_dim_name].data
    sorted_external_time.sort()

    min_index = max(
        np.searchsorted(sorted_external_time, ping_time.min(), side="left") - 1,
        0,
    )

    max_index = min(
        np.searchsorted(
            sorted_external_time,
            ping_time.max(),
            side="right",
        ),
        len(sorted_external_time) - 1,
    )

    return external_ds.sel(
        {
            ext_time_dim_name: np.logical_and(
                sorted_external_time[min_index] <= external_ds[ext_time_dim_name],
                external_ds[ext_time_dim_name] <= sorted_external_time[max_index],
            )
        }
    )


def get_mappings_expanded(logger, extra_platform_data, variable_mappings, platform):
    """
    Generate a dictionary of mappings between Platform group variables and external variables.

    Parameters
    ----------
    logger : logging.Logger
        A logger object to log warnings and errors.
    extra_platform_data : xr.Dataset
        An `xr.Dataset` containing the additional platform data to be added
        to the `EchoData["Platform"]` group.
    variable_mappings: Dict[str,str]
        A dictionary mapping Platform variable names (dict key) to the
        external-data variable name (dict value).
    platform: xr.Dataset
        An `xr.Dataset` containing the original Platform data.

    Returns
    -------
    mappings_expanded: Dict[str, Dict[str, Any]]
        A dictionary containing mappings between Platform group variables and external variables.

    Raises
    ------
    ValueError: If only one of latitude and longitude are specified.
                If the external latitude and longitude use different time dimensions.
    """

    mappings_expanded = {}
    for platform_var, external_var in variable_mappings.items():
        # TODO: instead of using existing Platform group variables, a better practice is to
        # define a set of allowable Platform variables (sonar_model dependent) for this check.
        # This set can be dynamically generated from an external source like a CDL or yaml.
        if platform_var in platform:
            platform_validvalues = not platform[platform_var].isnull().all()
            ext_present, ext_validvalues, ext_time_dim_name = _extvar_properties(
                extra_platform_data, external_var
            )
            if ext_present and ext_validvalues:
                mappings_expanded[platform_var] = dict(
                    external_var=external_var,
                    ext_time_dim_name=ext_time_dim_name,
                    platform_validvalues=platform_validvalues,
                )

    # Generate warning if mappings_expanded is empty
    if not mappings_expanded:
        logger.warning(
            "No variables will be updated, "
            "check variable_mappings to ensure variable names are correctly specified!"
        )

    # If longitude or latitude are requested, verify that both are present
    # and they share the same external time dimension
    for lat_name, lon_name in [
        ("latitude", "longitude"),
        ("latitude_idx", "longitude_idx"),
        ("latitude_mru1", "longitude_mru1"),
    ]:
        if lat_name in mappings_expanded or lon_name in mappings_expanded:
            if lat_name not in mappings_expanded or lon_name not in mappings_expanded:
                raise ValueError(
                    f"Only one of {lat_name} and {lon_name} are specified. Please include both, or neither."  # noqa
                )
            if (
                mappings_expanded[lat_name]["ext_time_dim_name"]
                != mappings_expanded[lon_name]["ext_time_dim_name"]
            ):
                raise ValueError(
                    "The external latitude and longitude use different time dimensions. "
                    "They must share the same time dimension."
                )

    # Generate warnings regarding variables that will be updated
    vars_not_handled = set(variable_mappings.keys()).difference(mappings_expanded.keys())
    if len(vars_not_handled) > 0:
        logger.warning(
            f"The following requested variables will not be updated: {', '.join(vars_not_handled)}"  # noqa
        )

    vars_notnan_replaced = [
        platform_var for platform_var, v in mappings_expanded.items() if v["platform_validvalues"]
    ]
    if len(vars_notnan_replaced) > 0:
        logger.warning(
            f"Some variables with valid data in the original Platform group will be overwritten: {', '.join(vars_notnan_replaced)}"  # noqa
        )

    return mappings_expanded
