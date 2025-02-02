import logging

import xarray as xr


def check_unique_ping_time_duplicates(ds_data: xr.Dataset, logger: logging.Logger) -> None:
    """
    Raises a warning if the data stored in duplicate pings is not unique.

    Parameters
    ----------
    ds_data : xr.Dataset
        Single freq beam dataset being processed in the `SetGroupsEK80.set_beams` class function.
    logger : logging.Logger
        Warning logger initialized in `SetGroupsEK80` file.
    """
    # Group the dataset by the "ping_time" coordinate
    groups = ds_data.groupby("ping_time")

    # Loop through each ping_time group
    for ping_time_val, group in groups:
        # Extract all data variable names to check
        data_vars = list(group.data_vars)

        # Use the first duplicate ping time index as a reference
        ref_duplicate_ping_time_index = 0

        # Iterate over each data variable in the group
        for var in data_vars:
            # Extract data array corresponding to the iterated variable
            data_array = group[var]

            # Use the slice corresponding to the reference index as the reference slice
            ref_slice = data_array.isel({"ping_time": ref_duplicate_ping_time_index})

            # Iterate over the remaining entries
            for i in range(1, data_array.sizes["ping_time"]):
                if not ref_slice.equals(data_array.isel({"ping_time": i})):
                    logger.warning(
                        f"Duplicate slices in variable '{var}' corresponding to 'ping_time' "
                        f"{ping_time_val} differ in data. All duplicate 'ping_time' entries "
                        "will be removed, which will result in data loss."
                    )
                    break
