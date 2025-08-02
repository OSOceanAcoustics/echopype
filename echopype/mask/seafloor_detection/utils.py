import numpy as np
import xarray as xr


def mask_below_seafloor(ds: xr.Dataset, varname: str = "Sv") -> xr.Dataset:
    ds_masked = ds.copy(deep=True)

    if varname not in ds_masked:
        raise ValueError(f"Dataset must contain '{varname}' variable.")

    ping_times = ds_masked["ping_time"].values
    # range_sample = ds_masked["range_sample"].values

    for chan in ds_masked["channel"].values:
        bottom_var = f"seafloor_sample_range_on_{chan}"
        if bottom_var not in ds_masked:
            print(f"No seafloor index found for channel '{chan}', skipping.")
            continue

        seafloor_idx = ds_masked[bottom_var].values.astype(float)

        for i, idx in enumerate(seafloor_idx):
            if np.isnan(idx):
                continue
            ping_val = ping_times[i]
            ds_masked[varname].loc[
                dict(channel=chan, ping_time=ping_val, range_sample=slice(int(idx) + 1, None))
            ] = np.nan

    return ds_masked
