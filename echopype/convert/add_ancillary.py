def update_platform(self, files=None, extra_platform_data=None):
    """
    Parameters
    ----------
    files : str / list
        path of converted .nc/.zarr files
    extra_platform_data : xarray dataset
        dataset containing platform information along a 'time' dimension
    """
    # self.extra_platform data passed into to_netcdf or from another function
    if extra_platform_data is None:
        return
    files = self.output_file if files is None else files
    if not isinstance(files, list):
        files = [files]
    engine = io.get_file_format(files[0])

    # saildrone specific hack
    if "trajectory" in extra_platform_data:
        extra_platform_data = extra_platform_data.isel(trajectory=0).drop("trajectory")
        extra_platform_data = extra_platform_data.swap_dims({"obs": "time"})

    # Try to find the time dimension in the extra_platform_data
    possible_time_keys = ["time", "ping_time", "location_time"]
    time_name = ""
    for k in possible_time_keys:
        if k in extra_platform_data:
            time_name = k
            break
    if not time_name:
        raise ValueError("Time dimension not found")

    for f in files:
        ds_beam = xr.open_dataset(f, group="Beam", engine=engine)
        ds_platform = xr.open_dataset(f, group="Platform", engine=engine)

        # only take data during ping times
        # start_time, end_time = min(ds_beam["ping_time"]), max(ds_beam["ping_time"])
        start_time, end_time = ds_beam.ping_time.min(), ds_beam.ping_time.max()

        extra_platform_data = extra_platform_data.sel(
            {time_name: slice(start_time, end_time)}
        )

        def mapping_get_multiple(mapping, keys, default=None):
            for key in keys:
                if key in mapping:
                    return mapping[key]
            return default

        if self.sonar_model in ["EK80", "EA640"]:
            ds_platform = ds_platform.reindex(
                {
                    "mru_time": extra_platform_data[time_name].values,
                    "location_time": extra_platform_data[time_name].values,
                }
            )
            # merge extra platform data
            num_obs = len(extra_platform_data[time_name])
            ds_platform = ds_platform.update(
                {
                    "pitch": (
                        "mru_time",
                        mapping_get_multiple(
                            extra_platform_data,
                            ["pitch", "PITCH"],
                            np.full(num_obs, np.nan),
                        ),
                    ),
                    "roll": (
                        "mru_time",
                        mapping_get_multiple(
                            extra_platform_data,
                            ["roll", "ROLL"],
                            np.full(num_obs, np.nan),
                        ),
                    ),
                    "heave": (
                        "mru_time",
                        mapping_get_multiple(
                            extra_platform_data,
                            ["heave", "HEAVE"],
                            np.full(num_obs, np.nan),
                        ),
                    ),
                    "latitude": (
                        "location_time",
                        mapping_get_multiple(
                            extra_platform_data,
                            ["lat", "latitude", "LATITUDE"],
                            default=np.full(num_obs, np.nan),
                        ),
                    ),
                    "longitude": (
                        "location_time",
                        mapping_get_multiple(
                            extra_platform_data,
                            ["lon", "longitude", "LONGITUDE"],
                            default=np.full(num_obs, np.nan),
                        ),
                    ),
                    "water_level": (
                        "location_time",
                        mapping_get_multiple(
                            extra_platform_data,
                            ["water_level", "WATER_LEVEL"],
                            np.ones(num_obs),
                        ),
                    ),
                }
            )
        elif self.sonar_model == "EK60":
            ds_platform = ds_platform.reindex(
                {
                    "ping_time": extra_platform_data[time_name].values,
                    "location_time": extra_platform_data[time_name].values,
                }
            )
            # merge extra platform data
            num_obs = len(extra_platform_data[time_name])
            ds_platform = ds_platform.update(
                {
                    "pitch": (
                        "ping_time",
                        mapping_get_multiple(
                            extra_platform_data,
                            ["pitch", "PITCH"],
                            np.full(num_obs, np.nan),
                        ),
                    ),
                    "roll": (
                        "ping_time",
                        mapping_get_multiple(
                            extra_platform_data,
                            ["roll", "ROLL"],
                            np.full(num_obs, np.nan),
                        ),
                    ),
                    "heave": (
                        "ping_time",
                        mapping_get_multiple(
                            extra_platform_data,
                            ["heave", "HEAVE"],
                            np.full(num_obs, np.nan),
                        ),
                    ),
                    "latitude": (
                        "location_time",
                        mapping_get_multiple(
                            extra_platform_data,
                            ["lat", "latitude", "LATITUDE"],
                            default=np.full(num_obs, np.nan),
                        ),
                    ),
                    "longitude": (
                        "location_time",
                        mapping_get_multiple(
                            extra_platform_data,
                            ["lon", "longitude", "LONGITUDE"],
                            default=np.full(num_obs, np.nan),
                        ),
                    ),
                    "water_level": (
                        "location_time",
                        mapping_get_multiple(
                            extra_platform_data,
                            ["water_level", "WATER_LEVEL"],
                            np.ones(num_obs),
                        ),
                    ),
                }
            )

        # need to close the file in order to remove it
        # (and need to close the file so to_netcdf can write to it)
        ds_platform.close()
        ds_beam.close()

        if engine == "netcdf4":
            # https://github.com/Unidata/netcdf4-python/issues/65
            # Copy groups over to temporary file
            # TODO: Add in documentation: recommended to use Zarr if using add_platform
            new_dataset_filename = f + ".temp"
            groups = ["Provenance", "Environment", "Beam", "Sonar", "Vendor"]
            with xr.open_dataset(f) as ds_top:
                ds_top.to_netcdf(new_dataset_filename, mode="w")
            for group in groups:
                with xr.open_dataset(f, group=group) as ds:
                    ds.to_netcdf(new_dataset_filename, mode="a", group=group)
            ds_platform.to_netcdf(new_dataset_filename, mode="a", group="Platform")
            # Replace original file with temporary file
            os.remove(f)
            os.rename(new_dataset_filename, f)
        elif engine == "zarr":
            # https://github.com/zarr-developers/zarr-python/issues/65
            old_dataset = zarr.open_group(f, mode="a")
            del old_dataset["Platform"]
            ds_platform.to_zarr(f, mode="a", group="Platform")
