# import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


class DetectBasic:
    """
    Basic threshold-based bottom detection method.

    Parameters
    ----------
    ds : xr.Dataset (EchoData)
    threshold : float or tuple of float
        Sv threshold range (min, max) in dB for bottom detection.
    offset : float
        Vertical offset (in meters) to add to the detected bottom.
    channel : int or str
        Index or name of the frequency channel to use for detection.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        threshold: float | list[float] | tuple[float, float] = -50,
        offset: float = 0.3,
        channel: int | str = 0,
    ):
        self.ds = ds
        self.offset = offset

        # --- Validate and store threshold ---
        if isinstance(threshold, (list, tuple)) and len(threshold) == 2:
            tmin, tmax = threshold
            if tmax <= tmin:
                raise ValueError("`threshold_max` must be strictly greater than `threshold_min`.")
            self.threshold_min = tmin
            self.threshold_max = tmax
        elif isinstance(threshold, (int, float)):
            self.threshold_min = threshold
            self.threshold_max = threshold + 10  # default range
        else:
            raise TypeError("`threshold` must be a float or a list/tuple of two floats.")

        # --- Validate and store channel ---
        channels = self.ds["channel"].values

        if isinstance(channel, int):
            if not (0 <= channel < len(channels)):
                raise ValueError(
                    f"Invalid channel index: {channel}. Available indices: 0 to {len(channels) - 1}"
                )
            self.channel_sel = dict(channel=channel)

        elif isinstance(channel, str):
            if channel not in channels:
                raise ValueError(
                    f"Channel '{channel}' not found. Available names: {list(channels)}"
                )
            self.channel_sel = dict(channel=channel)

        else:
            raise TypeError("`channel` must be an integer index or a string channel name.")

    def compute(self) -> xr.Dataset:
        print("Running basic bottom detection...")

        # Select the Sv for the requested channel
        Sv = self.ds["Sv"].sel(**self.channel_sel)

        # Get range_sample and ping_time
        if "range_sample" in self.ds.coords:
            range_sample = self.ds["range_sample"]
        else:
            raise KeyError("'range_sample' coordinate not found in dataset.")

        ping_time = self.ds["ping_time"]

        # Select matching depth slice
        if "depth" in self.ds.data_vars:
            depth = self.ds["depth"].sel(**self.channel_sel)
            # print("depth dims (selected):", depth.dims)
            # print("depth shape:", depth.shape)
        else:
            raise KeyError("'depth' variable not found in dataset.")

        # asserts
        assert Sv.shape == depth.shape, f"Shape mismatch: Sv {Sv.shape} vs depth {depth.shape}"
        if Sv.dims != depth.dims:
            raise ValueError(
                "Dimension names mismatch: Sv.dims = {Sv.dims}, depth.dims = {depth.dims}"
            )

        assert Sv.shape[0] == ping_time.size, "Mismatch: ping_time != number of Sv rows"
        assert Sv.shape[1] == range_sample.size, "Mismatch: range_sample != number of Sv columns"

        # Reference depth profile (1D)
        depth_ref = depth.isel(ping_time=0)
        max_diff = abs(depth - depth_ref).max(dim="range_sample")
        is_uniform_depth = max_diff < 1e-16

        if not is_uniform_depth.all():
            raise ValueError("Depth profile varies across ping_times.")

        # Limit Sv to depths below the first 200 range samples to account
        # for the moment for surface saturation zone
        Sv_deep = Sv.isel(range_sample=slice(200, None))

        condition = (Sv_deep > self.threshold_min) & (Sv_deep < self.threshold_max)

        bottom_sample_idx_local = condition.argmax(dim="range_sample")
        bottom_sample_idx = bottom_sample_idx_local + 200

        bottom_depth = depth_ref.values[bottom_sample_idx.values] - self.offset

        # Convert condition to float (1 for True, 0 for False) for plotting
        # condition_plot = condition.astype(float)

        # # Plot (can slice to zoom in, e.g., first 100 pings)
        # condition_plot.plot(
        #     x="ping_time",
        #     y="range_sample",
        #     cmap="Greys",
        #     vmin=0, vmax=1,
        #     cbar_kwargs={"label": "Condition met (1=True)"},
        #     yincrease=False
        # )

        # plt.plot(
        #     ping_time,
        #     bottom_sample_idx,  # y-axis: detected sample
        #     color="red",
        #     linewidth=2,
        #     label="Detected bottom"
        # )

        # plt.title("Threshold condition (True regions)")
        # plt.xlabel("Ping time")
        # plt.ylabel("Range sample (starting at 200)")
        # plt.tight_layout()
        # plt.show()

        for chan in self.ds["channel"].values:
            chan_name = str(chan)

            # Get depth_ref for this channel
            depth_chan = self.ds["depth"].sel(channel=chan)
            depth_ref_chan = depth_chan.isel(ping_time=0).values  # shape: (range_sample,)

            # print(f"\n Channel: {chan_name}")
            # print(f"   Depth range: {np.nanmin(depth_ref_chan):.2f} m
            # to {np.nanmax(depth_ref_chan):.2f} m")
            # print(f"   Max bottom_depth: {np.nanmax(bottom_depth):.2f} m")

            range_sample_vec = self.ds["range_sample"].values
            rs_values = []

            for i, bd in enumerate(bottom_depth):
                if not np.isnan(bd):
                    valid_idx = np.where(~np.isnan(depth_ref_chan))[0]
                    if valid_idx.size > 0:
                        diffs = np.abs(depth_ref_chan[valid_idx] - bd)
                        closest_idx = valid_idx[np.argmin(diffs)]
                        # closest = depth_ref_chan[closest_idx]
                        rs_value = range_sample_vec[closest_idx]
                    else:
                        # closest = np.nan
                        rs_value = np.nan
                else:
                    # closest = np.nan
                    rs_value = np.nan

                # print(f" ping {i:03}: bottom_depth={bd:.2f}:
                # closest={closest:.2f} @ range_sample={rs_value}")
                rs_values.append(rs_value)

            # Convert to array and store
            idx_per_ping = np.array(rs_values)

            da = xr.DataArray(
                idx_per_ping,
                dims=["ping_time"],
                coords={"ping_time": self.ds["ping_time"]},
                name=f"seafloor_sample_range_on_{chan_name}",
            )

            self.ds[da.name] = da

        return self.ds
