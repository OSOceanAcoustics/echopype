from typing import Optional

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

matplotlib.use("TkAgg")  # Must come BEFORE importing pyplot?


# first test to explore how we could implement it, to be updated
class DetectManual:
    def __init__(self, ds: xr.Dataset, initial_bottom: Optional[xr.DataArray] = None):
        """
        Parameters:
        - ds: Dataset with 'Sv' and 'depth' over multiple frequencies
        - initial_bottom: (frequency_nominal, ping_time) DataArray
        """
        self.ds = ds
        self.Sv_all = ds["Sv"]
        self.depth_all = ds["depth"]
        self.ping_time = ds["ping_time"]

        if initial_bottom is not None:
            self.bottom_all = initial_bottom
        else:
            self.bottom_all = xr.zeros_like(ds["depth"])

        self.current_channel = 0  # start with first channel

    def compute(self) -> xr.Dataset:
        """
        Interactive visualisation for manual bottom adjustment on one channel.
        (future: support GUI channel switching)
        """

        Sv = self.Sv_all.isel(frequency_nominal=self.current_channel)
        depth = self.depth_all  # verify
        bottom = self.bottom_all.isel(frequency_nominal=self.current_channel)

        fig, ax = plt.subplots(figsize=(12, 6))
        pcm = ax.pcolormesh(self.ping_time, depth, Sv.T, shading="nearest", cmap="RdYlBu_r")
        (line,) = ax.plot(self.ping_time, bottom, color="black", lw=1.5, label="Bottom")
        ax.set_ylim(depth.max(), depth.min())
        ax.set_title(f"Click to define new bottom — Channel {self.current_channel}")
        ax.set_xlabel("Ping time")
        ax.set_ylabel("Depth (m)")
        plt.colorbar(pcm, ax=ax, label="Sv (dB)")
        plt.legend()

        print("Click two points to interpolate bottom between them...")
        clicks = plt.ginput(n=2, timeout=0)
        plt.close()

        if len(clicks) != 2:
            print("You must click exactly two points. Keeping original bottom.")
            return xr.Dataset({"bottom_depth": self.bottom_all})

        # Unpack clicked points
        (t1, d1), (t2, d2) = sorted(clicks)

        print((t1, d1), (t2, d2))

        # Convert click times (matplotlib float days) to datetime64[ns]
        t1_dt64 = np.datetime64(mdates.num2date(t1))
        t2_dt64 = np.datetime64(mdates.num2date(t2))

        print((t1_dt64, d1), (t2_dt64, d2))

        # Create boolean mask between t1 and t2 (inclusive)
        mask = (self.ping_time >= t1_dt64) & (self.ping_time <= t2_dt64)

        # Interpolation of depths at matching ping_times
        interp_vals = np.interp(
            self.ping_time[mask].values.astype("datetime64[ns]").astype("float64"),
            [
                t1_dt64.astype("datetime64[ns]").astype("float64"),
                t2_dt64.astype("datetime64[ns]").astype("float64"),
            ],
            [d1, d2],
        )

        # Update the bottom values for this frequency
        self.bottom_all[self.current_channel, mask] = xr.DataArray(
            interp_vals, dims=["ping_time"], coords={"ping_time": self.ping_time[mask]}
        )

        return xr.Dataset({"bottom_depth": self.bottom_all})
