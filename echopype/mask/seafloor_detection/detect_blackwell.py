import numpy as np
import xarray as xr
from scipy.ndimage import label
from scipy.signal import convolve2d


class DetectBlackwell:
    """
    Bottom detection using Sv and split-beam angles (Blackwell et al. 2019).

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset with 'Sv', 'angle_major', and 'angle_minor' variables.
    threshold : float or list/tuple of 3 floats
        Threshold(s) for Sv (dB), angle_major (theta), angle_minor (phi).
        If a single float is provided, used as Sv threshold and default angles are used.
    offset : float
        Constant offset (m) to add to detected bottom.
    channel : int or str
        Channel index or name.
    r0 : float
        Minimum detection range (default: 10 m).
    r1 : float
        Maximum detection range (default: 500 m).
    wtheta : int
        Half-window size for rolling median on angle_major (samples).
    wphi : int
        Half-window size for rolling median on angle_minor (samples).
    """

    def __init__(
        self,
        ds: xr.Dataset,
        threshold: float | list | tuple = -75,
        offset: float = 0.3,
        channel: int | str = 0,
        r0: float = 0,
        r1: float = 500,
        wtheta: int = 28,
        wphi: int = 52,
    ):
        self.ds = ds
        self.offset = offset
        self.r0 = r0
        self.r1 = r1
        self.wtheta = wtheta
        self.wphi = wphi

        # --- Parse threshold(s) ---
        if isinstance(threshold, (list, tuple)):
            if len(threshold) == 3:
                self.tSv, self.ttheta, self.tphi = threshold
            elif len(threshold) == 2:
                self.tSv = threshold[0]
                self.ttheta = 702  # default from original code
                self.tphi = 282
            else:
                raise ValueError(
                    "`threshold` must have 1 (Sv), 2 (Sv + default angles),\
                    or 3 elements (Sv, theta, phi)"
                )
        elif isinstance(threshold, (int, float)):
            self.tSv = threshold
            self.ttheta = 702
            self.tphi = 282
        else:
            raise TypeError("`threshold` must be float or list/tuple of 1–3 floats")

        # --- Select channel ---
        channels = self.ds["channel"].values
        if isinstance(channel, int):
            if not (0 <= channel < len(channels)):
                raise ValueError(f"Invalid channel index: {channel}")
            self.channel_sel = dict(channel=channel)
        elif isinstance(channel, str):
            if channel not in channels:
                raise ValueError(f"Channel '{channel}' not found. Available: {list(channels)}")
            self.channel_sel = dict(channel=channel)
        else:
            raise TypeError("`channel` must be an int or str")

    # to move in utils???
    @staticmethod
    def lin(variable):
        """
        Convert from logarithmic (dB) to linear scale.
        """
        return 10 ** (variable / 10)

    def compute(self) -> xr.Dataset:
        print("Running Blackwell seafloor detection...")

        # Select data for the requested channel
        Sv = self.ds["Sv"].sel(**self.channel_sel)
        theta = self.ds["angle_alongship"].sel(**self.channel_sel)
        phi = self.ds["angle_athwartship"].sel(**self.channel_sel)
        depth = self.ds["depth"].sel(**self.channel_sel)
        ping_time = self.ds["ping_time"]

        # print(depth)
        # print(ping_time)

        # Reference profile
        depth_ref = depth.isel(ping_time=0).values
        r = depth_ref  # used as 1D range/depth vector

        # Extract numpy arrays
        Sv_vals = Sv.values
        theta_vals = theta.values
        phi_vals = phi.values

        # Run detection core
        print("Blackwell core...")
        mask, _ = self._blackwell_core(Sv_vals, theta_vals, phi_vals, r)

        # print(f"✅ Mask computed. Shape: {mask.shape}, Type: {mask.dtype}")
        # print(f"   ➤ Number of pings with detected seafloor:
        # {(mask.any(axis=1)).sum()} / {mask.shape[0]}")

        # Get bottom index and depth
        bottom_sample_idx = mask.argmax(axis=1)
        bottom_depth = r[bottom_sample_idx] - self.offset

        # print("Bottom detection summary:")
        # print(f"Offset applied: {self.offset} m")
        # print(f"Detected depths (min/max): {np.nanmin(bottom_depth):.2f} m
        # / {np.nanmax(bottom_depth):.2f} m")
        # print(f"Mean detected depth: {np.nanmean(bottom_depth):.2f} m")

        # Dataset output
        chan_name = str(self.channel_sel["channel"])
        ds_out = xr.Dataset(
            data_vars=dict(
                bottom_depth=(["ping_time"], bottom_depth),
                seafloor_sample_range=(["ping_time"], bottom_sample_idx.astype("float32")),
            ),
            coords=dict(
                ping_time=ping_time,
            ),
            attrs=dict(
                bottom_detection_method="blackwell",
                bottom_Sv_threshold=self.tSv,
                bottom_angle_thresholds=[self.ttheta, self.tphi],
                bottom_offset_applied=self.offset,
                bottom_channel_used=chan_name,
            ),
        )

        # Optional: Add per-channel index to main dataset (for follow-up)
        range_sample_vec = self.ds["range_sample"].values
        depth_ref_chan = r
        idx_per_ping = []

        for bd in bottom_depth:
            if not np.isnan(bd):
                diffs = np.abs(depth_ref_chan - bd)
                closest_idx = np.argmin(diffs)
                idx_per_ping.append(range_sample_vec[closest_idx])
            else:
                idx_per_ping.append(np.nan)

        da = xr.DataArray(
            idx_per_ping,
            dims=["ping_time"],
            coords={"ping_time": ping_time},
            name=f"seafloor_sample_range_on_{chan_name}",
        )

        self.ds[da.name] = da

        return ds_out

    def _blackwell_core(self, Sv, theta, phi, r):

        # print(f"r.shape: {r.shape}")
        # print(f"r0 (min depth): {self.r0}")
        # print(f"r1 (max depth): {self.r1}")

        # Index positions
        r0_idx = np.nanargmin(abs(r - self.r0))
        r1_idx = np.nanargmin(abs(r - self.r1)) + 1
        # print(f"r0_idx: {r0_idx}, r1_idx: {r1_idx}")
        # print(f"len(r): {len(r)}, len(r) - r1_idx: {len(r) - r1_idx}")

        Svchunk = Sv[:, r0_idx:r1_idx]
        thetachunk = theta[:, r0_idx:r1_idx]
        phichunk = phi[:, r0_idx:r1_idx]

        # print(f"Svchunk.shape: {Svchunk.shape}")
        # print(f"thetachunk.shape: {thetachunk.shape}")
        # print(f"phichunk.shape: {phichunk.shape}")

        # print(f"Sv range: {np.nanmin(Svchunk):.2f} to {np.nanmax(Svchunk):.2f} dB")
        # print(f"Theta range: {np.nanmin(thetachunk):.2f} to {np.nanmax(thetachunk):.2f}°")
        # print(f"Phi range: {np.nanmin(phichunk):.2f} to {np.nanmax(phichunk):.2f}°")

        ktheta = np.ones((self.wtheta, self.wtheta)) / self.wtheta**2
        kphi = np.ones((self.wphi, self.wphi)) / self.wphi**2

        # print(f"ktheta.shape: {ktheta.shape}")
        # print(f"kphi.shape: {kphi.shape}")

        thetamaskchunk = convolve2d(thetachunk, ktheta, "same", boundary="symm") ** 2 > self.ttheta
        phimaskchunk = convolve2d(phichunk, kphi, "same", boundary="symm") ** 2 > self.tphi
        anglemaskchunk = thetamaskchunk | phimaskchunk

        # print(f"thetamaskchunk.shape: {thetamaskchunk.shape}")
        # print(f"phimaskchunk.shape: {phimaskchunk.shape}")
        # print(f"anglemaskchunk.shape: {anglemaskchunk.shape}")

        if anglemaskchunk.any():
            # attention au log/lin qu'on utilise : utilsier celui d echopype
            Svmedian = np.log10(np.nanmedian(self.lin(Svchunk[anglemaskchunk])))
            # print(f"Svmedian.shape: {Svmedian.shape}")

            if np.isnan(Svmedian):
                Svmedian = np.inf
            if Svmedian < self.tSv:
                Svmedian = self.tSv

            # print(f"Svmedian.shape after if: {Svmedian.shape}")

            Svmask = Svchunk > Svmedian

            # print(f"Svmask.shape: {Svmask.shape}")

            items = label(Svmask)[0]
            intercepted = list(set(items[anglemaskchunk]))
            if 0 in intercepted:
                intercepted.remove(0)

            # print(f"intercepted.shape: {intercepted.shape}")

            maskchunk = np.zeros(Svchunk.shape, dtype=bool)
            for i in intercepted:
                maskchunk |= items == i

            # print(f"maskchunk.shape: {maskchunk.shape}")

            above = np.zeros((maskchunk.shape[0], r0_idx), dtype=bool)  # padding before r0
            below = np.zeros((maskchunk.shape[0], len(r) - r1_idx), dtype=bool)  # padding after r1
            mask = np.hstack([above, maskchunk, below])  # horizontally stack (depth = axis 1)

            # print(f"above.shape: {above.shape}")
            # print(f"below.shape: {below.shape}")
            # print(f"mask.shape: {mask.shape}")

            assert (
                mask.shape == Sv.shape
            ), f"Mask shape {mask.shape} doesn't match Sv shape {Sv.shape}"

            return mask, anglemaskchunk

        return np.zeros_like(Sv, dtype=bool), np.zeros_like(Sv, dtype=bool)
