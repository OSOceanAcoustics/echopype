from collections import defaultdict
from typing import Dict, Literal, Union

import numpy as np
import xarray as xr
from scipy import signal


def tapered_chirp(
    fs,
    transmit_duration_nominal,
    slope,
    transmit_power,
    implementation="Anderson",
    z_et=None,
    frequency_nominal=None,
    frequency_start=None,
    frequency_end=None,
):
    """Create a baseline chirp template."""
    if frequency_start is None and frequency_end is None:  # CW waveform
        frequency_start = frequency_nominal
        frequency_end = frequency_nominal

    if implementation == "Macaulay":
        # z_et is required for Macaulay implementation
        if z_et is None:
            raise ValueError("z_et is needed for Macaulay implementation of transmit chirp!")

        t = np.arange(0, transmit_duration_nominal, 1 / fs)
        nwtx = int(2 * np.floor(slope * t.size))  # length of tapering window
        wtx_tmp = np.hanning(nwtx)  # hanning window
        nwtxh = int(np.round(nwtx / 2))  # half length of the hanning window
        wtx = np.concatenate(
            [wtx_tmp[0:nwtxh], np.ones((t.size - nwtx)), wtx_tmp[nwtxh:]]
        )  # assemble full tapering window
        chirp_fac = (
            (frequency_end - frequency_start) / transmit_duration_nominal
        ) * t / 2 + frequency_start
        y_tmp = (
            np.sqrt((transmit_power / 4) * (2 * z_et))  # amplitude
            * np.cos(2 * np.pi * chirp_fac * t)  # chirp
            * wtx  # tapering
        )  # taper and scale linear chirp
        return y_tmp / np.max(np.abs(y_tmp)), t  # amplitude needs to be normalized

    elif implementation == "Anderson":
        # Substitute to keep original form in Anderson implementation
        # source: https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-Raw-To-Svf-TSf/blob/main/Core/Calculation.py  # noqa
        tau = transmit_duration_nominal
        f0 = frequency_start
        f1 = frequency_end

        nsamples = int(np.floor(tau * fs))
        t = np.linspace(0, nsamples - 1, num=nsamples) * 1 / fs
        a = np.pi * (f1 - f0) / tau
        b = 2 * np.pi * f0
        y = np.cos(a * t * t + b * t)
        L = int(np.round(tau * fs * slope * 2.0))  # Length of hanning window
        w = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(0, L, 1) / (L - 1)))
        N = len(y)
        w1 = w[0 : int(len(w) / 2)]
        w2 = w[int(len(w) / 2) : -1]
        i0 = 0
        i1 = len(w1)
        i2 = N - len(w2)
        i3 = N
        y[i0:i1] = y[i0:i1] * w1
        y[i2:i3] = y[i2:i3] * w2

        return y / np.max(y), t  # amplitude needs to be normalized

    else:
        raise ValueError("Input implementation type not recognized!")


def filter_decimate_chirp(coeff_ch: Dict, y_ch: np.array, fs: float):
    """Filter and decimate the transmit replica for one channel.

    Parameters
    ----------
    coeff_ch : dict
        a dictionary containing filter coefficients and decimation factors for ``ch_id``
    y_ch : np.array
        chirp from _tapered_chirp
    fs : float
        system sampling frequency [Hz]
    """
    # Get values

    # WBT filter and decimation
    ytx_wbt = signal.convolve(y_ch, coeff_ch["wbt_fil"])
    ytx_wbt_deci = ytx_wbt[0 :: coeff_ch["wbt_decifac"]]

    # PC filter and decimation
    if len(coeff_ch["pc_fil"].squeeze().shape) == 0:  # in case it is a single element
        coeff_ch["pc_fil"] = [coeff_ch["pc_fil"].squeeze()]
    ytx_pc = signal.convolve(ytx_wbt_deci, coeff_ch["pc_fil"])
    ytx_pc_deci = ytx_pc[0 :: coeff_ch["pc_decifac"]]
    ytx_pc_deci_time = (
        np.arange(ytx_pc_deci.size) * 1 / fs * coeff_ch["wbt_decifac"] * coeff_ch["pc_decifac"]
    )

    return ytx_pc_deci, ytx_pc_deci_time


def get_vend_filter_EK80(
    vend: xr.Dataset, channel_id: str, filter_name: str, param_type: Literal["coeff", "decimation"]
) -> Union[np.ndarray, int]:
    """
    Get filter coefficients stored in the Vendor_specific group attributes.

    Parameters
    ----------
    vend: xr.Dataset
        An xr.Dataset from EchoData["Vendor_specific"]
    channel_id : str
        channel id for which the param to be retrieved
    filter_name : str
        name of filter coefficients to retrieve
    param_type : str
        'coeff' or 'decimation'

    Returns
    -------
    np.ndarray or int
        The filter coefficient or the decimation factor
    """
    if param_type == "coeff":
        v = vend.attrs[f"{channel_id} {filter_name} filter_r"] + 1j * np.array(
            vend.attrs[f"{channel_id} {filter_name} filter_i"]
        )
        if v.size == 1:
            v = np.expand_dims(v, axis=0)  # expand dims for convolution
        return v
    else:
        return vend.attrs[f"{channel_id} {filter_name} decimation"]


def get_filter_coeff(vend: xr.Dataset) -> Dict:
    """
    Get WBT and PC filter coefficients for constructing the transmit replica.

    Parameters
    ----------
    vend: xr.Dataset
        An xr.Dataset from EchoData["Vendor_specific"]

    Returns
    -------
    dict
        A dictionary indexed by ``channel`` and values being dictionaries containing
        filter coefficients and decimation factors for constructing the transmit replica.
    """
    coeff = defaultdict(dict)
    for ch_id in vend["channel"].values:
        # filter coefficients and decimation factor
        coeff[ch_id]["wbt_fil"] = get_vend_filter_EK80(vend, ch_id, "WBT", "coeff")
        coeff[ch_id]["pc_fil"] = get_vend_filter_EK80(vend, ch_id, "PC", "coeff")
        coeff[ch_id]["wbt_decifac"] = get_vend_filter_EK80(vend, ch_id, "WBT", "decimation")
        coeff[ch_id]["pc_decifac"] = get_vend_filter_EK80(vend, ch_id, "PC", "decimation")

    return coeff


def get_tau_effective(
    ytx_dict: Dict[str, np.array],
    fs_deci_dict: Dict[str, float],
    waveform_mode: str,
    channel: xr.DataArray,
    ping_time: xr.DataArray,
):
    """Compute effective pulse length.

    Parameters
    ----------
    ytx_dict : dict
        A dict of transmit signals, with keys being the ``channel`` and
        values being either a vector when the transmit signals are identical across all pings
        or a 2D array when the transmit signals vary across ping
    fs_deci_dict : dict
        A dict of sampling frequency of the decimated (recorded) signal,
        with keys being the ``channel``
    waveform_mode : str
        ``CW`` for CW-mode samples, either recorded as complex or power samples
        ``BB`` for BB-mode samples, recorded as complex samples
    """
    tau_effective = {}
    for ch, ytx in ytx_dict.items():
        if waveform_mode == "BB":
            ytxa = signal.convolve(ytx, np.flip(np.conj(ytx))) / np.linalg.norm(ytx) ** 2
            ptxa = np.abs(ytxa) ** 2
        elif waveform_mode == "CW":
            ptxa = np.abs(ytx) ** 2  # energy of transmit signal
        tau_effective[ch] = ptxa.sum() / (ptxa.max() * fs_deci_dict[ch])

    # set up coordinates
    if len(ytx.shape) == 1:  # ytx is a vector (transmit signals are identical across pings)
        coords = {"channel": channel}
    elif len(ytx.shape) == 2:  # ytx is a matrix (transmit signals vary across pings)
        coords = {"channel": channel, "ping_time": ping_time}

    vals = np.array(list(tau_effective.values())).squeeze()
    if vals.size == 1:
        vals = np.expand_dims(vals, axis=0)

    tau_effective = xr.DataArray(
        data=vals,
        coords=coords,
    )

    return tau_effective


def get_transmit_signal(
    beam: xr.Dataset,
    coeff: Dict,
    waveform_mode: str,
    fs: Union[float, xr.DataArray],
):
    """Reconstruct transmit signal and compute effective pulse length.

    Parameters
    ----------
    beam : xr.Dataset
        EchoData["Sonar/Beam_group1"] selected with channel subset
    coeff : dict
        a dictionary indexed by ``channel`` and values being dictionaries containing
        filter coefficients and decimation factors for constructing the transmit replica.
    waveform_mode : str
        ``CW`` for CW-mode samples, either recorded as complex or power samples
        ``BB`` for BB-mode samples, recorded as complex samples

    Return
    ------
    y_all
        Transmit replica (BB: broadband chirp, CW: constant frequency sinusoid)
    y_time_all
        Timestamp for the transmit replica
    """
    # Make sure it is BB mode data
    # This is already checked in calibrate_ek
    # but keeping this here for use as standalone function
    if waveform_mode == "BB" and (("frequency_start" not in beam) or ("frequency_end" not in beam)):
        raise TypeError("File does not contain BB mode complex samples!")

    # Generate all transmit replica
    y_all = {}
    y_time_all = {}
    for ch in beam["channel"].values:
        # TODO: expand to deal with the case with varying tx param across ping_time
        if waveform_mode == "BB":
            tx_param_names = [
                "transmit_duration_nominal",
                "slope",
                "transmit_power",
                "frequency_start",
                "frequency_end",
            ]
        else:
            tx_param_names = [
                "transmit_duration_nominal",
                "slope",
                "transmit_power",
                "frequency_nominal",
            ]
        tx_params = {}
        for p in tx_param_names:
            tx_params[p] = np.unique(beam[p].sel(channel=ch))
            if tx_params[p].size != 1:
                raise TypeError("File contains changing %s!" % p)
        fs_chan = fs.sel(channel=ch).data if isinstance(fs, xr.DataArray) else fs
        tx_params["fs"] = fs_chan
        y_ch, _ = tapered_chirp(**tx_params)

        # Filter and decimate chirp template
        y_ch, y_tmp_time = filter_decimate_chirp(coeff_ch=coeff[ch], y_ch=y_ch, fs=fs_chan)

        # Fill into output dict
        y_all[ch] = y_ch
        y_time_all[ch] = y_tmp_time

    return y_all, y_time_all


def compress_pulse(backscatter: xr.DataArray, chirp: Dict) -> xr.DataArray:
    """Perform pulse compression on the backscatter data.

    Parameters
    ----------
    backscatter : xr.DataArray
        complex backscatter samples
    chirp : dict
        transmit chirp replica indexed by ``channel``

    Returns
    -------
    xr.DataArray
        A data array containing pulse compression output.
    """
    pc_all = []
    for chan in backscatter["channel"]:
        backscatter_chan = (
            backscatter.sel(channel=chan)
            # .dropna(dim="range_sample", how="all")
            .dropna(dim="beam", how="all")
            # .dropna(dim="ping_time")
        )

        tx = chirp[str(chan.values)]
        replica = np.flipud(np.conj(tx))
        pc = xr.apply_ufunc(
            lambda m: np.apply_along_axis(
                lambda m: (
                    signal.convolve(m, replica, mode="full")[tx.size - 1 :]
                    / np.linalg.norm(tx) ** 2
                ),
                axis=2,
                arr=m,
            ),
            backscatter_chan,
            input_core_dims=[["range_sample"]],
            output_core_dims=[["range_sample"]],
            # exclude_dims={"range_sample"},
        )

        pc_all.append(pc)

    pc_all = xr.DataArray(
        pc_all,
        coords={
            "channel": backscatter["channel"],
            "ping_time": backscatter["ping_time"],
            "beam": backscatter["beam"],
            "range_sample": backscatter["range_sample"],
        },
    )

    return pc_all
