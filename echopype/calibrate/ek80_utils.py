import numpy as np
import xarray as xr
from scipy import signal

from ..echodata import EchoData
from .cal_params import get_vend_cal_params_complex_EK80


def tapered_chirp(
    fs,
    z_et,
    transmit_duration_nominal,
    slope,
    transmit_power,
    frequency_nominal=None,
    frequency_start=None,
    frequency_end=None,
):
    """Create a baseline chirp template."""
    if frequency_start is None and frequency_end is None:  # CW waveform
        frequency_start = frequency_nominal
        frequency_end = frequency_nominal

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
    return y_tmp / np.max(np.abs(y_tmp)), t  # amp has no actual effect


def filter_decimate_chirp(echodata: EchoData, fs: float, y: np.array, ch_id: str):
    """Filter and decimate the chirp template.

    Parameters
    ----------
    y : np.array
        chirp from _tapered_chirp
    ch_id : str
        channel_id to select the right coefficients and factors
    """
    # filter coefficients and decimation factor
    wbt_fil = get_vend_cal_params_complex_EK80(echodata, ch_id, "WBT", "coeff")
    pc_fil = get_vend_cal_params_complex_EK80(echodata, ch_id, "PC", "coeff")
    wbt_decifac = get_vend_cal_params_complex_EK80(echodata, ch_id, "WBT", "decimation")
    pc_decifac = get_vend_cal_params_complex_EK80(echodata, ch_id, "PC", "decimation")

    # WBT filter and decimation
    ytx_wbt = signal.convolve(y, wbt_fil)
    ytx_wbt_deci = ytx_wbt[0::wbt_decifac]

    # PC filter and decimation
    if len(pc_fil.squeeze().shape) == 0:  # in case it is a single element
        pc_fil = [pc_fil.squeeze()]
    ytx_pc = signal.convolve(ytx_wbt_deci, pc_fil)
    ytx_pc_deci = ytx_pc[0::pc_decifac]
    ytx_pc_deci_time = np.arange(ytx_pc_deci.size) * 1 / fs * wbt_decifac * pc_decifac

    return ytx_pc_deci, ytx_pc_deci_time


def get_tau_effective(echodata: EchoData, ytx: np.array, waveform_mode: str):
    """Compute effective pulse length.

    Parameters
    ----------
    ytx : array
        transmit signal
    fs_deci : float
        sampling frequency of the decimated (recorded) signal
    waveform_mode : str
        ``CW`` for CW-mode samples, either recorded as complex or power samples
        ``BB`` for BB-mode samples, recorded as complex samples
    """
    # TODO: change this to handle a dictionary of ytx with keys being the channel_id

    # TODO: tau_effective_tmp has a ping_time dimension
    # because fs_deci has a ping_time dimension,
    # probably should be removed

    chan = ytx.keys()
    fs_deci = 1 / echodata["Sonar/Beam_group1"].sel(channel=chan)["sample_interval"].values

    if waveform_mode == "BB":
        ytxa = signal.convolve(ytx, np.flip(np.conj(ytx))) / np.linalg.norm(ytx) ** 2
        ptxa = np.abs(ytxa) ** 2
    elif waveform_mode == "CW":
        ptxa = np.abs(ytx) ** 2  # energy of transmit signal
    return ptxa.sum() / (ptxa.max() * fs_deci)


def get_transmit_chirp(echodata: EchoData, waveform_mode: str, fs: float, z_et: float):
    """Reconstruct transmit signal and compute effective pulse length.

    Parameters
    ----------
    waveform_mode : str
        ``CW`` for CW-mode samples, either recorded as complex or power samples
        ``BB`` for BB-mode samples, recorded as complex samples
    """
    # Make sure it is BB mode data
    if waveform_mode == "BB" and (
        ("frequency_start" not in echodata["Sonar/Beam_group1"])
        or ("frequency_end" not in echodata["Sonar/Beam_group1"])
    ):
        raise TypeError("File does not contain BB mode complex samples!")

    y_all = {}
    y_time_all = {}
    for chan in echodata["Sonar/Beam_group1"].channel.values:
        # TODO: currently only deal with the case with
        # a fixed tx key param values within a channel
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
            tx_params[p] = np.unique(echodata["Sonar/Beam_group1"][p].sel(channel=chan))
            if tx_params[p].size != 1:
                raise TypeError("File contains changing %s!" % p)
        tx_params["fs"] = fs
        tx_params["z_et"] = z_et
        y_tmp, _ = tapered_chirp(**tx_params)

        # Filter and decimate chirp template
        fs_deci = 1 / echodata["Sonar/Beam_group1"].sel(channel=chan)["sample_interval"].values
        y_tmp, y_tmp_time = filter_decimate_chirp(echodata=echodata, fs=fs, y=y_tmp, ch_id=chan)

        y_all[chan] = y_tmp
        y_time_all[chan] = y_tmp_time

    return y_all, y_time_all


def compress_pulse(echodata: EchoData, chirp, chan_BB=None):
    """Perform pulse compression on the backscatter data.

    Parameters
    ----------
    chirp : dict
        transmit chirp replica indexed by channel_id
    chan_BB : str
        channels that transmit in BB mode
        (since CW mode can be in mixed in complex samples too)
    """
    backscatter = echodata["Sonar/Beam_group1"]["backscatter_r"].sel(
        channel=chan_BB
    ) + 1j * echodata["Sonar/Beam_group1"]["backscatter_i"].sel(channel=chan_BB)

    pc_all = []
    for chan in chan_BB:
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
            exclude_dims={"range_sample"},
        )

        # Expand dimension and add name to allow merge
        pc = pc.expand_dims(dim="channel")
        pc.name = "pulse_compressed_output"
        pc_all.append(pc)

    pc_merge = xr.merge(pc_all)

    return pc_merge
