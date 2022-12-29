from typing import Dict, Union

import numpy as np
import xarray as xr

from ..echodata import EchoData

CAL_PARAMS = {
    "EK": ("sa_correction", "gain_correction", "equivalent_beam_angle"),
    "AZFP": ("EL", "DS", "TVR", "VTX", "equivalent_beam_angle", "Sv_offset"),
}


# TODO: need a function (something like "_check_param_freq_dep")
# to check user input cal_params and env_params


def get_cal_params_AZFP(echodata: EchoData, user_cal_dict: dict) -> dict:
    """
    Get cal params using user inputs or values from data file.

    Parameters
    ----------
    echodata : EchoData
        An EchoData object containing data to be calibrated
    user_cal_dict : dict
        A dictionary containing user-defined calibration parameters.
        The user-defined calibration parameters will overwrite values in the data file.

    Returns
    -------
    A dict containing the calibration parameters for the AZFP echosounder
    """
    out_dict = dict.fromkeys(CAL_PARAMS["AZFP"])

    # Get params from Beam_group1
    out_dict["equivalent_beam_angle"] = (
        user_cal_dict["equivalent_beam_angle"]
        if "equivalent_beam_angle" in user_cal_dict
        else echodata["Sonar/Beam_group1"]["equivalent_beam_angle"]
    )

    # Get params from the Vendor_specific group
    for p in ["EL", "DS", "TVR", "VTX", "Sv_offset"]:
        # substitute if None in user input
        out_dict[p] = user_cal_dict[p] if p in user_cal_dict else echodata["Vendor_specific"][p]

    return out_dict


def get_cal_params_EK(
    beam: xr.Dataset, vend: xr.Dataset, user_cal_dict: Dict[str, xr.DataArray]
) -> Dict:
    """
    Get cal params using user inputs or values from data file.

    Parameters
    ----------
    beam : xr.Dataset
        A subset of Sonar/Beam_groupX that contains only the channels specified for calibration
    vend : xr.Dataset
        A subset of Vendor_specific that contains only the channels specified for calibration
    user_cal_dict : dict
        A dictionary containing user-defined calibration parameters.
        The user-defined calibration parameters will overwrite values in the data file.

    Returns
    -------
    A dictionary containing the calibration parameters for the EK echosounders
    """
    out_dict = dict.fromkeys(CAL_PARAMS["EK"])
    if user_cal_dict is None:
        user_cal_dict = {}

    # Params from the Vendor_specific group
    params_from_vend = ["sa_correction", "gain_correction"]
    for p in params_from_vend:
        # substitute if p not in user input
        out_dict[p] = (
            user_cal_dict[p]
            if p in user_cal_dict
            else get_vend_cal_params_power(beam=beam, vend=vend, param=p)
        )

    # Other params
    out_dict["equivalent_beam_angle"] = (
        user_cal_dict["equivalent_beam_angle"]
        if "equivalent_beam_angle" in user_cal_dict
        else beam["equivalent_beam_angle"]
    )

    return out_dict


def get_vend_filter_EK80(
    vend: xr.Dataset, channel_id: str, filter_name: str, param_type: str
) -> Union[np.ndarray, int]:
    """
    Get filter coefficients stored in the Vendor_specific group attributes.

    Parameters
    ----------
    vend: xr.Dataset
        EchoData["Vendor_specific"]
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
        v = (
            vend.attrs["%s %s filter_r" % (channel_id, filter_name)]
            + 1j * np.array(vend.attrs["%s %s filter_i" % (channel_id, filter_name)])
        )
        if v.size == 1:
            v = np.expand_dims(v, axis=0)  # expand dims for convolution
        return v
    else:
        return vend.attrs["%s %s decimation" % (channel_id, filter_name)]


def get_vend_cal_params_power(beam: xr.Dataset, vend: xr.Dataset, param: str) -> xr.DataArray:
    """
    Get cal parameters stored in the Vendor_specific group
    by matching the transmit_duration_nominal with allowable pulse_length.

    Parameters
    ----------
    beam : xr.Dataset
        A subset of Sonar/Beam_groupX that contains only the channels specified for calibration
    vend : xr.Dataset
        A subset of Vendor_specific that contains only the channels specified for calibration
    param : str {"sa_correction", "gain_correction"}
        name of parameter to retrieve

    Returns
    -------
    An xr.DataArray containing the matched ``param``
    """

    # Check parameter is among those allowed
    if param not in ["sa_correction", "gain_correction"]:
        raise ValueError(f"Unknown parameter {param}")

    # Check parameter exists
    if param not in vend:
        raise ValueError(f"{param} does not exist in the Vendor_specific group!")

    # Find idx to select the corresponding param value
    # by matching beam["transmit_duration_nominal"] with ds_vend["pulse_length"]
    transmit_isnull = beam["transmit_duration_nominal"].isnull()
    idxmin = np.abs(beam["transmit_duration_nominal"] - vend["pulse_length"]).idxmin(
        dim="pulse_length_bin"
    )

    # fill nan position with 0 (will remove before return)
    # and convert to int for indexing
    idxmin = idxmin.where(~transmit_isnull, 0).astype(int)

    # Get param dataarray into correct shape
    da_param = (
        vend[param]
        .expand_dims(dim={"ping_time": idxmin["ping_time"]})  # expand dims for direct indexing
        .sortby(idxmin.channel)  # sortby in case channel sequence differs in vend and beam
    )

    # Select corresponding index and clean up the original nan elements
    da_param = da_param.sel(pulse_length_bin=idxmin, drop=True)
    return da_param.where(~transmit_isnull, np.nan)  # set the nan elements back to nan


def get_gain_BB(
    vend: xr.Dataset, freq_center: xr.DataArray, cal_params_CW: Dict[str, xr.DataArray]
) -> xr.DataArray:
    """
    Get broadband gain factor for calibrating complex samples.

    Interpolate ``gain`` in the Vendor_specific group to the center frequency of each ping
    for BB mode samples if nominal frequency is within the calibrated frequency range

    Parameters
    ----------
    waveform_mode : str
        ``CW`` for CW-mode samples, either recorded as complex or power samples
        ``BB`` for BB-mode samples, recorded as complex samples
    chan_sel : xr.DataArray
        Nominal channel for CW mode samples
        and a xr.DataArray of selected channels for BB mode samples

    Returns
    -------
    An xr.DataArray containing the interpolated gain factors
    """
    gain = []
    for ch_id in freq_center["channel"]:
        # if frequency-dependent gain exists in Vendor group, interpolate at center frequency
        if ch_id in vend["cal_channel_id"]:
            gain_temp = (
                vend["gain"]
                .sel(cal_channel_id=ch_id)
                .interp(cal_frequency=freq_center.sel(channel=ch_id))
                .drop(["cal_channel_id", "cal_frequency"])
                .expand_dims("channel")
            )
        # if no frequency-dependent gain exists, use CW gain
        else:
            gain_temp = (
                cal_params_CW["gain_correction"].sel(channel=ch_id)
                # .reindex_like(echodata["Sonar/Beam_group1"]["backscatter_r"], method="nearest")
                .expand_dims("channel")
            )
        gain_temp.name = "gain"
        gain.append(gain_temp)
    gain = xr.merge(gain).gain  # select the single data variable

    return gain
