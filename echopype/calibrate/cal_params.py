from typing import Dict, Union

import numpy as np
import xarray as xr

from ..echodata import EchoData

CAL_PARAMS = {
    "EK": ("sa_correction", "gain_correction", "equivalent_beam_angle"),
    "AZFP": ("EL", "DS", "TVR", "VTX", "equivalent_beam_angle", "Sv_offset"),
}

# mapping param name between input varname and Vendor/Beam group data variable name
PARAM_VEND = {
    "gain": "gain",
    "angle_offset_alongship": "angle_offset_alongship",
    "angle_offset_athwartship": "angle_offset_athwartship",
    "beamwidth_alongship": "beamwidth_alongship",
    "beamwidth_athwartship": "beamwidth_athwartship",
    "z_et": "impedance_transmit",
}
PARAM_BEAM = {
    "gain": "gain_correction",
    "angle_offset_alongship": "angle_offset_alongship",
    "angle_offset_athwartship": "angle_offset_athwartship",
    "beamwidth_alongship": "beamwidth_twoway_alongship",
    "beamwidth_athwartship": "beamwidth_twoway_athwartship",
    "z_et": "z_et",  # from default EK80 params
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
    if user_cal_dict is None:
        user_cal_dict = {}

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

    # Params from the Beam group
    params_from_beam = [
        "angle_offset_alongship", "angle_offset_athwartship",
        "beamwidth_twoway_alongship", "beamwidth_twoway_athwartship",
    ]
    for p in params_from_beam:
        # substitute if p not in user input
        out_dict[p] = (
            user_cal_dict[p]
            if p in user_cal_dict
            else beam[p]
        )

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
        v = vend.attrs["%s %s filter_r" % (channel_id, filter_name)] + 1j * np.array(
            vend.attrs["%s %s filter_i" % (channel_id, filter_name)]
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


def get_param_BB(
    vend: xr.Dataset,
    varname: str,
    freq_center: xr.DataArray,
    cal_params_CW: Dict[str, xr.DataArray]
) -> xr.DataArray:
    """
    Get broadband gain or angle factor for calibrating complex samples.

    Interpolate gain or angle factor in the Vendor_specific group to the center frequency
    of each ping for BB mode samples if nominal frequency is within the calibrated frequency range

    Parameters
    ----------
    vend : xr.Dataset
        Vendor_specific group in an EchoData object
    varname : str
        the desired parameter
    freq_center : xr.DataArray
        center frequency of the transmit BB signal
    cal_params_CW : dict
        a dictionary storing CW calibration parameters as xr.DataArray

    Returns
    -------
    An xr.DataArray containing the interpolated gain or angle factor
    """
    param = []
    for ch_id in freq_center["channel"]:
        # if frequency-dependent gain/angle factor exists in Vendor group,
        # interpolate at center frequency
        if ch_id in vend["cal_channel_id"]:
            param_temp = (
                vend[PARAM_VEND[varname]]
                .sel(cal_channel_id=ch_id)
                .interp(cal_frequency=freq_center.sel(channel=ch_id))
                .drop(["cal_channel_id", "cal_frequency"])
                .expand_dims("channel")
            )
        # if no frequency-dependent gain/angle factor exists, use CW gain or default value
        else:
            if varname != "z_et":
                param_temp = (
                    cal_params_CW[PARAM_BEAM[varname]].sel(channel=ch_id)
                    # .reindex_like(echodata["Sonar/Beam_group1"]["backscatter_r"], method="nearest")  # noqa
                    .expand_dims("channel")
                )
            else:  # make it a data array if param a single value (true for default EK80 params)
                param_temp = xr.DataArray(
                    [cal_params_CW[PARAM_BEAM[varname]]],
                    dims=["channel"],
                    coords={"channel": [ch_id.data.tolist()]},
                )
        param_temp.name = varname
        param.append(param_temp)
    param = xr.merge(param)[varname]  # select the single data variable

    return param
