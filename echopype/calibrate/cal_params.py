from typing import Dict, Union, List

import numpy as np
import xarray as xr

from ..echodata import EchoData

CAL_PARAMS = {
    "EK": (  # TODO: consider including impedance?
        "sa_correction",
        "gain_correction",
        "equivalent_beam_angle",
        "angle_offset_alongship",
        "angle_offset_athwartship",
        "beamwidth_alongship",
        "beamwidth_athwartship",
        "impedance_transceiver",
        "impedance_transducer",
        "receiver_sampling_frequency",
    ),
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


def param_dict2da(
    p_dict: Dict[str, Union[int, float]]
) -> xr.DataArray:
    """
    Organize calibration parameters in dict to xr.DataArray.

    Allowable input dictionary format:
    - dict{channel: values}
    - TODO: dict{frequency: values}

    Parameters
    ----------
    p_dict : dict
        dictionary holding calibration params for one or more channels
        each param has to be a scalar
    """
    # TODO: allow passing in np.array as dict values to assemble a frequency-dependent cal da

    k_list = []
    v_list = []
    for k, v in p_dict.items():
        k_list.append(k)  # coordinate
        v_list.append(v)  # values

    return xr.DataArray(v_list, coords=("channel", k_list), dims="channel")


# TODO: allow Dict[Union[float, str], np.array] as user_dict values
def sanitize_user_cal_dict(
    user_dict: Dict[str, Union[float, xr.DataArray]],
    channel: Union[List, xr.DataArray],
    sonar_model: str,
) -> Dict:
    """
    Check the format and organize user-provided cal_params dict.

    Parameters
    ----------
    user_dict : dict
        a dict containing user input calibration parameters as {parameter name: parameter value}
        parameter value has to be a scalar (int or float) or an xr.DataArray
    channel : list or xr.DataArray
        a list of channels to be calibrated
        for EK80 data, this list has to corresponds with the subset of channels
        selected based on waveform_mode and encode_mode
    """
    # Make channel a sorted list
    if not isinstance(channel, (list, xr.DataArray)):
        raise ValueError("'channel' has to be a list or an xr.DataArray")
    else:
        if isinstance(channel, xr.DataArray):
            return sorted(list(channel.data))  # TODO: check if it works with just sorted()
        else:
            return sorted(channel)

    # Screen parameters: only retain those defined in CAL_PARAMS
    out_dict = dict.fromkeys(CAL_PARAMS[sonar_model])
    for p_name in user_dict:
        if p_name in out_dict:
            out_dict[p_name] = user_dict[p_name]

    # Check allowable type of each param item
    # - scalar: no change
    # - list: does NOT allow because there is no explict channel or frequency correpsondence
    # - xr.DataArray: no change
    # TODO: - dict: convert to xr.DataArray
    for p_name, p_val in out_dict.items():
        if not isinstance(p_val, (int, float, xr.DataArray)):
            raise ValueError(
                f"{p_name} has to be a scalar (int or float) or an xr.DataArray"
            )

        # TODO: allow parameter xr.DataArray to be aligned with frequency as well
        # If a parameter is an xr.DataArray, its channel coordinate has to be identical with
        # that of data to be calibrated, ie no missing or extra channels/frequencies should exist
        # (in theory can allow extra ones, but seems better to just require things to be identical)
        if isinstance(p_val, xr.DataArray):
            if "channel" not in p_val.coords:
                raise ValueError(f"'channel' has to be one of the coordinates of {p_name}")
            else:
                # TODO: check if just sorted() works
                if not (sorted(list(p_val.coords)) == channel):
                    raise ValueError(
                        f"The 'channel' coordinate of {p_name} has to match "
                        "that of the data to be calibrated"
                    )
            # TODO: Pre-sort the param xr.DataArray

    return out_dict


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
    out_dict["equivalent_beam_angle"] = user_cal_dict.get(
        "equivalent_beam_angle", echodata["Sonar/Beam_group1"]["equivalent_beam_angle"]
    )

    # Get params from the Vendor_specific group
    for p in ["EL", "DS", "TVR", "VTX", "Sv_offset"]:
        # substitute if None in user input
        out_dict[p] = user_cal_dict.get(p, echodata["Vendor_specific"][p])

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
        "angle_offset_alongship",
        "angle_offset_athwartship",
        "beamwidth_twoway_alongship",
        "beamwidth_twoway_athwartship",
        "equivalent_beam_angle",
    ]
    for p in params_from_beam:
        # substitute if p not in user input
        out_dict[p] = user_cal_dict.get(p, beam[p])

    # Params from the Vendor_specific group
    params_from_vend = ["sa_correction", "gain_correction"]
    for p in params_from_vend:
        # substitute if p not in user input
        out_dict[p] = user_cal_dict.get(p, get_vend_cal_params_power(beam=beam, vend=vend, param=p))

    # Params with default values
    # - transceiver impedance
    # - transducer impedance
    # - receiver sampling frequency

    # Interpolate to center frequency if doing broadband calibration
    # - requires an extra input argument
    #   - EK80: freq_center (center frequency for BB mode or nominal frequency for CW mode)
    #   - EK60: frequency_nominal
    # - variables to include: gain, angle offset, beamwidth, transducer impedance

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
    cal_params_CW: Dict[str, xr.DataArray],
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
        if "cal_channel_id" in vend.coords and ch_id.data in vend["cal_channel_id"]:
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
