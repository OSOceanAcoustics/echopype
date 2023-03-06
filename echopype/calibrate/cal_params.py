from typing import Dict, List, Union

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
        "impedance_transmit",  # z_et
        "impedance_receive",  # z_er
        "receiver_sampling_frequency",
    ),
    "AZFP": ("EL", "DS", "TVR", "VTX", "equivalent_beam_angle", "Sv_offset"),
}

EK80_DEFAULT_PARAMS = {
    "impedance_transmit": 75,
    "impedance_receive": 1000,
    "receiver_sampling_frequency": {  # default full sampling frequency [Hz]
        "default": 1500000,
        "GPT": 500000,
        "SBT": 50000,
        "WBAT": 1500000,
        "WBT TUBE": 1500000,
        "WBT MINI": 1500000,
        "WBT": 1500000,
        "WBT HP": 187500,
        "WBT LF": 93750,
    },
}


# TODO: need a function (something like "_check_param_freq_dep")
# to check user input cal_params and env_params


def param2da(p_val: Union[int, float, list], channel: Union[list, xr.DataArray]) -> xr.DataArray:
    """
    Organize individual parameter in scalar or list to xr.DataArray with channel coordinate.

    Parameters
    ----------
    p_val : int, float, or list
        dictionary holding calibration params for one or more channels
        each param has to be a scalar
    channel : list or xr.DataArray
        values to use for the output channel coordinate

    Returns
    -------
    an xr.DataArray with channel coordinate
    """
    # TODO: allow passing in np.array as dict values to assemble a frequency-dependent cal da

    if not isinstance(p_val, (int, float, list)):
        raise ValueError("'p_val' needs to be one of type int, float, or list")
    else:
        if isinstance(p_val, list):
            # Check length if p_val a list
            if len(p_val) != len(channel):
                raise ValueError("The lengths of 'p_val' and 'channel' should be identical")

            return xr.DataArray(p_val, dims=["channel"], coords={"channel": channel})
        else:
            # if scalar, make a list to form data array
            return xr.DataArray(
                [p_val] * len(channel), dims=["channel"], coords={"channel": channel}
            )


# TODO: allow Dict[Union[float, str], np.array] as user_dict values
def sanitize_user_cal_dict(
    sonar_type: str,
    user_dict: Dict[str, Union[int, float, xr.DataArray]],
    channel: Union[List, xr.DataArray],
) -> Dict[str, Union[int, float, xr.DataArray]]:
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
    sonar_type : str
        type of sonar, either "EK" or "AZFP"
    """
    # Check sonar type
    if sonar_type not in ["EK", "AZFP"]:
        raise ValueError("'sonar_type' has to be either 'EK' or 'AZFP'")

    # Make channel a sorted list
    if not isinstance(channel, (list, xr.DataArray)):
        raise ValueError("'channel' has to be a list or an xr.DataArray")
    else:
        if isinstance(channel, xr.DataArray):
            channel = sorted(channel.data)
        else:
            channel = sorted(channel)

    # Screen parameters: only retain those defined in CAL_PARAMS
    #  -- transform params in scalar or list to xr.DataArray
    #  -- directly pass through those that are xr.DataArray
    out_dict = dict.fromkeys(CAL_PARAMS[sonar_type])
    for p_name, p_val in user_dict.items():
        if p_name in out_dict:
            # if p_val an xr.DataArray, check existence and correspondence of channel coordinate
            if isinstance(p_val, xr.DataArray):
                if "channel" not in p_val.coords:
                    raise ValueError(f"'channel' has to be one of the coordinates of {p_name}")
                else:
                    if not (sorted(p_val.coords["channel"].data) == channel):
                        raise ValueError(
                            f"The 'channel' coordinate of {p_name} has to match "
                            "that of the data to be calibrated"
                        )
                out_dict[p_name] = p_val

            # If p_val a scalar or list, make it xr.DataArray
            elif isinstance(p_val, (int, float, list)):
                # check for list dimension happens within param2da()
                out_dict[p_name] = param2da(p_val, channel)

            # p_val has to be one of int, float, xr.DataArray
            else:
                raise ValueError(f"{p_name} has to be a scalar, list, or an xr.DataArray")

            # TODO: Consider pre-sort the param xr.DataArray?

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


def _get_interp_da(
    da_param: Union[None, xr.DataArray],
    freq_center: xr.DataArray,
    alternative: Union[int, float, xr.DataArray],
) -> xr.DataArray:
    """
    Get interpolated xr.DataArray aligned with the channel coordinate.

    Interpolation at freq_center when da_param contains frequency-dependent xr.DataArray.
    When da_param is None or does not contain frequency-dependent xr.DataArray,
    the alternative (a const or an xr.DataArray with coordinate channel) is used.

    Parameters
    ----------
    da_param : xr.DataArray or None
        a data array from the Vendor group with frequency-dependent param values.
    freq_center : xr.DataArray
        center frequency (BB) or nominal frequency (CW)
    alternative : xr.DataArray or int or float
        alternative for when freq-dep values do not exist

    Returns
    -------
    an xr.DataArray aligned with the channel coordinate.

    Note
    ----
    Since ``da_param`` is an xr.DataArray from the Vendor-specific group with frequency-dependent
    param values, the case where ``da_param`` is an xr.DataArray without frequency-dependent
    param values do not exist. Check all use cases of this function to avoid confusion.
    """
    param = []
    for ch_id in freq_center["channel"].data:
        # if frequency-dependent param exists as a data array with desired channel
        if (
            da_param is not None
            and "cal_channel_id" in da_param.coords
            and ch_id in da_param["cal_channel_id"]
        ):
            param.append(
                da_param.sel(cal_channel_id=ch_id)
                .interp(cal_frequency=freq_center.sel(channel=ch_id).data)
                .data.squeeze()
            )
        # if no frequency-dependent param exists, use alternative
        else:
            if isinstance(alternative, xr.DataArray):
                param.append(alternative.sel(channel=ch_id).data.squeeze())
            else:  # int or float
                param.append(alternative)

    return xr.DataArray(param, dims=["channel"], coords={"channel": freq_center["channel"]})


def get_cal_params_EK_new(
    waveform_mode: str,
    freq_center: xr.DataArray,
    beam: xr.Dataset,
    vend: xr.Dataset,
    user_dict: Dict[str, Union[int, float, xr.DataArray]],
    default_dict: Dict[str, Union[int, float]] = EK80_DEFAULT_PARAMS,
) -> Dict:
    """
    Get cal parameters from user input, data file, or a set of default values.

    Parameters
    ----------
    waveform_mode : str
        transmit waveform mode, either "CW" or "BB"
    freq_center : xr.DataArray
        center frequency (BB mode) or nominal frequency (CW mode)
    beam : xr.Dataset
        a subset of Sonar/Beam_groupX that contains only the channels to be calibrated
    vend : xr.Dataset
        a subset of Vendor_specific that contains only the channels to be calibrated
    user_dict : dict
        a dictionary containing user-defined parameters.
        user-defined parameters take precedance over values in the data file or in default dict.
    default_dict : dict
        a dictionary containing default parameters
    """

    # Private function to get fs
    def _get_fs():
        if "fs_receiver" in vend:
            return vend["fs_receiver"]
        else:
            # loop through channel since transceiver can vary
            fs = []
            for ch in vend["channel"]:
                tcvr_type = vend["transceiver_type"].sel(channel=ch).data.tolist().upper()
                fs.append(default_dict["fs"][tcvr_type])
            return xr.DataArray(fs, dims=["channel"], coords={"channel": vend["channel"]})

    # Mapping between desired param name with Beam group data variable name
    PARAM_BEAM_NAME_MAP = {
        "angle_offset_alongship": "angle_offset_alongship",
        "angle_offset_athwartship": "angle_offset_athwartship",
        "beamwidth_alongship": "beamwidth_twoway_alongship",
        "beamwidth_athwartship": "beamwidth_twoway_athwartship",
        "equivalent_beam_angle": "equivalent_beam_angle",
    }
    if waveform_mode == "BB":
        PARAM_BEAM_NAME_MAP.pop("equivalent_beam_angle")

    # Use sanitized user dict as blueprint
    out_dict = sanitize_user_cal_dict(user_dict=user_dict, channel=beam["channel"], sonar_type="EK")

    # Only fill in params that are None
    for p, v in out_dict.items():
        if v is None:
            # Those without CW or BB complications
            if p == "sa_correction":  # pull from data file
                out_dict[p] = get_vend_cal_params_power(beam=beam, vend=vend, param=p)
            if p == "impedance_receive":  # from data file or default dict
                out_dict[p] = default_dict[p] if p not in vend else vend["impedance_receive"]
            if p == "receiver_sampling_frequency":  # from data file or default_dict
                out_dict[p] = _get_fs()

            # CW: params do not require except for impedance_transmit
            if waveform_mode == "CW":
                if p in PARAM_BEAM_NAME_MAP.keys():
                    for p, p_beam in PARAM_BEAM_NAME_MAP.items():
                        # pull from data file, these should always exist
                        out_dict[p] = beam[p_beam]
                if p == "gain_correction":
                    # pull from data file narrowband table
                    out_dict[p] = get_vend_cal_params_power(beam=beam, vend=vend, param=p)
                if p == "impedance_transmit":
                    # assemble each channel from data file or default dict
                    out_dict[p] = _get_interp_da(
                        da_param=None if p not in vend else vend[p],
                        freq_center=freq_center,
                        alternative=default_dict[p],  # pull from default dict
                    )

            # BB mode: params require interpolation
            else:
                # interpolate for center frequency or use CW values
                if p in PARAM_BEAM_NAME_MAP.keys():
                    for p, p_beam in PARAM_BEAM_NAME_MAP.items():
                        # TODO: beamwidth_along/athwartship should be scaled
                        #       like equivalent_beam_angle
                        out_dict[p] = _get_interp_da(
                            da_param=None if p not in vend else vend[p],
                            freq_center=freq_center,
                            alternative=beam[p_beam],  # these should always exist
                        )
                if p == "equivalent_beam_angle":
                    # scaled according to frequency ratio
                    out_dict[p] = beam[p] + 20 * np.log10(beam["frequency_nominal"] / freq_center)
                if p == "gain_correction":
                    # interpolate or pull from narrowband table
                    out_dict[p] = _get_interp_da(
                        da_param=None if "gain" not in vend else vend["gain"],  # freq-dep values
                        freq_center=freq_center,
                        alternative=get_vend_cal_params_power(beam=beam, vend=vend, param=p),
                    )
                if p == "impedance_transmit":
                    out_dict[p] = _get_interp_da(
                        da_param=None if p not in vend else vend[p],
                        freq_center=freq_center,
                        alternative=default_dict[p],  # pull from default dict
                    )

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
