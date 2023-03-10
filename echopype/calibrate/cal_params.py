from typing import Dict, List, Union

import numpy as np
import xarray as xr

CAL_PARAMS = {
    "EK60": (
        "sa_correction",
        "gain_correction",
        "equivalent_beam_angle",
        "angle_offset_alongship",
        "angle_offset_athwartship",
        "angle_sensitivity_alongship",
        "angle_sensitivity_athwartship",
        "beamwidth_alongship",
        "beamwidth_athwartship",
    ),
    "EK80": (
        "sa_correction",
        "gain_correction",
        "equivalent_beam_angle",
        "angle_offset_alongship",
        "angle_offset_athwartship",
        "angle_sensitivity_alongship",
        "angle_sensitivity_athwartship",
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


def sanitize_user_cal_dict(
    sonar_type: str,
    user_dict: Dict[str, Union[int, float, xr.DataArray]],
    channel: Union[List, xr.DataArray],
) -> Dict[str, Union[int, float, xr.DataArray]]:
    """
    Create a blue print for cal_params dict and check the format/organize user-provided params.

    Parameters
    ----------
    sonar_type : str
        type of sonar, one of "EK60", "EK80", or "AZFP"
    user_dict : dict
        a dict containing user input calibration parameters as {parameter name: parameter value}
        parameter value has to be a scalar (int or float) or an xr.DataArray
        if parameter value is an xr.DataArray, it has to either have 'channel' as a coorindate
        or have both 'cal_channel_id' and 'cal_frequency' as coordinates

    channel : list or xr.DataArray
        a list of channels to be calibrated
        for EK80 data, this list has to corresponds with the subset of channels
        selected based on waveform_mode and encode_mode
    """
    # Check sonar type
    if sonar_type not in ["EK60", "EK80", "AZFP"]:
        raise ValueError("'sonar_type' has to be one of: 'EK60', 'EK80', or 'AZFP'")

    # Make channel a sorted list
    if not isinstance(channel, (list, xr.DataArray)):
        raise ValueError("'channel' has to be a list or an xr.DataArray")
    else:
        if isinstance(channel, xr.DataArray):
            channel_sorted = sorted(channel.data)
        else:
            channel_sorted = sorted(channel)

    # Screen parameters: only retain those defined in CAL_PARAMS
    #  -- transform params in scalar or list to xr.DataArray
    #  -- directly pass through those that are xr.DataArray
    out_dict = dict.fromkeys(CAL_PARAMS[sonar_type])
    for p_name, p_val in user_dict.items():
        if p_name in out_dict:
            # if p_val an xr.DataArray, check existence and coordinates
            if isinstance(p_val, xr.DataArray):
                # if 'channel' is a coordinate, it has to match that of the data
                if "channel" in p_val.coords:
                    if not (sorted(p_val.coords["channel"].data) == channel_sorted):
                        raise ValueError(
                            f"The 'channel' coordinate of {p_name} has to match "
                            "that of the data to be calibrated"
                        )
                elif "cal_channel_id" in p_val.coords and "cal_frequency" in p_val.coords:
                    if not (sorted(p_val.coords["cal_channel_id"].data) == channel_sorted):
                        raise ValueError(
                            f"The 'cal_channel_id' coordinate of {p_name} has to match "
                            "that of the data to be calibrated"
                        )
                else:
                    raise ValueError(
                        f"{p_name} has to either have 'channel' as a coorindate "
                        "or have both 'cal_channel_id' and 'cal_frequency' as coordinates"
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


def _get_interp_da(
    da_param: Union[None, xr.DataArray],
    freq_center: xr.DataArray,
    alternative: Union[int, float, xr.DataArray],
    BB_factor: float = 1,
) -> xr.DataArray:
    """
    Get interpolated xr.DataArray aligned with the channel coordinate.

    Interpolation at freq_center when da_param contains frequency-dependent xr.DataArray.
    When da_param is None or does not contain frequency-dependent xr.DataArray,
    the alternative (a const or an xr.DataArray with coordinate channel) is used.

    Parameters
    ----------
    da_param : xr.DataArray or None
        a data array from the Vendor group or user dict with freq-dependent param values
    freq_center : xr.DataArray
        center frequency (BB) or nominal frequency (CW)
    alternative : xr.DataArray or int or float
        alternative for when freq-dep values do not exist
    BB_factor : float
        scaling factor due to BB transmit signal with different center frequency
        with respect to nominal channel frequency;
        only applies when ``alternative`` from the Sonar/Beam_groupX group is used
        for params ``angle_sensitivity_alongship/athwartship`` and
        ``beamwidth_alongship/athwartship`` (``see get_cal_params_EK`` for detail)

    Returns
    -------
    an xr.DataArray aligned with the channel coordinate.

    Note
    ----
    ``da_param`` is always an xr.DataArray from the Vendor-specific group.
    It is possible that only a subset of the channels have frequency-dependent parameter values.
    The output xr.DataArray here is constructed channel-by-channel to allow for this flexibility.

    ``alternative`` can be one of the following:

    - scalar (int or float): this is the case for impedance_transmit
    - xr.DataArray with coordinates channel, ping_time, and beam:
        this is the case for parameters angle_offset_alongship, angle_offset_athwartship,
                                        beamwidth_alongship, beamwidth_athwartship
    - xr.DataArray with coordinates channel, ping_time:
        this is the case for sa_correction and gain_correction,
        which will be direct output of get_vend_cal_params_power()
    """
    param = []
    for ch_id in freq_center["channel"].data:
        # if frequency-dependent param exists as a data array with desired channel
        if (
            da_param is not None
            and "cal_channel_id" in da_param.coords
            and ch_id in da_param["cal_channel_id"]
        ):
            # interp variable has ping_time dimension from freq_center
            param.append(
                da_param.sel(cal_channel_id=ch_id)
                .interp(cal_frequency=freq_center.sel(channel=ch_id))
                .data.squeeze()
            )
        # if no frequency-dependent param exists, use alternative
        else:
            BB_factor_ch = (
                BB_factor.sel(channel=ch_id) if isinstance(BB_factor, xr.DataArray) else BB_factor
            )
            if isinstance(alternative, xr.DataArray):
                # drop the redundant beam dimension if exist
                if "beam" in alternative.coords:
                    param.append(
                        (alternative.sel(channel=ch_id).isel(beam=0) * BB_factor_ch).data.squeeze()
                    )
                else:
                    param.append((alternative.sel(channel=ch_id) * BB_factor_ch).data.squeeze())
            elif isinstance(alternative, (int, float)):
                # expand to have ping_time dimension
                param.append(
                    np.array([alternative] * freq_center.sel(channel=ch_id).size).squeeze()
                    * BB_factor_ch
                )
            else:
                raise ValueError("'alternative' has to be of the type int, float, or xr.DataArray")

    param = np.array(param)

    if "ping_time" in freq_center.coords:
        if len(param.shape) == 1:  # this means ping_time has length=1
            param = np.expand_dims(param, axis=1)
        return xr.DataArray(
            param,
            dims=["channel", "ping_time"],
            coords={"channel": freq_center["channel"], "ping_time": freq_center["ping_time"]},
        )
    else:
        return xr.DataArray(param, dims=["channel"], coords={"channel": freq_center["channel"]})


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


def get_cal_params_AZFP(beam: xr.DataArray, vend: xr.DataArray, user_dict: dict) -> dict:
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
    # Use sanitized user dict as blueprint
    # out_dict contains only and all of the allowable cal params
    out_dict = sanitize_user_cal_dict(
        user_dict=user_dict, channel=beam["channel"], sonar_type="AZFP"
    )

    # Only fill in params that are None
    for p, v in out_dict.items():
        if v is None:
            # Params from Sonar/Beam_group1
            if p == "equivalent_beam_angle":
                # equivalent_beam_angle has dims: channel, ping_time, beam --> only need channel
                out_dict[p] = beam[p].isel(ping_time=0, beam=0).drop(["ping_time", "beam"])

            # Params from Vendor_specific group
            elif p in ["EL", "DS", "TVR", "VTX", "Sv_offset"]:
                out_dict[p] = vend[p]  # these params only have the channel dimension

    return out_dict


def get_cal_params_EK(
    waveform_mode: str,
    freq_center: xr.DataArray,
    beam: xr.Dataset,
    vend: xr.Dataset,
    user_dict: Dict[str, Union[int, float, xr.DataArray]],
    default_dict: Dict[str, Union[int, float]] = EK80_DEFAULT_PARAMS,
    sonar_type: str = "EK80",
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
    sonar_type : str
        type of EK sonar, either "EK60" or "EK80"
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
                fs.append(default_dict["receiver_sampling_frequency"][tcvr_type])
            return xr.DataArray(fs, dims=["channel"], coords={"channel": vend["channel"]})

    # Mapping between desired param name with Beam group data variable name
    PARAM_BEAM_NAME_MAP = {
        "angle_offset_alongship": "angle_offset_alongship",
        "angle_offset_athwartship": "angle_offset_athwartship",
        "angle_sensitivity_alongship": "angle_sensitivity_alongship",
        "angle_sensitivity_athwartship": "angle_sensitivity_athwartship",
        "beamwidth_alongship": "beamwidth_twoway_alongship",
        "beamwidth_athwartship": "beamwidth_twoway_athwartship",
        "equivalent_beam_angle": "equivalent_beam_angle",
    }
    if waveform_mode == "BB":
        # for BB data equivalent_beam_angle needs to be scaled wrt freq_center
        PARAM_BEAM_NAME_MAP.pop("equivalent_beam_angle")

    # Use sanitized user dict as blueprint
    # out_dict contains only and all of the allowable cal params
    out_dict = sanitize_user_cal_dict(
        user_dict=user_dict, channel=beam["channel"], sonar_type=sonar_type
    )

    # Interpolate user-input params that contain freq-dependent info
    # ie those that has coordinate combination (cal_channel_id, cal_frequency)
    # TODO: this will need to change for computing frequency-dependent TS
    for p, v in out_dict.items():
        if v is not None:
            if "cal_channel_id" in v.coords:
                out_dict[p] = _get_interp_da(v, freq_center, np.nan)

    # Only fill in params that are None
    for p, v in out_dict.items():
        if v is None:
            # Those without CW or BB complications
            if p == "sa_correction":  # pull from data file
                out_dict[p] = get_vend_cal_params_power(beam=beam, vend=vend, param=p)
            elif p == "impedance_receive":  # from data file or default dict
                out_dict[p] = default_dict[p] if p not in vend else vend["impedance_receive"]
            elif p == "receiver_sampling_frequency":  # from data file or default_dict
                out_dict[p] = _get_fs()
            else:
                # CW: params do not require interpolation, except for impedance_transmit
                if waveform_mode == "CW":
                    if p in PARAM_BEAM_NAME_MAP.keys():
                        p_beam = PARAM_BEAM_NAME_MAP[p]
                        # pull from data file, these should always exist
                        if "beam" in beam[p_beam].coords:
                            out_dict[p] = beam[p_beam].isel(beam=0).drop("beam")
                        else:
                            out_dict[p] = beam[p_beam]
                    elif p == "gain_correction":
                        # pull from data file narrowband table
                        out_dict[p] = get_vend_cal_params_power(beam=beam, vend=vend, param=p)
                    elif p == "impedance_transmit":
                        # assemble each channel from data file or default dict
                        out_dict[p] = _get_interp_da(
                            da_param=None if p not in vend else vend[p],
                            freq_center=freq_center,
                            alternative=default_dict[p],  # pull from default dict
                        )
                    else:
                        raise ValueError(f"{p} not in the defined set of calibration parameters.")

                # BB mode: params require interpolation
                else:
                    # interpolate for center frequency or use CW values
                    if p in PARAM_BEAM_NAME_MAP.keys():
                        # only scale these params if alternative is used
                        if p in [
                            "angle_sensitivity_alongship",
                            "angle_sensitivity_athwartship",
                            "beamwidth_alongship",
                            "beamwidth_athwartship",
                        ]:
                            BB_factor = freq_center / beam["frequency_nominal"]
                        else:
                            BB_factor = 1

                        p_beam = PARAM_BEAM_NAME_MAP[p]
                        out_dict[p] = _get_interp_da(
                            da_param=None if p not in vend else vend[p],
                            freq_center=freq_center,
                            alternative=beam[p_beam],  # these should always exist
                            BB_factor=BB_factor,
                        )
                    elif p == "equivalent_beam_angle":
                        # scaled according to frequency ratio
                        out_dict[p] = beam[p].isel(beam=0).drop("beam") + 20 * np.log10(
                            beam["frequency_nominal"] / freq_center
                        )
                    elif p == "gain_correction":
                        # interpolate or pull from narrowband table
                        out_dict[p] = _get_interp_da(
                            da_param=None
                            if "gain" not in vend
                            else vend["gain"],  # freq-dep values
                            freq_center=freq_center,
                            alternative=get_vend_cal_params_power(beam=beam, vend=vend, param=p),
                        )
                    elif p == "impedance_transmit":
                        out_dict[p] = _get_interp_da(
                            da_param=None if p not in vend else vend[p],
                            freq_center=freq_center,
                            alternative=default_dict[p],  # pull from default dict
                        )
                    else:
                        raise ValueError(f"{p} not in the defined set of calibration parameters.")

    return out_dict
