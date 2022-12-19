from typing import List

import numpy as np
import xarray as xr

from ..echodata import EchoData

CAL_PARAMS = {
    "EK": ("sa_correction", "gain_correction", "equivalent_beam_angle"),
    "AZFP": ("EL", "DS", "TVR", "VTX", "equivalent_beam_angle", "Sv_offset"),
}


def _check_param_chan_dep(p_val: List, da_chan: xr.DataArray):
    """
    Check type and dimension of calibration parameters that should be
    channel-dependent (i.e. frequency-dependent unless with duplicated frequency).

    Parameters
    ----------
    p_val
        Calibration parameter to check.
        If ``p`` is a list, the length of the list should be identical
        to the dimension length of ``chan``.
    da_chan
        An xr.DataArray containing channel or frequency information

    Return
    ------
    A xr.DataArray that contains the parameter with coordinate ``chan``
    """


#     # TODO: allow p to be xr.DataArray
#     #   in this case the coorindate should be ``chan`` or ``frequency`` or ``frequency_nominal``
#     #   and be identical to corresponding coordinate values in ``chan``

#     if isinstance(p, list):
#         if len(list) != len(da_chan):
#             raise ValueError("The number of elements in p should be identical with in da_chan")


def _get_param_from_user_or_echodata(p_name: str, user_cal_dict: dict, ed: EchoData):
    """
    Triangle getting parameter from either user input or an EchoData object.

    Parameter
    ---------
    p_name : str
        Name of the parameter to retrieve
    user_cal_dict : dict
        A dict of calibration parameter provided by user
    ed : an EchoData object
        An EchoData object containing the calibration parameter.
        In the current implement the parameter will be in either
        the ``Sonar/Beam_groupX`` group or the ``Vendor_specific`` group.
    """
    # # Check parameter form in user_cal_dict if exists
    # if p_name in user_cal_dict:
    #     _check_param_chan_dep(p_val=user_cal_dict[p_name], ed["Sonar/Beam_group"])

    # return user_cal_dict[p_name] if p_name in user_cal_dict else ed_group[p_name]


def get_cal_params_AZFP(echodata: EchoData, user_cal_dict: dict) -> dict:
    """
    Get cal params using user inputs or values from data file.

    Parameters
    ----------
    user_cal_dict : dict

    Returns
    -------
    A dict containing the calibration parameters
    """
    # TODO: For all the params below, check user input using _check_param_freq_dep()
    #  before storing in cal_params
    #  Such check is not needed if grabbning from EchoData since it's already a xr.DataArray
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
    echodata: EchoData, user_cal_dict: dict, waveform_mode: str, encode_mode: str
) -> dict:
    """
    Get cal params using user inputs or values from data file.

    Parameters
    ----------
    user_cal_dict : dict
    """
    out_dict = dict.fromkeys(CAL_PARAMS["EK"])

    if (
        encode_mode == "power"
        and waveform_mode == "CW"
        and echodata["Sonar/Beam_group2"] is not None
    ):
        beam = echodata["Sonar/Beam_group2"]
    else:
        beam = echodata["Sonar/Beam_group1"]

    # Params from the Vendor_specific group

    # only execute this if cw and power
    if waveform_mode == "CW" and beam is not None:
        params_from_vend = ["sa_correction", "gain_correction"]
        for p in params_from_vend:
            # substitute if None in user input
            out_dict[p] = (
                user_cal_dict[p]
                if p in user_cal_dict
                else get_vend_cal_params_power(
                    echodata=echodata, param=p, waveform_mode=waveform_mode
                )
            )

    # Other params
    out_dict["equivalent_beam_angle"] = (
        user_cal_dict["equivalent_beam_angle"]
        if "equivalent_beam_angle" in user_cal_dict
        else beam["equivalent_beam_angle"]
    )

    return out_dict


def get_vend_cal_params_complex_EK80(
    echodata: EchoData, channel_id: str, filter_name: str, param_type: str
):
    """
    Get filter coefficients stored in the Vendor_specific group attributes.

    Parameters
    ----------
    channel_id : str
        channel id for which the param to be retrieved
    filter_name : str
        name of filter coefficients to retrieve
    param_type : str
        'coeff' or 'decimation'
    """
    if param_type == "coeff":
        v = echodata["Vendor_specific"].attrs[
            "%s %s filter_r" % (channel_id, filter_name)
        ] + 1j * np.array(
            echodata["Vendor_specific"].attrs["%s %s filter_i" % (channel_id, filter_name)]
        )
        if v.size == 1:
            v = np.expand_dims(v, axis=0)  # expand dims for convolution
        return v
    else:
        return echodata["Vendor_specific"].attrs["%s %s decimation" % (channel_id, filter_name)]


def get_vend_cal_params_power(echodata: EchoData, param: str, waveform_mode: str):
    """
    Get cal parameters stored in the Vendor_specific group.

    Parameters
    ----------
    param : str {"sa_correction", "gain_correction"}
        name of parameter to retrieve
    """
    ds_vend = echodata["Vendor_specific"]

    if ds_vend is None or param not in ds_vend:
        return None

    if param not in ["sa_correction", "gain_correction"]:
        raise ValueError(f"Unknown parameter {param}")

    if waveform_mode == "CW" and echodata["Sonar/Beam_group2"] is not None:
        beam = echodata["Sonar/Beam_group2"]
    else:
        beam = echodata["Sonar/Beam_group1"]

    # indexes of frequencies that are for power, not complex
    relevant_indexes = np.where(np.isin(ds_vend["frequency_nominal"], beam["frequency_nominal"]))[0]

    # Find idx to select the corresponding param value
    # by matching beam["transmit_duration_nominal"] with ds_vend["pulse_length"]
    transmit_isnull = beam["transmit_duration_nominal"].isnull()
    idxmin = np.abs(
        beam["transmit_duration_nominal"] - ds_vend["pulse_length"][relevant_indexes]
    ).idxmin(dim="pulse_length_bin")

    # fill nan position with 0 (witll remove before return)
    # and convert to int for indexing
    idxmin = idxmin.where(~transmit_isnull, 0).astype(int)

    # Get param dataarray into correct shape
    da_param = (
        ds_vend[param][relevant_indexes]
        .expand_dims(dim={"ping_time": idxmin["ping_time"]})  # expand dims for direct indexing
        .sortby(idxmin.channel)  # sortby in case channel sequence different in vendor and beam
    )

    # Select corresponding index and clean up the original nan elements
    da_param = da_param.sel(pulse_length_bin=idxmin, drop=True)
    return da_param.where(~transmit_isnull, np.nan)  # set the nan elements back to nan


def get_gain_for_complex(
    echodata: EchoData, waveform_mode: str, chan_sel: xr.DataArray
) -> xr.DataArray:
    """
    Get gain factor for calibrating complex samples.

    Use values from ``gain_correction`` in the Vendor_specific group for CW mode samples,
    or interpolate ``gain`` to the center frequency of each ping for BB mode samples
    if nominal frequency is within the calibrated frequencies range

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
    An xr.DataArray
    """
    if waveform_mode == "BB":
        gain_single = get_vend_cal_params_power(
            echodata=echodata, param="gain_correction", waveform_mode=waveform_mode
        )
        gain = []
        if "gain" in echodata["Vendor_specific"].data_vars:
            # index using channel_id as order of frequency across channel can be arbitrary
            # reference to freq_center in case some channels are CW complex samples
            # (already dropped when computing freq_center in the calling function)
            for ch_id in chan_sel:
                # if channel gain exists in data
                if ch_id in echodata["Vendor_specific"].cal_channel_id:
                    gain_vec = echodata["Vendor_specific"].gain.sel(cal_channel_id=ch_id)
                    gain_temp = (
                        gain_vec.interp(
                            cal_frequency=echodata["Vendor_specific"].frequency_nominal.sel(
                                channel=ch_id
                            )
                        ).drop(["cal_channel_id", "cal_frequency"])
                    ).expand_dims("channel")
                # if no freq-dependent gain use CW gain
                else:
                    gain_temp = (
                        gain_single.sel(channel=ch_id)
                        .reindex_like(echodata["Sonar/Beam_group1"].backscatter_r, method="nearest")
                        .expand_dims("channel")
                    )
                gain_temp.name = "gain"
                gain.append(gain_temp)
            gain = xr.merge(gain).gain  # select the single data variable
        else:
            gain = gain_single
    elif waveform_mode == "CW":
        gain = get_vend_cal_params_power(
            echodata=echodata, param="gain_correction", waveform_mode=waveform_mode
        ).sel(channel=chan_sel)

    return gain
