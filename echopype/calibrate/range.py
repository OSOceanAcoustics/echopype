import datetime
from typing import Dict, Optional, Union

import numpy as np
import xarray as xr

from ..echodata import EchoData
from ..echodata.simrad import retrieve_correct_beam_group


def _harmonize_env_param_time(
    p: Union[int, float, xr.DataArray],
    ping_time: Optional[Union[xr.DataArray, datetime.datetime]] = None,
):
    """
    Harmonize time coordinate between Beam_groupX data and env_params to make sure
    the timestamps are broadcast correctly in calibration and range calculations.

    Regardless of the source, if `p` is an xr.DataArray, the time coordinate name
    needs to be `time1` to be consistent with the time coordinate in EchoData["Environment"].
    If `time1` is of length=1, the dimension `time1` is dropped.
    Otherwise, `p` is interpolated to `ping_time`.
    If `p` is not an xr.DataArray it is returned directly.

    Parameters
    ----------
    p
        The environment parameter for timestamp check/correction
    ping_time
        Beam_groupX ping_time to interpolate env_params timestamps to.
        Only used if p.time1 has length >1

    Returns
    -------
    Environment parameter with correctly broadcasted timestamps
    """
    if isinstance(p, xr.DataArray):
        if "time1" not in p.coords:
            return p
        else:
            # If there's only 1 time1 value,
            # or if after dropping NaN there's only 1 time1 value
            if p["time1"].size == 1 or p.dropna(dim="time1").size == 1:
                return p.dropna(dim="time1").squeeze(dim="time1").drop("time1")

            # Direct assignment if all timestamps are identical (EK60 data)
            elif np.all(p["time1"].values == ping_time.values):
                return p.rename({"time1": "ping_time"})

            elif ping_time is None:
                raise ValueError(f"ping_time needs to be provided for interpolating {p.name}")

            else:
                return p.dropna(dim="time1").interp(time1=ping_time)
    else:
        return p


def compute_range_AZFP(echodata: EchoData, env_params: Dict, cal_type: str) -> xr.DataArray:
    """
    Computes the range (``echo_range``) of AZFP backscatter data in meters.

    Parameters
    ----------
    echodata : EchoData
        An EchoData object holding data from an AZFP echosounder
    env_params : dict
        A dictionary holding environmental parameters needed for computing range
        See echopype.calibrate.env_params.get_env_params_AZFP()
    cal_type : {"Sv", "TS"}

        - `"Sv"` for calculating volume backscattering strength
        - `"TS"` for calculating target strength.

        This parameter needs to be specified for data from the AZFP echosounder
        due to a difference in the range computation given by the manufacturer

    Returns
    -------
    xr.DataArray
        The range (``echo_range``) of the data in meters.

    Notes
    -----
    For AZFP echosounder, the returned ``echo_range`` is duplicated along ``ping_time``
    to conform with outputs from other echosounders, even though within each data
    file the range is held constant.
    """
    # sound_speed should exist already
    if "sound_speed" not in env_params:
        raise RuntimeError(
            "sounds_speed not included in env_params, "
            "use echopype.calibrate.env_params.get_env_params_AZFP() to compute env_params "
            "by supplying temperature, salinity, and pressure."
        )
    else:
        sound_speed = env_params["sound_speed"]

    # Check cal_type
    if cal_type is None:
        raise ValueError('cal_type must be "Sv" or "TS"')

    # Groups to use
    vend = echodata["Vendor_specific"]
    beam = echodata["Sonar/Beam_group1"]

    # Notation below follows p.86 of user manual
    N = vend["number_of_samples_per_average_bin"]  # samples per bin
    f = vend["digitization_rate"]  # digitization rate
    L = vend["lockout_index"]  # number of lockout samples

    # keep this in ref of AZFP matlab code,
    # set to 1 since we want to calculate from raw data
    bins_to_avg = 1

    # Harmonize sound_speed time1 and Beam_group1 ping_time
    sound_speed = _harmonize_env_param_time(
        p=sound_speed,
        ping_time=beam.ping_time,
    )

    # Calculate range using parameters for each freq
    # This is "the range to the centre of the sampling volume
    # for bin m" from p.86 of user manual
    if cal_type == "Sv":
        range_offset = 0
    else:
        range_offset = sound_speed * beam["transmit_duration_nominal"] / 4  # from matlab code
    range_meter = (
        sound_speed * L / (2 * f)
        + (sound_speed / 4)
        * (
            ((2 * (beam["range_sample"] + 1) - 1) * N * bins_to_avg - 1) / f
            + beam["transmit_duration_nominal"]
        )
        - range_offset
    )

    range_meter.name = "echo_range"  # add name to facilitate xr.merge

    return range_meter


def compute_range_EK(
    echodata: EchoData, env_params: Dict, waveform_mode: str = "CW", encode_mode: str = "power"
):
    """
    Computes the range (``echo_range``) of EK backscatter data in meters.

    Parameters
    ----------
    echodata : EchoData
        An EchoData object holding data from an AZFP echosounder
    env_params : dict
        A dictionary holding environmental parameters needed for computing range
        See echopype.calibrate.env_params.get_env_params_EK60() or .get_env_params_EK80()
    waveform_mode : {"CW", "BB"}
        Type of transmit waveform.
        Required only for data from the EK80 echosounder.

        - `"CW"` for narrowband transmission,
            returned echoes recorded either as complex or power/angle samples
        - `"BB"` for broadband transmission,
            returned echoes recorded as complex samples

    encode_mode : {"complex", "power"}
        Type of encoded data format.
        Required only for data from the EK80 echosounder.

        - `"complex"` for complex samples
        - `"power"` for power/angle samples, only allowed when
          the echosounder is configured for narrowband transmission

    Returns
    -------
    xr.DataArray
        The range (``echo_range``) of the data in meters.

    Notes
    -----
    The EK80 echosounder can be configured to transmit
    either broadband (``waveform_mode="BB"``) or narrowband (``waveform_mode="CW"``) signals.
    When transmitting in broadband mode, the returned echoes must be
    encoded as complex samples (``encode_mode="complex"``).
    When transmitting in narrowband mode, the returned echoes can be encoded
    either as complex samples (``encode_mode="complex"``)
    or as power/angle combinations (``encode_mode="power"``) in a format
    similar to those recorded by EK60 echosounders (the "power/angle" format).
    """
    # sound_speed should exist already
    if echodata.sonar_model in ("EK60", "ES70"):
        ek_str = "EK60"
    elif echodata.sonar_model in ("EK80", "ES80", "EA640"):
        ek_str = "EK80"
    else:
        raise ValueError("The specified sonar_model is not supported!")

    if "sound_speed" not in env_params:
        raise RuntimeError(
            "sounds_speed not included in env_params, "
            f"use echopype.calibrate.env_params.get_env_params_{ek_str}() to compute env_params "
        )
    else:
        sound_speed = env_params["sound_speed"]

    # Get the right Sonar/Beam_groupX group according to encode_mode
    ed_group = retrieve_correct_beam_group(echodata, waveform_mode, encode_mode)
    beam = echodata[ed_group]
    vend = echodata["Vendor_specific"]

    # Harmonize sound_speed time1 and Beam_groupX ping_time
    sound_speed = _harmonize_env_param_time(
        p=sound_speed,
        ping_time=beam.ping_time,
    )

    # TVG correction factor changes depending when the echo recording starts
    # wrt when the transmit signal is sent out.
    # This depends on whether it is Ex60 or Ex80 style hardware
    # ref: https://github.com/CI-CMG/pyEcholab/blob/RHT-EK80-Svf/echolab2/instruments/EK80.py#L4297-L4308  # noqa
    def range_Ex60(ds):
        return (
            # 2-sample shift in the beginning
            (ds["range_sample"] - 2)
            * ds["sample_interval"]
            * sound_speed
            / 2
        )  # [frequency x range_sample]

    def range_Ex80(ds):
        return (
            ds["range_sample"] * ds["sample_interval"] * sound_speed / 2
            - sound_speed * ds["transmit_duration_nominal"] / 4
        )

    # If EK60
    if echodata.sonar_model in ["EK60", "ES70"]:
        range_meter = range_Ex60(beam)

    # If EK80:
    # - compute range first assuming all channels have Ex80 style hardware
    # - change range for channels with Ex60 style hardware (GPT)
    elif echodata.sonar_model in ["EK80", "ES80", "EA640"]:
        range_meter = range_Ex80(beam)

        # Change range for all channels with GPT
        if "GPT" in vend["transceiver_type"]:
            ch_GPT = vend["transceiver_type"] == "GPT"
            range_meter.loc[dict(channel=ch_GPT)] = range_Ex60(
                beam.sel(channel=vend["channel"][ch_GPT])
            )

    # make order of dims conform with the order of backscatter data
    range_meter = range_meter.transpose("channel", "ping_time", "range_sample")
    # range_meter = range_meter.where(range_meter > 0, 0)  # set negative ranges to 0

    # set entries with NaN backscatter data to NaN
    if "beam" in beam["backscatter_r"].dims:
        # Drop beam because echo_range does not have beam dimension
        valid_idx = ~beam["backscatter_r"].isel(beam=0).drop("beam").isnull()
    else:
        valid_idx = ~beam["backscatter_r"].isnull()
    range_meter = range_meter.where(valid_idx)

    # remove time1 if exists as a coordinate
    if "time1" in range_meter.coords:
        range_meter = range_meter.drop("time1")

    # add name to facilitate xr.merge
    range_meter.name = "echo_range"

    return range_meter
