import warnings

import xarray as xr

from ..echodata import EchoData
from .calibrate_azfp import CalibrateAZFP
from .calibrate_ek import CalibrateEK60, CalibrateEK80

CALIBRATOR = {"EK60": CalibrateEK60, "EK80": CalibrateEK80, "AZFP": CalibrateAZFP}


def _compute_cal(
    cal_type,
    echodata: EchoData,
    env_params=None,
    cal_params=None,
    waveform_mode=None,
    encode_mode=None,
):
    # Check on waveform_mode and encode_mode inputs
    if echodata.sonar_model == "EK80":
        if waveform_mode is None or encode_mode is None:
            raise ValueError(
                "waveform_mode and encode_mode must be specified for EK80 calibration"
            )
        elif waveform_mode not in ("BB", "CW"):
            raise ValueError("Input waveform_mode not recognized!")
        elif encode_mode not in ("complex", "power"):
            raise ValueError("Input encode_mode not recognized!")
        elif waveform_mode == "BB" and encode_mode == "power":
            raise ValueError(
                "Data from broadband ('BB') transmission must be recorded as complex samples"
            )
    elif echodata.sonar_model in ("EK60", "AZFP"):
        if waveform_mode is not None and waveform_mode != "CW":
            warnings.warn(
                "This sonar model transmits only narrowband signals (waveform_mode='CW'). "
                "Calibration will be in CW mode",
            )
        if encode_mode is not None and encode_mode != "power":
            warnings.warn(
                "This sonar model only record data as power or power/angle samples "
                "(encode_mode='power'). Calibration will be done on the power samples.",
            )

    # Set up calibration object
    cal_obj = CALIBRATOR[echodata.sonar_model](
        echodata,
        env_params=env_params,
        cal_params=cal_params,
        waveform_mode=waveform_mode,
        encode_mode=encode_mode,
    )
    # Perform calibration
    if cal_type == "Sv":
        return cal_obj.compute_Sv(waveform_mode=waveform_mode, encode_mode=encode_mode)
    else:
        return cal_obj.compute_Sp(waveform_mode=waveform_mode, encode_mode=encode_mode)


def compute_Sv(echodata: EchoData, **kwargs) -> xr.Dataset:
    """
    Compute volume backscattering strength (Sv) from raw data.

    The calibration routine varies depending on the sonar type.
    Currently this operation is supported for the following ``sonar_model``:
    EK60, AZFP, EK80 (see Notes below for detail).

    Parameters
    ----------
    echodata : EchoData
        An `EchoData` object created by using `open_raw` or `open_converted`

    env_params : dict, optional
        Environmental parameters needed for calibration.
        Users can supply `"sound speed"` and `"absorption"` directly,
        or specify other variables that can be used to compute them,
        including `"temperature"`, `"salinity"`, and `"pressure"`.

        For EK60 and EK80 echosounders, by default echopype uses
        environmental variables stored in the data files.
        For AZFP echosounder, all environmental parameters need to be supplied.
        AZFP echosounders typically are equipped with an internal temperature
        sensor, and some are equipped with a pressure sensor, but automatically
        using these pressure data is not currently supported.

    cal_params : dict, optional
        Intrument-dependent calibration parameters.

        For EK60, EK80, and AZFP echosounders, by default echopype uses
        environmental variables stored in the data files.
        Users can optionally pass in custom values shown below.

        - for EK60 echosounder, allowed parameters include:
          `"sa_correction"`, `"gain_correction"`, `"equivalent_beam_angle"`
        - for AZFP echosounder, allowed parameters include:
          `"EL"`, `"DS"`, `"TVR"`, `"VTX"`, `"equivalent_beam_angle"`, `"Sv_offset"`

        Passing in calibration parameters for other echosounders
        are not currently supported.

    waveform_mode : {"CW", "BB"}, optional
        Type of transmit waveform.
        Required only for data from the EK80 echosounder
        and not used with any other echosounder.

        - `"CW"` for narrowband transmission,
          returned echoes recorded either as complex or power/angle samples
        - `"BB"` for broadband transmission,
          returned echoes recorded as complex samples

    encode_mode : {"complex", "power"}, optional
        Type of encoded return echo data.
        Required only for data from the EK80 echosounder
        and not used with any other echosounder.

        - `"complex"` for complex samples
        - `"power"` for power/angle samples, only allowed when
          the echosounder is configured for narrowband transmission

    Returns
    -------
    xr.Dataset
        The calibrated Sv dataset, including calibration parameters
        and environmental variables used in the calibration operations.

    Notes
    -----
    The EK80 echosounder can be configured to transmit
    either broadband (``waveform_mode="BB"``)
    or narrowband (``waveform_mode="CW"``) signals.
    When transmitting in broadband mode, the returned echoes are
    encoded as complex samples (``encode_mode="complex"``).
    When transmitting in narrowband mode, the returned echoes can be encoded
    either as complex samples (``encode_mode="complex"``)
    or as power/angle combinations (``encode_mode="power"``) in a format
    similar to those recorded by EK60 echosounders.

    The current calibration implemented for EK80 broadband complex data
    uses band-integrated Sv with the gain computed at the center frequency
    of the transmit signal.
    """
    return _compute_cal(cal_type="Sv", echodata=echodata, **kwargs)


def compute_Sp(echodata: EchoData, **kwargs):
    """
    Compute point backscattering strength (Sp) from raw data.

    The calibration routine varies depending on the sonar type.
    Currently this operation is supported for the following ``sonar_model``:
    EK60, AZFP, EK80 (see Notes below for detail).

    Parameters
    ----------
    echodata : EchoData
        An `EchoData` object created by using `open_raw` or `open_converted`

    env_params : dict, optional
        Environmental parameters needed for calibration.
        Users can supply `"sound speed"` and `"absorption"` directly,
        or specify other variables that can be used to compute them,
        including `"temperature"`, `"salinity"`, and `"pressure"`.

        For EK60 and EK80 echosounders, by default echopype uses
        environmental variables stored in the data files.
        For AZFP echosounder, all environmental parameters need to be supplied.
        AZFP echosounders typically are equipped with an internal temperature
        sensor, and some are equipped with a pressure sensor, but automatically
        using these pressure data is not currently supported.

    cal_params : dict, optional
        Intrument-dependent calibration parameters.

        For EK60, EK80, and AZFP echosounders, by default echopype uses
        environmental variables stored in the data files.
        Users can optionally pass in custom values shown below.

        - for EK60 echosounder, allowed parameters include:
          `"sa_correction"`, `"gain_correction"`, `"equivalent_beam_angle"`
        - for AZFP echosounder, allowed parameters include:
          `"EL"`, `"DS"`, `"TVR"`, `"VTX"`, `"equivalent_beam_angle"`, `"Sv_offset"`

        Passing in calibration parameters for other echosounders
        are not currently supported.

    waveform_mode : {"CW", "BB"}, optional
        Type of transmit waveform.
        Required only for data from the EK80 echosounder
        and not used with any other echosounder.

        - `"CW"` for narrowband transmission,
          returned echoes recorded either as complex or power/angle samples
        - `"BB"` for broadband transmission,
          returned echoes recorded as complex samples

    encode_mode : {"complex", "power"}, optional
        Type of encoded return echo data.
        Required only for data from the EK80 echosounder
        and not used with any other echosounder.

        - `"complex"` for complex samples
        - `"power"` for power/angle samples, only allowed when
          the echosounder is configured for narrowband transmission

    Returns
    -------
    xr.Dataset
        The calibrated Sp dataset, including calibration parameters
        and environmental variables used in the calibration operations.

    Notes
    -----
    The EK80 echosounder can be configured to transmit
    either broadband (``waveform_mode="BB"``)
    or narrowband (``waveform_mode="CW"``) signals.
    When transmitting in broadband mode, the returned echoes are
    encoded as complex samples (``encode_mode="complex"``).
    When transmitting in narrowband mode, the returned echoes can be encoded
    either as complex samples (``encode_mode="complex"``)
    or as power/angle combinations (``encode_mode="power"``) in a format
    similar to those recorded by EK60 echosounders.

    The current calibration implemented for EK80 broadband complex data
    uses band-integrated Sp with the gain computed at the center frequency
    of the transmit signal.
    """
    return _compute_cal(cal_type="Sp", echodata=echodata, **kwargs)
