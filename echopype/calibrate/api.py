import numpy as np
import xarray as xr

from ..echodata import EchoData
from ..echodata.simrad import check_input_args_combination, retrieve_correct_beam_group
from ..utils.log import _init_logger
from ..utils.prov import echopype_prov_attrs, source_files_vars
from .calibrate_azfp import CalibrateAZFP
from .calibrate_ek import CalibrateEK60, CalibrateEK80

CALIBRATOR = {
    "EK60": CalibrateEK60,
    "EK80": CalibrateEK80,
    "AZFP": CalibrateAZFP,
    "ES70": CalibrateEK60,
    "ES80": CalibrateEK80,
    "EA640": CalibrateEK80,
}

logger = _init_logger(__name__)


def _compute_cal(
    cal_type,
    echodata: EchoData,
    env_params=None,
    cal_params=None,
    ecs_file=None,
    waveform_mode=None,
    encode_mode=None,
    assume_single_filter_time=None,
    drop_last_hanning_zero=False,
):
    # Copy echodata object
    echodata_copy = echodata.copy()

    # Make waveform_mode "FM" equivalent to "BB"
    waveform_mode = "BB" if waveform_mode == "FM" else waveform_mode

    # Check on waveform_mode, encode_mode inputs, and assumption on single filter time
    if echodata_copy.sonar_model == "EK80":
        if waveform_mode is None or encode_mode is None:
            raise ValueError("waveform_mode and encode_mode must be specified for EK80 calibration")
        check_input_args_combination(waveform_mode=waveform_mode, encode_mode=encode_mode)
    elif echodata_copy.sonar_model in ("EK60", "AZFP"):
        if waveform_mode is not None and waveform_mode != "CW":
            logger.warning(
                "This sonar model transmits only narrowband signals (waveform_mode='CW'). "
                "Calibration will be in CW mode",
            )
        if encode_mode is not None and encode_mode != "power":
            logger.warning(
                "This sonar model only record data as power or power/angle samples "
                "(encode_mode='power'). Calibration will be done on the power samples.",
            )

    # Check that assume_single_filter_time is correctly passed in
    if (
        echodata_copy.sonar_model != "EK80" or encode_mode != "complex"
    ) and assume_single_filter_time is not None:
        raise ValueError("assume_single_filter_time can only be used on complex EK80 data.")

    # Compute calibration dataset
    def _compute_cal_ds(echodata_copy):
        # Set up calibration object
        cal_obj = CALIBRATOR[echodata_copy.sonar_model](
            echodata_copy,
            env_params=env_params,
            cal_params=cal_params,
            ecs_file=ecs_file,
            waveform_mode=waveform_mode,
            encode_mode=encode_mode,
            drop_last_hanning_zero=drop_last_hanning_zero,
        )

        # Check Echodata Backscatter Size
        cal_obj._check_echodata_backscatter_size()

        # Perform calibration
        if cal_type == "Sv":
            cal_ds = cal_obj.compute_Sv()
        else:
            cal_ds = cal_obj.compute_TS()

        return cal_ds

    # Grab the correct ed_beam_group given waveform and encode mode and subset for
    # CW or BB if encode mode is complex
    ed_beam_group = retrieve_correct_beam_group(
        echodata=echodata_copy, waveform_mode=waveform_mode, encode_mode=encode_mode
    )

    # Collapse vendor specific's filter time dimension
    if assume_single_filter_time and "filter_time" in echodata_copy["Vendor_specific"].dims:
        transmit_duration_nominal_ds = echodata_copy[ed_beam_group]["transmit_duration_nominal"]
        # Grab a valid filter time for each channel
        channel_filter_time = {}
        for channel in transmit_duration_nominal_ds.channel.values:
            valid_ping_times = (
                transmit_duration_nominal_ds.sel(channel=channel)
                .dropna(dim="ping_time")
                .ping_time.values
            )
            channel_filter_time[channel] = valid_ping_times[0]
        vendor_specific_ds = echodata_copy["Vendor_specific"]

        # Grab filter time and channel pair subsets of Vendor_specific
        vendor_specific_collapsed_ds_list = []
        for channel, filter_time in channel_filter_time.items():
            vendor_specific_collapsed_ds = vendor_specific_ds.sel(
                filter_time=filter_time
            ).drop_vars("filter_time")
            # Ensure that we do not remove channel as dimension
            if len(vendor_specific_collapsed_ds["channel"]) > 1:
                vendor_specific_collapsed_ds = vendor_specific_collapsed_ds.sel(channel=channel)
            vendor_specific_collapsed_ds_list.append(vendor_specific_collapsed_ds)

        # Merge collapsed datasets and replace original Vendor specific
        vendor_specific_collapsed_combined_ds = xr.merge(vendor_specific_collapsed_ds_list)
        echodata_copy["Vendor_specific"] = vendor_specific_collapsed_combined_ds

    if (
        echodata_copy.sonar_model == "EK80"
        and "filter_time" in echodata_copy["Vendor_specific"].dims
        and len(echodata_copy["Vendor_specific"]["filter_time"]) > 1
    ):
        # List to accumulate calibration datasets
        cal_ds_list = []

        # Grab valid channel and ping time pairings and organize into
        # channel/filter-time dictionary
        valid = (
            echodata_copy[ed_beam_group]["transmit_duration_nominal"]
            .stack(pairs=("channel", "ping_time"))
            .dropna(dim="pairs")
        )
        filter_times_all = sorted(echodata_copy["Vendor_specific"]["filter_time"].data)
        channel_filter_times = {}
        for channel in valid["channel"].values:
            ping_times = valid["ping_time"].where(valid["channel"] == channel, drop=True).values
            channel_filter_times[channel] = np.array(
                sorted(ping_times[np.isin(ping_times, filter_times_all)])
            )

        # Calibrate for each channel and filter time pairing
        for channel in channel_filter_times.keys():
            filter_times_subset = channel_filter_times[channel]
            for filter_time in filter_times_subset:
                echodata_copy_copy = echodata_copy.copy()

                # Subset echodata object to grab vendor values and ping times corresponding to
                # the index's associated filter time
                echodata_copy_copy["Vendor_specific"] = echodata_copy_copy["Vendor_specific"].sel(
                    filter_time=filter_time
                )

                # Subset for channel
                echodata_copy_copy[ed_beam_group] = echodata_copy_copy[ed_beam_group].sel(
                    channel=[channel]
                )

                # We want to subset the beam group to calibrate for 1 specific set of calibration
                # parameters, but this can get complicated:
                # In the complex FM, CW, FM case, there will be a filter time that is specific
                # to each recording. However, FM complex will be completely contained in
                # Beam_group1, so there will be a gap in Beam_group1. The filter  time
                # corresponding to the CW recording is the end time for the first FM recording.
                start_time = filter_time
                filter_times_subset_index = np.where(filter_times_subset == start_time)[0][0]
                if filter_times_subset_index == len(filter_times_subset) - 1:
                    end_time = None
                else:
                    end_time = filter_times_subset[filter_times_subset_index + 1] - np.timedelta64(
                        1, "ns"
                    )
                echodata_copy_copy[ed_beam_group] = echodata_copy_copy[ed_beam_group].sel(
                    ping_time=slice(start_time, end_time)
                )

                # Calibrate and drop filter_time
                cal_ds_iteration = _compute_cal_ds(echodata_copy_copy)
                cal_ds_list.append(cal_ds_iteration.drop_vars("filter_time"))

        # Merge across both channel and ping time dimensions
        cal_ds = xr.merge(cal_ds_list)
    else:
        # Compute a single calibration dataset
        cal_ds = _compute_cal_ds(echodata_copy)
        if "filter_time" in cal_ds:
            cal_ds = cal_ds.drop_vars("filter_time")

    # Add attributes
    def _add_attrs(cal_type, ds):
        """Add attributes to backscattering strength dataset.
        cal_type: Sv or TS
        """
        ds["range_sample"].attrs = {"long_name": "Along-range sample number, base 0"}
        ds["echo_range"].attrs = {"long_name": "Range distance", "units": "m"}
        ds[cal_type].attrs = {
            "long_name": {
                "Sv": "Volume backscattering strength (Sv re 1 m-1)",
                "TS": "Target strength (TS re 1 m^2)",
            }[cal_type],
            "units": "dB",
        }
        if echodata.sonar_model == "EK80":
            ds[cal_type] = ds[cal_type].assign_attrs(
                {
                    "waveform_mode": waveform_mode,
                    "encode_mode": encode_mode,
                }
            )

    _add_attrs(cal_type, cal_ds)

    # Add provinance
    # Provenance source files may originate from raw files (echodata.source_files)
    # or converted files (echodata.converted_raw_path)
    if echodata.source_file is not None:
        source_file = echodata.source_file
    elif echodata.converted_raw_path is not None:
        source_file = echodata.converted_raw_path
    else:
        source_file = "SOURCE FILE NOT IDENTIFIED"

    prov_dict = echopype_prov_attrs(process_type="processing")
    prov_dict["processing_function"] = f"calibrate.compute_{cal_type}"
    files_vars = source_files_vars(source_file)
    cal_ds = (
        cal_ds.assign(**files_vars["source_files_var"])
        .assign_coords(**files_vars["source_files_coord"])
        .assign_attrs(prov_dict)
    )

    # Add water_level to the created xr.Dataset
    if "water_level" in echodata["Platform"].data_vars.keys():
        cal_ds["water_level"] = echodata["Platform"].water_level

    return cal_ds


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
          `"EL"`, `"DS"`, `"TVR"`, `"VTX0"`, `"equivalent_beam_angle"`, `"Sv_offset"`

        Passing in calibration parameters for other echosounders
        are not currently supported.

    waveform_mode : {"CW", "BB", "FM"}, optional
        Type of transmit waveform.
        Required only for data from the EK80 echosounder
        and not used with any other echosounder.

        - `"CW"` for narrowband transmission,
          returned echoes recorded either as complex or power/angle samples
        - `"BB"` or `"FM"` for broadband transmission,
          returned echoes recorded as complex samples

    encode_mode : {"complex", "power"}, optional
        Type of encoded return echo data.
        Required only for data from the EK80 echosounder
        and not used with any other echosounder.

        - `"complex"` for complex samples
        - `"power"` for power/angle samples, only allowed when
          the echosounder is configured for narrowband transmission

    assume_single_filter_time : boolean, optional
        If true, filter coefficients and decimation values will be used from the first
        filter time. If false, all filter times will be used.
        This can only be used for complex EK80 calibration.

    drop_last_hanning_zero: bool, default False
        If true, uses the pyEcholab implementation of dropping the hanning window's
        last index value (which is zero). Else, follows the CRIMAC implementation and
        keeps the last zero. This is here for CI test purposes.

    Returns
    -------
    xr.Dataset
        The calibrated Sv dataset, including calibration parameters
        and environmental variables used in the calibration operations.

    Notes
    -----
    The EK80 echosounder can be configured to transmit
    either broadband/frequency modulated (``waveform_mode="BB"`` or ``waveform_mode="FM"``)
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

    The returned xr.Dataset will contain the variable `water_level` from the
    EchoData object provided, if it exists. If `water_level` is not returned,
    it must be set using `EchoData.update_platform()`.
    """
    return _compute_cal(cal_type="Sv", echodata=echodata, **kwargs)


def compute_TS(echodata: EchoData, **kwargs):
    """
    Compute target strength (TS) from raw data.

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
          `"EL"`, `"DS"`, `"TVR"`, `"VTX0"`, `"equivalent_beam_angle"`, `"Sv_offset"`

        Passing in calibration parameters for other echosounders
        are not currently supported.

    waveform_mode : {"CW", "BB", "FM"}, optional
        Type of transmit waveform.
        Required only for data from the EK80 echosounder
        and not used with any other echosounder.

        - `"CW"` for narrowband transmission,
          returned echoes recorded either as complex or power/angle samples
        - `"BB"` or `"FM"` for broadband transmission,
          returned echoes recorded as complex samples

    encode_mode : {"complex", "power"}, optional
        Type of encoded return echo data.
        Required only for data from the EK80 echosounder
        and not used with any other echosounder.

        - `"complex"` for complex samples
        - `"power"` for power/angle samples, only allowed when
          the echosounder is configured for narrowband transmission

    assume_single_filter_time : boolean, optional
        If true, filter coefficients and decimation values will be used from the first
        filter time. If false, all filter times will be used.

    drop_last_hanning_zero: bool, default False
        If true, uses the pyEcholab implementation of dropping the hanning window's
        last index value (which is zero). Else, follows the CRIMAC implementation and
        keeps the last zero. This is here for CI test purposes.

    Returns
    -------
    xr.Dataset
        The calibrated TS dataset, including calibration parameters
        and environmental variables used in the calibration operations.

    Notes
    -----
    The EK80 echosounder can be configured to transmit
    either broadband/frequency modulated (``waveform_mode="BB"`` or ``waveform_mode="FM"``)
    or narrowband (``waveform_mode="CW"``) signals.
    When transmitting in broadband mode, the returned echoes are
    encoded as complex samples (``encode_mode="complex"``).
    When transmitting in narrowband mode, the returned echoes can be encoded
    either as complex samples (``encode_mode="complex"``)
    or as power/angle combinations (``encode_mode="power"``) in a format
    similar to those recorded by EK60 echosounders.

    The current calibration implemented for EK80 broadband complex data
    uses band-integrated TS with the gain computed at the center frequency
    of the transmit signal.

    Note that in the fisheries acoustics context, it is customary to
    associate TS to a single scatterer.
    TS is defined as: TS = 10 * np.log10 (sigma_bs), where sigma_bs
    is the backscattering cross-section.

    For details, see:
    MacLennan et al. 2002. A consistent approach to definitions and
    symbols in fisheries acoustics. ICES J. Mar. Sci. 59: 365-369.
    https://doi.org/10.1006/jmsc.2001.1158
    """
    return _compute_cal(cal_type="TS", echodata=echodata, **kwargs)
