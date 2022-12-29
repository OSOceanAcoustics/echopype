from collections import defaultdict
from typing import Dict

import numpy as np
import xarray as xr

from ..echodata import EchoData
from ..echodata.simrad import check_input_args_combination, retrieve_correct_beam_group
from ..utils.log import _init_logger
from .cal_params import get_cal_params_EK, get_gain_BB, get_vend_filter_EK80
from .calibrate_base import CalibrateBase
from .ek80_complex import compress_pulse, get_tau_effective, get_transmit_signal
from .env_params import get_env_params_EK60, get_env_params_EK80
from .range import compute_range_EK

logger = _init_logger(__name__)


class CalibrateEK(CalibrateBase):
    def __init__(self, echodata: EchoData, env_params, cal_params):
        super().__init__(echodata, env_params, cal_params)

    def compute_echo_range(self):
        """
        Compute echo range for EK echosounders.

        Returns
        -------
        range_meter : xr.DataArray
            range in units meter
        """
        self.range_meter = compute_range_EK(
            echodata=self.echodata,
            env_params=self.env_params,
            waveform_mode=self.waveform_mode,
            encode_mode=self.encode_mode,
        )

    def _cal_power_samples(self, cal_type: str, power_ed_group: str = None) -> xr.Dataset:
        """Calibrate power data from EK60 and EK80.

        Parameters
        ----------
        cal_type: str
            'Sv' for calculating volume backscattering strength, or
            'TS' for calculating target strength
        power_ed_group:
            The ``EchoData`` beam group path containing the power data

        Returns
        -------
        xr.Dataset
            The calibrated dataset containing Sv or TS
        """
        # Select source of backscatter data
        beam = self.echodata[power_ed_group]

        # Harmonize time coordinate between Beam_groupX data and env_params
        for p in self.env_params.keys():
            self.env_params[p] = self.echodata._harmonize_env_param_time(
                self.env_params[p], ping_time=beam.ping_time
            )

        # Derived params
        wavelength = self.env_params["sound_speed"] / beam["frequency_nominal"]  # wavelength
        range_meter = self.range_meter

        # Transmission loss
        spreading_loss = 20 * np.log10(range_meter.where(range_meter >= 1, other=1))
        absorption_loss = 2 * self.env_params["sound_absorption"] * range_meter

        if cal_type == "Sv":
            # Calc gain
            CSv = (
                10 * np.log10(beam["transmit_power"])
                + 2 * self.cal_params["gain_correction"]
                + self.cal_params["equivalent_beam_angle"]  # has beam dim
                + 10
                * np.log10(
                    wavelength**2
                    * beam["transmit_duration_nominal"]
                    * self.env_params["sound_speed"]
                    / (32 * np.pi**2)
                )
            )

            # Calibration and echo integration
            out = (
                beam["backscatter_r"]  # has beam dim
                + spreading_loss
                + absorption_loss
                - CSv  # has beam dim
                - 2 * self.cal_params["sa_correction"]
            )
            out.name = "Sv"

        elif cal_type == "TS":
            # Calc gain
            CSp = (
                10 * np.log10(beam["transmit_power"])
                + 2 * self.cal_params["gain_correction"]
                + 10 * np.log10(wavelength**2 / (16 * np.pi**2))
            )

            # Calibration and echo integration
            out = beam["backscatter_r"] + spreading_loss * 2 + absorption_loss - CSp
            out.name = "TS"

        # Attach calculated range (with units meter) into data set
        out = out.to_dataset()
        out = out.merge(range_meter)

        # Add frequency_nominal to data set
        out["frequency_nominal"] = beam["frequency_nominal"]

        # Add env and cal parameters
        out = self._add_params_to_output(out)

        # Remove time1 if exist as a coordinate
        if "time1" in out.coords:
            out = out.drop("time1")

        # Squeeze out the beam dim
        # doing it here because both out and self.cal_params["equivalent_beam_angle"] has beam dim
        return out.squeeze("beam", drop=True)


class CalibrateEK60(CalibrateEK):
    def __init__(self, echodata, env_params, cal_params, **kwargs):
        super().__init__(echodata, env_params, cal_params)

        # Get env_params
        self.env_params = get_env_params_EK60(echodata=echodata, user_env_dict=env_params)
        self.waveform_mode = "CW"
        self.encode_mode = "power"

        # Compute range
        self.compute_echo_range()

        # Get the right ed_group for CW power samples
        self.ed_group = retrieve_correct_beam_group(
            echodata=self.echodata, waveform_mode=self.waveform_mode, encode_mode=self.encode_mode
        )

        # Set the channels to calibrate
        # For EK60 this is all channels
        self.chan_sel = self.echodata[self.ed_group]["channel"]

        # Get cal_params
        self.cal_params = get_cal_params_EK(
            beam=echodata[self.ed_group].sel(channel=self.chan_sel),
            vend=echodata["Vendor_specific"].sel(channel=self.chan_sel),
            user_cal_dict=cal_params,
        )

    def compute_Sv(self, **kwargs):
        return self._cal_power_samples(cal_type="Sv", power_ed_group=self.ed_group)

    def compute_TS(self, **kwargs):
        return self._cal_power_samples(cal_type="TS", power_ed_group=self.ed_group)


class CalibrateEK80(CalibrateEK):

    # TODO: add option to get these from data file
    fs = 1.5e6  # default full sampling frequency [Hz]
    z_et = 75
    z_er = 1000

    def __init__(self, echodata, env_params, cal_params, waveform_mode, encode_mode):
        super().__init__(echodata, env_params, cal_params)

        # Check the combination of waveform and encode mode makes sense
        check_input_args_combination(waveform_mode, encode_mode)
        self.waveform_mode = waveform_mode
        self.encode_mode = encode_mode
        self.echodata = echodata

        # Get the right ed_group given waveform and encode mode
        self.ed_group = retrieve_correct_beam_group(
            echodata=self.echodata, waveform_mode=self.waveform_mode, encode_mode=self.encode_mode
        )

        # Select the channels to calibrate
        if self.encode_mode == "power":
            # Power sample only possible under CW mode,
            # and all power samples will live in the same group
            self.chan_sel = self.echodata[self.ed_group]["channel"]
        else:
            # Complex samples can be CW or BB, so select based on waveform mode
            chan_dict = self._get_chan_dict(self.echodata[self.ed_group])
            self.chan_sel = chan_dict[self.waveform_mode]

        # Subset of the right Sonar/Beam_groupX group given the selected channels
        beam = self.echodata[self.ed_group].sel(channel=self.chan_sel)

        # Use center frequency if in BB mode, else use nominal channel frequency
        if self.waveform_mode == "BB":
            # use true center frequency to interpolate for gain factor
            self.freq_center = (beam["frequency_start"] + beam["frequency_end"]).sel(
                channel=self.chan_sel
            ).isel(beam=0).drop("beam") / 2
        else:
            # use nominal channel frequency for CW pulse
            self.freq_center = beam["frequency_nominal"].sel(channel=self.chan_sel)

        # Get env_params: depends on waveform mode
        self.env_params = get_env_params_EK80(
            echodata=echodata, freq=self.freq_center, user_env_dict=env_params
        )

        # Get cal_params: depends on waveform and encode mode
        self.cal_params = get_cal_params_EK(
            beam=beam,
            vend=self.echodata["Vendor_specific"].sel(channel=self.chan_sel),
            user_cal_dict=cal_params,
        )

        # Compute echo range in meters
        self.compute_echo_range()

    @staticmethod
    def _get_chan_dict(beam: xr.Dataset) -> Dict:
        """
        Build dict to select BB and CW channels from complex samples where data
        from both waveform modes may co-exist.
        """
        # Use center frequency for each ping to select BB or CW channels
        # when all samples are encoded as complex samples
        if "frequency_start" in beam and "frequency_end" in beam:
            # At least some channels are BB
            # frequency_start and frequency_end are NaN for CW channels
            freq_center = (beam["frequency_start"] + beam["frequency_end"]) / 2  # has beam dim

            return {
                # For BB: drop channels containing CW samples (nan in freq start/end)
                "BB": freq_center.dropna(dim="channel").channel,
                # For CW: drop channels containing BB samples (not nan in freq start/end)
                "CW": freq_center.where(np.isnan(freq_center), drop=True).channel,
            }

        else:
            # All channels are CW
            return {"BB": None, "CW": beam.channel}

    def _get_filter_coeff(self, channel: xr.DataArray) -> Dict:
        """
        Get WBT and PC filter coefficients for constructing the transmit replica.

        Returns
        -------
        A dictionary indexed by ``channel`` and values being dictionaries containing
        filter coefficients and decimation factors for constructing the transmit replica.
        """
        vend = self.echodata["Vendor_specific"]
        coeff = defaultdict(dict)
        for ch_id in channel.values:
            # filter coefficients and decimation factor
            coeff[ch_id]["wbt_fil"] = get_vend_filter_EK80(vend, ch_id, "WBT", "coeff")
            coeff[ch_id]["pc_fil"] = get_vend_filter_EK80(vend, ch_id, "PC", "coeff")
            coeff[ch_id]["wbt_decifac"] = get_vend_filter_EK80(vend, ch_id, "WBT", "decimation")
            coeff[ch_id]["pc_decifac"] = get_vend_filter_EK80(vend, ch_id, "PC", "decimation")

        return coeff

    def _get_power_from_complex(
        self, beam: xr.Dataset, chan_sel: xr.DataArray, chirp: Dict
    ) -> xr.DataArray:
        """
        Get power from complex samples.

        Parameters
        ----------
        beam : xr.Dataset
            EchoData["Sonar/Beam_group1"]

        Returns
        -------
        prx : xr.DataArray
            Power computed from complex samples
        """

        def _get_prx(sig):
            return (
                beam["beam"].size  # number of transducer sectors
                * np.abs(sig.mean(dim="beam")) ** 2
                / (2 * np.sqrt(2)) ** 2
                * (np.abs(self.z_er + self.z_et) / self.z_er) ** 2
                / self.z_et
            )

        if self.waveform_mode == "BB":
            pc = compress_pulse(beam=beam, chirp=chirp, chan_BB=chan_sel)  # has beam dim
            prx = _get_prx(pc["pulse_compressed_output"])  # ensure prx is xr.DataArray
        else:
            bs_cw = beam["backscatter_r"].sel(channel=chan_sel) + 1j * beam["backscatter_i"].sel(
                channel=chan_sel
            )
            prx = _get_prx(bs_cw)

        prx.name = "received_power"

        return prx

    def _cal_complex_samples(self, cal_type: str, complex_ed_group: str) -> xr.Dataset:
        """Calibrate complex data from EK80.

        Parameters
        ----------
        cal_type : str
            'Sv' for calculating volume backscattering strength, or
            'TS' for calculating target strength
        complex_ed_group : str
            The ``EchoData`` beam group path containing complex data

        Returns
        -------
        xr.Dataset
            The calibrated dataset containing Sv or TS
        """
        # Select source of backscatter data
        beam = self.echodata[complex_ed_group]

        # Get transmit signal
        tx_coeff = self._get_filter_coeff(channel=self.chan_sel)
        tx, tx_time = get_transmit_signal(
            beam=beam,
            coeff=tx_coeff,
            waveform_mode=self.waveform_mode,
            channel=self.chan_sel,
            fs=self.fs,
            z_et=self.z_et,
        )

        # Get power from complex samples
        prx = self._get_power_from_complex(beam=beam, chan_sel=self.chan_sel, chirp=tx)

        # Harmonize time coordinate between Beam_groupX data and env_params
        # Use self.echodata["Sonar/Beam_group1"] because complex sample is always in Beam_group1
        for p in self.env_params.keys():
            if "channel" in self.env_params[p].coords:
                self.env_params[p] = self.env_params[p].sel(channel=self.chan_sel)
            self.env_params[p] = self.echodata._harmonize_env_param_time(
                self.env_params[p], ping_time=beam["ping_time"]
            )

        # Common params
        sound_speed = self.env_params["sound_speed"]
        absorption = self.env_params["sound_absorption"].sel(channel=self.chan_sel)
        range_meter = self.range_meter.sel(channel=self.chan_sel)
        wavelength = sound_speed / self.freq_center

        if self.waveform_mode == "BB" and "gain" in self.echodata["Vendor_specific"]:
            # If frequency-dependent gain exists, interpolate at true center frequency
            gain = get_gain_BB(
                vend=self.echodata["Vendor_specific"],
                freq_center=self.freq_center,
                cal_params_CW=self.cal_params,
            )
        else:
            # use gain already retrieved in init
            gain = self.cal_params["gain_correction"]

        spreading_loss = 20 * np.log10(range_meter.where(range_meter >= 1, other=1))
        absorption_loss = 2 * absorption * range_meter
        transmit_power = beam["transmit_power"].sel(channel=self.chan_sel)

        # Compute based on cal_type
        if cal_type == "Sv":
            # Compute effective pulse length
            tau_effective = get_tau_effective(
                ytx_dict=tx,
                fs_deci_dict={k: np.diff(v[:2]) for (k, v) in tx_time.items()},
                waveform_mode=self.waveform_mode,
                channel=self.chan_sel,
                ping_time=beam["ping_time"],
            )

            # equivalent_beam_angle:
            # identical within each channel regardless of ping_time/beam
            # but drop only the beam dimension here
            # to allow scaling for potential center frequency changes
            psifc = (
                beam["equivalent_beam_angle"].sel(channel=self.chan_sel).isel(beam=0).drop("beam")
            )
            if self.waveform_mode == "BB":
                # if BB scale according to true center frequency
                psifc += 20 * np.log10(  # TODO: BUGS! should be 20 * log10 [WJ resolved 2022/12/27]
                    beam["frequency_nominal"].sel(channel=self.chan_sel) / self.freq_center
                )

            out = (
                10 * np.log10(prx)
                + spreading_loss
                + absorption_loss
                - 10 * np.log10(wavelength**2 * transmit_power * sound_speed / (32 * np.pi**2))
                - 2 * gain
                - 10 * np.log10(tau_effective)
                - psifc
            )
            out.name = "Sv"
            # out = out.rename_vars({list(out.data_vars.keys())[0]: "Sv"})

        elif cal_type == "TS":
            out = (
                10 * np.log10(prx)
                + 2 * spreading_loss
                + absorption_loss
                - 10 * np.log10(wavelength**2 * transmit_power / (16 * np.pi**2))
                - 2 * gain
            )
            out.name = "TS"

        # Attach calculated range (with units meter) into data set
        out = out.to_dataset().merge(range_meter)

        # Add frequency_nominal to data set
        out["frequency_nominal"] = beam["frequency_nominal"]

        # Add env and cal parameters
        out = self._add_params_to_output(out)

        # Squeeze out the beam dim
        # out has beam dim, which came from absorption and absorption_loss
        # self.cal_params["equivalent_beam_angle"] also has beam dim

        # TODO: out should not have beam dimension at this stage
        # once that dimension is removed from equivalent_beam_angle
        return out.isel(beam=0).drop("beam")

    def _compute_cal(self, cal_type) -> xr.Dataset:
        """
        Private method to compute Sv or TS from EK80 data, called by compute_Sv or compute_TS.

        Parameters
        ----------
        cal_type : str
            'Sv' for calculating volume backscattering strength, or
            'TS' for calculating target strength

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing either Sv or TS.
        """
        # Set flag_complex: True-complex cal, False-power cal
        flag_complex = (
            True if self.waveform_mode == "BB" or self.encode_mode == "complex" else False
        )

        if flag_complex:
            # Complex samples can be BB or CW
            self.compute_echo_range()
            ds_cal = self._cal_complex_samples(cal_type=cal_type, complex_ed_group=self.ed_group)
        else:
            # Power samples only make sense for CW mode data
            self.compute_echo_range()
            ds_cal = self._cal_power_samples(cal_type=cal_type, power_ed_group=self.ed_group)

        return ds_cal

    def compute_Sv(self):
        """Compute volume backscattering strength (Sv).

        Returns
        -------
        Sv : xr.DataSet
            A DataSet containing volume backscattering strength (``Sv``)
            and the corresponding range (``echo_range``) in units meter.
        """
        return self._compute_cal(cal_type="Sv")

    def compute_TS(self):
        """Compute target strength (TS).

        Returns
        -------
        TS : xr.DataSet
            A DataSet containing target strength (``TS``)
            and the corresponding range (``echo_range``) in units meter.
        """
        return self._compute_cal(cal_type="TS")
