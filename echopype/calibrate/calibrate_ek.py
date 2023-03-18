from typing import Dict

import numpy as np
import xarray as xr

from ..echodata import EchoData
from ..echodata.simrad import retrieve_correct_beam_group
from ..utils.log import _init_logger
from .cal_params import get_cal_params_EK
from .calibrate_base import CalibrateBase
from .ecs import conform_channel_order, ecs_ds2dict, ecs_ev2ep
from .ek80_complex import compress_pulse, get_filter_coeff, get_tau_effective, get_transmit_signal
from .env_params import get_env_params_EK
from .range import compute_range_EK, range_mod_TVG_EK

logger = _init_logger(__name__)


class CalibrateEK(CalibrateBase):
    def __init__(self, echodata: EchoData, env_params, cal_params, ecs_file, **kwargs):
        super().__init__(echodata, env_params, cal_params, ecs_file)

        self.ed_beam_group = None  # will be assigned in child class

    def compute_echo_range(self, chan_sel: xr.DataArray = None):
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
            chan_sel=chan_sel,
        )

    def _cal_power_samples(self, cal_type: str) -> xr.Dataset:
        """Calibrate power data from EK60 and EK80.

        Parameters
        ----------
        cal_type: str
            'Sv' for calculating volume backscattering strength, or
            'TS' for calculating target strength

        Returns
        -------
        xr.Dataset
            The calibrated dataset containing Sv or TS
        """
        # Select source of backscatter data
        beam = self.echodata[self.ed_beam_group]

        # Derived params
        wavelength = self.env_params["sound_speed"] / beam["frequency_nominal"]  # wavelength
        # range_meter = self.range_meter

        # TVG compensation with modified range
        sound_speed = self.env_params["sound_speed"]
        absorption = self.env_params["sound_absorption"]
        tvg_mod_range = range_mod_TVG_EK(
            self.echodata, self.ed_beam_group, self.range_meter, sound_speed
        )
        tvg_mod_range = tvg_mod_range.where(tvg_mod_range > 0, np.nan)

        spreading_loss = 20 * np.log10(tvg_mod_range)
        absorption_loss = 2 * absorption * tvg_mod_range

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
        out = out.merge(self.range_meter)

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
    def __init__(self, echodata: EchoData, env_params, cal_params, ecs_file, **kwargs):
        super().__init__(echodata, env_params, cal_params, ecs_file)

        # Set sonar_type and waveform/encode mode
        self.sonar_type = "EK60"

        # Set cal type
        self.waveform_mode = "CW"
        self.encode_mode = "power"

        # Get the right ed_beam_group for CW power samples
        self.ed_beam_group = retrieve_correct_beam_group(
            echodata=self.echodata, waveform_mode=self.waveform_mode, encode_mode=self.encode_mode
        )

        # Set the channels to calibrate
        # For EK60 this is all channels
        self.chan_sel = self.echodata[self.ed_group]["channel"]
        beam = self.echodata[self.ed_group]

        # Convert env_params and cal_params if self.ecs_file exists
        # Note a warning if thrown out in CalibrateBase.__init__
        # to let user know cal_params and env_params are ignored if ecs_file is provided
        if self.ecs_file is not None:  # also means self.ecs_dict != {}
            ds_cal_tmp, ds_env_tmp = ecs_ev2ep(self.ecs_dict, "EK60")
            self.cal_params = ecs_ds2dict(
                conform_channel_order(ds_cal_tmp, beam["frequency_nominal"])
            )
            self.env_params = ecs_ds2dict(
                conform_channel_order(ds_env_tmp, beam["frequency_nominal"])
            )

        # Regardless of the source cal and env params,
        # go through the same sanitization and organization process
        self.env_params = get_env_params_EK(
            sonar_type=self.sonar_type,
            beam=self.echodata[self.ed_beam_group],
            env=self.echodata["Environment"],
            user_dict=self.env_params,
        )
        self.cal_params = get_cal_params_EK(
            waveform_mode=self.waveform_mode,
            freq_center=beam["frequency_nominal"],
            beam=beam,
            vend=self.echodata["Vendor_specific"],
            user_dict=self.cal_params,
            sonar_type=self.sonar_type,
        )

        # Compute range
        self.compute_echo_range()

    def compute_Sv(self, **kwargs):
        return self._cal_power_samples(cal_type="Sv")

    def compute_TS(self, **kwargs):
        return self._cal_power_samples(cal_type="TS")


class CalibrateEK80(CalibrateEK):
    # Default EK80 params: these parameters are only recorded in later versions of EK80 software
    EK80_params = {}
    EK80_params["z_et"] = 75  # transmit impedance
    EK80_params["z_er"] = 1000  # receive impedance
    EK80_params["fs"] = {  # default full sampling frequency [Hz]
        "default": 1500000,
        "GPT": 500000,
        "SBT": 50000,
        "WBAT": 1500000,
        "WBT TUBE": 1500000,
        "WBT MINI": 1500000,
        "WBT": 1500000,
        "WBT HP": 187500,
        "WBT LF": 93750,
    }

    def __init__(
        self,
        echodata: EchoData,
        env_params,
        cal_params,
        waveform_mode,
        encode_mode,
        ecs_file=None,
        **kwargs
    ):
        super().__init__(echodata, env_params, cal_params, ecs_file)

        # Set sonar_type
        self.sonar_type = "EK80"

        # The waveform and encode mode combination checked in calibrate/api.py::_compute_cal
        # so just doing assignment here
        self.waveform_mode = waveform_mode
        self.encode_mode = encode_mode
        self.echodata = echodata

        # Get the right ed_beam_group given waveform and encode mode
        self.ed_beam_group = retrieve_correct_beam_group(
            echodata=self.echodata, waveform_mode=self.waveform_mode, encode_mode=self.encode_mode
        )

        # Select the channels to calibrate
        if self.encode_mode == "power":
            # Power sample only possible under CW mode,
            # and all power samples will live in the same group
            self.chan_sel = self.echodata[self.ed_beam_group]["channel"]
        else:
            # Complex samples can be CW or BB, so select based on waveform mode
            chan_dict = self._get_chan_dict(self.echodata[self.ed_beam_group])
            self.chan_sel = chan_dict[self.waveform_mode]

        # Subset of the right Sonar/Beam_groupX group given the selected channels
        beam = self.echodata[self.ed_beam_group].sel(channel=self.chan_sel)

        # Use center frequency if in BB mode, else use nominal channel frequency
        if self.waveform_mode == "BB":
            # use true center frequency to interpolate for various cal params
            self.freq_center = (beam["frequency_start"] + beam["frequency_end"]).sel(
                channel=self.chan_sel
            ).isel(beam=0).drop("beam") / 2
        else:
            # use nominal channel frequency for CW pulse
            self.freq_center = beam["frequency_nominal"].sel(channel=self.chan_sel)

        # Get env_params: depends on waveform mode
        self.env_params = get_env_params_EK(
            sonar_type=self.sonar_type,
            beam=beam,
            env=self.echodata["Environment"],
            user_dict=self.env_params,
            freq=self.freq_center,
        )

        # Get cal_params: depends on waveform and encode mode
        self.cal_params = get_cal_params_EK(
            waveform_mode=self.waveform_mode,
            freq_center=self.freq_center,
            beam=beam,  # already subset above
            vend=self.echodata["Vendor_specific"].sel(channel=self.chan_sel),
            user_dict=self.cal_params,
        )

        # Compute echo range in meters
        self.compute_echo_range(chan_sel=self.chan_sel)

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

    def _get_power_from_complex(
        self,
        beam: xr.Dataset,
        chirp: Dict,
        z_et,
        z_er,
    ) -> xr.DataArray:
        """
        Get power from complex samples.

        Parameters
        ----------
        beam : xr.Dataset
            EchoData["Sonar/Beam_group1"] with selected channel subset
        chirp : dict
            a dictionary containing transmit chirp for BB channels

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
                * (np.abs(z_er + z_et) / z_er) ** 2
                / z_et
            )

        # Compute power
        if self.waveform_mode == "BB":
            pc = compress_pulse(
                backscatter=beam["backscatter_r"] + 1j * beam["backscatter_i"], chirp=chirp
            )  # has beam dim
            prx = _get_prx(pc)  # ensure prx is xr.DataArray
        else:
            bs_cw = beam["backscatter_r"] + 1j * beam["backscatter_i"]
            prx = _get_prx(bs_cw)

        prx.name = "received_power"

        return prx

    def _get_B_theta_phi_m(self):
        """
        Get transceiver gain compensation for BB mode.

        ref: https://github.com/CI-CMG/pyEcholab/blob/RHT-EK80-Svf/echolab2/instruments/EK80.py#L4263-L4274  # noqa
        """
        fac_along = (
            np.abs(-self.cal_params["angle_offset_alongship"])
            / (self.cal_params["beamwidth_alongship"] / 2)
        ) ** 2
        fac_athwart = (
            np.abs(-self.cal_params["angle_offset_athwartship"])
            / (self.cal_params["beamwidth_athwartship"] / 2)
        ) ** 2
        B_theta_phi_m = 0.5 * 6.0206 * (fac_along + fac_athwart - 0.18 * fac_along * fac_athwart)

        return B_theta_phi_m

    def _cal_complex_samples(self, cal_type: str) -> xr.Dataset:
        """Calibrate complex data from EK80.

        Parameters
        ----------
        cal_type : str
            'Sv' for calculating volume backscattering strength, or
            'TS' for calculating target strength

        Returns
        -------
        xr.Dataset
            The calibrated dataset containing Sv or TS
        """
        # Select source of backscatter data
        beam = self.echodata[self.ed_beam_group].sel(channel=self.chan_sel)
        vend = self.echodata["Vendor_specific"].sel(channel=self.chan_sel)

        # Get transmit signal
        tx_coeff = get_filter_coeff(vend)
        fs = self.cal_params["receiver_sampling_frequency"]

        # Switch to use Anderson implementation for transmit chirp starting v0.6.4
        tx, tx_time = get_transmit_signal(beam, tx_coeff, self.waveform_mode, fs)

        # Params to clarity in use below
        z_er = self.cal_params["impedance_receive"]
        z_et = self.cal_params["impedance_transmit"]
        gain = self.cal_params["gain_correction"]

        # Transceiver gain compensation for BB mode
        if self.waveform_mode == "BB":
            gain = gain - self._get_B_theta_phi_m()

        absorption = self.env_params["sound_absorption"]
        range_meter = self.range_meter
        sound_speed = self.env_params["sound_speed"]
        wavelength = sound_speed / self.freq_center
        transmit_power = beam["transmit_power"]

        # TVG compensation with modified range
        tvg_mod_range = range_mod_TVG_EK(
            self.echodata, self.ed_beam_group, range_meter, sound_speed
        )
        tvg_mod_range = tvg_mod_range.where(tvg_mod_range > 0, np.nan)

        spreading_loss = 20 * np.log10(tvg_mod_range)
        absorption_loss = 2 * absorption * tvg_mod_range

        # Get power from complex samples
        prx = self._get_power_from_complex(beam=beam, chirp=tx, z_et=z_et, z_er=z_er)
        prx = prx.where(prx > 0, np.nan)

        # Compute based on cal_type
        if cal_type == "Sv":
            # Effective pulse length
            # compute first assuming all channels are not GPT
            tau_effective = get_tau_effective(
                ytx_dict=tx,
                fs_deci_dict={k: 1 / np.diff(v[:2]) for (k, v) in tx_time.items()},  # decimated fs
                waveform_mode=self.waveform_mode,
                channel=self.chan_sel,
                ping_time=beam["ping_time"],
            )
            # Use pulse_duration in place of tau_effective for GPT channels
            # below assumesthat all transmit parameters are identical
            # and needs to be changed when allowing transmit parameters to vary by ping
            ch_GPT = vend["transceiver_type"] == "GPT"
            tau_effective[ch_GPT] = beam["transmit_duration_nominal"][ch_GPT].isel(ping_time=0)

            # equivalent_beam_angle
            # TODO: THIS ONE CARRIES THE BEAM DIMENSION AROUND
            psifc = self.cal_params["equivalent_beam_angle"]

            out = (
                10 * np.log10(prx)
                + spreading_loss
                + absorption_loss
                - 10 * np.log10(wavelength**2 * transmit_power * sound_speed / (32 * np.pi**2))
                - 2 * gain
                - 10 * np.log10(tau_effective)
                - psifc
            )

            # Correct for sa_correction if CW mode
            if self.waveform_mode == "CW":
                out = out - 2 * self.cal_params["sa_correction"]

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
        if "beam" in out.coords:
            return out.isel(beam=0).drop("beam")
        else:
            return out

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
            ds_cal = self._cal_complex_samples(cal_type=cal_type)
        else:
            # Power samples only make sense for CW mode data
            ds_cal = self._cal_power_samples(cal_type=cal_type)

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
