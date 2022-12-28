import numpy as np
import xarray as xr

from ..echodata import EchoData
from ..echodata.simrad import retrieve_correct_beam_group
from ..utils.log import _init_logger
from .cal_params import get_cal_params_EK, get_gain_for_complex
from .calibrate_base import CalibrateBase
from .ek80_utils import compress_pulse, get_tau_effective, get_transmit_signal
from .env_params_new import get_env_params_EK60, get_env_params_EK80

logger = _init_logger(__name__)


class CalibrateEK(CalibrateBase):
    def __init__(self, echodata: EchoData, env_params, cal_params):
        super().__init__(echodata, env_params, cal_params)

    def compute_range_meter(self, waveform_mode, encode_mode):
        """
        Parameters
        ----------
        waveform_mode : {"CW", "BB"}
            Type of transmit waveform.
            Required only for data from the EK80 echosounder.

            - `"CW"` for narrowband transmission,
              returned echoes recorded either as complex or power/angle samples
            - `"BB"` for broadband transmission,
              returned echoes recorded as complex samples

        encode_mode : {"complex", "power"}
            Type of encoded return echo data.
            Required only for data from the EK80 echosounder.

            - `"complex"` for complex samples
            - `"power"` for power/angle samples, only allowed when
              the echosounder is configured for narrowband transmission

        Returns
        -------
        range_meter : xr.DataArray
            range in units meter
        """
        self.range_meter = self.echodata.compute_range(
            self.env_params,
            ek_waveform_mode=waveform_mode,
            ek_encode_mode=encode_mode,
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

        # load env and cal parameters
        self.env_params = get_env_params_EK60(echodata=echodata, user_env_dict=env_params)

        if cal_params is None:
            cal_params = {}
        self.cal_params = get_cal_params_EK(
            echodata=echodata, user_cal_dict=cal_params, waveform_mode="CW", encode_mode="power"
        )

        # default to CW mode recorded as power samples
        self.compute_range_meter(waveform_mode="CW", encode_mode="power")

    def compute_Sv(self, **kwargs):
        power_ed_group = retrieve_correct_beam_group(
            echodata=self.echodata, waveform_mode="CW", encode_mode="power", pulse_compression=False
        )
        return self._cal_power_samples(cal_type="Sv", power_ed_group=power_ed_group)

    def compute_TS(self, **kwargs):
        power_ed_group = retrieve_correct_beam_group(
            echodata=self.echodata, waveform_mode="CW", encode_mode="power", pulse_compression=False
        )
        return self._cal_power_samples(cal_type="TS", power_ed_group=power_ed_group)


class CalibrateEK80(CalibrateEK):

    # TODO: add option to get these from data file
    fs = 1.5e6  # default full sampling frequency [Hz]
    z_et = 75
    z_er = 1000

    def __init__(self, echodata, env_params, cal_params, waveform_mode, encode_mode):
        super().__init__(echodata, env_params, cal_params)

        # TODO: make waveform_mode and encode_mode class attributes

        # load env and cal parameters
        self.env_params = get_env_params_EK80(
            echodata=echodata,
            user_env_dict=env_params,
            waveform_mode=waveform_mode,
            encode_mode=encode_mode,
        )

        if cal_params is None:
            cal_params = {}
        self.cal_params = get_cal_params_EK(
            echodata=echodata,
            user_cal_dict=cal_params,
            waveform_mode=waveform_mode,
            encode_mode=encode_mode,
        )

        # self.range_meter computed under self._compute_cal()
        # because the implementation is different depending on waveform_mode and encode_mode

    def _get_chan_dict(self):
        """
        Build dict to select BB and CW channels from complex samples where data
        from both waveform modes may co-exist.
        """
        # Use center frequency for each ping to select BB or CW channels
        # when all samples are encoded as complex samples
        if (
            "frequency_start" in self.echodata["Sonar/Beam_group1"]
            and "frequency_end" in self.echodata["Sonar/Beam_group1"]
        ):
            # At least some channels are BB
            # frequency_start and frequency_end are NaN for CW channels
            freq_center = (
                self.echodata["Sonar/Beam_group1"]["frequency_start"]
                + self.echodata["Sonar/Beam_group1"]["frequency_end"]
            ) / 2  # has beam dim

            return {
                # For BB: drop channels containing CW samples (nan in freq start/end)
                "BB": freq_center.dropna(dim="channel").channel,
                # For CW: drop channels containing BB samples (not nan in freq start/end)
                "CW": freq_center.where(np.isnan(freq_center), drop=True).channel,
            }

        else:
            # All channels are CW
            return {"BB": None, "CW": self.echodata["Sonar/Beam_group1"].channel}

    def _get_power_from_complex(self, waveform_mode, chan_dict, chirp):
        """
        Get power from complex samples.

        Returns
        -------
        prx
            Power computed from complex samples
        """

        def _get_prx(sig):
            return (
                self.echodata["Sonar/Beam_group1"].beam.size
                * np.abs(sig.mean(dim="beam")) ** 2
                / (2 * np.sqrt(2)) ** 2
                * (np.abs(self.z_er + self.z_et) / self.z_er) ** 2
                / self.z_et
            )

        if waveform_mode == "BB":
            pc = compress_pulse(
                echodata=self.echodata, chirp=chirp, chan_BB=chan_dict["BB"]
            )  # has beam dim
            prx = _get_prx(pc["pulse_compressed_output"])  # ensure prx is xr.DataArray
        else:
            backscatter_cw = self.echodata["Sonar/Beam_group1"]["backscatter_r"].sel(
                channel=chan_dict["CW"]
            ) + 1j * self.echodata["Sonar/Beam_group1"]["backscatter_i"].sel(
                channel=chan_dict["CW"]
            )
            prx = _get_prx(backscatter_cw)

        prx.name = "received_power"

        return prx

    def _cal_complex_samples(self, cal_type, waveform_mode, complex_ed_group) -> xr.Dataset:
        """Calibrate complex data from EK80.

        Parameters
        ----------
        cal_type : str
            'Sv' for calculating volume backscattering strength, or
            'TS' for calculating target strength
        waveform_mode : {"CW", "BB"}
            Type of transmit waveform.

            - `"CW"` for narrowband transmission,
              returned echoes recorded either as complex or power/angle samples
            - `"BB"` for broadband transmission,
              returned echoes recorded as complex samples
        complex_ed_group : str
            The ``EchoData`` beam group path containing complex data

        Returns
        -------
        xr.Dataset
            The calibrated dataset containing Sv or TS
        """
        # Select source of backscatter data
        beam = self.echodata[complex_ed_group]

        # Get dict for channel selection with keys "BB" and "CW"
        chan_dict = self._get_chan_dict()

        # Select channel according to waveform_mode
        chan_sel = chan_dict[waveform_mode]

        # Get transmit signal
        # tx_params = self._get_tx_params()
        tx, tx_time = get_transmit_signal(
            echodata=self.echodata,
            waveform_mode=waveform_mode,
            channel=chan_sel,
            fs=self.fs,
            z_et=self.z_et,
        )

        # Get power from complex samples
        prx = self._get_power_from_complex(
            waveform_mode=waveform_mode, chan_dict=chan_dict, chirp=tx
        )

        # Center frequency: only exists for BB
        if waveform_mode == "BB":
            freq_center = (beam["frequency_start"] + beam["frequency_end"]).sel(
                channel=chan_sel
            ).isel(beam=0).drop(
                "beam"
            ) / 2  # identical for all beams
        else:
            freq_center = None

        # Harmonize time coordinate between Beam_groupX data and env_params
        # Use self.echodata["Sonar/Beam_group1"] because complex sample is always in Beam_group1
        for p in self.env_params.keys():
            if "channel" in self.env_params[p].coords:
                self.env_params[p] = self.env_params[p].sel(channel=chan_sel)
            self.env_params[p] = self.echodata._harmonize_env_param_time(
                self.env_params[p], ping_time=beam["ping_time"]
            )

        # Common params
        sound_speed = self.env_params["sound_speed"]
        absorption = self.env_params["sound_absorption"].sel(channel=chan_sel)
        range_meter = self.range_meter.sel(channel=chan_sel)

        if waveform_mode == "BB":
            # use true center frequency for BB pulse  # TODO: BUG! [WJ resolved 2022/12/27]
            wavelength = sound_speed / freq_center

            # use true center frequency to interpolate for gain factor
            gain = get_gain_for_complex(
                echodata=self.echodata, waveform_mode=waveform_mode, chan_sel=chan_sel
            )
        else:
            # use nominal channel frequency for CW pulse
            wavelength = sound_speed / beam["frequency_nominal"].sel(channel=chan_sel)

            # use nominal channel frequency to select gain factor
            gain = get_gain_for_complex(
                echodata=self.echodata, waveform_mode=waveform_mode, chan_sel=chan_sel
            )

        spreading_loss = 20 * np.log10(range_meter.where(range_meter >= 1, other=1))
        absorption_loss = 2 * absorption * range_meter
        transmit_power = beam["transmit_power"].sel(channel=chan_sel)

        # Compute based on cal_type
        if cal_type == "Sv":
            # Compute effective pulse length
            tau_effective = get_tau_effective(
                ytx_dict=tx,
                fs_deci_dict={k: np.diff(v[:2]) for (k, v) in tx_time.items()},
                waveform_mode=waveform_mode,
                channel=chan_sel,
                ping_time=beam["ping_time"],
            )

            # equivalent_beam_angle:
            # identical within each channel regardless of ping_time/beam
            # but drop only the beam dimension here
            # to allow scaling for potential center frequency changes
            psifc = beam["equivalent_beam_angle"].sel(channel=chan_sel).isel(beam=0).drop("beam")
            if waveform_mode == "BB":
                # if BB scale according to true center frequency
                psifc += 20 * np.log10(  # TODO: BUGS! should be 20 * log10 [WJ resolved 2022/12/27]
                    beam["frequency_nominal"].sel(channel=chan_sel) / freq_center
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

    def _compute_cal(self, cal_type, waveform_mode, encode_mode) -> xr.Dataset:
        """
        Private method to compute Sv or TS from EK80 data, called by compute_Sv or compute_TS.

        Parameters
        ----------
        cal_type : str
            'Sv' for calculating volume backscattering strength, or
            'TS' for calculating target strength

        waveform_mode : {"CW", "BB"}
            Type of transmit waveform.

            - `"CW"` for narrowband transmission,
              returned echoes recorded either as complex or power/angle samples
            - `"BB"` for broadband transmission,
              returned echoes recorded as complex samples

        encode_mode : {"complex", "power"}
            Type of encoded return echo data.

            - `"complex"` for complex samples
            - `"power"` for power/angle samples, only allowed when
              the echosounder is configured for narrowband transmission

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing either Sv or TS.
        """
        # Get the right Sonar/Beam_groupX group according to encode_mode
        ed_group = retrieve_correct_beam_group(
            echodata=self.echodata,
            waveform_mode=waveform_mode,
            encode_mode=encode_mode,
            pulse_compression=False,
        )

        # Set flag_complex: True-complex cal, False-power cal
        flag_complex = True if waveform_mode == "BB" or encode_mode == "complex" else False

        if flag_complex:
            # Complex samples can be BB or CW
            self.compute_range_meter(waveform_mode=waveform_mode, encode_mode=encode_mode)
            ds_cal = self._cal_complex_samples(
                cal_type=cal_type, waveform_mode=waveform_mode, complex_ed_group=ed_group
            )
        else:
            # Power samples only make sense for CW mode data
            self.compute_range_meter(waveform_mode="CW", encode_mode=encode_mode)
            ds_cal = self._cal_power_samples(cal_type=cal_type, power_ed_group=ed_group)

        return ds_cal

    def compute_Sv(self, waveform_mode="BB", encode_mode="complex"):
        """Compute volume backscattering strength (Sv).

        Parameters
        ----------
        waveform_mode : {"CW", "BB"}
            Type of transmit waveform.

            - `"CW"` for narrowband transmission,
              returned echoes recorded either as complex or power/angle samples
            - (default) `"BB"` for broadband transmission,
              returned echoes recorded as complex samples

        encode_mode : {"complex", "power"}
            Type of encoded return echo data.

            - (default) `"complex"` for complex samples
            - `"power"` for power/angle samples, only allowed when
              the echosounder is configured for narrowband transmission

        Returns
        -------
        Sv : xr.DataSet
            A DataSet containing volume backscattering strength (``Sv``)
            and the corresponding range (``echo_range``) in units meter.
        """
        return self._compute_cal(
            cal_type="Sv", waveform_mode=waveform_mode, encode_mode=encode_mode
        )

    def compute_TS(self, waveform_mode="BB", encode_mode="complex"):
        """Compute target strength (TS).

        Parameters
        ----------
        waveform_mode : {"CW", "BB"}
            Type of transmit waveform.

            - `"CW"` for narrowband transmission,
              returned echoes recorded either as complex or power/angle samples
            - (default) `"BB"` for broadband transmission,
              returned echoes recorded as complex samples

        encode_mode : {"complex", "power"}
            Type of encoded return echo data.

            - (default) `"complex"` for complex samples
            - `"power"` for power/angle samples, only allowed when
              the echosounder is configured for narrowband transmission

        Returns
        -------
        TS : xr.DataSet
            A DataSet containing target strength (``TS``)
            and the corresponding range (``echo_range``) in units meter.
        """
        return self._compute_cal(
            cal_type="TS", waveform_mode=waveform_mode, encode_mode=encode_mode
        )
