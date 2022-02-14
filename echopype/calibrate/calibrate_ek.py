import numpy as np
import xarray as xr
from dask.array.core import Array
from scipy import signal

from ..utils import uwa
from .calibrate_base import CAL_PARAMS, CalibrateBase


class CalibrateEK(CalibrateBase):
    def __init__(self, echodata, env_params):
        super().__init__(echodata, env_params)

        # cal params specific to EK echosounders
        self.cal_params = dict.fromkeys(CAL_PARAMS["EK"])

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

    def _get_vend_cal_params_power(self, param, waveform_mode):
        """Get cal parameters stored in the Vendor group.

        Parameters
        ----------
        param : str
            name of parameter to retrieve
        """
        # TODO: need to test with EK80 power/angle data
        #  currently this has only been tested with EK60 data
        ds_vend = self.echodata.vendor

        if ds_vend is None or param not in ds_vend:
            return None

        if param not in ["sa_correction", "gain_correction"]:
            raise ValueError(f"Unknown parameter {param}")

        # Drop NaN ping_time for transmit_duration_nominal
        if np.any(np.isnan(self.echodata.beam["transmit_duration_nominal"])):
            # TODO: resolve performance warning:
            #  /Users/wu-jung/miniconda3/envs/echopype_jan2021/lib/python3.8/site-packages/xarray/core/indexing.py:1369:
            #  PerformanceWarning: Slicing is producing a large chunk. To accept the large
            #  chunk and silence this warning, set the option
            #      >>> with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            #      ...     array[indexer]
            #  To avoid creating the large chunks, set the option
            #      >>> with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            #      ...     array[indexer]
            #    return self.array[key]
            self.echodata.beam = self.echodata.beam.dropna(
                dim="ping_time", how="any", subset=["transmit_duration_nominal"]
            )

        if waveform_mode == "CW" and self.echodata.beam_power is not None:
            beam = self.echodata.beam_power
        else:
            beam = self.echodata.beam

        # indexes of frequencies that are for power, not complex
        relevant_indexes = np.where(np.isin(ds_vend["frequency"], beam["frequency"]))[0]

        unique_pulse_length = np.unique(beam["transmit_duration_nominal"], axis=1)

        pulse_length = ds_vend["pulse_length"][relevant_indexes]

        # Find index with correct pulse length
        idx_wanted = np.abs(pulse_length - unique_pulse_length).argmin(
            dim="pulse_length_bin"
        )

        # Checks for dask array and compute first
        if isinstance(idx_wanted.data, Array):
            idx_wanted = idx_wanted.data.compute()

        return (
            ds_vend[param]
            .isel(pulse_length_bin=idx_wanted, frequency=relevant_indexes)
            .drop("pulse_length_bin")
        )

    def get_cal_params(self, cal_params, waveform_mode, encode_mode):
        """Get cal params using user inputs or values from data file.

        Parameters
        ----------
        cal_params : dict
        """

        if (
            encode_mode == "power"
            and waveform_mode == "CW"
            and self.echodata.beam_power is not None
        ):
            beam = self.echodata.beam_power
        else:
            beam = self.echodata.beam

        # Params from the Vendor group
        # only execute this if cw and power
        if waveform_mode == "CW" and (
            self.echodata.beam_power is not None or "quadrant" not in self.echodata.beam
        ):
            params_from_vend = ["sa_correction", "gain_correction"]
            for p in params_from_vend:
                # substitute if None in user input
                self.cal_params[p] = (
                    cal_params[p]
                    if p in cal_params
                    else self._get_vend_cal_params_power(p, waveform_mode=waveform_mode)
                )

        # Other params
        self.cal_params["equivalent_beam_angle"] = (
            cal_params["equivalent_beam_angle"]
            if "equivalent_beam_angle" in cal_params
            else beam["equivalent_beam_angle"]
        )

    def _cal_power(self, cal_type, use_beam_power=False) -> xr.Dataset:
        """Calibrate power data from EK60 and EK80.

        Parameters
        ----------
        cal_type : str
            'Sv' for calculating volume backscattering strength, or
            'Sp' for calculating point backscattering strength
        use_beam_power : bool
            whether to use beam_power.
            If ``True`` use ``echodata.beam_power``; if ``False`` use ``echodata.beam``.
            Note ``echodata.beam_power`` could only exist for EK80 data.

        Returns
        -------
        xr.Dataset
            The calibrated dataset containing Sv or Sp
        """
        # Select source of backscatter data
        if use_beam_power:
            beam = self.echodata.beam_power
        else:
            beam = self.echodata.beam

        # Derived params
        wavelength = self.env_params["sound_speed"] / beam["frequency"]  # wavelength
        range_meter = self.range_meter

        # Transmission loss
        spreading_loss = 20 * np.log10(range_meter.where(range_meter >= 1, other=1))
        absorption_loss = 2 * self.env_params["sound_absorption"] * range_meter

        if cal_type == "Sv":
            # Calc gain
            CSv = (
                10 * np.log10(beam["transmit_power"])
                + 2 * self.cal_params["gain_correction"]
                + self.cal_params["equivalent_beam_angle"]
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
                beam["backscatter_r"]
                + spreading_loss
                + absorption_loss
                - CSv
                - 2 * self.cal_params["sa_correction"]
            )
            out.name = "Sv"

        elif cal_type == "Sp":
            # Calc gain
            CSp = (
                10 * np.log10(beam["transmit_power"])
                + 2 * self.cal_params["gain_correction"]
                + 10 * np.log10(wavelength**2 / (16 * np.pi**2))
            )

            # Calibration and echo integration
            out = beam["backscatter_r"] + spreading_loss * 2 + absorption_loss - CSp
            out.name = "Sp"

        # Attach calculated range (with units meter) into data set
        out = out.to_dataset()
        out = out.merge(range_meter)

        # Add env and cal parameters
        out = self._add_params_to_output(out)

        return out


class CalibrateEK60(CalibrateEK):
    def __init__(self, echodata, env_params, cal_params, **kwargs):
        super().__init__(echodata, env_params)

        # load env and cal parameters
        self.get_env_params()
        if cal_params is None:
            cal_params = {}
        self.get_cal_params(cal_params, waveform_mode="CW", encode_mode="power")

        # default to CW mode recorded as power samples
        self.compute_range_meter(waveform_mode="CW", encode_mode="power")

    def get_env_params(self, **kwargs):
        """Get env params using user inputs or values from data file.

        EK60 file by default contains only sound speed and absorption.
        In cases when temperature, salinity, and pressure values are supplied
        by the user simultaneously, the sound speed and absorption are re-calculated.

        Parameters
        ----------
        env_params : dict
        """
        # Re-calculate environment parameters if user supply all env variables
        if (
            ("temperature" in self.env_params)
            and ("salinity" in self.env_params)
            and ("pressure" in self.env_params)
        ):
            self.env_params["sound_speed"] = uwa.calc_sound_speed(
                temperature=self.env_params["temperature"],
                salinity=self.env_params["salinity"],
                pressure=self.env_params["pressure"],
            )
            self.env_params["sound_absorption"] = uwa.calc_absorption(
                frequency=self.echodata.beam["frequency"],
                temperature=self.env_params["temperature"],
                salinity=self.env_params["salinity"],
                pressure=self.env_params["pressure"],
            )
        # Otherwise get sound speed and absorption from user inputs or raw data file
        else:
            self.env_params["sound_speed"] = (
                self.env_params["sound_speed"]
                if "sound_speed" in self.env_params
                else self.echodata.environment["sound_speed_indicative"]
            )
            self.env_params["sound_absorption"] = (
                self.env_params["sound_absorption"]
                if "sound_absorption" in self.env_params
                else self.echodata.environment["absorption_indicative"]
            )

    def compute_Sv(self, **kwargs):
        return self._cal_power(cal_type="Sv")

    def compute_Sp(self, **kwargs):
        return self._cal_power(cal_type="Sp")


class CalibrateEK80(CalibrateEK):
    fs = 1.5e6  # default full sampling frequency [Hz]
    z_et = 75
    z_er = 1000

    def __init__(self, echodata, env_params, cal_params, waveform_mode, encode_mode):
        super().__init__(echodata, env_params)

        # initialize cal params
        # cal params are those used by both complex and power data calibration
        # TODO: add complex data-specific params, like the freq-dependent gain factor
        self.cal_params = dict.fromkeys(CAL_PARAMS["EK"])
        # TODO: make waveform_mode and encode_mode class attributes

        # load env and cal parameters
        self.get_env_params(waveform_mode=waveform_mode, encode_mode=encode_mode)
        if cal_params is None:
            cal_params = {}
        self.get_cal_params(
            cal_params, waveform_mode=waveform_mode, encode_mode=encode_mode
        )

        # self.range_meter computed under self._compute_cal()
        # because the implementation is different depending on waveform_mode and encode_mode

    def get_env_params(self, waveform_mode=None, encode_mode="complex"):
        """Get env params using user inputs or values from data file.

        EK80 file by default contains sound speed, temperature, depth, salinity, and acidity,
        therefore absorption is always calculated unless it is supplied by the user.
        In cases when temperature, salinity, and pressure values are supplied
        by the user simultaneously, both the sound speed and absorption are re-calculated.

        Parameters
        ----------
        env_params : dict

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
        """

        if (
            encode_mode == "power"
            and waveform_mode == "CW"
            and self.echodata.beam_power is not None
        ):
            beam = self.echodata.beam_power
        else:
            beam = self.echodata.beam

        # Use center frequency if in BB mode, else use nominal channel frequency
        if waveform_mode == "BB":
            freq = (beam["frequency_start"] + beam["frequency_end"]) / 2
        else:
            freq = beam["frequency"]

        # Re-calculate environment parameters if user supply all env variables
        if (
            ("temperature" in self.env_params)
            and ("salinity" in self.env_params)
            and ("pressure" in self.env_params)
        ):
            self.env_params["sound_speed"] = uwa.calc_sound_speed(
                temperature=self.env_params["temperature"],
                salinity=self.env_params["salinity"],
                pressure=self.env_params["pressure"],
            )
            self.env_params["sound_absorption"] = uwa.calc_absorption(
                frequency=freq,
                temperature=self.env_params["temperature"],
                salinity=self.env_params["salinity"],
                pressure=self.env_params["pressure"],
            )
        # Otherwise
        #  get temperature, salinity, and pressure from raw data file
        #  get sound speed from user inputs or raw data file
        #  get absorption from user inputs or computing from env params stored in raw data file
        else:
            # pressure is encoded as "depth" in EK80  # TODO: change depth to pressure in EK80 file?
            for p1, p2 in zip(
                ["temperature", "salinity", "pressure"],
                ["temperature", "salinity", "depth"],
            ):
                self.env_params[p1] = (
                    self.env_params[p1]
                    if p1 in self.env_params
                    else self.echodata.environment[p2]
                )
            self.env_params["sound_speed"] = (
                self.env_params["sound_speed"]
                if "sound_speed" in self.env_params
                else self.echodata.environment["sound_speed_indicative"]
            )
            self.env_params["sound_absorption"] = (
                self.env_params["sound_absorption"]
                if "sound_absorption" in self.env_params
                else uwa.calc_absorption(
                    frequency=freq,
                    temperature=self.env_params["temperature"],
                    salinity=self.env_params["salinity"],
                    pressure=self.env_params["pressure"],
                )
            )

    def _get_vend_cal_params_complex(self, channel_id, filter_name, param_type):
        """Get filter coefficients stored in the Vendor group attributes.

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
            v = self.echodata.vendor.attrs[
                "%s %s filter_r" % (channel_id, filter_name)
            ] + 1j * np.array(
                self.echodata.vendor.attrs["%s %s filter_i" % (channel_id, filter_name)]
            )
            if v.size == 1:
                v = np.expand_dims(v, axis=0)  # expand dims for convolution
            return v
        else:
            return self.echodata.vendor.attrs[
                "%s %s decimation" % (channel_id, filter_name)
            ]

    def _tapered_chirp(
        self,
        transmit_duration_nominal,
        slope,
        transmit_power,
        frequency=None,
        frequency_start=None,
        frequency_end=None,
    ):
        """Create a baseline chirp template."""
        if frequency_start is None and frequency_end is None:  # CW waveform
            frequency_start = frequency
            frequency_end = frequency

        t = np.arange(0, transmit_duration_nominal, 1 / self.fs)
        nwtx = int(2 * np.floor(slope * t.size))  # length of tapering window
        wtx_tmp = np.hanning(nwtx)  # hanning window
        nwtxh = int(np.round(nwtx / 2))  # half length of the hanning window
        wtx = np.concatenate(
            [wtx_tmp[0:nwtxh], np.ones((t.size - nwtx)), wtx_tmp[nwtxh:]]
        )  # assemble full tapering window
        y_tmp = (
            np.sqrt((transmit_power / 4) * (2 * self.z_et))  # amplitude
            * signal.chirp(t, frequency_start, t[-1], frequency_end)
            * wtx
        )  # taper and scale linear chirp
        return y_tmp / np.max(np.abs(y_tmp)), t  # amp has no actual effect

    def _filter_decimate_chirp(self, y, ch_id):
        """Filter and decimate the chirp template.

        Parameters
        ----------
        y : np.array
            chirp from _tapered_chirp
        ch_id : str
            channel_id to select the right coefficients and factors
        """
        # filter coefficients and decimation factor
        wbt_fil = self._get_vend_cal_params_complex(ch_id, "WBT", "coeff")
        pc_fil = self._get_vend_cal_params_complex(ch_id, "PC", "coeff")
        wbt_decifac = self._get_vend_cal_params_complex(ch_id, "WBT", "decimation")
        pc_decifac = self._get_vend_cal_params_complex(ch_id, "PC", "decimation")

        # WBT filter and decimation
        ytx_wbt = signal.convolve(y, wbt_fil)
        ytx_wbt_deci = ytx_wbt[0::wbt_decifac]

        # PC filter and decimation
        if len(pc_fil.squeeze().shape) == 0:  # in case it is a single element
            pc_fil = [pc_fil.squeeze()]
        ytx_pc = signal.convolve(ytx_wbt_deci, pc_fil)
        ytx_pc_deci = ytx_pc[0::pc_decifac]
        ytx_pc_deci_time = (
            np.arange(ytx_pc_deci.size) * 1 / self.fs * wbt_decifac * pc_decifac
        )

        return ytx_pc_deci, ytx_pc_deci_time

    @staticmethod
    def _get_tau_effective(ytx, fs_deci, waveform_mode):
        """Compute effective pulse length.

        Parameters
        ----------
        ytx :array
            transmit signal
        fs_deci : float
            sampling frequency of the decimated (recorded) signal
        waveform_mode : str
            ``CW`` for CW-mode samples, either recorded as complex or power samples
            ``BB`` for BB-mode samples, recorded as complex samples
        """
        if waveform_mode == "BB":
            ytxa = (
                signal.convolve(ytx, np.flip(np.conj(ytx))) / np.linalg.norm(ytx) ** 2
            )
            ptxa = abs(ytxa) ** 2
        elif waveform_mode == "CW":
            ptxa = np.abs(ytx) ** 2  # energy of transmit signal
        return ptxa.sum() / (
            ptxa.max() * fs_deci
        )  # TODO: verify fs_deci = 1.5e6 in spheroid data sets

    def get_transmit_chirp(self, waveform_mode):
        """Reconstruct transmit signal and compute effective pulse length.

        Parameters
        ----------
        waveform_mode : str
            ``CW`` for CW-mode samples, either recorded as complex or power samples
            ``BB`` for BB-mode samples, recorded as complex samples
        """
        # Make sure it is BB mode data
        if waveform_mode == "BB" and (
            ("frequency_start" not in self.echodata.beam)
            or ("frequency_end" not in self.echodata.beam)
        ):
            raise TypeError("File does not contain BB mode complex samples!")

        y_all = {}
        y_time_all = {}
        tau_effective = {}
        for freq in self.echodata.beam.frequency.values:
            # TODO: currently only deal with the case with
            # a fixed tx key param values within a channel
            if waveform_mode == "BB":
                tx_param_names = [
                    "transmit_duration_nominal",
                    "slope",
                    "transmit_power",
                    "frequency_start",
                    "frequency_end",
                ]
            else:
                tx_param_names = [
                    "transmit_duration_nominal",
                    "slope",
                    "transmit_power",
                    "frequency",
                ]
            tx_params = {}
            for p in tx_param_names:
                tx_params[p] = np.unique(self.echodata.beam[p].sel(frequency=freq))
                if tx_params[p].size != 1:
                    raise TypeError("File contains changing %s!" % p)
            y_tmp, _ = self._tapered_chirp(**tx_params)

            # Filter and decimate chirp template
            channel_id = str(
                self.echodata.beam.sel(frequency=freq)["channel_id"].values
            )
            fs_deci = (
                1 / self.echodata.beam.sel(frequency=freq)["sample_interval"].values
            )
            y_tmp, y_tmp_time = self._filter_decimate_chirp(y_tmp, channel_id)

            # Compute effective pulse length
            tau_effective_tmp = self._get_tau_effective(
                y_tmp, fs_deci, waveform_mode=waveform_mode
            )

            y_all[channel_id] = y_tmp
            y_time_all[channel_id] = y_tmp_time
            tau_effective[channel_id] = tau_effective_tmp

        return y_all, y_time_all, tau_effective

    def compress_pulse(self, chirp, freq_BB=None):
        """Perform pulse compression on the backscatter data.

        Parameters
        ----------
        chirp : dict
            transmit chirp replica indexed by channel_id
        freq_BB : int or float
            frequency channels that transmit in BB mode
            (since CW mode can be in mixed in complex samples too)
        """
        backscatter = self.echodata.beam["backscatter_r"].sel(
            frequency=freq_BB
        ) + 1j * self.echodata.beam["backscatter_i"].sel(frequency=freq_BB)

        pc_all = []
        for freq in freq_BB:
            backscatter_freq = (
                backscatter.sel(frequency=freq)
                .dropna(dim="range_bin", how="all")
                .dropna(dim="quadrant", how="all")
                .dropna(dim="ping_time")
            )
            channel_id = str(
                self.echodata.beam.sel(frequency=freq)["channel_id"].values
            )
            replica = xr.DataArray(np.conj(chirp[channel_id]), dims="window")
            # Pulse compression via rolling
            pc = (
                backscatter_freq.rolling(range_bin=replica.size)
                .construct("window")
                .dot(replica)
                / np.linalg.norm(chirp[channel_id]) ** 2
            )
            # Expand dimension and add name to allow merge
            pc = pc.expand_dims(dim="frequency")
            pc.name = "pulse_compressed_output"
            pc_all.append(pc)

        pc_merge = xr.merge(pc_all)

        return pc_merge

    def _get_gain_for_complex(self, waveform_mode, freq_center) -> xr.DataArray:
        """Get gain factor for calibrating complex samples.

        Use values from ``gain_correction`` in the Vendor group for CW mode samples,
        or interpolate ``gain`` to the center frequency of each ping for BB mode samples
        if nominal frequency is within the calibrated frequencies range

        Parameters
        ----------
        waveform_mode : str
            ``CW`` for CW-mode samples, either recorded as complex or power samples
            ``BB`` for BB-mode samples, recorded as complex samples
        freq_center : xr.DataArray
            Nominal channel frequency for CW mode samples
            and an xr.DataArray with coorindate ``frequency`` and ``ping_time`` for BB mode samples

        Returns
        -------
        An xr.DataArray
        """
        if waveform_mode == "BB":
            gain_single = self._get_vend_cal_params_power(
                "gain_correction", waveform_mode=waveform_mode
            )
            gain = []
            if "gain" in self.echodata.vendor.data_vars:
                # index using channel_id as order of frequency across channel can be arbitrary
                # reference to freq_center in case some channels are CW complex samples
                # (already dropped when computing freq_center in the calling function)
                for fn in freq_center.frequency:
                    ch_id = self.echodata.beam.channel_id.sel(frequency=fn)
                    # if freq-dependent gain exists in data
                    if ch_id in self.echodata.vendor.cal_channel_id:
                        gain_vec = self.echodata.vendor.gain.sel(cal_channel_id=ch_id)
                        gain_temp = (
                            gain_vec.interp(
                                cal_frequency=freq_center.sel(frequency=fn)
                            ).drop(["cal_channel_id", "cal_frequency"])
                        ).expand_dims("frequency")
                    # if no freq-dependent gain use CW gain
                    else:
                        gain_temp = (
                            gain_single.sel(frequency=fn)
                            .assign_coords(ping_time=np.datetime64(0, "ns"))
                            .expand_dims("ping_time")
                            .reindex_like(
                                self.echodata.beam.backscatter_r, method="nearest"
                            )
                            .expand_dims("frequency")
                        )
                    gain_temp.name = "gain"
                    gain.append(gain_temp)
                gain = xr.merge(gain).gain  # select the single data variable
            else:
                gain = gain_single
        elif waveform_mode == "CW":
            gain = self._get_vend_cal_params_power(
                "gain_correction", waveform_mode=waveform_mode
            ).sel(frequency=freq_center.frequency)

        return gain

    def _cal_complex(self, cal_type, waveform_mode) -> xr.Dataset:
        """Calibrate complex data from EK80.

        Parameters
        ----------
        cal_type : str
            'Sv' for calculating volume backscattering strength, or
            'Sp' for calculating point backscattering strength
        waveform_mode : {"CW", "BB"}
            Type of transmit waveform.

            - `"CW"` for narrowband transmission,
              returned echoes recorded either as complex or power/angle samples
            - `"BB"` for broadband transmission,
              returned echoes recorded as complex samples

        Returns
        -------
        xr.Dataset
            The calibrated dataset containing Sv or Sp
        """
        # Transmit replica and effective pulse length
        chirp, _, tau_effective = self.get_transmit_chirp(waveform_mode=waveform_mode)

        # use center frequency for each ping to select BB or CW channels
        # when all samples are encoded as complex samples
        if (
            "frequency_start" in self.echodata.beam
            and "frequency_end" in self.echodata.beam
        ):
            freq_center = (
                self.echodata.beam["frequency_start"]
                + self.echodata.beam["frequency_end"]
            ) / 2
        else:
            freq_center = None

        if waveform_mode == "BB":
            if freq_center is None:
                raise ValueError(
                    "frequency_start and frequency_end should exist in BB mode data, "
                    "double check the EchoData object!"
                )
            # if CW and BB complex samples co-exist
            # drop those that contain CW samples (nan in freq start/end)
            freq_sel = freq_center.dropna(dim="frequency")

            # backscatter data
            pc = self.compress_pulse(chirp, freq_BB=freq_sel.frequency)
            prx = (
                self.echodata.beam.quadrant.size
                * np.abs(pc.mean(dim="quadrant")) ** 2
                / (2 * np.sqrt(2)) ** 2
                * (np.abs(self.z_er + self.z_et) / self.z_er) ** 2
                / self.z_et
            )
        else:
            if freq_center is None:
                # when only have CW complex samples
                freq_sel = self.echodata.beam.frequency
            else:
                # if BB and CW complex samples co-exist
                # drop those that contain BB samples (not nan in freq start/end)
                freq_sel = freq_center.where(np.isnan(freq_center), drop=True).frequency

            # backscatter data
            backscatter_cw = (
                self.echodata.beam["backscatter_r"]
                + 1j * self.echodata.beam["backscatter_i"]
            )
            prx = (
                self.echodata.beam.quadrant.size
                * np.abs(backscatter_cw.mean(dim="quadrant")) ** 2
                / (2 * np.sqrt(2)) ** 2
                * (np.abs(self.z_er + self.z_et) / self.z_er) ** 2
                / self.z_et
            )
            prx.name = "received_power"
            prx = prx.to_dataset()

        # derived params
        sound_speed = self.env_params["sound_speed"].squeeze()
        absorption = (
            self.env_params["sound_absorption"]
            .sel(frequency=freq_sel.frequency)
            .squeeze()
        )
        range_meter = self.range_meter.sel(frequency=freq_sel.frequency).squeeze()
        if waveform_mode == "BB":
            # use true center frequency for BB pulse
            wavelength = sound_speed / freq_sel

            # use true center frequency to interpolate for gain factor
            gain = self._get_gain_for_complex(
                waveform_mode=waveform_mode, freq_center=freq_sel
            )

        else:
            # use nominal channel frequency for CW pulse
            wavelength = sound_speed / freq_sel

            # use nominal channel frequency to select gain factor
            gain = self._get_gain_for_complex(
                waveform_mode=waveform_mode, freq_center=freq_sel
            )

        # Transmission loss
        spreading_loss = 20 * np.log10(range_meter.where(range_meter >= 1, other=1))
        absorption_loss = 2 * absorption * range_meter

        # TODO: both Sv and Sp are off by ~<0.5 dB from matlab outputs.
        #  Is this due to the use of 'single' in matlab code?
        if cal_type == "Sv":
            # effective pulse length
            tau_effective = xr.DataArray(
                data=list(tau_effective.values()),
                coords=[self.echodata.beam.frequency, self.echodata.beam.ping_time],
                dims=["frequency", "ping_time"],
            ).sel(frequency=freq_sel.frequency)

            # other params
            transmit_power = self.echodata.beam["transmit_power"].sel(
                frequency=freq_sel.frequency
            )
            if waveform_mode == "BB":
                psifc = self.echodata.beam["equivalent_beam_angle"].sel(
                    frequency=freq_sel.frequency
                ) + 10 * np.log10(freq_sel.frequency / freq_center)
            elif waveform_mode == "CW":
                psifc = self.echodata.beam["equivalent_beam_angle"].sel(
                    frequency=freq_sel.frequency
                )

            out = (
                10 * np.log10(prx)
                + spreading_loss
                + absorption_loss
                - 10
                * np.log10(
                    wavelength**2 * transmit_power * sound_speed / (32 * np.pi**2)
                )
                - 2 * gain
                - 10 * np.log10(tau_effective)
                - psifc
            )
            out = out.rename_vars({list(out.data_vars.keys())[0]: "Sv"})

        elif cal_type == "Sp":
            transmit_power = self.echodata.beam["transmit_power"].sel(
                frequency=freq_sel.frequency
            )

            out = (
                10 * np.log10(prx)
                + 2 * spreading_loss
                + absorption_loss
                - 10 * np.log10(wavelength**2 * transmit_power / (16 * np.pi**2))
                - 2 * gain
            )
            out = out.rename_vars({list(out.data_vars.keys())[0]: "Sp"})

        # Attach calculated range (with units meter) into data set
        out = out.merge(range_meter)

        # Add env and cal parameters
        out = self._add_params_to_output(out)

        return out

    def _compute_cal(self, cal_type, waveform_mode, encode_mode) -> xr.Dataset:
        """
        Private method to compute Sv or Sp from EK80 data, called by compute_Sv or compute_Sp.

        Parameters
        ----------
        cal_type : str
            'Sv' for calculating volume backscattering strength, or
            'Sp' for calculating point backscattering strength

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
            An xarray Dataset containing either Sv or Sp.
        """
        # Raise error for wrong inputs
        if waveform_mode not in ("BB", "CW"):
            raise ValueError(
                "Input waveform_mode not recognized! "
                "waveform_mode must be either 'BB' or 'CW' for EK80 data."
            )
        if encode_mode not in ("complex", "power"):
            raise ValueError(
                "Input encode_mode not recognized! "
                "encode_mode must be either 'complex' or 'power' for EK80 data."
            )

        # Set flag_complex
        #  - True: complex cal
        #  - False: power cal
        # BB: complex only, CW: complex or power
        if waveform_mode == "BB":
            if encode_mode == "power":  # BB waveform forces to collect complex samples
                raise ValueError(
                    "encode_mode='power' not allowed when waveform_mode='BB'!"
                )
            flag_complex = True
        else:  # waveform_mode="CW"
            if encode_mode == "complex":
                flag_complex = True
            else:
                flag_complex = False

        # Raise error when waveform_mode and actual recording mode do not match
        # This simple check is only possible for BB-only data,
        #   since for data with both BB and CW complex samples,
        #   frequency_start will exist in echodata.beam for the BB channels
        if waveform_mode == "BB" and "frequency_start" not in self.echodata.beam:
            raise ValueError("waveform_mode='BB' but broadband data not found!")

        # Set use_beam_power
        #  - True: use self.echodata.beam_power for cal
        #  - False: use self.echodata.beam for cal
        use_beam_power = False

        # Warn user about additional data in the raw file if another type exists
        # When both power and complex samples exist:
        #   complex samples will be stored in echodata.beam
        #   power samples will be stored in echodata.beam_power
        # When only one type of samples exist,
        #   all samples with be stored in echodata.beam
        if self.echodata.beam_power is not None:  # both power and complex samples exist
            # If both beam and beam_power groups exist,
            #   this means that CW data are encoded as power samples and in beam_power group
            if waveform_mode == "CW" and encode_mode == "complex":
                raise ValueError("File does not contain CW complex samples")

            if encode_mode == "power":
                use_beam_power = True  # switch source of backscatter data
                print(
                    "Only power samples are calibrated, but complex samples also exist in the raw data file!"  # noqa
                )
            else:
                print(
                    "Only complex samples are calibrated, but power samples also exist in the raw data file!"  # noqa
                )
        else:  # only power OR complex samples exist
            if (
                "quadrant" in self.echodata.beam.dims
            ):  # data contain only complex samples
                if encode_mode == "power":
                    raise TypeError(
                        "File does not contain power samples! Use encode_mode='complex'"
                    )  # user selects the wrong encode_mode
            else:  # data contain only power samples
                if encode_mode == "complex":
                    raise TypeError(
                        "File does not contain complex samples! Use encode_mode='power'"
                    )  # user selects the wrong encode_mode

        # Compute Sv
        if flag_complex:
            # Complex samples can be BB or CW
            self.compute_range_meter(
                waveform_mode=waveform_mode, encode_mode=encode_mode
            )
            ds_cal = self._cal_complex(cal_type=cal_type, waveform_mode=waveform_mode)
        else:
            # Power samples only make sense for CW mode data
            self.compute_range_meter(waveform_mode="CW", encode_mode=encode_mode)
            ds_cal = self._cal_power(cal_type=cal_type, use_beam_power=use_beam_power)

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
            and the corresponding range (``range``) in units meter.
        """
        return self._compute_cal(
            cal_type="Sv", waveform_mode=waveform_mode, encode_mode=encode_mode
        )

    def compute_Sp(self, waveform_mode="BB", encode_mode="complex"):
        """Compute point backscattering strength (Sp).

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
        Sp : xr.DataSet
            A DataSet containing point backscattering strength (``Sp``)
            and the corresponding range (``range``) in units meter.
        """
        return self._compute_cal(
            cal_type="Sp", waveform_mode=waveform_mode, encode_mode=encode_mode
        )
