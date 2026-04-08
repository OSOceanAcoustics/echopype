import numpy as np
import xarray
from scipy.interpolate import LinearNDInterpolator

from ..echodata import EchoData
from ..utils.log import _init_logger
from .cal_params import get_cal_params_AZFP
from .calibrate_ek import CalibrateBase
from .env_params import get_env_params_AZFP
from .range import compute_range_AZFP

logger = _init_logger(__name__)

# Common Sv_offset values for frequency > 38 kHz
SV_OFFSET_HF = {
    150: 1.4,
    200: 1.4,
    250: 1.3,
    300: 1.1,
    500: 0.8,
    700: 0.5,
    900: 0.3,
    1000: 0.3,
}
SV_OFFSET_LF = {
    500: 1.1,
    1000: 0.7,
}
SV_OFFSET = {
    38000.0: {**SV_OFFSET_LF},
    67000.0: {
        **SV_OFFSET_HF,
        500: 1.1,
    },
    120000.0: {
        **SV_OFFSET_HF,
        150: 1.4,
        250: 1.3,
    },
    125000.0: {
        **SV_OFFSET_HF,
        150: 1.4,
        250: 1.3,
    },
    130000.0: {
        **SV_OFFSET_HF,
        150: 1.4,
        250: 1.3,
    },
    200000.0: {
        **SV_OFFSET_HF,
        150: 1.4,
        250: 1.3,
    },
    417000.0: {
        **SV_OFFSET_HF,
    },
    455000.0: {
        **SV_OFFSET_HF,
        250: 1.3,
    },
    769000.0: {
        **SV_OFFSET_HF,
        150: 1.4,
    },
}


def _calc_azfp_Sv_offset(
    frequency: float | int,
    pulse_length: float | int,
) -> float:
    """

    Based on code provided by @IanTBlack

    Linearly interpolate an SV offset for a given pulse length and frequency.
    The predefined offsets used in this function can be found in Table 3 of
    the AZFP Operator's Manual.

    :param pulse_length: A pulse length value (in microseconds) from an
        AZFP config file for a given frequency channel.
    :param frequency: The AZFP frequency channel.
    :return: Either the known Sv offset if it is predefined, or an interpolated
        value for less common pulse lengths.
    """

    pulse_length = int(pulse_length)  # Convert to an integer for consistency.
    frequency = int(frequency)

    # Check if the specified freq is known values
    if frequency in SV_OFFSET.keys() and pulse_length in SV_OFFSET[frequency]:
        return SV_OFFSET[frequency][pulse_length]

    # convert dict to a grid
    xs, ys, zs = [], [], []
    for x_val, inner in SV_OFFSET.items():  # frequencies
        for y_val, z_val in inner.items():  # phases
            xs.append(x_val)
            ys.append(y_val)
            zs.append(z_val)  # known offsets
    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    points = np.column_stack([xs, ys])
    interp = LinearNDInterpolator(points, zs)
    zq = interp(frequency, pulse_length)

    # Check if outside of calibration grid
    if np.isnan(zq):
        raise ValueError(
            f"Pulse lengths less than 150 or greater than 1000 usecs for {frequency}kHz "
            "are not supported. Set cal_params={'Sv_offset' : VALUES} "
            "to provide your own Sv_offset."
        )

    return np.round(zq, 1)


class CalibrateAZFP(CalibrateBase):
    def __init__(
        self, echodata: EchoData, env_params=None, cal_params=None, ecs_file=None, **kwargs
    ):
        super().__init__(echodata, env_params, cal_params, ecs_file)

        # Set sonar_type
        self.sonar_type = "AZFP"  # ULS5 and ULS6 use the same calculations currently

        # Screen for ECS file: currently not support
        if self.ecs_file is not None:
            raise ValueError("Using ECS file for calibration is not currently supported for AZFP!")

        # load env and cal parameters
        self.env_params = get_env_params_AZFP(echodata=self.echodata, user_dict=self.env_params)
        self.cal_params = get_cal_params_AZFP(
            beam=self.echodata["Sonar/Beam_group1"],
            vend=self.echodata["Vendor_specific"],
            user_dict=self.cal_params,
        )

        # self.range_meter computed under self._cal_power_samples()
        # because the implementation is different for Sv and TS
        self.compute_Sv_offset()

    def compute_Sv_offset(self):
        """
        If Sv_offset isn't in cal_params or the echodata then calculate it.
        """
        if self.cal_params["Sv_offset"] is None:
            Sv_offset = []
            for freq, pulse_len in zip(
                self.echodata["Vendor_specific"]["frequency_nominal"].values,
                self.echodata["Vendor_specific"]["XML_transmit_duration_nominal"].values[0],
            ):
                try:
                    Sv_offset.append(_calc_azfp_Sv_offset(freq, pulse_len * 1e6))
                except ValueError:
                    logger.warning(
                        f"The Sv for {freq}kHz and pulse length {pulse_len}us "
                        "is uncalibrated (Sv_offset=0.0)"
                    )
                    Sv_offset.append(0.0)

            Sv_offset = xarray.DataArray(
                Sv_offset,
                coords=[("channel", self.echodata["Vendor_specific"].coords["channel"].values)],
                dims=["channel"],
                name="Sv_offset",
            )
            Sv_offset.channel.attrs = (
                self.echodata["Vendor_specific"].coords["channel"].attrs.copy()
            )
            self.cal_params["Sv_offset"] = Sv_offset

    def compute_echo_range(self, cal_type):
        """Calculate range (``echo_range``) in meter using AZFP formula.

        Note the range calculation differs for Sv and TS per AZFP matlab code.

        Parameters
        ----------
        cal_type : str
            'Sv' for calculating volume backscattering strength, or
            'TS' for calculating target strength
        """
        self.range_meter = compute_range_AZFP(
            echodata=self.echodata, env_params=self.env_params, cal_type=cal_type
        )

    def _cal_power_samples(self, cal_type, **kwargs):
        """Calibrate to get volume backscattering strength (Sv) from AZFP power data.

        The calibration formulae used here is based on Appendix G in
        the GU-100-AZFP-01-R50 Operator's Manual.
        Note a Sv_offset factor that varies depending on frequency is used
        in the calibration as documented on p.90.
        """

        # Compute range in meters
        # range computation different for Sv and TS per AZFP matlab code
        self.compute_echo_range(cal_type=cal_type)

        # Compute derived params
        # TODO: take care of dividing by zero encountered in log10
        spreading_loss = 20 * np.log10(self.range_meter)
        absorption_loss = 2 * self.env_params["sound_absorption"] * self.range_meter
        SL = self.cal_params["TVR"] + 20 * np.log10(self.cal_params["VTX0"])  # eq.(2)

        # scaling factor (slope) in Fig.G-1, units Volts/dB], see p.84
        a = self.cal_params["DS"]
        EL = (
            self.cal_params["EL"]
            - 2.5 / a
            + self.echodata["Sonar/Beam_group1"]["backscatter_r"] / (26214 * a)
        )  # eq.(5)

        if cal_type == "Sv":
            # eq.(9)
            out = (
                EL
                - SL
                + spreading_loss
                + absorption_loss
                - 10
                * np.log10(
                    0.5
                    * self.env_params["sound_speed"]
                    * self.echodata["Sonar/Beam_group1"]["transmit_duration_nominal"]
                    * self.cal_params["equivalent_beam_angle"]
                )
                + self.cal_params["Sv_offset"]
            )  # see p.90-91 for this correction to Sv
            out.name = "Sv"

        elif cal_type == "TS":
            # eq.(10)
            out = EL - SL + 2 * spreading_loss + absorption_loss
            out.name = "TS"
        else:
            raise ValueError("cal_type not recognized!")

        # Attach calculated range (with units meter) into data set
        out = out.to_dataset()
        out = out.merge(self.range_meter)

        # Add frequency_nominal to data set
        out["frequency_nominal"] = self.echodata["Sonar/Beam_group1"]["frequency_nominal"]

        # Add env and cal parameters
        out = self._add_params_to_output(out)

        return out

    def compute_Sv(self, **kwargs):
        return self._cal_power_samples(cal_type="Sv")

    def compute_TS(self, **kwargs):
        return self._cal_power_samples(cal_type="TS")
