import numpy as np

from ..echodata import EchoData
from .cal_params import get_cal_params_AZFP
from .calibrate_ek import CalibrateBase
from .env_params import get_env_params_AZFP
from .range import compute_range_AZFP


class CalibrateAZFP(CalibrateBase):
    def __init__(
        self, echodata: EchoData, env_params=None, cal_params=None, ecs_file=None, **kwargs
    ):
        super().__init__(echodata, env_params, cal_params, ecs_file)

        # Set sonar_type
        self.sonar_type = "AZFP"

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
        See calc_Sv_offset() in convert/azfp.py
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
