import numpy as np

from ..echodata import EchoData
from .cal_params import get_cal_params_AZFP
from .calibrate_ek import CalibrateBase
from .env_params_new import get_env_params_AZFP


class CalibrateAZFP(CalibrateBase):
    def __init__(self, echodata: EchoData, env_params=None, cal_params=None, **kwargs):
        super().__init__(echodata, env_params)

        # load env and cal parameters
        self.env_params = get_env_params_AZFP(echodata=echodata, user_env_dict=env_params)

        if cal_params is None:
            cal_params = {}
        self.cal_params = get_cal_params_AZFP(echodata=echodata, user_cal_dict=cal_params)

        # self.range_meter computed under self._cal_power()
        # because the implementation is different for Sv and TS

    def compute_range_meter(self, cal_type):
        """Calculate range (``echo_range``) in meter using AZFP formula.

        Note the range calculation differs for Sv and TS per AZFP matlab code.

        Parameters
        ----------
        cal_type : str
            'Sv' for calculating volume backscattering strength, or
            'TS' for calculating target strength
        """
        self.range_meter = self.echodata.compute_range(self.env_params, azfp_cal_type=cal_type)

    def _cal_power(self, cal_type, **kwargs):
        """Calibrate to get volume backscattering strength (Sv) from AZFP power data.

        The calibration formulae used here is based on Appendix G in
        the GU-100-AZFP-01-R50 Operator's Manual.
        Note a Sv_offset factor that varies depending on frequency is used
        in the calibration as documented on p.90.
        See calc_Sv_offset() in convert/azfp.py
        """
        # Compute range in meters
        self.compute_range_meter(
            cal_type=cal_type
        )  # range computation different for Sv and TS per AZFP matlab code

        # Compute derived params

        # Harmonize time coordinate between Beam_groupX data and env_params
        # Use self.echodata["Sonar/Beam_group1"] because complex sample is always in Beam_group1
        for p in self.env_params.keys():
            self.env_params[p] = self.echodata._harmonize_env_param_time(
                self.env_params[p], ping_time=self.echodata["Sonar/Beam_group1"].ping_time
            )

        # TODO: take care of dividing by zero encountered in log10
        spreading_loss = 20 * np.log10(self.range_meter)
        absorption_loss = 2 * self.env_params["sound_absorption"] * self.range_meter
        SL = self.cal_params["TVR"] + 20 * np.log10(self.cal_params["VTX"])  # eq.(2)

        # scaling factor (slope) in Fig.G-1, units Volts/dB], see p.84
        a = self.cal_params["DS"]
        EL = (
            self.cal_params["EL"]
            - 2.5 / a
            + self.echodata["Sonar/Beam_group1"].backscatter_r / (26214 * a)
        )  # eq.(5)  # has beam dim due to backscatter_r

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

        # Order the dimensions
        out["echo_range"] = out["echo_range"].transpose("channel", "ping_time", "range_sample")

        # Squeeze out the beam dim
        # doing it here because both out and self.cal_params["equivalent_beam_angle"] has beam dim
        return out.squeeze("beam", drop=True)

    def compute_Sv(self, **kwargs):
        return self._cal_power(cal_type="Sv")

    def compute_TS(self, **kwargs):
        return self._cal_power(cal_type="TS")
