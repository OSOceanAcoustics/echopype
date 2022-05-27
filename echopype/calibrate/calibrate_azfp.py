import numpy as np

from ..utils import uwa
from .calibrate_base import CAL_PARAMS
from .calibrate_ek import CalibrateBase


class CalibrateAZFP(CalibrateBase):
    def __init__(self, echodata, env_params, cal_params, **kwargs):
        super().__init__(echodata, env_params)

        # initialize cal params
        self.cal_params = dict.fromkeys(CAL_PARAMS["AZFP"])

        # load env and cal parameters
        self.get_env_params()
        if cal_params is None:
            cal_params = {}
        self.get_cal_params(cal_params)

        # self.range_meter computed under self._cal_power()
        # because the implementation is different for Sv and TS

    def get_cal_params(self, cal_params):
        """Get cal params using user inputs or values from data file.

        Parameters
        ----------
        cal_params : dict
        """

        # Get params from Beam_group1
        self.cal_params["equivalent_beam_angle"] = (
            cal_params["equivalent_beam_angle"]
            if "equivalent_beam_angle" in cal_params
            else self.echodata.beam["equivalent_beam_angle"]
        )

        # Get params from the Vendor_specific group
        for p in ["EL", "DS", "TVR", "VTX", "Sv_offset"]:
            # substitute if None in user input
            self.cal_params[p] = cal_params[p] if p in cal_params else self.echodata.vendor[p]

    def get_env_params(self):
        """Get env params using user inputs or values from data file.

        Parameters
        ----------
        env_params : dict
        """
        # Temperature comes from either user input or data file
        # Below, renaming time1 to ping_time is necessary because we are performing
        # calculations with the beam groups that use ping_time
        self.env_params["temperature"] = (
            self.env_params["temperature"]
            if "temperature" in self.env_params
            else self.echodata.environment["temperature"].rename({"time1": "ping_time"})
        )

        # Salinity and pressure always come from user input
        if ("salinity" not in self.env_params) or ("pressure" not in self.env_params):
            raise ReferenceError("Please supply both salinity and pressure in env_params.")
        else:
            self.env_params["salinity"] = self.env_params["salinity"]
            self.env_params["pressure"] = self.env_params["pressure"]

        # Always calculate sound speed and absorption
        self.env_params["sound_speed"] = uwa.calc_sound_speed(
            temperature=self.env_params["temperature"],
            salinity=self.env_params["salinity"],
            pressure=self.env_params["pressure"],
            formula_source="AZFP",
        )
        self.env_params["sound_absorption"] = uwa.calc_absorption(
            frequency=self.echodata.beam["frequency_nominal"],
            temperature=self.env_params["temperature"],
            salinity=self.env_params["salinity"],
            pressure=self.env_params["pressure"],
            formula_source="AZFP",
        )

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

        # Compute various params

        # TODO: take care of dividing by zero encountered in log10
        spreading_loss = 20 * np.log10(self.range_meter)
        absorption_loss = 2 * self.env_params["sound_absorption"] * self.range_meter
        SL = self.cal_params["TVR"] + 20 * np.log10(self.cal_params["VTX"])  # eq.(2)

        # scaling factor (slope) in Fig.G-1, units Volts/dB], see p.84
        a = self.cal_params["DS"]
        EL = (
            self.cal_params["EL"] - 2.5 / a + self.echodata.beam.backscatter_r / (26214 * a)
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
                    * self.echodata.beam["transmit_duration_nominal"]
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
        out["frequency_nominal"] = self.echodata.beam["frequency_nominal"]

        # Add env and cal parameters
        out = self._add_params_to_output(out)

        # Squeeze out the beam dim
        # doing it here because both out and self.cal_params["equivalent_beam_angle"] has beam dim
        return out.squeeze("beam", drop=True)

    def compute_Sv(self, **kwargs):
        return self._cal_power(cal_type="Sv")

    def compute_TS(self, **kwargs):
        return self._cal_power(cal_type="TS")
