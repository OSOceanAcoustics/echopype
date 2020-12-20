from datetime import datetime as dt
import numpy as np
import xarray as xr
from ..utils import io
from .process_base import ProcessBase


class ProcessAZFP(ProcessBase):
    """
    Class for processing data from ASL Env Sci AZFP echosounder.
    """
    def __init__(self, model='AZFP'):
        super().__init__(model)

    def calc_range(self, ed, env_params, tilt_corrected=False):
        """Calculates range in meters using AZFP formula,
        instead of from sample_interval directly.
        """

        ds_vend = ed.get_vend_from_raw()

        # WJ: same as "range_samples_per_bin" used to calculate "sample_interval"
        range_samples = ds_vend.number_of_samples_per_average_bin
        pulse_length = ed.raw.transmit_duration_nominal   # units: seconds
        bins_to_avg = 1   # set to 1 since we want to calculate from raw data
        sound_speed = env_params['speed_of_sound_in_water']
        dig_rate = ds_vend.digitization_rate
        lockout_index = ds_vend.lockout_index

        # Below is from LoadAZFP.m, the output is effectively range_bin+1 when bins_to_avg=1
        range_mod = xr.DataArray(np.arange(1, len(ed.raw.range_bin) - bins_to_avg + 2, bins_to_avg),
                                 coords=[('range_bin', ed.raw.range_bin)])

        # Calculate range using parameters for each freq
        range_meter = (lockout_index / (2 * dig_rate) * sound_speed + sound_speed / 4 *
                       (((2 * range_mod - 1) * range_samples * bins_to_avg - 1) / dig_rate +
                        pulse_length))

        if tilt_corrected:
            range_meter = ed.raw.cos_tilt_mag.mean() * range_meter

        ds_vend.close()

        return range_meter

    def get_Sv(self, ed, env_params, cal_params, save=True, save_path=None, save_format='zarr'):
        """Calibrate to get volume backscattering strength (Sv) from AZFP power data.

        The calibration formula used here is documented in eq.(9) on p.85
        of GU-100-AZFP-01-R50 Operator's Manual.
        Note a Sv_offset factor that varies depending on frequency is used
        in the calibration as documented on p.90.
        See calc_Sv_offset() in convert/azfp.py
        """
        if ed.range is None:
            ed.range = self.calc_range(ed, env_params)
        Sv = (cal_params['EL'] - 2.5 / cal_params['DS'] +
              ed.raw.backscatter_r / (26214 * cal_params['DS']) -
              cal_params['TVR'] - 20 * np.log10(cal_params['VTX']) + 20 * np.log10(ed.range) +
              2 * env_params['seawater_absorption'] * ed.range -
              10 * np.log10(0.5 * env_params['speed_of_sound_in_water'] *
                            cal_params['transmit_duration_nominal'] *
                            cal_params['equivalent_beam_angle']) + cal_params['Sv_offset'])

        Sv.name = 'Sv'
        Sv = Sv.to_dataset()

        # Attached calculated range into the dataset
        Sv['range'] = (('frequency', 'ping_time', 'range_bin'), ed.range)

        # Save calibrated data into the calling instance and
        # to a separate .nc file in the same directory as the data
        if save:
            # Update pointer in EchoData
            Sv_path = io.validate_proc_path(ed, '_Sv', save_path)
            print(f"{dt.now().strftime('%H:%M:%S')}  saving calibrated Sv to {Sv_path}")
            ed._save_dataset(Sv, Sv_path, mode="w", save_format=save_format)
            ed.Sv_path = Sv_path
        else:
            # TODO Add to docs
            ed.Sv = Sv

    def get_Sp(self, ed, env_params, cal_params, save=True, save_path=None, save_format='zarr'):
        """Calibrate to get Target Strength (TS) from AZFP power data.
        """
        if ed.range is None:
            ed.range = self.calc_range(ed, env_params)

        TS = (cal_params['EL'] - 2.5 / cal_params['DS'] +
              ed.raw.backscatter_r / (26214 * cal_params['DS']) -
              cal_params['TVR'] - 20 * np.log10(cal_params['VTX']) + 40 * np.log10(ed.range) +
              2 * env_params['seawater_absorption'] * ed.range)

        TS.name = "TS"
        TS = TS.to_dataset()

        # Attached calculated range into the dataset
        TS['range'] = (('frequency', 'ping_time', 'range_bin'), ed.range)

        if save:
            # Update pointer in EchoData
            TS_path = io.validate_proc_path(ed, '_TS', save_path)
            print(f"{dt.now().strftime('%H:%M:%S')}  saving calibrated TS to {TS_path}")
            ed._save_dataset(TS, TS_path, mode="w", save_format=save_format)
            ed.TS_path = TS_path
        else:
            ed.TS = TS
