from .process_base import ProcessEK


class ProcessEK60(ProcessEK):
    """
    Class for processing data from Simrad EK60 echosounder.
    """
    def __init__(self, model='EK60'):
        super().__init__(model)

        self.tvg_correction_factor = 2  # EK60 specific parameter

    def get_Sv(self, ed, env_params, cal_params=None, save=True, save_path=None, save_format='zarr'):
        """Calibrate to get volume backscattering strength (Sv) from EK60 data.
        """
        return self._cal_narrowband(ed=ed,
                                    env_params=env_params,
                                    cal_params=cal_params,
                                    cal_type='Sv',
                                    save=save,
                                    save_path=save_path,
                                    save_format=save_format)

    def get_Sp(self, ed, env_params, cal_params=None, save=True, save_path=None, save_format='zarr'):
        """Calibrate to get point backscattering strength (Sp) from EK60 data.
        """
        return self._cal_narrowband(ed=ed,
                                    env_params=env_params,
                                    cal_params=cal_params,
                                    cal_type='Sp',
                                    save=save,
                                    save_path=save_path,
                                    save_format=save_format)

    def calc_range(self, ed, env_params, cal_params):
        """Calculates range in meters.
        """
        sample_thickness = self.calc_sample_thickness(ed, env_params, cal_params)
        range_meter = (ed.raw.range_bin
                       - self.tvg_correction_factor - 1/2) * sample_thickness  # DataArray [frequency x range_bin]
        range_meter = range_meter.where(range_meter > 0, other=0)
        return range_meter
