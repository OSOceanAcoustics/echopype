from datetime import datetime as dt
import numpy as np
import xarray as xr
from ..utils import uwa
from ..utils import io


class ProcessBase:
    """Class for processing sonar data.
    """
    def __init__(self, model=None):
        self.sonar_model = model   # type of echosounder

    def calc_sound_speed(self, ed, env_params=None, src='user', formula_source='Mackenzie'):
        """Base method for calculating sound speed.
        """
        if src == 'file':
            with xr.open_dataset(ed.raw_path, group="Environment", engine=ed._file_format) as ds_env:
                if 'sound_speed_indicative' in ds_env:
                    return ds_env.sound_speed_indicative
                else:
                    ValueError("Sound speed not found in file")
        elif src == 'user':
            if env_params is None:
                raise ValueError("`env_params` required for calculating sound speed")
            ss = uwa.calc_sound_speed(salinity=env_params['water_salinity'],
                                      temperature=env_params['water_temperature'],
                                      pressure=env_params['water_pressure'],
                                      formula_source=formula_source)
            return ss
        else:
            ValueError("Not sure how to update sound speed!")

    def calc_absorption(self, ed, env_params, src, formula_source='FG'):
        """Base method for calculating water absorption.
        """
        if src != 'user':
            raise ValueError("'src' can only be 'user'")
        freq = ed.raw.frequency.astype(np.int64)  # should already be in unit [Hz]
        return uwa.calc_absorption(freq,
                                   temperature=env_params['water_temperature'],
                                   salinity=env_params['water_salinity'],
                                   pressure=env_params['water_pressure'],
                                   formula_source=formula_source)

    def calc_sample_thickness(self, ed):
        """Base method for calculating sample thickness.
        This method is only used for EK echosounders.
        """

    def calc_range(self, ed):
        """Base method for calculating range.

        Parameters
        ----------
        ed : EchoDataBase
        """

    def get_Sv(self, ed, env_params, cal_params, save=True, save_format='zarr'):
        """Base method to be overridden for calculating Sv from raw backscatter data.
        """
        # Issue warning when subclass methods not available
        print('Calibration has not been implemented for this sonar model!')

    def get_TS(self, ed, env_params, cal_params, save=True, save_format='zarr'):
        """Base method to be overridden for calculating TS from raw backscatter data.
        """
        # Issue warning when subclass methods not available
        print('Calibration has not been implemented for this sonar model!')

    def _get_tile_params(self, ed, da, env_params, cal_params, proc_params):
        # Get number of pings per tile
        if proc_params['MVBS_time_interval'] is not None:
            print("Averaging by time interval is not yet implemented")
            return
        elif proc_params['MVBS_ping_num'] is not None:
            pings_per_tile = proc_params['MVBS_ping_num']
        else:
            raise ValueError("No ping tile size provided")

        # Get number of range_bins per tile
        if proc_params['MVBS_distance_interval'] is not None:
            # TODO MVBS_distance_interval: can use .groupby().mean(),
            # based on distance calculated by lat/lon from Platform group,
            print("Averaging by distance interval is not yet implemented")
            return
        elif proc_params['MVBS_range_interval'] is not None:
            print("Averaging by range inteval is not yet implemented")
            return
        elif proc_params['MVBS_range_bin_num'] is not None:
            range_bins_per_tile = proc_params['MVBS_range_bin_num']
        else:
            raise ValueError("No range_bin tile size provided")

        return pings_per_tile, range_bins_per_tile

    def get_MVBS(self, ed, env_params, cal_params, proc_params,
                 save=True, save_path=None, save_format='zarr'):
        """Calculate Mean Volume Backscattering Strength (MVBS).

        The calculation uses class attributes MVBS_ping_size and MVBS_range_bin_size to
        calculate and save MVBS as a new attribute to the calling Process instance.
        MVBS is an xarray DataArray with dimensions ``ping_time`` and ``range_bin``
        that are from the first elements of each tile along the corresponding dimensions
        in the original Sv or Sv_clean DataArray.
        """
        #  - MVBS_source: 'Sv' or 'Sv_cleaned'
        #  - MVBS_type: 'binned' or 'rolling'
        #               so far we've only had binned averages (what coarsen is doing)
        #               let's add the functionality to use rolling
        #  - MVBS_ping_num or MVBS_time_interval (one of them has to be given)
        #     - MVBS_ping_num:
        #     - MVBS_time_interval: can use .groupby/resample().mean() or .rolling().mean(),
        #                           based on ping_time
        #       ?? x.resample(time='30s').mean()
        #     - MVBS_distance_interval: can use .groupby().mean(),
        #                               based on distance calculated by lat/lon from Platform group,
        #                               let's put this the last to add
        #  - MVBS_range_bin_num or MVBS_range_interval (use left close right open intervals for now)
        #     - MVBS_range_bin_num:
        #     - MVBS_range_interval: can use .groupby.resample().mean() or .rolling().mean(),
        #                            based on the actual range in meter

        # Check to see if the source exists. If it does, convert to linear domain
        if proc_params['MVBS_source'] in ['Sv', 'Sv_clean']:
            if getattr(ed, proc_params['MVBS_source']) is not None:
                Sv_linear = 10 ** (getattr(ed, proc_params['MVBS_source']).Sv / 10)
            else:
                if proc_params['MVBS_source'] == 'Sv':
                    raise ValueError("Sv data has not been found. Please calibrate with get_Sv")
                else:
                    raise ValueError("Sv_clean data has not been found. Please clean Sv data with remove_noise")
        else:
            raise ValueError("MVBS_source must be either Sv or Sv_clean")

        pings_per_tile, range_bins_per_tile = self._get_tile_params(ed, Sv_linear, env_params,
                                                                    cal_params, proc_params)
        if proc_params['MVBS_type'] == 'binned':
            MVBS = Sv_linear.coarsen(
                ping_time=pings_per_tile,
                range_bin=range_bins_per_tile,
                boundary='pad', keep_attrs=True).mean()
        elif proc_params['MVBS_type'] == 'rolling':
            # TODO: likely bad. Look into memory usage of rolling
            # Assuming file size 100 mb and RAM 4 gb. Limits the memory usage when rolling
            if pings_per_tile * range_bins_per_tile > 40:
                Sv_linear = Sv_linear.load()
            MVBS = Sv_linear.rolling(ping_time=pings_per_tile,
                                     range_bin=range_bins_per_tile).mean(keep_attrs=True)
        else:
            raise ValueError("MVBS_type must be either binned or rolling")
        # Convert to log domain
        MVBS = 10 * np.log10(MVBS)
        MVBS.name = 'MVBS'
        MVBS = MVBS.to_dataset()

        if save:
            # Update pointer in EchoData
            MVBS_path = io.validate_proc_path(ed, '_MVBS', save_path)
            print(f"{dt.now().strftime('%H:%M:%S')}  saving calibrated TS to {MVBS_path}")
            ed._save_dataset(MVBS, MVBS_path, mode="w", save_format=save_format)
            ed.MVBS_path = MVBS_path
        else:
            ed.MVBS = MVBS

    def remove_noise(self, ed, env_params, proc_params, save=True, save_format='zarr'):
        """Remove noise by using noise estimates obtained from the minimum mean calibrated power level
        along each column of tiles.

        See method noise_estimates() for details of noise estimation.
        Reference: De Robertis & Higginbottom, 2007, ICES Journal of Marine Sciences
        """

    def get_noise_estimates(self):
        """Obtain noise estimates from the minimum mean calibrated power level along each column of tiles.

        The tiles here are defined by class attributes noise_est_range_bin_size and noise_est_ping_size.
        This method contains redundant pieces of code that also appear in method remove_noise(),
        but this method can be used separately to determine the exact tile size for noise removal before
        noise removal is actually performed.
        """

    def db_diff(self, ed, proc_params, save=True, save_format='zarr'):
        """Perform dB-differencing (frequency-differencing) for specified thresholds.
        """


class ProcessEK(ProcessBase):
    """
    Class for processing data from Simrad EK echosounders.
    """
    def __init__(self, model=None):
        super().__init__(model)

    def calc_sa_correction(self, ed):
        ds_vend = ed.get_vend_from_raw()

        if 'sa_correction' not in ds_vend:
            return

        sa_correction_table = ds_vend.sa_correction
        pulse_length_table = ds_vend.pulse_length
        pulse_length = np.unique(ed.raw.transmit_duration_nominal, axis=1).flatten()

        if pulse_length.ndim > 1:
            raise ValueError("Pulse length changes over time")
        idx = [np.argwhere(np.isclose(pulse_length[i], pulse_length_table[i])).squeeze()
               for i in range(pulse_length_table.shape[0])]

        sa_correction = np.array([ch[x] for ch, x in zip(sa_correction_table, np.array(idx))])

        return xr.DataArray(sa_correction, dims='frequency').assign_coords(frequency=sa_correction_table.frequency)

    def calc_sample_thickness(self, ed, env_params, cal_params):
        """Calculate sample thickness.
        """
        return env_params['speed_of_sound_in_water'] * cal_params['sample_interval'] / 2

    def _cal_narrowband(self, ed, env_params, cal_params, cal_type,
                        save=True, save_path=None, save_format='zarr'):
        """Calibrate narrowband data from EK60 and EK80.
        """
        # Derived params
        wavelength = env_params['speed_of_sound_in_water'] / ed.raw.frequency  # wavelength
        if ed.range is None:
            ed.range = self.calc_range(ed, env_params, cal_params)
        # Get TVG and absorption
        TVG = np.real(20 * np.log10(ed.range.where(ed.range >= 1, other=1)))
        ABS = 2 * env_params['absorption'] * ed.range
        if cal_type == 'Sv':
            # Print raw data nc file

            # Calc gain
            CSv = 10 * np.log10((cal_params['transmit_power'] * (10 ** (cal_params['gain_correction'] / 10)) ** 2 *
                                wavelength ** 2 * env_params['speed_of_sound_in_water'] *
                                cal_params['transmit_duration_nominal'] *
                                10 ** (cal_params['equivalent_beam_angle'] / 10)) /
                                (32 * np.pi ** 2))

            # Calibration and echo integration
            Sv = ed.raw.backscatter_r + TVG + ABS - CSv - 2 * cal_params['sa_correction']
            Sv.name = 'Sv'
            Sv = Sv.to_dataset()

            # Attach calculated range into data set
            Sv['range'] = (('frequency', 'ping_time', 'range_bin'), ed.range)

            # Save calibrated data into the calling instance and
            #  to a separate .nc file in the same directory as the data filef.Sv = Sv
            if save:
                # Update pointer in EchoData
                Sv_path = io.validate_proc_path(ed, '_Sv', save_path)
                print(f"{dt.now().strftime('%H:%M:%S')}  saving calibrated Sv to {Sv_path}")
                ed._save_dataset(Sv, Sv_path, mode="w", save_format=save_format)
                ed.Sv_path = Sv_path
            else:
                ed.Sv = Sv
        elif cal_type == 'TS':
            # calculate TS
            # Open data set for Environment and Beam groups

            # Calc gain
            CSp = 10 * np.log10((cal_params['transmit_power'] *
                                (10 ** (cal_params['gain_correction'] / 10)) ** 2 *
                                wavelength ** 2) / (16 * np.pi ** 2))

            # Calibration and echo integration
            TS = ed.raw.backscatter_r + TVG * 2 + ABS - CSp
            TS.name = 'TS'
            TS = TS.to_dataset()

            # Attach calculated range into data set
            TS['range'] = (('frequency', 'ping_time', 'range_bin'), ed.range)

            if save:
                # Update pointer in EchoData
                TS_path = io.validate_proc_path(ed, '_TS', save_path)
                print(f"{dt.now().strftime('%H:%M:%S')}  saving calibrated TS to {TS_path}")
                ed._save_dataset(TS, TS_path, mode="w", save_format=save_format)
                ed.TS_path = TS_path
            else:
                ed.TS = TS
