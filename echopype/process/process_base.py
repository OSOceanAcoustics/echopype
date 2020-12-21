from datetime import datetime as dt
import os
import numpy as np
import xarray as xr
from ..utils import uwa
from ..utils import io


class ProcessBase:
    """Class for processing sonar data.
    """
    def __init__(self, model=None):
        self.sonar_model = model   # type of echosounder

    def validate_proc_path(ed, postfix, save_path=None):
        """Creates a directory if it doesn't exist. Returns a valid save path.
        """
        # TODO: It might be better to merge this with convert validate_path
        def _assemble_path():
            file_in = os.path.basename(ed.raw_path[0])
            file_name, file_ext = os.path.splitext(file_in)
            return file_name + postfix + file_ext

        if save_path is None:
            save_dir = os.path.dirname(ed.raw_path[0])
            file_out = _assemble_path()
        else:
            path_ext = os.path.splitext(save_path)[1]
            # If given save_path is file, split into directory and file
            if path_ext != '':
                save_dir, file_out = os.path.split(save_path)
                if save_dir == '':  # save_path is only a filename without directory
                    save_dir = os.path.dirname(ed.raw_path)  # use directory from input file
            # If given save_path is a directory, get a filename from input .nc file
            else:
                save_dir = save_path
                file_out = _assemble_path()

        # Create folder if not already exists
        if save_dir == '':
            save_dir = os.getcwd()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        return os.path.join(save_dir, file_out)

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

    def get_Sp(self, ed, env_params, cal_params, save=True, save_format='zarr'):
        """Base method to be overridden for calculating Sp from raw backscatter data.
        """
        # Issue warning when subclass methods not available
        print('Calibration has not been implemented for this sonar model!')

    def get_MVBS(self, ed, env_params=None, cal_params=None, proc_params=None,
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
        if proc_params['MVBS']['source'] in ['Sv', 'Sv_clean']:
            if getattr(ed, proc_params['MVBS']['source']) is not None:
                Sv_linear = 10 ** (getattr(ed, proc_params['MVBS']['source']).Sv / 10)
            else:
                if proc_params['MVBS']['source'] == 'Sv':
                    raise ValueError("Sv data has not been found. Please calibrate with get_Sv")
                else:
                    raise ValueError("Sv_clean data has not been found. Please clean Sv data with remove_noise")
        else:
            raise ValueError("MVBS_source must be either Sv or Sv_clean")

        if proc_params['MVBS']['type'] == 'binned':
            MVBS = Sv_linear.coarsen(
                ping_time=proc_params['MVBS']['ping_num'],
                range_bin=proc_params['MVBS']['range_bin_num'],
                boundary='pad', keep_attrs=True).mean()
            MVBS.coords['range_bin'] = ('range_bin', np.arange(MVBS['range_bin'].size))
        elif proc_params['MVBS']['type'] == 'rolling':
            # TODO: likely bad. Look into memory usage of rolling
            # Assuming file size 100 mb and RAM 4 gb. Limits the memory usage when rolling
            if proc_params['MVBS']['ping_num'] * proc_params['MVBS']['range_bin_num'] > 40:
                Sv_linear = Sv_linear.load()
            MVBS = Sv_linear.rolling(ping_time=proc_params['MVBS']['ping_num'],
                                     range_bin=proc_params['MVBS']['range_bin_num']).mean(keep_attrs=True)
        else:
            raise ValueError("MVBS_type must be either binned or rolling")
        # Convert to log domain
        MVBS = 10 * np.log10(MVBS)
        MVBS.name = 'MVBS'
        MVBS = MVBS.to_dataset()

        if save:
            # Update pointer in EchoData
            MVBS_path = io.validate_proc_path(ed, '_MVBS', save_path)
            print(f"{dt.now().strftime('%H:%M:%S')}  saving calibrated MVBS to {MVBS_path}")
            ed._save_dataset(MVBS, MVBS_path, mode="w", save_format=save_format)
            ed.MVBS_path = MVBS_path
        else:
            ed.MVBS = MVBS

    def remove_noise(self, ed, env_params, cal_params, proc_params={}, save=True, save_format='zarr'):
        """Remove noise by using noise estimates obtained from the minimum mean calibrated power level
        along each column of tiles.

        See method noise_estimates() for details of noise estimation.
        Reference: De Robertis & Higginbottom, 2007, ICES Journal of Marine Sciences
        """
        # TODO: @leewujung: incorporate an user-specified upper limit of noise level

        if ed.range is None:
            ed.range = self.calc_range(ed, env_params, cal_params)

        # Transmission loss
        spreading_loss = 20 * np.log10(ed.range.where(ed.range >= 1, other=1))
        absorption_loss = 2 * env_params['absorption'] * ed.range

        # Noise estimates
        power_cal = ed.Sv['Sv'] - spreading_loss - absorption_loss  # calibrated power
        power_cal_binned_avg = 10 * np.log10(   # binned averages of calibrated power
            (10 ** (power_cal / 10)).coarsen(
                ping_time=proc_params['noise_est']['ping_num'],
                range_bin=proc_params['noise_est']['range_bin_num'],
                boundary='pad'
            ).mean())
        noise_est = power_cal_binned_avg.min(dim='range_bin')
        noise_est['ping_time'] = power_cal['ping_time'][::proc_params['noise_est']['ping_num']]
        Sv_noise = (noise_est.reindex({'ping_time': power_cal['ping_time']}, method='ffill')  # forward fill empty index
                    + spreading_loss + absorption_loss)

        # Sv corrected for noise
        Sv_corr = 10 * np.log10(10 ** (ed.Sv['Sv'] / 10) - 10 ** (Sv_noise / 10))
        Sv_corr = Sv_corr.where(Sv_corr - Sv_noise > proc_params['noise_est']['SNR'], other=np.nan)

        Sv_corr.name = 'Sv_clean'
        Sv_corr = Sv_corr.to_dataset()

        # Attach calculated range into data set
        Sv_corr['range'] = (('frequency', 'ping_time', 'range_bin'),
                            ed.range.transpose('frequency', 'ping_time', 'range_bin'))

        # Save Sp into the calling instance and
        #  to a separate zarr/nc file in the same directory as the data file
        # TODO: @ngkavin: could you finish the file saving part of this?

    def get_noise_estimates(self, ed, proc_params, save=True, save_format='zarr'):
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
        """ Calculate the sa correction from the sa correction table and the pulse length.
        For EK80 BB mode, there is no sa correction table so the sa correction is saved as none
        """
        ds_vend = ed.get_vend_from_raw()

        if 'sa_correction' not in ds_vend:
            return None

        sa_correction_table = ds_vend.sa_correction
        pulse_length_table = ds_vend.pulse_length
        unique_pulse_length = np.unique(ed.raw.transmit_duration_nominal, axis=1).squeeze()

        if unique_pulse_length.size != ds_vend.frequency.size:
            raise ValueError("Pulse length changes over time")
        idx = [np.argwhere(np.isclose(unique_pulse_length[i], pulse_length_table[i])).squeeze()
               for i in range(pulse_length_table.shape[0])]

        sa_correction = np.array([ch[x] for ch, x in zip(sa_correction_table, np.array(idx))])

        return xr.DataArray(sa_correction, dims='frequency').assign_coords(frequency=sa_correction_table.frequency)

    def calc_sample_thickness(self, ed, env_params, cal_params):
        """Calculate sample thickness.
        """
        return env_params['speed_of_sound_in_water'] * ed.raw.sample_interval / 2

    def _cal_narrowband(self, ed, env_params, cal_params, cal_type,
                        save=True, save_path=None, save_format='zarr'):
        """Calibrate narrowband data from EK60 and EK80.
        """
        # Derived params
        wavelength = env_params['speed_of_sound_in_water'] / ed.raw.frequency  # wavelength
        if ed.range is None:
            ed.range = self.calc_range(ed, env_params, cal_params)

        # Transmission loss
        spreading_loss = 20 * np.log10(ed.range.where(ed.range >= 1, other=1))
        absorption_loss = 2 * env_params['absorption'] * ed.range

        if cal_type == 'Sv':
            # Calc gain
            CSv = (10 * np.log10((cal_params['transmit_power']))
                   + 2 * cal_params['gain_correction']
                   + cal_params['equivalent_beam_angle']
                   + 10 * np.log10(wavelength**2
                                   * ed.raw.transmit_duration_nominal
                                   * env_params['speed_of_sound_in_water']
                                   / (32 * np.pi**2)))

            # Calibration and echo integration
            Sv = ed.raw.backscatter_r + spreading_loss + absorption_loss - CSv - 2 * cal_params['sa_correction']
            Sv.name = 'Sv'
            Sv = Sv.to_dataset()

            # Attach calculated range into data set
            Sv['range'] = (('frequency', 'ping_time', 'range_bin'),
                           ed.range.transpose('frequency', 'ping_time', 'range_bin'))

            # Save Sv into the calling instance and
            #  to a separate zarr/nc file in the same directory as the data file
            if save:
                # Update pointer in EchoData
                Sv_path = io.validate_proc_path(ed, '_Sv', save_path)
                print(f"{dt.now().strftime('%H:%M:%S')}  saving calibrated Sv to {Sv_path}")
                ed._save_dataset(Sv, Sv_path, mode="w", save_format=save_format)
                ed.Sv_path = Sv_path
            else:
                ed.Sv = Sv

        elif cal_type == 'Sp':
            # Calc gain
            CSp = (10 * np.log10(cal_params['transmit_power'])
                   + 2 * cal_params['gain_correction']
                   + 10 * np.log10(wavelength**2 / (16 * np.pi**2)))

            # Calibration and echo integration
            Sp = ed.raw.backscatter_r + spreading_loss * 2 + absorption_loss - CSp
            Sp.name = 'Sp'
            Sp = Sp.to_dataset()

            # Attach calculated range into data set
            Sp['range'] = (('frequency', 'ping_time', 'range_bin'),
                           ed.range.transpose('frequency', 'ping_time', 'range_bin'))

            # Save Sp into the calling instance and
            #  to a separate zarr/nc file in the same directory as the data file
            if save:
                # Update pointer in EchoData
                Sp_path = io.validate_proc_path(ed, '_Sp', save_path)
                print(f"{dt.now().strftime('%H:%M:%S')}  saving calibrated Sp to {Sp_path}")
                ed._save_dataset(Sp, Sp_path, mode="w", save_format=save_format)
                ed.Sp_path = Sp_path
            else:
                ed.Sp = Sp
