import os
import numpy as np
import xarray as xr
from echopype.convert.ek60 import ConvertEK60
from echopype.model.ek60 import EchoData

raw_path = './echopype/data/DY1801_EK60-D20180211-T164025.raw'
nc_path = os.path.join(os.path.dirname(raw_path),
                       os.path.splitext(os.path.basename(raw_path))[0] + '.nc')
Sv_path = os.path.join(os.path.dirname(raw_path),
                       os.path.splitext(os.path.basename(raw_path))[0] + '_Sv.nc')


def test_noise_estimates_removal():
    """Check noise estimation and noise removal using xarray and brute force using numpy.
    """

    # Noise estimation via EchoData method =========
    # Unpack data and convert to .nc file
    tmp = ConvertEK60(raw_path)
    tmp.raw2nc()

    # Read .nc file into an EchoData object and calibrate
    e_data = EchoData(nc_path)
    e_data.calibrate()
    noise_est = e_data.noise_estimates()
    e_data.remove_noise()

    # Noise estimation via numpy brute force =======
    proc_data = xr.open_dataset(Sv_path)

    # Get tile indexing parameters
    e_data.noise_est_range_bin_size, add_idx, range_bin_tile_bin_edge = \
        e_data.get_tile_params(r_data_sz=proc_data.range_bin.size,
                               p_data_sz=proc_data.ping_time.size,
                               r_tile_sz=e_data.noise_est_range_bin_size,
                               p_tile_sz=e_data.noise_est_ping_size,
                               sample_thickness=e_data.sample_thickness)

    power_cal_test = (10 ** ((proc_data.Sv - e_data.ABS - e_data.TVG) / 10)).values

    num_ping_bins = np.unique(add_idx).size
    num_range_bins = range_bin_tile_bin_edge.size - 1
    noise_est_tmp = np.empty((proc_data.frequency.size, num_range_bins, num_ping_bins))  # all tiles
    noise_est_test = np.empty((proc_data.frequency.size, num_ping_bins))  # all columns
    p_sz = e_data.noise_est_ping_size
    p_idx = np.arange(p_sz, dtype=int)
    r_sz = (e_data.noise_est_range_bin_size.max() / e_data.sample_thickness[0].values).astype(int)
    r_idx = np.arange(r_sz, dtype=int)

    # Get noise estimates manually
    for f, f_seq in enumerate(np.arange(proc_data.frequency.size)):
        for p, p_seq in enumerate(np.arange(num_ping_bins)):
            for r, r_seq in enumerate(np.arange(num_range_bins)):
                if p_idx[-1] + p_sz * p_seq < power_cal_test.shape[1]:
                    pp_idx = p_idx + p_sz * p_seq
                else:
                    pp_idx = np.arange(p_sz * p_seq, power_cal_test.shape[1])
                if r_idx[-1] + r_sz * r_seq < power_cal_test.shape[2]:
                    rr_idx = r_idx + r_sz * r_seq
                else:
                    rr_idx = np.arange(r_sz * r_seq, power_cal_test.shape[2])
                nn = power_cal_test[f_seq, :, :][np.ix_(pp_idx, rr_idx)]
                noise_est_tmp[f_seq, r_seq, p_seq] = 10 * np.log10(nn.mean())
            noise_est_test[f_seq, p_seq] = noise_est_tmp[f_seq, :, p_seq].min()

    # Check xarray and numpy noise estimates
    assert np.all(np.isclose(noise_est_test, noise_est.noise_est.values))

    # Remove noise manually
    Sv_clean_test = np.empty(proc_data.Sv.shape)
    for f, f_seq in enumerate(np.arange(proc_data.frequency.size)):
        for p, p_seq in enumerate(np.arange(num_ping_bins)):
            if p_idx[-1] + p_sz * p_seq < power_cal_test.shape[1]:
                pp_idx = p_idx + p_sz * p_seq
            else:
                pp_idx = np.arange(p_sz * p_seq, power_cal_test.shape[1])
            ss_tmp = proc_data.Sv.values[f_seq, pp_idx, :]
            nn_tmp = (noise_est_test[f_seq, p_seq] +
                      e_data.ABS.isel(frequency=f_seq) + e_data.TVG.isel(frequency=f_seq)).values
            Sv_clean_tmp = ss_tmp.copy()
            Sv_clean_tmp[Sv_clean_tmp < nn_tmp] = np.nan
            Sv_clean_test[f_seq, pp_idx, :] = Sv_clean_tmp

    # Check xarray and numpy noise removal
    assert ~np.any(e_data.Sv_clean.Sv_clean.values[~np.isnan(e_data.Sv_clean.Sv_clean.values)]
                   != Sv_clean_test[~np.isnan(Sv_clean_test)])

    proc_data.close()
    del tmp
    del e_data
    os.remove(nc_path)
    os.remove(Sv_path)
