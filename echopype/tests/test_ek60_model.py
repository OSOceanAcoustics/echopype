import os
import numpy as np
import xarray as xr
from echopype.convert import Convert
from echopype.model import EchoData

# ek60_raw_path = './echopype/test_data/ek60/2015843-D20151023-T190636.raw'   # Varying ranges
ek60_raw_path = './echopype/test_data/ek60/DY1801_EK60-D20180211-T164025.raw'     # Constant ranges
ek60_test_path = './echopype/test_data/ek60/from_matlab/DY1801_EK60-D20180211-T164025_Sv_TS.nc'
nc_path = os.path.join(os.path.dirname(ek60_raw_path),
                       os.path.splitext(os.path.basename(ek60_raw_path))[0] + '.nc')
Sv_path = os.path.join(os.path.dirname(ek60_raw_path),
                       os.path.splitext(os.path.basename(ek60_raw_path))[0] + '_Sv.nc')


def test_noise_estimates_removal():
    """Check noise estimation and noise removal using xarray and brute force using numpy.
    """

    # Noise estimation via EchoData method =========
    # Unpack data and convert to .nc file
    tmp = Convert(ek60_raw_path)
    tmp.raw2nc()

    # Read .nc file into an EchoData object and calibrate
    e_data = EchoData(nc_path)
    e_data.calibrate(save=True)
    noise_est = e_data.noise_estimates()

    with xr.open_dataset(ek60_test_path) as ds_test:
        ds_Sv = ds_test.Sv

    assert np.allclose(ds_Sv.values, e_data.Sv['Sv'].values, atol=1e-10)  # TODO: now identical to 1e-5 with matlab output
    # assert np.allclose(ds_TS.values, e_data.TS.TS.values, atol=1e-10)
    # Noise estimation via numpy brute force =======
    proc_data = xr.open_dataset(Sv_path)

    # Get tile indexing parameters
    e_data.noise_est_range_bin_size, range_bin_tile_bin_edge, ping_tile_bin_edge = \
        e_data.get_tile_params(r_data_sz=proc_data.range_bin.size,
                               p_data_sz=proc_data.ping_time.size,
                               r_tile_sz=e_data.noise_est_range_bin_size,
                               p_tile_sz=e_data.noise_est_ping_size,
                               sample_thickness=e_data.sample_thickness)

    range_bin_tile_bin_edge = np.unique(range_bin_tile_bin_edge)

    range_meter = e_data.range
    TVG = np.real(20 * np.log10(range_meter.where(range_meter >= 1, other=1)))
    ABS = 2 * e_data.seawater_absorption * range_meter
    power_cal_test = (10 ** ((proc_data.Sv - ABS - TVG) / 10)).values

    num_ping_bins = ping_tile_bin_edge.size - 1
    num_range_bins = range_bin_tile_bin_edge.size - 1
    noise_est_tmp = np.empty((proc_data.frequency.size, num_range_bins, num_ping_bins))  # all tiles
    noise_est_test = np.empty((proc_data.frequency.size, num_ping_bins))  # all columns
    p_sz = e_data.noise_est_ping_size
    p_idx = np.arange(p_sz, dtype=int)
    r_sz = (e_data.noise_est_range_bin_size.max() / e_data.sample_thickness[0].values).astype(int).values
    r_idx = np.arange(r_sz, dtype=int)

    # Get noise estimates manually
    for f_seq in np.arange(proc_data.frequency.size):
        for p_seq in np.arange(num_ping_bins):
            for r_seq in np.arange(num_range_bins):
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

    # Remove noise using .remove_noise()
    e_data.remove_noise()

    # Remove noise manually
    Sv_clean_test = np.empty(proc_data.Sv.shape)
    for ff, freq in enumerate(proc_data.frequency.values):
        for pp in np.arange(num_ping_bins):
            if pp == num_ping_bins - 1:    # if the last ping bin
                pp_idx = np.arange(p_sz * pp, power_cal_test.shape[1])
            else:                          # all other ping bins
                pp_idx = p_idx + p_sz * pp
            ss_tmp = proc_data['Sv'].sel(frequency=freq).values[pp_idx, :]   # all data in this ping bin
            nn_tmp = (noise_est['noise_est'].sel(frequency=freq).isel(ping_time=pp) +
                      ABS.sel(frequency=freq) + TVG.sel(frequency=freq)).values
            Sv_clean_tmp = ss_tmp.copy()
            Sv_clean_tmp[Sv_clean_tmp <= nn_tmp] = np.nan
            Sv_clean_test[ff, pp_idx, :] = Sv_clean_tmp

    # Check xarray and numpy noise removal
    assert ~np.any(e_data.Sv_clean['Sv'].values[~np.isnan(e_data.Sv_clean['Sv'].values)]
                   != Sv_clean_test[~np.isnan(Sv_clean_test)])

    proc_data.close()
    del tmp
    del e_data
    os.remove(nc_path)
    os.remove(Sv_path)
