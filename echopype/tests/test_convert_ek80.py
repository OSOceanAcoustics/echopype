from pathlib import Path
import shutil
import numpy as np
import xarray as xr
import pandas as pd
from scipy.io import loadmat
from echopype import open_raw

ek80_path = Path('./echopype/test_data/ek80/')
output_dir = Path('./echopype/test_data/ek80/echopype_test_export')

# raw_path_simrad  = ['./echopype/test_data/ek80/simrad/EK80_SimradEcho_WC381_Sequential-D20150513-T090935.raw',
#                     './echopype/test_data/ek80/simrad/EK80_SimradEcho_WC381_Sequential-D20150513-T091004.raw',
#                     './echopype/test_data/ek80/simrad/EK80_SimradEcho_WC381_Sequential-D20150513-T091034.raw',
#                     './echopype/test_data/ek80/simrad/EK80_SimradEcho_WC381_Sequential-D20150513-T091105.raw']
# raw_paths = ['./echopype/test_data/ek80/Summer2018--D20180905-T033113.raw',
#              './echopype/test_data/ek80/Summer2018--D20180905-T033258.raw']  # Multiple files (CW and BB)


def test_convert_ek80_complex_matlab():
    """Compare parsed EK80 CW power/angle data with Matlab parsed data.
    """
    ek80_raw_path_bb = str(ek80_path.joinpath('D20170912-T234910.raw'))
    ek80_matlab_path_bb = str(ek80_path.joinpath('from_matlab/D20170912-T234910_data.mat'))

    # Convert file
    echodata = open_raw(raw_file=ek80_raw_path_bb, sonar_model='EK80')
    echodata.to_netcdf(save_path=output_dir)

    # Test complex parsed data
    ds_matlab = loadmat(ek80_matlab_path_bb)
    with xr.open_dataset(echodata.converted_raw_path, group='Beam') as ds_beam:
        assert np.array_equal(
            ds_beam.backscatter_r.isel(frequency=0, ping_time=0).dropna('range_bin').values[1:, :],
            np.real(ds_matlab['data']['echodata'][0][0][0,0]['complexsamples'])  # real part
        )
        assert np.array_equal(
            ds_beam.backscatter_i.isel(frequency=0, ping_time=0).dropna('range_bin').values[1:, :],
            np.imag(ds_matlab['data']['echodata'][0][0][0,0]['complexsamples'])  # imag part
        )

    Path(echodata.converted_raw_path).unlink()
    output_dir.rmdir()


def test_convert_ek80_cw_power_echoview():
    """Compare parsed EK80 CW power/angle data with csv exported by EchoView.
    """
    ek80_raw_path_cw = str(ek80_path.joinpath('D20190822-T161221.raw'))  # Small file (CW)
    freq_list = [18, 38, 70, 120, 200]
    ek80_echoview_power_csv = [
        ek80_path.joinpath('from_echoview/D20190822-T161221/%dkHz.power.csv' % freq)
        for freq in freq_list
    ]
    # ek80_echoview_angle_csv = [
    #     ek80_path.joinpath('from_echoview/D20190822-T161221/%dkHz.angles.points.csv' % freq)
    #     for freq in freq_list
    # ]

    # Convert file
    echodata = open_raw(ek80_raw_path_cw, sonar_model='EK80')
    echodata.to_netcdf(save_path=output_dir)

    # Test power
    with xr.open_dataset(echodata.converted_raw_path, group='Beam') as ds_beam:
        # single point error in original raw data. Read as -2000 by echopype and -999 by EchoView
        ds_beam.backscatter_r[3, 4, 13174] = -999
        for file, freq in zip(ek80_echoview_power_csv, freq_list):
            test_power = pd.read_csv(file, delimiter=';').iloc[:, 13:].values
            assert np.allclose(
                test_power,
                ds_beam.backscatter_r.sel(frequency=freq * 1e3).dropna('range_bin'),
                rtol=0, atol=1.1e-5
            )

    # # Test angle  TODO: fix angle test: bug in parser
    # with xr.open_dataset(echodata.converted_raw_path, group='Beam') as ds_beam:
    #     # Convert from electrical angles to physical angle [deg]
    #     major = (ds_beam['angle_athwartship'] * 1.40625
    #              / ds_beam['angle_sensitivity_athwartship']
    #              - ds_beam['angle_offset_athwartship'])
    #     minor = (ds_beam['angle_alongship'] * 1.40625
    #              / ds_beam['angle_sensitivity_alongship']
    #              - ds_beam['angle_offset_alongship'])
    #     for file, freq in zip(ek80_echoview_angle_csv, freq_list):
    #         df_angle = pd.read_csv(file)
    #         for ping_idx in df_angle['Ping_index'].value_counts().index:
    #             assert np.allclose(
    #                 df_angle.loc[df_angle['Ping_index'] == ping_idx, ' Major'],
    #                 major.isel(frequency=0, ping_time=0),
    #                 rtol=0, atol=1e-5
    #             )
    #             assert np.allclose(
    #                 df_angle.loc[df_angle['Ping_index'] == ping_idx, ' Minor'],
    #                 minor.isel(frequency=0, ping_time=0),
    #                 rtol=0, atol=1e-5
    #             )

    Path(echodata.converted_raw_path).unlink()
    output_dir.rmdir()


def test_convert_ek80_complex_echoview():
    """Compare parsed EK80 BB data with csv exported by EchoView.
    """
    ek80_raw_path_bb = ek80_path.joinpath('D20170912-T234910.raw')
    ek80_echoview_bb_power_csv = ek80_path.joinpath('from_echoview/D20170912-T234910/70 kHz raw power.complex.csv')

    # Convert file
    echodata = open_raw(raw_file=ek80_raw_path_bb, sonar_model='EK80')
    echodata.to_netcdf(save_path=output_dir)

    # Test complex parsed data
    df_bb = pd.read_csv(ek80_echoview_bb_power_csv, header=None, skiprows=[0])  # averaged across quadrants
    with xr.open_dataset(echodata.converted_raw_path, group='Beam') as ds_beam:
        assert np.allclose(
            ds_beam.backscatter_r.sel(frequency=70e3).dropna('range_bin').mean(dim='quadrant'),
            df_bb.iloc[::2, 14:],  # real rows
            rtol=0, atol=8e-6
        )
        assert np.allclose(
            ds_beam.backscatter_i.sel(frequency=70e3).dropna('range_bin').mean(dim='quadrant'),
            df_bb.iloc[1::2, 14:],  # imag rows
            rtol=0, atol=4e-6
        )

    Path(echodata.converted_raw_path).unlink()
    output_dir.rmdir()


def test_convert_ek80_cw_bb_in_single_file():
    """Make sure can convert a single EK80 file containing both CW and BB mode data.
    """
    ek80_raw_path_bb_cw = str(ek80_path.joinpath('Summer2018--D20180905-T033113.raw'))
    echodata = open_raw(raw_file=ek80_raw_path_bb_cw, sonar_model='EK80')
    echodata.to_zarr(save_path=output_dir)

    # Check there are both Beam and Beam_power groups in the converted file
    ds_beam = xr.open_zarr(echodata.converted_raw_path, group='Beam')
    ds_beam_power = xr.open_zarr(echodata.converted_raw_path, group='Beam_power')

    ds_beam.close()
    ds_beam_power.close()
    shutil.rmtree(echodata.converted_raw_path)
    output_dir.rmdir()


def test_convert_ek80_freq_subset():
    """Make sure can convert EK80 file with multiple frequency channels off.
    """
    ek80_raw_path_freq_subset = str(ek80_path.joinpath('2019118 group2survey-D20191214-T081342.raw'))
    echodata = open_raw(raw_file=ek80_raw_path_freq_subset, sonar_model='EK80')
    echodata.to_zarr(save_path=output_dir)

    # Check if converted output has only 2 frequency channels
    with xr.open_dataset(echodata.converted_raw_path, group='Beam', engine='zarr') as ds_beam:
        assert ds_beam.frequency.size == 2
    shutil.rmtree(echodata.converted_raw_path)
    output_dir.rmdir()


# def test_xml():
#     # Tests the exporting of the configuration xml as well as the environment xml
#     tmp = Convert(raw_file=raw_path_bb_cw, sonar_model='EK80')
#     tmp.to_xml(data_type='CONFIG')
#     assert os.path.exists(tmp.converted_raw_path)
#     os.remove(tmp.converted_raw_path)
#
#     tmp.to_xml(save_path='env.xml', data_type='ENV')
#     assert os.path.exists(tmp.converted_raw_path)
#     os.remove(tmp.converted_raw_path)
#
#
# def test_add_platform():
#     # Construct lat/lon dataset with fake data using a date range that includes
#     # the ping_time ranges of the raw EK80 file. 7 pings over 28.166 seconds.
#     # (2019-08-22T16:12:21.398000128 to 2019-08-22T16:12:49.564000256)
#     location_time = pd.date_range(start='2019-08-22T16:00:00.0',
#                                   end='2019-08-22T16:15:00.0', periods=100)
#     lat = np.random.rand(100)
#     lon = np.random.rand(100)
#     testing_ds = xr.Dataset({'lat': (['location_time'], lat),
#                              'lon': (['location_time'], lon)},
#                             coords={'location_time': (['location_time'], location_time)})
#     tmp = Convert(raw_file=raw_path_cw, sonar_model='EK80')
#     tmp.to_netcdf(overwrite=True, extra_platform_data=testing_ds)
#     with xr.open_dataset(tmp.converted_raw_path, group='Platform') as ds_plat:
#         # Test if the slicing the location_time with the ping_time worked
#         assert len(ds_plat.location_time) == 3
#     os.remove(tmp.converted_raw_path)
