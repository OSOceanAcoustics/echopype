import os

import xarray as xr
import numpy as np

from ..convert import Convert

GROUPS = [
    "Environment",
    "Platform",
    "Beam",
    "Vendor"
]

FILES = [
    # "./echopype/test_data/ad2cp/average_only.366.00000.ad2cp",
    "./echopype/test_data/ad2cp/avg_bur_echo.366.00000.ad2cp",
    "./echopype/test_data/ad2cp/burst_echosoun.366.00000.ad2cp",
    "./echopype/test_data/ad2cp/burst_only.366.00000.ad2cp",
]

BASE_FILES = [
    # "./echopype/test_data/ad2cp/average_only.366.00000.base.nc",
    "./echopype/test_data/ad2cp/avg_bur_echo.366.00000.base.nc",
    "./echopype/test_data/ad2cp/burst_echosoun.366.00000.base.nc",
    "./echopype/test_data/ad2cp/burst_only.366.00000.base.nc",
]

def test_process():
    for file, base_file in zip(FILES, BASE_FILES):
        tmp = Convert(file=file, model="AD2CP")
        tmp_file = f"{file}.test.nc"
        tmp.to_netcdf(save_path=tmp_file, overwrite=True)

        for group in GROUPS:
            testing_group = xr.open_dataset(tmp_file, group=group)
            base_group = xr.open_dataset(base_file, group=group)
            assert testing_group.equals(base_group), f"process ad2cp failed on {file} with group {group}"

        os.remove(tmp_file)

OCEAN_CONTOUR_UNPARSED = [
    "./echopype/test_data/ad2cp/SG198_0097_a.ad2cp",
    "./echopype/test_data/ad2cp/SG198_0097_b.ad2cp",
    "./echopype/test_data/ad2cp/SG198_0099_a.ad2cp",
    "./echopype/test_data/ad2cp/SG198_0099_b.ad2cp"
]

OCEAN_CONTOUR_PARSED = [
    "./echopype/test_data/ad2cp/SG198_0097_a.ad2cp.nc",
    "./echopype/test_data/ad2cp/SG198_0097_b.ad2cp.nc",
    "./echopype/test_data/ad2cp/SG198_0099_a.ad2cp.nc",
    "./echopype/test_data/ad2cp/SG198_0099_b.ad2cp.nc"
]

THRESHOLD = 0.02

def test_ocean_contour():
    def close(a, b):
        return abs(np.float32(a) - np.float32(b)) < THRESHOLD

    for file, parsed_file in zip(OCEAN_CONTOUR_UNPARSED, OCEAN_CONTOUR_PARSED):
        test_convert = Convert(file, model="AD2CP")
        test_file = ".".join(file.split(".")[:-1]) + ".nc"
        test_convert.to_netcdf(save_path=test_file, overwrite=True)

        # Config
        base = xr.open_dataset(parsed_file, group="Config")
        
        test = xr.open_dataset(test_file, group="Vendor")
        assert close(base.attrs["Instrument_pressure"], test.attrs["pressure_sensor_valid"])
        assert close(base.attrs["Instrument_temperature"], test.attrs["temperature_sensor_valid"])
        assert close(base.attrs["Instrument_compass"], test.attrs["compass_sensor_valid"])
        assert close(base.attrs["Instrument_tilt"], test.attrs["tilt_sensor_valid"])
        assert close(base.attrs["Instrument_avg_nCells"], test.dims["range_bin_average"])
        assert close(base.attrs["Instrument_avg_nBeams"], test.dims["beam"])
        assert close(base.attrs["Instrument_echo_enable"], test.attrs["echosounder_data_included"])
        test.close()

        test = xr.open_dataset(test_file, group="Beam")
        assert close(base.attrs["Instrument_avg_cellSize"], test["cell_size"].data[0])
        assert close(base.attrs["Instrument_avg_blankingDistance"], (test["blanking"].data[0]).astype(np.float32))
        # FIXME: test["velocity_range"] is nan
        # assert close(base.attrs["Instrument_avg_velocityRange"], test["velocity_range"].data[0])
        # TODO: transmit power vs transmit energy?
        # assert close(base.attrs["Instrument_avg_transmitPower"], test["transmit_energy"].data[0])
        test.close()

        # test = xr.open_dataset(r"C:\Users\strea\Desktop\UW\echopype\fork3\echopype\echopype\test_data\ad2cp\SG198_0097_a.nc", group="Environment")
        # assert close(base.attrs["DataInfo_pressure_min"], min(test["pressure"].data))
        # assert close(base.attrs["DataInfo_pressure_max"], max(test["pressure"].data))
        # test.close()

        base.close()

        # Data

        def test_field(base, test, base_field_name, test_field_name):
            for base_value, test_value in zip(base[base_field_name], test[test_field_name]):
                assert close(base_value.data[()], test_value.data[()])

        base = xr.open_dataset(parsed_file, group="Data/Avg")
        
        test = xr.open_dataset(test_file, group="Platform")
        # ocean contour timestamps have rounding errors so we can't compare times
        for base_heading, test_heading in zip(base["Heading"], test["heading"]):
            assert close(base_heading.data[()], (test_heading.data[()]).astype(np.float32))
        for base_pitch, test_pitch in zip(base["Pitch"], test["pitch"]):
            assert close(base_pitch.data[()], (test_pitch.data[()]).astype(np.float32))
        for base_roll, test_roll in zip(base["Roll"], test["roll"]):
            assert close(base_roll.data[()], (test_roll.data[()]).astype(np.float32))

        test_field(base, test, "Magnetometer_X", "magnetometer_raw_x")
        test_field(base, test, "Magnetometer_Y", "magnetometer_raw_y")
        test_field(base, test, "Magnetometer_Z", "magnetometer_raw_z")
        test.close()

        test = xr.open_dataset(test_file, group="Beam")
        test_field(base, test, "NumberofCells", "number_of_cells")
        test_field(base, test, "CellSize", "cell_size")
        test_field(base, test, "Blanking", "blanking")
        test_field(base, test, "TransmitEnergy", "transmit_energy")
        test_field(base, test, "Ambiguity", "ambiguity_velocity")

        for beam in test["beam"]:
            for range_bin in range(15):
                for base_vel, test_vel in zip(base[f"Vel_Beam{beam.data[()]}"].isel(AvgVelocityBeam_Range=range_bin), test["velocity_average"].sel(beam=beam.data[()]).isel(range_bin_average=range_bin)):
                    assert close(base_vel.data[()], test_vel.data[()])
                for base_cor, test_cor in zip(base[f"Cor_Beam{beam.data[()]}"].isel(AvgCorrelationBeam_Range=range_bin), test["correlation_average"].sel(beam=beam.data[()]).isel(range_bin_average=range_bin)):
                    assert close(base_cor.data[()], test_cor.data[()])
                for base_amp, test_amp in zip(base[f"Amp_Beam{beam.data[()]}"].isel(AvgAmplitudeBeam_Range=range_bin), test["amplitude_average"].sel(beam=beam.data[()]).isel(range_bin_average=range_bin)):
                    assert close(base_amp.data[()], test_amp.data[()])
        test.close()

        test = xr.open_dataset(test_file, group="Vendor")
        # uncalibrated (no units so cannot compare)
        # test_field(base, test, "MagnetometerTemperature", "magnetometer_temperature")
        test_field(base, test, "EnsembleCount", "ensemble_counter")
        test_field(base, test, "RTCTemperature", "real_time_clock_temperature")
        test_field(base, test, "NominalCor", "nominal_correlation")
        test.close()

        base.close()