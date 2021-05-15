from pathlib import Path

import xarray as xr
import numpy as np
import netCDF4

from echopype import open_raw, open_converted


normal_test_file_dir = Path("./echopype/test_data/ad2cp/normal")
raw_test_file_dir = Path("./echopype/test_data/ad2cp/raw")
ocean_contour_export_dir = Path("./echopype/test_data/ad2cp/ocean-contour")
ocean_contour_export_076_dir = ocean_contour_export_dir / "076"
ocean_contour_export_090_dir = ocean_contour_export_dir / "090"
output_dir = Path("./echopype/test_data/ad2cp/echopype_test-export")


ABSOLUTE_TOLERANCE = 1e-6


def test_convert():
    for filepath in normal_test_file_dir.glob("**/*.ad2cp"):
        print("converting", filepath)
        echodata = open_raw(raw_file=str(filepath), sonar_model="AD2CP")
        echodata.to_netcdf(save_path=output_dir)

raw_echodata = {}

def test_convert_raw():
    for filepath in raw_test_file_dir.glob("**/*.ad2cp"):
        print("converting raw", filepath)
        echodata = open_raw(raw_file=str(filepath), sonar_model="AD2CP")
        echodata.to_netcdf(save_path=output_dir)
        raw_echodata[filepath] = echodata

# TODO: xarray has precision issues with saving/loading np.datetime64 to/from disk in v0.16.2,
# so when we get errors when we try to select using timestamps. Instead, we'll just keep
# the data in memory so we know the representation will be correct. 
# 
# When xarray version gets bumped, try deleting raw_echodata 
# and using the first few commented lines in this function
# (it is known to behave correctly in v0.17.0).
def test_raw_output():
    # for filepath in raw_test_file_dir.glob("**/*.ad2cp"):
    for filepath, echodata in raw_echodata.items():
        print("checking raw", filepath)
        # echodata = open_converted(
        #     converted_raw_path=output_dir.joinpath(filepath.with_suffix(".nc").name)
        # )
        if "090" in filepath.parts:
            ocean_contour_converted_config_path = ocean_contour_export_090_dir.joinpath(
                filepath.with_suffix(filepath.suffix + ".00000.nc").name
            )
            ocean_contour_converted_transmit_data_path = (
                ocean_contour_converted_config_path
            )
            ocean_contour_converted_data_path = ocean_contour_converted_config_path
        else:
            ocean_contour_converted_config_path = (
                ocean_contour_export_076_dir
                / filepath.with_suffix("").name
                / "Raw Echo 1_1000 kHz_001.nc"
            )
            ocean_contour_converted_transmit_data_path = (
                ocean_contour_export_076_dir
                / filepath.with_suffix("").name
                / "Raw Echo 1_1000 kHz Tx_001.nc"
            )
            ocean_contour_converted_data_path = (
                ocean_contour_export_076_dir
                / filepath.with_suffix("").name
                / "Raw Echo 1_1000 kHz_001.nc"
            )
        if not all(
            (
                ocean_contour_converted_config_path.exists(),
                ocean_contour_converted_transmit_data_path.exists(),
                ocean_contour_converted_data_path.exists(),
            )
        ):
            continue

        # check pulse compression
        base = xr.open_dataset(str(ocean_contour_converted_config_path), group="Config")
        pulse_compressed = 0
        for i in range(1, 4):
            if "090" in filepath.parts:
                if base.attrs[f"echo_pulseComp{i}"]:
                    pulse_compressed = i
                    break
            else:
                if base.attrs[f"Instrument_echo_pulseComp{i}"]:
                    pulse_compressed = i
                    break
        assert echodata.beam_complex.attrs["pulse_compressed"] == pulse_compressed
        base.close()

        # check raw data transmit samples
        try:
            netCDF4.Dataset(str(ocean_contour_converted_transmit_data_path))[
                "Data/RawEcho1_1000kHzTx"
            ]
        except IndexError:
            # no transmit data in this dataset
            pass
        else:
            base = xr.open_dataset(
                str(ocean_contour_converted_transmit_data_path),
                group="Data/RawEcho1_1000kHzTx",
            )
            if "090" in filepath.parts:
                assert np.isclose(
                    echodata.beam_complex["echosounder_raw_transmit_samples_r"]
                    .sel(
                        ping_time=echodata.beam_complex["time_echosounder_raw_transmit"]
                    )
                    .data.flatten(),
                    base["DataI"].data.flatten(),
                    atol=ABSOLUTE_TOLERANCE,
                ).all()
                assert np.isclose(
                    echodata.beam_complex["echosounder_raw_transmit_samples_i"]
                    .sel(
                        ping_time=echodata.beam_complex["time_echosounder_raw_transmit"]
                    )
                    .data.flatten(),
                    base["DataQ"].data.flatten(),
                    atol=ABSOLUTE_TOLERANCE,
                ).all()
            else:
                assert np.isclose(
                    echodata.beam_complex["echosounder_raw_transmit_samples_r"]
                    .sel(
                        ping_time=echodata.beam_complex["time_echosounder_raw_transmit"]
                    )
                    .data.flatten(),
                    base["Data_I"].data.flatten(),
                    atol=ABSOLUTE_TOLERANCE,
                ).all()
                assert np.isclose(
                    echodata.beam_complex["echosounder_raw_transmit_samples_i"]
                    .sel(
                        ping_time=echodata.beam_complex["time_echosounder_raw_transmit"]
                    )
                    .data.flatten(),
                    base["Data_Q"].data.flatten(),
                    atol=ABSOLUTE_TOLERANCE,
                ).all()
            base.close()

        # check raw data samples
        base = xr.open_dataset(
            str(ocean_contour_converted_data_path), group="Data/RawEcho1_1000kHz"
        )
        if "090" in filepath.parts:
            assert np.isclose(
                echodata.beam_complex["echosounder_raw_samples_r"]
                .sel(ping_time=echodata.beam_complex["time_echosounder_raw"])
                .data.flatten(),
                base["DataI"].data.flatten(),
                atol=ABSOLUTE_TOLERANCE,
            ).all()
            assert np.isclose(
                echodata.beam_complex["echosounder_raw_samples_i"]
                .sel(ping_time=echodata.beam_complex["time_echosounder_raw"])
                .data.flatten(),
                base["DataQ"].data.flatten(),
                atol=ABSOLUTE_TOLERANCE,
            ).all()
        else:
            # note the transpose
            assert np.isclose(
                echodata.beam_complex["echosounder_raw_samples_r"]
                .sel(ping_time=echodata.beam_complex["time_echosounder_raw"])
                .data.flatten(),
                base["Data_I"].data.T.flatten(),
                atol=ABSOLUTE_TOLERANCE,
            ).all()
            assert np.isclose(
                echodata.beam_complex["echosounder_raw_samples_i"]
                .sel(ping_time=echodata.beam_complex["time_echosounder_raw"])
                .data.flatten(),
                base["Data_Q"].data.T.flatten(),
                atol=ABSOLUTE_TOLERANCE,
            ).all()
        base.close()
