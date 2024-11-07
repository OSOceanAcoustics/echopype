import pytest
import echopype as ep
import numpy as np
import xarray as xr


@pytest.mark.integration
def test_regrid_Sv(test_data_samples):
    """
    Test running through from open_raw to compute_MVBS.
    """
    (
        filepath,
        sonar_model,
        azfp_xml_path,
        range_kwargs,
    ) = test_data_samples
    ed = ep.open_raw(filepath, sonar_model, azfp_xml_path)
    if ed.sonar_model.lower() == "azfp":
        avg_temperature = ed["Environment"]["temperature"].values.mean()
        env_params = {
            "temperature": avg_temperature,
            "salinity": 27.9,
            "pressure": 59,
        }
        range_kwargs["env_params"] = env_params
        if "azfp_cal_type" in range_kwargs:
            range_kwargs.pop("azfp_cal_type")
    Sv = ep.calibrate.compute_Sv(ed, **range_kwargs)

    # Setup output grid
    depth_data = Sv.echo_range.isel(ping_time=0)
    time_data = Sv.ping_time.astype("float64")

    # Evenly spaced grid
    target_grid = xr.Dataset(
        {
            "ping_time": (
                ["ping_time"],
                np.linspace(time_data.min().values, time_data.max().values, 300).astype(
                    "datetime64[ns]"
                ),
            ),
            "echo_range": (
                ["echo_range"],
                np.linspace(depth_data.min().values, depth_data.max().values, 300),
            ),
        }
    )

    regridded_Sv = ep.commongrid.regrid_Sv(Sv, target_grid=target_grid)
    assert regridded_Sv is not None

    # Test to see if values average are close
    for channel in regridded_Sv.channel:
        original_vals = Sv.sel(channel=channel).Sv.values
        regridded_vals = regridded_Sv.sel(channel=channel).Sv.values
        assert np.allclose(
            np.nanmean(original_vals),
            np.nanmean(regridded_vals),
            atol=1.0,
            rtol=1.0,
            equal_nan=True,
        )
