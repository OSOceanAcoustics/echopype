import math

import pytest
import numpy as np
import xarray as xr

import echopype as ep
from echopype.consolidate.loc_utils import sel_nmea


@pytest.mark.unit
def test_sel_nmea_value_error():
    """
    Check that the appropriate ValueError is raised when nmea_sentence!=None and datagram_type!=None.
    This would imply NMEA sentence selection of location variable not from the NMEA datagrams, which
    is wrong.
    """
    # Check if the expected error is logged
    with pytest.raises(ValueError) as exc_info:
        # Pass in non-None values for nmea_sentence and datagram_type and leave the rest as
        # None (blank) since those are not needed to test this part of the function
        sel_nmea(
            echodata=None,
            loc_name=None,
            datagram_type="MRU1",
            nmea_sentence="GGA"
        )
    assert ("If datagram_type is not `None`, then `nmea_sentence` cannot be specified.") == str(exc_info.value)


@pytest.mark.unit
def test_add_location_datagram_type_specified_not_ek_error():
    """
    Check that the appropriate ValueError is raised when datagram_type is passed in and EchoData sonar model is
    not EK.
    """
    # Compute raw and compute Sv
    ed = ep.open_raw(
        "echopype/test_data/azfp/17082117.01A",
        sonar_model="AZFP",
        xml_path="echopype/test_data/azfp/23081211.XML"
    )
    avg_temperature = ed["Environment"]['temperature'].values.mean()
    env_params = {
        'temperature': avg_temperature,
        'salinity': 27.9,
        'pressure': 59,
    }
    ds_Sv = ep.calibrate.compute_Sv(echodata=ed, env_params=env_params)

    # Check if the expected error is logged
    with pytest.raises(ValueError) as exc_info:
        ep.consolidate.add_location(
            ds=ds_Sv,
            echodata=ed,
            datagram_type="MRU1",
        )
    assert ("Sonar Model must be EK in order to specify datagram_type.") == str(exc_info.value)


@pytest.mark.integration
@pytest.mark.parametrize(
    ["location_type", "sonar_model", "path_model", "raw_and_xml_paths", "lat_lon_name_dict", "extras"],
    [
        (
            "empty-location",
            "EK60",
            "EK60",
            ("ooi/CE02SHBP-MJ01C-07-ZPLSCB101_OOI-D20191201-T000000.raw", None),
            {"lat_name": "latitude", "lon_name": "longitude"},
            None,
        ),
        (
            "with-track-location",
            "EK60",
            "EK60",
            ("Winter2017-D20170115-T150122.raw", None),
            {"lat_name": "latitude", "lon_name": "longitude"},
            None,
        ),
        (
            "fixed-location",
            "AZFP",
            "AZFP",
            ("17082117.01A", "17041823.XML"),
            {"lat_name": "latitude", "lon_name": "longitude"},
            {'longitude': -60.0, 'latitude': 45.0, 'salinity': 27.9, 'pressure': 59},
        ),
    ],
)
def test_add_location(
        location_type,
        sonar_model,
        path_model,
        raw_and_xml_paths,
        lat_lon_name_dict,
        extras,
        test_path
):
    # Prepare the Sv dataset
    raw_path = test_path[path_model] / raw_and_xml_paths[0]
    if raw_and_xml_paths[1]:
        xml_path = test_path[path_model] / raw_and_xml_paths[1]
    else:
        xml_path = None

    ed = ep.open_raw(raw_path, xml_path=xml_path, sonar_model=sonar_model)
    if location_type == "fixed-location":
        point_ds = xr.Dataset(
            {
                lat_lon_name_dict["lat_name"]: (["time"], np.array([float(extras['latitude'])])),
                lat_lon_name_dict["lon_name"]: (["time"], np.array([float(extras['longitude'])])),
            },
            coords={
                "time": (["time"], np.array([ed["Sonar/Beam_group1"]["ping_time"].values.min()]))
            },
        )
        ed.update_platform(
            point_ds,
            variable_mappings={
                lat_lon_name_dict["lat_name"]: lat_lon_name_dict["lat_name"],
                lat_lon_name_dict["lon_name"]: lat_lon_name_dict["lon_name"]
            }
        )

    env_params = None
    # AZFP data require external salinity and pressure
    if sonar_model == "AZFP":
        env_params = {
            "temperature": ed["Environment"]["temperature"].values.mean(),
            "salinity": extras["salinity"],
            "pressure": extras["pressure"],
        }

    ds = ep.calibrate.compute_Sv(echodata=ed, env_params=env_params)

    # add_location tests
    if location_type == "empty-location":
        with pytest.raises(Exception) as exc:
            ep.consolidate.add_location(ds=ds, echodata=ed)
        assert exc.type is ValueError
        assert "Coordinate variables are all NaN." in str(exc.value)
    else:
        def _tests(ds_test, location_type, nmea_sentence=None):
            # lat,lon & time1 existence
            assert "latitude" in ds_test
            assert "longitude" in ds_test
            assert "time1" not in ds_test

            # lat & lon have a single dimension: 'ping_time'
            assert len(ds_test["latitude"].dims) == 1 and ds_test["latitude"].dims[0] == "ping_time" # noqa
            assert len(ds_test["longitude"].dims) == 1 and ds_test["longitude"].dims[0] == "ping_time" # noqa

            # Check interpolated or broadcast values
            if location_type == "with-track-location":
                for ed_position, ds_position in [
                    (lat_lon_name_dict["lat_name"], "latitude"),
                    (lat_lon_name_dict["lon_name"], "longitude")
                ]:
                    position_var = ed["Platform"][ed_position]
                    if nmea_sentence:
                        position_var = position_var[ed["Platform"]["sentence_type"] == nmea_sentence]
                    position_interp = position_var.interp(
                        {"time1": ds_test["ping_time"]},
                        method="linear",
                        kwargs={"fill_value": "extrapolate"},
                    )
                    # interpolated values are identical
                    assert np.allclose(ds_test[ds_position].values, position_interp.values, equal_nan=True) # noqa
            elif location_type == "fixed-location":
                for position in ["latitude", "longitude"]:
                    position_uniq = set(ds_test[position].values)
                    # contains a single repeated value equal to the value passed to update_platform
                    assert (
                            len(position_uniq) == 1 and
                            math.isclose(list(position_uniq)[0], extras[position])
                    )

        ds_all = ep.consolidate.add_location(ds=ds, echodata=ed)
        _tests(ds_all, location_type)

        # the test for nmea_sentence="GGA" is limited to the with-track-location case
        if location_type == "with-track-location" and sonar_model.startswith("EK"):
            ds_sel = ep.consolidate.add_location(ds=ds, echodata=ed, nmea_sentence="GGA")
            _tests(ds_sel, location_type, nmea_sentence="GGA")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("raw_path, sonar_model, datagram_type, parse_idx, time_dim_name, compute_Sv_kwargs"),
    [
        (
            "echopype/test_data/ek80/D20170912-T234910.raw",
            "EK80",
            None,
            False,
            "time1",
            {
                "waveform_mode": "BB",
                "encode_mode": "complex"
            }
        ),
        (
            "echopype/test_data/ek80/RL2407_ADCP-D20240709-T150437.raw",
            "EK80",
            "MRU1",
            False,
            "time3",
            {
                "waveform_mode": "CW",
                "encode_mode": "complex"
            }
        ),
        (
            "echopype/test_data/ek80/idx_bot/Hake-D20230711-T181910.raw",
            "EK80",
            "IDX",
            True,
            "time4",
            {
                "waveform_mode": "CW",
                "encode_mode": "power"
            }
        ),
    ],
)
def test_add_location_time_duplicates_value_error(
    raw_path, sonar_model, datagram_type, parse_idx, time_dim_name, compute_Sv_kwargs,
):   
    """Tests for duplicate time value error in ``add_location``.""" 
    # Open raw and compute the Sv dataset
    if parse_idx:
        ed = ep.open_raw(raw_path, include_idx=True, sonar_model=sonar_model)
    else:
        ed = ep.open_raw(raw_path, sonar_model=sonar_model)
    ds = ep.calibrate.compute_Sv(
        echodata=ed,
        **compute_Sv_kwargs,
    )

    # Add duplicates to `time_dim`
    ed["Platform"][time_dim_name].data[0] = ed["Platform"][time_dim_name].data[1]
    
    # Check if the expected error is logged
    with pytest.raises(ValueError) as exc_info:
        # Run add location with duplicated time
        ep.consolidate.add_location(ds=ds, echodata=ed, datagram_type=datagram_type)

    # Check if the specific error message is in the logs
    assert (
        f'The ``echodata["Platform"]["{time_dim_name}"]`` array contains duplicate values. '
        "Downstream interpolation on the position variables requires unique time values."
    ) == str(exc_info.value)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("raw_path, sonar_model, datagram_type, parse_idx, compute_Sv_kwargs, error_type, expected_error_message"),
    [
        (
            "echopype/test_data/ek80/D20170912-T234910.raw",
            "EK80",
            None,
            False,
            {
                "waveform_mode": "BB",
                "encode_mode": "complex"
            },
            "missing",
            "Coordinate variables not present.",
        ),
        (
            "echopype/test_data/ek80/RL2407_ADCP-D20240709-T150437.raw",
            "EK80",
            "MRU1",
            False,
            {
                "waveform_mode": "CW",
                "encode_mode": "complex"
            },
            "all_nan",
            "Coordinate variables are all NaN.",
        ),
        (
            "echopype/test_data/ek80/idx_bot/Hake-D20230711-T181910.raw",
            "EK80",
            "IDX",
            True,
            {
                "waveform_mode": "CW",
                "encode_mode": "power"
            },
            "all_nan",
            "Coordinate variables are all NaN. Consider setting datagram_type to any of [None].",
        ),
    ],
)
def test_add_location_lat_lon_missing_all_NaN_errors(
    raw_path, sonar_model, datagram_type, parse_idx, compute_Sv_kwargs, error_type, expected_error_message
):
    """Tests for lat lon missing or all NaN values errors."""
    # Open raw and compute the Sv dataset
    if parse_idx:
        ed = ep.open_raw(raw_path, include_idx=True, sonar_model=sonar_model)
    else:
        ed = ep.open_raw(raw_path, sonar_model=sonar_model)
    ds = ep.calibrate.compute_Sv(
        echodata=ed,
        **compute_Sv_kwargs,
    )

    # Set NaN/None to Lat/Lon
    if datagram_type in ["MRU1", "IDX"]:
        if error_type == "missing":
            ed["Platform"] = ed["Platform"].drop_vars(f"longitude_{datagram_type.lower()}")
        elif error_type == "all_nan":
            ed["Platform"][f"latitude_{datagram_type.lower()}"].data = (
                [np.nan] * len(ed["Platform"][f"latitude_{datagram_type.lower()}"])
            )
    else:
        if error_type == "missing":
            ed["Platform"] = ed["Platform"].drop_vars("longitude")
        if error_type == "all_nan":
            ed["Platform"]["latitude"].data = (
                [np.nan] * len(ed["Platform"]["latitude"])
            )

    # Check if the expected error is logged
    with pytest.raises(ValueError) as exc_info:
        # Run add location with duplicated time
        ep.consolidate.add_location(ds=ds, echodata=ed, datagram_type=datagram_type)

    # Check expected error message is in the logs
    assert expected_error_message == str(exc_info.value)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("raw_path, sonar_model, datagram_type, parse_idx, compute_Sv_kwargs, expected_warnings"),
    [
        (
            "echopype/test_data/ek80/D20170912-T234910.raw",
            "EK80",
            "NMEA",
            False,
            {
                "waveform_mode": "BB",
                "encode_mode": "complex"
            },
            [
                (
                    "Coordinate variables contain NaN(s). "
                    "Interpolation may be negatively impacted, "
                    "consider handling these values before calling ``add_location``."
                ),
                (
                    "Coordinate variables contain zero(s). "
                    "Interpolation may be negatively impacted, "
                    "consider handling these values before calling ``add_location``."
                ),
            ]
        ),
        (
            "echopype/test_data/ek80/RL2407_ADCP-D20240709-T150437.raw",
            "EK80",
            "MRU1",
            False,
            {
                "waveform_mode": "CW",
                "encode_mode": "complex"
            },
            [
                (
                    "Coordinate variables contain NaN(s). "
                    "Interpolation may be negatively impacted, "
                    "consider handling these values before calling ``add_location``."
                ),
                (
                    "Coordinate variables contain zero(s). "
                    "Interpolation may be negatively impacted, "
                    "consider handling these values before calling ``add_location``."
                ),
            ]
        ),
        (
            "echopype/test_data/ek80/idx_bot/Hake-D20230711-T181910.raw",
            "EK80",
            "IDX",
            True,
            {
                "waveform_mode": "CW",
                "encode_mode": "power"
            },
            [
                (
                    "Coordinate variables contain NaN(s). " +
                    "Interpolation may be negatively impacted, " +
                    "consider handling these values before calling ``add_location``. "
                    "Consider setting datagram_type to any of [None]."
                ),
                (
                    "Coordinate variables contain zero(s). "
                    "Interpolation may be negatively impacted, "
                    "consider handling these values before calling ``add_location``. "
                    "Consider setting datagram_type to any of [None]."
                )
            ]
        ),
    ],
)
def test_add_location_lat_lon_0_NaN_warnings(
    raw_path, sonar_model, datagram_type, parse_idx, compute_Sv_kwargs, expected_warnings, caplog
):
    """Tests for lat lon 0 and NaN value warnings."""
    # Open raw and compute the Sv dataset
    if parse_idx:
        ed = ep.open_raw(raw_path, include_idx=True, sonar_model=sonar_model)
    else:
        ed = ep.open_raw(raw_path, sonar_model=sonar_model)
    ds = ep.calibrate.compute_Sv(
        echodata=ed,
        **compute_Sv_kwargs,
    )

    # Add NaN to latitude and 0 to longitude
    if datagram_type in ["MRU1", "IDX"]:
        ed["Platform"][f"latitude_{datagram_type.lower()}"][0] = np.nan
        ed["Platform"][f"longitude_{datagram_type.lower()}"][0] = 0
    else:
        ed["Platform"]["latitude"][0] = np.nan
        ed["Platform"]["longitude"][0] = 0

    # Turn on logger verbosity
    ep.utils.log.verbose(override=False)

    # Run add location with 0 and NaN lat/lon values
    ep.consolidate.add_location(ds=ds, echodata=ed, datagram_type=datagram_type)
    
    # Check if the expected warnings are logged
    for warning in expected_warnings:
        assert any(warning in record.message for record in caplog.records)
    
    # Turn off logger verbosity
    ep.utils.log.verbose(override=True)
