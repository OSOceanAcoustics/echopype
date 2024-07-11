# Test conversion functionality that is the same for both EK60 and EK80.
import pytest
import numpy as np

from echopype import open_raw
from echopype.convert.utils.ek_raw_io import RawSimradFile, SimradEOF
from echopype.convert.parse_base import ParseEK


def expected_array_shape(file, datagram_type, datagram_item):
    """Extract array shape from user-specified parsed datagram type and item."""
    fid = RawSimradFile(file.replace("raw", datagram_type))
    fid.read(1)
    datagram_item_list = []
    while True:
        try:
            datagram_item_list.append(fid.read(1)[datagram_item])
        except SimradEOF:
            break
    return np.array(datagram_item_list).shape


@pytest.mark.integration
def test_convert_ek60_with_missing_bot_idx_file():
    """
    Check appropriate FileNotFoundError when attempting to parse .BOT and IDX file that do
    not exist in the same folder for which the .RAW file exists."""
    with pytest.raises(FileNotFoundError):
        open_raw(
            "echopype/test_data/ek60/ncei-wcsd/SH1701/TEST-D20170114-T202932.raw",
            sonar_model="EK60",
            include_bot=True,
        )
    with pytest.raises(FileNotFoundError):
        open_raw(
            "echopype/test_data/ek80/ncei-wcsd/SH2106/EK80/Reduced_Hake-D20210701-T131325.raw",
            sonar_model="EK80",
            include_idx=True,
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    "file, sonar_model",
    [
        ("echopype/test_data/ek60/idx_bot/Summer2017-D20170620-T011027.raw", "EK60"),
        ("echopype/test_data/ek60/idx_bot/Summer2017-D20170707-T150923.raw", "EK60"),
        ("echopype/test_data/ek80/idx_bot/Hake-D20230711-T181910.raw", "EK80"),
        ("echopype/test_data/ek80/idx_bot/Hake-D20230711-T182702.raw", "EK80"),
    ]
)
def test_convert_ek_with_bot_file(file, sonar_model):
    """Check variable dimensions, time encodings, and attributes when BOT file is parsed."""
    # Open Raw and Parse BOT
    ed = open_raw(
        file,
        sonar_model=sonar_model,
        include_bot=True,
    )

    # Check data variable shape
    seafloor_depth_da = ed["Vendor_specific"]["detected_seafloor_depth"]
    parsed_seafloor_depth_shape = expected_array_shape(file, "bot", "depth")
    assert len(seafloor_depth_da["ping_time"]) == parsed_seafloor_depth_shape[0]
    assert len(seafloor_depth_da["channel"]) == parsed_seafloor_depth_shape[1]

    # Check time encodings
    time_encoding = ed["Vendor_specific"]["ping_time"].encoding
    assert time_encoding["units"] == "nanoseconds since 1970-01-01T00:00:00Z"
    assert time_encoding["calendar"] == "gregorian"
    assert time_encoding["dtype"] == "int64"

    # Check `detected_seafloor_depth` attribute
    assert (
        ed["Vendor_specific"]["detected_seafloor_depth"].attrs[
            "long_name"
        ] == "Echosounder detected seafloor depth from the BOT datagrams."
    )

    # Check time attributes
    time_attrs = ed["Vendor_specific"]["ping_time"].attrs
    assert time_attrs["long_name"] == "Timestamps from the BOT datagrams"
    assert time_attrs["standard_name"] == "time"
    assert time_attrs["axis"] == "T"
    assert time_attrs["comment"] == "Time coordinate corresponding to seafloor detection data."


@pytest.mark.integration
@pytest.mark.parametrize(
    "file, sonar_model",
    [
        ("echopype/test_data/ek60/idx_bot/Summer2017-D20170620-T011027.raw", "EK60"),
        ("echopype/test_data/ek60/idx_bot/Summer2017-D20170707-T150923.raw", "EK60"),
        ("echopype/test_data/ek80/idx_bot/Hake-D20230711-T181910.raw", "EK80"),
        ("echopype/test_data/ek80/idx_bot/Hake-D20230711-T182702.raw", "EK80"),
    ]
)
def test_convert_ek_with_idx_file(file, sonar_model):
    """Check variable dimensions and attributes when IDX file is parsed."""
    # Open Raw and Parse IDX
    ed = open_raw(
        file,
        sonar_model=sonar_model,
        include_idx=True,
    )
    platform = ed["Platform"]

    # Check data variable lengths
    assert (
        len(platform["idx_ping_number"]) == \
        len(platform["idx_file_offset"]) == \
        len(platform["idx_vessel_distance"]) == \
        len(platform["idx_latitude"]) == \
        len(platform["idx_longitude"]) == \
        expected_array_shape(file, "idx", "ping_number")[0] == \
        expected_array_shape(file, "idx", "file_offset")[0] == \
        expected_array_shape(file, "idx", "distance")[0] == \
        expected_array_shape(file, "idx", "latitude")[0] == \
        expected_array_shape(file, "idx", "longitude")[0]
    )

    # Check attributes (sanity check)
    platform["time3"].attrs == (
        {
            "axis": "T",
            "long_name": "Timestamps from IDX datagrams",
            "standard_name": "time",
            "comment": "Time coordinate corresponding to index file vessel "
            + "distance and latitude/longitude data.",
        }
    )
    assert platform["idx_vessel_distance"].attrs == (
        {
            "long_name": "Vessel distance in nautical miles (nmi) from start of recording.",
            "comment": "Data from the IDX datagrams. Aligns time-wise with this "
            + "dataset's `time3` dimension.",
        }
    )
    assert platform["idx_latitude"].attrs == (
        {
            "long_name": "Index File Derived Platform Latitude",
            "comment": "Data from the IDX datagrams. Aligns time-wise with this "
            + "dataset's `time3` dimension. "
            + "This is different from latitude stored in the NMEA datagram.",
        }
    )
    assert platform["idx_longitude"].attrs == (
        {
            "long_name": "Index File Derived Platform Longitude",
            "comment": "Data from the IDX datagrams. Aligns time-wise with this "
            + "dataset's `time3` dimension. "
            + "This is different from longitude from the NMEA datagram.",
        }
    )


@pytest.mark.unit
def test_pad_short_complex_pings():
    """
    Test padding of short complex pings.
    """
    # Create empty parser
    empty_parser = ParseEK(None, None, None, None, None)

    # Create ping list with sizes 3, 2, 1 samples per ping
    data_list = [
        np.array(
            [
                1.0 + 2.5j,
                7.3 - 3.2j,
                3.2 - 9.1j,
            ],
            dtype=np.complex64
        ),
        np.array(
            [
                1.8 - 4.1j,
                1.2 - 8.9j,
            ],
            dtype=np.complex64
        ),
        np.array(
            [
                6.1 - 4.8j,
            ],
            dtype=np.complex64
        ),
    ]

    # Pad shorter pings
    output = empty_parser.pad_shorter_ping(data_list)

    # Set expected output
    expected_output = np.array(
        [
            [1. +2.5j, 7.3-3.2j, 3.2-9.1j],
            [1.8-4.1j, 1.2-8.9j, np.nan+0.j ],
            [6.1-4.8j, np.nan+0.j , np.nan+0.j ]
        ],
        dtype=np.complex64
    )
    # Check dtype
    assert output.dtype == np.complex64

    # Check output against expected
    assert np.allclose(
        output,
        expected_output,
        equal_nan=True
    )
