import copy
import dask.array
import numpy as np

import pytest
from echopype.convert.parse_base import FILENAME_DATETIME_EK60, ParseBase, ParseEK, INDEX2POWER
from echopype.convert.utils.ek_swap import calc_final_shapes


class TestParseBase:
    file = "./my_file.raw"
    storage_options = {}
    sonar_model = "EK60"
    data_types = ["power", "angle", "complex"]
    raw_types = ["receive", "transmit"]

    def test_constructor(self):
        parser = ParseBase(
            file=self.file,
            storage_options=self.storage_options,
            sonar_model=self.sonar_model,
        )

        assert isinstance(parser, ParseBase)
        assert parser.source_file == self.file
        assert parser.sonar_model == self.sonar_model
        assert parser.storage_options == self.storage_options
        assert parser.data_types == self.data_types
        assert parser.raw_types == self.raw_types
        assert parser.timestamp_pattern is None
        assert parser.ping_time == []

    def test__print_status(self):
        parser = ParseBase(
            file=self.file,
            storage_options=self.storage_options,
            sonar_model=self.sonar_model,
        )
        assert parser._print_status() is None


class TestParseEK:
    file = "./my_file.raw"
    storage_options = {}
    data_types = ["power", "angle", "complex"]
    raw_types = ["receive", "transmit"]

    def _get_parser(self, sonar_model, ping_data_dict):
        parser = ParseEK(
            file=self.file,
            bot_file="",
            idx_file="",
            storage_options=self.storage_options,
            sonar_model=sonar_model,
        )
        parser.ping_data_dict = copy.deepcopy(ping_data_dict)
        parser.ping_time = parser.ping_data_dict["timestamp"]
        return parser

    @pytest.mark.parametrize("sonar_model", ["EK60", "EK80"])
    def test_constructor(self, sonar_model):
        parser = ParseEK(
            file=self.file,
            bot_file="",
            idx_file="",
            storage_options=self.storage_options,
            sonar_model=sonar_model,
        )

        assert isinstance(parser, ParseEK)
        assert parser.source_file == self.file
        assert parser.sonar_model == sonar_model
        assert parser.storage_options == self.storage_options
        assert parser.data_types == self.data_types
        assert parser.raw_types == self.raw_types
        assert parser.timestamp_pattern == FILENAME_DATETIME_EK60

    @pytest.mark.skip(reason="Test not implemented at this time.")
    def test_parse_raw(self):
        pass

    def _setup_rectangularize_data(self, mocker, sonar_model, use_swap, data_fixture):
        fixture_param, orig_data_dict = data_fixture
        parser = self._get_parser(sonar_model, orig_data_dict)

        # Patch use swap to either True or False
        # based on the test fixture parameterization
        mocker.patch(
            "echopype.convert.parse_base.ParseEK._ParseEK__should_use_swap", return_value=use_swap
        )

        parser.rectangularize_data(
            use_swap="auto",
            max_chunk_size="100MB",
        )
        return parser, orig_data_dict, fixture_param, mocker

    @pytest.mark.parametrize(
        ["use_swap", "expected_type"], [(True, dask.array.Array), (False, np.ndarray)]
    )
    @pytest.mark.parametrize("data_type", ["power", "angle"])
    def test_rectangularize_data_EK60(
        self, mocker, request, use_swap, expected_type, data_type, mock_ping_data_dict_power_angle
    ):
        sonar_model = "EK60"
        parser, orig_data_dict, fixture_param, mocker = self._setup_rectangularize_data(
            mocker, sonar_model, use_swap, mock_ping_data_dict_power_angle
        )

        # Check for each channel for the array values
        for ch, arr in parser.ping_data_dict[data_type].items():
            if arr is not None:
                # Check for expected type
                assert isinstance(arr, expected_type)

                # Check expanded shape
                if use_swap:
                    # Check for each channel for the array expansion shape
                    expected_final_shape = calc_final_shapes(self.data_types, orig_data_dict)
                    data_shape = expected_final_shape[data_type]
                    # Check that the array ping time dimension matches
                    assert len(parser.ping_time[ch]) == arr.shape[0]
                    assert arr.shape == (len(parser.ping_time[ch]),) + data_shape[1:]

                    # Load the dask array into memory
                    arr = arr.compute()
                else:
                    # This is separate since the array is not expanded until later point
                    # for across channels
                    assert parser.pad_shorter_ping(orig_data_dict[data_type][ch]).shape == arr.shape

                # Check for array values for regular dataset
                if fixture_param == "regular":
                    orig_arr = np.array(orig_data_dict[data_type][ch])
                    if data_type == "power":
                        # Do power conversion to ensure same data result
                        orig_arr = orig_arr.astype("float32") * INDEX2POWER
                    # Check if arrays are equal
                    assert np.array_equal(orig_arr, arr)
                # Check for array values for irregular dataset
                elif fixture_param == "irregular":
                    # Iterate through each range sample
                    for i in range(len(arr) - 1):
                        mask = ~np.isnan(arr[i])
                        if data_type == "power":
                            # Filter out all NaNs in array
                            darr = arr[i][mask]
                            if len(darr) > 0:
                                orig_arr = orig_data_dict[data_type][ch][i]
                                # Do power conversion to ensure same data result
                                orig_arr = orig_arr.astype("float32") * INDEX2POWER
                                assert np.array_equal(darr, orig_arr)
                        elif data_type == "angle":
                            # Filter out all Nans in array
                            # Create a 1D mask to filter out NaNs when both
                            # angle data is available
                            dmask = np.alltrue(mask, axis=1)
                            darr = arr[i][dmask]
                            if len(darr) > 0:
                                orig_arr = orig_data_dict[data_type][ch][i]
                                assert np.array_equal(darr, orig_arr)

    @pytest.mark.parametrize(
        ["use_swap", "expected_type"], [(True, dask.array.Array), (False, np.ndarray)]
    )
    @pytest.mark.parametrize("complex_part", ["real", "imag"])
    def test_rectangularize_data_EK80(
        self, mocker, request, use_swap, expected_type, complex_part, mock_ping_data_dict_complex
    ):
        sonar_model = "EK80"
        complex_keys = {
            "real": np.real,
            "imag": np.imag,
        }
        parser, orig_data_dict, fixture_param, mocker = self._setup_rectangularize_data(
            mocker, sonar_model, use_swap, mock_ping_data_dict_complex
        )

        for ch, arr_dct in parser.ping_data_dict["complex"].items():
            assert complex_part in arr_dct

            # Check if array is of expected type
            arr = arr_dct[complex_part]
            assert isinstance(arr, expected_type)

            if use_swap:
                # Check for each channel for the array expansion shape
                expected_final_shape = calc_final_shapes(self.data_types, orig_data_dict)
                data_shape = expected_final_shape["complex"]
                # Check that the array ping time dimension matches
                assert len(parser.ping_time[ch]) == arr.shape[0]
                assert arr.shape == (len(parser.ping_time[ch]),) + data_shape[1:]

                # Load the dask array into memory
                arr = arr.compute()
            else:
                # This is separate since the array is not expanded until later point
                # for across channels
                assert parser.pad_shorter_ping(orig_data_dict["complex"][ch]).shape == arr.shape

            # Check for array values for regular dataset
            if fixture_param == "regular":
                # Check if arrays are equal
                orig_arr = np.array(orig_data_dict["complex"][ch])
                orig_arr = complex_keys[complex_part](orig_arr)
                assert np.array_equal(arr, orig_arr)
            # Check for array values for irregular dataset
            elif fixture_param == "irregular":
                # Iterate through each range sample
                for i in range(len(arr) - 1):
                    mask = ~np.isnan(arr[i])
                    # Filter out all NaNs in array
                    darr = arr[i][mask]
                    if len(darr) > 0:
                        orig_arr = orig_data_dict["complex"][ch][i]
                        orig_arr = complex_keys[complex_part](orig_arr)
                        assert np.array_equal(darr, orig_arr)

    def test__parse_and_pad_datagram_no_zarr_root(self, mock_ping_data_dict_power_angle_simple):
        sonar_model = "EK60"
        parser = self._get_parser(sonar_model, mock_ping_data_dict_power_angle_simple)

        raw_type = "receive"
        expanded_data_shapes = parser._get_data_shapes()
        data_type_shapes = expanded_data_shapes[raw_type]

        expected_error_message = r"zarr_root cannot be None when use_swap is True"
        with pytest.raises(ValueError, match=expected_error_message):
            parser._parse_and_pad_datagram(
                data_type="power",
                data_type_shapes=data_type_shapes,
                raw_type=raw_type,
                use_swap=True,
                zarr_root=None,
            )
