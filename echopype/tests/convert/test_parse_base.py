import copy
import dask.array
import numpy as np

import pytest
from echopype.convert.parse_base import FILENAME_DATETIME_EK60, ParseBase, ParseEK, INDEX2POWER


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
    params = "ALL"
    data_types = ["power", "angle", "complex"]
    raw_types = ["receive", "transmit"]

    def _get_parser(self, sonar_model, ping_data_dict):
        parser = ParseEK(
            file=self.file,
            params=self.params,
            storage_options=self.storage_options,
            sonar_model=sonar_model,
        )
        parser.ping_data_dict = copy.deepcopy(ping_data_dict)
        parser.ping_time = parser.ping_data_dict["timestamp"]
        return parser

    @pytest.fixture
    def mock_ek60_parser_obj(self, mocker, mock_ping_data_dict_power_angle):
        parser = ParseEK(
            file=self.file,
            params=self.params,
            storage_options=self.storage_options,
            sonar_model="EK60",
        )
        parser.ping_data_dict = copy.deepcopy(mock_ping_data_dict_power_angle)
        parser.ping_time = parser.ping_data_dict["timestamp"]
        return parser

    @pytest.fixture
    def mock_ek80_parser_obj(self, mocker, mock_ping_data_dict_complex):
        parser = ParseEK(
            file=self.file,
            params=self.params,
            storage_options=self.storage_options,
            sonar_model="EK80",
        )
        parser.ping_data_dict = copy.deepcopy(mock_ping_data_dict_complex)
        parser.ping_time = parser.ping_data_dict["timestamp"]
        return parser

    @pytest.mark.parametrize("sonar_model", ["EK60", "EK80"])
    def test_constructor(self, sonar_model):
        parser = ParseEK(
            file=self.file,
            params=self.params,
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
        sonar_model = sonar_model
        fixture_param, orig_data_dict = data_fixture
        parser = self._get_parser(sonar_model, orig_data_dict)

        # Patch use swap to either True or False
        # based on the test fixture parameterization
        mocker.patch(
            "echopype.convert.parse_base.ParseEK._ParseEK__should_use_swap", return_value=use_swap
        )

        parser.rectangularize_data(
            dest_path="auto",
            dest_storage_options=None,
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

        for ch, arr in parser.ping_data_dict[data_type].items():
            if arr is not None:
                # For cases that angle data is not available
                assert isinstance(arr, expected_type)
                if use_swap:
                    # Load the dask array into memory
                    arr = arr.compute()

                if fixture_param == "regular":
                    orig_arr = np.array(orig_data_dict[data_type][ch])
                    if data_type == "power":
                        # Do power conversion to ensure same data result
                        orig_arr = orig_arr.astype("float32") * INDEX2POWER
                    # Check if arrays are equal
                    assert np.array_equal(orig_arr, arr)
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
                # Load the dask array into memory
                arr = arr.compute()
            
            if fixture_param == "regular":
                # Check if arrays are equal
                orig_arr = np.array(orig_data_dict["complex"][ch])
                orig_arr = complex_keys[complex_part](orig_arr)
                assert np.array_equal(arr, orig_arr)
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