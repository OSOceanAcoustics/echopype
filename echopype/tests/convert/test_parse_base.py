import dask.array
import numpy as np

import pytest
from echopype.convert.parse_base import FILENAME_DATETIME_EK60, ParseBase, ParseEK

class TestParseBase:
    file="./my_file.raw"
    storage_options={}
    sonar_model="EK60"
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

    @pytest.fixture
    def mock_ek60_parser_obj(self, mocker, mock_ping_data_dict_power_angle):
        parser = ParseEK(
            file=self.file,
            params=self.params,
            storage_options=self.storage_options,
            sonar_model="EK60",
        )
        parser.ping_data_dict = mock_ping_data_dict_power_angle
        parser.ping_time = mock_ping_data_dict_power_angle["timestamp"]
        return parser
    
    @pytest.fixture
    def mock_ek80_parser_obj(self, mocker, mock_ping_data_dict_complex):
        parser = ParseEK(
            file=self.file,
            params=self.params,
            storage_options=self.storage_options,
            sonar_model="EK80",
        )
        parser.ping_data_dict = mock_ping_data_dict_complex
        parser.ping_time = mock_ping_data_dict_complex["timestamp"]
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
    
    @pytest.mark.parametrize("sonar_model", ["EK60", "EK80"])
    @pytest.mark.parametrize(
        ["use_swap", "expected_type"],
        [
            (True, dask.array.Array),
            (False, np.ndarray)
        ]
    )
    def test_rectangularize_data(self, mocker, request, sonar_model, use_swap, expected_type):
        """
        parser.rectangularize_data(
            dest_path=destination_path,
            dest_storage_options=destination_storage_options,
            max_chunk_size=max_chunk_size,
        )
        """
        if sonar_model == "EK60":
            parser = request.getfixturevalue("mock_ek60_parser_obj")
            
            for data_type in ["power", "angle"]:
                assert data_type in parser.ping_data_dict
            assert "complex" not in parser.ping_data_dict
        elif sonar_model == "EK80":
            parser = request.getfixturevalue("mock_ek80_parser_obj")
            assert "complex" in parser.ping_data_dict
        else:
            pytest.skip(reason=f"Not implement for {sonar_model}")
        
        # Patch use swap to either True or False
        # based on the test fixture parameterization
        mocker.patch(
            "echopype.convert.parse_base.ParseEK._ParseEK__should_use_swap", 
            return_value=use_swap
        )
        
        parser.rectangularize_data(
            dest_path="auto",
            dest_storage_options=None,
            max_chunk_size="100MB",
        )
        
        # Check for expected data types
        if sonar_model == "EK60":
            for data_type in ["power", "angle"]:
                for val in parser.ping_data_dict[data_type].values():
                    assert isinstance(val, expected_type)
        else:
            complex_keys = ["real", "imag"]
            for key in complex_keys:
                for val in parser.ping_data_dict["complex"].values():
                    assert key in val
                    assert isinstance(val[key], expected_type)
                    
        # TODO: Check for data shapes
        # TODO: Check for actual data values
            
            
    
    
