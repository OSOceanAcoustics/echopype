from echopype.convert.parse_base import ParseBase

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
