from collections import defaultdict
from struct import pack
from unittest.mock import MagicMock, patch

import pytest

from echopype.convert.parse_bi500 import ParseBI500

pytestmark = pytest.mark.unit


@pytest.fixture
def parser():
    """ParseBI500 instance with filesystem validation bypassed."""
    with patch.object(ParseBI500, "_validate_folder_path", return_value=MagicMock()):
        yield ParseBI500(file="/fake/bi500_folder")


class TestValidateFolderPath:
    def _make_parser(self):
        parser = ParseBI500.__new__(ParseBI500)
        parser.storage_options = {}
        parser.file_types = ["-Data", "-Info", "-Ping", "-Vlog", "-Snap", "-Work"]
        parser.file_type_map = defaultdict(None)
        return parser

    @patch("echopype.convert.parse_bi500.fsspec.get_mapper")
    def test_raises_when_path_is_not_a_directory(self, mock_get_mapper):
        parser = self._make_parser()
        mock_get_mapper.return_value.fs.ls.side_effect = NotADirectoryError()

        with pytest.raises(ValueError, match="Expecting a folder"):
            parser._validate_folder_path("/not/a/folder")

    @patch("echopype.convert.parse_bi500.fsspec.get_mapper")
    def test_raises_when_required_files_are_missing(self, mock_get_mapper):
        parser = self._make_parser()
        mock_get_mapper.return_value.fs.ls.return_value = ["/fake/unrelated.txt"]

        with pytest.raises(ValueError, match="required file missing"):
            parser._validate_folder_path("/fake/folder")


class TestParseRaw:
    def test_zero_trace_count_appends_zero_placeholders(self, parser):
        parser.index_counts = {
            "pelagic_count": [1],
            "bottom_count": [0],
            "echotrace_count": [0],
        }
        parser.file_type_map["-Data"] = "/fake/data"
        parser.fsmap.fs.open.return_value.read.return_value = pack(">h", 100)

        with patch.object(parser, "load_BI500_info"), patch.object(
            parser, "load_BI500_ping"
        ), patch.object(parser, "load_BI500_vlog"):
            parser.parse_raw()

        assert len(parser.unpacked_data["pelagic"]) == 1
        assert parser.unpacked_data["TargetDepth"] == [0.0]
        assert parser.unpacked_data["CompTS"] == [0.0]
        assert parser.unpacked_data["UncompTS"] == [0.0]
        assert parser.unpacked_data["Alongship"] == [0.0]
        assert parser.unpacked_data["Athwartship"] == [0.0]
