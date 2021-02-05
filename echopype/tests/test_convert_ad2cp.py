import json
from typing import List, Dict, Any

import numpy as np

from ..convert import ParseAd2cp

FILES = [
    # "./echopype/test_data/ad2cp/average_only.366.00000.ad2cp",
    "./echopype/test_data/ad2cp/avg_bur_echo.366.00000.ad2cp",
    "./echopype/test_data/ad2cp/burst_echosoun.366.00000.ad2cp",
    "./echopype/test_data/ad2cp/burst_only.366.00000.ad2cp",
]

BASE_FILES = [f"{f}.raw.json" for f in FILES]

class Utils:
    @staticmethod
    def make_json(parser: ParseAd2cp) -> List[Dict[str, Any]]:
        for packet in parser.packets:
            for k, v in packet.data.items():
                if isinstance(v, np.ndarray):
                    packet.data[k] = v.tolist()
        return [packet.data for packet in parser.packets]

def test_convert():
    for file, base_file in zip(FILES, BASE_FILES):
        parser = ParseAd2cp(file, None)
        parser.parse_raw()

        with open(base_file) as f:
            assert Utils.make_json(parser) == json.load(f), f"convert ad2cp failed on {file}"

# def create_base_files():
#     for file, base_file in zip(FILES, BASE_FILES):
#         parser = ParseAd2cp(file, None)
#         parser.parse_raw()

#         with open(base_file, "w") as f:
#             json.dump(Utils.make_json(parser), f)