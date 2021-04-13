"""run-test.py

Script to run tests in Github.

"""
import argparse
import glob
import os
from pathlib import Path

import pytest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests listed.")
    parser.add_argument("touchedfiles", metavar="TOUCHED_FILES", type=str)
    args = parser.parse_args()
    file_list = args.touchedfiles.split(",")

    pytest_args = ["--log-cli-level=WARNING", "-vv"]
    test_to_run = {}
    for f in file_list:
        file_path = Path(f)
        file_name, file_ext = os.path.splitext(os.path.basename(f))
        if file_ext == ".py":
            if any(
                [
                    (file_path.match("echopype/convert/*")),
                    (file_path.match("echopype/tests/test_convert*")),
                ]
            ):
                if "convert" not in test_to_run:
                    test_to_run["convert"] = []
                test_to_run["convert"].append(file_path)
            elif any(
                [
                    (file_path.match("echopype/calibrate/*")),
                    (file_path.match("echopype/tests/test_calibrate*")),
                ]
            ):
                if "calibrate" not in test_to_run:
                    test_to_run["calibrate"] = []
                test_to_run["calibrate"].append(file_path)

    for k, v in test_to_run.items():
        print(f"=== RUNNING {k.upper()} TESTS===")
        print(f"Touched files: {','.join([os.path.basename(p) for p in v])}")
        test_files = glob.glob(f"echopype/tests/test_{k}*.py")
        final_args = pytest_args + test_files
        print(f"Pytest args: {final_args}")
        pytest.main(final_args)
