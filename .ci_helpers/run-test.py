"""run-test.py

Script to run tests in Github.

"""
import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

import pytest
from pytest import ExitCode

EXIT_CODES = {
    ExitCode.OK: 0,
    ExitCode.TESTS_FAILED: 1,
    ExitCode.INTERRUPTED: 2,
    ExitCode.INTERNAL_ERROR: 3,
    ExitCode.USAGE_ERROR: 4,
    ExitCode.NO_TESTS_COLLECTED: 5,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests listed.")
    parser.add_argument(
        "touchedfiles",
        metavar="TOUCHED_FILES",
        type=str,
        nargs="?",
        default="",
        help="Comma separated list of changed files."
    )
    parser.add_argument(
        "--pytest-args", type=str, help="Optional pytest args", default=""
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Optional flag for running tests locally, not in continuous integration.",
    )
    args = parser.parse_args()
    if args.local:
        temp_path = Path("temp_echopype_output")
        dump_path = Path("echopype/test_data/dump")
        if temp_path.exists():
            shutil.rmtree(temp_path)

        if dump_path.exists():
            shutil.rmtree(dump_path)
        echopype_folder = Path("echopype")
        file_list = glob.glob(str(echopype_folder / "**" / "*.py"))
    else:
        file_list = args.touchedfiles.split(",")
    pytest_args = []
    if args.pytest_args:
        pytest_args = args.pytest_args.split(",")
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
            elif any(
                [
                    (file_path.match("echopype/echodata/*")),
                    (file_path.match("echopype/tests/test_echodata*")),
                ]
            ):
                if "echodata" not in test_to_run:
                    test_to_run["echodata"] = []
                test_to_run["echodata"].append(file_path)

    total_exit_codes = []
    for k, v in test_to_run.items():
        print(f"=== RUNNING {k.upper()} TESTS===")
        print(f"Touched files: {','.join([os.path.basename(p) for p in v])}")
        test_files = glob.glob(f"echopype/tests/test_{k}*.py")
        final_args = pytest_args + test_files
        print(f"Pytest args: {final_args}")
        exit_code = pytest.main(final_args)
        total_exit_codes.append(EXIT_CODES[exit_code])

    if len(total_exit_codes) == 0:
        print("No test(s) were run.")
        sys.exit(0)
    if all([True if e == 0 else False for e in total_exit_codes]):
        print("All test run successful")
        sys.exit(0)
    else:
        print("Some run failed. Please see log.")
        sys.exit(1)
