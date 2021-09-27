"""run-test.py

Script to run tests in Github.

"""
import argparse
import glob
import os
import re
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

MODULES_TO_TEST = {
    "root": {},  # This is to test the root folder.
    "convert": {},
    "calibrate": {},
    "echodata": {},
    "preprocess": {},
    "utils": {},
    "old": {"extra_globs": ["echopype/convert/convert.py", "echopype/process/*"]},
    "metrics": {},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests listed.")
    parser.add_argument(
        "touchedfiles",
        metavar="TOUCHED_FILES",
        type=str,
        nargs="?",
        default="",
        help="Comma separated list of changed files.",
    )
    parser.add_argument(
        "--pytest-args", type=str, help="Optional pytest args", default=""
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Optional flag for running tests locally, not in continuous integration.",
    )
    parser.add_argument(
        "--include-cov",
        action="store_true",
        help="Optional flag for including coverage. Exports to coverage.xml by default.",
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
        file_list = glob.glob(str(echopype_folder / "**" / "*.py"), recursive=True)
    else:
        file_list = args.touchedfiles.split(",")
    pytest_args = []
    if args.pytest_args:
        pytest_args = args.pytest_args.split(",")
        if args.include_cov:
            # Checks for cov in pytest_args
            for arg in pytest_args:
                if re.match("--cov", arg) is not None:
                    raise ValueError(
                        "pytest args may not have any cov arguments if --include-cov is set."
                    )
            pytest_args = pytest_args + [
                "--cov-report=xml",
                "--cov-append",
            ]
    test_to_run = {}
    for module, mod_extras in MODULES_TO_TEST.items():
        if module == "root":
            file_globs = [
                "echopype/*",
                "echopype/tests/*",
            ]
        else:
            file_globs = [
                f"echopype/{module}/*",
                f"echopype/tests/{module}/*",
            ]
        if "extra_globs" in mod_extras:
            file_globs = file_globs + mod_extras["extra_globs"]
        for f in file_list:
            file_path = Path(f)
            file_name, file_ext = os.path.splitext(os.path.basename(f))
            if file_ext == ".py":
                if any(((file_path.match(fg)) for fg in file_globs)):
                    if module not in test_to_run:
                        test_to_run[module] = []
                    test_to_run[module].append(file_path)

    original_pytest_args = pytest_args.copy()
    total_exit_codes = []
    for k, v in test_to_run.items():
        print(f"=== RUNNING {k.upper()} TESTS===")
        print(f"Touched files: {','.join([os.path.basename(p) for p in v])}")
        if k == "root":
            file_glob_str = "echopype/tests/test_*.py"
            cov_mod_arg = ["--cov=echopype"]
        else:
            file_glob_str = f"echopype/tests/{k}/*.py"
            cov_mod_arg = [f"--cov=echopype/{k}"]
        if args.include_cov:
            if k == "old":
                pytest_args = original_pytest_args + [
                    "--cov=echopype/convert",
                    "--cov=echopype/process",
                ]
            else:
                pytest_args = original_pytest_args + cov_mod_arg
        test_files = glob.glob(file_glob_str)
        final_args = pytest_args + test_files
        print(f"Pytest args: {final_args}")
        exit_code = pytest.main(final_args)
        total_exit_codes.append(EXIT_CODES[exit_code])

    if len(total_exit_codes) == 0:
        print("No test(s) were run.")
        sys.exit(0)
    if all(True if e == 0 else False for e in total_exit_codes):
        print("All test run successful")
        sys.exit(0)
    else:
        print("Some run failed. Please see log.")
        sys.exit(1)
