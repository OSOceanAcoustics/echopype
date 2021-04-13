"""run-test.py

Script to run tests in Github.

"""
import os
import pytest
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run tests listed.')
    parser.add_argument('testfiles', metavar='TEST_FILES', type=str)
    args = parser.parse_args()
    file_list = args.testfiles.split(',')

    pytest_args = ["--log-cli-level=WARNING", "-vv"]
    test_files = []
    for f in file_list:
        file_name, file_ext = os.path.splitext(os.path.basename(f))
        if (file_ext == ".py") and ("test_" in file_name):
            test_files.append(f)

    final_args = pytest_args + test_files
    pytest.main(final_args)
