import argparse
import sys

import echopype

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check current echopype version.")
    parser.add_argument(
        "expected_version",
        type=str,
        nargs="?",
        default="0.5.0",
        help="Expected Echopype Version to check",
    )
    args = parser.parse_args()
    expected_version = args.expected_version
    installed_version = echopype.__version__
    if installed_version != expected_version:
        print(
            f"!! Installed version {installed_version} does not match expected version {expected_version}."  # noqa
        )
        sys.exit(1)
    else:
        print(f"Installed version {installed_version} is expected.")
        sys.exit(0)
