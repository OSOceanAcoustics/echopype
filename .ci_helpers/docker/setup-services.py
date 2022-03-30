"""setup-services.py

Script to help bring up docker services for testing.

"""
import argparse
import os
import shutil
import sys
from pathlib import Path

HERE = Path(".").absolute()
BASE = Path(__file__).parent.absolute()
COMPOSE_FILE = BASE / "docker-compose.yaml"
TEST_DATA_PATH = HERE / "echopype" / "test_data"


def parse_args():
    parser = argparse.ArgumentParser(description="Setup services for testing")
    parser.add_argument("--deploy", action="store_true", help="Flag to setup docker services")
    parser.add_argument(
        "--tear-down",
        action="store_true",
        help="Flag to tear down docker services",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if all([args.deploy, args.tear_down]):
        print("Cannot have both --deploy and --tear-down. Exiting.")
        sys.exit(1)

    if not any([args.deploy, args.tear_down]):
        print("Please provide either --deploy or --tear-down flags. For more help use --help flag.")
        sys.exit(0)

    if args.deploy:
        print("1) Starting test services deployment.")

        print("2) Pulling latest images.")
        os.system(f"docker-compose -f {COMPOSE_FILE} pull")

        print("3) Bringing up services.")
        os.system(f"docker-compose -f {COMPOSE_FILE} up -d --remove-orphans --force-recreate")

        print(f"4) Deleting old test folder at {TEST_DATA_PATH}")
        if TEST_DATA_PATH.exists():
            print("SKIPPED.")
            shutil.rmtree(TEST_DATA_PATH)

        print("5) Copying new test folder from http service")
        os.system(
            f"docker cp -L docker_httpserver_1:/usr/local/apache2/htdocs/data {TEST_DATA_PATH}"
        )

        print("6) Done.")
        os.system("docker ps --last 2")

    if args.tear_down:
        print("1) Stopping test services deployment.")
        os.system(f"docker-compose -f {COMPOSE_FILE} down --remove-orphans")
        print("2) Done.")
        os.system("docker ps --last 2")
