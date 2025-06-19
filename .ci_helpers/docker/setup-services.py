"""setup-services.py

Script to help bring up docker services for testing.

"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import fsspec

logger = logging.getLogger("setup-services")
streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(message)s")
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)
logger.setLevel(level=logging.INFO)

HERE = Path(".").absolute()
BASE = Path(__file__).parent.absolute()
COMPOSE_FILE = BASE / "docker-compose.yaml"
TEST_DATA_PATH = HERE / "echopype" / "test_data"


def parse_args():
    parser = argparse.ArgumentParser(description="Setup services for testing")
    parser.add_argument("--deploy", action="store_true", help="Flag to setup docker services")
    parser.add_argument(
        "--http-server",
        default="docker-httpserver-1",
        help="Flag for specifying docker http server id.",
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Optional flag to skip pulling the latest images from dockerhub",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="""Optional flag to only copy over data to http server,
        and setup minio bucket and not deploy any services. NOTE: MUST HAVE SERVICES RUNNING!""",
    )
    parser.add_argument(
        "--tear-down",
        action="store_true",
        help="Flag to tear down docker services",
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help="Optional flag to remove images also during tear down",
    )

    return parser.parse_args()


def run_commands(commands: List[Dict]) -> None:
    for idx, command in enumerate(commands, start=1):
        msg = command.get("msg")
        cmd = command.get("cmd")
        args = command.get("args", None)
        logger.info(f"{idx}) {msg}")
        if cmd is None:
            continue
        elif isinstance(cmd, list):
            subprocess.run(cmd)
        elif callable(cmd):
            cmd(args)
        else:
            raise ValueError(f"command of {type(cmd)} is invalid.")


def load_s3(*args, **kwargs) -> None:
    common_storage_options = dict(
        client_kwargs=dict(endpoint_url="http://localhost:9000/"),
        key="minioadmin",
        secret="minioadmin",
    )
    bucket_name = "ooi-raw-data"
    fs = fsspec.filesystem(
        "s3",
        **common_storage_options,
    )
    test_data = "data"
    if not fs.exists(test_data):
        fs.mkdir(test_data)

    if not fs.exists(bucket_name):
        fs.mkdir(bucket_name)

    # Load test data into bucket
    for d in TEST_DATA_PATH.iterdir():
        source_path = f"echopype/test_data/{d.name}"
        fs.put(source_path, f"{test_data}/{d.name}", recursive=True)


if __name__ == "__main__":
    args = parse_args()
    commands = []
    if all([args.deploy, args.tear_down]):
        print("Cannot have both --deploy and --tear-down. Exiting.")
        sys.exit(1)

    if not any([args.deploy, args.tear_down]):
        print("Please provide either --deploy or --tear-down flags. For more help use --help flag.")
        sys.exit(0)

    if args.deploy:
        commands.append({"msg": "Starting test services deployment ...", "cmd": None})
        if not args.data_only:
            if not args.no_pull:
                commands.append(
                    {
                        "msg": "Pulling latest images ...",
                        "cmd": ["docker", "compose", "-f", COMPOSE_FILE, "pull"],
                    }
                )
            commands.append(
                {
                    "msg": "Bringing up services ...",
                    "cmd": [
                        "docker",
                        "compose",
                        "-f",
                        COMPOSE_FILE,
                        "up",
                        "-d",
                        "--remove-orphans",
                        "--force-recreate",
                    ],
                }
            )

        if TEST_DATA_PATH.exists():
            commands.append(
                {
                    "msg": f"Deleting old test folder at {TEST_DATA_PATH} ...",
                    "cmd": shutil.rmtree,
                    "args": TEST_DATA_PATH,
                }
            )
        commands.append(
            {
                "msg": "Copying new test folder from http service ...",
                "cmd": [
                    "docker",
                    "cp",
                    "-L",
                    f"{args.http_server}:/usr/local/apache2/htdocs/data",
                    TEST_DATA_PATH,
                ],
            }
        )

        commands.append({"msg": "Setting up minio s3 bucket ...", "cmd": load_s3})

    if args.tear_down:
        command = ["docker", "compose", "-f", COMPOSE_FILE, "down", "--remove-orphans", "--volumes"]
        if args.images:
            command = command + ["--rmi", "all"]
        commands.append({"msg": "Stopping test services deployment ...", "cmd": command})
    commands.append({"msg": "Done.", "cmd": ["docker", "ps", "--last", "2"]})
    run_commands(commands)
