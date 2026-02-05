import subprocess
import sys
from pathlib import Path

CONTAINER_NAME = "echopype-test-http-server"
DOCKER_IMAGE = "cormorack/http"


def check_docker():
    try:
        subprocess.run(["docker", "--version"], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("Error: Docker is not installed or the daemon is not running.")
        sys.exit(1)


def run_container():
    subprocess.run(["docker", "run", "-d", "--name", CONTAINER_NAME, DOCKER_IMAGE], check=True)


def copy_test_data(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "docker", "cp", "-L",
        f"{CONTAINER_NAME}:/usr/local/apache2/htdocs/data/.",
        str(destination)
    ], check=True)


def cleanup_container():
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], check=True)


def main():
    script_dir = Path(__file__).resolve().parent
    destination_dir = script_dir.parent / "test_data"
    check_docker()
    run_container()

    print(f"Copying test data to {destination_dir} ...")
    copy_test_data(destination_dir)

    print("Cleaning up container...")
    cleanup_container()

    print(f"Test data downloaded successfully to {destination_dir}")


if __name__ == "__main__":
    main()
