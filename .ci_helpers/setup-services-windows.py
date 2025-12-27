# .ci_helpers/setup-services-windows.py
"""
Spin up local services for Windows CI without Docker:
- Start a MinIO server on localhost:9000
- Seed S3 from the Pooch cache (unzipped test bundles)
- Start a simple HTTP server on :8080 that exposes ./data (mirrors Linux job)
- Write PID files for clean teardown

Usage:
  python .ci_helpers/setup-services-windows.py start [--no-http]
  python .ci_helpers/setup-services-windows.py stop
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import time
from urllib.request import urlretrieve

import fsspec
import pooch

MINIO_URL = "https://dl.min.io/server/minio/release/windows-amd64/minio.exe"
MINIO_BIN = pathlib.Path(".ci_helpers") / "minio.exe"
STATE_DIR = pathlib.Path(".ci_helpers") / ".state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
MINIO_PID = STATE_DIR / "minio.pid"
HTTP_PID = STATE_DIR / "http.pid"

# Use localhost everywhere to match tests (which hit http://localhost:9000/...)
MINIO_ENDPOINT = "http://localhost:9000/"
MINIO_USER = "minioadmin"
MINIO_PASS = "minioadmin"


def get_pooch_cache() -> pathlib.Path:
    """Return the Pooch cache dir for the configured dataset version."""
    ver = os.getenv("ECHOPYPE_DATA_VERSION", "v0.11.1a2")
    root = pathlib.Path(pooch.os_cache("echopype"))
    path = root / ver
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_minio_downloaded() -> None:
    if MINIO_BIN.exists():
        return
    MINIO_BIN.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MinIO -> {MINIO_BIN}", flush=True)
    urlretrieve(MINIO_URL, MINIO_BIN)
    MINIO_BIN.chmod(0o755)


def start_minio() -> None:
    """Start MinIO on localhost:9000 and wait until ready."""
    ensure_minio_downloaded()
    data_dir = pathlib.Path(os.getenv("USERPROFILE", str(pathlib.Path.home()))) / "minio" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MINIO_ROOT_USER"] = MINIO_USER
    env["MINIO_ROOT_PASSWORD"] = MINIO_PASS

    print(f"Starting MinIO on {MINIO_ENDPOINT} (data: {data_dir})", flush=True)
    proc = subprocess.Popen(
        [
            str(MINIO_BIN),
            "server",
            str(data_dir),
            "--address=localhost:9000",
            "--console-address=localhost:9001",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    MINIO_PID.write_text(str(proc.pid))

    # Wait for readiness
    import urllib.request

    for _ in range(60):
        try:
            with urllib.request.urlopen(MINIO_ENDPOINT + "minio/health/ready", timeout=1):
                break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("MinIO did not become ready on :9000")


def seed_s3_from_pooch() -> None:
    """Upload unzipped Pooch bundles into MinIO under the 'data/' prefix,
    and also mirror a local ./data/ folder for the HTTP server."""
    cache = get_pooch_cache()

    # Seed S3 (MinIO)
    fs = fsspec.filesystem(
        "s3",
        client_kwargs=dict(endpoint_url=MINIO_ENDPOINT),
        key=MINIO_USER,
        secret=MINIO_PASS,
    )

    for base in ("data", "ooi-raw-data"):
        try:
            fs.mkdir(base)
        except Exception:
            pass

    for d in cache.iterdir():
        if d.suffix == ".zip":
            continue
        tgt = f"data/{d.name}"
        print(f"Uploading {d} -> {tgt}", flush=True)
        fs.put(str(d), tgt, recursive=True)

    # Build local ./data for HTTP tests that hit http://localhost:8080/data/...
    local_data = pathlib.Path("data")
    local_data.mkdir(exist_ok=True)
    for d in cache.iterdir():
        if d.suffix == ".zip":
            continue
        dst = local_data / d.name
        if not dst.exists():
            if d.is_dir():
                shutil.copytree(d, dst)
            else:
                shutil.copy2(d, dst)


def start_http_server() -> None:
    """Serve the repository root so /data/... exists on :8080."""
    root = pathlib.Path(".").absolute()
    print(f"Starting local HTTP server on :8080 (root={root})", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8080", "--directory", str(root)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    HTTP_PID.write_text(str(proc.pid))


def stop_pid_file(path: pathlib.Path) -> None:
    try:
        pid = int(path.read_text().strip())
    except Exception:
        return
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)
        else:
            os.kill(pid, 9)
    except Exception:
        pass
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def cmd_start(no_http: bool) -> None:
    # Ensure cache exists (prefetch step should have populated it)
    _ = get_pooch_cache()
    start_minio()
    seed_s3_from_pooch()
    if not no_http:
        start_http_server()
    print("Services up.", flush=True)


def cmd_stop() -> None:
    stop_pid_file(HTTP_PID)
    stop_pid_file(MINIO_PID)
    print("Services stopped.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--no-http", action="store_true", help="Do not start the local HTTP server")
    sub.add_parser("stop")
    args = ap.parse_args()
    if args.cmd == "start":
        cmd_start(no_http=args.no_http)
    else:
        cmd_stop()


if __name__ == "__main__":
    main()
