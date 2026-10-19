# .ci_helpers/setup-services-windows.py
"""
Spin up local services for Windows CI without Docker:
- Start a SeaweedFS server on localhost:9000
- Seed S3 from the Pooch cache (unzipped test bundles)
- Start a simple HTTP server on :8080 that exposes /data
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
import tempfile
import time
import zipfile
from urllib.request import urlretrieve

import fsspec
import pooch

SEAWEEDFS_VERSION = "4.48"
SEAWEEDFS_URL = (
    "https://github.com/seaweedfs/seaweedfs/releases/download/"
    f"{SEAWEEDFS_VERSION}/windows_amd64.zip"
)
SEAWEEDFS_BIN = pathlib.Path(".ci_helpers") / f"seaweedfs-{SEAWEEDFS_VERSION}" / "weed.exe"
STATE_DIR = pathlib.Path(".ci_helpers") / ".state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
S3_PID = STATE_DIR / "seaweedfs.pid"
HTTP_PID = STATE_DIR / "http.pid"

HTTP_ROOT = pathlib.Path(tempfile.gettempdir()) / "echopype-test-services"
HTTP_DATA = HTTP_ROOT / "data"

# Use localhost everywhere to match tests.
S3_ENDPOINT = "http://localhost:9000/"
S3_USER = "s3admin"
S3_PASS = "s3admin"


def get_pooch_cache() -> pathlib.Path:
    """Return the Pooch cache dir for the configured dataset version."""
    ver = os.getenv("ECHOPYPE_DATA_VERSION", "v0.11.1a2")
    root = pathlib.Path(pooch.os_cache("echopype"))
    path = root / ver
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_seaweedfs_downloaded() -> None:
    if SEAWEEDFS_BIN.exists():
        return
    SEAWEEDFS_BIN.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SeaweedFS -> {SEAWEEDFS_BIN}", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = pathlib.Path(tmp) / "seaweedfs.zip"
        urlretrieve(SEAWEEDFS_URL, archive)
        with zipfile.ZipFile(archive) as release:
            # Extract only the executable required by this helper.
            SEAWEEDFS_BIN.write_bytes(release.read("weed.exe"))
    SEAWEEDFS_BIN.chmod(0o755)


def start_seaweedfs() -> None:
    """Start SeaweedFS on localhost:9000 and wait until ready."""
    ensure_seaweedfs_downloaded()
    data_dir = (
        pathlib.Path(os.getenv("USERPROFILE", str(pathlib.Path.home())))
        / "echopype-seaweedfs"
        / "data"
    )
    data_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = S3_USER
    env["AWS_SECRET_ACCESS_KEY"] = S3_PASS

    print(f"Starting SeaweedFS on {S3_ENDPOINT} (data: {data_dir})", flush=True)
    proc = subprocess.Popen(
        [
            str(SEAWEEDFS_BIN),
            "mini",
            f"-dir={data_dir}",
            "-ip=127.0.0.1",
            "-ip.bind=127.0.0.1",
            "-s3.port=9000",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    S3_PID.write_text(str(proc.pid))

    # Wait for the authenticated S3 API, not just the process to start.
    fs = fsspec.filesystem(
        "s3",
        client_kwargs=dict(endpoint_url=S3_ENDPOINT),
        key=S3_USER,
        secret=S3_PASS,
    )
    for attempt in range(60):
        if proc.poll() is not None:
            raise RuntimeError("SeaweedFS exited before the S3 endpoint became ready")
        try:
            fs.ls("", refresh=True)
            return
        except Exception:
            if attempt == 59:
                proc.terminate()
                proc.wait(timeout=10)
                S3_PID.unlink(missing_ok=True)
                raise RuntimeError("SeaweedFS S3 did not become ready on :9000")
            time.sleep(1)


def seed_s3_from_pooch() -> None:
    """Upload unzipped Pooch bundles into SeaweedFS and prepare HTTP test data."""
    cache = get_pooch_cache()

    # Seed S3 (SeaweedFS)
    fs = fsspec.filesystem(
        "s3",
        client_kwargs=dict(endpoint_url=S3_ENDPOINT),
        key=S3_USER,
        secret=S3_PASS,
    )

    for base in ("data", "data-output"):
        if not fs.exists(base):
            fs.mkdir(base)

    for d in cache.iterdir():
        if d.suffix == ".zip":
            continue
        tgt = f"data/{d.name}"
        print(f"Uploading {d} -> {tgt}", flush=True)
        fs.put(str(d), tgt, recursive=True)

    # Build a temporary data mirror for HTTP tests that hit
    # http://localhost:8080/data/...
    shutil.rmtree(HTTP_ROOT, ignore_errors=True)
    HTTP_DATA.mkdir(parents=True, exist_ok=True)

    for d in cache.iterdir():
        if d.suffix == ".zip":
            continue
        dst = HTTP_DATA / d.name
        if d.is_dir():
            shutil.copytree(d, dst)
        else:
            shutil.copy2(d, dst)


def start_http_server() -> None:
    """Serve the temporary HTTP root so /data/... exists on :8080."""
    root = HTTP_ROOT.absolute()
    print(f"Starting local HTTP server on :8080 (root={root})", flush=True)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            "8080",
            "--directory",
            str(root),
        ],
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
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
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
    start_seaweedfs()
    seed_s3_from_pooch()
    if not no_http:
        start_http_server()
    print("Services up.", flush=True)


def cmd_stop() -> None:
    stop_pid_file(HTTP_PID)
    stop_pid_file(S3_PID)
    shutil.rmtree(HTTP_ROOT, ignore_errors=True)
    print("Services stopped.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument(
        "--no-http",
        action="store_true",
        help="Do not start the local HTTP server",
    )
    sub.add_parser("stop")
    args = ap.parse_args()
    if args.cmd == "start":
        cmd_start(no_http=args.no_http)
    else:
        cmd_stop()


if __name__ == "__main__":
    main()
