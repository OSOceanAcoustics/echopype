"""pytest configuration with minimal Pooch fallback for CI"""

import os
import pytest
from pathlib import Path

if os.getenv("USE_POOCH") == "True":
    import pooch

    # Lock to the known-good assets release (can be overridden via env if needed)
    ver = os.getenv("ECHOPYPE_DATA_VERSION", "v0.11.1a1")
    base = os.getenv(
        "ECHOPYPE_DATA_BASEURL",
        "https://github.com/OSOceanAcoustics/echopype/releases/download/{version}/",
    )
    cache_dir = pooch.os_cache("echopype")

    print(
        "\n[echopype-ci] POOCH CONFIG\n"
        f"  USE_POOCH           = {os.getenv('USE_POOCH')}\n"
        f"  ECHOPYPE_DATA_VERSION = {ver}\n"
        f"  ECHOPYPE_DATA_BASEURL = {base}\n"
        f"  pooch cache_dir       = {cache_dir}\n",
        flush=True,
    )

    bundles = [
        "ad2cp.zip", "azfp.zip", "azfp6.zip", "ea640.zip", "ecs.zip", "ek60.zip",
        "ek60_calibrate_chunks.zip", "ek60_missing_channel_power.zip", "ek80.zip",
        "ek80_bb_complex_multiplex.zip", "ek80_bb_with_calibration.zip",
        "ek80_duplicate_ping_times.zip", "ek80_ext.zip", "ek80_invalid_env_datagrams.zip",
        "ek80_missing_sound_velocity_profile.zip", "ek80_new.zip", "ek80_sequence.zip",
        "es60.zip", "es70.zip", "es80.zip", "legacy_datatree.zip",
    ]

    # v0.11.1a1 checksums (GitHub release assets)
    registry = {
        "ad2cp.zip": "sha256:78c634c7345991177b267c4cbb31f391990d2629b7f4a546da20d5126978b98a",
        "azfp.zip": "sha256:43976d3e6763c19278790bf39923b6827143ea3024c5ba5f46126addecf235de",
        "azfp6.zip": "sha256:98228329333064fb4b44d3044296c79d58ac22f6d81f7f22cf770bacf0e882fd",
        "ea640.zip": "sha256:49f70bd6f2355cb3c4c7a5b31fc00f7ae8c8a9ae888f0df1efe759032f9580df",
        "ecs.zip": "sha256:dcc312baa1e9da4488f33bef625b1f86c8a92e3262e34fc90ccd0a4f90d1e313",
        "ek60.zip": "sha256:66735de0ac584ec8a150b54b1a54024a92195f64036134ffdc9d472d7e155bb2",
        "ek60_calibrate_chunks.zip": "sha256:bf435b1f7fc055f51afd55c4548713ba8e1eb0e919a0d74f4b9dd5f60b7fe327",
        "ek60_missing_channel_power.zip": "sha256:f3851534cdc6ad3ae1d7c52a11cb279305d316d0086017a305be997d4011e20e",
        "ek80.zip": "sha256:a114a8272e4c0e08c45c75241c50e3fd9e954f85791bb5eda25be98f6f782397",
        "ek80_bb_complex_multiplex.zip": "sha256:9feaa90ce60036db8110ff14e732910786c6deff96836f2d6504ec1e9de57533",
        "ek80_bb_with_calibration.zip": "sha256:53f018b6dae051cc86180e13cb3f28848750014dfcf84d97cf2191be2b164ccb",
        "ek80_duplicate_ping_times.zip": "sha256:11a2dcb5cf113fa1bb03a6724524ac17bdb0db66cb018b0a3ca7cad87067f4bb",
        "ek80_ext.zip": "sha256:79dd12b2d9e0399f88c98ab53490f5d0a8d8aed745650208000fcd947dbdd0de",
        "ek80_invalid_env_datagrams.zip": "sha256:dece27d90f30d1a13b56d99350c3254e81622af3199fda0112d3b9e1d7db270c",
        "ek80_missing_sound_velocity_profile.zip": "sha256:1635585026ae5c4ffdff09ca4d63aeff0b33471c5ee0e1b8a520f87469535852",
        "ek80_new.zip": "sha256:f799cde453762c46ad03fee178c76cd9fbb00eec92a5d1038c32f6a9479b2e57",
        "ek80_sequence.zip": "sha256:9d8fac39dd31f587d55b9978ba4d2b52bbc85daa85d320ef2ac34b3ae947bb1f",
        "es60.zip": "sha256:a6c2a15c664ef8b6ac17cb36a28162c271fca361509cf43313038f1bdc9b6c7c",
        "es70.zip": "sha256:a6b4f27f33f09bace26264de6984fdb4111a3a0337bc350c3c1d25c8b3effc7c",
        "es80.zip": "sha256:b37ee01462f46efe055702c20be67d2b8c6b786844b183b16ffc249c7c5ec704",
        "legacy_datatree.zip": "sha256:820cd252047dbf35fa5fb04a9aafee7f7659e0fe4f7d421d69901c57deb6c9d5",
    }

    EP = pooch.create(
        path=cache_dir,
        base_url=base,
        version=ver,
        registry=registry,
        retry_if_failed=1,
    )

    def _unpack(fname, action, pooch_instance):
        z = Path(fname)
        out = z.parent / z.stem

        print(
            "\n[echopype-ci] UNPACK\n"
            f"  zip file   = {z}\n"
            f"  action     = {action}\n"
            f"  extract_to = {out}\n",
            flush=True,
        )

        if action in ("update", "download") or not out.exists():
            from zipfile import ZipFile

            with ZipFile(z, "r") as f:
                f.extractall(out)

            # flatten single nested dir if needed
            try:
                entries = list(out.iterdir())
                if len(entries) == 1 and entries[0].is_dir():
                    inner = entries[0]
                    for child in inner.iterdir():
                        target = out / child.name
                        if not target.exists():
                            child.rename(target)
                    try:
                        inner.rmdir()
                    except Exception:
                        pass
            except Exception:
                pass

        # Print tree after extraction/flatten
        try:
            if out.exists():
                print("[echopype-ci] extracted tree (depth ≤ 2):")
                for p in sorted(out.glob("*")):
                    print(f"   - {p.name}")
                    if p.is_dir():
                        for q in sorted(p.glob("*")):
                            print(f"       • {q.name}")
        except Exception:
            pass

        return str(out)


    for b in bundles:
        url = base.format(version=ver) + b
        print(f"[echopype-ci] fetching bundle: {b}")
        print(f"[echopype-ci]   → URL: {url}")
        EP.fetch(b, processor=_unpack, progressbar=False)

    TEST_DATA_FOLDER = Path(cache_dir) / ver
    
    print(
        "\n[echopype-ci] TEST_DATA_FOLDER\n"
        f"  path = {TEST_DATA_FOLDER}\n"
        f"  exists = {TEST_DATA_FOLDER.exists()}\n",
        flush=True,
    )

    if TEST_DATA_FOLDER.exists():
        print("[echopype-ci] top-level contents:")
        for p in sorted(TEST_DATA_FOLDER.iterdir()):
            print(f"   - {p.name}")


@pytest.fixture(scope="session")
def dump_output_dir():
    return TEST_DATA_FOLDER / "dump"


@pytest.fixture(scope="session")
def test_path():
    return {
        "ROOT": TEST_DATA_FOLDER,
        "EA640": TEST_DATA_FOLDER / "ea640",
        "EK60": TEST_DATA_FOLDER / "ek60",
        "EK60_CAL_CHUNKS": TEST_DATA_FOLDER / "ek60_calibrate_chunks",
        "EK60_MISSING_CHANNEL_POWER": TEST_DATA_FOLDER / "ek60_missing_channel_power",
        "EK80": TEST_DATA_FOLDER / "ek80",
        "EK80_NEW": TEST_DATA_FOLDER / "ek80_new",
        "ES60": TEST_DATA_FOLDER / "es60",
        "ES70": TEST_DATA_FOLDER / "es70",
        "ES80": TEST_DATA_FOLDER / "es80",
        "AZFP": TEST_DATA_FOLDER / "azfp",
        "AZFP6": TEST_DATA_FOLDER / "azfp6",
        "AD2CP": TEST_DATA_FOLDER / "ad2cp",
        "EK80_MULTIPLEX": TEST_DATA_FOLDER / "ek80_bb_complex_multiplex",
        "EK80_DUPE_PING": TEST_DATA_FOLDER / "ek80_duplicate_ping_times",
        "EK80_MISSING_SOUND": TEST_DATA_FOLDER / "ek80_missing_sound_velocity_profile",
        "EK80_INVALID_ENV": TEST_DATA_FOLDER / "ek80_invalid_env_datagrams",
        "EK80_SEQUENCE": TEST_DATA_FOLDER / "ek80_sequence",
        "EK80_CAL": TEST_DATA_FOLDER / "ek80_bb_with_calibration",
        "EK80_EXT": TEST_DATA_FOLDER / "ek80_ext",
        "ECS": TEST_DATA_FOLDER / "ecs",
        "LEGACY_DATATREE": TEST_DATA_FOLDER / "legacy_datatree",
    }


@pytest.fixture(scope="session")
def minio_bucket():
    return dict(
        client_kwargs=dict(endpoint_url="http://localhost:9000/"),
        key="minioadmin",
        secret="minioadmin",
    )

# recap of the errors, and failures
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    tr = terminalreporter
    # Counts
    sections = [
        ("passed",   "green"),
        ("xfailed",  "yellow"),
        ("xpassed",  "yellow"),
        ("skipped",  "yellow"),
        ("failed",   "red"),
        ("error",    "red"),
    ]

    tr.write_line("\n─ Test Outcome Summary ─", bold=True)
    for key, color in sections:
        reports = tr.getreports(key) or []
        if reports:
            tr.write_line(f"{key:>8}: {len(reports)}",
                          **{color: True})

    # Warnings count (pytest already prints a warnings summary block)
    w_reports = tr.stats.get("warnings", [])
    if w_reports:
        tr.write_line(f"{'warnings':>8}: {len(w_reports)}", yellow=True)

    # List skipped with reasons (short)
    skipped = tr.stats.get("skipped", [])
    if skipped:
        tr.write_line("\nSkipped tests:", yellow=True, bold=True)
        for rep in skipped[:20]:
            reason = ""
            try:
                # Best effort to extract reason text
                reason = getattr(rep, "longrepr", "") or ""
                reason = getattr(reason, "reprcrash", None) and reason.reprcrash.message or str(reason)
            except Exception:
                pass
            tr.write_line(f"  • {rep.nodeid}" + (f"  — {reason}" if reason else ""))

        if len(skipped) > 20:
            tr.write_line(f"  … and {len(skipped) - 20} more", yellow=True)

    # List xfailed (expected fails)
    xfailed = tr.stats.get("xfailed", [])
    if xfailed:
        tr.write_line("\nExpected failures (xfail):", yellow=True, bold=True)
        for rep in xfailed[:20]:
            tr.write_line(f"  • {rep.nodeid}")
        if len(xfailed) > 20:
            tr.write_line(f"  … and {len(xfailed) - 20} more", yellow=True)

    tr.write_line("")  # trailing newline
