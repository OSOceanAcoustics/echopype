from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

cache_root = Path(r"C:\Users\lloyd\AppData\Local\echopype\echopype\Cache\v0.11.1a2")
out_root = Path(r"C:\Users\lloyd\AppData\Local\Temp\echopype_asset_check\rebuilt_py")

out_root.mkdir(parents=True, exist_ok=True)


def build_zip(src_dir: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file():
                # IMPORTANT: force POSIX paths inside the zip
                arcname = p.relative_to(src_dir.parent).as_posix()
                zf.write(p, arcname)

    print(f"Built: {zip_path}")


build_zip(cache_root / "ek80", out_root / "ek80.zip")
build_zip(cache_root / "ek80_ext", out_root / "ek80_ext.zip")

print("\nDone rebuilding zips with POSIX paths.")