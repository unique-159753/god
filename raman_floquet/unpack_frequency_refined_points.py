"""Validate and restore the three lossless Git LFS raw-point ZIP shards."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile
import zlib

BASE = Path(__file__).resolve().parent
DEFAULT = BASE / "results" / "frequency_phase_100x100_phi_0_pi"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def file_crc(path):
    crc = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            crc = zlib.crc32(block, crc)
    return crc & 0xffffffff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT)
    parser.add_argument("--if-needed", action="store_true",
                        help="Do nothing for a complete local points tree or a fresh scan without archives")
    args = parser.parse_args()
    out = args.results.resolve()
    points = out / "points"
    if args.if_needed:
        finalized = [p for p in points.glob("*.npz") if not p.name.endswith(".tmp.npz")]
        if len(finalized) == 30300:
            print("All30300 local raw-point files are present; no archive extraction needed.", flush=True)
            return
        if not (out / "raw_point_archives.json").exists():
            print("No completed raw-point archives found; start or resume the exact scan normally.", flush=True)
            return
    metadata = json.loads((out / "raw_point_archives.json").read_text())
    if metadata["total_points"] != 30300 or len(metadata["archives"]) != 3:
        raise RuntimeError("Unexpected raw archive contract")
    restored = reused = 0
    # Check every complete archive before writing any extracted data.
    for iv in range(3):
        name = f"raw_points_V{iv}.zip"
        expected = metadata["archives"][name]
        path = out / name
        if not path.is_file():
            raise RuntimeError(f"Missing {name}; fetch the Git LFS files with git lfs pull")
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise RuntimeError(f"SHA256/size mismatch for {name}; verify git lfs pull completed")
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) != 10100 or len({item.filename for item in members}) != 10100:
                raise RuntimeError(f"Incorrect or duplicate member count: {name}")
            for item in members:
                match = re.fullmatch(rf"points/V{iv}_p(\d{{2}}|100)_w(\d{{2}})\.npz", item.filename)
                if not match or int(match[1]) > 100 or int(match[2]) > 99 or item.is_dir():
                    raise RuntimeError(f"Unexpected archive member path: {item.filename}")
                destination = out / item.filename
                if not destination.resolve().is_relative_to(out):
                    raise RuntimeError(f"Extraction path escapes results directory: {item.filename}")
                unix_mode = item.external_attr >> 16
                if (unix_mode & 0o170000) == 0o120000:
                    raise RuntimeError(f"Archive symlink is not permitted: {item.filename}")
        print(f"SHA256 and member checks passed: {name}", flush=True)
    points.mkdir(parents=True, exist_ok=True)
    for iv in range(3):
        name = f"raw_points_V{iv}.zip"
        with zipfile.ZipFile(out / name) as archive:
            for item in archive.infolist():
                destination = out / item.filename
                if (destination.is_file() and destination.stat().st_size == item.file_size
                        and file_crc(destination) == item.CRC):
                    reused += 1
                    continue
                temp = destination.with_suffix(".unpack.tmp")
                if not temp.resolve().is_relative_to(out):
                    raise RuntimeError(f"Temporary extraction path escapes results directory: {item.filename}")
                with archive.open(item) as source, temp.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1 << 20)
                if temp.stat().st_size != item.file_size or file_crc(temp) != item.CRC:
                    raise RuntimeError(f"Extracted CRC/size mismatch: {item.filename}")
                temp.replace(destination)
                restored += 1
    print(f"Raw checkpoint restoration complete: {restored} extracted, {reused} existing files verified.", flush=True)


if __name__ == "__main__":
    main()
