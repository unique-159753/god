"""Package only the completed, validated 100x100 full-spectrum IPR/Dn study."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import zipfile

from PIL import Image

BASE = Path(__file__).resolve().parent
DEFAULT = BASE / "results" / "frequency_phase_100x100_phi_0_pi"
ARCHIVE_RESULTS = Path("results/frequency_phase_100x100_phi_0_pi")
SOURCES = (
    "model.py", "frequency_solver.py", "frequency_fast_native.py", "frequency_baselines.py",
    "scan_frequency_phase_refined.py", "validate_frequency_phase_refined.py",
    "plot_frequency_phase_refined.py", "write_frequency_refined_report.py",
    "package_frequency_refined.py", "read_frequency_point.py", "run_frequency_refined.sh",
    "unpack_frequency_refined_points.py", "requirements.txt",
)
FIGURES = (
    "three_V_refined_IPR_Dn_summary", "three_V_refined_mean_IPR", "three_V_refined_mean_Dn",
    "three_V_refined_frequency_cuts",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def matches_raw_archive(path, selected, point_hashes):
    """Reuse a preuploaded shard only after comparing every source SHA256."""
    if not path.is_file():
        return False
    expected_names = [str(Path("points") / p.name) for p in selected]
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != expected_names:
                return False
            for source, member in zip(selected, archive.infolist()):
                if member.compress_type != zipfile.ZIP_STORED or member.file_size != source.stat().st_size:
                    return False
                archive_hash = hashlib.sha256()
                with archive.open(member) as stream:
                    for block in iter(lambda: stream.read(1 << 20), b""):
                        archive_hash.update(block)
                actual = digest(source)
                if archive_hash.hexdigest() != actual:
                    return False
                point_hashes[source] = actual
    except (OSError, zipfile.BadZipFile):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.results
    completion = json.loads((out / "scan_completion.json").read_text())
    if (completion["grid_points"] != 30000 or completion["additional_pi_over_2_points"] != 300
            or completion["nonconverged"]):
        raise RuntimeError("Refuse to package incomplete or unconverged results")
    validation = json.loads((out / "frequency_refined_validation.json").read_text())
    if (not validation["data_integrity_passed"] or not validation["numerical_validation_passed"]
            or validation["failures"]):
        raise RuntimeError("Refuse to package results without a passed independent validation")
    for name in ("research_report_zh.md", "README.md", "data_dictionary.md", "analysis_summary.json",
                 "frequency_phase_scan.npz", "frequency_phase_scan.csv", "frequency_phase_slices.npz",
                 "frequency_extra_pi_over_2.csv", "eigenstate_observables.csv",
                 "frequency_refined_validation_modes.npz", "frequency_phase_figures_summary.json",
                 "frequency_refined_convergence_diagnostics.csv", "frequency_refined_method_crosscheck.csv",
                 "frequency_refined_phase_symmetry.csv"):
        if not (out / name).is_file():
            raise RuntimeError(f"Missing required completed output: {name}")
    point_files = sorted((out / "points").glob("*.npz"))
    if len(point_files) != 30300 or any(p.name.endswith(".tmp.npz") for p in point_files):
        raise RuntimeError("Expected exactly 30300 finalized raw-state checkpoints")
    metadata = json.loads((out / "scan_parameters.json").read_text())
    for name, expected in metadata["code_sha256"].items():
        if digest(BASE / name) != expected:
            raise RuntimeError(f"Production source changed after the scan: {name}")
    raw_archives, point_hashes = {}, {}
    for iv in range(3):
        selected = [p for p in point_files if p.name.startswith(f"V{iv}_")]
        if len(selected) != 10100:
            raise RuntimeError(f"Expected 10100 raw checkpoints for V index {iv}")
        name = f"raw_points_V{iv}.zip"
        dest = out / name
        reused = matches_raw_archive(dest, selected, point_hashes)
        if dest.exists() and not reused:
            raise RuntimeError(f"Existing {name} does not match finalized point data; do not overwrite an incomplete or uploaded shard")
        if not reused:
            with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for p in selected:
                    archive.write(p, str(Path("points") / p.name))
        with zipfile.ZipFile(dest) as archive:
            bad = None if reused else archive.testzip()
            if bad:
                raise RuntimeError(f"Raw point archive CRC failure: {bad}")
            raw_archives[name] = {"V_index": iv, "V": [.5, 2., 4.][iv], "bytes": dest.stat().st_size,
                                  "members": len(archive.namelist()), "sha256": digest(dest),
                                  "CRC_passed": True, "compression": "ZIP_STORED",
                                  "member_prefix": "points/"}
        print(json.dumps({name: raw_archives[name]}), flush=True)
    raw_metadata = {"total_points": 30300, "archives": raw_archives,
                    "restore": "python3 unpack_frequency_refined_points.py",
                    "purpose": "lossless Git LFS storage of all raw-state checkpoints; full delivery ZIP contains actual points instead"}
    (out / "raw_point_archives.json").write_text(json.dumps(raw_metadata, indent=2) + "\n")
    code = out / "code"
    code.mkdir(exist_ok=True)
    for name in SOURCES:
        shutil.copy2(BASE / name, code / name)
    figure_meta = []
    for name in FIGURES:
        png, pdf = out / (name + ".png"), out / (name + ".pdf")
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (0., 0.))
            if min(dpi) < 349.9:
                raise RuntimeError(f"Expected 350 DPI PNG: {name}")
            dimensions = list(image.size)
        if not pdf.exists() or re.search(rb"/Subtype\s*/Image\b", pdf.read_bytes()):
            raise RuntimeError(f"Missing fully vector PDF: {name}")
        figure_meta.append({"png": png.name, "pdf": pdf.name, "pixels": dimensions,
                            "dpi": list(dpi), "vector_pdf_image_objects": 0})
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "libraries": {name: importlib.metadata.version(name) for name in
                      ("numpy", "scipy", "matplotlib", "threadpoolctl", "Pillow")},
        "sources": [{"path": name, "sha256": digest(code / name)} for name in SOURCES],
        "figures": figure_meta,
        "grid_points": 30000, "extra_phi_pi_over_2_points": 300, "eigenstate_rows": 5393400,
        "frequency_units": "dimensionless omega/t0, range 0.2..10 inclusive, 100 points",
        "phase_grid": "0..pi inclusive, 100 points; pi/2 separately calculated",
        "observables": "all-spectrum spin-resolved IPR; Dn=-ln(IPR)/ln89",
        "wavefunctions": "lossless Q-sector coefficients reconstruct all178 complete eigenstates",
        "git_raw_data": "three Git LFS ZIP_STORED per-V archives,10100 raw point files each; SHA256 in raw_point_archives.json",
        "removed_observables": "No undriven-baseline IPR/Dn differences; no outer20% selection; no effective controls",
        "finite_size_limit": "L=89 only; no thermodynamic localization transition claim",
        "validation_file": "frequency_refined_validation.json",
    }
    (out / "delivery_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    sections = "\n".join(
        '<section><p><a href="' + item["png"] + '">PNG</a> · <a href="' + item["pdf"]
        + '">矢量 PDF</a></p><img src="' + item["png"] + '" alt="' + item["png"] + '"></section>'
        for item in figure_meta)
    index = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1">'
             '<title>100×100 Raman Floquet IPR / Dn</title><style>'
             'body{max-width:1500px;margin:30px auto;padding:0 24px;font:17px/1.7 sans-serif}'
             'img{width:100%;height:auto}section{margin:35px 0}a{color:#145a8a}</style>'
             '<h1>双驱动 Raman 晶格：100×100 全谱 IPR / Dn</h1>'
             '<p>ω/t₀=0.2…10；φ=0…π；V=0.5、2、4。共30000个真实主网格点，'
             '另有300个真实π/2切线点。L=89，完整178态。</p>'
             '<p>spin-resolved IPR；Dn=−lnIPR/ln89。每种指标跨三V共享色标；没有参数插值。'
             '有限尺寸参与指标不能单独确立相变。</p>'
             '<p><a href="research_report_zh.md">中文报告</a> · <a href="README.md">复现说明</a> · '
             '<a href="data_dictionary.md">字段定义</a> · '
             '<a href="frequency_refined_validation.json">独立数值验证</a></p>' + sections + '</html>')
    (out / "index.html").write_text(index, encoding="utf-8")
    excluded = {"file_manifest.json", "SHA256SUMS", "archive_checksums.json"}
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in excluded
                   and p.suffix not in {".zip", ".log"} and not p.name.endswith(".tmp.npz")
                   and "__pycache__" not in p.parts)
    manifest = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size,
                 "sha256": point_hashes[p] if p in point_hashes else digest(p)} for p in files]
    (out / "file_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "SHA256SUMS").write_text("".join(row["sha256"] + "  " + row["path"] + "\n" for row in manifest))
    files += [out / "file_manifest.json", out / "SHA256SUMS"]
    checks = {}
    for light in (True, False):
        name = "frequency_refined_figures_report_code.zip" if light else "frequency_refined_complete_data.zip"
        dest = out / name
        if light:
            chosen = [p for p in files if p.name not in {"file_manifest.json", "SHA256SUMS"}
                      and (p.suffix in {".png", ".pdf", ".md", ".py", ".txt", ".html", ".json", ".sh"}
                           or (p.suffix == ".csv" and p.stat().st_size < 50_000_000)
                           or p.name in {"frequency_phase_plot_data.npz", "frequency_phase_slices.npz"})]
        else:
            chosen = files
        with zipfile.ZipFile(dest, "w", allowZip64=True) as archive:
            for p in chosen:
                compression = zipfile.ZIP_STORED if p.suffix in {".npz", ".png", ".pdf"} else zipfile.ZIP_DEFLATED
                archive.write(p, str(ARCHIVE_RESULTS / p.relative_to(out)), compress_type=compression,
                              compresslevel=1 if compression == zipfile.ZIP_DEFLATED else None)
            for source in sorted(code.iterdir()):
                if source.is_file():
                    archive.write(source, source.name, compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
        with zipfile.ZipFile(dest) as archive:
            bad = archive.testzip()
            if bad:
                raise RuntimeError(f"Archive CRC failure: {bad}")
            checks[name] = {"bytes": dest.stat().st_size, "sha256": digest(dest),
                            "members": len(archive.namelist()), "CRC_passed": True}
        print(json.dumps({name: checks[name]}), flush=True)
    (out / "archive_checksums.json").write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    main()
