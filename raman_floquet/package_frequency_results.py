"""Archive completed real scan outputs and all required reproducible sources."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import zipfile

from PIL import Image

BASE = Path(__file__).resolve().parent
DEFAULT = BASE / "results" / "frequency_phase_three_V"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.results
    completion = json.loads((out/"scan_completion.json").read_text())
    if completion["nonconverged"] or completion["grid_points"] != 7500:
        raise RuntimeError("Do not package an incomplete or unconverged scan as final")
    required = ["research_report_zh.md", "analysis_summary.json", "frequency_scan_validation.json",
                "three_V_frequency_phase_summary.png", "frequency_phase_scan.npz",
                "frequency_phase_slices.npz", "eigenstate_observables.csv"]
    for name in required:
        if not (out/name).is_file():
            raise RuntimeError(f"Missing required final result: {name}")
    validation = json.loads((out/"frequency_scan_validation.json").read_text())
    if not validation["data_integrity_passed"] or validation["failures"]:
        raise RuntimeError("Do not package results that failed the independent data audit")
    code = out/"code"
    code.mkdir(exist_ok=True)
    names = {"model.py", "requirements.txt", "method_benchmark.py"}
    for pattern in ["frequency_*.py", "scan_frequency*.py", "plot_frequency*.py",
                    "validate_frequency*.py", "analyze_frequency*.py", "read_frequency*.py",
                    "package_frequency*.py", "write_frequency*.py", "run_frequency*.sh"]:
        names.update(p.name for p in BASE.glob(pattern))
    for name in sorted(names):
        if (BASE/name).is_file():
            shutil.copy2(BASE/name, code/name)
    figmeta = []
    for png in sorted(out.glob("*.png")):
        with Image.open(png) as image:
            dpi = image.info.get("dpi", [0., 0.])
            if min(dpi) < 299.9 or not png.with_suffix(".pdf").exists():
                raise RuntimeError(f"Missing 300DPI/vector counterpart for {png}")
            figmeta.append({"png": png.name, "pdf": png.with_suffix(".pdf").name,
                            "pixels": list(image.size), "dpi": list(dpi)})
    provenance = {"python": platform.python_version(),
                  "libraries": {name: importlib.metadata.version(name) for name in
                                ["numpy", "scipy", "matplotlib", "threadpoolctl", "Pillow"]},
                  "source_files": sorted(names), "figures": figmeta,
                  "data_definition": "spin-resolved IPR, Dn=-ln(IPR)/ln89; omega dimensionless",
                  "zero_frequency": "genuine frozen static states, outer Floquet metrics NaN",
                  "wavefunction_storage": "lossless sector coefficients; read_frequency_point reconstructs all178 modes",
                  "finite_size_limit": "L=89 only; no claim of a thermodynamic phase transition"}
    (out/"delivery_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    sections = "\n".join(
        '<section><p><a href="'+item["png"]+'">PNG</a> · <a href="'+item["pdf"]+'">PDF</a></p>'
        '<img src="'+item["png"]+'" alt="'+item["png"]+'"></section>' for item in figmeta)
    index = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1">'
             '<title>Three-reference Raman Floquet scan</title><style>'
             'body{max-width:1550px;margin:30px auto;padding:0 24px;font:17px/1.7 sans-serif}'
             'img{width:100%;height:auto}section{margin:40px 0}a{color:#145a8a}</style>'
             '<h1>双驱动Raman晶格：三静态基准相的真实频率—相位扫描</h1>'
             '<p>ω/t₀=0…10；3×50×50真实网格及额外150个π/2切线点；自旋分辨IPR，Dn=−lnIPR/ln89。</p>'
             '<p>ω=0为冻结静态态，无Floquet外侧准能量指标。不同V的同一指标共享色标。'
             '本结果比较有限尺寸参与指标，不能单独确立热力学相变。</p>'
             '<p><a href="research_report_zh.md">完整中文科研报告</a> · '
             '<a href="README.md">复现说明</a> · <a href="data_dictionary.md">数据字典</a> · '
             '<a href="frequency_scan_validation.json">独立验证</a></p>'+sections+'</html>')
    (out/"index.html").write_text(index, encoding="utf-8")
    exclude = {"file_manifest.json", "SHA256SUMS", "archive_checksums.json"}
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in exclude
                   and p.suffix != ".zip" and p.name != "scan.log")
    manifest = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]
    (out/"file_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (out/"SHA256SUMS").write_text("".join(x["sha256"]+"  "+x["path"]+"\n" for x in manifest))
    files += [out/"file_manifest.json", out/"SHA256SUMS"]
    checks = {}
    for light in [False, True]:
        name = "frequency_three_V_figures_report_code.zip" if light else "frequency_three_V_complete_data.zip"
        dest = out/name
        chosen = [p for p in files if not light or p.suffix in
                  {".png", ".pdf", ".md", ".py", ".txt", ".html", ".json"}
                  or (p.suffix == ".csv" and p.stat().st_size < 500_000)]
        # A full manifest describes the full archive, not a lightweight subset.
        if light:
            chosen = [p for p in chosen if p.name not in {"file_manifest.json", "SHA256SUMS"}]
        with zipfile.ZipFile(dest, "w", allowZip64=True) as archive:
            for p in chosen:
                compression = zipfile.ZIP_STORED if p.suffix in {".npz", ".png", ".pdf"} else zipfile.ZIP_DEFLATED
                archive.write(p, str(Path("results/frequency_phase_three_V")/p.relative_to(out)),
                              compress_type=compression, compresslevel=1 if compression == zipfile.ZIP_DEFLATED else None)
            # Scripts beside results preserve their actual relative output paths.
            for source in sorted(code.iterdir()):
                if source.is_file():
                    archive.write(source, source.name, compress_type=zipfile.ZIP_DEFLATED)
        with zipfile.ZipFile(dest) as archive:
            bad = archive.testzip()
            if bad:
                raise RuntimeError(f"Archive checksum failure: {bad}")
            checks[name] = {"bytes": dest.stat().st_size, "sha256": sha256(dest),
                            "members": len(archive.namelist()), "CRC_passed": True}
    (out/"archive_checksums.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
