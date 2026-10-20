"""Package computed results, source code and a local figure index; no new physics."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import zipfile

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
SOURCE_TASK = BASE / "original_task_zh.txt"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    code = OUT / "code"
    code.mkdir(exist_ok=True)
    sources = sorted(p for p in BASE.iterdir()
                     if p.is_file() and p.suffix in {".py", ".sh", ".md", ".txt"})
    for source in sources:
        shutil.copy2(source, code / source.name)
    if SOURCE_TASK.exists():
        original = SOURCE_TASK.read_text(encoding="utf-8")
        accepted = ("\n\n## 本轮已由用户确认的观测量规则\n\n"
                    "站点ρj=|ψj↑|²+|ψj↓|²、IPRsite=Σρj²、Dsite=−lnIPRsite/ln89；"
                    "准能折叠至[−ω/2,ω/2)后选|ε|最大的36态。另保留178分量IPR/ln178、"
                    "静态外侧投影及重叠匹配对照。用户答复：采用上述明确规则，保留两套对照。\n")
        (OUT / "task_specification_zh.md").write_text(original + accepted, encoding="utf-8")

    inventory = []
    for path in sorted(OUT.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                array = data[key]
                inventory.append({"file": path.name, "key": key,
                                  "shape": json.dumps(array.shape),
                                  "dtype": str(array.dtype)})
    with (OUT / "npz_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "key", "shape", "dtype"])
        writer.writeheader()
        writer.writerows(inventory)

    figures = []
    for path in sorted(OUT.glob("*.png")):
        with Image.open(path) as im:
            dpi = im.info.get("dpi", [None, None])
            if min(dpi) < 299.9:
                raise RuntimeError(f"PNG resolution below 300 DPI: {path}")
            if not path.with_suffix(".pdf").exists():
                raise RuntimeError(f"Missing vector PDF: {path}")
            figures.append({"png": path.name, "pdf": path.with_suffix(".pdf").name,
                            "pixels": list(im.size), "dpi": list(dpi)})
    validation = json.loads((OUT / "grid_validation.json").read_text())
    solver = json.loads((OUT / "solver_validation.json").read_text())
    if not validation["data_integrity_passed"] or validation["failures"] or not solver["passed"]:
        raise RuntimeError("Underlying scientific validation did not pass")
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "libraries": {p: importlib.metadata.version(p)
                      for p in ["numpy", "scipy", "matplotlib", "threadpoolctl", "Pillow"]},
        "grid_points": 8991, "additional_pi_over_3_points": 111,
        "main_time_steps": 64, "representative_time_steps": 128,
        "validation": "whole-grid integrity plus selected convergence; see numerical warnings",
        "figures": figures,
        "archive_scope": "All scientific output including every point checkpoint, source code, reports and diagnostics; venv/temp binaries excluded",
        "uniform_accuracy_claim": False,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    main_figures = [
        ("fig1_resonance_vs_localization", "图1：一光子共振与局域化差值热图"),
        ("fig2_resonance_correlation", "图2：相位切线和定量相关性"),
        ("fig3_resonant_state_pairs", "图3：真实共振态对、空间密度和投影"),
        ("fig4_NNN_vs_resonance", "图4：静态、有效、exact及补偿对照"),
        ("fig5_sambe_mixing", "图5：Sambe截断及多态混合"),
    ]
    sections = "\n".join(
        f'<section><h2>{html.escape(caption)}</h2><a href="{name}.png">PNG</a> · '
        f'<a href="{name}.pdf">矢量PDF</a><img src="{name}.png" alt="{html.escape(caption)}"></section>'
        for name, caption in main_figures)
    index = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1">'
             '<title>Raman Floquet 真实数值结果</title><style>'
             'body{max-width:1500px;margin:30px auto;padding:0 22px;font:17px/1.7 sans-serif;color:#17212b}'
             'img{display:block;max-width:100%;height:auto;margin:12px 0}a{color:#145a8a}section{margin:45px 0}'
             '</style><h1>双驱动准周期 Raman 晶格</h1>'
             '<p>L=89，111×81参数网格及额外111个π/3切线点。以下图片均由真实数据绘制，提供≥300DPI PNG及矢量PDF。</p>'
             '<p>存在一光子混合通道，但裸响应与空间局域化差值的峰值和相位规律不同；抵消首阶NNN项后低频增强仍保留。'
             '当前数据不证明新相变。</p>'
             '<p><a href="research_report_zh.md">完整中文科研报告（含6问回答及误差）</a> · '
             '<a href="data_dictionary.md">数据字典</a> · <a href="code/README.md">复现说明</a> · '
             '<a href="grid_validation.json">全网格验证</a> · <a href="npz_inventory.csv">NPZ字段表</a></p>'
             '<p>主网格M64；最敏感的抽样点误差约4.77×10⁻⁴，追加M512后降为1.10×10⁻⁷。'
             '近简并基底敏感性及抽样限制见报告。幺正残差不等于IPR误差上界。</p>'
             + sections + '</html>')
    (OUT / "index.html").write_text(index, encoding="utf-8")

    excluded = {"SHA256SUMS", "file_manifest.json", "archive_checksums.json"}
    files = sorted(p for p in OUT.rglob("*")
                   if p.is_file() and p.suffix != ".zip" and p.name not in excluded)
    manifest = [{"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size,
                 "sha256": sha256(p)} for p in files]
    (OUT / "file_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "SHA256SUMS").write_text("".join(
        f'{row["sha256"]}  {row["path"]}\n' for row in manifest))

    # Put scripts beside results in the full archive, preserving their default paths.
    root_files = [(source, Path("raman_floquet") / source.name) for source in sources]
    result_files = [(p, Path("raman_floquet/results") / p.relative_to(OUT)) for p in files]
    result_files.extend((OUT / n, Path("raman_floquet/results") / n)
                        for n in ["file_manifest.json", "SHA256SUMS"])
    full_path = OUT / "raman_floquet_full_results.zip"
    with zipfile.ZipFile(full_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for source, target in root_files + result_files:
            z.write(source, str(target))
    light_path = OUT / "raman_floquet_figures_report_code.zip"
    light_files = [(p, target) for p, target in result_files
                   if (p.suffix in {".png", ".pdf", ".md", ".html", ".py", ".sh", ".txt", ".json"}
                       or (p.suffix == ".csv" and p.stat().st_size < 500_000))
                   and "representative_cache" not in str(p) and "localization_points" not in str(p)
                   and p.name not in {"SHA256SUMS", "file_manifest.json"}]
    with zipfile.ZipFile(light_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for source, target in root_files + light_files:
            z.write(source, str(target))
    checks = {}
    for archive in [full_path, light_path]:
        with zipfile.ZipFile(archive) as z:
            bad = z.testzip()
            if bad is not None:
                raise RuntimeError(f"CRC failure in {archive}: {bad}")
            checks[archive.name] = {"bytes": archive.stat().st_size,
                                    "sha256": sha256(archive),
                                    "members": len(z.namelist()), "CRC_passed": True}
    (OUT / "archive_checksums.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"figures": len(figures), "npz_fields": len(inventory),
                      "manifest_files": len(manifest), "archives": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
