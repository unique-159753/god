"""Package already-computed results and verify both downloadable ZIP files."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent
EXCLUDED = {
    'TASK_SPECIFICATION.txt', 'SHA256SUMS.txt', 'FIGURE_SHA256SUMS.txt',
    'DOWNLOAD_SHA256SUMS.txt', 'package_summary.json',
}
COMPACT_NAMES = {
    'README.md', 'REPORT.md', 'METHODS.md', 'key_frequency_comparison.csv',
    'quantitative_summary.json', 'validation_summary.json',
    'refinement_summary.json', 'provenance.json', 'artifact_audit.json',
    'run_summary.json', 'requirements.txt',
}


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(name, files):
    (ROOT / name).write_text(''.join(f'{checksum(p)}  {p.name}\n' for p in files))


def archive(name, files):
    target = ROOT / name
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as out:
        for path in files:
            out.write(path, arcname='folding_study/' + path.name)
    with zipfile.ZipFile(target) as saved:
        assert saved.testzip() is None
        assert len(saved.namelist()) == len(files)
    return dict(file=name, bytes=target.stat().st_size,
                sha256=checksum(target), files=len(files))


def main():
    payload = sorted(p for p in ROOT.iterdir()
                     if p.is_file() and p.suffix != '.zip' and p.name not in EXCLUDED)
    compact = [p for p in payload if p.name in COMPACT_NAMES
               or (p.name.startswith('fig') and p.suffix in {'.png', '.pdf'})]
    manifest('SHA256SUMS.txt', payload)
    manifest('FIGURE_SHA256SUMS.txt', compact)
    compact.append(ROOT / 'FIGURE_SHA256SUMS.txt')
    payload.extend([ROOT / 'SHA256SUMS.txt', ROOT / 'FIGURE_SHA256SUMS.txt'])
    result = [archive('figures_and_report.zip', compact),
              archive('full_study.zip', payload)]
    manifest('DOWNLOAD_SHA256SUMS.txt', [ROOT / r['file'] for r in result])
    (ROOT / 'package_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
