#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/raman_matplotlib}"
RAMAN_PYTHON="${RAMAN_PYTHON:-python3}"
"$RAMAN_PYTHON" scan_frequency_phases.py --workers 4
"$RAMAN_PYTHON" validate_frequency_scan.py
"$RAMAN_PYTHON" analyze_frequency_scan.py
"$RAMAN_PYTHON" plot_frequency_phases.py --verify-modal-statistics
"$RAMAN_PYTHON" write_frequency_report.py
"$RAMAN_PYTHON" package_frequency_results.py
