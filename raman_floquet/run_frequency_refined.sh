#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/raman_matplotlib}"
RAMAN_PYTHON="${RAMAN_PYTHON:-python3}"
"$RAMAN_PYTHON" unpack_frequency_refined_points.py --if-needed
"$RAMAN_PYTHON" scan_frequency_phase_refined.py --workers 4
"$RAMAN_PYTHON" validate_frequency_phase_refined.py
"$RAMAN_PYTHON" plot_frequency_phase_refined.py --verify-modal-statistics
"$RAMAN_PYTHON" write_frequency_refined_report.py
"$RAMAN_PYTHON" package_frequency_refined.py
