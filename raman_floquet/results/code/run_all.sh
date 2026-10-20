#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
RAMAN_PYTHON="${RAMAN_PYTHON:-python3}"
"$RAMAN_PYTHON" validate_model.py
"$RAMAN_PYTHON" resonance.py
"$RAMAN_PYTHON" representatives.py
"$RAMAN_PYTHON" kick_baseline.py
"$RAMAN_PYTHON" sambe.py
"$RAMAN_PYTHON" sambe_diagnostics.py
"$RAMAN_PYTHON" sambe_figure.py
"$RAMAN_PYTHON" scan.py --steps 64 --workers 4
"$RAMAN_PYTHON" figures12.py
"$RAMAN_PYTHON" validate_grid.py
"$RAMAN_PYTHON" single_drive_controls.py
