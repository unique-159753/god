#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg .venv/bin/python validate_model.py
