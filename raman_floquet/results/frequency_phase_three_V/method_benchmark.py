"""Reproduce the legacy method stress test, not the primary frequency scan.

Only explicit dimensionless frequencies are used; no Hz calibration is
assumed. D fields here use the old ln(178) normalization, and are clearly
separate from the new main Dn=-ln(IPR)/ln(89). The static parameters and
full time-dependent Hamiltonian are unchanged. No artificial data appear.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import DEFAULT_PARAMS, exact_modes


def run_sequence(omega, steps_values, params):
    previous, rows = None, []
    reflection = np.eye(89)[(-np.arange(89)-2) % 89]
    q = np.kron(reflection, np.diag([1., -1.]))
    for steps in steps_values:
        x = exact_modes(omega, np.pi/3, steps=steps, params=params, return_operator=True)
        u = x["operator"]
        selected = np.argsort(abs(x["quasienergies"]), kind="stable")[-36:]
        row = {"omega": omega, "steps": steps, "seconds": x["total_seconds"],
               "mean_D": float(x["D_component"].mean()),
               "mean_IPR": float(x["ipr_component"].mean()),
               "outer_D": float(x["D_component"][selected].mean()),
               "unitarity": x["unitarity_error"],
               "Q_commutator": float(np.linalg.norm(u@q-q@u)/np.sqrt(178))}
        if previous is not None:
            overlap = abs(previous["modes"].conj().T@x["modes"])**2
            i, j = linear_sum_assignment(-overlap)
            row.update(operator_change=float(np.linalg.norm(u-previous["operator"])/np.sqrt(178)),
                       max_D_change=float(abs(x["D_component"][j]-previous["D_component"][i]).max()),
                       min_match_overlap=float(overlap[i, j].min()))
        rows.append(row)
        previous = x
        print(json.dumps(row), flush=True)
    return rows


def main():
    params = replace(DEFAULT_PARAMS, V=4., W=0.)
    rows = []
    for omega in [.1, 1., 10.]:
        rows.extend(run_sequence(omega, [32, 64, 128, 256, 512, 1024], params))
        if omega in [.1, 10.]:
            x = exact_modes(omega, np.pi/3, steps=128, params=params,
                            return_operator=True, backend="fft")
            q = np.kron(np.eye(89)[(-np.arange(89)-2) % 89], np.diag([1., -1.]))
            rows.append({"omega": omega, "steps": 128, "backend": "fft",
                         "seconds": x["total_seconds"], "unitarity": x["unitarity_error"],
                         "Q_commutator": float(np.linalg.norm(x["operator"]@q-q@x["operator"])/np.sqrt(178)),
                         "mean_D": float(x["D_component"].mean())})
    extension = run_sequence(.1, [1024, 2048, 4096, 8192, 16384], params)
    data = {"dimensionless_only": True, "physical_t0_over_h_not_assumed": True,
            "not_primary_scan": True, "D_normalization": "legacy ln(178), not primary ln(89)",
            "parameters": {"V": 4., "W": 0., "phi": float(np.pi/3)},
            "rows": rows, "low_frequency_extension": extension}
    Path(__file__).with_name("method_benchmark.json").write_text(json.dumps(data, indent=2)+"\n")


if __name__ == "__main__":
    main()
