"""Actual exact/effective Raman Floquet scan, retaining modal observables."""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from model import DEFAULT_PARAMS, effective_modes, exact_modes, numerical_backend, static_eigensystem
from observables import summarize_pair


SPECTRAL_KEYS = ["quasienergies", "D_site", "D_component", "D_component_logL", "ipr_site", "ipr_component"]
LINE_PHASES = np.array([0., np.pi / 4, np.pi / 3, np.pi / 2, 3 * np.pi / 4, np.pi])


def flat_comparison(exact, effective):
    pair = summarize_pair(exact, effective)
    flat = {}
    for prefix, modes in [("exact", exact), ("effective", effective)]:
        for key in SPECTRAL_KEYS:
            flat[f"{prefix}_{key}"] = np.asarray(modes[key])
        if prefix == "effective":
            flat["effective_energies"] = modes["energies"]
        for key, value in pair[prefix].items():
            flat[f"{prefix}_{key}"] = value
    for key, value in pair.items():
        if key not in ["exact", "effective"]:
            flat[key] = value
    for key in ["unitarity_error", "eigen_residual", "orthogonality_error",
                "eigenvalue_modulus_error", "schur_offdiagonal_error",
                "min_eigenphase_gap", "degenerate_eigenphase_gaps"]:
        flat[f"exact_{key}"] = exact[key]
    flat["backend"] = str(exact["backend"])
    return flat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    out = args.results
    checkpoints = out / "localization_points"
    checkpoints.mkdir(parents=True, exist_ok=True)
    omega = np.linspace(5.8, 8., 111)
    phases = np.linspace(0., np.pi, 81)
    all_phases = np.r_[phases, np.pi / 3]
    base = Path(__file__).parent
    metadata = {
        "parameters": asdict(DEFAULT_PARAMS), "steps": args.steps,
        "omega_range": [5.8, 8., 111], "phi_range": [0., float(np.pi), 81],
        "extra_exact_slice_phi": float(np.pi / 3), "physical_drive_initial_time": 0.,
        "outer_count": 36, "outer_rule": "center-fold [-omega/2,omega/2), then highest 36 |epsilon|",
        "static_outer_rule": "highest 36 |unfolded E|, fixed across scan",
        "primary_D": "-log(sum_j (|up_j|^2+|down_j|^2)^2)/log(89)",
        "secondary_D": "-log(sum_components |psi|^4)/log(178)",
        "effective_frame": "lab-frame exp(-i K1(0)); K1(0)=B sin(phi) S/omega",
        "matching": "maximum-total-squared-overlap Hungarian assignment in lab frame",
        "model_sha256": hashlib.sha256((base / "model.py").read_bytes()).hexdigest(),
        "observables_sha256": hashlib.sha256((base / "observables.py").read_bytes()).hexdigest(),
    }
    path = out / "localization_parameters.json"
    if path.exists() and json.loads(path.read_text()) != metadata:
        raise RuntimeError("Existing checkpoint parameters or source hashes differ")
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    numerical_backend()
    static_eigensystem()

    def work(item):
        ip, iw = item
        phi, w = float(all_phases[ip]), float(omega[iw])
        path = checkpoints / f"p{ip:03d}_w{iw:03d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                result = {key: data[key] for key in data.files}
            if int(result["steps"]) != args.steps or float(result["omega"]) != w or float(result["phi"]) != phi:
                raise RuntimeError(f"Stale checkpoint {path}")
            return ip, iw, result
        start = time.monotonic()
        exact = exact_modes(w, phi, steps=args.steps)
        effective = effective_modes(w, phi)
        result = flat_comparison(exact, effective)
        result.update(omega=w, phi=phi, steps=args.steps,
                      seconds=time.monotonic() - start)
        for metric in ["exact_unitarity_error", "exact_eigen_residual", "exact_orthogonality_error"]:
            if result[metric] > 1e-8:
                raise RuntimeError(f"Numerically unreliable point {ip,iw}: {metric}={result[metric]}")
        for prefix in ["exact", "effective"]:
            site, comp = result[f"{prefix}_ipr_site"], result[f"{prefix}_ipr_component"]
            if not (np.isfinite(site).all() and np.all(site >= 1/89-1e-12)
                    and np.all(site <= 1+1e-12) and np.all(comp <= site+1e-12)
                    and np.all(site <= 2*comp+1e-12)):
                raise RuntimeError(f"Invalid probabilities at {ip,iw}")
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **result)
        temporary.replace(path)
        return ip, iw, result

    tasks = [(ip, iw) for ip in range(len(all_phases)) for iw in range(len(omega))]
    arrays = None
    csv_rows = []
    total = len(tasks)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, (ip, iw, result) in enumerate(pool.map(work, tasks), 1):
            if arrays is None:
                arrays = {key: np.empty((len(all_phases), len(omega)) + np.shape(value),
                                       dtype=np.asarray(value).dtype)
                          for key, value in result.items() if key != "backend"}
            for key in arrays:
                arrays[key][ip, iw] = result[key]
            scalar_row = {key: np.asarray(value).item() for key, value in result.items()
                          if np.ndim(value) == 0}
            scalar_row.update(index_phi=ip, index_omega=iw, extra_slice=int(ip == 81))
            csv_rows.append(scalar_row)
            if done % 111 == 0 or done == total:
                print(f"{done}/{total} computed/cached; phi/pi={all_phases[ip]/np.pi:.6g}; "
                      f"elapsed={time.monotonic()-started:.1f}s; "
                      f"last deltaD={float(result['delta_D_site']):+.6g}", flush=True)

    metadata_text = json.dumps(metadata)
    primary = {key: value[:81] for key, value in arrays.items()
               if key not in ["omega", "phi"]}
    np.savez_compressed(out / "localization_grid.npz", omega=omega, phi=phases,
                        metadata=metadata_text, **primary)
    line_indices = [0, 20, 81, 40, 60, 80]
    slices = {key: value[line_indices] for key, value in arrays.items()
              if key not in ["omega", "phi"]}
    np.savez_compressed(out / "localization_slices.npz", omega=omega, phi=LINE_PHASES,
                        metadata=metadata_text, **slices)
    fieldnames = list(csv_rows[0])
    with (out / "localization_grid.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row for row in csv_rows if not row["extra_slice"])
    with (out / "localization_extra_pi_over_3.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row for row in csv_rows if row["extra_slice"])
    print(f"Saved 8991 grid points and 111 additional exact phase-cut points in {out}", flush=True)


if __name__ == "__main__":
    main()
