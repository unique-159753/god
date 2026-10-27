"""Refined 100 x 100 exact Floquet IPR/Dn scans of three Raman regimes.

Dimensionless omega/t0: 0.2..10, phase: 0..pi, both inclusive. Every point
is propagated independently with the time-ordered solver. No undriven-baseline
difference observables or effective-Hamiltonian substitutes are computed.
Adaptive time-discretization differences are numerical convergence diagnostics.
"""
from __future__ import annotations

import os
for _thread_variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/raman_matplotlib")

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from frequency_baselines import parameters_for_V
from frequency_fast_native import solve_point


BASE = Path(__file__).resolve().parent
DEFAULT_RESULTS = BASE / "results" / "frequency_phase_100x100_phi_0_pi"
V_VALUES = np.array([.5, 2., 4.])
OMEGA = np.linspace(.2, 10., 100)
PHASE = np.linspace(0., np.pi, 100)
ALL_PHASE = np.r_[PHASE, np.pi / 2]
N = 178


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert_result(result, omega):
    """Spin-resolved full-spectrum observables in the fixed stroboscopic frame."""
    value = dict(result)
    value["IPR"] = np.asarray(value["ipr_component"])
    value["Dn"] = -np.log(value["IPR"]) / np.log(89)
    value["all_IPR"] = float(value["IPR"].mean())
    value["all_Dn"] = float(value["Dn"].mean())
    return value


def comparison(coarse, fine, omega):
    """Check operators and observables; near-degenerate labels are matched."""
    u0, u1 = coarse["sector_operator"], fine["sector_operator"]
    operator_delta = float(np.linalg.norm(u1-u0) / np.sqrt(u1.shape[0]))
    overlap = abs(coarse["modes"].conj().T @ fine["modes"]) ** 2
    # Opposite parity states must not be swapped in a near-degenerate pair.
    if "parity_labels" in coarse and "parity_labels" in fine:
        allowed = coarse["parity_labels"][:, None] == fine["parity_labels"][None, :]
        cost = np.where(allowed, -overlap, 10.)
    else:
        cost = -overlap
    rows, cols = linear_sum_assignment(cost)
    mapping = np.empty(N, int)
    mapping[rows] = cols
    delta = fine["Dn"][mapping] - coarse["Dn"]
    phase_delta = ((fine["quasienergies"][mapping] - coarse["quasienergies"]
                    + omega/2) % omega) - omega/2
    return {
        "operator_delta": operator_delta,
        "max_matched_Dn_delta": float(np.max(abs(delta))),
        "mean_Dn_delta": float(abs(fine["all_Dn"]-coarse["all_Dn"])),
        "minimum_mode_overlap": float(overlap[rows, cols].min()),
        "mean_mode_overlap": float(overlap[rows, cols].mean()),
        "max_matched_quasienergy_delta": float(np.max(abs(phase_delta))),
    }


def tolerance_passed(metrics, args):
    return (metrics["operator_delta"] <= args.operator_tolerance
            and metrics["max_matched_Dn_delta"] <= args.mode_tolerance
            and metrics["mean_Dn_delta"] <= args.mean_tolerance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--initial-steps", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=32768)
    parser.add_argument("--operator-tolerance", type=float, default=1e-6)
    parser.add_argument("--mode-tolerance", type=float, default=1e-4)
    parser.add_argument("--mean-tolerance", type=float, default=1e-5)
    parser.add_argument("--optional-t0-over-h-hz", type=float, default=500.)
    parser.add_argument("--limit", type=int, default=0,
                        help="Compute this many positive-frequency points only; do not pack a partial grid")
    args = parser.parse_args()
    if (args.initial_steps < 1 or args.max_steps < 2*args.initial_steps
            or args.workers < 1 or min(args.operator_tolerance, args.mode_tolerance,
                                     args.mean_tolerance) <= 0):
        raise ValueError("Require positive tolerances/workers and max_steps >= 2*initial_steps")
    out = args.results
    checkpoints = out / "points"
    checkpoints.mkdir(parents=True, exist_ok=True)
    metadata = {
        "frequency_units": "dimensionless omega/t0, no Hz axis",
        "optional_physical_calibration_t0_over_h_hz": args.optional_t0_over_h_hz,
        "parameters": [asdict(parameters_for_V(v)) for v in V_VALUES],
        "omega": OMEGA.tolist(), "phi": PHASE.tolist(),
        "extra_phase_pi_over_2": float(np.pi/2),
        "primary_IPR": "sum over j,spin of |psi_jspin(tau=0)|^4",
        "primary_Dn": "-ln(IPR)/ln(L=89); can exceed one",
        "zero_frequency": "excluded: minimum scanned omega/t0 is 0.2",
        "phase_endpoint_rule": "100 inclusive samples from 0 to pi; no duplicated periodic point",
        "physical_difference_observables": "not computed: no Delta IPR or Delta Dn",
        "convergence_differences": "numerical time-slice diagnostics only; not physical baseline differences",
        "initial_steps": args.initial_steps, "max_steps": args.max_steps,
        "integration_order": 6,
        "integration_backend": "reflection-preserving native bond/star gates; FFT independently validated",
        "operator_tolerance": args.operator_tolerance,
        "mode_tolerance": args.mode_tolerance, "mean_tolerance": args.mean_tolerance,
        "code_sha256": {name: sha256(BASE/name) for name in
                        ["model.py", "frequency_solver.py", "frequency_fast_native.py", "frequency_baselines.py",
                         "scan_frequency_phase_refined.py"]},
        "raw_wavefunctions": "lossless symmetry-sector coefficients plus supplied reconstruction code",
    }
    meta_file = out/"scan_parameters.json"
    if meta_file.exists() and json.loads(meta_file.read_text()) != metadata:
        raise RuntimeError("Existing scan source or parameters differ; use a fresh results directory")
    meta_file.write_text(json.dumps(metadata, indent=2) + "\n")
    def work(item):
        iv, ip, iw = item
        v, phi, omega = float(V_VALUES[iv]), float(ALL_PHASE[ip]), float(OMEGA[iw])
        dest = checkpoints/f"V{iv}_p{ip:02d}_w{iw:02d}.npz"
        if dest.exists():
            with np.load(dest, allow_pickle=False) as data:
                value = {key: data[key] for key in data.files}
            if float(value["omega"]) != omega or float(value["phi"]) != phi:
                raise RuntimeError(f"Stale checkpoint: {dest}")
            return iv, ip, iw, value
        params = parameters_for_V(v)
        started = time.monotonic()
        history = []
        steps = args.initial_steps
        coarse = convert_result(solve_point(omega, phi, params, steps,
                                           return_operator=True, order=6), omega)
        converged = False
        while steps < args.max_steps:
            steps *= 2
            fine = convert_result(solve_point(omega, phi, params, steps,
                                             return_operator=True, order=6), omega)
            metrics = comparison(coarse, fine, omega)
            history.append({"steps": steps, **metrics})
            if tolerance_passed(metrics, args):
                converged = True
                result = fine
                break
            coarse = fine
        if not converged:
            result = coarse
        if float(result["unitarity_error"]) > 1e-8:
            raise RuntimeError(f"Nonunitary result at {item}")
        flat = {key: np.asarray(result[key]) for key in
                ["IPR", "Dn", "quasienergies", "all_IPR", "all_Dn",
                 "unitarity_error", "min_eigenphase_gap"]}
        for key in ["eigen_residual", "orthogonality_error", "parity_labels",
                    "sector_vectors", "sector_quasienergies", "reconstruction_order",
                    "sector_parity", "hermiticity_error", "Q_commutator_error",
                    "particle_hole_error", "sector_projection_error", "sector_unitarity_error",
                    "schur_offdiagonal_error", "eigenvalue_modulus_error"]:
            if key in result:
                flat[key] = np.asarray(result[key])
        if "parity_labels" not in flat:
            raise RuntimeError("The solver must retain a parity label for each of the 178 modes")
        for key in ["eigen_residual", "orthogonality_error", "hermiticity_error",
                    "Q_commutator_error", "particle_hole_error", "sector_projection_error",
                    "sector_unitarity_error", "schur_offdiagonal_error", "eigenvalue_modulus_error"]:
            flat.setdefault(key, np.asarray(np.nan))
        flat.update(omega=omega, phi=phi, V=v, steps=steps, converged=converged,
                    is_frozen=False, seconds=time.monotonic()-started,
                    convergence_history_json=json.dumps(history))
        for name in ["operator_delta", "max_matched_Dn_delta", "mean_Dn_delta",
                     "minimum_mode_overlap", "mean_mode_overlap",
                     "max_matched_quasienergy_delta"]:
            flat["final_"+name] = history[-1][name] if history else np.nan
        temp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(temp, **flat)
        temp.replace(dest)
        return iv, ip, iw, flat

    tasks = [(iv, ip, iw) for iv in range(3) for ip in range(len(ALL_PHASE)) for iw in range(len(OMEGA))]
    if args.limit:
        tasks = tasks[:args.limit]
    arrays = None
    rows = []
    failed = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, (iv, ip, iw, value) in enumerate(pool.map(work, tasks), 1):
            if arrays is None:
                arrays = {key: np.empty((len(V_VALUES), len(ALL_PHASE), len(OMEGA)) + np.shape(item), dtype=np.asarray(item).dtype)
                          for key, item in value.items()
                          if key not in ["sector_vectors", "sector_quasienergies", "reconstruction_order",
                                         "sector_parity", "frozen_modes", "convergence_history_json",
                                         "omega", "phi", "V"]}
            for key in arrays:
                if key in value:
                    arrays[key][iv, ip, iw] = value[key]
                else:
                    arrays[key][iv, ip, iw] = np.nan
            scalar = {key: np.asarray(item).item() for key, item in value.items()
                      if np.ndim(item) == 0 and key != "convergence_history_json"}
            scalar.update(index_V=iv, index_phi=ip, index_omega=iw, extra_slice=(ip == len(PHASE)))
            rows.append(scalar)
            if not bool(value["converged"]):
                failed.append([iv, ip, iw])
            if done % 50 == 0 or done == len(tasks):
                print(f"{done}/{len(tasks)} V={V_VALUES[iv]:g} phi/pi={ALL_PHASE[ip]/np.pi:.4g} "
                      f"omega={OMEGA[iw]:.4g} M={int(value['steps'])} "
                      f"elapsed={time.monotonic()-started:.1f}s nonconverged={len(failed)}", flush=True)
    if args.limit:
        print("Diagnostic subset saved; no partial grid exported", flush=True)
        return
    grid = {key: item[:, :len(PHASE)] for key, item in arrays.items()}
    slices = {key: item[:, [0, len(PHASE)]] for key, item in arrays.items()}
    np.savez_compressed(out/"frequency_phase_scan.npz", V=V_VALUES, omega=OMEGA, phi=PHASE,
                        metadata=json.dumps(metadata), **grid)
    np.savez_compressed(out/"frequency_phase_slices.npz", V=V_VALUES, omega=OMEGA,
                        phi=np.array([0., np.pi/2]), metadata=json.dumps(metadata), **slices)
    fields = list(rows[0])
    for name, extra in [("frequency_phase_scan.csv", False), ("frequency_extra_pi_over_2.csv", True)]:
        with (out/name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row for row in rows if bool(row["extra_slice"]) == extra)
    # One row per actual Floquet eigenstate; no zero-frequency or interpolated data.
    with (out/"eigenstate_observables.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["V", "phi", "omega", "state", "quasienergy", "IPR", "Dn", "parity"])
        for iv in range(3):
            for ip in range(len(ALL_PHASE)):
                for iw in range(len(OMEGA)):
                    for n in range(N):
                        writer.writerow([V_VALUES[iv], ALL_PHASE[ip], OMEGA[iw], n,
                                         arrays["quasienergies"][iv, ip, iw, n],
                                         arrays["IPR"][iv, ip, iw, n], arrays["Dn"][iv, ip, iw, n],
                                         arrays["parity_labels"][iv, ip, iw, n]])
    summary = {"grid_points": int(len(V_VALUES)*len(PHASE)*len(OMEGA)),
               "additional_pi_over_2_points": int(len(V_VALUES)*len(OMEGA)),
               "frozen_points_in_grid": 0,
               "positive_frequency_points_in_grid": int(len(V_VALUES)*len(PHASE)*len(OMEGA)),
               "nonconverged": failed,
               "positive_steps_range": [int(arrays["steps"].min()),
                                        int(arrays["steps"].max())],
               "seconds": time.monotonic()-started}
    (out/"scan_completion.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise RuntimeError("Some points failed adaptive convergence; do not present as converged results")


if __name__ == "__main__":
    main()
