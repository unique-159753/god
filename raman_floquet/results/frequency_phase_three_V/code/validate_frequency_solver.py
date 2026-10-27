"""Independent full-space FFT/ODE checks for the symmetry-sector Floquet solver."""
from __future__ import annotations

import os
for _name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import schur
from scipy.optimize import linear_sum_assignment

from frequency_baselines import parameters_for_V
from frequency_solver import (reconstruct_modes, reconstruct_operator, sector_bases,
                              solve_point, spin_swap, symmetry_diagnostics)
from model import build_floquet_operator, fold_quasienergies, localization_observables
from validate_model import reference_propagator


def reference_sector_modes(operator, omega):
    """Resolve Q before mode comparisons, not an arbitrary full-space basis."""
    basis, _ = sector_bases(89)
    sector = basis.T @ operator @ basis
    triangular, vectors = schur(sector, output="complex", check_finite=False)
    values = np.diag(triangular)
    eps = fold_quasienergies(-np.angle(values)*omega/(2*np.pi), omega)
    index = np.argsort(eps, kind="stable")
    physical = basis @ vectors[:, index]
    return {"sector_operator": sector, "vectors": physical,
            "Dn": localization_observables(physical)["D_component_logL"],
            "eps": eps[index]}


def compare_reference(candidate, reference, omega):
    basis, _ = sector_bases(89)
    physical = basis @ candidate["sector_vectors"]
    overlap = abs(reference["vectors"].conj().T @ physical)**2
    rows, cols = linear_sum_assignment(-overlap)
    mapping = np.empty(89, int)
    mapping[rows] = cols
    d = localization_observables(physical)["D_component_logL"]
    return {
        "operator_relative_Frobenius_error": float(np.linalg.norm(candidate["sector_operator"]-reference["sector_operator"])/np.sqrt(89)),
        "max_matched_Dn_error": float(np.max(abs(d[mapping]-reference["Dn"]))),
        "all_Dn_error": float(abs(d.mean()-reference["Dn"].mean())),
        "minimum_mode_overlap": float(overlap[rows,cols].min()),
        "max_matched_quasienergy_error": float(np.max(abs(fold_quasienergies(candidate["sector_quasienergies"][mapping]-reference["eps"], omega)))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent/"results/frequency_phase_three_V")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    rows, arrays, failures = [], {}, []
    started = time.perf_counter()
    for V in [.5, 2., 4.]:
        params = parameters_for_V(V)
        for omega in [1., 10.]:
            phi = np.pi/3
            ode, ode_meta = reference_propagator(params, omega, phi)
            reference = reference_sector_modes(ode, omega)
            values = []
            for steps in [32, 64, 128, 256, 512]:
                result = solve_point(omega, phi, params, steps)
                comparison = compare_reference(result, reference, omega)
                row = {"V": V, "omega": omega, "phi": phi, "steps": steps,
                       "unitarity_error": result["unitarity_error"],
                       "eigen_residual": result["eigen_residual"],
                       "sector_projection_error": result["sector_projection_error"],
                       **comparison}
                rows.append(row)
                values.append(result)
            final = values[-1]
            tag = f'V{str(V).replace(".","p")}_w{int(omega)}'
            arrays[f"{tag}_reference_operator"] = ode
            arrays[f"{tag}_sector_vectors"] = final["sector_vectors"]
            arrays[f"{tag}_reconstruction_order"] = final["reconstruction_order"]
            full_fft = build_floquet_operator(omega, phi, 128, params, backend="fft")
            reconstructed = reconstruct_operator(values[2]["sector_operator"])
            fft_error = float(np.linalg.norm(full_fft-reconstructed)/np.sqrt(178))
            mode_error = float(np.max(abs(final["modes"]-reconstruct_modes(final["sector_vectors"], final["reconstruction_order"]))))
            reverse = solve_point(omega, 2*np.pi-phi, params, 512)
            reverse_u = reconstruct_operator(reverse["sector_operator"])
            final_u = reconstruct_operator(final["sector_operator"])
            transpose_error = float(np.linalg.norm(reverse_u-final_u.T)/np.sqrt(178))
            c_identity_error = float(np.linalg.norm(spin_swap(spin_swap(final_u.conj().T,89).T,89)-final_u)/np.sqrt(178))
            reverse_ref = reference_sector_modes(reverse_u,omega)
            # U(phi)^T has conjugated eigenmodes with the same quasienergies.
            mapped = dict(reference_sector_modes(final_u,omega))
            mapped["vectors"] = mapped["vectors"].conj()
            reverse_modes = sector_bases()[0] @ reverse["sector_vectors"]
            overlap = abs(mapped["vectors"].conj().T @ reverse_modes)**2
            r,c = linear_sum_assignment(-overlap)
            dn_reverse = localization_observables(reverse_modes)["D_component_logL"]
            phi_dn_error = float(np.max(abs(mapped["Dn"][r]-dn_reverse[c])))
            row = {"V":V,"omega":omega,"symmetry":symmetry_diagnostics(params),
                   "reference_method":ode_meta,
                   "full_fft_M128_relative_error":fft_error,
                   "reconstruction_max_error":mode_error,
                   "phase_reversal_U_transpose_error_M512":transpose_error,
                   "phase_reversal_max_matched_Dn_error_M512":phi_dn_error,
                   "particle_hole_U_identity_error":c_identity_error,
                   "M512_vs_ODE":compare_reference(final,reference,omega)}
            arrays[f"{tag}_diagnostics_json"] = json.dumps(row)
            if max(fft_error,mode_error,transpose_error,c_identity_error)>1e-9:
                failures.append(f"Structural/symmetry check failed at V={V},omega={omega}")
            if phi_dn_error>1e-6:
                failures.append(f"Phase reversal Dn failed at V={V},omega={omega}")
            print(json.dumps(row),flush=True)
    output = {"scope":"full-space independent FFT/ODE validation; no Hz interpretation",
              "validation_integrator_order":4,
              "frequency_solver_sha256":hashlib.sha256((Path(__file__).parent/"frequency_solver.py").read_bytes()).hexdigest(),
              "primary_IPR":"sum over 178 spin-resolved components |psi|^4",
              "primary_Dn":"-log(IPR)/log(89)",
              "raw_mode_reconstruction":"Q+ basis and exact C conjugation; no interpolation",
              "particle_hole_error_field":"solve_point scalar reports maximum generator C anticommutator; dynamical identity tested here independently",
              "rows":rows,"failures":failures,"passed":not failures,
              "seconds":time.perf_counter()-started}
    (args.results/"frequency_solver_validation.json").write_text(json.dumps(output,indent=2)+"\n")
    with (args.results/"frequency_solver_validation.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.results/"frequency_solver_validation.npz",**arrays)
    if failures:
        raise SystemExit("; ".join(failures))
    print("Frequency solver structural validation PASS",flush=True)


if __name__ == "__main__":
    main()
