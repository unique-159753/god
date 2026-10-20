"""True one-drive controls at fixed amplitudes and laboratory time origin.

No A-only phase dependence is introduced: phi occurs only in the B term.
For B-only, phi is an absolute driving phase and changes the tau=0 modes,
although the exact Floquet quasienergies are invariant under this time shift.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

# Keep this small supplementary calculation within the shared CPU budget.
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import numpy as np

from model import DEFAULT_PARAMS, effective_modes, exact_modes
from observables import extract_stats, serializable_stats


METRICS = ("outer_D_site", "projected_D_site", "outer_D_component",
           "projected_D_component", "outer_D_component_logL",
           "all_D_site", "all_D_component", "outer_ipr_site",
           "projected_ipr_site", "outer_ipr_component", "projected_ipr_component")
SPECTRA = ("quasienergies", "ipr_site", "ipr_component", "D_site",
           "D_component", "D_component_logL")


def _double_reference(root):
    source = root / "kick_only_baseline.csv"
    if not source.exists():
        return {}, None
    with source.open(newline="") as stream:
        data = list(csv.DictReader(stream))
    indexed = {(round(float(row["omega"]), 8), round(float(row["phi"]), 8)): row
               for row in data}
    return indexed, hashlib.sha256(source.read_bytes()).hexdigest()


def run(root: Path, steps: int = 128):
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    omegas = np.array([6.2, 8., 16.])
    phases = np.array([0., np.pi / 4, np.pi / 2])
    kinds = ("A_only", "B_only")
    controls = (replace(DEFAULT_PARAMS, A=0.8, B=0.),
                replace(DEFAULT_PARAMS, A=0., B=0.8))
    nstates = 2 * DEFAULT_PARAMS.L
    shape = (len(kinds), len(omegas), len(phases), nstates)
    arrays = {f"{model}_{metric}": np.empty(shape)
              for model in ("static", "kick", "exact") for metric in SPECTRA}
    outer = {f"{model}_outer_indices": np.empty(shape[:-1] + (36,), dtype=int)
             for model in ("static", "kick", "exact")}
    projected = {f"{model}_static_outer_weights": np.empty(shape)
                 for model in ("static", "kick", "exact")}
    double_reference, reference_hash = _double_reference(root)
    rows, phase_checks, convergence = [], [], []
    for ik, (kind, params) in enumerate(zip(kinds, controls)):
        for iw, omega in enumerate(omegas):
            previous = None
            cached_A_exact = None
            for ip, phi in enumerate(phases):
                static = effective_modes(omega, phi, params,
                                         include_first_order=False,
                                         kick_corrected=False)
                kicked = effective_modes(omega, phi, params,
                                         include_first_order=False,
                                         kick_corrected=True)
                # A-only Hamiltonian has no phi. Calculate it once per omega.
                if kind == "A_only" and cached_A_exact is not None:
                    exact = cached_A_exact
                else:
                    exact = exact_modes(omega, phi, steps=steps, params=params)
                    if kind == "A_only":
                        cached_A_exact = exact
                results = {"static": static, "kick": kicked, "exact": exact}
                stats = {name: extract_stats(result, params)
                         for name, result in results.items()}
                row = {"control": kind, "omega": float(omega), "phi": float(phi),
                       "phi_over_pi": float(phi / np.pi),
                       "A": params.A, "B": params.B, "g_F": exact.get("g_F", 0.),
                       "steps": steps, "backend": exact["backend"]}
                for name, result in results.items():
                    for metric in SPECTRA:
                        arrays[f"{name}_{metric}"][ik, iw, ip] = result[metric]
                    outer[f"{name}_outer_indices"][ik, iw, ip] = stats[name]["outer_indices"]
                    projected[f"{name}_static_outer_weights"][ik, iw, ip] = stats[name]["static_outer_weights"]
                    for metric in METRICS:
                        row[f"{metric}_{name}"] = stats[name][metric]
                for metric in ("outer_D_site", "projected_D_site", "outer_D_component",
                               "projected_D_component", "all_D_site", "all_D_component"):
                    row[f"{metric}_exact_minus_static"] = stats["exact"][metric] - stats["static"][metric]
                    row[f"{metric}_kick_minus_static"] = stats["kick"][metric] - stats["static"][metric]
                    row[f"{metric}_exact_minus_kick"] = stats["exact"][metric] - stats["kick"][metric]
                for diagnostic in ("unitarity_error", "eigen_residual", "orthogonality_error",
                                   "eigenvalue_modulus_error", "schur_offdiagonal_error",
                                   "min_eigenphase_gap", "degenerate_eigenphase_gaps"):
                    row[diagnostic] = exact[diagnostic]
                for name in results:
                    row[f"{name}_static_outer_weight_sum_error"] = stats[name]["static_outer_weight_sum_error"]
                ref = double_reference.get((round(float(omega), 8), round(float(phi), 8)))
                if ref is not None:
                    for metric in ("outer_D_site", "projected_D_site", "outer_D_component", "projected_D_component"):
                        for ref_model in ("A_static", "B_effective", "C_exact", "D_compensated"):
                            row[f"{metric}_double_{ref_model}"] = float(ref[f"{metric}_{ref_model}"])
                        row[f"{metric}_single_minus_double_exact"] = stats["exact"][metric] - float(ref[f"{metric}_C_exact"])
                rows.append(row)
                if previous is None:
                    previous = exact
                phase_checks.append({"control": kind, "omega": float(omega),
                                     "phi_over_pi": float(phi / np.pi),
                                     "quasienergies_vs_phi0_max_abs": float(np.max(np.abs(exact["quasienergies"] - previous["quasienergies"]))),
                                     "outer_D_site_vs_phi0": stats["exact"]["outer_D_site"] - extract_stats(previous, params)["outer_D_site"]})
                print(json.dumps({key: row[key] for key in ("control", "omega", "phi_over_pi",
                                 "outer_D_site_static", "outer_D_site_kick", "outer_D_site_exact",
                                 "outer_D_site_exact_minus_static", "projected_D_site_exact_minus_static",
                                 "unitarity_error")}), flush=True)
            # Supplement the already validated M128 solver at the lowest frequency
            # for one point of each genuinely different one-drive Hamiltonian.
            if iw == 0:
                phi = phases[-1]
                refined = exact_modes(omega, phi, steps=2 * steps, params=params)
                base_stats = extract_stats(exact, params)
                fine_stats = extract_stats(refined, params)
                convergence.append({"control": kind, "omega": float(omega),
                                    "phi_over_pi": 0.5, "coarse_steps": steps,
                                    "refined_steps": 2 * steps,
                                    "max_quasienergy_difference": float(np.max(np.abs(exact["quasienergies"] - refined["quasienergies"]))),
                                    **{f"abs_difference_{metric}": abs(base_stats[metric] - fine_stats[metric])
                                       for metric in ("outer_D_site", "projected_D_site", "outer_D_component", "projected_D_component")}})
    with (root / "single_drive_controls.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(root / "single_drive_controls.npz", omega=omegas, phi=phases,
                        control_names=np.array(kinds), **arrays, **outer, **projected)
    summary = {
        "description": "True one-drive controls with unchanged H0, site-spin-summed and component localization, laboratory tau=0 retained",
        "base_parameters": asdict(DEFAULT_PARAMS),
        "controls": {kind: asdict(params) for kind, params in zip(kinds, controls)},
        "steps": steps, "tau0": 0., "outer_selection": "highest 36 absolute centered folded quasienergies, separately in each model",
        "projected_selection": "fixed highest-|E| static 36-state projector; mode weights normalized by 36",
        "D_definitions": {"site": "-log(sum_j (p_up+p_down)^2)/log(L)",
                          "component": "-log(sum_j,s p_js^2)/log(2L)"},
        "g_F_zero_for_both_single_drive_controls": True,
        "A_only_phi_independence": "phi enters only the B term and B=0, so the same exact U is reused across phi at each omega",
        "B_only_phi_interpretation": "Changing phi shifts the single drive in time: quasienergies are invariant, but tau=0 modes and their localization need not be. The leading kick is B sin(phi) S/omega.",
        "not_an_undriven_control": "g_F=0 or phi=0 does not remove the time-dependent one-drive Hamiltonian. Static A is the undriven reference.",
        "causal_limit": "These finite-size representative-point differences are not additive causal contributions and do not alone prove a new mechanism or phase transition.",
        "double_reference_source": "kick_only_baseline.csv" if double_reference else None,
        "double_reference_sha256": reference_hash,
        "max_unitarity_error": max(row["unitarity_error"] for row in rows),
        "max_eigen_residual": max(row["eigen_residual"] for row in rows),
        "phase_checks": phase_checks, "M128_M256_checks": convergence,
        "wall_seconds": time.perf_counter() - started, "rows": rows}
    assert summary["max_unitarity_error"] < 1e-10
    assert all(check["quasienergies_vs_phi0_max_abs"] < 1e-5 for check in phase_checks)
    (root / "single_drive_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--steps", type=int, default=128)
    arguments = parser.parse_args()
    run(arguments.output_dir, arguments.steps)
