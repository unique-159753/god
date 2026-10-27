"""Undriven and zero-frequency Raman baselines, with fixed parity convention.

The primary IPR resolves all 2L spin components and primary Dn=-ln(IPR)/ln(L).
Zero angular frequency defines a frozen Hamiltonian, not a Floquet period:
its quasienergies and Floquet-outer means are undefined (NaN).
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

from model import DEFAULT_PARAMS, localization_observables, operators


V_VALUES = np.array([0.5, 2.0, 4.0])
DEFAULT_PHASES = 2 * np.pi * np.arange(50) / 50
SLICE_PHASES = np.array([0., np.pi / 2])
OUTER_FRACTION = 0.2


def parameters_for_V(V):
    return replace(DEFAULT_PARAMS, L=89, alpha=55/89, t0=1., delta0=.5,
                   A=.8, B=.8, W=0., theta=0., V=float(V))


@lru_cache(maxsize=8)
def reflection_spin_bases(L=89):
    """Return deterministic Q=+1,-1 bases for Q=R tensor sigma_z.

    j runs from one to L, R:j->-j mod L, so zero-index R(i)=(-i-2)%L.
    Columns are ordered by spin, then first site of each reflection orbit.
    No numerical diagonalization of the multiply-degenerate Q is needed.
    """
    reflection = (-np.arange(L)-2) % L
    bases = []
    for q in (1, -1):
        columns = []
        for spin, sz in ((0, 1), (1, -1)):
            for i, partner in enumerate(reflection):
                j = int(partner)
                if i > j:
                    continue
                column = np.zeros(2*L)
                if i == j:
                    if q != sz:
                        continue
                    column[2*i+spin] = 1.
                else:
                    column[2*i+spin] = 1/np.sqrt(2.)
                    column[2*j+spin] = (q*sz)/np.sqrt(2.)
                columns.append(column)
        basis = np.column_stack(columns)
        basis.setflags(write=False)
        bases.append(basis)
    return tuple(bases)


@lru_cache(maxsize=8)
def reflection_spin_operator(L=89):
    reflection = (-np.arange(L)-2) % L
    q = np.kron(np.eye(L)[reflection], np.diag([1., -1.]))
    q.setflags(write=False)
    return q


def energy_outer_indices(energies):
    count = int(np.ceil(OUTER_FRACTION * len(energies)))
    return np.argsort(abs(np.asarray(energies)), kind="stable")[-count:]


def diagonalize_static(H, params=None):
    """Parity-resolved Hermitian diagonalization of an H commuting with Q.

    The `outer_*` values here are *energy*-outer diagnostics. A caller must
    not label them Floquet outer values at omega=0. `D_component` and `Dn`
    use ln(L); `D_component_log2L` retains the former ln(2L) convention.
    Full residuals are measured against the unprojected input H.
    """
    if params is None:
        params = parameters_for_V(2.)
    H = np.asarray(H)
    n, L = H.shape[0], params.L
    if H.shape != (2*L, 2*L):
        raise ValueError("H must have shape (2L,2L)")
    scale = max(float(np.linalg.norm(H)), np.finfo(float).tiny)
    hermiticity_error = float(np.linalg.norm(H-H.conj().T)/scale)
    q = reflection_spin_operator(L)
    q_error = float(np.linalg.norm(q@H-H@q)/scale)
    if hermiticity_error > 1e-11 or q_error > 1e-10:
        raise ValueError(f"Expected Hermitian Q-symmetric H: {hermiticity_error=}, {q_error=}")
    values, vectors, labels, block_gaps = [], [], [], []
    for parity, basis in zip((1, -1), reflection_spin_bases(L)):
        block = basis.conj().T @ H @ basis
        block = (block+block.conj().T)/2
        e, v = eigh(block, check_finite=False)
        full = basis @ v
        # Reproducible irrelevant phase: largest-magnitude entry real positive.
        pivot = np.argmax(abs(full), axis=0)
        phase = full[pivot, np.arange(full.shape[1])]
        if np.iscomplexobj(full):
            full *= np.exp(-1j*np.angle(phase))[None, :]
        else:
            full *= np.where(phase < 0., -1., 1.)[None, :]
        values.append(e)
        vectors.append(full)
        labels.append(np.full(len(e), parity, dtype=np.int8))
        block_gaps.append(float(np.min(np.diff(e))))
    energies = np.concatenate(values)
    order = np.argsort(energies, kind="stable")
    energies = energies[order]
    modes = np.column_stack(vectors)[:, order]
    parity_labels = np.concatenate(labels)[order]
    obs = localization_observables(modes)
    primary = -np.log(obs["ipr_component"])/np.log(L)
    result = dict(obs)
    result.update(energies=energies, modes=modes, parity_labels=parity_labels,
                  IPR=obs["ipr_component"], Dn=primary,
                  D_component=primary, D_component_logL=primary,
                  D_component_log2L=obs["D_component"],
                  hermiticity_error=hermiticity_error, Q_commutator_error=q_error,
                  eigen_residual=float(np.linalg.norm(H@modes-modes*energies[None, :])/scale),
                  orthogonality_error=float(np.linalg.norm(modes.conj().T@modes-np.eye(n))/np.sqrt(n)),
                  parity_residual=float(np.linalg.norm(q@modes-modes*parity_labels[None, :])/np.sqrt(n)),
                  min_energy_gap=float(np.min(np.diff(energies))),
                  min_parity_block_energy_gap=float(min(block_gaps)))
    spin_flip = np.arange(n) ^ 1
    result["particle_hole_error"] = float(np.linalg.norm(H.conj()[spin_flip][:, spin_flip]+H)/scale)
    selected = energy_outer_indices(energies)
    magnitude = np.sort(abs(energies))
    result["outer_indices"] = selected
    result["energy_outer_indices"] = selected
    result["energy_outer_selection_gap"] = float(magnitude[-len(selected)]-magnitude[-len(selected)-1])
    for key in ("ipr_component", "ipr_site", "D_component", "D_component_logL",
                "D_component_log2L", "D_site"):
        result[f"all_{key}"] = float(np.mean(result[key]))
        result[f"outer_{key}"] = float(np.mean(result[key][selected]))
        result[f"energy_outer_{key}"] = result[f"outer_{key}"]
    result["all_IPR"] = result["all_ipr_component"]
    result["all_Dn"] = result["all_D_component"]
    result["energy_outer_Dn"] = result["energy_outer_D_component"]
    return result


def frozen_hamiltonian(phi, params):
    ops = operators(params)
    return ((params.t0+params.A)*ops["T"]
            + (params.delta0+params.B*np.cos(phi))*ops["S"]+ops["HZ"])


def _summarize_result(result):
    names = ["all_ipr_component", "all_D_component", "all_D_component_log2L",
             "all_ipr_site", "all_D_site", "energy_outer_ipr_component",
             "energy_outer_D_component", "energy_outer_D_component_log2L",
             "energy_outer_ipr_site", "energy_outer_D_site", "energy_outer_selection_gap",
             "hermiticity_error", "Q_commutator_error", "eigen_residual",
             "orthogonality_error", "parity_residual", "min_energy_gap",
             "min_parity_block_energy_gap", "particle_hole_error"]
    return {name: float(result[name]) for name in names}


def build_baselines(output_dir, phases=None, save_modes=True):
    """Compute independent H0 baselines and frozen omega=0 states.

    NPZ `zero_outer_Dn` and `zero_outer_D_component` are NaN. Separate
    `zero_energy_outer_*` arrays explicitly retain static-energy diagnostics.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    phases = DEFAULT_PHASES if phases is None else np.asarray(phases, float)
    static_results, zero_results, slice_results = [], [], []
    rows, states = [], []
    for iv, V in enumerate(V_VALUES):
        params = parameters_for_V(V)
        result = diagonalize_static(operators(params)["H0"], params)
        static_results.append(result)
        rows.append(dict(kind="undriven_H0", V=float(V), phi=float("nan"),
                         floquet_outer_Dn=float("nan"), **_summarize_result(result)))
        zero_row, slice_row = [], []
        for phi in phases:
            value = diagonalize_static(frozen_hamiltonian(float(phi), params), params)
            zero_row.append(value)
            rows.append(dict(kind="frozen_omega_zero", V=float(V), phi=float(phi),
                             floquet_outer_Dn=float("nan"), **_summarize_result(value)))
        for phi in SLICE_PHASES:
            slice_row.append(diagonalize_static(frozen_hamiltonian(float(phi), params), params))
        zero_results.append(zero_row)
        slice_results.append(slice_row)
        for kind, collection in [("undriven_H0", [(np.nan, result)]),
                                 ("frozen_omega_zero", zip(phases, zero_row)),
                                 ("frozen_line_slice", zip(SLICE_PHASES, slice_row))]:
            for phi, value in collection:
                selected = set(value["energy_outer_indices"].tolist())
                for i, e in enumerate(value["energies"]):
                    states.append(dict(kind=kind, V=float(V), phi=float(phi), mode=i,
                                       energy=float(e), parity=int(value["parity_labels"][i]),
                                       ipr_component=float(value["ipr_component"][i]),
                                       Dn=float(value["D_component"][i]),
                                       D_component_log2L=float(value["D_component_log2L"][i]),
                                       ipr_site=float(value["ipr_site"][i]),
                                       D_site=float(value["D_site"][i]),
                                       selected_by_static_energy=int(i in selected),
                                       quasienergy=float("nan"), floquet_outer_defined=0))
        print(f"Baselines V={V:g}: H0 mean Dn={result['all_Dn']:.9f}; {len(phases)} frozen phases", flush=True)

    metadata = {
        "parameters": [asdict(parameters_for_V(v)) for v in V_VALUES],
        "primary_IPR": "sum over j,spin of |psi_jspin|^4",
        "primary_Dn": "-ln(IPR_component)/ln(L=89); can exceed one",
        "alternative_D_component_log2L": "-ln(IPR_component)/ln(178)",
        "basis": "Q=reflection(j->-j mod L) tensor sigma_z resolved, fixed deterministic real bases",
        "Q_column_order": "parity + then -; within each basis spin then first site in reflection orbit",
        "state_sort": "increasing energy, stable within numerical ties",
        "zero_frequency": "frozen H=(t0+A)T+(delta0+B cos(phi))S+HZ; no Floquet period or quasienergy",
        "zero_floquet_outer": "undefined: NaN with indices -1; energy_outer diagnostics separately labeled",
        "undriven_reference": "H0=t0 T+delta0 S+HZ separately at each V; no drive A/B terms",
        "energy_outer_rule": "highest ceil(0.2*178)=36 absolute unfolded static energies; not Floquet outer",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(Path(__file__).with_name("model.py").read_bytes()).hexdigest(),
    }
    arrays = {"V": V_VALUES, "phi": phases, "slice_phi": SLICE_PHASES,
              "metadata": json.dumps(metadata),
              "Q_plus_basis": reflection_spin_bases(89)[0],
              "Q_minus_basis": reflection_spin_bases(89)[1]}
    state_keys = ["energies", "parity_labels", "ipr_component", "D_component",
                  "D_component_logL", "D_component_log2L", "ipr_site", "D_site",
                  "density", "energy_outer_indices"]
    if save_modes:
        state_keys.append("modes")
    for key in state_keys + list(_summarize_result(static_results[0])):
        arrays[f"static_{key}"] = np.asarray([x[key] for x in static_results])
        arrays[f"zero_{key}"] = np.asarray([[x[key] for x in row] for row in zero_results])
        arrays[f"zero_slice_{key}"] = np.asarray([[x[key] for x in row] for row in slice_results])
    arrays.update(static_IPR=arrays["static_ipr_component"], static_Dn=arrays["static_D_component"],
                  static_all_IPR=arrays["static_all_ipr_component"], static_all_Dn=arrays["static_all_D_component"],
                  zero_IPR=arrays["zero_ipr_component"], zero_Dn=arrays["zero_D_component"],
                  zero_all_IPR=arrays["zero_all_ipr_component"], zero_all_Dn=arrays["zero_all_D_component"],
                  zero_outer_Dn=np.full((len(V_VALUES), len(phases)), np.nan),
                  zero_outer_D_component=np.full((len(V_VALUES), len(phases)), np.nan),
                  zero_quasienergies=np.full((len(V_VALUES), len(phases), 178), np.nan),
                  zero_outer_indices=np.full((len(V_VALUES), len(phases), 36), -1, dtype=np.int16))
    arrays["zero_delta_all_Dn_vs_undriven"] = arrays["zero_all_Dn"]-arrays["static_all_Dn"][:, None]
    np.savez_compressed(out/"baselines.npz", **arrays)
    for name, data in [("baselines.csv", rows), ("baselines_states.csv", states)]:
        with (out/name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    # Compare independently constructed/eigen-solved partner phases, never an interpolation.
    phase_checks = []
    for iv, V in enumerate(V_VALUES):
        params = parameters_for_V(V)
        for ip, phi in enumerate(phases):
            distance = abs(np.angle(np.exp(1j*(phases+phi))))
            jp = int(np.argmin(distance))
            if distance[jp] > 1e-10 or jp < ip:
                continue
            left, right = zero_results[iv][ip], zero_results[iv][jp]
            overlap = abs(left["modes"].conj().T@right["modes"])**2
            allowed = left["parity_labels"][:, None] == right["parity_labels"][None, :]
            i, j = linear_sum_assignment(np.where(allowed, -overlap, 10.))
            phase_checks.append(dict(V=float(V), phi=float(phi), partner_phi=float(phases[jp]),
                H_max_error=float(np.max(abs(frozen_hamiltonian(phi, params)-frozen_hamiltonian(phases[jp], params)))),
                spectrum_max_error=float(np.max(abs(left["energies"]-right["energies"]))),
                matched_IPR_max_error=float(np.max(abs(left["ipr_component"][i]-right["ipr_component"][j]))),
                mean_Dn_error=abs(left["all_Dn"]-right["all_Dn"]),
                minimum_matched_overlap=float(overlap[i,j].min())))
    all_results = static_results+[x for row in zero_results+slice_results for x in row]
    maxima = {key: float(max(x[key] for x in all_results)) for key in
              ["hermiticity_error", "Q_commutator_error", "eigen_residual",
               "orthogonality_error", "parity_residual", "particle_hole_error"]}
    symmetry = {key: max(x[key] for x in phase_checks) for key in
                ["H_max_error", "spectrum_max_error", "matched_IPR_max_error", "mean_Dn_error"]}
    min_block_gap = float(min(x["min_parity_block_energy_gap"] for x in all_results))
    validation = {
        "scope": "independent static and frozen-frequency-zero calculations; no physical frequency calibration assumed",
        "passed": bool(max(maxima.values()) < 1e-10 and symmetry["mean_Dn_error"] < 1e-8
                       and symmetry["matched_IPR_max_error"] < 1e-7),
        "max_residuals": maxima, "phase_symmetry_maxima": symmetry,
        "minimum_same_parity_energy_gap": min_block_gap,
        "static_summary": [{"V":float(v), **_summarize_result(x)} for v,x in zip(V_VALUES,static_results)],
        "phase_checks": phase_checks,
        "near_degenerate_basis_note": "Cross-parity degeneracies fixed by exact Q convention; individual IPR is basis dependent in an unresolved same-parity degenerate subspace",
        "zero_floquet_outer_is_nan": bool(np.isnan(arrays["zero_outer_Dn"]).all()),
        "primary_Dn_can_exceed_one": True,
        "metadata": metadata,
    }
    (out/"baselines_validation.json").write_text(json.dumps(validation, indent=2)+"\n", encoding="utf-8")
    if not validation["passed"]:
        raise RuntimeError(f"Baseline validation failed: {maxima=}, {symmetry=}")
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path(__file__).parent/"results"/"frequency_phase_three_V")
    parser.add_argument("--no-modes", action="store_true")
    args = parser.parse_args()
    build_baselines(args.results, save_modes=not args.no_modes)
    print(f"Saved genuine static/frozen baselines in {args.results}", flush=True)


if __name__ == "__main__":
    main()
