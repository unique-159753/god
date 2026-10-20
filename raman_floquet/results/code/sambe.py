"""Reference-matched Floquet-Sambe checks, with explicit replica accounting.

Fourier convention: H(t)=sum_l H_l exp(+il omega t),
u(t)=sum_m u_m exp(+im omega t), HF_mn=H_(m-n)+m omega delta_mn.
Every physical-mode label below comes from an exact Floquet reference, never
from treating all folded Sambe copies as independent physical bands.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

from model import (DEFAULT_PARAMS, Parameters, exact_modes, fold_quasienergies,
                   fourier_components, localization_observables, static_eigensystem)


def sambe_hamiltonian(omega: float, phi: float, mcut: int,
                     params: Parameters = DEFAULT_PARAMS):
    if omega <= 0 or int(mcut) != mcut or mcut < 1:
        raise ValueError("Require omega>0 and positive integer mcut")
    fourier = fourier_components(phi, params)
    sectors = np.arange(-mcut, mcut + 1)
    n = 2 * params.L
    h = np.zeros((n * len(sectors), n * len(sectors)), dtype=complex)
    for i, m in enumerate(sectors):
        for j, photon in enumerate(sectors):
            if m - photon in fourier:
                h[i*n:(i+1)*n, j*n:(j+1)*n] = fourier[m - photon]
        block = h[i*n:(i+1)*n, i*n:(i+1)*n]
        block.flat[::n + 1] += m * omega
    return h, sectors


def wrapped_distance(a, b, omega):
    return abs(fold_quasienergies(np.asarray(a) - np.asarray(b), omega))


def solve_sambe(omega: float, phi: float, mcut: int,
                exact: dict | None = None, params: Parameters = DEFAULT_PARAMS,
                steps: int = 128) -> dict:
    """Return one Sambe representative matched to each exact physical mode.

    In-zone raw eigenvalues have priority. If a truncation gives fewer than N
    in-zone candidates, missing references are matched to explicitly flagged
    out-of-zone copies; their photon gauge is aligned before weight comparison.
    Low-confidence matches and central-zone count deficits remain visible.
    """
    started = time.perf_counter()
    if exact is None:
        exact = exact_modes(omega, phi, steps=steps, params=params,
                            return_modes=True, return_operator=True)
    h, sectors = sambe_hamiltonian(omega, phi, mcut, params)
    hermiticity = float(np.linalg.norm(h - h.conj().T))
    if hermiticity > 1e-12:
        raise RuntimeError("Sambe Hamiltonian is not Hermitian")
    values, vectors = eigh(h, check_finite=False, driver="evd")
    n = 2 * params.L
    blocks = vectors.reshape(len(sectors), n, len(values))
    raw_reconstructed = np.sum(blocks, axis=0)
    norms = np.linalg.norm(raw_reconstructed, axis=0)
    reconstructed = raw_reconstructed / np.maximum(norms, 1e-15)[None, :]
    exact_vectors = exact["modes"]
    exact_epsilon = exact["quasienergies"]
    overlaps = abs(exact_vectors.conj().T @ reconstructed) ** 2
    folded = fold_quasienergies(values, omega)
    distances = wrapped_distance(exact_epsilon[:, None], folded[None, :], omega)
    raw_weights = np.sum(abs(blocks) ** 2, axis=1)
    edge_weights = raw_weights[0] + raw_weights[-1]
    # Overlap dominates; the energy term excludes unrelated nearby replicas.
    # This is a matching cost, not a physical probability or new observable.
    cost = (1 - overlaps + .05 * (distances / .05) ** 2
            + .002 * edge_weights[None, :])
    central = np.flatnonzero((values >= -omega / 2) & (values < omega / 2))
    chosen = np.full(n, -1, dtype=int)
    if len(central):
        rows, cols = linear_sum_assignment(cost[:, central])
        chosen[rows] = central[cols]
    missing = np.flatnonzero(chosen < 0)
    if len(missing):
        outside = np.flatnonzero(~((values >= -omega / 2) & (values < omega / 2)))
        shifts = np.rint((values[outside] - folded[outside]) / omega).astype(int)
        fallback_cost = cost[np.ix_(missing, outside)] + .01 * abs(shifts)[None, :]
        rows, cols = linear_sum_assignment(fallback_cost)
        chosen[missing[rows]] = outside[cols]
    if np.any(chosen < 0) or len(np.unique(chosen)) != n:
        raise RuntimeError("Reference matching did not cover each exact state once")

    selected_vectors = reconstructed[:, chosen]
    selected_blocks = blocks[:, :, chosen]
    selected_epsilon = folded[chosen]
    shifts = np.rint((values[chosen] - selected_epsilon) / omega).astype(int)
    axis_extent = mcut + int(np.max(abs(shifts)))
    photon_axis = np.arange(-axis_extent, axis_extent + 1)
    aligned = np.zeros((n, len(photon_axis)))
    E0, V0 = static_eigensystem(params)
    static_coeff = np.einsum("ab,qbc->qac", V0.conj().T, selected_blocks, optimize=True)
    aligned_static = np.zeros((n, len(photon_axis), n))
    for alpha in range(n):
        for row, raw_m in enumerate(sectors):
            aligned_m = raw_m - shifts[alpha]
            location = int(aligned_m + axis_extent)
            aligned[alpha, location] = raw_weights[row, chosen[alpha]]
            aligned_static[alpha, location] = abs(static_coeff[row, :, alpha]) ** 2
    statistics = localization_observables(selected_vectors)
    exact_statistics = localization_observables(exact_vectors)
    selected_overlaps = overlaps[np.arange(n), chosen]
    selected_distances = distances[np.arange(n), chosen]
    selected_central = (values[chosen] >= -omega / 2) & (values[chosen] < omega / 2)
    outer_count = int(np.ceil(.2 * n))
    outer = np.argsort(abs(exact_epsilon), kind="stable")[-outer_count:]
    entropy = -np.sum(aligned * np.log(np.maximum(aligned, 1e-300)), axis=1)
    participation = 1 / np.sum(aligned ** 2, axis=1)
    gram_error = float(np.linalg.norm(selected_vectors.conj().T @ selected_vectors - np.eye(n)) / np.sqrt(n))
    U = exact.get("operator")
    if U is not None:
        eigenvalues = np.exp(-2j * np.pi * selected_epsilon / omega)
        residual = float(np.linalg.norm(U @ selected_vectors - selected_vectors * eigenvalues[None, :]) / np.sqrt(n))
    else:
        residual = float("nan")
    summary = {
        "omega": float(omega), "phi": float(phi), "mcut": int(mcut),
        "exact_steps": int(steps), "matrix_dimension": int(len(values)),
        "exact_reference_mode_count": n,
        "raw_central_zone_candidate_count": int(len(central)),
        "selected_central_zone_representatives": int(np.sum(selected_central)),
        "outside_zone_gauge_fallback_count": int(np.sum(~selected_central)),
        "high_confidence_matches": int(np.count_nonzero((selected_overlaps >= .8) & (selected_distances <= .05))),
        "minimum_mode_overlap": float(np.min(selected_overlaps)),
        "mean_mode_overlap": float(np.mean(selected_overlaps)),
        "outer_minimum_overlap": float(np.min(selected_overlaps[outer])),
        "outer_mean_overlap": float(np.mean(selected_overlaps[outer])),
        "max_quasienergy_error": float(np.max(selected_distances)),
        "outer_max_quasienergy_error": float(np.max(selected_distances[outer])),
        "outer_mean_D_site": float(np.mean(statistics["D_site"][outer])),
        "exact_outer_mean_D_site": float(np.mean(exact_statistics["D_site"][outer])),
        "outer_mean_D_difference": float(np.mean(statistics["D_site"][outer] - exact_statistics["D_site"][outer])),
        "mean_raw_edge_weight": float(np.mean(edge_weights[chosen])),
        "outer_mean_raw_edge_weight": float(np.mean(edge_weights[chosen][outer])),
        "mean_photon_participation": float(np.mean(participation)),
        "outer_mean_photon_participation": float(np.mean(participation[outer])),
        "outer_mean_photon_entropy": float(np.mean(entropy[outer])),
        "reconstructed_orthogonality_error": gram_error,
        "max_reconstructed_norm_error": float(np.max(abs(norms[chosen] - 1))),
        "exact_operator_mode_residual": residual,
        "hermiticity_error": hermiticity,
        "total_seconds": time.perf_counter() - started,
        "replica_policy": "In-zone raw eigenvalues first; missing exact references get explicitly flagged gauge-aligned outside-zone candidates",
    }
    return {"summary": summary, "omega": float(omega), "phi": float(phi), "mcut": int(mcut),
            "raw_sambe_eigenvalues": values, "candidate_indices": chosen,
            "selected_raw_eigenvalues": values[chosen], "quasienergies": selected_epsilon,
            "exact_quasienergies": exact_epsilon, "mode_overlap": selected_overlaps,
            "quasienergy_error": selected_distances, "raw_gauge_shift": shifts,
            "original_in_zone": selected_central, "photon_axis": photon_axis,
            "photon_weights": aligned, "static_photon_weights": aligned_static,
            "static_energies": E0, "photon_participation": participation,
            "photon_entropy": entropy, "raw_edge_weight": edge_weights[chosen],
            "reconstructed_norm": norms[chosen], "modes": selected_vectors,
            "outer_indices_exact_reference": outer,
            "exact_D_site": exact_statistics["D_site"], **statistics}


def run_representative_points(output: str | Path = "results", steps: int = 128,
                              mcuts=(1, 2, 3)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = [(6.2, np.pi/3), (6.2, 0.), (6.2, np.pi/2), (8., np.pi/2)]
    summaries = []
    for i, (omega, phi) in enumerate(settings):
        exact = exact_modes(omega, phi, steps=steps, return_modes=True, return_operator=True)
        for mcut in mcuts:
            result = solve_sambe(omega, phi, mcut, exact=exact, steps=steps)
            summary = result.pop("summary")
            np.savez_compressed(output / f"sambe_point{i}_M{mcut}.npz",
                                summary=json.dumps(summary), **result)
            summary["file"] = f"sambe_point{i}_M{mcut}.npz"
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    with (output / "sambe_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    payload = {"points": summaries, "exact_steps": steps,
               "mcuts": list(mcuts), "outer_rule": "36 highest |exact centered quasienergies|; same exact reference indices for all Sambe comparisons",
               "caution": "Noncentral photon weight is gauge dependent; a folded static state can already occupy photon ±1. Use participation/entropy and static-photon mixing, not sideband weight alone, as diagnostics."}
    (output / "sambe_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--mcuts", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    run_representative_points(args.output, args.steps, tuple(args.mcuts))
