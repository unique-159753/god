"""Time-ordered exact-Floquet numerics in fixed reflection/spin sectors.

Frequencies are dimensionless angular energies omega/t0, never Hertz.
Finite-step propagation is fourth-order FFT/onsite splitting, not an effective
Hamiltonian.  The theta=0 reflection/spin symmetry Q separates two sectors.
The exact C U* C=U identity (C=I tensor sigma_x) reconstructs the second
sector from the first without interpolation or a second physical model.
"""
from __future__ import annotations

from functools import lru_cache
import time
from typing import Any

import numpy as np
from scipy.linalg import schur

from model import (Parameters, _kinetic_rotation, fold_quasienergies,
                   localization_observables, operators)


def spin_swap(vectors: np.ndarray, L: int) -> np.ndarray:
    """Apply C=I tensor sigma_x to columns in the interleaved spin basis."""
    return vectors.reshape(L, 2, -1)[:, ::-1, :].reshape(2 * L, -1)


@lru_cache(maxsize=4)
def sector_bases(L: int = 89) -> tuple[np.ndarray, np.ndarray]:
    from frequency_baselines import reflection_spin_bases
    plus, _ = reflection_spin_bases(L)
    plus = np.asarray(plus, dtype=float)
    minus = spin_swap(plus, L)
    if plus.shape != (2 * L, L):
        raise ValueError("Q+ basis must have shape (2L,L)")
    if np.linalg.norm(plus.T @ minus) > 1e-10:
        raise ValueError("Q+ and C Q+ bases must be orthogonal")
    plus.setflags(write=False)
    minus.setflags(write=False)
    return plus, minus


@lru_cache(maxsize=16)
def symmetry_diagnostics(params: Parameters) -> dict[str, float]:
    plus, minus = sector_bases(params.L)
    q = plus @ plus.T - minus @ minus.T
    errors = {}
    for name in ["T", "S", "HZ"]:
        h = operators(params)[name]
        scale = max(np.linalg.norm(h), 1.)
        errors[f"Q_commutator_{name}"] = float(np.linalg.norm(h @ q - q @ h) / scale)
        errors[f"Hermiticity_{name}"] = float(np.linalg.norm(h-h.conj().T) / scale)
        transformed = spin_swap(spin_swap(h.T, params.L).T, params.L)
        errors[f"C_anticommutator_{name}"] = float(np.linalg.norm(transformed+h) / scale)
    return errors


def reconstruct_modes(sector_vectors: np.ndarray, reconstruction_order: np.ndarray,
                      L: int = 89) -> np.ndarray:
    """Losslessly reconstruct all 2L modes from stored L-by-L sector vectors.

    The Q+ basis is reproducible from ``frequency_baselines``.  Full columns
    before sorting are [Q+@sector_vectors, C conjugate(Q+@sector_vectors)].
    ``reconstruction_order`` is the saved permutation into quasienergy order.
    No information about the second-sector eigenvectors is discarded.
    """
    plus, _ = sector_bases(L)
    physical_plus = plus @ np.asarray(sector_vectors)
    all_modes = np.column_stack([physical_plus, spin_swap(physical_plus.conj(), L)])
    return all_modes[:, np.asarray(reconstruction_order, dtype=int)]


def reconstruct_operator(sector_operator: np.ndarray, L: int = 89) -> np.ndarray:
    """Build full U for independent checks; production need only store a sector."""
    plus, minus = sector_bases(L)
    return plus @ sector_operator @ plus.T + minus @ sector_operator.conj() @ minus.T


def _propagate_plus(omega: float, phi: float, params: Parameters, steps: int,
                    order: int = 4) -> np.ndarray:
    plus, _ = sector_bases(params.L)
    dt = 2 * np.pi / (omega * steps)
    if order == 4:
        weight1 = 1 / (2 - 2 ** (1 / 3))
        weights = (weight1, 1 - 2 * weight1, weight1)
    elif order == 6:
        a, b, c = .7845136104775573, .2355732133593581, -1.177679984178871
        weights = (a, b, c, 1-2*(a+b+c), c, b, a)
    else:
        raise ValueError("order must be 4 or 6")
    z = operators(params)["z"]
    phases = [np.exp(-.5j * dt * weight * z)[:, None] for weight in weights]
    state = np.array(plus, dtype=np.complex128).reshape(params.L, 2, params.L)
    for step in range(steps):
        start = step * dt
        for weight, phase in zip(weights, phases):
            duration = weight * dt
            midpoint = start + duration / 2
            state[:, 0, :] *= phase
            state[:, 1, :] *= phase.conj()
            state = _kinetic_rotation(
                state, duration,
                params.t0 + params.A * np.cos(omega * midpoint),
                params.delta0 + params.B * np.cos(omega * midpoint + phi))
            state[:, 0, :] *= phase
            state[:, 1, :] *= phase.conj()
            start += duration
    return state.reshape(2 * params.L, params.L)


def solve_point(omega: float, phi: float, params: Parameters, steps: int,
                return_operator: bool = True, order: int = 4) -> dict[str, Any]:
    """Solve one nonzero-frequency point, preserving all physical eigenmodes.

    Primary ``Dn``, ``D_component``, and ``D_component_logL`` use spin-resolved
    IPR and ln(L). ``D_component_log2L`` preserves the old ln(2L) control.
    The caller must perform time-step convergence; no finite step count is
    silently advertised as fully converged.  omega=0 is intentionally rejected.
    """
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("solve_point requires finite omega>0; handle omega=0 separately")
    if not np.isfinite(phi) or int(steps) != steps or steps < 1:
        raise ValueError("Require finite phi and positive integer steps")
    symmetry = symmetry_diagnostics(params)
    if max(symmetry.values()) > 1e-10:
        raise ValueError("Hamiltonian does not have the required Q/C symmetries")
    started = time.perf_counter()
    plus, minus = sector_bases(params.L)
    propagated = _propagate_plus(float(omega), float(phi), params, int(steps), order)
    propagation_done = time.perf_counter()
    sector_operator = plus.T @ propagated
    projection_error = float(np.linalg.norm(propagated-plus @ sector_operator) / np.sqrt(params.L))
    triangular, sector_vectors = schur(sector_operator, output="complex", check_finite=False)
    values = np.diag(triangular).copy()
    sector_eps = fold_quasienergies(-np.angle(values)*omega/(2*np.pi), omega)
    sector_order = np.argsort(sector_eps, kind="stable")
    sector_eps, sector_vectors, values = (sector_eps[sector_order],
                                         sector_vectors[:, sector_order], values[sector_order])
    unsorted_eps = np.r_[sector_eps, fold_quasienergies(-sector_eps, omega)]
    reconstruction_order = np.argsort(unsorted_eps, kind="stable")
    eps = unsorted_eps[reconstruction_order]
    modes = reconstruct_modes(sector_vectors, reconstruction_order, params.L)
    eigenvalues = np.r_[values, values.conj()][reconstruction_order]
    N = 2 * params.L
    physical_plus = plus @ sector_vectors
    residual_plus = propagated @ sector_vectors-physical_plus*values[None,:]
    gram_error = propagated.conj().T @ propagated-np.eye(params.L)
    cross_gram = propagated.conj().T @ spin_swap(propagated.conj(), params.L)
    phases = np.sort(np.mod(np.angle(eigenvalues), 2*np.pi))
    phase_gaps = np.diff(np.r_[phases, phases[0]+2*np.pi])
    observables = localization_observables(modes)
    result = {
        "omega": float(omega), "phi": float(phi), "steps": int(steps), "order":int(order),
        "quasienergies": eps, "modes": modes,
        "sector_vectors": sector_vectors, "sector_operator": sector_operator,
        "sector_quasienergies": sector_eps, "sector_eigenvalues": values,
        "reconstruction_order": reconstruction_order,
        "parity_labels": np.r_[np.ones(params.L, int), -np.ones(params.L, int)][reconstruction_order],
        "unitarity_error": float(np.sqrt((np.linalg.norm(gram_error)**2+np.linalg.norm(cross_gram)**2)/params.L)),
        "sector_unitarity_error": float(np.linalg.norm(sector_operator.conj().T @ sector_operator-np.eye(params.L))/np.sqrt(params.L)),
        "eigen_residual": float(np.linalg.norm(residual_plus)/np.sqrt(params.L)),
        "orthogonality_error": float(np.linalg.norm(sector_vectors.conj().T @ sector_vectors-np.eye(params.L))/np.sqrt(params.L)),
        "eigenvalue_modulus_error": float(np.max(abs(abs(eigenvalues)-1))),
        "sector_projection_error": projection_error,
        "schur_offdiagonal_error": float(np.linalg.norm(triangular-np.diag(np.diag(triangular)))/np.sqrt(params.L)),
        "min_eigenphase_gap": float(phase_gaps.min()),
        "degenerate_eigenphase_gaps": int(np.count_nonzero(phase_gaps<1e-9)),
        "hermiticity_error": max(value for key, value in symmetry.items()
                                  if key.startswith("Hermiticity_")),
        "Q_commutator_error": max(value for key, value in symmetry.items()
                                  if key.startswith("Q_commutator_")),
        "particle_hole_error": max(value for key, value in symmetry.items()
                                   if key.startswith("C_anticommutator_")),
        "sector_parity": np.ones(params.L, dtype=int),
        "propagation_seconds": propagation_done-started,
        "total_seconds": time.perf_counter()-started,
        "backend": f"Qplus_FFT_Yoshida{order}_C_conjugate_reconstruction",
        **observables,
    }
    result["D_component_log2L"] = result["D_component"]
    result["D_component"] = result["D_component_logL"]
    result["Dn"] = result["D_component_logL"]
    if return_operator:
        result["propagated_plus"] = propagated
    return result
