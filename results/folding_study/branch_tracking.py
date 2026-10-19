"""Symmetry-resolved overlap tracking of Floquet eigenstate branches.

This module contains no physical simulation and writes no result files.  The
columns of every supplied vector matrix are normalized eigenvectors in the
same physical basis; eigenvectors may be in any order.

Gauge convention for the accompanying study: anchor principal-zone q at
omega=8, track toward both lower and higher omega, lift the normalized
eigenphase with ``unwrap_q``, and plot omega*q_lift.  A different initial
integer adds k*omega to that displayed energy.  Neither the lift nor its
integer is a unique ordinary energy or a photon occupation.

Maximum-overlap assignments define overlap-continuous branches.  High
overlap alone cannot establish adiabatic continuation: a narrow avoided
crossing may be skipped and tracked diabatically.  Rank-swap, small-gap,
margin, and overlap flags identify intervals for refinement; they do not
establish a phase transition.  Midpoint-composition checks remain necessary.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


OVERLAP_THRESHOLD = 0.8
MARGIN_THRESHOLD = 0.1
NEAR_GAP_THRESHOLD = 1.0e-6


def _principal_q(q):
    """Return q in [-0.5, 0.5), including the +0.5 endpoint convention."""
    return np.remainder(np.asarray(q, dtype=float) + 0.5, 1.0) - 0.5


def _nearest_circular_gap(q):
    """Nearest distinct eigenphase distance within one symmetry sector."""
    if len(q) < 2:
        return np.full(len(q), np.inf)
    distance = np.abs(q[:, None] - q[None, :])
    distance = np.minimum(distance, 1.0 - distance)
    np.fill_diagonal(distance, np.inf)
    return np.min(distance, axis=1)


def unwrap_q(qprev_lift, qnew_matched):
    """Lift matched normalized eigenphases continuously from a chosen anchor.

    This is phase unwrapping in q, not nearest-energy matching while omega
    changes.  The integer lift is unique only when a phase increment is
    resolved to less than half a turn.  Exactly half a turn is ambiguous;
    NumPy's nearest-even tie rule supplies no physical resolution of it.
    """
    previous, current = np.broadcast_arrays(
        np.asarray(qprev_lift, dtype=float),
        np.asarray(qnew_matched, dtype=float),
    )
    if not (np.all(np.isfinite(previous)) and np.all(np.isfinite(current))):
        raise ValueError("Eigenphases must be finite.")
    return current + np.rint(previous - current)


def align(
    previous_vectors,
    current_vectors,
    parity_previous,
    parity_current,
    q_previous,
    q_current,
):
    """Assign current states to previous labels independently per parity.

    ``permutation[i]`` is the current column matched to previous column i.
    Every per-state return array uses previous-column order.  ``overlaps``
    contains squared absolute overlaps.  ``margins`` is the assigned value
    minus the strongest alternative in its row OR column; a constrained
    Hungarian assignment can consequently have a negative margin.

    ``rank_swap`` detects a change in within-sector eigenphase ordering
    after correcting its common cyclic shift at the principal-zone cut.
    That shift is inferred from matched cut crossings, assuming consecutive
    phase changes are below half a turn.  ``near_gap`` flags a same-parity
    circular gap below 1e-6 at either endpoint.  ``ambiguous`` is the union
    of overlap<0.8, margin<0.1, rank_swap, and near_gap; it is a warning for
    follow-up checks, not an error estimate or proof of physical mixing.
    """
    before = np.asarray(previous_vectors)
    after = np.asarray(current_vectors)
    if before.ndim != 2 or after.ndim != 2 or before.shape != after.shape:
        raise ValueError("Vector matrices must have the same 2D shape.")
    nstate = before.shape[1]
    if nstate == 0:
        raise ValueError("At least one state is required.")
    if not (np.all(np.isfinite(before)) and np.all(np.isfinite(after))):
        raise ValueError("Eigenvectors must be finite.")
    for vectors in (before, after):
        norm2 = np.sum(np.abs(vectors) ** 2, axis=0)
        if not np.allclose(norm2, 1.0, atol=1.0e-6, rtol=0.0):
            raise ValueError("Input eigenvector columns must be normalized.")

    pbefore = np.asarray(parity_previous)
    pafter = np.asarray(parity_current)
    qbefore = np.asarray(q_previous, dtype=float)
    qafter = np.asarray(q_current, dtype=float)
    for array in (pbefore, pafter, qbefore, qafter):
        if array.shape != (nstate,):
            raise ValueError("Parity and q arrays must have one entry per state.")
    if not (np.all(np.isfinite(qbefore)) and np.all(np.isfinite(qafter))):
        raise ValueError("Eigenphases must be finite.")
    qbefore = _principal_q(qbefore)
    qafter = _principal_q(qafter)
    if not np.array_equal(np.unique(pbefore), np.unique(pafter)):
        raise ValueError("The symmetry sectors must agree between endpoints.")

    permutation = np.empty(nstate, dtype=np.int64)
    overlaps = np.empty(nstate, dtype=float)
    margins = np.empty(nstate, dtype=float)
    rank_swap = np.zeros(nstate, dtype=bool)
    gap_before = np.full(nstate, np.inf)
    gap_after_matched = np.full(nstate, np.inf)
    zone_crossings = np.empty(nstate, dtype=np.int64)
    sector_diagnostics = []

    for parity in np.unique(pbefore):
        old_indices = np.flatnonzero(pbefore == parity)
        new_indices = np.flatnonzero(pafter == parity)
        count = len(old_indices)
        if count != len(new_indices):
            raise ValueError("Each symmetry sector must preserve its dimension.")
        overlap_matrix = np.abs(
            before[:, old_indices].conj().T @ after[:, new_indices]
        ) ** 2
        rows, columns = linear_sum_assignment(-overlap_matrix)
        # scipy returns rows in sorted order; build an explicit row map so
        # the contract does not depend on that implementation detail.
        col_for_row = np.empty(count, dtype=np.int64)
        col_for_row[rows] = columns
        assigned = overlap_matrix[np.arange(count), col_for_row]
        permutation[old_indices] = new_indices[col_for_row]
        overlaps[old_indices] = assigned

        if count == 1:
            margins[old_indices] = assigned
        else:
            alternatives = overlap_matrix.copy()
            alternatives[np.arange(count), col_for_row] = -np.inf
            strongest_alternative = np.maximum(
                np.max(alternatives, axis=1),
                np.max(alternatives, axis=0)[col_for_row],
            )
            margins[old_indices] = assigned - strongest_alternative

        old_q = qbefore[old_indices]
        new_q = qafter[new_indices]
        matched_q = new_q[col_for_row]
        cut_crossings = np.rint(old_q - matched_q).astype(np.int64)
        zone_crossings[old_indices] = cut_crossings
        cyclic_shift = int(np.sum(cut_crossings)) % count

        old_order = np.argsort(old_q, kind="stable")
        new_order = np.argsort(new_q, kind="stable")
        old_rank = np.empty(count, dtype=np.int64)
        new_rank = np.empty(count, dtype=np.int64)
        old_rank[old_order] = np.arange(count)
        new_rank[new_order] = np.arange(count)
        rank_offset = (new_rank[col_for_row] - old_rank) % count
        rank_swap[old_indices] = rank_offset != cyclic_shift
        gap_before[old_indices] = _nearest_circular_gap(old_q)
        gap_after_matched[old_indices] = _nearest_circular_gap(new_q)[col_for_row]
        modal_shift = int(np.argmax(np.bincount(rank_offset, minlength=count)))
        sector_diagnostics.append(
            {
                "parity": parity.item() if hasattr(parity, "item") else parity,
                "dimension": count,
                "cyclic_cut_shift": cyclic_shift,
                "modal_rank_shift": modal_shift,
                "cut_crossing_count": int(np.count_nonzero(cut_crossings)),
                "rank_swap_count": int(np.count_nonzero(rank_swap[old_indices])),
            }
        )

    near_gap_before = gap_before < NEAR_GAP_THRESHOLD
    near_gap_after = gap_after_matched < NEAR_GAP_THRESHOLD
    near_gap = near_gap_before | near_gap_after
    ambiguous = (
        (overlaps < OVERLAP_THRESHOLD)
        | (margins < MARGIN_THRESHOLD)
        | rank_swap
        | near_gap
    )
    matched_q = qafter[permutation]
    return {
        "permutation": permutation,
        "overlaps": overlaps,
        "margins": margins,
        "rank_swap": rank_swap,
        "near_gap": near_gap,
        "ambiguous": ambiguous,
        "near_gap_previous": near_gap_before,
        "near_gap_current": near_gap_after,
        "nearest_gap_previous": gap_before,
        "nearest_gap_current": gap_after_matched,
        "zone_crossings": zone_crossings,
        "assigned_q_step": unwrap_q(qbefore, matched_q) - qbefore,
        "sector_diagnostics": sector_diagnostics,
    }


def _synthetic_self_test():
    """Small algebraic checks only; these are not physical result data."""
    checks = []
    identity = np.eye(8, dtype=complex)
    parity = np.repeat([-1, 1], 4)
    q = np.tile([-0.4, -0.1, 0.15, 0.4], 2)

    reorder = np.array([6, 0, 4, 2, 7, 1, 5, 3])
    phase = np.exp(1j * np.arange(8) * 0.37)
    result = align(
        identity, identity[:, reorder] * phase,
        parity, parity[reorder], q, q[reorder],
    )
    assert np.array_equal(result["permutation"], np.argsort(reorder))
    assert np.allclose(result["overlaps"], 1.0)
    assert not np.any(result["ambiguous"])
    checks.append("permutation_and_arbitrary_vector_phases")

    q_old = np.array([-0.3, -0.1, 0.1, 0.49])
    q_new = np.array([-0.29, -0.09, 0.11, -0.49])
    result = align(np.eye(4), np.eye(4), np.ones(4), np.ones(4), q_old, q_new)
    assert not np.any(result["rank_swap"])
    assert result["sector_diagnostics"][0]["cyclic_cut_shift"] == 1
    assert np.allclose(unwrap_q(q_old, q_new), [-0.29, -0.09, 0.11, 0.51])
    checks.append("zone_cut_wrap_is_not_rank_swap")

    q_new = np.array([-0.3, 0.1, -0.1, 0.4])
    result = align(
        np.eye(4), np.eye(4), np.ones(4), np.ones(4),
        np.array([-0.3, -0.1, 0.1, 0.4]), q_new,
    )
    assert np.array_equal(result["rank_swap"], [False, True, True, False])
    assert np.allclose(result["overlaps"], 1.0)
    checks.append("high_overlap_diabatic_rank_swap_is_flagged")

    vectors = np.eye(4)
    angle = np.pi / 4
    vectors[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]]
    result = align(
        np.eye(4), vectors, np.ones(4), np.ones(4),
        [-0.3, -0.1, 0.1, 0.4], [-0.3, -0.1, 0.1, 0.4],
    )
    assert np.all(result["ambiguous"][:2])
    assert np.allclose(result["overlaps"][:2], 0.5)
    assert np.allclose(result["margins"][:2], 0.0)
    checks.append("mixed_subspace_has_low_overlap_and_margin")

    close_q = np.array([-0.3, -0.1, -0.1 + 1e-8, 0.4])
    result = align(np.eye(4), np.eye(4), np.ones(4), np.ones(4), close_q, close_q)
    assert np.array_equal(result["near_gap"], [False, True, True, False])
    checks.append("same_sector_near_degeneracy")

    result = align(
        np.eye(2), np.eye(2), [-1, 1], [-1, 1], [0.1, 0.1], [0.1, 0.1],
    )
    assert not np.any(result["near_gap"])
    assert not np.any(result["ambiguous"])
    checks.append("opposite_sector_degeneracy_is_resolved")
    return {"kind": "synthetic_utility_test_not_physical_data", "passed": checks}


if __name__ == "__main__":
    import json
    print(json.dumps(_synthetic_self_test(), indent=2))
