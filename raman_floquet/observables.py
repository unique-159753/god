"""Explicit finite-size localization and common Floquet-zone selection rules."""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from model import DEFAULT_PARAMS, static_eigensystem


OUTER_FRACTION = 0.2


def outer_indices(quasienergies, fraction=OUTER_FRACTION):
    values = np.asarray(quasienergies)
    count = math.ceil(fraction * values.size)
    return np.argsort(np.abs(values), kind="stable")[-count:]


def static_outer_indices(params=DEFAULT_PARAMS):
    energies, _ = static_eigensystem(params)
    return outer_indices(energies)


def extract_stats(result, params=DEFAULT_PARAMS):
    """Use the folded spectrum, then select its highest-|epsilon| 20%."""
    selected = outer_indices(result["quasienergies"])
    absolute = np.sort(np.abs(result["quasienergies"]))
    count = selected.size
    stats = {"outer_indices": selected,
             "outer_count": int(count),
             "outer_selection_gap": float(absolute[-count] - absolute[-count - 1])}
    for key in ["D_site", "D_component", "D_component_logL", "ipr_site", "ipr_component"]:
        values = np.asarray(result[key])
        stats[f"outer_{key}"] = float(np.mean(values[selected]))
        stats[f"all_{key}"] = float(np.mean(values))
    if "modes" in result:
        _, reference = static_eigensystem(params)
        weights = np.abs(reference.conj().T @ result["modes"]) ** 2
        static_outer = static_outer_indices(params)
        projected = np.sum(weights[static_outer], axis=0)
        stats["static_outer_weights"] = projected
        stats["static_outer_weight_sum_error"] = float(abs(projected.sum() - count))
        stats["dominant_static_state"] = np.argmax(weights, axis=0)
        stats["dominant_static_overlap"] = np.max(weights, axis=0)
        for key in ["D_site", "D_component", "ipr_site", "ipr_component"]:
            stats[f"projected_{key}"] = float(np.dot(projected, result[key]) / count)
    return stats


def summarize_pair(exact, effective, params=DEFAULT_PARAMS):
    """Separate paired wavefunction changes from independent hard selection.

    For exact outer states alpha, match beta(alpha) in the lab-frame effective
    eigenbasis by maximum total squared overlap. Then
    raw_delta = matched_delta + selection_delta exactly (up to rounding).
    The projector-weighted comparison does not require one-to-one labels.
    """
    x, y = extract_stats(exact, params), extract_stats(effective, params)
    overlaps = np.abs(exact["modes"].conj().T @ effective["modes"]) ** 2
    rows, columns = linear_sum_assignment(-overlaps)
    permutation = np.empty(len(rows), dtype=np.int16)
    permutation[rows] = columns
    matched_overlaps = overlaps[np.arange(len(rows)), permutation]
    selected = x["outer_indices"]
    output = {"exact": x, "effective": y,
              "exact_to_effective": permutation,
              "matched_overlaps": matched_overlaps,
              "mean_outer_matched_overlap": float(np.mean(matched_overlaps[selected])),
              "min_outer_matched_overlap": float(np.min(matched_overlaps[selected]))}
    for key in ["D_site", "D_component", "D_component_logL"]:
        exact_mean = float(np.mean(exact[key][selected]))
        matched_effective_mean = float(np.mean(effective[key][permutation[selected]]))
        effective_mean = y[f"outer_{key}"]
        raw = exact_mean - effective_mean
        matched = exact_mean - matched_effective_mean
        selection = matched_effective_mean - effective_mean
        output[f"delta_{key}"] = raw
        output[f"matched_delta_{key}"] = matched
        output[f"selection_delta_{key}"] = selection
        output[f"decomposition_error_{key}"] = float(abs(raw - matched - selection))
    for key in ["D_site", "D_component"]:
        output[f"projected_delta_{key}"] = x[f"projected_{key}"] - y[f"projected_{key}"]
    return output


def serializable_stats(stats):
    return {key: value for key, value in stats.items()
            if isinstance(value, (float, int, np.floating, np.integer))}
