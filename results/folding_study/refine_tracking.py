"""Refine suspicious overlap-tracked Floquet branch intervals.

All physical eigenvectors are recomputed by exact midpoint evolution with
M=128.  The existing tracked curves remain unchanged.  Refinement tests
discretization dependence of Hungarian assignments; consistent assignments
alone do not establish adiabatic continuation or a phase transition.
"""

from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from branch_tracking import align
from floquet_core import calculate


ROOT = Path(__file__).resolve().parent
PHASES = (("0", 0.0), ("pi4", np.pi / 4), ("pi2", np.pi / 2))


def select_intervals(curve, count=3):
    """Select high-suspicion non-touching intervals at omega <= 8.

    The supplied curve's parent indices point toward its omega=8 anchor.
    Existing ambiguity, low-overlap, and rank-swap counts are used only to
    prioritize checks; no original branch labels are changed.
    """
    omega = curve["omega"]
    candidates = []
    for target, parent in enumerate(curve["parent_index"]):
        parent = int(parent)
        if parent < 0 or max(omega[target], omega[parent]) > 8.0 + 1e-12:
            continue
        low = int(np.count_nonzero(curve["assigned_overlap"][target] < 0.8))
        ambiguous = int(np.count_nonzero(curve["ambiguous"][target]))
        rank = int(np.count_nonzero(curve["rank_swap"][target]))
        minimum_overlap = float(np.min(curve["assigned_overlap"][target]))
        candidates.append(
            {
                "parent_index": parent,
                "target_index": int(target),
                "omega_start": float(omega[parent]),
                "omega_end": float(omega[target]),
                "source_ambiguous_count": ambiguous,
                "source_low_overlap_count": low,
                "source_rank_swap_count": rank,
                "source_min_overlap": minimum_overlap,
                "suspicion_score": ambiguous + 2 * low + 3 * rank,
            }
        )
    candidates.sort(
        key=lambda item: (
            -item["suspicion_score"],
            item["source_min_overlap"],
            item["omega_end"],
        )
    )
    chosen = []
    used_endpoints = set()
    for candidate in candidates:
        endpoints = {candidate["parent_index"], candidate["target_index"]}
        if endpoints & used_endpoints:
            continue
        candidate["interval_id"] = len(chosen)
        chosen.append(candidate)
        used_endpoints.update(endpoints)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError(f"Requested {count} disjoint intervals, found {len(chosen)}.")
    return chosen


def _circular_gap(q, parity):
    smallest = np.inf
    for label in np.unique(parity):
        values = np.sort(q[parity == label])
        if len(values) < 2:
            continue
        gaps = np.diff(np.r_[values, values[0] + 1.0])
        smallest = min(smallest, float(np.min(gaps)))
    return float(smallest)


def _phase_refinement(job):
    tag, phi, count, max_level = job
    start_wall = time.perf_counter()
    path = ROOT / f"curves_phi_{tag}.npz"
    with np.load(path, allow_pickle=False) as curve:
        intervals = select_intervals(curve, count)
    cache = {}
    compute_seconds = 0.0

    def point(omega):
        nonlocal compute_seconds
        key = round(float(omega), 14)
        if key not in cache:
            result = calculate(89, 55, key, phi, 128, "exact", return_matrices=True)
            cache[key] = {
                "vectors": result["eigenvectors"],
                "q": result["quasienergy"] / key,
                "parity": result["parity"],
                "min_gap_energy": result["min_within_sector_quasienergy_gap"],
                "min_gap_q_circular": _circular_gap(
                    result["quasienergy"] / key, result["parity"]
                ),
                "eigen_residual": result["eigen_residual_relative_fro"],
                "unitarity_residual": result["unitarity_relative_fro"],
            }
            compute_seconds += result["seconds"]
        return cache[key]

    def link(old, new):
        return align(
            old["vectors"], new["vectors"],
            old["parity"], new["parity"], old["q"], new["q"],
        )

    rows = []
    summaries = []
    for interval in intervals:
        old = point(interval["omega_start"])
        new = point(interval["omega_end"])
        direct = link(old, new)
        direct_permutation = direct["permutation"]
        last_permutation = None
        interval_rows = []
        for level in range(max_level + 1):
            level_start = time.perf_counter()
            steps = 2 ** level
            frequencies = np.linspace(
                interval["omega_start"], interval["omega_end"], steps + 1
            )
            values = [point(omega) for omega in frequencies]
            permutation = np.arange(len(direct_permutation))
            overlap_sum = 0.0
            minimum_overlap = 1.0
            low_overlap_count = 0
            small_margin_count = 0
            rank_swap_count = 0
            ambiguous_count = 0
            near_gap_count = 0
            ambiguous_origin = np.zeros(len(permutation), dtype=bool)
            rank_swap_origin = np.zeros(len(permutation), dtype=bool)
            for before, after in zip(values[:-1], values[1:]):
                result = link(before, after)
                # Result flags have the current link's previous raw-column
                # order; pull them back before composing the permutation.
                ambiguous_origin |= result["ambiguous"][permutation]
                rank_swap_origin |= result["rank_swap"][permutation]
                permutation = result["permutation"][permutation]
                overlap_sum += float(np.sum(result["overlaps"]))
                minimum_overlap = min(minimum_overlap, float(np.min(result["overlaps"])))
                low_overlap_count += int(np.count_nonzero(result["overlaps"] < 0.8))
                small_margin_count += int(np.count_nonzero(result["margins"] < 0.1))
                rank_swap_count += int(np.count_nonzero(result["rank_swap"]))
                ambiguous_count += int(np.count_nonzero(result["ambiguous"]))
                near_gap_count += int(np.count_nonzero(result["near_gap"]))

            changed = np.flatnonzero(permutation != direct_permutation)
            previous_mismatches = (
                None if last_permutation is None
                else int(np.count_nonzero(permutation != last_permutation))
            )
            # For a changed group, the endpoint subspace can remain stable
            # despite sensitivity of individual assignment labels.  This
            # check is supplementary and does not relabel any curve.
            if len(changed):
                singular = np.linalg.svd(
                    old["vectors"][:, changed].conj().T
                    @ new["vectors"][:, permutation[changed]],
                    compute_uv=False,
                )
                min_subspace_overlap = float(np.min(singular ** 2))
                mean_subspace_overlap = float(np.mean(singular ** 2))
            else:
                min_subspace_overlap = None
                mean_subspace_overlap = None
            row = {
                "phase_tag": tag,
                "phi": float(phi),
                **interval,
                "slices": 128,
                "level": level,
                "steps": steps,
                "absolute_delta_omega": float(abs(frequencies[1] - frequencies[0])),
                "mismatch_vs_direct": int(len(changed)),
                "mismatch_vs_previous_level": previous_mismatches,
                "direct_high_overlap_mismatches": int(np.count_nonzero(
                    direct["overlaps"][changed] >= 0.8
                )),
                "min_assigned_overlap": minimum_overlap,
                "mean_assigned_overlap": overlap_sum / (steps * len(permutation)),
                "low_overlap_count": low_overlap_count,
                "small_margin_count": small_margin_count,
                "rank_swap_count": rank_swap_count,
                "near_gap_count": near_gap_count,
                "ambiguous_count": ambiguous_count,
                "ambiguous_origin_state_count": int(np.count_nonzero(ambiguous_origin)),
                "rank_swap_origin_state_count": int(np.count_nonzero(rank_swap_origin)),
                "min_within_sector_quasienergy_gap": float(min(
                    value["min_gap_energy"] for value in values
                )),
                "min_within_sector_circular_q_gap": float(min(
                    value["min_gap_q_circular"] for value in values
                )),
                "max_eigen_residual": float(max(
                    value["eigen_residual"] for value in values
                )),
                "max_unitarity_residual": float(max(
                    value["unitarity_residual"] for value in values
                )),
                "changed_subspace_min_overlap_sv2": min_subspace_overlap,
                "changed_subspace_mean_overlap_sv2": mean_subspace_overlap,
                "level_wall_seconds": time.perf_counter() - level_start,
            }
            rows.append(row)
            interval_rows.append(row)
            last_permutation = permutation.copy()
        final = interval_rows[-1]
        final_stable = final["mismatch_vs_previous_level"] == 0
        flags_remain = final["ambiguous_count"] > 0
        summaries.append({
            **interval,
            "phi": float(phi),
            "phase_tag": tag,
            "assignment_grid_stable_last_two_levels": final_stable,
            "coarse_assignment_changed": final["mismatch_vs_direct"] > 0,
            "final_mismatch_vs_direct": final["mismatch_vs_direct"],
            "final_mismatch_vs_previous_level": final["mismatch_vs_previous_level"],
            "final_direct_high_overlap_mismatches": final["direct_high_overlap_mismatches"],
            "final_ambiguity_remains": flags_remain,
            "unresolved": bool(not final_stable or flags_remain),
            "final_min_assigned_overlap": final["min_assigned_overlap"],
            "final_mean_assigned_overlap": final["mean_assigned_overlap"],
            "final_ambiguous_origin_state_count": final["ambiguous_origin_state_count"],
            "final_rank_swap_origin_state_count": final["rank_swap_origin_state_count"],
            "changed_subspace_min_overlap_sv2": final["changed_subspace_min_overlap_sv2"],
            "interpretation": (
                "This is an assignment discretization test. Even agreement and high "
                "overlap do not prove adiabatic continuation; original curves are unchanged."
            ),
        })
        print(
            f"refinement phi={tag} interval={interval['interval_id']} "
            f"{interval['omega_start']:.6f}->{interval['omega_end']:.6f}: "
            f"final direct mismatches={final['mismatch_vs_direct']}, "
            f"last-level mismatches={final['mismatch_vs_previous_level']}, "
            f"ambiguities={final['ambiguous_count']}",
            flush=True,
        )
    return {
        "tag": tag,
        "phi": float(phi),
        "source_curve": path.name,
        "source_curve_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "unique_exact128_points": len(cache),
        "sum_point_compute_seconds": compute_seconds,
        "phase_wall_seconds": time.perf_counter() - start_wall,
        "intervals": summaries,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals", type=int, default=3)
    parser.add_argument("--max-level", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        parser.error("--workers must be between 1 and 3.")
    if args.intervals < 1 or args.max_level < 1:
        parser.error("--intervals and --max-level must be positive.")
    for tag, _ in PHASES:
        if not (ROOT / f"curves_phi_{tag}.npz").exists():
            parser.error(f"Missing completed curve: curves_phi_{tag}.npz")
    start_wall = time.perf_counter()
    jobs = [(tag, phi, args.intervals, args.max_level) for tag, phi in PHASES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_phase_refinement, jobs))
    rows = [row for result in results for row in result.pop("rows")]
    csv_path = ROOT / "refinement_intervals.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    all_intervals = [interval for phase in results for interval in phase["intervals"]]
    summary = {
        "purpose": "Hungarian-assignment discretization and ambiguity checks",
        "physical_method": "exact midpoint Floquet time evolution; M=128; L=89; alpha=55/89",
        "levels": list(range(args.max_level + 1)),
        "selection_rule": (
            "Three non-touching omega<=8 intervals per phase, ranked by "
            "ambiguous_count + 2*low_overlap_count + 3*rank_swap_count; "
            "ties resolved by lower minimum overlap."
        ),
        "original_curve_files_modified": False,
        "adiabatic_continuation_claimed": False,
        "total_wall_seconds": time.perf_counter() - start_wall,
        "unique_exact128_points": sum(result["unique_exact128_points"] for result in results),
        "assignment_grid_stable_interval_count": sum(
            item["assignment_grid_stable_last_two_levels"] for item in all_intervals
        ),
        "coarse_assignment_changed_interval_count": sum(
            item["coarse_assignment_changed"] for item in all_intervals
        ),
        "unresolved_interval_count": sum(item["unresolved"] for item in all_intervals),
        "phases": results,
        "data_csv": csv_path.name,
    }
    (ROOT / "refinement_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "phases"}, indent=2))


if __name__ == "__main__":
    main()
