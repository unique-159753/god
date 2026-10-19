"""Fresh convergence and high-frequency checks for folding_study.

Run with one process and one BLAS thread. All reference Floquet operators are
products of instantaneous Hamiltonian exponentials; no effective model is used
to produce reference data. Output paths are relative to this source file.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import csv
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import schur
from floquet_core import calculate, operators, row
from analysis_core import (effective, match_by_overlap, outer_indices,
                           range_info, vector_metrics, DIM, METRIC_NAMES)

ROOT = Path(__file__).resolve().parent
PHASES = (0., np.pi/4, np.pi/2)
FREQUENCIES = (6., 6.3, 6.5, 7., 8., 10.)


def write_csv(name, rows):
    with (ROOT/name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def circular_difference(a, b):
    return (a-b+.5) % 1-.5


def extra_checks(result):
    """Check the actual returned vectors against the actual propagated U."""
    _, qs, _, _ = operators(89, 55)
    V = result['eigenvectors']
    residuals = np.empty(DIM)
    for parity in (-1, 1):
        columns = np.flatnonzero(result['parity']==parity)
        v = np.asarray(qs[parity].T@V[:, columns])
        residual = (result['sector_U'][parity]@v
                    -v*result['eigenvalues'][columns])
        residuals[columns] = np.linalg.norm(residual, axis=0)
    q = result['quasienergy']/result['omega']
    idx = result['outer_indices']
    ipr_again, D_again, norm_error = vector_metrics(V)
    assert np.max(abs(ipr_again-result['ipr'])) < 1e-13
    checks = dict(
        returned_vector_norm_max_error=norm_error,
        returned_vector_orthogonality_relative_fro=float(
            np.linalg.norm(V.conj().T@V-np.eye(DIM), 'fro')/np.sqrt(DIM)),
        actual_eigen_residual_max=float(residuals.max()),
        actual_eigen_residual_relative_fro=float(np.linalg.norm(residuals)/np.sqrt(DIM)),
        quasienergy_mean=float(result['quasienergy'].mean()),
        quasienergy_min=float(result['quasienergy'].min()),
        quasienergy_max=float(result['quasienergy'].max()),
        **range_info(q, q, idx))
    assert checks['actual_eigen_residual_max'] < 1e-10
    assert checks['returned_vector_norm_max_error'] < 1e-11
    return checks


def same_sector_circular_gap(q, parity):
    gaps = []
    near_pairs = 0
    for p in (-1, 1):
        qp = np.sort(q[parity==p])
        gap = np.diff(np.r_[qp, qp[0]+1])
        gaps.extend(gap.tolist())
        near_pairs += int(np.count_nonzero(gap < 1e-10))
    return float(min(gaps)), near_pairs


def schur_basis_recheck(result):
    """Re-Schur after an independent similarity permutation within each sector.

    This checks basis sensitivity to a numerically equivalent eigensolve. It is
    not a proof that every nearly degenerate subspace has a unique physical IPR.
    Symmetry sectors remain separate in both eigensolves.
    """
    _, qs, _, _ = operators(89, 55)
    all_q, all_v, all_p = [], [], []
    for p in (-1, 1):
        U = result['sector_U'][p]
        order = np.arange(len(U))[::-1]
        Us = U[np.ix_(order, order)]
        tt, vv = schur(Us, output='complex', check_finite=False)
        restored = np.empty_like(vv)
        restored[order] = vv
        all_v.append(np.asarray(qs[p]@restored))
        all_q.append(-np.angle(np.diag(tt))/(2*np.pi))
        all_p.append(np.full(len(U), p))
    v = np.concatenate(all_v, axis=1)
    q = np.concatenate(all_q)
    parity = np.concatenate(all_p)
    ipr, D, _ = vector_metrics(v)
    assignment, overlap, _ = match_by_overlap(
        result['eigenvectors'], v, result['parity'], parity)
    idx = result['outer_indices']
    return dict(
        repeated_schur_min_assigned_overlap=float(overlap.min()),
        repeated_schur_mean_assigned_overlap=float(overlap.mean()),
        repeated_schur_D_max_difference=float(np.max(abs(result['D']-D[assignment]))),
        repeated_schur_IPR_max_difference=float(np.max(abs(result['ipr']-ipr[assignment]))),
        repeated_schur_D_outer_matched_difference=float(
            np.mean(result['D'][idx]-D[assignment[idx]])),
        repeated_schur_q_max_circular_difference=float(np.max(abs(
            circular_difference(result['quasienergy']/result['omega'], q[assignment])))),
    )


def run():
    started = time.perf_counter()
    convergence, state_rows, high_frequency = [], [], []
    exact128 = {}
    for omega in FREQUENCIES:
        for phi in PHASES:
            runs = {M: calculate(89, 55, omega, phi, M, return_matrices=True)
                    for M in (32, 64, 128)}
            reference = runs[128]
            exact128[(omega, phi)] = reference
            qref = reference['quasienergy']/omega
            reference_outer_set = set(reference['outer_indices'].tolist())
            for M, result in runs.items():
                q = result['quasienergy']/omega
                gap, near_pairs = same_sector_circular_gap(q, result['parity'])
                assignment, overlap, participation = match_by_overlap(
                    result['eigenvectors'], reference['eigenvectors'],
                    result['parity'], reference['parity'])
                matched_outer_set = set(assignment[result['outer_indices']].tolist())
                record = {**row(result), **extra_checks(result),
                          'phi_over_pi': float(phi/np.pi),
                          'min_within_sector_circular_q_gap': gap,
                          'within_sector_gap_below_1e_10_count': near_pairs,
                          'q_vs128_matched_max_error': float(np.max(abs(
                              circular_difference(q, qref[assignment])))),
                          'IPR_vs128_matched_max_error': float(np.max(abs(
                              result['ipr']-reference['ipr'][assignment]))),
                          'D_vs128_matched_max_error': float(np.max(abs(
                              result['D']-reference['D'][assignment]))),
                          'matching_vs128_min_overlap': float(overlap.min()),
                          'matching_vs128_mean_overlap': float(overlap.mean()),
                          'matching_vs128_max_participation': float(participation.max()),
                          'outer_set_replaced_vs128': len(reference_outer_set-matched_outer_set)}
                for k, name in enumerate(METRIC_NAMES):
                    record[name+'_minus128'] = float(result['metrics'][k]-reference['metrics'][k])
                if M == 128:
                    record.update(schur_basis_recheck(result))
                else:
                    record.update({key: '' for key in (
                        'repeated_schur_min_assigned_overlap',
                        'repeated_schur_mean_assigned_overlap',
                        'repeated_schur_D_max_difference',
                        'repeated_schur_IPR_max_difference',
                        'repeated_schur_D_outer_matched_difference',
                        'repeated_schur_q_max_circular_difference')})
                convergence.append(record)
                is_outer = np.zeros(DIM, dtype=int)
                is_outer[result['outer_indices']] = 1
                for n in range(DIM):
                    state_rows.append(dict(omega=omega, phi=phi, slices=M,
                        state_index=n, parity=int(result['parity'][n]),
                        quasienergy=float(result['quasienergy'][n]), q=float(q[n]),
                        ipr=float(result['ipr'][n]), D=float(result['D'][n]),
                        outer_selected=int(is_outer[n]),
                        assigned_M128_state_index=int(assignment[n]),
                        assigned_M128_overlap=float(overlap[n])))
            print(f'convergence omega={omega:g} phi/pi={phi/np.pi:g} done', flush=True)
    write_csv('validation_convergence.csv', convergence)
    write_csv('validation_eigenstates.csv', state_rows)

    for omega in (8., 16., 32., 64.):
        for phi in PHASES:
            ex = exact128.get((omega, phi))
            if ex is None:
                ex = calculate(89, 55, omega, phi, 128, return_matrices=True)
            eff = effective(omega, phi, need_vectors=True)
            # An independently evaluated core effective spectrum verifies that
            # folding changes the window rather than individual bare states.
            independent = calculate(89, 55, omega, phi, 32,
                                    model='effective', return_matrices=True)
            map_ind, overlap_ind, _ = match_by_overlap(
                eff['vectors_bare'], independent['eigenvectors'],
                eff['parity'], independent['parity'])
            bare_ipr_again, bare_D_again, norm_bare = vector_metrics(eff['vectors_bare'])
            kick_ipr_again, kick_D_again, norm_kick = vector_metrics(eff['vectors_kick'])
            idx_fold_recomputed = outer_indices(eff['q'])
            assert np.array_equal(idx_fold_recomputed, eff['folded_indices'])
            # There is exactly one folded window, used before/after dressing.
            kick_window = outer_indices(eff['q'].copy())
            assert np.array_equal(kick_window, eff['folded_indices'])
            assert np.max(abs(bare_ipr_again-eff['ipr_bare'])) < 1e-13
            assert np.max(abs(kick_ipr_again-eff['ipr_kick'])) < 1e-13
            kick_map = eff['vectors_kick']@eff['vectors_bare'].conj().T
            kick_unitarity = float(np.linalg.norm(
                kick_map.conj().T@kick_map-np.eye(DIM), 'fro')/np.sqrt(DIM))
            assert kick_unitarity < 1e-11
            if phi == 0.:
                assert eff['g_F']==0. and eff['kick_coefficient']==0.
                assert np.max(abs(eff['ipr_bare']-eff['ipr_kick'])) < 1e-12
                assert np.max(abs(eff['metrics'][1]-eff['metrics'][2])) < 1e-12
            record = dict(omega=omega, phi=phi, phi_over_pi=float(phi/np.pi),
                slices=128, **{name: float(ex['metrics'][k])
                               for k, name in enumerate(METRIC_NAMES)},
                **extra_checks(ex),
                unitarity_relative_fro=ex['unitarity_relative_fro'],
                g_F=eff['g_F'], kick_coefficient=eff['kick_coefficient'],
                eff_bare_norm_error=norm_bare, eff_kick_norm_error=norm_kick,
                kick_unitarity_relative_fro=kick_unitarity,
                folded_and_kick_selection_identical=True,
                bare_and_fold_single_state_IPR_identical=True,
                independent_bare_IPR_matched_max_difference=float(np.max(abs(
                    eff['ipr_bare']-independent['ipr'][map_ind]))),
                independent_bare_D_matched_max_difference=float(np.max(abs(
                    eff['D_bare']-independent['D'][map_ind]))),
                independent_bare_min_overlap=float(overlap_ind.min()),
                independent_folded_D_outer_difference=float(
                    eff['metrics'][1,1]-independent['metrics'][1]),
                folded_state_count=eff['folded_state_count'],
                bare_vs_fold_replaced=eff['bare_vs_fold_replaced'])
            for k, label in enumerate(('unfolded_bare', 'folded_bare', 'folded_kick')):
                record.update({label+'_'+name: float(eff['metrics'][k, j])
                               for j, name in enumerate(METRIC_NAMES)})
                record['D_outer_exact_minus_'+label] = float(ex['metrics'][1]-eff['metrics'][k,1])
                record.update({label+'_'+name: value for name, value in eff['ranges'][k].items()})
            for label, vectors_key, D_key, ipr_key in (
                    ('bare', 'vectors_bare', 'D_bare', 'ipr_bare'),
                    ('kick', 'vectors_kick', 'D_kick', 'ipr_kick')):
                assignment, overlap, participation = match_by_overlap(
                    ex['eigenvectors'], eff[vectors_key], ex['parity'], eff['parity'])
                chosen = ex['outer_indices']
                dD = ex['D']-eff[D_key][assignment]
                dIPR = ex['ipr']-eff[ipr_key][assignment]
                dq = circular_difference(ex['quasienergy']/omega, eff['q'][assignment])
                record.update({
                    label+'_matched_D_outer_difference': float(dD[chosen].mean()),
                    label+'_matched_D_outer_mean_abs_error': float(abs(dD[chosen]).mean()),
                    label+'_matched_D_outer_RMS_error': float(np.sqrt(np.mean(dD[chosen]**2))),
                    label+'_matched_D_all_RMS_error': float(np.sqrt(np.mean(dD**2))),
                    label+'_matched_IPR_outer_RMS_error': float(np.sqrt(np.mean(dIPR[chosen]**2))),
                    label+'_matched_q_all_max_error': float(abs(dq).max()),
                    label+'_matched_q_outer_RMS_error': float(np.sqrt(np.mean(dq[chosen]**2))),
                    label+'_matched_overlap_all_mean': float(overlap.mean()),
                    label+'_matched_overlap_all_min': float(overlap.min()),
                    label+'_matched_overlap_outer_mean': float(overlap[chosen].mean()),
                    label+'_matched_overlap_outer_min': float(overlap[chosen].min()),
                    label+'_matched_participation_outer_mean': float(participation[chosen].mean()),
                    label+'_matched_outer_window_replaced': int(len(chosen)-len(
                        np.intersect1d(assignment[chosen], eff['folded_indices']))),
                })
            bare_abs = abs(record['D_outer_exact_minus_folded_bare'])
            kick_abs = abs(record['D_outer_exact_minus_folded_kick'])
            record['kick_abs_outer_mean_error_reduction'] = bare_abs-kick_abs
            record['kick_abs_outer_mean_error_reduction_fraction'] = (
                1-kick_abs/bare_abs if bare_abs>1e-14 else 0.)
            record['kick_outer_matched_RMS_error_reduction'] = (
                record['bare_matched_D_outer_RMS_error']-record['kick_matched_D_outer_RMS_error'])
            high_frequency.append(record)
            print(f'high frequency omega={omega:g} phi/pi={phi/np.pi:g} done', flush=True)
    write_csv('validation_high_frequency.csv', high_frequency)

    summary = dict(
        description='Fresh exact midpoint-time-ordered validation with symmetry-sector Schur eigenvectors.',
        L=89, alpha='55/89', outer_count=36,
        exact_evaluations=54+9, convergence_rows=len(convergence),
        high_frequency_rows=len(high_frequency),
        maximum_hermiticity_residual=max(r['hermiticity_max'] for r in convergence),
        maximum_unitarity_relative_fro=max(r['unitarity_relative_fro'] for r in convergence+high_frequency),
        maximum_actual_eigen_residual=max(r['actual_eigen_residual_max'] for r in convergence+high_frequency),
        maximum_returned_vector_norm_error=max(r['returned_vector_norm_max_error'] for r in convergence+high_frequency),
        maximum_32_to128_D_outer_difference=max(abs(r['D_outer_minus128']) for r in convergence if r['slices']==32),
        maximum_64_to128_D_outer_difference=max(abs(r['D_outer_minus128']) for r in convergence if r['slices']==64),
        maximum_32_to128_IPR_outer_difference=max(abs(r['IPR_outer_minus128']) for r in convergence if r['slices']==32),
        maximum_64_to128_IPR_outer_difference=max(abs(r['IPR_outer_minus128']) for r in convergence if r['slices']==64),
        maximum_32_to128_selected_state_replacement=max(r['outer_set_replaced_vs128'] for r in convergence if r['slices']==32),
        maximum_64_to128_selected_state_replacement=max(r['outer_set_replaced_vs128'] for r in convergence if r['slices']==64),
        maximum_128_repeated_schur_D_difference=max(r['repeated_schur_D_max_difference'] for r in convergence if r['slices']==128),
        maximum_128_repeated_schur_IPR_difference=max(r['repeated_schur_IPR_max_difference'] for r in convergence if r['slices']==128),
        minimum_128_repeated_schur_assigned_overlap=min(r['repeated_schur_min_assigned_overlap'] for r in convergence if r['slices']==128),
        minimum_within_sector_circular_q_gap=min(r['min_within_sector_circular_q_gap'] for r in convergence),
        within_sector_near_degenerate_pairs_below_1e_10_max=max(r['within_sector_gap_below_1e_10_count'] for r in convergence),
        phi_zero_fold_and_kick_identical=True,
        folded_and_kick_selection_identical=True,
        bare_and_fold_single_state_IPR_identical=True,
        maximum_kick_norm_error=max(r['eff_kick_norm_error'] for r in high_frequency),
        maximum_kick_unitarity_relative_fro=max(r['kick_unitarity_relative_fro'] for r in high_frequency),
        limitations=[
            'All state matches respect the exact reflection-spin symmetry sector.',
            'A repeated Schur check tests numerical reproducibility, not a unique IPR under arbitrary rotations in an exactly degenerate subspace.',
            'Window-mean errors may cancel; matched-state RMS and overlaps are reported separately.',
            'No pointwise or monotonic kick improvement is assumed.',
            'Bare-unfolded and bare-folded use identical eigenvectors; only selected indices differ.',
            'Propagator kick-sign certification is in theory_checks.json, a separate fresh calculation.'
        ],
        high_frequency_by_phase=[],
    )
    for phi in PHASES:
        records = [r for r in high_frequency if r['phi']==phi]
        slope = lambda key: float(-np.polyfit(
            np.log([r['omega'] for r in records[-3:]]),
            np.log([max(r[key], 1e-20) for r in records[-3:]]), 1)[0])
        summary['high_frequency_by_phase'].append(dict(
            phi=phi, phi_over_pi=float(phi/np.pi),
            omega8_folded_outer_error=records[0]['D_outer_exact_minus_folded_bare'],
            omega8_kick_outer_error=records[0]['D_outer_exact_minus_folded_kick'],
            omega64_folded_outer_error=records[-1]['D_outer_exact_minus_folded_bare'],
            omega64_kick_outer_error=records[-1]['D_outer_exact_minus_folded_kick'],
            omega8_bare_matched_D_RMS=records[0]['bare_matched_D_outer_RMS_error'],
            omega8_kick_matched_D_RMS=records[0]['kick_matched_D_outer_RMS_error'],
            omega64_bare_matched_D_RMS=records[-1]['bare_matched_D_outer_RMS_error'],
            omega64_kick_matched_D_RMS=records[-1]['kick_matched_D_outer_RMS_error'],
            omega64_bare_outer_overlap_mean=records[-1]['bare_matched_overlap_outer_mean'],
            omega64_kick_outer_overlap_mean=records[-1]['kick_matched_overlap_outer_mean'],
            matched_kick_D_RMS_power_16_to64=slope('kick_matched_D_outer_RMS_error'),
            kick_outer_mean_error_improved_point_count=sum(r['kick_abs_outer_mean_error_reduction']>1e-12 for r in records),
            kick_outer_matched_RMS_improved_point_count=sum(r['kick_outer_matched_RMS_error_reduction']>1e-12 for r in records),
        ))
    summary['seconds'] = time.perf_counter()-started
    (ROOT/'validation_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    run()
