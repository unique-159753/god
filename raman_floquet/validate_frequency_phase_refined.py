"""Independent, read-only validation of the refined full-spectrum IPR/Dn scan.

The primary IPR sums all 178 spin components; Dn uses ln(89). Numerical
errors below compare discretizations or independent algorithms, never a
physical observable with an undriven baseline. No energy-ranked subset is
selected. Mirror phases outside the plotted 0..pi range are independently
propagated for the time-reversal check.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import schur
from scipy.optimize import linear_sum_assignment

from frequency_baselines import parameters_for_V
from frequency_fast_native import solve_point
from frequency_solver import reconstruct_modes, solve_point as fft_solve_point
from model import operators

BASE = Path(__file__).resolve().parent
DEFAULT_RESULTS = BASE / "results" / "frequency_phase_100x100_phi_0_pi"
L, N = 89, 178


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fold(values, omega):
    return (np.asarray(values)+omega/2) % omega-omega/2


def stats(modes):
    ipr = np.sum(abs(modes)**4, axis=0)
    return {"IPR": ipr, "Dn": -np.log(ipr)/np.log(L)}


def independent_plus_basis():
    """Build Q+ from reflection orbits without the production basis helper."""
    columns = []
    reflected = (-np.arange(L)-2) % L
    for spin, spin_z in ((0, 1), (1, -1)):
        for j, k in enumerate(reflected):
            if j > k or (j == k and spin_z != 1):
                continue
            v = np.zeros(N)
            if j == k:
                v[2*j+spin] = 1.
            else:
                v[2*j+spin] = 1/np.sqrt(2.)
                v[2*k+spin] = spin_z/np.sqrt(2.)
            columns.append(v)
    return np.column_stack(columns)


def spin_flip(modes):
    return np.asarray(modes).reshape(L, 2, -1)[:, ::-1, :].reshape(N, -1)


def solution(value, omega, operator=None):
    modes = value["modes"]
    eps = value["quasienergies"]
    if operator is None:
        operator = (modes*np.exp(-2j*np.pi*eps/omega)[None, :]) @ modes.conj().T
    return {"modes": modes, "quasienergies": eps,
            "parity_labels": value["parity_labels"], "operator": operator,
            **stats(modes)}


def checkpoint(out, iv, ip, iw):
    name = f"V{iv}_p{ip:02d}_w{iw:02d}.npz"
    with np.load(out/"points"/name, allow_pickle=False) as data:
        value = {key: data[key] for key in data.files}
    value["modes"] = reconstruct_modes(value["sector_vectors"], value["reconstruction_order"], L)
    return value


def compare(reference, candidate, omega, mirror=False):
    left = reference["modes"].conj() if mirror else reference["modes"]
    overlap = abs(left.conj().T @ candidate["modes"])**2
    allowed = reference["parity_labels"][:, None] == candidate["parity_labels"][None, :]
    rows, columns = linear_sum_assignment(np.where(allowed, -overlap, 10.))
    mapping = np.empty(N, int)
    mapping[rows] = columns
    reference_u = reference["operator"].T if mirror else reference["operator"]
    eps_error = fold(candidate["quasienergies"][mapping]-reference["quasienergies"], omega)
    return {
        "operator_relative_Frobenius_error": float(np.linalg.norm(candidate["operator"]-reference_u)/np.sqrt(N)),
        "max_circular_quasienergy_error": float(abs(eps_error).max()),
        "max_eigenphase_error": float(abs(eps_error).max()*2*np.pi/omega),
        "mean_IPR_error": float(abs(candidate["IPR"].mean()-reference["IPR"].mean())),
        "mean_Dn_error": float(abs(candidate["Dn"].mean()-reference["Dn"].mean())),
        "max_matched_IPR_error": float(abs(candidate["IPR"][mapping]-reference["IPR"]).max()),
        "max_matched_Dn_error": float(abs(candidate["Dn"][mapping]-reference["Dn"]).max()),
        "minimum_matched_mode_overlap": float(overlap[rows, columns].min()),
        "mean_matched_mode_overlap": float(overlap[rows, columns].mean()),
    }


def independent_ode(omega, phi, params):
    """Integrate the unsplit physical H(t), not the production gate sequence."""
    plus = independent_plus_basis()
    minus = spin_flip(plus)
    ops = operators(params)
    t_block = plus.T @ ops["T"] @ plus
    s_block = plus.T @ ops["S"] @ plus
    z_block = plus.T @ ops["HZ"] @ plus

    def rhs(tau, flat):
        h = ((params.t0+params.A*np.cos(omega*tau))*t_block
             +(params.delta0+params.B*np.cos(omega*tau+phi))*s_block+z_block)
        return (-1j*h @ flat.reshape(L, L)).ravel()

    started = time.monotonic()
    ivp = solve_ivp(rhs, (0., 2*np.pi/omega), np.eye(L, dtype=complex).ravel(),
                    method="DOP853", rtol=2e-12, atol=2e-14)
    if not ivp.success:
        raise RuntimeError(ivp.message)
    block = ivp.y[:, -1].reshape(L, L)
    triangular, vectors = schur(block, output="complex", check_finite=False)
    values = np.diag(triangular)
    eps = fold(-np.angle(values)*omega/(2*np.pi), omega)
    physical = plus @ vectors
    modes = np.column_stack([physical, spin_flip(physical.conj())])
    eps_all = np.r_[eps, fold(-eps, omega)]
    parity = np.r_[np.ones(L, int), -np.ones(L, int)]
    order = np.argsort(eps_all, kind="stable")
    u = plus @ block @ plus.T + minus @ block.conj() @ minus.T
    result = solution({"modes": modes[:, order], "quasienergies": eps_all[order],
                       "parity_labels": parity[order]}, omega, u)
    lam = np.r_[values, values.conj()][order]
    result["ode"] = {"method": "DOP853", "rtol": 2e-12, "atol": 2e-14,
                     "rhs_evaluations": int(ivp.nfev), "seconds": time.monotonic()-started,
                     "unitarity_error": float(np.linalg.norm(u.conj().T@u-np.eye(N))/np.sqrt(N)),
                     "eigen_residual": float(np.linalg.norm(u@result["modes"]-result["modes"]*lam[None, :])/np.sqrt(N))}
    return result


def audit_arrays(path, shape, label):
    failures, errors = [], {}
    with np.load(path, allow_pickle=False) as data:
        def close(name, actual, expected, tolerance=1e-11):
            error = float(np.max(abs(np.asarray(actual)-np.asarray(expected))))
            errors[name] = error
            if not np.allclose(actual, expected, rtol=0, atol=tolerance):
                failures.append(f"{label}: {name}, maximum error {error:g}")

        for name in data.files:
            if name in {"V", "omega", "phi", "metadata"}:
                continue
            if data[name].shape[:3] != shape:
                failures.append(f"{label}: {name} leading shape")
            if (name.startswith("outer") or name.startswith("delta_all")
                    or name.startswith("effective") or name.startswith("static_all")):
                failures.append(f"{label}: unrequested physical field {name}")
        ipr, dn, eps, parity = (data[name] for name in ("IPR", "Dn", "quasienergies", "parity_labels"))
        for name, value in (("IPR", ipr), ("Dn", dn), ("quasienergies", eps), ("parity_labels", parity)):
            if value.shape != shape+(N,):
                failures.append(f"{label}: {name} modal shape")
        if (not np.isfinite(ipr).all() or not np.isfinite(dn).all()
                or ipr.min() < 1/N-1e-12 or ipr.max() > 1+1e-12
                or dn.min() < -1e-12 or dn.max() > np.log(N)/np.log(L)+1e-12):
            failures.append(f"{label}: spin-component probability bounds")
        close("Dn_definition", dn, -np.log(ipr)/np.log(L))
        close("mean_IPR", data["all_IPR"], ipr.mean(axis=-1))
        close("mean_Dn", data["all_Dn"], dn.mean(axis=-1))
        close("positive_parity_count", (parity == 1).sum(axis=-1), L, 0.)
        close("negative_parity_count", (parity == -1).sum(axis=-1), L, 0.)
        omega = data["omega"][None, None, :, None]
        if (not np.isfinite(eps).all() or np.any(eps < -omega/2-1e-12)
                or np.any(eps >= omega/2+1e-12) or np.any(np.diff(eps, axis=-1) < -1e-12)):
            failures.append(f"{label}: centered, sorted quasienergies")
        steps = data["steps"]
        if np.any(steps < 64) or np.any(steps.astype(np.int64) & (steps.astype(np.int64)-1)):
            failures.append(f"{label}: invalid adaptive power-of-two step count")
        if not np.asarray(data["converged"], bool).all() or np.asarray(data["is_frozen"], bool).any():
            failures.append(f"{label}: nonconverged or frozen positive-frequency point")
        unity = data["unitarity_error"]
        if not np.isfinite(unity).all() or unity.min() < 0 or unity.max() > 1e-8:
            failures.append(f"{label}: nonunitarity")
        diagnostics = {key: float(np.max(data[key])) for key in
                       ("unitarity_error", "eigen_residual", "orthogonality_error", "hermiticity_error",
                        "Q_commutator_error", "particle_hole_error", "sector_projection_error",
                        "sector_unitarity_error", "schur_offdiagonal_error", "eigenvalue_modulus_error")}
        for key, value in diagnostics.items():
            if not np.isfinite(value) or value > 1e-8:
                failures.append(f"{label}: {key} above tolerance")
        axes = {key: data[key].copy() for key in ("V", "omega", "phi")}
    return {"label": label, "shape": list(shape), "failures": failures,
            "max_recomputation_errors": errors, "maximum_residuals": diagnostics,
            "max_unitarity_error": float(unity.max()),
            "IPR_range": [float(ipr.min()), float(ipr.max())],
            "Dn_range": [float(dn.min()), float(dn.max())],
            "positive_steps_range": [int(steps.min()), int(steps.max())]}, axes


def count_rows(path):
    with Path(path).open("rb") as stream:
        header = stream.readline().decode().strip().split(",")
        rows = sum(block.count(b"\n") for block in iter(lambda: stream.read(1 << 20), b""))
    return rows, header


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--integrity-only", action="store_true", help="Write a separate integrity report; no new numerical propagation")
    parser.add_argument("--sample-wavefunctions", action="store_true", help="Reconstruct a deterministic subset instead of all saved modes")
    args = parser.parse_args()
    out, started = args.results, time.monotonic()
    metadata = json.loads((out/"scan_parameters.json").read_text())
    failures, warnings = [], []
    report = {"scope": "Read-only production-data audit and independent time-ordered numerical validation",
              "primary_IPR": "sum_j,spin |psi_jspin(tau=0)|^4",
              "primary_Dn": "-ln(IPR)/ln89; upper bound ln178/ln89",
              "physical_baseline_differences": "not computed",
              "failures": failures, "warnings": warnings}
    report["grid"], axes = audit_arrays(out/"frequency_phase_scan.npz", (3, 100, 100), "grid")
    report["slices"], slice_axes = audit_arrays(out/"frequency_phase_slices.npz", (3, 2, 100), "slices")
    failures.extend(report["grid"]["failures"]+report["slices"]["failures"])
    for name, expected in (("V", [.5, 2., 4.]), ("omega", np.linspace(.2, 10., 100)),
                           ("phi", np.linspace(0., np.pi, 100))):
        if not np.allclose(axes[name], expected, rtol=0, atol=1e-14):
            failures.append(f"grid axis {name}")
    if len(np.unique(axes["phi"])) != 100 or not np.array_equal(axes["phi"], np.asarray(metadata["phi"])):
        failures.append("phase uniqueness/metadata")
    if not np.allclose(slice_axes["phi"], [0., np.pi/2], rtol=0, atol=1e-14):
        failures.append("exact slice phases")
    link_fields = ("IPR", "Dn", "quasienergies", "parity_labels", "all_IPR", "all_Dn", "steps", "unitarity_error")
    with np.load(out/"frequency_phase_scan.npz", allow_pickle=False) as grid, np.load(out/"frequency_phase_slices.npz", allow_pickle=False) as slices:
        for label, data in (("grid", grid), ("slices", slices)):
            if json.loads(str(data["metadata"].item())) != metadata:
                failures.append(f"{label}: aggregate metadata differs from scan_parameters.json")
        for key in grid.files:
            if key not in {"V", "omega", "phi", "metadata"}:
                if not np.array_equal(grid[key][:, 0], slices[key][:, 0], equal_nan=True):
                    failures.append(f"phi=0 exact slice/grid mismatch: {key}")
        # Load these arrays once. Each checkpoint must be linked to its actual
        # aggregate slot, beyond checking that both independently obey formulas.
        grid_link = {key: grid[key] for key in link_fields}
        slice_link = {key: slices[key] for key in link_fields}
    source_checks = {}
    for name, expected in metadata["code_sha256"].items():
        actual = digest(BASE/name)
        source_checks[name] = {"expected": expected, "actual": actual, "passed": actual == expected}
        if actual != expected:
            failures.append(f"production source hash changed: {name}")
    report["source_hashes"] = source_checks

    expected_names = {f"V{iv}_p{ip:02d}_w{iw:02d}.npz" for iv in range(3) for ip in range(101) for iw in range(100)}
    actual_names = {p.name for p in (out/"points").glob("*.npz")}
    missing, extra = sorted(expected_names-actual_names), sorted(actual_names-expected_names)
    report["point_files"] = {"expected": 30300, "actual": len(actual_names), "missing": missing, "extra": extra}
    if missing or extra:
        failures.append("checkpoint set incomplete or unexpected")
    point_failures, maxima = [], {}
    aggregate_link_maxima = {key: 0. for key in link_fields}
    reconstruction_maxima = {"IPR_error": 0., "Dn_error": 0., "column_probability_error": 0.}
    reconstructed, full_phases = 0, np.r_[axes["phi"], np.pi/2]
    limits = {"operator_delta": metadata["operator_tolerance"],
              "max_matched_Dn_delta": metadata["mode_tolerance"], "mean_Dn_delta": metadata["mean_tolerance"]}
    for iv in range(3):
        for ip in range(101):
            for iw in range(100):
                name = f"V{iv}_p{ip:02d}_w{iw:02d}.npz"
                if name not in actual_names:
                    continue
                with np.load(out/"points"/name, allow_pickle=False) as point:
                    if (float(point["V"]) != axes["V"][iv] or float(point["phi"]) != full_phases[ip]
                            or float(point["omega"]) != axes["omega"][iw]):
                        point_failures.append(name+":coordinates")
                    history = json.loads(str(point["convergence_history_json"].item()))
                    expected_steps = [2**power for power in range(int(np.log2(metadata["initial_steps"]))+1,
                                                                 int(np.log2(int(point["steps"])))+1)]
                    if not history or [int(x["steps"]) for x in history] != expected_steps:
                        point_failures.append(name+":adaptive_history_steps")
                        continue
                    for key, limit in limits.items():
                        value = float(history[-1][key])
                        maxima[key] = max(maxima.get(key, 0.), value)
                        if not np.isfinite(value) or value > limit*(1+1e-10) or value != float(point["final_"+key]):
                            point_failures.append(name+":"+key)
                    if not bool(point["converged"]) or bool(point["is_frozen"]):
                        point_failures.append(name+":not_converged_positive_frequency")
                    if any(k.startswith(("outer", "delta_all", "effective", "static_all")) for k in point.files):
                        point_failures.append(name+":unrequested_observable")
                    packed = grid_link if ip < 100 else slice_link
                    packed_index = (iv, ip if ip < 100 else 1, iw)
                    local_observables = {}
                    for key in link_fields:
                        actual, expected = point[key], packed[key][packed_index]
                        local_observables[key] = actual
                        error = float(np.max(abs(np.asarray(actual)-np.asarray(expected))))
                        aggregate_link_maxima[key] = max(aggregate_link_maxima[key], error)
                        if not np.array_equal(actual, expected, equal_nan=True):
                            point_failures.append(name+":aggregate_link_"+key)
                    raw_sector_vectors = point["sector_vectors"]
                    order = point["reconstruction_order"]
                    if raw_sector_vectors.shape != (L, L) or order.shape != (N,):
                        point_failures.append(name+":lossless_coefficient_shape")
                    if not np.array_equal(np.sort(order), np.arange(N)):
                        point_failures.append(name+":reconstruction_not_a_permutation")
                    else:
                        reconstructed_parity = np.r_[np.ones(L, int), -np.ones(L, int)][order]
                        sector_eps = point["sector_quasienergies"]
                        reconstructed_eps = np.r_[sector_eps, fold(-sector_eps, float(point["omega"]))][order]
                        if not np.array_equal(reconstructed_parity, local_observables["parity_labels"]):
                            point_failures.append(name+":raw_reconstruction_parity")
                        if not np.allclose(reconstructed_eps, local_observables["quasienergies"], rtol=0, atol=1e-13):
                            point_failures.append(name+":raw_reconstruction_quasienergies")
                    if not args.sample_wavefunctions or (ip in (0, 25, 49, 75, 99, 100) and iw in (0, 1, 10, 30, 60, 99)):
                        modes = reconstruct_modes(raw_sector_vectors, order, L)
                        raw_stats = stats(modes)
                        local = {"IPR_error": float(abs(raw_stats["IPR"]-local_observables["IPR"]).max()),
                                 "Dn_error": float(abs(raw_stats["Dn"]-local_observables["Dn"]).max()),
                                 "column_probability_error": float(abs((abs(modes)**2).sum(axis=0)-1).max())}
                        for key, value in local.items():
                            reconstruction_maxima[key] = max(reconstruction_maxima[key], value)
                            if not np.isfinite(value) or value > 1e-10:
                                point_failures.append(name+":"+key)
                        reconstructed += 1
            print(f"Audited checkpoints V-index {iv}, phase-index {ip}/100", flush=True) if ip % 25 == 0 else None
    report["checkpoint_audit"] = {"final_maximum_metrics": maxima, "failures": point_failures,
                                  "aggregate_link_maximum_errors": aggregate_link_maxima,
                                  "aggregate_link_fields": list(link_fields),
                                  "reconstructed_point_count": reconstructed,
                                  "reconstruction_maximum_errors": reconstruction_maxima,
                                  "wavefunction_audit": "deterministic subset" if args.sample_wavefunctions else "all 30300 points / all 5393400 eigenstates"}
    if point_failures:
        failures.append("checkpoint history or raw-mode audit failed")
    del grid_link, slice_link
    completion = json.loads((out/"scan_completion.json").read_text())
    for key, expected in (("grid_points", 30000), ("additional_pi_over_2_points", 300),
                           ("frozen_points_in_grid", 0), ("positive_frequency_points_in_grid", 30000)):
        if completion.get(key) != expected:
            failures.append(f"completion record: {key}")
    expected_steps_range = [min(report[label]["positive_steps_range"][0] for label in ("grid", "slices")),
                            max(report[label]["positive_steps_range"][1] for label in ("grid", "slices"))]
    if completion.get("nonconverged") or completion.get("positive_steps_range") != expected_steps_range:
        failures.append("completion record: convergence/step range")
    report["scan_completion"] = completion
    csv_counts = {}
    for name, expected in (("frequency_phase_scan.csv", 30000), ("frequency_extra_pi_over_2.csv", 300),
                           ("eigenstate_observables.csv", 30300*N)):
        rows, header = count_rows(out/name)
        csv_counts[name] = {"rows": rows, "expected": expected, "columns": header}
        if rows != expected:
            failures.append(f"CSV row count: {name}")
        if any(k.startswith(("outer", "delta_all", "effective", "static_all")) for k in header):
            failures.append(f"unrequested CSV observable: {name}")
    report["csv_counts"] = csv_counts

    hermiticity = []
    for v in axes["V"]:
        params, worst = parameters_for_V(float(v)), 0.
        ops = operators(params)
        for phase in (0., np.pi/2, np.pi):
            for drive_angle in (0., .35, 1.6, 3.9, 5.7):
                h = ((params.t0+params.A*np.cos(drive_angle))*ops["T"]
                     +(params.delta0+params.B*np.cos(drive_angle+phase))*ops["S"]+ops["HZ"])
                worst = max(worst, float(np.linalg.norm(h-h.conj().T)/max(np.linalg.norm(h), 1.)))
        hermiticity.append({"V": float(v), "sample_count": 15, "maximum_relative_error": worst})
        if worst > 1e-12:
            failures.append(f"Hamiltonian Hermiticity V={v}")
    report["Hermiticity"] = hermiticity

    raw, convergence_rows, cross_rows, representatives, phase_rows = {}, [], [], [], []
    if not args.integrity_only:
        frequencies = [0, int(np.argmin(abs(axes["omega"]-1))), int(np.argmin(abs(axes["omega"]-6))), 99]
        for iv in range(3):
            params = parameters_for_V(float(axes["V"][iv]))
            for iw in frequencies:
                for ip in (0, 100):
                    stored = checkpoint(out, iv, ip, iw)
                    omega, phi = float(stored["omega"]), float(stored["phi"])
                    computed = {m: solve_point(omega, phi, params, m, order=6) for m in (32, 64, 128)}
                    reference_steps = max(128, int(stored["steps"]))
                    ref = solution(stored if int(stored["steps"]) >= 128 else computed[128], omega)
                    case = len(representatives)
                    entry = {"case": case, "V": float(axes["V"][iv]), "omega": omega, "phi": phi,
                             "reference_steps": reference_steps, "production_steps": int(stored["steps"]), "resolutions": []}
                    for m, value in computed.items():
                        row = {"case": case, "V": float(axes["V"][iv]), "omega": omega, "phi": phi,
                               "steps": m, "reference": f"adaptive_or_M128_{reference_steps}",
                               **compare(ref, solution(value, omega), omega)}
                        entry["resolutions"].append(row)
                        convergence_rows.append(row)
                        for key in ("sector_vectors", "reconstruction_order", "quasienergies", "parity_labels"):
                            raw[f"case{case}_M{m}_{key}"] = value[key]
                    for key in ("sector_vectors", "reconstruction_order", "IPR", "Dn", "quasienergies", "parity_labels"):
                        raw[f"case{case}_production_{key}"] = stored[key]
                    if iw == 0:
                        ode = independent_ode(omega, phi, params)
                        metrics = compare(ode, solution(stored, omega), omega)
                        entry["independent_ODE"] = {**ode["ode"], "production_comparison": metrics}
                        for key in ("modes", "IPR", "Dn", "quasienergies", "parity_labels"):
                            raw[f"case{case}_ODE_{key}"] = ode[key]
                        if (metrics["operator_relative_Frobenius_error"] > 2e-6
                                or metrics["max_matched_Dn_error"] > 3e-4
                                or metrics["max_matched_IPR_error"] > 1e-4
                                or metrics["max_eigenphase_error"] > 3e-6):
                            warnings.append(f"independent ODE deviation needs inspection at case {case}")
                            print("WARNING: "+warnings[-1]+" "+json.dumps(metrics), flush=True)
                        for m, value in computed.items():
                            convergence_rows.append({"case": case, "V": float(axes["V"][iv]), "omega": omega,
                                                     "phi": phi, "steps": m, "reference": "independent_DOP853",
                                                     **compare(ode, solution(value, omega), omega)})
                    representatives.append(entry)
                    print(f"Validated convergence {case+1}/24 V={axes['V'][iv]:g}, omega={omega:g}, phi/pi={phi/np.pi:g}", flush=True)
            for iw in (0, 99):
                ip = 25 if iw == 0 else 49
                stored = checkpoint(out, iv, ip, iw)
                omega, phi, steps = float(stored["omega"]), float(stored["phi"]), int(stored["steps"])
                mirror_phi = 2*np.pi-phi
                mirrored = solve_point(omega, mirror_phi, params, steps, order=6)
                metrics = compare(solution(stored, omega), solution(mirrored, omega), omega, mirror=True)
                phase_rows.append({"V": float(axes["V"][iv]), "omega": omega, "phi": phi,
                                   "partner_phi": mirror_phi, "steps": steps, **metrics})
                k = len(phase_rows)-1
                for key in ("sector_vectors", "reconstruction_order", "quasienergies", "parity_labels"):
                    raw[f"mirror{k}_{key}"] = mirrored[key]
                    raw[f"mirror{k}_reference_{key}"] = stored[key]
                if (metrics["operator_relative_Frobenius_error"] > 3e-6
                        or metrics["max_matched_Dn_error"] > 3e-4
                        or metrics["max_matched_IPR_error"] > 1e-4
                        or metrics["max_eigenphase_error"] > 3e-6):
                    warnings.append(f"phase-reversal symmetry needs inspection V={axes['V'][iv]}, omega={omega}")
                    print("WARNING: "+warnings[-1]+" "+json.dumps(metrics), flush=True)
                fft_steps = steps
                fft_value = fft_solve_point(omega, phi, params, fft_steps, order=6)
                fft_metrics = compare(solution(stored, omega), solution(fft_value, omega), omega)
                while (fft_metrics["operator_relative_Frobenius_error"] > 2e-6
                       or fft_metrics["max_matched_Dn_error"] > 3e-4
                       or fft_metrics["max_matched_IPR_error"] > 1e-4
                       or fft_metrics["max_eigenphase_error"] > 3e-6) and fft_steps < metadata["max_steps"]:
                    fft_steps *= 2
                    fft_value = fft_solve_point(omega, phi, params, fft_steps, order=6)
                    fft_metrics = compare(solution(stored, omega), solution(fft_value, omega), omega)
                cross_rows.append({"V": float(axes["V"][iv]), "omega": omega, "phi": phi,
                                   "native_steps": steps, "FFT_steps": fft_steps, **fft_metrics})
                for key in ("sector_vectors", "reconstruction_order", "quasienergies", "parity_labels"):
                    raw[f"FFT_cross{k}_{key}"] = fft_value[key]
                    raw[f"FFT_cross{k}_native_{key}"] = stored[key]
                if (fft_metrics["operator_relative_Frobenius_error"] > 2e-6
                        or fft_metrics["max_matched_Dn_error"] > 3e-4
                        or fft_metrics["max_matched_IPR_error"] > 1e-4
                        or fft_metrics["max_eigenphase_error"] > 3e-6):
                    warnings.append(f"independent FFT needs inspection V={axes['V'][iv]}, omega={omega}")
                    print("WARNING: "+warnings[-1]+" "+json.dumps(fft_metrics), flush=True)
        save_csv(out/"frequency_refined_convergence_diagnostics.csv", convergence_rows)
        save_csv(out/"frequency_refined_method_crosscheck.csv", cross_rows)
        save_csv(out/"frequency_refined_phase_symmetry.csv", phase_rows)
        np.savez_compressed(out/"frequency_refined_validation_modes.npz", **raw)
    report["representative_convergence"] = representatives
    report["independent_ODE_points"] = sum("independent_ODE" in entry for entry in representatives)
    report["phase_reversal_selected_modes"] = phase_rows
    report["FFT_crosschecks"] = cross_rows
    report["independent_check_thresholds"] = {
        "operator_relative_Frobenius_error": 2e-6,
        "phase_reversal_operator_relative_Frobenius_error": 3e-6,
        "max_matched_Dn_error": 3e-4,
        "max_matched_IPR_error": 1e-4,
        "max_eigenphase_error": 3e-6,
        "on_exceedance": "Retain a warning and set numerical_validation_passed=false; never modify production data",
    }
    report["integrity_only"] = args.integrity_only
    report["data_integrity_passed"] = not failures
    report["numerical_validation_passed"] = None if args.integrity_only else not (failures or warnings)
    report["interpretation"] = ("Unitarity alone does not establish time-discretization accuracy. Coarse M32/64/128 results "
        "are diagnostics; only the adaptive production data are plotted. The phase-reversal partners outside 0..pi are "
        "new independent propagations, never filled into the requested grid. Small finite-L IPR/Dn changes alone "
        "do not establish a thermodynamic localization transition. Near degeneracies can make individual eigenstate "
        "observables sensitive to basis choice; fixed reflection sectors and explicit mode matching are retained.")
    report["seconds"] = time.monotonic()-started
    name = "frequency_refined_integrity.json" if args.integrity_only else "frequency_refined_validation.json"
    (out/name).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"data_integrity_passed": not failures, "failures": failures, "warnings": warnings,
                      "independent_ODE_points": report["independent_ODE_points"], "seconds": report["seconds"]}, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
