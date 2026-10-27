"""Independent audit of the exact three-V frequency/phase scan.

The main Dn is -ln(spin-resolved IPR)/ln(89), so its upper bound exceeds one.
No production output is changed.  Coarse M32/64/128 runs are diagnostics,
not replacements for the adaptively converged grid.  Independent DOP853
integrates the actual time-dependent Hamiltonian in an explicitly built Q+
sector; it does not use the production time-stepping algorithm.
"""
from __future__ import annotations

import os
for _key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_key] = "1"

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import schur, svdvals
from scipy.optimize import linear_sum_assignment

from model import operators
from frequency_baselines import parameters_for_V
from frequency_solver import reconstruct_modes
from frequency_fast_native import solve_point

BASE = Path(__file__).resolve().parent
DEFAULT_RESULTS = BASE / "results" / "frequency_phase_three_V"
L, N, COUNT = 89, 178, 36


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stats(modes):
    p = abs(modes) ** 2
    ipr = np.sum(p*p, axis=0)
    density = p.reshape(L, 2, -1).sum(axis=1)
    site = np.sum(density*density, axis=0)
    return {"IPR": ipr, "Dn": -np.log(ipr)/np.log(L),
            "IPR_site": site, "D_site": -np.log(site)/np.log(L)}


def outer(eps):
    return np.argsort(abs(eps), kind="stable")[-COUNT:]


def fold(values, omega):
    return (np.asarray(values)+omega/2) % omega-omega/2


def own_plus_basis():
    """Construct Q+ directly from reflection orbits, independently of solver."""
    cols = []
    reflected = (-np.arange(L)-2) % L
    for spin, spin_z in [(0, 1), (1, -1)]:
        for i, k in enumerate(reflected):
            if i > k:
                continue
            v = np.zeros(N)
            if i == k:
                if spin_z != 1:
                    continue
                v[2*i+spin] = 1.
            else:
                v[2*i+spin] = 1/np.sqrt(2.)
                v[2*k+spin] = spin_z/np.sqrt(2.)
            cols.append(v)
    return np.column_stack(cols)


def spin_flip(v):
    return np.asarray(v).reshape(L, 2, -1)[:, ::-1, :].reshape(N, -1)


def solution(modes, eps, parity, omega, operator=None):
    modes, eps = np.asarray(modes), np.asarray(eps)
    if operator is None:
        operator = (modes*np.exp(-2j*np.pi*eps/omega)[None, :]) @ modes.conj().T
    return {"modes": modes, "quasienergies": eps,
            "parity_labels": np.asarray(parity), "operator": operator, **stats(modes)}


def from_production(value, omega):
    return solution(value["modes"], value["quasienergies"], value["parity_labels"], omega)


def checkpoint(out, iv, ip, iw):
    with np.load(out/"points"/f"V{iv}_p{ip:02d}_w{iw:02d}.npz", allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    if iw == 0:
        data["modes"] = data["frozen_modes"]
    else:
        data["modes"] = reconstruct_modes(data["sector_vectors"], data["reconstruction_order"], L)
    return data


def compare(ref, candidate, omega, conjugate_reference=False):
    left = ref["modes"].conj() if conjugate_reference else ref["modes"]
    overlap = abs(left.conj().T @ candidate["modes"]) ** 2
    allowed = ref["parity_labels"][:, None] == candidate["parity_labels"][None, :]
    i, j = linear_sum_assignment(np.where(allowed, -overlap, 10.))
    mapping = np.empty(N, int)
    mapping[i] = j
    ix, iy = outer(ref["quasienergies"]), outer(candidate["quasienergies"])
    singular = svdvals(left[:, ix].conj().T @ candidate["modes"][:, iy])
    phase = fold(candidate["quasienergies"][mapping]-ref["quasienergies"], omega)
    reference_operator = ref["operator"].T if conjugate_reference else ref["operator"]
    return {
        "operator_relative_Frobenius_error": float(np.linalg.norm(candidate["operator"]-reference_operator)/np.sqrt(N)),
        "max_circular_quasienergy_error": float(abs(phase).max()),
        "max_eigenphase_error": float(abs(phase).max()*2*np.pi/omega),
        "mean_Dn_error": float(abs(candidate["Dn"].mean()-ref["Dn"].mean())),
        "outer_Dn_error": float(abs(candidate["Dn"][iy].mean()-ref["Dn"][ix].mean())),
        "max_matched_Dn_error": float(abs(candidate["Dn"][mapping]-ref["Dn"]).max()),
        "max_matched_IPR_error": float(abs(candidate["IPR"][mapping]-ref["IPR"]).max()),
        "minimum_matched_mode_overlap": float(overlap[i, j].min()),
        "minimum_outer_mode_overlap": float(overlap[ix, mapping[ix]].min()),
        "outer_subspace_min_squared_singular_value": float(np.min(singular**2)),
        "outer_subspace_mean_squared_overlap": float(np.mean(singular**2)),
        "outer_membership_changes": int(np.count_nonzero(~np.isin(mapping[ix], iy))),
    }


def independent_ode(omega, phi, params):
    plus = own_plus_basis()
    minus = spin_flip(plus)
    ops = operators(params)
    t_block = plus.T @ ops["T"] @ plus
    s_block = plus.T @ ops["S"] @ plus
    z_block = plus.T @ ops["HZ"] @ plus
    period = 2*np.pi/omega

    def rhs(tau, flat):
        h = ((params.t0+params.A*np.cos(omega*tau))*t_block
             +(params.delta0+params.B*np.cos(omega*tau+phi))*s_block+z_block)
        return (-1j*h @ flat.reshape(L, L)).ravel()

    started = time.monotonic()
    ivp = solve_ivp(rhs, (0., period), np.eye(L, dtype=complex).ravel(),
                    method="DOP853", rtol=2e-12, atol=2e-14)
    if not ivp.success:
        raise RuntimeError(ivp.message)
    block = ivp.y[:, -1].reshape(L, L)
    triangular, vectors = schur(block, output="complex", check_finite=False)
    eigenvalues = np.diag(triangular)
    eps = fold(-np.angle(eigenvalues)*omega/(2*np.pi), omega)
    physical = plus @ vectors
    all_modes = np.column_stack([physical, spin_flip(physical.conj())])
    all_eps = np.r_[eps, fold(-eps, omega)]
    all_parity = np.r_[np.ones(L, int), -np.ones(L, int)]
    order = np.argsort(all_eps, kind="stable")
    u = plus @ block @ plus.T + minus @ block.conj() @ minus.T
    result = solution(all_modes[:, order], all_eps[order], all_parity[order], omega, u)
    lam = np.r_[eigenvalues, eigenvalues.conj()][order]
    result["ode"] = {"method": "DOP853", "rtol": 2e-12, "atol": 2e-14,
                     "rhs_evaluations": int(ivp.nfev), "seconds": time.monotonic()-started,
                     "unitarity_error": float(np.linalg.norm(u.conj().T@u-np.eye(N))/np.sqrt(N)),
                     "eigen_residual": float(np.linalg.norm(u@result["modes"]-result["modes"]*lam[None, :])/np.sqrt(N))}
    return result


def audit_arrays(data, shape, label):
    failed, errors = [], {}

    def close(name, a, b, tolerance=1e-11):
        a, b = np.asarray(a), np.asarray(b)
        value = float(np.nanmax(abs(a-b))) if a.size and b.size else 0.
        errors[name] = value
        if not np.allclose(a, b, rtol=0, atol=tolerance, equal_nan=True):
            failed.append(f"{label}: {name} (maximum finite error {value:g})")

    for name in data.files:
        if name in {"V", "omega", "phi", "metadata"}:
            continue
        if data[name].shape[:3] != shape:
            failed.append(f"{label}: {name} leading shape {data[name].shape}")
    for name in ["IPR", "Dn", "IPR_site", "D_site", "D_component_log2L", "quasienergies", "energies", "parity_labels"]:
        if data[name].shape != shape+(N,):
            failed.append(f"{label}: {name} modal shape")
    ipr, site, dn = data["IPR"], data["IPR_site"], data["Dn"]
    if (not np.isfinite(ipr).all() or not np.isfinite(dn).all()
            or np.min(ipr) < 1/N-1e-12 or np.max(ipr) > 1+1e-12
            or np.min(dn) < -1e-12 or np.max(dn) > np.log(N)/np.log(L)+1e-12
            or np.min(site) < 1/L-1e-12 or np.max(site) > 1+1e-12
            or np.max(ipr-site) > 1e-12 or np.max(site-2*ipr) > 1e-12):
        failed.append(f"{label}: IPR/D probability bounds")
    close("Dn_definition", dn, -np.log(ipr)/np.log(L))
    close("D_component_log2L_definition", data["D_component_log2L"], -np.log(ipr)/np.log(N))
    close("D_site_definition", data["D_site"], -np.log(site)/np.log(L))
    for name in ["IPR", "Dn", "IPR_site", "D_site"]:
        close(f"all_{name}", data[f"all_{name}"], data[name].mean(axis=-1))
    close("delta_all_Dn", data["delta_all_Dn"], data["all_Dn"]-data["static_all_Dn"])
    eps = data["quasienergies"]
    if not np.isnan(eps[:, :, 0]).all() or not np.isnan(data["outer_Dn"][:, :, 0]).all():
        failed.append(f"{label}: zero-frequency Floquet values must be NaN")
    if not np.isnan(data["unitarity_error"][:, :, 0]).all():
        failed.append(f"{label}: zero-frequency unitarity must be NaN")
    if not np.all(data["outer_indices"][:, :, 0] == -1):
        failed.append(f"{label}: zero-frequency outer indices must be -1")
    if not np.all(data["steps"][:, :, 0] == 0):
        failed.append(f"{label}: frozen steps must be zero")
    if (not np.isfinite(data["energies"][:, :, 0]).all()
            or not np.isnan(data["energies"][:, :, 1:]).all()):
        failed.append(f"{label}: frozen energy/Floquet-energy distinction")
    expected_frozen = np.broadcast_to(np.arange(shape[2]) == 0, shape)
    close("is_frozen", data["is_frozen"].astype(int), expected_frozen.astype(int), 0.)
    if not np.asarray(data["converged"], bool).all():
        failed.append(f"{label}: nonconverged production points")
    omega = data["omega"][None, None, 1:, None]
    if (not np.isfinite(eps[:, :, 1:]).all() or np.any(eps[:, :, 1:] < -omega/2-1e-12)
            or np.any(eps[:, :, 1:] >= omega/2+1e-12)):
        failed.append(f"{label}: centered quasienergy bounds")
    indices = np.argsort(abs(eps[:, :, 1:]), axis=-1, kind="stable")[..., -COUNT:]
    close("positive_outer_indices", data["outer_indices"][:, :, 1:], indices, 0.)
    for name in ["Dn", "IPR", "D_site"]:
        close(f"outer_{name}", data[f"outer_{name}"][:, :, 1:],
              np.take_along_axis(data[name][:, :, 1:], indices, axis=-1).mean(axis=-1))
    close("parity_count", np.sum(data["parity_labels"] == 1, axis=-1), L, 0.)
    steps = data["steps"][:, :, 1:]
    if (np.any(steps < 64) or np.any((steps.astype(np.int64) & (steps.astype(np.int64)-1)) != 0)):
        failed.append(f"{label}: invalid adaptive power-of-two step count")
    unity = data["unitarity_error"][:, :, 1:]
    if not np.isfinite(unity).all() or unity.min() < 0 or unity.max() > 1e-8:
        failed.append(f"{label}: unitarity above tolerance")
    return {"label": label, "shape": list(shape), "failures": failed,
            "max_recomputation_errors": errors,
            "max_unitarity_error": float(unity.max()),
            "IPR_range": [float(ipr.min()), float(ipr.max())],
            "Dn_range": [float(dn.min()), float(dn.max())],
            "positive_steps_range": [int(steps.min()), int(steps.max())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--skip-convergence", action="store_true")
    parser.add_argument("--skip-ode", action="store_true")
    parser.add_argument("--fft-crosscheck-results", type=Path, default=None,
                        help="Retained independent FFT checkpoints; default is sibling *_fft_crosscheck")
    args = parser.parse_args()
    out = args.results
    started = time.monotonic()
    metadata = json.loads((out/"scan_parameters.json").read_text())
    integration_order = int(metadata.get("integration_order", 4))
    failures, warnings = [], []
    report = {"scope": "read-only data audit with independent selected convergence/ODE checks",
              "integration_order": integration_order,
              "primary_Dn": "-ln(sum178 |psi|^4)/ln89; upper bound ln178/ln89",
              "zero_frequency": "frozen static; Floquet quasienergy and outer quantities undefined",
              "failures": failures, "warnings": warnings}
    with np.load(out/"frequency_phase_scan.npz", allow_pickle=False) as grid, np.load(out/"frequency_phase_slices.npz", allow_pickle=False) as slices:
        for key, expected in [("V", [.5, 2., 4.]), ("omega", np.linspace(0, 10, 50)),
                              ("phi", 2*np.pi*np.arange(50)/50)]:
            if not np.allclose(grid[key], expected, rtol=0, atol=1e-14):
                failures.append(f"grid axis {key}")
        if len(np.unique(grid["phi"])) != 50 or np.max(grid["phi"]) >= 2*np.pi:
            failures.append("phase sampling not 50 unique periodic points")
        if not np.allclose(slices["phi"], [0, np.pi/2], rtol=0, atol=1e-14):
            failures.append("exact slice phases")
        report["grid"] = audit_arrays(grid, (3, 50, 50), "grid")
        report["slices"] = audit_arrays(slices, (3, 2, 50), "slices")
        failures.extend(report["grid"]["failures"]+report["slices"]["failures"])
        for key in grid.files:
            if key in {"V", "omega", "phi", "metadata"}:
                continue
            if not np.allclose(grid[key][:, 0], slices[key][:, 0], rtol=0, atol=0, equal_nan=True):
                failures.append(f"phi=0 exact slice/grid mismatch: {key}")
        with np.load(out/"baselines.npz", allow_pickle=False) as baseline:
            expected = baseline["static_all_Dn"][:, None, None]
            if not np.allclose(grid["static_all_Dn"], expected, rtol=0, atol=1e-12):
                failures.append("undriven Dn baseline mismatch")
            for iv in range(3):
                ipr = np.sum(abs(baseline["static_modes"][iv])**4, axis=0)
                if not np.allclose(ipr, baseline["static_IPR"][iv], rtol=0, atol=1e-12):
                    failures.append(f"static raw-mode IPR V index {iv}")
        source_checks = {}
        for name, expected in metadata["code_sha256"].items():
            actual = digest(BASE/name)
            source_checks[name] = {"expected": expected, "actual": actual, "passed": actual == expected}
            if actual != expected:
                failures.append(f"source hash changed: {name}")
        report["source_hashes"] = source_checks
        axes = {k: np.asarray(grid[k]).copy() for k in ["V", "omega", "phi"]}
        # Every independently computed phase pair is present; no symmetry interpolation.
        partner = (-np.arange(50)) % 50
        phase_symmetry = {}
        for key in ["all_IPR", "all_Dn", "outer_Dn", "delta_all_Dn"]:
            delta = abs(grid[key]-grid[key][:, partner])
            phase_symmetry[key] = {"max_finite_absolute_error": float(np.nanmax(delta))}
            if np.nanmax(delta) > 1e-3:
                warnings.append(f"all-grid phase reversal {key} exceeds 1e-3; inspect selected mode matching")
        report["phase_reversal_grid_statistics"] = phase_symmetry
        degeneracy = []
        all_eps = grid["quasienergies"][:, :, 1:]
        all_parity = grid["parity_labels"][:, :, 1:]
        plus_eps = all_eps[all_parity == 1].reshape(3, 50, 49, L)
        positive_omega = grid["omega"][None, None, 1:, None]
        plus_gaps = np.diff(np.concatenate([plus_eps, plus_eps[..., :1]+positive_omega], axis=-1), axis=-1)
        same_parity_phase_gap = plus_gaps.min(axis=-1)*2*np.pi/positive_omega[..., 0]
        for iv, v in enumerate(grid["V"]):
            gap = grid["outer_selection_gap"][iv, :, 1:]
            phase_gap = grid["min_eigenphase_gap"][iv, :, 1:]
            degeneracy.append({"V": float(v), "minimum_outer_selection_gap": float(gap.min()),
                               "outer_gaps_below_1e_minus_9": int(np.count_nonzero(gap < 1e-9)),
                               "minimum_eigenphase_gap": float(phase_gap.min()),
                               "eigenphase_gaps_below_1e_minus_9": int(np.count_nonzero(phase_gap < 1e-9)),
                               "minimum_same_Q_eigenphase_gap": float(same_parity_phase_gap[iv].min()),
                               "same_Q_eigenphase_gaps_below_1e_minus_9": int(np.count_nonzero(same_parity_phase_gap[iv] < 1e-9))})
        report["near_degeneracy_grid_statistics"] = degeneracy

    expected_names = {f"V{iv}_p{ip:02d}_w{iw:02d}.npz"
                      for iv in range(3) for ip in range(51) for iw in range(50)}
    actual_names = {p.name for p in (out/"points").glob("*.npz")}
    missing, extra = sorted(expected_names-actual_names), sorted(actual_names-expected_names)
    if missing or extra:
        failures.append("checkpoint set incomplete or unexpected")
    report["point_files"] = {"expected": len(expected_names), "actual": len(actual_names),
                             "missing": missing, "extra": extra}
    history_failures, history_max = [], {}
    for iv in range(3):
        for ip in range(51):
            phi = float(axes["phi"][ip]) if ip < 50 else np.pi/2
            for iw in range(50):
                name = f"V{iv}_p{ip:02d}_w{iw:02d}.npz"
                if name not in actual_names:
                    continue
                with np.load(out/"points"/name, allow_pickle=False) as p:
                    if (float(p["V"]) != axes["V"][iv] or float(p["phi"]) != phi
                            or float(p["omega"]) != axes["omega"][iw]):
                        failures.append(f"point coordinate mismatch: {name}")
                    if iw == 0:
                        continue
                    history = json.loads(str(p["convergence_history_json"].item()))
                    if not history or int(history[-1]["steps"]) != int(p["steps"]):
                        history_failures.append(name)
                        continue
                    last = history[-1]
                    limits = {"operator_delta": metadata["operator_tolerance"],
                              "max_matched_Dn_delta": metadata["mode_tolerance"],
                              "mean_Dn_delta": metadata["mean_tolerance"],
                              "outer_Dn_delta": metadata["mean_tolerance"]}
                    for key, limit in limits.items():
                        history_max[key] = max(history_max.get(key, 0.), float(last[key]))
                        if not np.isfinite(last[key]) or last[key] > limit*(1+1e-10):
                            history_failures.append(name+":"+key)
                    if last["outer_membership_changes"] != 0 or not bool(p["converged"]):
                        history_failures.append(name+":nonconverged")
    report["adaptive_history"] = {"final_maximum_metrics": history_max, "failures": history_failures}
    if history_failures:
        failures.append("adaptive final checks not satisfied")
    csv_counts = {}
    for name, count in [("frequency_phase_scan.csv", 7500), ("frequency_extra_pi_over_2.csv", 150),
                        ("eigenstate_observables.csv", 7650*N)]:
        with (out/name).open() as stream:
            rows = sum(1 for _ in stream)-1
        csv_counts[name] = {"rows": rows, "expected": count}
        if rows != count:
            failures.append(f"CSV row count: {name}")
    report["csv_counts"] = csv_counts

    diagnostic_rows, raw = [], {}
    raw.update(V=axes["V"], omega=axes["omega"], phi=axes["phi"])
    representative_points = []
    frequency_indices = [1, int(np.argmin(abs(axes["omega"]-1))),
                         int(np.argmin(abs(axes["omega"]-6))), 49]
    if not args.skip_convergence:
        for iv in range(3):
            for iw in frequency_indices:
                for ip in [0, 50]:
                    omega = float(axes["omega"][iw])
                    phi = 0. if ip == 0 else np.pi/2
                    params = parameters_for_V(axes["V"][iv])
                    stored = checkpoint(out, iv, ip, iw)
                    computed = {m: solve_point(omega, phi, params, m, return_operator=True,
                                               order=integration_order)
                                for m in [32, 64, 128]}
                    ref_raw = computed[128] if int(stored["steps"]) < 128 else stored
                    ref = from_production(ref_raw, omega)
                    case = len(representative_points)
                    entry = {"case": case, "V_index": iv, "omega_index": iw,
                             "phi_index": ip, "omega": omega, "phi": phi,
                             "reference_steps": max(128, int(stored["steps"])),
                             "production_steps": int(stored["steps"]), "resolutions": []}
                    raw[f"case{case}_reference_Dn"] = ref["Dn"]
                    raw[f"case{case}_reference_IPR"] = ref["IPR"]
                    raw[f"case{case}_reference_quasienergies"] = ref["quasienergies"]
                    raw[f"case{case}_production_sector_vectors"] = stored["sector_vectors"]
                    raw[f"case{case}_production_reconstruction_order"] = stored["reconstruction_order"]
                    for m, value in computed.items():
                        candidate = from_production(value, omega)
                        metrics = compare(ref, candidate, omega)
                        row = {"case": case, "V": float(axes["V"][iv]), "omega": omega,
                               "phi": phi, "steps": m, "reference": f"adaptive_or_M128_{entry['reference_steps']}",
                               **metrics}
                        diagnostic_rows.append(row)
                        entry["resolutions"].append(row)
                        for key in ["sector_vectors", "reconstruction_order", "quasienergies", "parity_labels"]:
                            raw[f"case{case}_M{m}_{key}"] = value[key]
                        raw[f"case{case}_M{m}_Dn"] = candidate["Dn"]
                    if iw == 1 and not args.skip_ode:
                        ode = independent_ode(omega, phi, params)
                        metrics = compare(ode, from_production(stored, omega), omega)
                        entry["independent_ODE"] = {**ode["ode"], "production_comparison": metrics}
                        raw[f"case{case}_ODE_modes"] = ode["modes"]
                        raw[f"case{case}_ODE_quasienergies"] = ode["quasienergies"]
                        raw[f"case{case}_ODE_Dn"] = ode["Dn"]
                        raw[f"case{case}_ODE_IPR"] = ode["IPR"]
                        if metrics["operator_relative_Frobenius_error"] > 2e-6 or metrics["max_matched_Dn_error"] > 3e-4:
                            warnings.append(f"ODE deviation needs inspection at case {case}")
                        for m, value in computed.items():
                            row = {"case": case, "V": float(axes["V"][iv]), "omega": omega,
                                   "phi": phi, "steps": m, "reference": "independent_DOP853",
                                   **compare(ode, from_production(value, omega), omega)}
                            diagnostic_rows.append(row)
                    representative_points.append(entry)
                    print(f"Validated representative {case+1}/24: V={axes['V'][iv]:g}, omega={omega:.6g}, phi/pi={phi/np.pi:g}", flush=True)

    phase_points = []
    for iv in range(3):
        for iw in [1, 49]:
            ip, jp = 7, 43
            left_raw, right_raw = checkpoint(out, iv, ip, iw), checkpoint(out, iv, jp, iw)
            omega = float(axes["omega"][iw])
            left, right = from_production(left_raw, omega), from_production(right_raw, omega)
            metrics = compare(left, right, omega, conjugate_reference=True)
            phase_points.append({"V": float(axes["V"][iv]), "omega": omega,
                                 "phi": float(axes["phi"][ip]), "partner_phi": float(axes["phi"][jp]),
                                 "left_steps": int(left_raw["steps"]), "right_steps": int(right_raw["steps"]),
                                 **metrics})
            if metrics["operator_relative_Frobenius_error"] > 3e-6 or metrics["max_matched_Dn_error"] > 3e-4:
                warnings.append(f"selected independent phase symmetry needs inspection V index {iv}, omega index {iw}")
    report["phase_reversal_selected_modes"] = phase_points
    cross_root = (args.fft_crosscheck_results if args.fft_crosscheck_results is not None
                  else out.with_name(out.name+"_fft_crosscheck"))
    cross_rows = []
    if (cross_root/"points").is_dir():
        for iv in range(3):
            for ip in [0, 12, 25]:
                for iw in frequency_indices:
                    name = f"V{iv}_p{ip:02d}_w{iw:02d}.npz"
                    if not (cross_root/"points"/name).is_file():
                        continue
                    old = checkpoint(cross_root, iv, ip, iw)
                    native = checkpoint(out, iv, ip, iw)
                    omega = float(axes["omega"][iw])
                    metrics = compare(from_production(old, omega), from_production(native, omega), omega)
                    row = {"V": float(axes["V"][iv]), "omega": omega,
                           "phi": float(axes["phi"][ip]), "FFT_steps": int(old["steps"]),
                           "native_steps": int(native["steps"]), **metrics}
                    if metrics["outer_membership_changes"]:
                        a_modes, b_modes = old["modes"], native["modes"]
                        overlap = abs(a_modes.conj().T@b_modes)**2
                        allowed = old["parity_labels"][:, None] == native["parity_labels"][None, :]
                        aa, bb = linear_sum_assignment(np.where(allowed, -overlap, 10.))
                        mapping = np.empty(N, int)
                        mapping[aa] = bb
                        a_outer, b_outer = outer(old["quasienergies"]), outer(native["quasienergies"])
                        lost = a_outer[~np.isin(mapping[a_outer], b_outer)]
                        added = b_outer[~np.isin(b_outer, mapping[a_outer])]
                        midpoint = float(np.mean(np.sort(abs(native["quasienergies"]))[-37:-35]))
                        cluster = np.flatnonzero(abs(abs(native["quasienergies"])-midpoint) <= 1e-8)
                        detail = {
                            "V": float(axes["V"][iv]), "omega": omega, "phi": float(axes["phi"][ip]),
                            "FFT_outer_gap": float(old["outer_selection_gap"]),
                            "native_outer_gap": float(native["outer_selection_gap"]),
                            "FFT_lost_indices": lost.tolist(), "native_added_indices": added.tolist(),
                            "FFT_lost_Dn": old["Dn"][lost].tolist(),
                            "native_added_Dn": native["Dn"][added].tolist(),
                            "boundary_cluster_tolerance": 1e-8,
                            "native_boundary_cluster_indices": cluster.tolist(),
                            "native_boundary_cluster_abs_epsilon": abs(native["quasienergies"][cluster]).tolist(),
                            "native_boundary_cluster_Dn": native["Dn"][cluster].tolist(),
                            "native_matched_FFT_outer_minus_native_hard_outer_Dn": float(
                                native["Dn"][mapping[a_outer]].mean()-native["Dn"][b_outer].mean()),
                            "interpretation": "The hard rank cutoff crosses a near-degenerate |epsilon| cluster. "
                                "Individually matched modes agree, but two distinct members are selected across backends. "
                                "This is a selection sensitivity, not loss of eigenvector agreement or a phase transition.",
                        }
                        report.setdefault("outer_selection_backend_sensitivities", []).append(detail)
                        warnings.append(f"hard outer membership differs by {metrics['outer_membership_changes']} modes "
                                        f"between native and retained FFT at V={axes['V'][iv]:g}, "
                                        f"omega={omega:.9g}, phi/pi={axes['phi'][ip]/np.pi:g}; "
                                        "inspect outer_selection_backend_sensitivities")
                    cross_rows.append(row)
                    k = len(cross_rows)-1
                    for prefix, value in [("FFT", old), ("native", native)]:
                        raw[f"backend_cross{k}_{prefix}_sector_vectors"] = value["sector_vectors"]
                        raw[f"backend_cross{k}_{prefix}_reconstruction_order"] = value["reconstruction_order"]
                        raw[f"backend_cross{k}_{prefix}_IPR"] = value["IPR"]
                        raw[f"backend_cross{k}_{prefix}_Dn"] = value["Dn"]
                    if (metrics["operator_relative_Frobenius_error"] > 3e-6
                            or metrics["max_matched_Dn_error"] > 3e-4):
                        warnings.append(f"native/retained FFT needs inspection at V index {iv}, phase index {ip}, omega index {iw}")
    report["retained_FFT_crosscheck_directory"] = str(cross_root)
    report["retained_FFT_crosschecks"] = cross_rows
    if cross_rows:
        with (out/"frequency_method_crosscheck.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cross_rows[0]))
            writer.writeheader()
            writer.writerows(cross_rows)
    report["representative_convergence"] = representative_points
    report["independent_ODE_points"] = sum("independent_ODE" in x for x in representative_points)
    report["convergence_skipped"] = args.skip_convergence
    report["ODE_skipped"] = args.skip_ode
    report["data_integrity_passed"] = not failures
    report["interpretation"] = ("Integrity and time-discretization accuracy are distinct. Near degeneracy can make individual "
        "IPRs basis-sensitive even with tiny unitarity residuals. Warnings retain observed deviations and do not modify "
        "production data. A finite L scan does not establish a thermodynamic localization transition.")
    report["seconds"] = time.monotonic()-started
    (out/"frequency_scan_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    if diagnostic_rows:
        with (out/"frequency_convergence_diagnostics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(diagnostic_rows[0]))
            writer.writeheader()
            writer.writerows(diagnostic_rows)
    np.savez_compressed(out/"frequency_validation_modes.npz", **raw)
    print(json.dumps({"data_integrity_passed": not failures, "failures": failures,
                      "warnings": warnings, "representative_points": len(representative_points),
                      "seconds": report["seconds"]}, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
