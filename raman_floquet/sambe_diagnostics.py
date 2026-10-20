"""Reproduce the Sambe zero-drive check and static-photon channel summaries."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm

from model import (Parameters, operators, static_eigensystem, fold_quasienergies,
                   localization_observables)
from sambe import solve_sambe


def zero_drive_check(output: Path):
    params = Parameters(A=0., B=0.)
    omega = 6.2
    E, V = static_eigensystem(params)
    eps = fold_quasienergies(E, omega)
    order = np.argsort(eps)
    exact = {"quasienergies": eps[order], "modes": V[:, order],
             "operator": expm(-2j*np.pi*operators(params)["H0"]/omega)}
    result = solve_sambe(omega, np.pi/3, 1, exact=exact, params=params)
    summary = result["summary"]
    checks = {"scope": "Independent zero-drive Sambe gauge/reconstruction limit",
              "raw_central_candidates": summary["raw_central_zone_candidate_count"],
              "minimum_overlap": summary["minimum_mode_overlap"],
              "max_quasienergy_error": summary["max_quasienergy_error"],
              "max_abs_D_difference": float(np.max(abs(result["D_site"]-result["exact_D_site"]))),
              "max_participation_error": float(np.max(abs(result["photon_participation"]-1))),
              "photon_weight_sum_error": float(np.max(abs(result["photon_weights"].sum(axis=1)-1))),
              "static_photon_sum_error": float(np.max(abs(result["static_photon_weights"].sum(axis=(1,2))-1))),
              "operator_residual": summary["exact_operator_mode_residual"]}
    checks["passed"] = bool(checks["raw_central_candidates"] == 178 and
                             checks["minimum_overlap"] > 1-1e-10 and
                             checks["max_quasienergy_error"] < 1e-10 and
                             checks["max_abs_D_difference"] < 1e-10 and
                             checks["max_participation_error"] < 1e-10)
    (output/"sambe_validation.json").write_text(json.dumps(checks, indent=2)+"\n")
    if not checks["passed"]:
        raise RuntimeError("Zero-drive Sambe check failed")
    return checks


def analyze_outer_static_photon(output: Path):
    E, V = static_eigensystem()
    static = localization_observables(V)
    rho = static["density"]
    payload = []
    for point in range(4):
        with np.load(output/f"sambe_point{point}_M3.npz") as z:
            outer = z["outer_indices_exact_reference"]
            weights, axis = z["static_photon_weights"], z["photon_axis"]
            omega, phi = float(z["omega"]), float(z["phi"])
            rows = []
            for alpha in outer:
                flat = weights[alpha].ravel()
                top = np.argsort(flat)[-8:][::-1]
                q, n = np.unravel_index(top, weights[alpha].shape)
                i, j = int(n[0]), int(n[1])
                dp = int(axis[q[1]]-axis[q[0]])
                row = {"mode": int(alpha), "quasienergy_exact": float(z["exact_quasienergies"][alpha]),
                       "D_exact": float(z["exact_D_site"][alpha]),
                       "photon_participation": float(z["photon_participation"][alpha]),
                       "top2_weight": float(flat[top[:2]].sum()),
                       "static_n1": i, "static_n2": j, "photon_m1": int(axis[q[0]]),
                       "photon_m2": int(axis[q[1]]), "delta_photon": dp,
                       "static_E1": float(E[i]), "static_E2": float(E[j]),
                       "sambe_detuning": float(E[j]-E[i]+dp*omega),
                       "static_D1": float(static["D_site"][i]), "static_D2": float(static["D_site"][j]),
                       "density_bhattacharyya": float(np.sum(np.sqrt(rho[:, i]*rho[:, j]))),
                       "top_terms": [{"photon": int(axis[qi]), "static_n": int(ni),
                                      "E": float(E[ni]), "weight": float(flat[ti])}
                                     for ti, qi, ni in zip(top, q, n)]}
                rows.append(row)
            summary = {"point": point, "omega": omega, "phi": phi,
                       "mean_top2_weight": float(np.mean([r["top2_weight"] for r in rows])),
                       "fraction_top2_one_photon": float(np.mean([abs(r["delta_photon"]) == 1 for r in rows])),
                       "fraction_top2_resonant_within_0p1": float(np.mean([
                           abs(r["sambe_detuning"]) < .1 and abs(r["delta_photon"]) == 1 for r in rows])),
                       "mean_top_pair_density_overlap": float(np.mean([r["density_bhattacharyya"] for r in rows])),
                       "mean_pair_static_D": float(np.mean([.5*(r["static_D1"]+r["static_D2"]) for r in rows])),
                       "modes": rows}
            payload.append(summary)
    (output/"sambe_outer_static_photon_analysis.json").write_text(json.dumps(payload, indent=2)+"\n")
    return payload


if __name__ == "__main__":
    output = Path(__file__).parent/"results"
    print(json.dumps(zero_drive_check(output), indent=2))
    analyses = analyze_outer_static_photon(output)
    print(json.dumps([{k:v for k,v in p.items() if k!="modes"} for p in analyses], indent=2))
