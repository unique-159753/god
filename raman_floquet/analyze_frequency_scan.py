"""Analyze the completed, genuinely computed three-V Raman frequency scan.

No interpolation, synthetic grid, high-frequency replacement, or causal
contribution percentages are used. Incomplete/nonconverged scans are rejected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parent
DEFAULT_RESULTS = BASE/"results"/"frequency_phase_three_V"
EXPECTED_V = np.array([.5, 2., 4.])
EXPECTED_OMEGA = np.linspace(0., 10., 50)
EXPECTED_PHI = 2*np.pi*np.arange(50)/50
PRIMARY_FIELDS = ["all_IPR", "all_Dn", "outer_Dn", "delta_all_Dn",
                  "static_all_IPR", "static_all_Dn", "effective_all_Dn",
                  "effective_outer_Dn", "effective_no_kick_all_Dn",
                  "kick_only_all_Dn", "converged", "is_frozen", "steps"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def rms(values):
    values = np.asarray(values)
    return float(np.sqrt(np.mean(values**2)))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_inputs(out):
    required = ["frequency_phase_scan.npz", "frequency_phase_slices.npz",
                "scan_completion.json", "baselines.npz", "baselines_validation.json"]
    for name in required:
        require((out/name).is_file(), f"Missing completed actual data: {out/name}")
    completion = json.loads((out/"scan_completion.json").read_text())
    require(completion.get("grid_points") == 7500, "Main grid is incomplete")
    require(completion.get("additional_pi_over_2_points") == 150,
            "The true pi/2 line cut is incomplete")
    require(completion.get("nonconverged") == [], "Refuse nonconverged production points")
    baseline_validation = json.loads((out/"baselines_validation.json").read_text())
    require(baseline_validation.get("passed") is True, "Baseline checks did not pass")
    grid = load_npz(out/"frequency_phase_scan.npz")
    slices = load_npz(out/"frequency_phase_slices.npz")
    baselines = load_npz(out/"baselines.npz")
    for label, data, nphi in [("grid", grid, 50), ("slices", slices, 2)]:
        require(np.array_equal(data["V"], EXPECTED_V), f"Unexpected {label} V axis")
        require(np.array_equal(data["omega"], EXPECTED_OMEGA), f"Unexpected {label} omega axis")
        expected_phi = EXPECTED_PHI if nphi == 50 else np.array([0., np.pi/2])
        require(np.array_equal(data["phi"], expected_phi), f"Unexpected {label} phase axis")
        for field in PRIMARY_FIELDS:
            require(field in data, f"Missing {label} field {field}")
            require(data[field].shape == (3, nphi, 50), f"Wrong {label} field shape: {field}")
        require(np.asarray(data["converged"], bool).all(), f"Some {label} points did not converge")
        expected_frozen = np.zeros((3, nphi, 50), bool)
        expected_frozen[:, :, 0] = True
        require(np.array_equal(data["is_frozen"], expected_frozen), f"Wrong {label} omega=0 marker")
        require(np.isfinite(data["all_IPR"]).all() and np.isfinite(data["all_Dn"]).all(),
                f"Nonfinite {label} primary full-spectrum data")
        require(np.isnan(data["outer_Dn"][:, :, 0]).all(),
                f"Undefined {label} zero-frequency Floquet outer value was replaced")
        require(np.isfinite(data["outer_Dn"][:, :, 1:]).all(), f"Nonfinite {label} outer data")
        for field in ["effective_all_Dn", "effective_outer_Dn", "effective_no_kick_all_Dn",
                      "kick_only_all_Dn"]:
            require(np.isfinite(data[field][:, :, 1:]).all(), f"Nonfinite {label} {field}")
        for field in ["IPR", "Dn", "quasienergies"]:
            require(data[field].shape == (3, nphi, 50, 178), f"Missing modal {label} {field}")
        require(np.isnan(data["quasienergies"][:, :, 0]).all(),
                f"Undefined {label} zero-frequency quasienergies were replaced")
        expected_dn = -np.log(data["IPR"])/np.log(89)
        require(np.max(abs(data["Dn"]-expected_dn)) < 2e-11, f"Wrong {label} Dn normalization")
        require(np.max(abs(data["all_Dn"]-data["Dn"].mean(axis=-1))) < 2e-11,
                f"Wrong {label} modal mean Dn")
        require(np.max(abs(data["all_IPR"]-data["IPR"].mean(axis=-1))) < 2e-11,
                f"Wrong {label} modal mean IPR")
        require(np.min(data["IPR"]) >= 1/178-1e-10 and np.max(data["IPR"]) <= 1+1e-10,
                f"Invalid normalized {label} IPR")
        static_dn = baselines["static_all_Dn"][:, None, None]
        static_ipr = baselines["static_all_IPR"][:, None, None]
        require(np.max(abs(data["static_all_Dn"]-static_dn)) < 2e-11,
                f"Mismatched {label} undriven Dn baseline")
        require(np.max(abs(data["static_all_IPR"]-static_ipr)) < 2e-11,
                f"Mismatched {label} undriven IPR baseline")
        require(np.max(abs(data["delta_all_Dn"]-(data["all_Dn"]-static_dn))) < 2e-11,
                f"Wrong {label} undriven difference")
    for field in PRIMARY_FIELDS:
        require(np.allclose(grid[field][:, 0], slices[field][:, 0], equal_nan=True,
                            rtol=0., atol=2e-11), f"Phi=0 grid/slice mismatch: {field}")
    metadata = json.loads(str(grid["metadata"]))
    slice_metadata = json.loads(str(slices["metadata"]))
    require(metadata == slice_metadata, "Grid/slice metadata mismatch")
    for name, expected in metadata.get("code_sha256", {}).items():
        require((BASE/name).is_file(), f"Missing input source {name}")
        require(sha256(BASE/name) == expected, f"Scan source changed after computation: {name}")
    return grid, slices, baselines, completion, metadata


def grid_extreme(values, omega, phi, which="max"):
    if which == "max_abs":
        index = np.unravel_index(np.argmax(abs(values)), values.shape)
    elif which == "max":
        index = np.unravel_index(np.argmax(values), values.shape)
    else:
        index = np.unravel_index(np.argmin(values), values.shape)
    ip, iw = map(int, index)
    return {"omega": float(omega[iw]), "phi": float(phi[ip]),
            "phi_over_pi": float(phi[ip]/np.pi), "value": float(values[ip, iw])}


def response_statistics(values, omega, phi):
    return {"mean": float(np.mean(values)), "rms": rms(values),
            "minimum": grid_extreme(values, omega, phi, "min"),
            "maximum": grid_extreme(values, omega, phi, "max"),
            "largest_absolute": grid_extreme(values, omega, phi, "max_abs")}


def cut_statistics(values, omega):
    largest = int(np.argmax(abs(values)))
    maximum, minimum = int(np.argmax(values)), int(np.argmin(values))
    return {"definition": "phi=pi/2 minus phi=0 at the same positive omega",
            "rms": rms(values), "mean": float(np.mean(values)),
            "maximum": {"omega": float(omega[maximum]), "value": float(values[maximum])},
            "minimum": {"omega": float(omega[minimum]), "value": float(values[minimum])},
            "largest_absolute": {"omega": float(omega[largest]), "value": float(values[largest])}}


def finite_or_none(value):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    value = float(value)
    return value if np.isfinite(value) else None


def analyze(out):
    grid, slices, baselines, completion, metadata = validate_inputs(out)
    omega, phi = grid["omega"], grid["phi"]
    positive, high = omega > 0, omega >= 6
    window_specs = [
        ("all_positive", positive),
        ("low_0_lt_omega_lt_2", (omega > 0) & (omega < 2)),
        ("middle_2_le_omega_lt_6", (omega >= 2) & (omega < 6)),
        ("high_omega_ge_6", high),
    ]
    per_V, rows, rankings = [], [], []
    for iv, V in enumerate(grid["V"]):
        static_dn = float(baselines["static_all_Dn"][iv])
        static_ipr = float(baselines["static_all_IPR"][iv])
        # Explicit [:, mask] preserves the (phase,frequency) orientation.
        delta = grid["delta_all_Dn"][iv][:, positive]
        ipr_change = grid["all_IPR"][iv][:, positive]-static_ipr
        full_cut = slices["all_Dn"][iv, 1, positive]-slices["all_Dn"][iv, 0, positive]
        outer_cut = slices["outer_Dn"][iv, 1, positive]-slices["outer_Dn"][iv, 0, positive]
        dn_field = grid["all_Dn"][iv]
        phase_range = np.ptp(dn_field, axis=0)
        phase_rms = np.sqrt(np.mean((dn_field-dn_field.mean(axis=0, keepdims=True))**2, axis=0))
        spread_peak = int(np.flatnonzero(positive)[np.argmax(phase_range[positive])])
        high_residuals = {}
        for label, exact_key, reference_key in [
            ("all_Dn_exact_minus_effective_with_kick", "all_Dn", "effective_all_Dn"),
            ("outer_Dn_exact_minus_effective_with_kick", "outer_Dn", "effective_outer_Dn"),
            ("all_Dn_exact_minus_effective_without_kick", "all_Dn", "effective_no_kick_all_Dn"),
            ("all_Dn_exact_minus_kick_only_static", "all_Dn", "kick_only_all_Dn"),
        ]:
            diff = (grid[exact_key][iv]-grid[reference_key][iv])[:, high]
            high_residuals[label] = response_statistics(diff, omega[high], phi)
        high_residuals["kick_only_minus_undriven"] = response_statistics(
            grid["kick_only_all_Dn"][iv][:, high]-static_dn, omega[high], phi)
        high_residuals["effective_with_minus_without_kick"] = response_statistics(
            (grid["effective_all_Dn"][iv]-grid["effective_no_kick_all_Dn"][iv])[:, high],
            omega[high], phi)
        high_residuals["effective_without_kick_minus_undriven"] = response_statistics(
            grid["effective_no_kick_all_Dn"][iv][:, high]-static_dn, omega[high], phi)
        high_residuals["effective_with_kick_minus_undriven"] = response_statistics(
            grid["effective_all_Dn"][iv][:, high]-static_dn, omega[high], phi)
        static_energy = baselines["static_energies"][iv]
        zero = dn_field[:, 0]
        zero_delta = zero-static_dn
        zero_max, zero_min = int(np.argmax(zero)), int(np.argmin(zero))
        item = {
            "V": float(V), "static_mean_Dn": static_dn, "static_mean_IPR": static_ipr,
            "positive_grid_points": int(delta.size),
            "delta_all_Dn_vs_undriven": response_statistics(delta, omega[positive], phi),
            "all_IPR_minus_undriven": response_statistics(ipr_change, omega[positive], phi),
            "all_IPR_relative_change": response_statistics(ipr_change/static_ipr, omega[positive], phi),
            "phase_pi_over_2_minus_zero_all_Dn": cut_statistics(full_cut, omega[positive]),
            "phase_pi_over_2_minus_zero_outer_Dn": cut_statistics(outer_cut, omega[positive]),
            "positive_phase_spread": {
                "mean_range": float(np.mean(phase_range[positive])),
                "rms_range": rms(phase_range[positive]),
                "mean_RMS_about_phase_mean": float(np.mean(phase_rms[positive])),
                "global_RMS_about_phase_mean": rms(dn_field[:, positive]-dn_field[:, positive].mean(axis=0, keepdims=True)),
                "maximum_range": {"omega": float(omega[spread_peak]), "value": float(phase_range[spread_peak])}},
            "static_spectral_scale": {
                "minimum_energy": float(np.min(static_energy)),
                "maximum_energy": float(np.max(static_energy)),
                "bandwidth": float(np.ptp(static_energy)),
                "omega_ge_6_samples_not_above_static_bandwidth": int(np.count_nonzero(high & (omega <= np.ptp(static_energy)))),
                "interpretation": "omega>=6 is a reporting window; it is not uniformly asymptotic across V. Being within the bare bandwidth permits but does not demonstrate a one-photon resonance."},
            "high_frequency_residuals": high_residuals,
            "frozen_omega_zero": {
                "definition": "independent frozen Hamiltonians; no Floquet outer statistic",
                "floquet_outer_Dn": None,
                "phase_mean_Dn": float(np.mean(zero)), "phase_range_Dn": float(np.ptp(zero)),
                "minimum_Dn": {"phi": float(phi[zero_min]), "phi_over_pi": float(phi[zero_min]/np.pi), "value": float(zero[zero_min])},
                "maximum_Dn": {"phi": float(phi[zero_max]), "phi_over_pi": float(phi[zero_max]/np.pi), "value": float(zero[zero_max])},
                "mean_delta_vs_undriven": float(np.mean(zero_delta)),
                "rms_delta_vs_undriven": rms(zero_delta),
                "phi0_Dn": float(slices["all_Dn"][iv, 0, 0]),
                "phi_pi_over_2_Dn": float(slices["all_Dn"][iv, 1, 0]),
                "phi_pi_over_2_minus_zero_Dn": float(slices["all_Dn"][iv, 1, 0]-slices["all_Dn"][iv, 0, 0]),
            },
        }
        per_V.append(item)
        for iw, w in enumerate(omega):
            dn, ipr = dn_field[:, iw], grid["all_IPR"][iv, :, iw]
            dd = dn-static_dn
            peak = int(np.argmax(abs(dd)))
            row = dict(V=float(V), omega=float(w), is_frozen=bool(w == 0),
                static_mean_Dn=static_dn, static_mean_IPR=static_ipr,
                phase_mean_Dn=float(np.mean(dn)), minimum_Dn=float(np.min(dn)), maximum_Dn=float(np.max(dn)),
                phase_range_Dn=float(phase_range[iw]), phase_RMS_Dn=float(phase_rms[iw]),
                mean_delta_Dn=float(np.mean(dd)), rms_delta_Dn=rms(dd),
                maximum_abs_delta_Dn=float(abs(dd[peak])), signed_peak_delta_Dn=float(dd[peak]),
                peak_delta_phi=float(phi[peak]), peak_delta_phi_over_pi=float(phi[peak]/np.pi),
                phase_mean_IPR=float(np.mean(ipr)), phase_range_IPR=float(np.ptp(ipr)),
                mean_relative_IPR_change=float(np.mean((ipr-static_ipr)/static_ipr)),
                rms_relative_IPR_change=rms((ipr-static_ipr)/static_ipr),
                phi_pi_over_2_minus_zero_all_Dn=float(slices["all_Dn"][iv, 1, iw]-slices["all_Dn"][iv, 0, iw]),
                phi_pi_over_2_minus_zero_outer_Dn=None,
                outer_phase_mean_Dn=None, outer_phase_range_Dn=None, outer_phase_RMS_Dn=None,
                rms_all_exact_minus_effective=None, max_abs_all_exact_minus_effective=None,
                rms_outer_exact_minus_effective=None, max_abs_outer_exact_minus_effective=None,
                rms_all_exact_minus_effective_no_kick=None, rms_all_exact_minus_kick_only=None,
                mean_kick_only_minus_undriven=None,
                minimum_steps=int(np.min(grid["steps"][iv, :, iw])),
                maximum_steps=int(np.max(grid["steps"][iv, :, iw])))
            if w > 0:
                outer = grid["outer_Dn"][iv, :, iw]
                reff = dn-grid["effective_all_Dn"][iv, :, iw]
                rout = outer-grid["effective_outer_Dn"][iv, :, iw]
                row.update(phi_pi_over_2_minus_zero_outer_Dn=float(slices["outer_Dn"][iv, 1, iw]-slices["outer_Dn"][iv, 0, iw]),
                    outer_phase_mean_Dn=float(np.mean(outer)), outer_phase_range_Dn=float(np.ptp(outer)),
                    outer_phase_RMS_Dn=rms(outer-np.mean(outer)),
                    rms_all_exact_minus_effective=rms(reff), max_abs_all_exact_minus_effective=float(np.max(abs(reff))),
                    rms_outer_exact_minus_effective=rms(rout), max_abs_outer_exact_minus_effective=float(np.max(abs(rout))),
                    rms_all_exact_minus_effective_no_kick=rms(dn-grid["effective_no_kick_all_Dn"][iv, :, iw]),
                    rms_all_exact_minus_kick_only=rms(dn-grid["kick_only_all_Dn"][iv, :, iw]),
                    mean_kick_only_minus_undriven=float(np.mean(grid["kick_only_all_Dn"][iv, :, iw]-static_dn)))
            rows.append(row)

    for name, mask in window_specs:
        scores = []
        for iv, V in enumerate(grid["V"]):
            dd = grid["delta_all_Dn"][iv][:, mask]
            scores.append({"V": float(V), "rms_delta_Dn": rms(dd),
                           "mean_delta_Dn": float(np.mean(dd)),
                           "maximum_abs_delta_Dn": float(np.max(abs(dd))),
                           "mean_phase_range_Dn": float(np.mean(np.ptp(grid["all_Dn"][iv][:, mask], axis=0))),
                           "phase_centered_RMS_Dn": rms(grid["all_Dn"][iv][:, mask]-grid["all_Dn"][iv][:, mask].mean(axis=0, keepdims=True))})
        rankings.append({"window": name, "sampled_omega_min": float(omega[mask].min()),
                         "sampled_omega_max": float(omega[mask].max()), "frequency_points": int(mask.sum()),
                         "scores": scores,
                         "descending_V_by_rms_delta_Dn": [x["V"] for x in sorted(scores, key=lambda x:x["rms_delta_Dn"], reverse=True)],
                         "descending_V_by_mean_phase_range": [x["V"] for x in sorted(scores, key=lambda x:x["mean_phase_range_Dn"], reverse=True)]})
    by_frequency = []
    for iw, w in enumerate(omega):
        here = [row for row in rows if row["omega"] == float(w)]
        by_frequency.append({"omega": float(w), "is_frozen": bool(w == 0),
            "descending_V_by_rms_delta_Dn": [x["V"] for x in sorted(here, key=lambda x:x["rms_delta_Dn"], reverse=True)],
            "descending_V_by_phase_range_Dn": [x["V"] for x in sorted(here, key=lambda x:x["phase_range_Dn"], reverse=True)]})
    summary = {
        "scope": "only completed genuine time-ordered exact Floquet data; zero frequency separate",
        "primary_IPR": "spin-resolved sum |psi_jspin|^4",
        "primary_Dn": "-ln(IPR)/ln89, not ln178; values above one are allowed",
        "positive_frequency_definition": "omega>0; omega=0 excluded from dynamic aggregate/RMS",
        "high_frequency_definition": "sampled dimensionless omega>=6",
        "high_frequency_caution": "This common plotting/statistics window need not be asymptotic for every V; static bandwidth is supplied per V.",
        "phase_spread_definition": "max_phi minus min_phi and RMS about own phase mean at each omega",
        "residual_definition": "exact minus specified effective/control observable; not additive causal fractions",
        "delta_definition": "mean of individual exact Dn minus mean of individual undriven H0 Dn for the same V",
        "statistical_scope": "structured finite-size grid, no thermodynamic transition or significance inferred",
        "frequency_calibration": "dimensionless omega/t0; no fixed Hz equivalence is assumed",
        "number_of_V": 3, "unique_phase_points": 50, "frequency_points": 50,
        "positive_points_per_V": 2450, "pi_over_2_slice": "independently computed, not interpolated",
        "per_V": per_V, "window_rankings": rankings, "frequency_rankings": by_frequency,
        "scan_completion": completion, "source_metadata": metadata,
        "input_sha256": {name:sha256(out/name) for name in
                         ["frequency_phase_scan.npz", "frequency_phase_slices.npz", "baselines.npz"]},
        "analyzer_sha256": sha256(Path(__file__)),
    }
    (out/"analysis_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with (out/"response_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    brief = [{"V": item["V"],
              "positive_rms_delta_all_Dn": item["delta_all_Dn_vs_undriven"]["rms"],
              "largest_delta_all_Dn": item["delta_all_Dn_vs_undriven"]["maximum"],
              "smallest_delta_all_Dn": item["delta_all_Dn_vs_undriven"]["minimum"],
              "phi_pi2_minus_zero_all_Dn_RMS": item["phase_pi_over_2_minus_zero_all_Dn"]["rms"],
              "high_all_exact_effective_RMS": item["high_frequency_residuals"]["all_Dn_exact_minus_effective_with_kick"]["rms"]}
             for item in per_V]
    print(json.dumps(brief, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    analyze(args.results)


if __name__ == "__main__":
    main()
