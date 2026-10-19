#!/usr/bin/env python3
"""Supplementary selectivity controls, using existing computed NPZ/CSV data."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/floquet-folding-selectivity-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/floquet-folding-selectivity-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter
import numpy as np


PARAMETERS = (r"$L=89$, $\alpha=55/89$, $t_0=1$, $\Delta_0=0.5$, $V=2.5$, $W=0.01$, "
              r"$A=B=0.8$, $\theta=0$, PBC")


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


def load_csv(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values) -> dict:
    values = np.asarray(values)
    return {"min": float(np.nanmin(values)), "max": float(np.nanmax(values)),
            "mean": float(np.nanmean(values))}


def frequency_edges(centers: np.ndarray) -> np.ndarray:
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
    return edges


def main(outdir: str | Path | None = None) -> dict:
    started = time.perf_counter()
    output = Path(outdir).resolve() if outdir is not None else Path(__file__).resolve().parent
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 11,
        "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8, "mathtext.fontset": "dejavusans",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 350,
        "figure.dpi": 130, "grid.alpha": 0.22, "lines.linewidth": 1.7,
    })
    tags = ("0", "pi4", "pi2")
    data = {tag: load_npz(output / f"curves_phi_{tag}.npz") for tag in tags}
    csv_data = {tag: load_csv(output / f"curves_phi_{tag}.csv") for tag in tags}
    omega = data["0"]["omega"]
    for tag in tags:
        if not np.array_equal(omega, data[tag]["omega"]) or not np.allclose(omega, csv_data[tag]["omega"], atol=1e-12, rtol=0):
            raise ValueError("Supplementary comparisons require the same frequency samples")
    selected = (omega >= 7.0 - 1e-10) & (omega <= 10.0 + 1e-10)
    x = omega[selected]
    static = load_npz(output / "static_spectrum.npz")
    omega_fold = float(np.asarray(static["omega_fold"]))
    outer = data["pi2"]["metrics"][:, 0, 1] - data["0"]["metrics"][:, 0, 1]
    central = csv_data["pi2"]["exact_D_central"] - csv_data["0"]["exact_D_central"]

    # Validate the independently stored central means using the stated 36-state rule.
    central_reconstruction_error = {}
    for tag in ("0", "pi2"):
        indices = np.argsort(np.abs(data[tag]["q_exact"]), axis=1)[:, :36]
        reconstructed = np.take_along_axis(data[tag]["D_exact"], indices, axis=1).mean(axis=1)
        error = float(np.max(abs(reconstructed - csv_data[tag]["exact_D_central"])))
        if error > 1e-10:
            raise ValueError(f"Central-state definition differs between NPZ and CSV for phase {tag}: {error}")
        central_reconstruction_error[tag] = error

    edges = data["0"]["qbin_edges"]
    if not np.array_equal(edges, data["pi2"]["qbin_edges"]):
        raise ValueError("Fixed-q bins must agree between phases")
    d0, dp = data["0"]["qbin_D"][:, 0], data["pi2"]["qbin_D"][:, 0]
    c0, cp = data["0"]["qbin_counts"][:, 0], data["pi2"]["qbin_counts"][:, 0]
    valid = (c0 > 0) & (cp > 0) & np.isfinite(d0) & np.isfinite(dp)
    response = np.where(valid, dp - d0, np.nan)
    plotted_response = np.ma.masked_invalid(response[selected].T)
    limit = max(float(np.nanmax(abs(response[selected]))), np.finfo(float).eps)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.6))
    fig.subplots_adjust(left=0.085, right=0.95, top=0.89, bottom=0.22, hspace=0.40, wspace=0.38)
    ax = axes[0, 0]
    ax.plot(x, outer[selected], color="#4166a2", label="Outer 20%")
    ax.plot(x, central[selected], color="#d48823", label="Central 20%")
    ax.axhline(0, color="0.55", lw=0.8)
    ax.set(title="(a) Exact Floquet phase response", xlabel=r"Drive frequency $\omega/t_0$",
           ylabel=r"$\overline{D_n}(\pi/2)-\overline{D_n}(0)$", xlim=(7, 10))
    ax.legend(frameon=False)
    ax.grid(True)

    ax = axes[0, 1]
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.80")
    mesh = ax.pcolormesh(frequency_edges(x), edges, plotted_response, cmap=cmap,
                         norm=norm, shading="flat", rasterized=False)
    ax.set(title="(b) Exact response in fixed quasienergy bins", xlabel=r"Drive frequency $\omega/t_0$",
           ylabel=r"Fixed bin in $|q_n|$", xlim=(7, 10), ylim=(edges[0], edges[-1]))
    ax.set_yticks(edges)
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.047, pad=0.035)
    colorbar.set_label(r"$\Delta\overline{D_n}$ ($\pi/2-0$)")
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)

    ax = axes[1, 0]
    control = csv_data["pi2"]
    ax.plot(x, control["J_dynamic_D_outer"][selected], color="#4166a2", label="Outer: dynamic folded set")
    ax.plot(x, control["J_dynamic_D_central"][selected], color="#d48823", label="Central: dynamic folded set")
    ax.plot(x, control["J_matched_D_outer"][selected], color="0.25", ls="--", lw=1.1,
            label="Outer: overlap-matched states")
    ax.axhline(0, color="0.55", lw=0.8)
    ax.set(title=r"(c) Generated-$J_2^x$ control at $\varphi=\pi/2$",
           xlabel=r"Drive frequency $\omega/t_0$", ylabel=r"$\overline{D_n}(g_F)-\overline{D_n}(0)$", xlim=(7, 10))
    ax.legend(frameon=False, fontsize=7.6)
    ax.grid(True)

    ax = axes[1, 1]
    phase_labels = (r"$\varphi=0$", r"$\varphi=\pi/4$", r"$\varphi=\pi/2$")
    for tag, label, color in zip(tags, phase_labels, ("#4166a2", "#d48823", "#258c76")):
        ax.plot(omega, csv_data[tag]["kickbasis_participation_outer"], color=color, label=label)
    ax.axhline(1, color="0.5", lw=0.9)
    ax.axvline(omega_fold, color="0.25", ls=(0, (3, 2)), lw=0.9)
    ax.set_yscale("log")
    ax.set_yticks((1, 2, 5, 10, 20))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set(title="(d) Mixing in the kick-corrected effective basis",
           xlabel=r"Drive frequency $\omega/t_0$", ylabel=r"Mean participation number $\overline{P}_{\mathrm{eff}}$ (outer)",
           xlim=(5.8, 10), ylim=(0.95, 23))
    ax.legend(frameon=False)
    ax.grid(True, which="major")
    fig.suptitle("Quasienergy-selective response and effective-model controls", fontsize=14, y=0.97)
    fig.text(0.5, 0.015, "\n".join((
        PARAMETERS,
        r"Outer / central sets: largest / smallest 36 values of $|q_n|$ at each point. (b) Actual fixed-bin means; empty bins are gray and counts are exported to CSV.",
        r"(c) $H_0+g_FJ_2^x$ versus $H_0$, with the same kick in both: a truncated-model contrast, not exclusive causal proof for exact Floquet dynamics.",
        r"(d) $P_{\mathrm{eff}}=[\sum_m|\langle u_m^{\mathrm{eff,kick}}|\psi_n^{\mathrm{exact}}\rangle|^4]^{-1}$; one means a single basis vector. Dashed vertical line: static folding reference."
    )), ha="center", va="bottom", fontsize=8, linespacing=1.6)

    files = []
    for suffix in ("png", "pdf"):
        path = output / f"figS1_selectivity_controls.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        files.append(path.name)
    plt.close(fig)

    rows = []
    for index, frequency in enumerate(omega):
        rows.append({"omega": float(frequency), "in_7_to_10_window": int(selected[index]),
                     "exact_outer_phase_response_pi2_minus_0": float(outer[index]),
                     "exact_central_phase_response_pi2_minus_0": float(central[index]),
                     "J_dynamic_D_outer_pi2": float(control["J_dynamic_D_outer"][index]),
                     "J_dynamic_D_central_pi2": float(control["J_dynamic_D_central"][index]),
                     "J_matched_D_outer_pi2": float(control["J_matched_D_outer"][index]),
                     **{f"kickbasis_participation_outer_phi_{tag}": float(csv_data[tag]["kickbasis_participation_outer"][index])
                        for tag in tags}})
    write_csv(output / "supplemental_selectivity_responses.csv", rows)
    rows = []
    for index in np.flatnonzero(selected):
        for bin_index in range(len(edges) - 1):
            good = bool(valid[index, bin_index])
            rows.append({"omega": float(omega[index]), "absq_min": float(edges[bin_index]),
                         "absq_max": float(edges[bin_index + 1]),
                         "count_phi_0": int(c0[index, bin_index]), "count_phi_pi2": int(cp[index, bin_index]),
                         "D_phi_0": float(d0[index, bin_index]), "D_phi_pi2": float(dp[index, bin_index]),
                         "phase_response_pi2_minus_0": float(response[index, bin_index]), "valid_both_phases": int(good)})
    write_csv(output / "supplemental_qbin_response.csv", rows)
    summary = {
        "files": files, "png_dpi": 350, "interpolation": "None; fixed-bin means shown as constant-color cells.",
        "frequency_window_for_panels_abc": [7, 10],
        "exact_outer_phase_response": stats(outer[selected]),
        "exact_central_phase_response": stats(central[selected]),
        "central_reconstruction_max_error": central_reconstruction_error,
        "fixed_qbin_phase_response": stats(response[selected]),
        "fixed_qbin_masked_cell_count": int(np.count_nonzero(~valid[selected])),
        "J_dynamic_D_outer": stats(control["J_dynamic_D_outer"][selected]),
        "J_dynamic_D_central": stats(control["J_dynamic_D_central"][selected]),
        "J_matched_D_outer": stats(control["J_matched_D_outer"][selected]),
        "kickbasis_participation_outer_all_frequencies": {
            tag: stats(csv_data[tag]["kickbasis_participation_outer"]) for tag in tags},
        "kickbasis_participation_outer_7_to_10": {
            tag: stats(csv_data[tag]["kickbasis_participation_outer"][selected]) for tag in tags},
        "pdf_image_xobjects": (output / files[1]).read_bytes().count(b"/Subtype /Image"),
        "plotting_wall_seconds": time.perf_counter() - started,
    }
    (output / "supplemental_plots_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path)
    main(parser.parse_args().outdir)
