"""Figures 1/2 and descriptive correlations from computed Raman Floquet data.

No localization solver, interpolation, or image-generation model is used here.
All plots consume the completed, explicitly aligned numerical grids.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

_CACHE = Path(__file__).resolve().parent / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import pearsonr, spearmanr


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def phase_centered(values):
    """Subtract each frequency column's phase average, not a global mean."""
    values = np.asarray(values, dtype=float)
    return values - np.mean(values, axis=0, keepdims=True)


def coefficients(x, y):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": int(len(x)), "pearson": None, "spearman": None,
                "status": "insufficient variation"}
    return {"n": int(len(x)), "pearson": float(pearsonr(x, y).statistic),
            "spearman": float(spearmanr(x, y).statistic), "status": "computed"}


def save_figure(fig, output_dir, stem):
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def coordinate(array, omega, phi, mode="max"):
    index = np.unravel_index(np.argmax(array) if mode == "max" else np.argmin(array), array.shape)
    ip, iw = map(int, index)
    return {"index_phi": ip, "index_omega": iw, "omega": float(omega[iw]),
            "phi": float(phi[ip]), "phi_over_pi": float(phi[ip] / np.pi),
            "value": float(array[ip, iw])}


def set_axes(ax, omega, phi, xlabel=True, ylabel=True):
    ax.set_xlim(float(omega[0]), float(omega[-1]))
    ax.set_ylim(float(phi[0] / np.pi), float(phi[-1] / np.pi))
    if xlabel:
        ax.set_xlabel(r"$\omega/t_0$")
    if ylabel:
        ax.set_ylabel(r"$\varphi/\pi$")


def fold_line(ax, reference):
    ax.axvline(reference, color="0.22", linewidth=1.1, linestyle="--", alpha=.9)


def mark_peaks(ax, resonance_peak, delta_peak):
    ax.plot(resonance_peak["omega"], resonance_peak["phi_over_pi"], "*",
            markersize=11, markerfacecolor="white", markeredgecolor="#17458a",
            markeredgewidth=1.2, zorder=5, clip_on=False)
    ax.plot(delta_peak["omega"], delta_peak["phi_over_pi"], "D",
            markersize=6, markerfacecolor="white", markeredgecolor="#be342b",
            markeredgewidth=1.2, zorder=6, clip_on=False)


def render_main_figure(output_dir, grid, resonance, eta_index, reference, peaks):
    omega, phi = grid["omega"], grid["phi"]
    delta = grid["delta_D_site"]
    R, R_outer = resonance["R_full"][eta_index], resonance["R_outer"][eta_index]
    response_norm = colors.LogNorm(vmin=float(min(R.min(), R_outer.min())),
                                   vmax=float(max(R.max(), R_outer.max())))
    amplitude = max(float(np.max(np.abs(delta))), np.finfo(float).eps)
    difference_norm = colors.TwoSlopeNorm(vmin=-amplitude, vcenter=0, vmax=amplitude)
    # Reserve separate rows for the title, shared legend, panels, and caption.
    # An outside legend together with constrained_layout can overlap a suptitle
    # and leave an unnecessarily large blank row above these short panels.
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.85))
    fig.subplots_adjust(left=.052, right=.973, bottom=.175, top=.78, wspace=.42)
    quantities = [(R, response_norm, "viridis", r"(a) Full static spectrum", r"$\mathcal{R}$"),
                  (R_outer, response_norm, "viridis", r"(b) At least one static-outer state", r"$\mathcal{R}_{\rm outer}$"),
                  (delta, difference_norm, "RdBu_r", r"(c) Exact $-$ kick-corrected effective", r"$\Delta\overline{D}_{\rm outer}$")]
    for i, (ax, (values, norm, cmap, title, label)) in enumerate(zip(axes, quantities)):
        im = ax.pcolormesh(omega, phi / np.pi, values, shading="nearest", cmap=cmap, norm=norm)
        fold_line(ax, reference)
        mark_peaks(ax, peaks["R_full"], peaks["delta"])
        set_axes(ax, omega, phi, ylabel=i == 0)
        ax.set_title(title, fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.025)
        cb.set_label(label)
    handles = [Line2D([], [], marker="*", markersize=9, color="none", markerfacecolor="white",
                      markeredgecolor="#17458a", label=r"maximum $\mathcal{R}$"),
               Line2D([], [], marker="D", markersize=5, color="none", markerfacecolor="white",
                      markeredgecolor="#be342b", label=r"maximum $\Delta\overline{D}_{\rm outer}$"),
               Line2D([], [], color=".22", linestyle="--", label=r"$2\max|E_n|=6.5968t_0$")]
    fig.suptitle(r"$L=89$, $\alpha=55/89$, $A=B=0.8t_0$, $\eta=0.05t_0$; PBC", y=.985, fontsize=11)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .925),
               ncol=3, frameon=False)
    fig.text(.5, .035, r"Site IPR; $D=-\ln({\rm IPR}_{\rm site})/\ln89$. Outer: 36 largest $|\epsilon|$ after zone folding. $\mathcal{R}$ is a diagnostic, not a transition rate.",
             ha="center", fontsize=8)
    save_figure(fig, output_dir, "fig1_resonance_vs_localization")


def render_correlation_figure(output_dir, grid, slices, resonance, resonance_slices,
                              eta_index, reference, primary):
    omega = grid["omega"]
    R = resonance["R_full"][eta_index]
    delta = grid["delta_D_site"]
    R_centered, delta_centered = phase_centered(R), phase_centered(delta)
    Rs = resonance_slices["R_full"][eta_index]
    deltas = slices["delta_D_site"]
    phase_labels = [r"$0$", r"$\pi/4$", r"$\pi/3$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]
    palette = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), constrained_layout=True)
    for ip, label in enumerate(phase_labels):
        axes[0, 0].plot(omega, Rs[ip], color=palette[ip], label=label, linewidth=1.45)
        axes[0, 1].plot(omega, deltas[ip], color=palette[ip], label=label, linewidth=1.45)
    for ax in axes[0]:
        fold_line(ax, reference)
        ax.set_xlim(omega[0], omega[-1])
        ax.set_xlabel(r"$\omega/t_0$")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel(r"$\mathcal{R}$ (full spectrum; log scale)")
    axes[0, 1].set_ylabel(r"$\Delta\overline{D}_{\rm outer}$")
    axes[0, 1].axhline(0, color=".65", linewidth=.7)
    axes[0, 0].set_title(r"(a) One-photon response, $\eta/t_0=0.05$")
    axes[0, 1].set_title("(b) Localization difference; independent vertical scale")
    axes[0, 0].legend(title=r"$\varphi$", ncol=3, frameon=False, fontsize=9)
    frequency_colors = np.broadcast_to(omega[None, :], R.shape).ravel()
    scatter = axes[1, 0].scatter(R.ravel(), delta.ravel(), c=frequency_colors,
                                cmap="plasma", s=3, alpha=.42, linewidths=0)
    axes[1, 1].scatter(R_centered.ravel(), delta_centered.ravel(), c=frequency_colors,
                       cmap="plasma", s=3, alpha=.42, linewidths=0)
    axes[1, 0].set_xlabel(r"$\mathcal{R}$")
    axes[1, 0].set_ylabel(r"$\Delta\overline{D}_{\rm outer}$")
    axes[1, 1].set_xlabel(r"$\mathcal{R}-\langle\mathcal{R}\rangle_\varphi(\omega)$")
    axes[1, 1].set_ylabel(r"$\Delta\overline{D}-\langle\Delta\overline{D}\rangle_\varphi(\omega)$")
    for ax, key, title in [(axes[1, 0], "raw", "(c) All grid points"),
                           (axes[1, 1], "phase_centered", "(d) Frequency-wise phase means removed")]:
        values = primary[key]
        ax.set_title(title)
        ax.text(.03, .97, f"Pearson r = {values['pearson']:.3f}\nSpearman ρ = {values['spearman']:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": .87, "edgecolor": "none"})
        ax.axhline(0, color=".75", linewidth=.6, zorder=0)
    fig.colorbar(scatter, ax=list(axes[1]), label=r"$\omega/t_0$", fraction=.025, pad=.02)
    fig.text(.5, -.025, r"All curves use numerical values without amplitude rescaling. The $\pi/3$ cut is calculated separately, not interpolated. Correlations are descriptive on a structured grid.",
             ha="center", fontsize=8)
    save_figure(fig, output_dir, "fig2_resonance_correlation")


def render_branch_figure(output_dir, grid, reference):
    keys = ["delta_D_site", "matched_delta_D_site", "selection_delta_D_site", "projected_delta_D_site"]
    titles = ["(a) Independent outer selections", "(b) Matched-mode contribution",
              "(c) Selection contribution", "(d) Static-outer projector weighted"]
    amplitude = max(float(np.max(np.abs(grid[key]))) for key in keys)
    amplitude = max(amplitude, np.finfo(float).eps)
    norm = colors.TwoSlopeNorm(vmin=-amplitude, vcenter=0, vmax=amplitude)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.7), constrained_layout=True)
    for ax, key, title in zip(axes.ravel(), keys, titles):
        im = ax.pcolormesh(grid["omega"], grid["phi"] / np.pi, grid[key],
                           cmap="RdBu_r", norm=norm, shading="nearest")
        fold_line(ax, reference)
        set_axes(ax, grid["omega"], grid["phi"])
        ax.set_title(title)
    fig.colorbar(im, ax=list(axes.ravel()), label=r"$\Delta\overline{D}_{\rm site}$", fraction=.025, pad=.02)
    fig.text(.5, -.025, r"Exact identity: (a) = (b) + (c). Matching maximizes lab-frame mode overlaps. (d) weights all modes by a fixed static-outer projector; it has no hard branch labels.",
             ha="center", fontsize=8)
    save_figure(fig, output_dir, "figS_branch_selection_diagnostics")


def render_eta_figure(output_dir, resonance, reference):
    omega, phi, eta = resonance["omega"], resonance["phi"], resonance["eta"]
    values = [resonance["R_full"], resonance["R_outer"]]
    norm = colors.LogNorm(vmin=min(float(v.min()) for v in values),
                          vmax=max(float(v.max()) for v in values))
    fig, axes = plt.subplots(2, len(eta), figsize=(13.5, 7), constrained_layout=True)
    for row, kind in enumerate(["full", "outer"]):
        for ie, width in enumerate(eta):
            ax = axes[row, ie]
            im = ax.pcolormesh(omega, phi / np.pi, resonance[f"R_{kind}"][ie],
                               norm=norm, cmap="viridis", shading="nearest")
            fold_line(ax, reference)
            set_axes(ax, omega, phi, ylabel=ie == 0)
            ax.set_title(rf"{'All states' if row == 0 else 'Involves static outer'}; $\eta/t_0={width:g}$")
    fig.colorbar(im, ax=list(axes.ravel()), label=r"$\mathcal{R}$ (shared logarithmic color scale)", fraction=.022, pad=.02)
    fig.text(.5, -.02, r"The broadening is a diagnostic choice. $\mathcal{R}_{\rm norm}=\eta^2\mathcal{R}$ is exported; it is not a transition probability and gives the same fixed-$\eta$ correlations.",
             ha="center", fontsize=8)
    save_figure(fig, output_dir, "figS_broadening_sensitivity")


def run(output_dir):
    output_dir = Path(output_dir)
    grid = load_npz(output_dir / "localization_grid.npz")
    slices = load_npz(output_dir / "localization_slices.npz")
    resonance = load_npz(output_dir / "resonance_grid.npz")
    resonance_slices = load_npz(output_dir / "resonance_slices.npz")
    resonance_summary = json.loads((output_dir / "resonance_summary.json").read_text())
    omega, phi = grid["omega"], grid["phi"]
    expected = (len(phi), len(omega))
    np.testing.assert_allclose(omega, resonance["omega"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(phi, resonance["phi"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(slices["omega"], omega, rtol=0, atol=1e-13)
    np.testing.assert_allclose(slices["phi"], resonance_slices["phi"], rtol=0, atol=1e-13)
    for key in ["delta_D_site", "delta_D_component", "matched_delta_D_site",
                "selection_delta_D_site", "projected_delta_D_site",
                "exact_outer_D_site", "effective_outer_D_site"]:
        if grid[key].shape != expected or not np.isfinite(grid[key]).all():
            raise ValueError(f"Incomplete or misaligned grid field {key}")
    decomposition_error = float(np.max(np.abs(grid["delta_D_site"] -
                           grid["matched_delta_D_site"] - grid["selection_delta_D_site"])))
    if decomposition_error > 1e-12:
        raise ValueError("Matched + selection decomposition is inconsistent")
    eta_index = int(np.argmin(abs(resonance["eta"] - .05)))
    if abs(float(resonance["eta"][eta_index]) - .05) > 1e-12:
        raise ValueError("The requested primary broadening eta=.05 is absent")
    reference = float(resonance_summary["folding_reference_2Emax"])
    reconstructed = (resonance["term_sums"][:, 0, 0, None, :]
                     + resonance["term_sums"][:, 0, 1, None, :]
                     + resonance["term_sums"][:, 0, 2, None, :] * np.cos(phi)[None, :, None])
    cosine_error = float(np.max(abs(reconstructed - resonance["R_full"])))
    np.testing.assert_allclose(resonance["Rnorm_full"],
                               resonance["R_full"] * resonance["eta"][:, None, None] ** 2,
                               rtol=1e-12, atol=1e-14)
    delta = grid["delta_D_site"]
    peaks = {"R_full": coordinate(resonance["R_full"][eta_index], omega, phi),
             "R_outer": coordinate(resonance["R_outer"][eta_index], omega, phi),
             "delta": coordinate(delta, omega, phi),
             "delta_minimum": coordinate(delta, omega, phi, mode="min")}
    primary = {"raw": coefficients(resonance["R_full"][eta_index], delta),
               "phase_centered": coefficients(phase_centered(resonance["R_full"][eta_index]),
                                                phase_centered(delta))}
    windows = {"full_5.8_to_8": np.ones(len(omega), dtype=bool),
               "focus_6_to_6.6": (omega >= 6 - 1e-12) & (omega <= 6.6 + 1e-12)}
    quantities = {"raw_site_delta": "delta_D_site", "matched_site_delta": "matched_delta_D_site",
                  "selection_site_delta": "selection_delta_D_site", "projector_site_delta": "projected_delta_D_site",
                  "raw_component_delta": "delta_D_component"}
    rows = []
    for ie, width in enumerate(resonance["eta"]):
        for region in ["full", "outer"]:
            response = resonance[f"R_{region}"][ie]
            for quantity, key in quantities.items():
                for process in ["raw", "phase_centered"]:
                    xx = phase_centered(response) if process == "phase_centered" else response
                    yy = phase_centered(grid[key]) if process == "phase_centered" else grid[key]
                    for window, mask in windows.items():
                        rows.append({"dataset": "grid", "eta": float(width), "response_region": region,
                                     "quantity": quantity, "processing": process, "window": window,
                                     "phi": None, "phi_over_pi": None, **coefficients(xx[:, mask], yy[:, mask])})
            # These use real evaluated phase slices, including the extra pi/3 cut.
            for ip, phase in enumerate(slices["phi"]):
                response_slice = resonance_slices[f"R_{region}"][ie, ip]
                for process in ["raw", "phase_centered"]:
                    xx = (response_slice - response.mean(axis=0)) if process == "phase_centered" else response_slice
                    yy = (slices["delta_D_site"][ip] - delta.mean(axis=0)) if process == "phase_centered" else slices["delta_D_site"][ip]
                    for window, mask in windows.items():
                        rows.append({"dataset": "phase_slice", "eta": float(width), "response_region": region,
                                     "quantity": "raw_site_delta", "processing": process, "window": window,
                                     "phi": float(phase), "phi_over_pi": float(phase / np.pi),
                                     **coefficients(xx[mask], yy[mask])})
    with (output_dir / "correlation_summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    ip, iw = peaks["delta"]["index_phi"], peaks["delta"]["index_omega"]
    local_peak = {key: float(grid[key][ip, iw]) for key in
                  ["delta_D_site", "matched_delta_D_site", "selection_delta_D_site", "projected_delta_D_site",
                   "exact_outer_D_site", "effective_outer_D_site", "delta_D_component"]}
    if abs(local_peak["delta_D_site"]) > 1e-14:
        local_peak["signed_selection_over_raw"] = local_peak["selection_delta_D_site"] / local_peak["delta_D_site"]
        local_peak["signed_matched_over_raw"] = local_peak["matched_delta_D_site"] / local_peak["delta_D_site"]
    i62 = int(np.argmin(abs(omega - 6.2)))
    profile = {"omega": float(omega[i62]),
               "R_full_max_phi_over_pi": float(phi[np.argmax(resonance["R_full"][eta_index, :, i62])] / np.pi),
               "R_outer_max_phi_over_pi": float(phi[np.argmax(resonance["R_outer"][eta_index, :, i62])] / np.pi),
               "delta_max_phi_over_pi": float(phi[np.argmax(delta[:, i62])] / np.pi),
               "delta_max": float(np.max(delta[:, i62])), "delta_min": float(np.min(delta[:, i62]))}
    report = {"grid_shape_phi_omega": list(expected), "eta_primary": .05,
              "primary_response": "Full-spectrum R; eta=.05; original values, not log-transformed for correlations",
              "primary_correlations": primary,
              "phase_centering": "Subtract each quantity's own arithmetic phase average separately at every frequency, then correlate residual grid values",
              "p_values": "Not reported: these structured grid samples are not independent experimental observations",
              "normalization_note": "For fixed eta, Rnorm=eta^2 R gives identical Pearson/Spearman coefficients; it is not an independent validation",
              "outer_selection": "Center-folded Floquet zone, then highest abs(epsilon) ceil(0.2*178)=36 modes; site-IPR/log89",
              "line_cut_note": "pi/3 is directly recomputed; other requested phases reused from the main grid",
              "folding_reference_2Emax": reference,
              "max_cosine_reconstruction_error_R": cosine_error,
              "matched_selection_identity_max_error": decomposition_error,
              "peaks": peaks, "decomposition_at_delta_peak": local_peak,
              "decomposition_ratio_note": "Signed algebraic contributions under mode matching; not additive Hamiltonian causal attribution fractions",
              "phase_profile_omega_6_2": profile,
              "grid_correlations": [row for row in rows if row["dataset"] == "grid"],
              "slice_correlations": [row for row in rows if row["dataset"] == "phase_slice"]}
    (output_dir / "correlation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(output_dir / "fig1_fig2_plot_data.npz", omega=omega, phi=phi,
                        R_full=resonance["R_full"][eta_index], R_outer=resonance["R_outer"][eta_index],
                        delta_D_site=delta, R_phase_centered=phase_centered(resonance["R_full"][eta_index]),
                        delta_phase_centered=phase_centered(delta),
                        slice_phi=slices["phi"], R_full_slices=resonance_slices["R_full"][eta_index],
                        delta_slices=slices["delta_D_site"],
                        matched_delta_D_site=grid["matched_delta_D_site"],
                        selection_delta_D_site=grid["selection_delta_D_site"],
                        projected_delta_D_site=grid["projected_delta_D_site"],
                        folding_reference=reference)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.linewidth": .8, "savefig.facecolor": "white",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    render_main_figure(output_dir, grid, resonance, eta_index, reference, peaks)
    render_correlation_figure(output_dir, grid, slices, resonance, resonance_slices, eta_index, reference, primary)
    render_branch_figure(output_dir, grid, reference)
    render_eta_figure(output_dir, resonance, reference)
    print(json.dumps({"primary_correlations": primary, "peaks": peaks,
                      "decomposition_at_delta_peak": local_peak,
                      "phase_profile_omega_6_2": profile}, indent=2))
    print("Saved Figure 1/2, branch-selection and broadening figures, CSV/NPZ/JSON data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    run(args.results)
