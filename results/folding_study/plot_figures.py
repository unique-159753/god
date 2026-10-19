#!/usr/bin/env python3
"""Plot the independently computed Floquet folding study (no interpolation).

Every point comes from the NPZ files in the selected directory. Scatter and
line artists are retained as vector objects in PDFs; PNGs use 350 DPI.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/floquet-folding-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/floquet-folding-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


PARAMETERS = (
    r"$L=89$, $\alpha=55/89$, $t_0=1$, $\Delta_0=0.5$, $V=2.5$, $W=0.01$, "
    r"$A=B=0.8$, $\theta=0$, PBC"
)
PHASE_NAMES = ("0", "pi4", "pi2")
PHASE_TITLES = (r"$\varphi=0$", r"$\varphi=\pi/4$", r"$\varphi=\pi/2$")
PHASE_TICKS = np.linspace(0, np.pi, 5)
PHASE_LABELS = (r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$")
BRANCH_COLORS = ("#3b528b", "#21918c", "#e39b18", "#c54459")


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "mathtext.fontset": "dejavusans", "axes.titlesize": 11,
        "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "savefig.dpi": 350, "figure.dpi": 130,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "lines.linewidth": 1.6, "grid.alpha": 0.22,
    })


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


def scalar(data: dict, key: str, default=None):
    if key not in data:
        if default is None:
            raise KeyError(key)
        return default
    return np.asarray(data[key]).item()


def save(fig: plt.Figure, outdir: Path, name: str, files: list[str]) -> None:
    for suffix in ("png", "pdf"):
        path = outdir / f"{name}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        files.append(path.name)
    plt.close(fig)


def footer(fig: plt.Figure, *lines: str, y: float = 0.014) -> None:
    fig.text(0.5, y, "\n".join(lines), ha="center", va="bottom", fontsize=8,
             linespacing=1.5)


def folding_line(ax: plt.Axes, value: float) -> None:
    ax.axvline(value, color="0.18", linestyle=(0, (3, 2)), lw=1.0, zorder=5)


def dnorm_from(*arrays) -> Normalize:
    minimum = min(float(np.nanmin(array)) for array in arrays)
    maximum = max(float(np.nanmax(array)) for array in arrays)
    if not np.isfinite(minimum + maximum):
        raise ValueError("Nonfinite D values cannot be plotted")
    return Normalize(minimum, maximum if maximum > minimum else minimum + 1e-12)


def difference_norm(*arrays) -> TwoSlopeNorm:
    extent = max(float(np.nanmax(np.abs(array))) for array in arrays)
    if not np.isfinite(extent):
        raise ValueError("Nonfinite difference values cannot be plotted")
    extent = max(extent, np.finfo(float).eps)
    return TwoSlopeNorm(vmin=-extent, vcenter=0, vmax=extent)


def fig1(outdir: Path, static: dict, curve: dict, files: list[str]) -> dict:
    omega = curve["omega"]
    keep = (omega >= 5.8 - 1e-10) & (omega <= 8.0 + 1e-10)
    frequency = omega[keep]
    if len(frequency) < 2:
        raise ValueError("Figure 1 needs computed frequency samples in [5.8,8]")
    energy, static_d = static["energy"], static["D"]
    unfolded = energy[None, :] / frequency[:, None]
    static_colors = np.broadcast_to(static_d, unfolded.shape)
    values = (unfolded, curve["q_eff"][keep], curve["q_exact"][keep])
    colors = (static_colors, curve["D_eff_bare"][keep], curve["D_exact"][keep])
    norm = dnorm_from(*colors)
    omega_fold = float(scalar(static, "omega_fold"))
    emax = float(scalar(static, "Emaxabs"))
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.4))
    fig.subplots_adjust(left=0.06, right=0.915, top=0.83, bottom=0.29, wspace=0.25)
    titles = ("(a) Static spectrum: no folding", "(b) Effective spectrum: folded", "(c) Exact Floquet spectrum")
    for index, ax in enumerate(axes):
        x = np.broadcast_to(frequency[:, None], values[index].shape)
        scatter = ax.scatter(x.ravel(), values[index].ravel(), c=colors[index].ravel(),
                             s=1.4, marker="o", linewidths=0, cmap="viridis", norm=norm,
                             alpha=1.0, rasterized=False)
        ax.set(xlim=(5.8, 8.0), xlabel=r"Drive frequency $\omega/t_0$", title=titles[index])
        ax.set_xticks((6.0, 6.5, 7.0, 7.5, 8.0))
        folding_line(ax, omega_fold)
        if index:
            ax.set_ylim(-0.5, 0.5)
            ax.set_ylabel(r"Quasienergy $q_n=\varepsilon_n/\omega$")
        else:
            limit = max(0.53, float(np.max(np.abs(unfolded))) * 1.04)
            ax.set_ylim(-limit, limit)
            ax.set_ylabel(r"Unfolded static energy $E_n/\omega$")
            for zone_edge in (-0.5, 0.5):
                ax.axhline(zone_edge, color="0.28", lw=0.9, ls=":", zorder=5)
    color_axis = fig.add_axes((0.935, 0.29, 0.012, 0.54))
    colorbar = fig.colorbar(scatter, cax=color_axis)
    colorbar.set_label(r"State dimension $D_n$")
    # Matplotlib otherwise automatically rasterizes dense colorbar gradients.
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    fig.suptitle(r"Floquet-zone folding at $\varphi=\pi/2$", fontsize=14, y=0.96)
    footer(fig, PARAMETERS,
           rf"Static reference: $\max_n|E_n|/t_0={emax:.6f}$, $\omega_{{\mathrm{{fold}}}}/t_0={omega_fold:.6f}$ (vertical dashed line).",
           r"The static folding scale is a reference, not a driven-resonance criterion. All markers are computed eigenstates; effective-state colors are bare $D_n$.")
    save(fig, outdir, "fig1_quasienergy_folding", files)
    return {"D_color_limits": [norm.vmin, norm.vmax], "omega_fold": omega_fold,
            "static_Emaxabs": emax, "frequency_samples": int(len(frequency)),
            "scatter_points_per_panel": int(values[0].size)}


def map_axes(ax: plt.Axes, grid: dict) -> None:
    ax.set(xlim=(grid["omega"][0], grid["omega"][-1]),
           ylim=(grid["phi"][0], grid["phi"][-1]),
           xlabel=r"Drive frequency $\omega/t_0$", ylabel=r"Relative phase $\varphi$")
    ax.set_yticks(PHASE_TICKS, PHASE_LABELS)


def fig2(outdir: Path, grid: dict, files: list[str]) -> dict:
    metrics = grid["metrics"]
    expected = (4, len(grid["phi"]), len(grid["omega"]), 4)
    if metrics.shape != expected:
        raise ValueError(f"Grid metrics have shape {metrics.shape}; expected {expected}")
    exact, corrected = metrics[0, ..., 1], metrics[3, ..., 1]
    corrected_error = exact - corrected
    original_error = exact - metrics[1, ..., 1]
    norm = dnorm_from(exact, corrected)
    error_norm = difference_norm(corrected_error, original_error)
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.8))
    fig.subplots_adjust(left=0.085, right=0.94, top=0.90, bottom=0.22, hspace=0.41, wspace=0.38)
    titles = ("(a) Exact Floquet", "(b) Effective: folded selection + kick",
              "(c) Exact − corrected effective", "(d) Exact − unfolded-selection effective")
    for index, (ax, values) in enumerate(zip(axes.flat, (exact, corrected, corrected_error, original_error))):
        mesh = ax.pcolormesh(grid["omega"], grid["phi"], values, shading="nearest",
                             cmap="viridis" if index < 2 else "RdBu_r",
                             norm=norm if index < 2 else error_norm, rasterized=False)
        map_axes(ax, grid)
        ax.set_title(titles[index], pad=8)
        label = r"$\overline{D_n}$ (outer 20%)" if index < 2 else r"$\Delta\overline{D_n}$ (outer 20%)"
        colorbar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.035)
        colorbar.set_label(label)
        if colorbar.solids is not None:
            colorbar.solids.set_rasterized(False)
    fig.suptitle("Outer-state localization with consistent Floquet-zone selection", fontsize=13, y=0.97)
    slices = int(scalar(grid, "slices", 128))
    footer(fig, PARAMETERS,
           rf"Exact Floquet: $M={slices}$ midpoint slices. Folded selection: largest 20% of $|q_n|$, $q_n\in[-1/2,1/2)$.",
           r"Kick-corrected states: $e^{-iK^{(1)}(0)}|\psi_n^{\mathrm{eff}}\rangle$. Panel (d): bare states selected by largest 20% of $|E_n^{\mathrm{eff}}/\omega|$.",
           r"Finite-size localization diagnostics; color scales are shared within each row.")
    save(fig, outdir, "fig2_corrected_Douter_comparison", files)
    return {"D_color_limits": [norm.vmin, norm.vmax],
            "difference_color_limits": [error_norm.vmin, error_norm.vmax],
            "time_slices": slices, "grid_shape": [len(grid["phi"]), len(grid["omega"])]}


def branch_flags(curve: dict, selected: np.ndarray) -> np.ndarray:
    flags = np.asarray(curve["ambiguous"][:, selected], dtype=bool)
    flags |= curve["assigned_overlap"][:, selected] < 0.8
    if "near_gap" in curve:
        gap = curve["near_gap"]
        if gap.dtype.kind != "b":
            raise ValueError("near_gap must be a Boolean warning flag")
        flags |= gap[:, selected]
    return flags


def fig3(outdir: Path, curves: list[dict], files: list[str]) -> dict:
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 11.0), sharex="col", sharey="row")
    fig.subplots_adjust(left=0.085, right=0.98, top=0.91, bottom=0.22, hspace=0.32, wspace=0.17)
    summary = {}
    refinement_path = outdir / "refinement_summary.json"
    refinement = json.loads(refinement_path.read_text()) if refinement_path.exists() else {}
    unresolved_by_phase = {
        phase["tag"]: [sorted((float(interval["omega_start"]), float(interval["omega_end"])))
                        for interval in phase.get("intervals", []) if interval.get("unresolved", False)]
        for phase in refinement.get("phases", [])
    }
    for column, (curve, phase_title) in enumerate(zip(curves, PHASE_TITLES)):
        omega = curve["omega"]
        keep = (omega >= 5.8 - 1e-10) & (omega <= 8.0 + 1e-10)
        selected = np.asarray(curve["representative_branch_ids"], dtype=int)
        if len(selected) != 4:
            raise ValueError("Four representative branch IDs are required")
        anchor = int(np.argmin(abs(omega - 8.0)))
        selected = selected[np.argsort(curve["q_exact"][anchor, selected])]
        flags = branch_flags(curve, selected)[keep]
        x = omega[keep]
        arrays = (curve["energy_unwrapped"][keep][:, selected],
                  curve["D_exact"][keep][:, selected],
                  curve["assigned_overlap"][keep][:, selected])
        for row, values in enumerate(arrays):
            ax = axes[row, column]
            for lower, upper in unresolved_by_phase.get(PHASE_NAMES[column], []):
                ax.axvspan(lower, upper, facecolor="0.88", edgecolor="0.65",
                           hatch="////", linewidth=0.65, zorder=0.4)
                # A thin center line keeps sub-pixel intervals visible in previews.
                ax.axvline((lower + upper) / 2, color="0.78", lw=0.9, zorder=0.5)
            for branch, color in enumerate(BRANCH_COLORS):
                ax.plot(x, values[:, branch], color=color, lw=1.6, zorder=3)
                flagged = flags[:, branch]
                ax.plot(x[flagged], values[flagged, branch], linestyle="none", marker="x",
                        color=color, markersize=4.0, markeredgewidth=0.85, zorder=6)
            ax.grid(True)
            ax.set_xlim(5.8, 8.0)
            ax.set_xticks((6.0, 6.5, 7.0, 7.5, 8.0))
            ax.text(0.025, 0.97, f"({chr(97 + row * 3 + column)})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=10,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1})
        axes[0, column].set_title(phase_title, fontsize=12, pad=10)
        controls = (
            (curve["fixed_reference_D_outer"][keep], "0.45", "--"),
            (curve["metrics"][keep, 0, 1], "0.10", ":"),
        )
        for control_values, color, linestyle in controls:
            axes[1, column].plot(x, control_values, color=color, ls=linestyle, lw=1.1, zorder=2)
        axes[2, column].axhline(0.8, color="0.55", ls="--", lw=0.9, zorder=1)
        axes[2, column].set_ylim(-0.025, 1.035)
        axes[2, column].set_xlabel(r"Drive frequency $\omega/t_0$")
        summary[PHASE_NAMES[column]] = {
            "representative_branch_ids": selected.tolist(),
            "anchor_q": curve["q_exact"][anchor, selected].tolist(),
            "flagged_samples_per_branch": np.sum(flags, axis=0).tolist(),
            "minimum_assigned_overlap_per_branch": np.min(arrays[2], axis=0).tolist(),
            "unresolved_connection_intervals_any_tracked_state": unresolved_by_phase.get(PHASE_NAMES[column], []),
        }
    axes[0, 0].set_ylabel(r"Unwrapped quasienergy $\widetilde{\varepsilon}_n/t_0$")
    axes[1, 0].set_ylabel(r"State dimension $D_n$")
    axes[2, 0].set_ylabel(r"Assigned-neighbor overlap $|\langle\psi_i|\psi_{i+1}\rangle|^2$")
    handles = [Line2D([], [], color=color, label=f"Branch {index + 1}")
               for index, color in enumerate(BRANCH_COLORS)]
    handles += [Line2D([], [], color="0.45", ls="--", label="Fixed-reference outer-set mean"),
                Line2D([], [], color="0.10", ls=":", label="Dynamic outer-set mean"),
                Line2D([], [], color="0.2", ls="none", marker="x", label="Tracking / near-gap warning"),
                Line2D([], [], color="0.55", ls="--", label="Overlap = 0.8"),
                Patch(facecolor="0.88", edgecolor="0.65", hatch="////", label="Unresolved connection interval")]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.53, 0.105),
               frameon=False, fontsize=8, columnspacing=1.7, handlelength=2.5)
    fig.suptitle("Overlap-continuous Floquet branches", fontsize=14, y=0.98)
    footer(fig, PARAMETERS,
           r"Gauge anchored at $\omega/t_0=8$; four positive-$q$ branches ordered by their anchor $q$. Assignment compares neighbors toward the anchor.",
           r"Crosses: low overlap ($<0.8$), ambiguity, or near gaps. Shaded: unresolved branch-connection intervals (any tracked state).",
           r"These tracks are not guaranteed adiabatic branches; low-confidence changes do not establish a transition.")
    save(fig, outdir, "fig3_eigenstate_branch_tracking", files)
    return summary


def fig4(outdir: Path, static: dict, curves: list[dict], files: list[str]) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 5.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.34, wspace=0.16)
    omega_fold = float(scalar(static, "omega_fold"))
    colors = ("#4b66a2", "#d37e24", "#228b75")
    linestyles = ("-", "--", "-.")
    summary = {}
    for index, (ax, curve, phase_title) in enumerate(zip(axes, curves, PHASE_TITLES)):
        omega = curve["omega"]
        metrics = curve["metrics"]
        exact = metrics[:, 0, 1]
        differences = [exact - metrics[:, model, 1] for model in (1, 2, 3)]
        for values, color, linestyle in zip(differences, colors, linestyles):
            ax.plot(omega, values, color=color, ls=linestyle, lw=1.7)
        for key in (6.0, 6.3, 6.5, 7.0, 8.0, 10.0):
            ax.axvline(key, color="0.8", lw=0.6, ls=":", zorder=0)
        ax.text(6.3, 0.985, "6.3", ha="right", va="top", rotation=90, fontsize=7,
                color="0.5", transform=ax.get_xaxis_transform())
        folding_line(ax, omega_fold)
        ax.axhline(0, color="0.55", lw=0.8, zorder=1)
        ax.set(xlim=(5.8, 10.0), xlabel=r"Drive frequency $\omega/t_0$",
               title=f"({chr(97 + index)}) {phase_title}")
        ax.set_xticks((6.0, 6.5, 7.0, 8.0, 9.0, 10.0))
        ax.grid(axis="y")
        summary[PHASE_NAMES[index]] = {
            name: {"rms": float(np.sqrt(np.mean(values**2))), "max_abs": float(np.max(np.abs(values)))}
            for name, values in zip(("unfolded_bare", "folded_bare", "folded_kick"), differences)
        }
    axes[0].set_ylabel(r"Outer-state difference $\Delta\overline{D_n}$ (exact − effective)")
    labels = ("Unfolded energy selection, bare states", "Folded quasienergy selection, bare states",
              "Folded quasienergy selection + kick")
    handles = [Line2D([], [], color=color, ls=linestyle, label=label)
               for color, linestyle, label in zip(colors, linestyles, labels)]
    handles += [Line2D([], [], color="0.18", ls=(0, (3, 2)), label=r"Static reference $\omega_{\mathrm{fold}}$")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.145),
               ncol=2, frameon=False, fontsize=8.5, columnspacing=2.0, handlelength=2.8)
    fig.suptitle("Effective-model errors under three selection and state conventions", fontsize=13, y=0.97)
    footer(fig, PARAMETERS,
           r"Unfolded: select largest 20% of $|E_n^{\mathrm{eff}}/\omega|$. Folded: select largest 20% of $|q_n^{\mathrm{eff}}|$.",
           r"Kick: use the same folded set after $e^{-iK^{(1)}(0)}$. Curves are methodological contrasts, not strictly additive physical corrections.")
    save(fig, outdir, "fig4_effective_model_errors", files)
    return summary


def main(outdir: Path | str | None = None) -> dict:
    started = time.perf_counter()
    output = Path(outdir).resolve() if outdir is not None else Path(__file__).resolve().parent
    style()
    static = load_npz(output / "static_spectrum.npz")
    grid = load_npz(output / "grid_41.npz")
    curves = [load_npz(output / f"curves_phi_{name}.npz") for name in PHASE_NAMES]
    files = []
    summary = {"source_files": ["static_spectrum.npz", "grid_41.npz"] +
               [f"curves_phi_{name}.npz" for name in PHASE_NAMES],
               "png_dpi": 350, "pdf_artists": "Vector scatter, lines, text, and heatmap cells; no rasterized artists.",
               "interpolation": "None. All plots use computed data samples."}
    summary["fig1"] = fig1(output, static, curves[2], files)
    summary["fig2"] = fig2(output, grid, files)
    summary["fig3"] = fig3(output, curves, files)
    summary["fig4"] = fig4(output, static, curves, files)
    summary.update({"files": files, "figure_count": len(files) // 2,
                    "plotting_wall_seconds": time.perf_counter() - started,
                    "pdf_image_xobjects": {
                        name: (output / name).read_bytes().count(b"/Subtype /Image")
                        for name in files if name.endswith(".pdf")}})
    (output / "plots_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure_count": summary["figure_count"], "plotting_wall_seconds": summary["plotting_wall_seconds"]}))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path)
    main(parser.parse_args().outdir)
