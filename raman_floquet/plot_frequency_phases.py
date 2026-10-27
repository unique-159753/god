"""Plot real dimensionless Raman Floquet scans; never interpolate scan data.

The primary definition is spin-resolved IPR and Dn=-log(IPR)/log(L).
The omega=0 column is a frozen static Hamiltonian, not a Floquet limit.
"""
from __future__ import annotations

import argparse
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
import numpy as np


DEFAULT_RESULTS = Path(__file__).resolve().parent / "results" / "frequency_phase_three_V"
FIELDS = ["all_IPR", "all_Dn", "outer_Dn", "delta_all_Dn"]
ROW_LABELS = [r"Mean spin-resolved IPR", r"Full-spectrum $\overline{D_n}$",
              r"Outer 20% $\overline{D_n}$", r"$\Delta\overline{D_n}$ vs undriven static"]
COLOR_LABELS = [r"$\overline{\mathrm{IPR}}$", r"$\overline{D_n}$",
                r"$\overline{D}_{\rm outer}$", r"$\Delta\overline{D_n}$"]
REGIMES = ["Extended static reference", "Critical static reference", "Localized static reference"]


def read_fields(path, names, optional=()):
    with np.load(path, allow_pickle=False) as file:
        missing = [name for name in names if name not in file]
        if missing:
            raise ValueError(f"{path.name} missing fields: {missing}")
        return {name: file[name] for name in list(names) + list(optional) if name in file}


def scalar_number(value):
    return float(np.asarray(value).reshape(()))


def validate_grid(data):
    V, omega, phi = data["V"], data["omega"], data["phi"]
    if V.shape != (3,) or not np.allclose(V, [.5, 2., 4.], atol=1e-14, rtol=0):
        raise ValueError("Expected the three V values .5, 2, 4")
    if omega.shape != (50,) or not np.allclose(omega, np.linspace(0., 10., 50), atol=1e-13, rtol=0):
        raise ValueError("Expected 50 true dimensionless omega points in [0,10]")
    if phi.shape != (50,) or not np.allclose(phi, 2*np.pi*np.arange(50)/50, atol=1e-13, rtol=0):
        raise ValueError("Expected 50 distinct periodic phase samples, endpoint excluded")
    shape = (3, 50, 50)
    for name in FIELDS:
        if data[name].shape != shape:
            raise ValueError(f"Unexpected shape for {name}: {data[name].shape}")
        if name == "outer_Dn":
            if not np.isnan(data[name][:, :, 0]).all() or not np.isfinite(data[name][:, :, 1:]).all():
                raise ValueError("omega=0 outer Dn must be NaN; positive frequencies must be finite")
        elif not np.isfinite(data[name]).all():
            raise ValueError(f"Nonfinite computed values for {name}")
    L = int(data.get("L", 89))
    limit = np.log(2*L)/np.log(L)
    if np.min(data["all_IPR"]) < 1/(2*L)-1e-10 or np.max(data["all_IPR"]) > 1+1e-10:
        raise ValueError("Mean IPR outside its probability bounds")
    for name in ["all_Dn", "outer_Dn"]:
        if np.nanmin(data[name]) < -1e-9 or np.nanmax(data[name]) > limit+1e-9:
            raise ValueError(f"{name} outside spin-resolved log(L) participation bounds")
    if "converged" in data:
        if data["converged"].shape != shape:
            raise ValueError("converged mask shape mismatch")
        if not np.asarray(data["converged"], bool).all():
            raise ValueError("Incomplete convergence mask: refuse to present unconverged points as final results")
    return L


def periodic_close(values, phi):
    """Append the exact first data row solely for the periodic plot boundary."""
    return np.r_[phi, 2*np.pi], np.concatenate([values, values[:1]], axis=0)


def common_norms(data, L):
    ipr_max = float(np.max(data["all_IPR"]))
    amplitude = max(float(np.max(np.abs(data["delta_all_Dn"]))), 1e-12)
    # Spin-resolved IPR with log(L) normalization can legitimately give Dn>1.
    Dmax = float(np.log(2*L)/np.log(L))
    return [colors.Normalize(vmin=1/(2*L), vmax=max(ipr_max, 1/(2*L)+1e-12)),
            colors.Normalize(vmin=0., vmax=Dmax),
            colors.Normalize(vmin=0., vmax=Dmax),
            colors.TwoSlopeNorm(vmin=-amplitude, vcenter=0., vmax=amplitude)]


def set_axes(ax, omega, xlabel=True, ylabel=False):
    ax.set_xlim(0., float(omega[-1]))
    ax.set_ylim(0., 2.)
    ax.set_yticks([0., .5, 1., 1.5, 2.])
    ax.set_xticks([0., 2., 4., 6., 8., 10.])
    ax.set_xticklabels(["0\nstatic limit", "2", "4", "6", "8", "10"])
    if xlabel:
        ax.set_xlabel(r"Dimensionless $\omega/t_0$")
    if ylabel:
        ax.set_ylabel(r"$\varphi/\pi$")
    # Centers nearest omega=0 form a half-width cell after clipping x>=0.
    ax.axvline(float(omega[1])/2, color="0.25", linewidth=.65, linestyle=":", zorder=5)


def heatmap(ax, data, field, vindex, norm, row, xlabel, ylabel, cmap_name=None):
    phi_closed, values = periodic_close(data[field][vindex], data["phi"])
    cmap = plt.get_cmap(cmap_name or ("RdBu_r" if field == "delta_all_Dn" else "viridis")).copy()
    cmap.set_bad("#d6d6d6")
    mesh = ax.pcolormesh(data["omega"], phi_closed/np.pi, np.ma.masked_invalid(values),
                         shading="nearest", cmap=cmap, norm=norm, antialiased=False,
                         rasterized=False)
    set_axes(ax, data["omega"], xlabel=xlabel, ylabel=ylabel)
    return mesh


def save(fig, results, stem):
    paths = []
    for suffix in ["png", "pdf"]:
        path = results / f"{stem}.{suffix}"
        fig.savefig(path, dpi=350, bbox_inches="tight", facecolor="white")
        paths.append(path.name)
    plt.close(fig)
    return paths


def vector_colorbar(fig, mesh, cax, label):
    """Keep the color scale itself vector, as well as the data cell mesh."""
    bar = fig.colorbar(mesh, cax=cax)
    bar.set_label(label)
    if bar.solids is not None:
        bar.solids.set_rasterized(False)
    return bar


def render_grid(data, results, L):
    norms = common_norms(data, L)
    outputs = []
    fig, axes = plt.subplots(4, 3, figsize=(14.7, 14.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.885, bottom=.13, top=.918,
                        wspace=.11, hspace=.30)
    for row, field in enumerate(FIELDS):
        for column in range(3):
            mesh = heatmap(axes[row, column], data, field, column, norms[row], row,
                           xlabel=row == 3, ylabel=column == 0)
            if row == 0:
                axes[row, column].set_title(REGIMES[column]+f"\n$V/t_0={data['V'][column]:g}$", fontsize=11)
            axes[row, column].text(.025, 1.035, f"({chr(97+3*row+column)})",
                                   transform=axes[row, column].transAxes, fontsize=10)
        axes[row, 0].text(-.18, .5, ROW_LABELS[row], rotation=90,
                          transform=axes[row, 0].transAxes, va="center", ha="center", fontsize=10)
        box = axes[row, -1].get_position()
        cax = fig.add_axes([.902, box.y0, .015, box.height])
        vector_colorbar(fig, mesh, cax, COLOR_LABELS[row])
    fig.suptitle(r"Exact time-ordered Floquet response: $L=89$, $A=B=0.8$, $W=0$, PBC", y=.979, fontsize=15)
    fig.text(.5, .947, r"Spin-resolved IPR; $D_n=-\ln(\mathrm{IPR}_n)/\ln89$; shared color scale in each row", ha="center", fontsize=10)
    fig.text(.5, .037,
             "50 true frequency points × 50 distinct phase points per V. The 2π row copies phase 0 only to close the plot.\n"
             "ω=0 is the frozen static Hamiltonian, separated by a dotted boundary; outer Floquet D is undefined (gray).\n"
             "These finite-size participation indicators do not by themselves establish a localization phase transition.",
             ha="center", va="center", fontsize=8.7, linespacing=1.5)
    outputs += save(fig, results, "three_V_frequency_phase_summary")

    stems = ["mean_IPR", "mean_Dn", "outer_Dn", "delta_mean_Dn"]
    for row, field in enumerate(FIELDS):
        fig, axes = plt.subplots(1, 3, figsize=(14.7, 5.15), sharex=True, sharey=True)
        fig.subplots_adjust(left=.055, right=.887, bottom=.30, top=.77, wspace=.10)
        for column, ax in enumerate(axes):
            mesh = heatmap(ax, data, field, column, norms[row], row, True, column == 0)
            ax.set_title(REGIMES[column]+f"\n$V/t_0={data['V'][column]:g}$", fontsize=11)
        cax = fig.add_axes([.906, .30, .014, .47])
        vector_colorbar(fig, mesh, cax, COLOR_LABELS[row])
        fig.suptitle(ROW_LABELS[row], y=.96, fontsize=14)
        fig.text(.5, .865, r"$L=89$, PBC; spin-resolved IPR; $D_n=-\ln(\mathrm{IPR}_n)/\ln89$", ha="center", fontsize=10)
        fig.text(.5, .053,
                 "Shared color scale across V; 50 × 50 computed points. The periodic boundary copies phase 0 without interpolation.\n"
                 "Dotted line separates the frozen ω=0 static column; no Floquet outer-state observable exists there.",
                 ha="center", fontsize=8.5, linespacing=1.45)
        outputs += save(fig, results, f"three_V_{stems[row]}")
    return outputs, norms


def render_effective_residuals(data, results):
    if not all(key in data for key in ["effective_all_Dn", "effective_outer_Dn"]):
        return [], {}
    fields = ["exact_minus_effective_all_Dn", "exact_minus_effective_outer_Dn"]
    labels = [r"Full spectrum: $\overline{D}_{\rm exact}-\overline{D}_{\rm eff+kick}$",
              r"Outer 20%: $\overline{D}_{\rm exact}-\overline{D}_{\rm eff+kick}$"]
    norms = []
    for name, exact, effective in zip(fields, ["all_Dn", "outer_Dn"],
                                     ["effective_all_Dn", "effective_outer_Dn"]):
        if data[effective].shape != (3, 50, 50):
            raise ValueError(f"Invalid effective-grid shape: {effective}")
        data[name] = data[exact]-data[effective]
        data[name][..., 0] = np.nan
        if not np.isfinite(data[name][..., 1:]).all():
            raise ValueError(f"Missing positive-frequency residual values for {name}")
        amplitude = max(float(np.nanmax(np.abs(data[name]))), 1e-12)
        norms.append(colors.TwoSlopeNorm(vmin=-amplitude, vcenter=0., vmax=amplitude))
    fig, axes = plt.subplots(2, 3, figsize=(14.7, 8.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.063, right=.885, bottom=.205, top=.80, wspace=.11, hspace=.31)
    for row, field in enumerate(fields):
        for column in range(3):
            mesh = heatmap(axes[row, column], data, field, column, norms[row], row,
                           xlabel=row == 1, ylabel=column == 0, cmap_name="RdBu_r")
            if row == 0:
                axes[row, column].set_title(REGIMES[column]+f"\n$V/t_0={data['V'][column]:g}$", fontsize=11)
        box = axes[row, -1].get_position()
        cax = fig.add_axes([.903, box.y0, .014, box.height])
        vector_colorbar(fig, mesh, cax, labels[row])
    fig.suptitle("Exact minus first-order + kick: finite-frequency residuals", y=.978, fontsize=14)
    fig.text(.5, .899,
             r"Spin-resolved IPR; $D_n=-\ln(\mathrm{IPR}_n)/\ln89$; independent symmetric shared color scale in each row",
             ha="center", fontsize=10)
    fig.text(.5, .066,
             "The first-order high-frequency expansion is uncontrolled at low frequency; these differences locate approximation failure.\n"
             "Residuals are not isolated resonance strengths or additive causal contributions. Main phase diagrams use exact time ordering.\n"
             "ω=0 has no effective Floquet model and is gray; the 2π plot boundary copies the phase-0 data without interpolation.",
             ha="center", fontsize=8.8, linespacing=1.5)
    outputs = save(fig, results, "three_V_exact_minus_effective_residual")
    return outputs, {name: [float(norm.vmin), float(norm.vmax)] for name, norm in zip(fields, norms)}


def load_slices(path):
    names = ["V", "omega"] + FIELDS
    optional = ["phi", "phase", "converged", "effective_all_Dn", "effective_outer_Dn",
                "static_kick_all_Dn", "static_all_Dn", "delta_effective_all_Dn"]
    data = read_fields(path, names, optional)
    phase = data.get("phase", data.get("phi"))
    if phase is None or phase.shape != (2,) or not np.allclose(phase, [0., np.pi/2], atol=1e-13, rtol=0):
        raise ValueError("Frequency slices must contain independently computed phi=0 and pi/2")
    for name in FIELDS:
        if data[name].shape != (3, 2, 50):
            raise ValueError(f"Invalid slice shape for {name}")
    if "converged" in data and not np.asarray(data["converged"], bool).all():
        raise ValueError("Frequency slices include unconverged points")
    data["phase"] = phase
    return data


def render_slices(data, results):
    fig, axes = plt.subplots(4, 3, figsize=(14.7, 12.2), sharex=True, sharey="row")
    fig.subplots_adjust(left=.075, right=.975, bottom=.12, top=.90, wspace=.16, hspace=.18)
    colors_ = ["#2469a1", "#be4933"]
    for row, field in enumerate(FIELDS):
        allvalues = data[field]
        if field == "delta_all_Dn":
            magnitude = max(float(np.nanmax(np.abs(allvalues))), 1e-12)
            bounds = (-1.07*magnitude, 1.07*magnitude)
        else:
            lower, upper = float(np.nanmin(allvalues)), float(np.nanmax(allvalues))
            padding = max(.07*(upper-lower), 1e-4)
            bounds = (lower-padding, upper+padding)
        for column, ax in enumerate(axes[row]):
            for ip, label in enumerate([r"$\varphi=0$", r"$\varphi=\pi/2$"]):
                ax.plot(data["omega"][1:], data[field][column, ip, 1:], color=colors_[ip],
                        linestyle="-" if ip == 0 else "--", marker=".", markersize=3,
                        linewidth=1.2, label=label)
                if np.isfinite(data[field][column, ip, 0]):
                    ax.plot([0.], [data[field][column, ip, 0]], color=colors_[ip],
                            linestyle="none", marker="o", markersize=4.2,
                            markerfacecolor="white", markeredgewidth=1.1, clip_on=False)
            if row == 0:
                ax.set_title(REGIMES[column]+f"; $V/t_0={data['V'][column]:g}$", fontsize=11)
            if column == 0:
                ax.set_ylabel(COLOR_LABELS[row])
            if row == 3:
                ax.set_xlabel(r"Dimensionless $\omega/t_0$")
            if field == "delta_all_Dn":
                ax.axhline(0., color=".5", linewidth=.7)
            ax.set_ylim(*bounds)
            ax.set_xlim(0., 10.)
            ax.set_xticks([0., 2., 4., 6., 8., 10.])
            ax.set_xticklabels(["0\nstatic limit", "2", "4", "6", "8", "10"])
            ax.grid(alpha=.16)
            ax.axvline(float(data["omega"][1])/2, color=".4", linestyle=":", linewidth=.65)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .954), ncol=2, frameon=False)
    fig.suptitle("True frequency cuts at two drive phases", y=.983, fontsize=15)
    fig.text(.5, .030,
             "Phase π/2 is computed separately because it is absent from the 50-point periodic phase grid.\n"
             "Dots are computed points; connecting lines guide the eye. Open ω=0 markers are separate frozen static values, not a Floquet limit.",
             ha="center", fontsize=8.7, linespacing=1.5)
    return save(fig, results, "three_V_frequency_cuts")


def render_effective_slices(data, results):
    if "effective_all_Dn" not in data:
        return []
    effective = data["effective_all_Dn"]
    if effective.shape != (3, 2, 50):
        raise ValueError("Invalid effective slice shape")
    selected = data["omega"] >= 6.
    if not np.isfinite(effective[..., selected]).all():
        raise ValueError("Missing high-frequency effective control points")
    fig, axes = plt.subplots(2, 3, figsize=(14.7, 7.1), sharex=True, sharey="row")
    fig.subplots_adjust(left=.072, right=.975, bottom=.16, top=.78, wspace=.16, hspace=.18)
    phase_colors = ["#2469a1", "#be4933"]
    for column in range(3):
        for ip, phase_label in enumerate([r"$\varphi=0$", r"$\varphi=\pi/2$"]):
            x = data["omega"][selected]
            exact = data["all_Dn"][column, ip, selected]
            approximate = effective[column, ip, selected]
            axes[0, column].plot(x, exact, color=phase_colors[ip], marker=".", linewidth=1.25,
                                 label=phase_label+" exact")
            axes[0, column].plot(x, approximate, color=phase_colors[ip], linestyle="--", linewidth=1.25,
                                 label=phase_label+" first order + kick")
            axes[1, column].plot(x, exact-approximate, color=phase_colors[ip], marker=".", linewidth=1.25)
        axes[0, column].set_title(f"$V/t_0={data['V'][column]:g}$", fontsize=12)
        axes[1, column].axhline(0., color=".4", linewidth=.7)
        axes[1, column].set_xlabel(r"Dimensionless $\omega/t_0$")
        for ax in axes[:, column]:
            ax.grid(alpha=.16)
            ax.set_xlim(6., 10.)
    axes[0, 0].set_ylabel(r"Full-spectrum $\overline{D_n}$")
    axes[1, 0].set_ylabel(r"$\overline{D}_{\rm exact}-\overline{D}_{\rm eff+kick}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .922), ncol=2, frameon=False)
    fig.suptitle("High-frequency first-order control against exact propagation", y=.985, fontsize=14)
    fig.text(.5, .045,
             "The main phase diagrams use exact time ordering. This additional first-order van Vleck + lab-kick control\n"
             "is an approximation check; model differences are not additive causal contributions or proof of a new transition.",
             ha="center", fontsize=8.7, linespacing=1.5)
    return save(fig, results, "three_V_high_frequency_effective_control")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--verify-modal-statistics", action="store_true")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    source = args.results / "frequency_phase_scan.npz"
    data = read_fields(source, ["V", "omega", "phi"] + FIELDS,
                       ["L", "converged", "steps", "metadata", "static_all_Dn",
                        "effective_all_Dn", "effective_outer_Dn"])
    L = validate_grid(data)
    if args.verify_modal_statistics:
        raw = read_fields(source, ["IPR", "Dn", "outer_indices"])
        if raw["IPR"].shape != (3, 50, 50, 2*L):
            raise ValueError("Invalid saved per-state IPR shape")
        checks = [np.max(np.abs(raw["Dn"]+np.log(raw["IPR"])/np.log(L))),
                  np.max(np.abs(np.mean(raw["IPR"], axis=-1)-data["all_IPR"])),
                  np.max(np.abs(np.mean(raw["Dn"], axis=-1)-data["all_Dn"]))]
        outer = np.mean(np.take_along_axis(raw["Dn"], raw["outer_indices"], axis=-1), axis=-1)
        checks.append(np.max(np.abs(outer[..., 1:]-data["outer_Dn"][..., 1:])))
        if max(checks) > 2e-11:
            raise ValueError(f"Modal statistics do not reproduce figure arrays: {checks}")
    outputs, norms = render_grid(data, args.results, L)
    residual_outputs, residual_bounds = render_effective_residuals(data, args.results)
    outputs += residual_outputs
    slices_path = args.results / "frequency_phase_slices.npz"
    if slices_path.exists():
        slices = load_slices(slices_path)
        outputs += render_slices(slices, args.results)
        outputs += render_effective_slices(slices, args.results)
    plot_data = {name: data[name] for name in ["V", "omega", "phi"] + FIELDS}
    for name in ["effective_all_Dn", "effective_outer_Dn", "exact_minus_effective_all_Dn",
                 "exact_minus_effective_outer_Dn"]:
        if name in data:
            plot_data[name] = data[name]
    plot_data["phi_plot_closed"] = np.r_[data["phi"], 2*np.pi]
    np.savez_compressed(args.results / "frequency_phase_plot_data.npz", **plot_data)
    summary = {
        "source": source.name, "V": data["V"].tolist(), "grid_shape_V_phi_omega": [3, 50, 50],
        "primary_IPR": "sum of fourth powers over all 2L spin-resolved amplitudes",
        "primary_Dn": f"-ln(IPR)/ln({L})", "Dn_probability_upper_bound": float(np.log(2*L)/np.log(L)),
        "frequency_axis": "dimensionless omega/t0, 50 points inclusive 0 and 10; no Hz assumption",
        "phase_closure": "Only first data row duplicated at 2pi; no data interpolation",
        "zero_frequency": "Frozen static Hamiltonian; outer Floquet Dn undefined and shown gray",
        "shared_row_color_bounds": {field: [float(norm.vmin), float(norm.vmax)]
                                    for field, norm in zip(FIELDS, norms)},
        "exact_minus_effective_row_bounds": residual_bounds,
        "files": outputs, "png_dpi": 350, "pdf_mesh": "vector pcolormesh, rasterized=False",
        "slices_available": slices_path.exists(), "finite_size_note": "Participation diagnostics alone do not establish a phase transition",
    }
    (args.results / "frequency_phase_figures_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
