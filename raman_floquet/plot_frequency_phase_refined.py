"""Plot the computed 100 x 100 Raman Floquet IPR and Dn grids.

Only full-spectrum observables are shown. No static subtraction, effective
model residual, interpolation, or outer-quasienergy selection is used here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/raman_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/raman_plot_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


DEFAULT_RESULTS = Path(__file__).resolve().parent / "results" / "frequency_phase_100x100_phi_0_pi"
FIELDS = ("all_IPR", "all_Dn")
ROW_LABELS = ("Full-spectrum mean spin-resolved IPR", r"Full-spectrum mean $D_n$")
COLOR_LABELS = (r"$\overline{\mathrm{IPR}}$", r"$\overline{D_n}$")
REGIMES = ("Extended static reference", "Critical static reference", "Localized static reference")
L = 89


def read_grid(path: Path):
    with np.load(path, allow_pickle=False) as file:
        required = ("V", "omega", "phi", *FIELDS, "converged")
        missing = [name for name in required if name not in file]
        if missing:
            raise ValueError(f"Missing computed fields: {missing}")
        data = {name: file[name] for name in required}
    if data["V"].shape != (3,) or not np.allclose(data["V"], [.5, 2., 4.], atol=1e-14, rtol=0):
        raise ValueError("Expected V=.5, 2, 4")
    if data["omega"].shape != (100,) or not np.allclose(data["omega"], np.linspace(.2, 10., 100), atol=1e-13, rtol=0):
        raise ValueError("Expected 100 computed dimensionless frequencies .2 through 10, inclusive")
    if data["phi"].shape != (100,) or not np.allclose(data["phi"], np.linspace(0., np.pi, 100), atol=1e-13, rtol=0):
        raise ValueError("Expected 100 phase values 0 through pi, inclusive")
    shape = (3, 100, 100)
    for name in (*FIELDS, "converged"):
        if data[name].shape != shape:
            raise ValueError(f"Unexpected {name} shape {data[name].shape}, expected {shape}")
        if not np.isfinite(data[name]).all():
            raise ValueError(f"Nonfinite {name}")
    if not np.asarray(data["converged"], bool).all():
        raise ValueError("Refuse to plot a grid containing unconverged points as a final result")
    if data["all_IPR"].min() < 1/(2*L)-1e-10 or data["all_IPR"].max() > 1+1e-10:
        raise ValueError("Mean IPR outside normalization bounds")
    if data["all_Dn"].min() < -1e-10 or data["all_Dn"].max() > np.log(2*L)/np.log(L)+1e-10:
        raise ValueError("Mean Dn outside log(L) participation bounds")
    return data


def verify_modes(path: Path, data):
    """Check means against actual 178 eigenstate values, with bounded memory."""
    expected = (3, 100, 100, 2*L)
    with np.load(path, allow_pickle=False) as file:
        for modal_name, mean_name in (("IPR", "all_IPR"), ("Dn", "all_Dn")):
            values = file[modal_name]
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(f"Invalid raw {modal_name} array")
            if not np.allclose(values.mean(axis=-1), data[mean_name], atol=2e-13, rtol=0):
                raise ValueError(f"Saved {mean_name} does not equal mean of actual eigenstate values")
        ipr, dn = file["IPR"], file["Dn"]
        identity_error = float(np.max(np.abs(dn + np.log(ipr)/np.log(L))))
        if identity_error > 2e-13:
            raise ValueError(f"Dn does not use the specified IPR/log(L): error={identity_error}")
    return {"state_count_per_point": 2*L, "Dn_identity_max_error": identity_error,
            "modal_mean_check": "passed"}


def norms_for(data):
    result = []
    for field in FIELDS:
        low, high = float(data[field].min()), float(data[field].max())
        if high <= low:
            high = low + 1e-12
        result.append(colors.Normalize(vmin=low, vmax=high))
    return result


def heatmap(ax, data, field, iv, norm, xlabel, ylabel):
    mesh = ax.pcolormesh(data["omega"], data["phi"]/np.pi, data[field][iv],
                         shading="nearest", cmap="viridis", norm=norm,
                         antialiased=False, rasterized=False)
    ax.set_xlim(.2, 10.)
    ax.set_ylim(0., 1.)
    ax.set_xticks([.2, 2., 4., 6., 8., 10.])
    ax.set_yticks([0., .25, .5, .75, 1.])
    if xlabel:
        ax.set_xlabel(r"Dimensionless $\omega/t_0$")
    if ylabel:
        ax.set_ylabel(r"$\varphi/\pi$")
    return mesh


def colorbar(fig, mesh, cax, label):
    bar = fig.colorbar(mesh, cax=cax)
    bar.set_label(label)
    if bar.solids is not None:
        bar.solids.set_rasterized(False)
    return bar


def save(fig, out, stem):
    paths = []
    for suffix in ("png", "pdf"):
        path = out / f"{stem}.{suffix}"
        fig.savefig(path, dpi=350, bbox_inches="tight", facecolor="white")
        paths.append(path.name)
    plt.close(fig)
    return paths


def render_maps(data, out):
    norms = norms_for(data)
    fig, axes = plt.subplots(2, 3, figsize=(14.7, 8.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.885, bottom=.15, top=.85,
                        wspace=.11, hspace=.32)
    for row, field in enumerate(FIELDS):
        for column in range(3):
            ax = axes[row, column]
            mesh = heatmap(ax, data, field, column, norms[row], row == 1, column == 0)
            if row == 0:
                ax.set_title(REGIMES[column]+f"\n$V/t_0={data['V'][column]:g}$", fontsize=11)
            ax.text(.025, 1.035, f"({chr(97+3*row+column)})", transform=ax.transAxes, fontsize=10)
        axes[row, 0].text(-.18, .5, ROW_LABELS[row], rotation=90,
                          transform=axes[row, 0].transAxes, va="center", ha="center", fontsize=10)
        box = axes[row, -1].get_position()
        cax = fig.add_axes([.902, box.y0, .015, box.height])
        colorbar(fig, mesh, cax, COLOR_LABELS[row])
    fig.suptitle("Exact time-ordered Floquet participation: 100 × 100 computed grid", y=.979, fontsize=15)
    fig.text(.5, .923, r"$L=89$, $A=B=0.8$, $W=0$, PBC; spin-resolved IPR; $D_n=-\ln(\mathrm{IPR}_n)/\ln89$",
             ha="center", fontsize=10)
    fig.text(.5, .043,
             "ω/t₀ = 0.2–10 and φ = 0–π, with both endpoints included; all 30,000 parameter points are independently evolved.\n"
             "Each row shares one color scale across the three static references. No interpolation or static subtraction is used.\n"
             "Finite-size participation indicators alone do not establish a localization phase transition.",
             ha="center", va="center", fontsize=8.7, linespacing=1.45)
    outputs = save(fig, out, "three_V_refined_IPR_Dn_summary")
    stems = ("mean_IPR", "mean_Dn")
    for row, field in enumerate(FIELDS):
        fig, axes = plt.subplots(1, 3, figsize=(14.7, 5.0), sharex=True, sharey=True)
        fig.subplots_adjust(left=.055, right=.887, bottom=.28, top=.77, wspace=.10)
        for column, ax in enumerate(axes):
            mesh = heatmap(ax, data, field, column, norms[row], True, column == 0)
            ax.set_title(REGIMES[column]+f"\n$V/t_0={data['V'][column]:g}$", fontsize=11)
        cax = fig.add_axes([.906, .28, .014, .49])
        colorbar(fig, mesh, cax, COLOR_LABELS[row])
        fig.suptitle(ROW_LABELS[row], y=.96, fontsize=14)
        fig.text(.5, .867, r"$L=89$, PBC; spin-resolved IPR; $D_n=-\ln(\mathrm{IPR}_n)/\ln89$", ha="center", fontsize=10)
        fig.text(.5, .057,
                 "100 true frequency values × 100 true phase values per V; shared color scale across the three static references.\n"
                 "ω/t₀ = 0.2–10; φ = 0–π. Both endpoints are included. No interpolation or static subtraction.",
                 ha="center", fontsize=8.5, linespacing=1.45)
        outputs += save(fig, out, f"three_V_refined_{stems[row]}")
    return outputs, {field: [float(norm.vmin), float(norm.vmax)] for field, norm in zip(FIELDS, norms)}


def read_slices(path):
    with np.load(path, allow_pickle=False) as file:
        data = {key: file[key] for key in ("V", "omega", "phi", *FIELDS, "converged")}
    if data["phi"].shape != (2,) or not np.allclose(data["phi"], [0., np.pi/2], atol=1e-13, rtol=0):
        raise ValueError("Frequency curves require exact phi=0 and separately calculated phi=pi/2")
    if data["omega"].shape != (100,) or not np.allclose(data["omega"], np.linspace(.2, 10., 100), atol=1e-13, rtol=0):
        raise ValueError("Frequency curve axis mismatch")
    for field in (*FIELDS, "converged"):
        if data[field].shape != (3, 2, 100) or not np.isfinite(data[field]).all():
            raise ValueError(f"Invalid frequency curve field {field}")
    if not np.asarray(data["converged"], bool).all():
        raise ValueError("Frequency curves contain unconverged points")
    return data


def render_slices(data, out):
    fig, axes = plt.subplots(2, 3, figsize=(14.7, 7.6), sharex=True, sharey="row")
    fig.subplots_adjust(left=.075, right=.975, bottom=.18, top=.80, wspace=.16, hspace=.24)
    palette = ("#2469a1", "#be4933")
    for row, field in enumerate(FIELDS):
        low, high = float(data[field].min()), float(data[field].max())
        pad = max(.07*(high-low), 1e-4)
        for column, ax in enumerate(axes[row]):
            for ip, label in enumerate((r"$\varphi=0$", r"$\varphi=\pi/2$")):
                ax.plot(data["omega"], data[field][column, ip], color=palette[ip],
                        linestyle="-" if ip == 0 else "--", marker=".", markersize=3,
                        linewidth=1.2, label=label)
            ax.set_xlim(.2, 10.)
            ax.set_ylim(max(0., low-pad), high+pad)
            ax.grid(alpha=.20, linewidth=.6)
            ax.set_xticks([.2, 2., 4., 6., 8., 10.])
            if row == 0:
                ax.set_title(REGIMES[column]+f"\n$V/t_0={data['V'][column]:g}$", fontsize=11)
            else:
                ax.set_xlabel(r"Dimensionless $\omega/t_0$")
            if column == 0:
                ax.set_ylabel(COLOR_LABELS[row])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .91), ncol=2, frameon=False)
    fig.suptitle("Full-spectrum frequency response for in-phase and quadrature drives", y=.983, fontsize=14)
    fig.text(.5, .063,
             "100 independently evolved frequencies per phase and V. The π/2 curves use additional exact calculations, not adjacent-grid interpolation.\n"
             r"Spin-resolved IPR; $D_n=-\ln(\mathrm{IPR}_n)/\ln89$; $L=89$, $A=B=0.8$, $W=0$, PBC.",
             ha="center", fontsize=8.7, linespacing=1.5)
    return save(fig, out, "three_V_refined_frequency_cuts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--verify-modal-statistics", action="store_true")
    args = parser.parse_args()
    grid_path = args.results / "frequency_phase_scan.npz"
    data = read_grid(grid_path)
    verification = verify_modes(grid_path, data) if args.verify_modal_statistics else {"modal_mean_check": "not requested"}
    outputs, norm_ranges = render_maps(data, args.results)
    slices_path = args.results / "frequency_phase_slices.npz"
    if slices_path.exists():
        outputs += render_slices(read_slices(slices_path), args.results)
    with (args.results / "frequency_phase_plot_data.npz").open("wb") as stream:
        np.savez_compressed(stream, **{key: data[key] for key in ("V", "omega", "phi", *FIELDS)},
                            L=L, interpolation="none; nearest-cell display of independently evolved grid values")
    summary = {"figures": outputs, "computed_grid_shape": [3, 100, 100], "computed_main_points": 30000,
               "omega_interval_inclusive": [.2, 10.], "phase_interval_inclusive": [0., float(np.pi)],
               "observables": list(FIELDS), "color_limits_shared_across_V": norm_ranges,
               "IPR_definition": "sum_j,spin |psi_jspin|^4",
               "Dn_definition": "mean_n[-ln(IPR_n)/ln(89)]", "interpolation": "none",
               "png_dpi": 350, "pdf": "vector mesh and vector colorbar", **verification}
    for filename in outputs:
        if filename.endswith(".pdf") and b"/Subtype /Image" in (args.results/filename).read_bytes():
            raise ValueError(f"PDF unexpectedly contains a raster image: {filename}")
    (args.results / "frequency_phase_figures_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
