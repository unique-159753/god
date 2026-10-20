"""Paper-style Figure 5 from actual Sambe calculations, without fitted curves."""
from __future__ import annotations

import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).parent / ".cache"))

import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def render(output: str | Path = Path(__file__).parent / "results"):
    output = Path(output)
    summary = json.loads((output / "sambe_summary.json").read_text())
    records = summary["points"]
    settings = [(6.2, np.pi/3), (6.2, 0.), (6.2, np.pi/2), (8., np.pi/2)]
    case_labels = [r"$6.2,\pi/3$", r"$6.2,0$", r"$6.2,\pi/2$", r"$8,\pi/2$"]
    case_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 11, "axes.titlesize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.8),
                             gridspec_kw={"width_ratios": [1.15, 1.15, 1.35]})
    fig.subplots_adjust(left=.055, right=.97, bottom=.32, top=.84, wspace=.35)
    fig.suptitle("Floquet–Sambe truncation and multichannel mixing", y=.98, fontsize=14)
    plotted = []
    ax = axes[0]
    for case, (omega, phi) in enumerate(settings):
        rows = sorted([r for r in records if abs(r["omega"]-omega)<1e-12 and
                       abs(r["phi"]-phi)<1e-12], key=lambda r:r["mcut"])
        ax.semilogy([r["mcut"] for r in rows],
                    [abs(r["outer_mean_D_difference"]) for r in rows],
                    "o-", color=case_colors[case], ms=4, lw=1.3,
                    label=case_labels[case])
        for row in rows:
            plotted.append({"case":case,"omega":omega,"phi":phi,"mcut":row["mcut"],
                            "absolute_outer_D_error":abs(row["outer_mean_D_difference"]),
                            "raw_central_zone_count":row["raw_central_zone_candidate_count"],
                            "fallback_count":row["outside_zone_gauge_fallback_count"],
                            "outer_photon_participation":row["outer_mean_photon_participation"]})
    ax.set_xticks([1,2,3]); ax.set_xlabel(r"Photon cutoff $m_{\max}$")
    ax.set_ylabel(r"$|\overline{D}_{\rm Sambe}-\overline{D}_{\rm exact}|$")
    ax.set_title("(a) Same 36 exact-reference outer states")
    ax.grid(alpha=.22, which="major")
    ax.legend(title=r"$\omega/t_0,\varphi$", fontsize=8, title_fontsize=9, loc="upper right")
    ax.text(.03,.06,"mₘₐₓ=1 central-zone counts:\n162 (phase 0), 166 (phase π/3)\nmₘₐₓ=2,3: 178 for all four cases",transform=ax.transAxes,
            fontsize=8, va="bottom", bbox={"facecolor":"white","edgecolor":".8","alpha":.9})

    m3=[]
    for case in range(4):
        with np.load(output/f"sambe_point{case}_M3.npz", allow_pickle=False) as z:
            m3.append({key:z[key] for key in z.files if key!="summary"})
    categories=[-2,-1,0,1,2,"other"]
    photon_colors=["#542788","#998EC3","#D9D9D9","#F1A340","#B35806","#444444"]
    bottom=np.zeros(4)
    weights_table=[]
    for category,color in zip(categories,photon_colors):
        means=[]
        for case,z in enumerate(m3):
            outer=z["outer_indices_exact_reference"]
            mask=(abs(z["photon_axis"])>=3) if category=="other" else (z["photon_axis"]==category)
            value=float(z["photon_weights"][outer][:,mask].sum(axis=1).mean())
            means.append(value)
            weights_table.append({"case":case,"omega":float(z["omega"]),"phi":float(z["phi"]),
                                  "aligned_photon_sector":str(category),"outer_mean_weight":value})
        label=(r"$|m|\geq3$" if category=="other" else f"$m={category}$")
        axes[1].bar(np.arange(4),means,bottom=bottom,color=color,label=label,width=.65,
                    linewidth=.3,edgecolor="white")
        bottom+=means
    axes[1].set_ylim(0,1.02); axes[1].set_xticks(np.arange(4),case_labels)
    axes[1].set_xlabel(r"$(\omega/t_0,\varphi)$")
    axes[1].set_ylabel("Mean outer-mode photon weight")
    axes[1].set_title(r"(b) $m_{\max}=3$; centered-zone gauge")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6,fontsize=8,loc="upper center",
               bbox_to_anchor=(.50,.225),handlelength=1.2,columnspacing=1.2)
    for case,z in enumerate(m3):
        outer=z["outer_indices_exact_reference"]
        pp=float(z["photon_participation"][outer].mean())
        axes[1].text(case,.04,rf"$\langle P_\gamma\rangle$"+f"\n{pp:.3f}",
                     ha="center",va="bottom",fontsize=8,bbox={"facecolor":"white","alpha":.85,"edgecolor":"none"})

    z=m3[0]; outer=z["outer_indices_exact_reference"]
    mode=int(outer[np.argmax(z["photon_participation"][outer])])
    matrix=z["static_photon_weights"][mode]
    positive=matrix[matrix>0]
    norm=LogNorm(vmin=1e-7,vmax=float(np.max(positive)))
    image=axes[2].pcolormesh(np.arange(len(z["static_energies"])),z["photon_axis"],matrix,
                             shading="nearest",cmap="magma",norm=norm,rasterized=False)
    axes[2].set_xlabel(r"Energy-ordered static eigenstate $n$")
    axes[2].set_ylabel(r"Photon sector $m$")
    axes[2].set_yticks(z["photon_axis"])
    axes[2].set_xlim(-.5,len(z["static_energies"])-.5)
    axes[2].set_ylim(-3.5,3.5)
    eps=float(z["exact_quasienergies"][mode])
    axes[2].set_title(r"(c) $\omega/t_0=6.2,\ \varphi=\pi/3$"+f"\nexact ε/t₀={eps:.4f}; mode {mode}")
    colorbar=fig.colorbar(image,ax=axes[2],fraction=.05,pad=.025)
    colorbar.set_label(r"$|\langle n,m|u\rangle|^2$",fontsize=10)

    fig.text(.055,.13,"Exact reference: 178 physical Floquet states; the same highest-|ε| 36 states define each outer comparison. "
             "In-zone deficits at photon cutoff 1 are truncation failures; flagged fallback copies are not new bands.",fontsize=8)
    fig.text(.055,.085,"Photon participation is averaged per mode. A folded static state may already occupy m=±1; nonzero sidebands alone do not establish resonance. "
             "Panel (c) shows actual static-state/photon coefficients, with color floor 10⁻⁷.",fontsize=8)
    fig.text(.055,.04,"Photon cutoff 3: outer-mean D agrees with exact propagation within 1.2×10⁻⁵ at these points; "
             "reconstructed-wavefunction residuals remain 10⁻⁵–10⁻⁴, rather than machine precision.",fontsize=8)
    for suffix in ["png","pdf"]:
        fig.savefig(output/f"fig5_sambe_mixing.{suffix}",dpi=300,bbox_inches="tight")
    plt.close(fig)
    for filename,rows in [("sambe_figure_convergence.csv",plotted),("sambe_figure_photon_weights.csv",weights_table)]:
        with (output/filename).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(output/"sambe_figure_representative_mode.npz",mode_index=mode,
                        exact_quasienergy=eps,static_energies=z["static_energies"],
                        photon_axis=z["photon_axis"],weights=matrix)
    caption=("Figure 5. Actual Floquet-Sambe checks. (a) Absolute difference in the same exact-reference "
             "outer-mode mean D versus photon cutoff m_max (not the number of time slices). At omega=6.2 the m_max=1 central-zone candidate counts "
             "are 162 at phi=0 and 166 at phi=pi/3; out-of-zone fallback representatives are flagged and "
             "are not additional physical bands. m_max=2 and m_max=3 have 178 centered-zone candidates at all "
             "four tested points. (b) Mean outer-mode photon weights for m_max=3 in a fixed centered-zone "
             "gauge, with mean per-mode photon participation. A folded zero-drive state can occupy "
             "photon ±1 without any resonance. (c) Actual static-energy/photon coefficients for the "
             "outer mode with maximum photon participation at omega=6.2, phi=pi/3; columns label discrete energy-ordered static states. Multiple nonnegligible "
             "static channels occur; no uncomputed avoided crossing is inferred. m_max=3 average-D and "
             "quasienergy agreement do not imply machine-precision convergence of the reconstructed "
             "wavefunctions. Data and plotting code are exported alongside this figure.")
    (output/"fig5_sambe_mixing_caption.txt").write_text(caption+"\n")
    print(output/"fig5_sambe_mixing.png")
    print(output/"fig5_sambe_mixing.pdf")


if __name__=="__main__":
    render()
