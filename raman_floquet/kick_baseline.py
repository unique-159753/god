"""Separate first-order lab-frame kick effects from NNN-compensated dynamics."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).resolve().parent/".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME",str(Path(__file__).resolve().parent/".cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import effective_modes
from observables import extract_stats
from representatives import cached_result


def run(root,steps=128):
    root=Path(root)
    omega=np.array([6.2,8.,12.,16.]);phi=np.array([0.,np.pi/4,np.pi/2])
    metrics=["outer_D_site","projected_D_site","outer_D_component","projected_D_component"]
    kinds=["A_static","A_kick","B_effective","C_exact","D_compensated"]
    measurements={metric:np.empty((3,4,5)) for metric in metrics}
    spectra={metric:np.empty((3,4,178)) for metric in
             ["quasienergies","ipr_site","ipr_component","D_site","D_component"]}
    outer_indices=np.empty((3,4,36),dtype=int)
    rows=[]
    for ip,p in enumerate(phi):
        for iw,w in enumerate(omega):
            results={kind:cached_result(root,kind,w,p,steps)
                     for kind in kinds if kind!="A_kick"}
            kicked=effective_modes(w,p,include_first_order=False,kick_corrected=True)
            results["A_kick"]=kicked
            stats={kind:extract_stats(result) for kind,result in results.items()}
            row=dict(omega=w,phi=p,phi_over_pi=p/np.pi)
            for metric in metrics:
                vals={kind:float(stats[kind][metric]) for kind in kinds}
                for ik,kind in enumerate(kinds):
                    measurements[metric][ip,iw,ik]=vals[kind]
                    row[f"{metric}_{kind}"]=vals[kind]
                row[f"{metric}_A_kick_minus_A"]=vals["A_kick"]-vals["A_static"]
                row[f"{metric}_Dcomp_minus_A_kick"]=vals["D_compensated"]-vals["A_kick"]
                row[f"{metric}_C_minus_B"]=vals["C_exact"]-vals["B_effective"]
                row[f"{metric}_Dcomp_minus_A"]=vals["D_compensated"]-vals["A_static"]
            for metric in spectra:spectra[metric][ip,iw]=kicked[metric]
            outer_indices[ip,iw]=stats["A_kick"]["outer_indices"]
            rows.append(row)
    with (root/"kick_only_baseline.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(root/"kick_only_baseline.npz",omega=omega,phi=phi,
        model_names=np.array(kinds),outer_indices=outer_indices,
        **measurements,**{f"A_kick_spectrum_{key}":value for key,value in spectra.items()})
    summary={"definition":"A_kick=H0 eigenvectors dressed with e^{-i K1(0)}, K1(0)=B sin(phi) S/omega; folded H0 energies select outer states",
        "purpose":"Dcomp-minus-bare-A contains micromotion dressing as well as higher-order effects. Dcomp-minus-A_kick removes this known first-order kick baseline.",
        "main_metric":"site-spin-summed finite-size D; both hard Floquet-zone outer and static-outer-projector weighted means retained",
        "not_additive_causal_attribution":True,
        "phi_zero_A_kick_equals_A_max_abs_error":max(abs(row[f"{metric}_A_kick_minus_A"])
            for row in rows if row["phi"]==0 for metric in metrics),
        "rows":rows}
    assert summary["phi_zero_A_kick_equals_A_max_abs_error"]<1e-12
    (root/"kick_only_baseline_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,3,figsize=(12.6,7),sharex=True,sharey="row",constrained_layout=True)
    series=[("A_kick minus A","#0072B2","o"),("Dcomp minus A_kick","#D55E00","s"),
            ("C minus B","#009E73","^")]
    for ip,p in enumerate(phi):
        for ir,metric in enumerate(["outer_D_site","projected_D_site"]):
            arr=measurements[metric]
            values=[arr[ip,:,1]-arr[ip,:,0],arr[ip,:,4]-arr[ip,:,1],arr[ip,:,3]-arr[ip,:,2]]
            for (label,color,marker),value in zip(series,values):
                axes[ir,ip].plot(omega,value,marker=marker,color=color,lw=1.2,ms=4.5,label=label)
            axes[ir,ip].axhline(0,color=".45",lw=.6)
            axes[ir,ip].set_xticks(omega);axes[ir,ip].grid(alpha=.15,lw=.5)
            axes[ir,ip].text(.03,.97,f"({chr(97+ir*3+ip)})",transform=axes[ir,ip].transAxes,va="top")
        axes[0,ip].set_title([r"$\varphi=0$",r"$\varphi=\pi/4$",r"$\varphi=\pi/2$"][ip])
        axes[1,ip].set_xlabel(r"$\omega/t_0$")
    axes[0,0].set_ylabel(r"Outer Floquet-zone $\Delta D$ (site)")
    axes[1,0].set_ylabel(r"Static-outer-projector weighted $\Delta D$")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,fontsize=9)
    fig.suptitle("Kick-only baseline and residual finite-frequency response",fontsize=12)
    fig.savefig(root/"figureS_kick_baseline.png",dpi=350,bbox_inches="tight")
    fig.savefig(root/"figureS_kick_baseline.pdf",bbox_inches="tight")
    plt.close(fig)
    for row in rows:
        print(json.dumps({key:row[key] for key in ["omega","phi_over_pi",
          "outer_D_site_A_kick_minus_A","outer_D_site_Dcomp_minus_A_kick",
          "outer_D_site_C_minus_B","projected_D_site_A_kick_minus_A",
          "projected_D_site_Dcomp_minus_A_kick"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent/"results")
    parser.add_argument("--steps",type=int,default=128)
    args=parser.parse_args();run(args.output_dir,args.steps)
