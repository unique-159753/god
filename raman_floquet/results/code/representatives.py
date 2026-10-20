"""Measured static-pair projections and four-model Raman comparisons."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent/".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent/".cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import (DEFAULT_PARAMS, exact_modes, effective_modes,
                   localization_observables, static_eigensystem)
from observables import extract_stats, serializable_stats, summarize_pair


COLORS={"A_static":"#555555","B_effective":"#0072B2",
        "C_exact":"#D55E00","D_compensated":"#009E73"}
LABELS={"A_static":"A: static", "B_effective":"B: van Vleck + kick",
        "C_exact":"C: exact Floquet", "D_compensated":"D: exact, NNN compensated"}


def write_csv(path, rows):
    with path.open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def cached_result(root,kind,omega,phi,steps):
    directory=root/"representative_cache";directory.mkdir(exist_ok=True)
    key=f"{kind}_w{omega:.12g}_phi{phi:.12g}_M{steps}"
    path=directory/f"{key}.npz"
    if path.exists():
        with np.load(path,allow_pickle=False) as data:
            return {name:data[name] for name in data.files}
    if kind=="A_static":
        result=effective_modes(omega,phi,include_first_order=False,kick_corrected=False)
    elif kind=="B_effective":
        result=effective_modes(omega,phi,kick_corrected=True)
    else:
        result=exact_modes(omega,phi,steps=steps,compensate=(kind=="D_compensated"))
    result={name:value for name,value in result.items() if value is not None}
    result["model_kind"]=kind
    result["omega"]=float(omega)
    result["phi"]=float(phi)
    result["physical_drive_phase_zero"]=True
    np.savez_compressed(path,**result)
    print(f"Cached {kind}, omega={omega:g}, phi/pi={phi/np.pi:g}",flush=True)
    return result


def save_figure(fig,root,name):
    fig.savefig(root/f"{name}.png",dpi=350,bbox_inches="tight")
    fig.savefig(root/f"{name}.pdf",bbox_inches="tight")
    plt.close(fig)


def state_pair_cases(root,steps):
    with (root/"resonance_state_pairs.csv").open() as file:
        table=list(csv.DictReader(file))
    pair_summary=json.loads((root/"resonance_summary.json").read_text())["representative_pairs"]
    E,V0=static_eigensystem()
    density=localization_observables(V0)["density"]
    cases=[(6.2,np.pi/3),(6.2,0.),(6.2,np.pi/2),(8.,np.pi/2)]
    outputs=[];projection_rows=[];all_projection=[];all_modes=[];all_quasienergies=[]
    for case,(omega,phi) in enumerate(cases):
        leading=next(row for row in table if int(row["case"])==case and int(row["rank"])==1)
        m,n=int(leading["m"]),int(leading["n"])
        result=cached_result(root,"C_exact",omega,phi,steps)
        stats=extract_stats(result)
        projections=abs(V0.conj().T@result["modes"])**2
        pair_weight=projections[m]+projections[n]
        alpha=int(np.argmax(pair_weight))
        outer=stats["outer_indices"]
        outer_alpha=int(outer[np.argmax(pair_weight[outer])])
        selected=projections[:,alpha]
        top=np.argsort(selected)[::-1][:10]
        row=dict(case=case,omega=omega,phi=phi,phi_over_pi=phi/np.pi,m=m,n=n,state_index_base=0,
            E_m=float(E[m]),E_n=float(E[n]),energy_gap=float(E[m]-E[n]),
            detuning=float(E[m]-E[n]-omega),M_squared=float(leading["M_squared"]),
            weighted_static_pair_fraction=float(leading["score_fraction_all_positive"]),
            selected_floquet_index=alpha,selected_quasienergy=float(result["quasienergies"][alpha]),
            selected_is_quasienergy_outer=bool(alpha in outer),
            selected_pair_projection=float(pair_weight[alpha]),
            selected_static_basis_effective_state_count=float(1/np.sum(selected**2)),
            selected_D_site=float(result["D_site"][alpha]),
            selected_D_component=float(result["D_component"][alpha]),
            best_outer_floquet_index=outer_alpha,
            best_outer_quasienergy=float(result["quasienergies"][outer_alpha]),
            best_outer_pair_projection=float(pair_weight[outer_alpha]),
            mean_outer_D_site=float(stats["outer_D_site"]),
            static_top1_resonance_fraction=pair_summary[case]["top1_fraction_all_positive"],
            static_top10_resonance_fraction=pair_summary[case]["top10_fraction_all_positive"],
            no_coupled_pair_within_detuning_window_010=pair_summary[case]["no_coupled_pair_within_detuning_window_010"],
            strongest_static_projection_states=[int(i) for i in top],
            strongest_static_projection_weights=[float(selected[i]) for i in top])
        outputs.append(row)
        for index in range(len(E)):
            projection_rows.append(dict(case=case,omega=omega,phi=phi,
               floquet_index=alpha,quasienergy=result["quasienergies"][alpha],
               static_state=index,static_energy=E[index],
               projection=selected[index],member_of_leading_pair=bool(index in (m,n))))
        all_projection.append(projections);all_modes.append(result["modes"])
        all_quasienergies.append(result["quasienergies"])
    write_csv(root/"representative_floquet_projections.csv",projection_rows)
    np.savez_compressed(root/"representative_floquet_projections.npz",static_energies=E,
        static_modes=V0,projection=np.asarray(all_projection),modes=np.asarray(all_modes),
        quasienergies=np.asarray(all_quasienergies),omega=np.array([c[0] for c in cases]),
        phi=np.array([c[1] for c in cases]),
        selected_indices=np.array([o["selected_floquet_index"] for o in outputs]),
        best_outer_indices=np.array([o["best_outer_floquet_index"] for o in outputs]))
    (root/"representative_state_pair_summary.json").write_text(json.dumps(outputs,indent=2)+"\n")

    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    lead=outputs[0];n,m=lead["n"],lead["m"]
    en,em=E[n],E[m];target=en+6.2
    ax=axes[0]
    ax.hlines(en,.1,.7,color="#0072B2",lw=2)
    ax.hlines(em,.9,1.5,color="#D55E00",lw=2)
    ax.hlines(target,.9,1.5,color=".45",ls="--",lw=1)
    ax.annotate("",xy=(.8,target),xytext=(.8,en),
                arrowprops=dict(arrowstyle="->",color="black",lw=1.3))
    ax.text(.05,en-.32,rf"$E_n={en:.4f}$",color="#0072B2",fontsize=9)
    ax.text(1.0,em-.4,rf"$E_m={em:.4f}$",color="#D55E00",fontsize=9)
    ax.text(.95,target+.18,rf"$E_n+\omega={target:.4f}$",fontsize=8)
    ax.text(.12,.5*(en+em),r"$\omega=6.2$"+"\n"+rf"$\delta={lead['detuning']:.4f}$",fontsize=10)
    ax.set_xlim(0,1.75);ax.set_ylim(en-.7,target+.8);ax.set_xticks([])
    ax.set_ylabel(r"Energy / $t_0$")
    ax.set_title("(a) Strong weighted pair; not isolated")
    ax=axes[1]
    j=np.arange(1,90)
    ax.plot(j,density[:,n],lw=1.2,label=rf"$n={n}$, $D={localization_observables(V0)['D_site'][n]:.3f}$")
    ax.plot(j,density[:,m],lw=1.2,label=rf"$m={m}$, $D={localization_observables(V0)['D_site'][m]:.3f}$")
    ax.set_xlabel("Site j");ax.set_ylabel(r"$\rho_j=|\psi_{j\uparrow}|^2+|\psi_{j\downarrow}|^2$")
    ax.set_title("(b) Static-state spatial densities");ax.legend(fontsize=8)
    ax=axes[2];alpha=lead["selected_floquet_index"]
    selected=all_projection[0][:,alpha]
    ax.vlines(E,0,selected,color=".55",lw=.8)
    ax.scatter(E[[n,m]],selected[[n,m]],s=28,c=["#0072B2","#D55E00"],zorder=3)
    ax.set_xlabel(r"Static energy $E_n/t_0$")
    ax.set_ylabel(r"$|\langle n|u_\alpha(0)\rangle|^2$")
    ax.set_title("(c) Exact Floquet projection")
    ax.text(.03,.96,rf"$\epsilon_\alpha={lead['selected_quasienergy']:.3f}$"+"\n"+
        rf"pair weight={lead['selected_pair_projection']:.3f}"+"\n"+
        f"outer-zone state: {lead['selected_is_quasienergy_outer']}",
        transform=ax.transAxes,va="top",fontsize=8)
    fig.suptitle(r"$L=89$, $\omega=6.2$, $\varphi=\pi/3$; site IPR convention",fontsize=11)
    save_figure(fig,root,"fig3_resonant_state_pairs")

    fig,axes=plt.subplots(4,2,figsize=(10.5,11.5),constrained_layout=True)
    for case,row in enumerate(outputs):
        n,m=row["n"],row["m"]
        axes[case,0].plot(j,density[:,n],lw=1,label=f"n={n}")
        axes[case,0].plot(j,density[:,m],lw=1,label=f"m={m}")
        axes[case,0].set_ylabel(r"$\rho_j$");axes[case,0].legend(fontsize=8)
        axes[case,0].set_title(rf"$\omega={row['omega']:g},\ \varphi/\pi={row['phi_over_pi']:.3g}$"+
            f"; delta={row['detuning']:.3f}")
        selected=all_projection[case][:,row["selected_floquet_index"]]
        axes[case,1].vlines(E,0,selected,color=".45",lw=.8)
        axes[case,1].scatter(E[[n,m]],selected[[n,m]],s=23,c=["#0072B2","#D55E00"],zorder=3)
        axes[case,1].set_ylabel("Static-basis weight")
        axes[case,1].set_title(f"pair weight={row['selected_pair_projection']:.3f}; "+
             f"outer={row['selected_is_quasienergy_outer']}")
    for ax in axes[:,0]: ax.set_xlabel("Site j")
    for ax in axes[:,1]: ax.set_xlabel(r"$E_n/t_0$")
    save_figure(fig,root,"fig3_representative_cases")
    return outputs


def four_models(root,steps):
    omega_axis=np.array([6.2,8.,12.,16.])
    phase_axis=np.array([0.,np.pi/4,np.pi/2])
    kinds=["A_static","B_effective","C_exact","D_compensated"]
    rows=[];differences=[]
    arrays={key:np.empty((3,4,4)) for key in ["outer_D_site","outer_D_component",
        "outer_D_component_logL","projected_D_site","projected_D_component",
        "outer_selection_gap"]}
    spectra={key:np.empty((3,4,4,178)) for key in
             ["quasienergies","D_site","D_component","D_component_logL","ipr_site","ipr_component"]}
    for ip,phi in enumerate(phase_axis):
        for iw,omega in enumerate(omega_axis):
            results={}
            for ik,kind in enumerate(kinds):
                result=cached_result(root,kind,omega,phi,steps)
                stats=extract_stats(result);results[kind]=result
                row=dict(omega=omega,phi=phi,phi_over_pi=phi/np.pi,model=kind,
                         **serializable_stats(stats))
                rows.append(row)
                for key in arrays: arrays[key][ip,iw,ik]=stats[key]
                for key in spectra: spectra[key][ip,iw,ik]=result[key]
            pair=summarize_pair(results["C_exact"],results["B_effective"])
            d=dict(omega=omega,phi=phi,phi_over_pi=phi/np.pi)
            for metric in ["outer_D_site","projected_D_site","outer_D_component","projected_D_component"]:
                vals={kind:extract_stats(results[kind])[metric] for kind in kinds}
                d[metric+"_C_minus_A"]=vals["C_exact"]-vals["A_static"]
                d[metric+"_B_minus_A"]=vals["B_effective"]-vals["A_static"]
                d[metric+"_D_minus_A"]=vals["D_compensated"]-vals["A_static"]
                d[metric+"_C_minus_D"]=vals["C_exact"]-vals["D_compensated"]
                d[metric+"_C_minus_B"]=vals["C_exact"]-vals["B_effective"]
            for key in ["delta_D_site","matched_delta_D_site","selection_delta_D_site",
                        "projected_delta_D_site","mean_outer_matched_overlap","min_outer_matched_overlap"]:
                d[key]=pair[key]
            differences.append(d)
    write_csv(root/"four_model_outer_comparison.csv",rows)
    write_csv(root/"four_model_differences.csv",differences)
    np.savez_compressed(root/"four_model_outer_comparison.npz",omega=omega_axis,
        phi=phase_axis,model_names=np.array(kinds),axes=np.array(["phi","omega","model"]),
        **arrays,**{f"spectrum_{key}":value for key,value in spectra.items()})
    fig,axes=plt.subplots(2,3,figsize=(12.8,7.1),sharex=True,sharey="row",constrained_layout=True)
    markers=["s","^","o","D"];lines=["--","-.","-",":"]
    for ip,phi in enumerate(phase_axis):
        for ik,kind in enumerate(kinds):
            for ir,metric in enumerate(["outer_D_site","projected_D_site"]):
                axes[ir,ip].plot(omega_axis,arrays[metric][ip,:,ik],color=COLORS[kind],
                    marker=markers[ik],ls=lines[ik],lw=1.3,ms=4.5,label=LABELS[kind])
        axes[0,ip].set_title([r"$\varphi=0$",r"$\varphi=\pi/4$",r"$\varphi=\pi/2$"][ip])
        for ir in range(2):
            axes[ir,ip].grid(alpha=.18,lw=.5)
            axes[ir,ip].text(.03,.97,f"({chr(97+ir*3+ip)})",transform=axes[ir,ip].transAxes,va="top")
            axes[ir,ip].set_xticks(omega_axis)
        axes[1,ip].set_xlabel(r"$\omega/t_0$")
    axes[0,0].set_ylabel(r"Outer Floquet-zone mean $D$ (site)")
    axes[1,0].set_ylabel(r"Static-outer-projector weighted $D$")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=4,fontsize=9)
    fig.suptitle("NNN cancellation and finite-frequency response; L=89, PBC",fontsize=12)
    save_figure(fig,root,"fig4_NNN_vs_resonance")
    summary={"steps":steps,"IPR_main":"spin-summed site probability, D=-log(IPR_site)/log(L)",
             "hard_outer":"top ceil(20% of 178)=36 |folded epsilon|, independently each model",
             "projected":"weighted by overlap with the fixed static outer subspace; denominator 36",
             "compensation":"H_D(t)=H(t)-AB sin(phi)/omega J2x",
             "differences_are_not_additive_causal_fractions":True,
             "comparison_rows":differences}
    (root/"four_model_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent/"results")
    parser.add_argument("--steps",type=int,default=128)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "pdf.fonttype":42,"ps.fonttype":42,"axes.spines.top":False,
                         "axes.spines.right":False})
    cases=state_pair_cases(args.output_dir,args.steps)
    four_models(args.output_dir,args.steps)
    print(json.dumps({"case_summaries":cases},indent=2),flush=True)


if __name__=="__main__":
    main()
