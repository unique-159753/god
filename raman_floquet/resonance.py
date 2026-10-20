"""One-photon diagnostics and static-state-pair data for the Raman model.

Everything is calculated from the specified static matrix. No Floquet or
localization anomaly is assumed by this module. Arrays use (eta, phi, omega).
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.linalg import eigh


def independent_operators(L=89, alpha=55/89, t0=1., delta0=.5, V=2.5,
                          W=.01, A=.8, B=.8, theta=0.):
    """Construct directly from the individual bonds, independently of model.py."""
    N = 2 * L
    T, S = np.zeros((N, N)), np.zeros((N, N))
    for j in range(L):
        jp = (j + 1) % L
        a, b, ap, bp = 2*j, 2*j+1, 2*jp, 2*jp+1
        T[a, ap] -= 1; T[ap, a] -= 1
        T[b, bp] += 1; T[bp, b] += 1
        S[a, bp] += 1; S[bp, a] += 1
        S[b, ap] -= 1; S[ap, b] -= 1
    potential = V * np.cos(2*np.pi*alpha*np.arange(1, L+1)+theta)+W
    HZ = np.diag(np.repeat(potential, 2)*np.tile([1., -1.], L))
    return dict(T=T, S=S, HZ=HZ, H0=t0*T+delta0*S+HZ)


def localization(vectors):
    N = vectors.shape[0]
    density = np.sum(abs(vectors.reshape(N//2, 2, -1))**2, axis=1)
    ipr_component = np.sum(abs(vectors)**4, axis=0)
    ipr_site = np.sum(density**2, axis=0)
    return dict(density=density, ipr_component=ipr_component, ipr_site=ipr_site,
                D_component=-np.log(ipr_component)/np.log(N),
                D_site=-np.log(ipr_site)/np.log(N//2),
                D_component_logL=-np.log(ipr_component)/np.log(N//2))


def resonance_grid(E, Tmn, Smn, omega, phi, eta, A=.8, B=.8,
                   outer_fraction=.2):
    """Vectorize frequency and analytic cosine phase dependence."""
    N = len(E)
    outer = np.zeros(N, dtype=bool)
    count = int(np.ceil(outer_fraction*N))
    outer[np.argsort(abs(E), kind="stable")[-count:]] = True
    off_diagonal = ~np.eye(N, dtype=bool)
    involves_outer = outer[:, None] | outer[None, :]
    difference = E[:, None] - E[None, :]
    terms = np.stack([A*A*Tmn*Tmn/4, B*B*Smn*Smn/4,
                      A*B*Tmn*Smn/2])
    term_sums = np.empty((len(eta), 2, 3, len(omega)))
    unwindowed = np.empty((2, 3))
    for region, mask in enumerate([off_diagonal, off_diagonal & involves_outer]):
        unwindowed[region] = np.einsum("qmn,mn->q", terms, mask)/N
        for ie, width in enumerate(eta):
            kernel = 1/((difference[None, :, :] - omega[:, None, None])**2 + width**2)
            kernel *= mask[None, :, :]
            term_sums[ie, region] = np.einsum("qmn,wmn->qw", terms, kernel, optimize=True)/N
    response = (term_sums[:, :, 0, None, :] + term_sums[:, :, 1, None, :]
                + term_sums[:, :, 2, None, :]*np.cos(phi)[None, None, :, None])
    normalized = response * np.asarray(eta)[:, None, None, None]**2
    phase_upper_bounds = (unwindowed[:, 0, None] + unwindowed[:, 1, None]
                         + unwindowed[:, 2, None]*np.cos(phi)[None, :])
    return dict(R_full=response[:, 0], R_outer=response[:, 1],
                Rnorm_full=normalized[:, 0], Rnorm_outer=normalized[:, 1],
                term_sums=term_sums, unwindowed_terms=unwindowed,
                phase_upper_bounds=phase_upper_bounds,
                static_outer_mask=outer, static_outer_count=count,
                involves_outer_mask=involves_outer, energy_difference=difference)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = dict(L=89, alpha=55/89, t0=1., delta0=.5, V=2.5, W=.01,
                      A=.8, B=.8, theta=0.)
    ops = independent_operators(**parameters)
    shared_check = {"available": False}
    try:
        from model import operators, static_eigensystem, DEFAULT_PARAMS
        common = operators(DEFAULT_PARAMS)
        differences = {key: float(np.max(abs(ops[key]-common[key]))) for key in ops}
        if max(differences.values()) > 1e-13:
            raise RuntimeError(f"Independent and common operators disagree: {differences}")
        shared_check = {"available": True, "max_element_differences": differences}
    except ImportError:
        pass
    E, vectors = eigh(ops["H0"], driver="evd", check_finite=True)
    shared_basis_check={"available":False}
    if shared_check["available"]:
        common_E,common_V=static_eigensystem(DEFAULT_PARAMS)
        shared_basis_check={"available":True,
            "energy_max_absolute_difference":float(np.max(abs(E-common_E))),
            "minimum_diagonal_squared_overlap":float(np.min(np.diag(abs(common_V.T@vectors)**2)))}
    Tmn, Smn = vectors.T @ ops["T"] @ vectors, vectors.T @ ops["S"] @ vectors
    loc = localization(vectors)
    omega = np.linspace(5.8, 8., 111)
    phi = np.linspace(0., np.pi, 81)
    eta = np.array([.02, .05, .10])
    grid = resonance_grid(E, Tmn, Smn, omega, phi, eta)
    np.savez_compressed(output_dir / "resonance_static.npz", energies=E,
                        eigenvectors=vectors, Tmn=Tmn, Smn=Smn,
                        T=ops["T"], S=ops["S"], HZ=ops["HZ"], H0=ops["H0"],
                        static_outer_mask=grid["static_outer_mask"], **loc)
    np.savez_compressed(output_dir / "resonance_grid.npz", omega=omega, phi=phi,
                        eta=eta, **grid,
                        metadata=json.dumps({"parameters": parameters,
                          "axes": ["eta", "phi", "omega"],
                          "term_sums_axes": ["eta", "region(full,outer)", "term(T,S,cross*cos(phi))", "omega"],
                          "outer_rule": "at least one state among ceil(0.2*N)=36 largest abs(E)",
                          "diagonal": "exclude m=n", "absorption": "E_m-E_n≈omega uses H_minus1",
                          "R": "sum |M_mn|^2/((E_m-E_n-omega)^2+eta^2)/N",
                          "Rnorm": "eta^2 R; bounded weighted matrix-element sum, not probability"}))

    slice_phi=np.array([0.,np.pi/4,np.pi/3,np.pi/2,3*np.pi/4,np.pi])
    terms=grid["term_sums"]
    slices=(terms[:,:,0,None,:]+terms[:,:,1,None,:]
            +terms[:,:,2,None,:]*np.cos(slice_phi)[None,None,:,None])
    np.savez_compressed(output_dir/"resonance_slices.npz",omega=omega,
        phi=slice_phi,eta=eta,R_full=slices[:,0],R_outer=slices[:,1],
        Rnorm_full=slices[:,0]*eta[:,None,None]**2,
        Rnorm_outer=slices[:,1]*eta[:,None,None]**2,
        phase_note="All six requested phases evaluated directly; pi/3 is not interpolated")
    slice_rows=[]
    for ie,width in enumerate(eta):
        for ip,p in enumerate(slice_phi):
            for iw,w in enumerate(omega):
                slice_rows.append(dict(eta=width,omega=w,phi=p,phi_over_pi=p/np.pi,
                    R_full=slices[ie,0,ip,iw],R_outer=slices[ie,1,ip,iw],
                    Rnorm_full=slices[ie,0,ip,iw]*width**2,
                    Rnorm_outer=slices[ie,1,ip,iw]*width**2))
    write_csv(output_dir/"resonance_slices.csv",slice_rows)

    direct_error, phase_even_error = 0., 0.
    for p in [0., np.pi/7, np.pi/4, np.pi/3, np.pi/2, 3*np.pi/4, np.pi]:
        direct = abs(.4*Tmn+.4*np.exp(-1j*p)*Smn)**2
        analytic = .16*(Tmn*Tmn+Smn*Smn+2*Tmn*Smn*np.cos(p))
        direct_error = max(direct_error, float(np.max(abs(direct-analytic))))
        minus = abs(.4*Tmn+.4*np.exp(1j*p)*Smn)**2
        phase_even_error = max(phase_even_error, float(np.max(abs(direct-minus))))
    assert direct_error < 1e-12 and phase_even_error < 1e-12
    np.testing.assert_allclose(grid["Rnorm_full"], grid["R_full"]*eta[:,None,None]**2)
    assert np.all(grid["Rnorm_full"] <= grid["phase_upper_bounds"][0][None,:,None]+1e-12)
    for key in ["T", "S", "HZ", "H0"]:
        assert np.max(abs(ops[key]-ops[key].T)) < 1e-13
    eig_residual = np.linalg.norm(ops["H0"]@vectors-vectors*E)/np.sqrt(len(E))
    maxima = []
    for ie, width in enumerate(eta):
        for region in ["full", "outer"]:
            array = grid[f"R_{region}"][ie]
            ip, iw = np.unravel_index(np.argmax(array), array.shape)
            maxima.append(dict(eta=float(width), region=region,
                               omega=float(omega[iw]), phi=float(phi[ip]),
                               phi_over_pi=float(phi[ip]/np.pi), R=float(array[ip,iw]),
                               Rnorm=float(grid[f"Rnorm_{region}"][ie,ip,iw])))
    csv_rows = []
    for ie, width in enumerate(eta):
        for ip, p in enumerate(phi):
            for iw, w in enumerate(omega):
                csv_rows.append(dict(eta=width, omega=w, phi=p, phi_over_pi=p/np.pi,
                  **{key: float(grid[key][ie,ip,iw]) for key in
                     ["R_full", "R_outer", "Rnorm_full", "Rnorm_outer"]}))
    write_csv(output_dir/"resonance_grid.csv", csv_rows)
    static_rows = [dict(state=i, energy=E[i], outer=bool(grid["static_outer_mask"][i]),
        ipr_component=loc["ipr_component"][i], ipr_site=loc["ipr_site"][i],
        D_component=loc["D_component"][i], D_site=loc["D_site"][i],
        D_component_logL=loc["D_component_logL"][i]) for i in range(len(E))]
    write_csv(output_dir/"resonance_static_states.csv", static_rows)

    representative = [(6.2, np.pi/3), (6.2,0.), (6.2,np.pi/2), (8.,np.pi/2)]
    pair_rows, near_pair_rows, pair_summaries, density_records = [], [], [], []
    gaps = grid["energy_difference"]
    for case, (w,p) in enumerate(representative):
        M2 = abs(.4*Tmn+.4*np.exp(-1j*p)*Smn)**2
        detuning = gaps-w
        positive = gaps > 0
        near = positive & (abs(detuning) <= .10)
        # A zero-by-symmetry matrix element is not an observable resonance.
        coupled_near = near & (M2 > 1e-14)
        score = M2/(detuning**2+.05**2)
        indices = np.flatnonzero(positive)
        top = indices[np.argsort(score.ravel()[indices])[::-1][:10]]
        near_indices = np.flatnonzero(coupled_near)
        near_top = near_indices[np.argsort(score.ravel()[near_indices])[::-1][:10]]
        total_score = float(np.sum(score[positive]))
        top_scores = []
        for rank, linear in enumerate(top, start=1):
            m,n = map(int, np.unravel_index(linear, score.shape))
            row = dict(case=case, omega=w, phi=p, phi_over_pi=p/np.pi, rank=rank,
                       m=m,n=n,E_n=E[n],E_m=E[m],energy_gap=gaps[m,n],
                       detuning=detuning[m,n],T_squared=Tmn[m,n]**2,
                       S_squared=Smn[m,n]**2,M_squared=M2[m,n],
                       interference_coefficient=.32*Tmn[m,n]*Smn[m,n],
                       coupling_to_abs_detuning=float(np.sqrt(M2[m,n])/max(abs(detuning[m,n]),1e-300)),
                       isolated_two_level_mixing_diagnostic=float(4*M2[m,n]/(detuning[m,n]**2+4*M2[m,n])),
                       weighted_score_eta005=score[m,n],
                       score_fraction_all_positive=score[m,n]/total_score,
                       involves_static_outer=bool(grid["involves_outer_mask"][m,n]),
                       near_resonant_abs_detuning_le_010=bool(abs(detuning[m,n])<=.10))
            for label,index in [("n",n),("m",m)]:
                for metric in ["ipr_component", "ipr_site", "D_component", "D_site", "D_component_logL"]:
                    row[f"{metric}_{label}"]=loc[metric][index]
            pair_rows.append(row)
            density_records.append(np.stack([loc["density"][:,n],loc["density"][:,m]]))
            top_scores.append(float(score[m,n]/total_score))
        for rank,linear in enumerate(near_top,start=1):
            m,n=map(int,np.unravel_index(linear,score.shape))
            near_pair_rows.append(dict(case=case,omega=w,phi=p,rank=rank,m=m,n=n,
                E_n=E[n],E_m=E[m],energy_gap=gaps[m,n],detuning=detuning[m,n],
                T_squared=Tmn[m,n]**2,S_squared=Smn[m,n]**2,M_squared=M2[m,n],
                weighted_score_eta005=score[m,n],
                score_fraction_all_positive=score[m,n]/total_score,
                involves_static_outer=bool(grid["involves_outer_mask"][m,n])))
        pair_summaries.append(dict(case=case,omega=w,phi=p,phi_over_pi=p/np.pi,
            total_near_pairs=int(near.sum()),coupled_near_pairs=int(coupled_near.sum()),
            smallest_positive_gap_detuning=float(np.min(abs(detuning[positive]))),
            returned_pairs=len(top),ranking="M2/(detuning2+0.05²), all E_m>E_n; near-only pairs separately exported",
            top1_fraction_all_positive=top_scores[0],top10_fraction_all_positive=sum(top_scores),
            all_near_window_fraction_all_positive=float(score[coupled_near].sum()/total_score),
            coupled_pairs_with_absM_ge_absdetuning=int(np.count_nonzero(positive&(M2>1e-14)&(M2>=detuning**2))),
            no_coupled_pair_within_detuning_window_010=not bool(coupled_near.any()),
            omega_exceeds_all_static_energy_differences=bool(w>2*np.max(abs(E)))))
    write_csv(output_dir/"resonance_state_pairs.csv", pair_rows)
    write_csv(output_dir/"resonance_near_only_pairs.csv", near_pair_rows)
    np.savez_compressed(output_dir/"resonance_state_pairs.npz",
        omega=np.array([r[0] for r in representative]),phi=np.array([r[1] for r in representative]),
        density=np.asarray(density_records).reshape(4,10,2,89),
        m=np.array([r["m"] for r in pair_rows]).reshape(4,10),
        n=np.array([r["n"] for r in pair_rows]).reshape(4,10),
        detuning=np.array([r["detuning"] for r in pair_rows]).reshape(4,10),
        M_squared=np.array([r["M_squared"] for r in pair_rows]).reshape(4,10))

    # Endpoint monotonicity is measured, rather than presumed from a chosen pair.
    cross = grid["term_sums"][:,:,2]
    summary = dict(parameters=parameters, N=178, static_outer_count=grid["static_outer_count"],
       static_outer_rule="ceil(20% of 178)=36 largest |E|, at least one endpoint",
       energy_min=float(E[0]), energy_max=float(E[-1]), folding_reference_2Emax=float(2*np.max(abs(E))),
       static_eigen_residual=float(eig_residual),
       shared_model_operator_crosscheck=shared_check,
       shared_static_basis_crosscheck=shared_basis_check,
       state_index_base=0,physical_site_index_base=1,
       matrix_element_decomposition_max_error=direct_error,
       phi_to_minus_phi_matrix_element_max_error=phase_even_error,
       phase_dependence="R(omega,phi;eta)=a(omega;eta)+b(omega;eta)*cos(phi), exact for this real static basis",
       cross_coefficient_range=[float(cross.min()),float(cross.max())],
       cross_coefficient_negative_at_all_grid_points=bool(np.all(cross < 0)),
       interior_phase_maximum_count=int(np.count_nonzero(
           (np.argmax(grid["R_full"],axis=1)>0)&(np.argmax(grid["R_full"],axis=1)<len(phi)-1))),
       maxima=maxima, representative_pairs=pair_summaries,
       Rnorm_quantity="eta²*R, bounded by unwindowed off-diagonal matrix-element norm; not a transition probability",
       eta_window_caution="The |detuning|<=0.10 diagnostic window is not a physical resonance criterion; |M|/|detuning| can be large outside it.",
       Rnorm_upper_bound_range=[float(grid["phase_upper_bounds"].min()),float(grid["phase_upper_bounds"].max())])
    (output_dir/"resonance_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({key:summary[key] for key in ["energy_max","folding_reference_2Emax",
          "matrix_element_decomposition_max_error","cross_coefficient_range",
          "cross_coefficient_negative_at_all_grid_points","maxima","representative_pairs"]},indent=2),flush=True)
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent/"results")
    args=parser.parse_args(); run(args.output_dir)
