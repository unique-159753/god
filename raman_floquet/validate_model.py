"""Independent Raman Floquet validation, including the selected outer spectrum.

Run with BLAS threads limited to one.  All test inputs are fixed; this script
does not construct or alter the production parameter-grid data.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import inspect
import json
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eig, expm, svdvals
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

import model


def independent_operators(p):
    n = 2 * p.L
    T = np.zeros((n, n))
    S = np.zeros((n, n))
    J2 = np.zeros((n, n), dtype=complex)
    for j in range(p.L):
        k = (j + 1) % p.L
        for spin, sign in [(0, -1.), (1, 1.)]:
            a, b = 2*j+spin, 2*k+spin
            T[a, b] += sign
            T[b, a] += sign
        for a, b, sign in [(2*j, 2*k+1, 1.), (2*j+1, 2*k, -1.)]:
            S[a, b] += sign
            S[b, a] += sign
        q = (j + 2) % p.L
        for spin in [0, 1]:
            a, b = 2*j+spin, 2*q+1-spin
            J2[a, b] += 1j
            J2[b, a] -= 1j
    field = p.V * np.cos(2*np.pi*p.alpha*np.arange(1,p.L+1)+p.theta) + p.W
    HZ = np.diag(np.column_stack([field, -field]).ravel())
    return dict(T=T, S=S, HZ=HZ, J2=J2, H0=p.t0*T+p.delta0*S+HZ)


def reference_propagator(p, omega, phi, compensate=False):
    ops = independent_operators(p)
    T, S, H0, J2 = [csr_matrix(ops[key]) for key in ["T", "S", "H0", "J2"]]
    n = 2*p.L
    g = p.A*p.B*np.sin(phi)/omega if compensate else 0.
    def rhs(tau, flat):
        u = flat.reshape(n,n)
        h_u = H0 @ u + p.A*np.cos(omega*tau)*(T@u) + p.B*np.cos(omega*tau+phi)*(S@u)
        if g:
            h_u -= g*(J2@u)
        return (-1j*h_u).ravel()
    started = time.perf_counter()
    solution = solve_ivp(rhs, [0.,2*np.pi/omega], np.eye(n,dtype=complex).ravel(),
                         method="DOP853", rtol=2e-12, atol=2e-14, t_eval=[2*np.pi/omega])
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[:,-1].reshape(n,n), {
        "method":"DOP853", "rtol":2e-12,"atol":2e-14,
        "rhs_evaluations":solution.nfev, "seconds":time.perf_counter()-started}


def independent_modes(u, omega, L):
    values, vectors = eig(u, check_finite=False)
    vectors /= np.linalg.norm(vectors,axis=0)[None,:]
    energies = (-np.angle(values)*omega/(2*np.pi) + omega/2) % omega - omega/2
    order = np.argsort(energies)
    vectors, energies = vectors[:,order], energies[order]
    density = (abs(vectors.reshape(L,2,-1))**2).sum(axis=1)
    ipr = np.sum(density**2,axis=0)
    return {"modes":vectors,"quasienergies":energies,"D_site":-np.log(ipr)/np.log(L)}


def outer_indices(energies):
    count = int(np.ceil(.2*len(energies)))
    return np.sort(np.argsort(abs(energies), kind="stable")[-count:])


def comparison(reference, candidate):
    ref_i = outer_indices(reference["quasienergies"])
    can_i = outer_indices(candidate["quasienergies"])
    overlap = abs(reference["modes"].conj().T @ candidate["modes"])**2
    rows, columns = linear_sum_assignment(-overlap)
    mapping = np.empty(len(rows),dtype=int)
    mapping[rows] = columns
    singular = svdvals(reference["modes"][:,ref_i].conj().T @ candidate["modes"][:,can_i])
    matched_singular = svdvals(reference["modes"][:,ref_i].conj().T @ candidate["modes"][:,mapping[ref_i]])
    ref_mean = float(np.mean(reference["D_site"][ref_i]))
    can_mean = float(np.mean(candidate["D_site"][can_i]))
    return {"outer_count":len(ref_i),"reference_outer_mean_D":ref_mean,
            "candidate_outer_mean_D":can_mean,"outer_mean_D_absolute_error":abs(can_mean-ref_mean),
            "outer_mean_D_relative_error":abs(can_mean-ref_mean)/max(abs(ref_mean),1e-15),
            "outer_subspace_mean_overlap":float(np.mean(singular**2)),
            "outer_subspace_min_squared_singular_value":float(np.min(singular**2)),
            "matched_outer_subspace_mean_overlap":float(np.mean(matched_singular**2)),
            "outer_membership_changes_after_matching":int(np.count_nonzero(~np.isin(mapping[ref_i],can_i))),
            "minimum_matched_mode_overlap":float(np.min(overlap[rows,columns])),
            "minimum_outer_matched_mode_overlap":float(np.min(overlap[ref_i,mapping[ref_i]]))}


def build(omega,phi,steps,p,compensate=False,backend=None):
    kwargs=dict(steps=steps,params=p,compensate=compensate)
    if backend and "backend" in inspect.signature(model.build_floquet_operator).parameters:
        kwargs["backend"] = backend
    return model.build_floquet_operator(omega,phi,**kwargs)


def main():
    p = model.DEFAULT_PARAMS
    omega, phi = 6.2, np.pi/3
    report = {"scope":"Independent solver validation, not production results",
              "parameters":asdict(p),"outer_definition":"top ceil(0.2*2L) of absolute centered-folded quasienergy; site IPR/log L",
              "representative_point":{"omega":omega,"phi":float(phi)},"checks":{}}
    own, independent = model.operators(p), independent_operators(p)
    report["checks"]["operator_construction_max_error"] = max(float(np.max(abs(own[k]-independent[k]))) for k in ["T","S","HZ","J2","H0"])
    report["checks"]["hermiticity_max_error"] = max(float(np.max(abs(own[k]-own[k].conj().T))) for k in ["T","S","HZ","J2","H0"])
    h = model.fourier_components(phi,p)
    comm = (h[1]@h[-1]-h[-1]@h[1])/omega
    report["checks"]["Fourier_commutator_positive_gJ_max_error"] = float(np.max(abs(comm-model.g_f(omega,phi,p)*own["J2"])))
    report["checks"]["kick_generator_max_error"] = float(np.max(abs(model.kick_operator(omega,phi,p)-p.B*np.sin(phi)/omega*own["S"])))
    identity = np.eye(2*p.L,dtype=complex)
    kick_direct = expm(-1j*model.kick_operator(omega,phi,p))
    report["checks"]["kick_exponential_max_error"] = float(np.max(abs(kick_direct-model.apply_kick(identity,omega,phi,p))))

    small = replace(p,L=7,alpha=4/7)
    references = {}
    convergence = []
    for name,params,compensate in [("small",small,False),("full",p,False),("compensated_full",p,True)]:
        reference_u, ode = reference_propagator(params,omega,phi,compensate)
        reference = independent_modes(reference_u,omega,params.L)
        references[name] = reference_u
        reference_unitarity = float(np.linalg.norm(reference_u.conj().T@reference_u-np.eye(2*params.L))/np.sqrt(2*params.L))
        entries=[]
        # Compensation contains a complex NNN term and deliberately takes
        # the FFT route; otherwise test both independent factorizations.
        backends=["auto"] if compensate else ["auto","fft"]
        for backend in backends:
            for count in [32,64,128]:
                started=time.perf_counter()
                u=build(omega,phi,count,params,compensate,backend=backend)
                modes=model.floquet_modes(u,omega)
                backend_used = ("fft" if compensate or backend == "fft" else
                                model.numerical_backend() if hasattr(model,"numerical_backend") else "fft")
                row={"steps":count,"backend_requested":backend,"backend_used":backend_used,"seconds":time.perf_counter()-started,
                     "operator_relative_Frobenius_error":float(np.linalg.norm(u-reference_u)/np.sqrt(2*params.L)),
                     "unitarity_error":modes["unitarity_error"],"eigen_residual":modes["eigen_residual"],
                     **comparison(reference,modes)}
                entries.append(row)
                print(name,backend,count,row["operator_relative_Frobenius_error"],row["outer_mean_D_absolute_error"],flush=True)
        ratios=[]
        for backend in backends:
            seq=[entry for entry in entries if entry["backend_requested"]==backend]
            ratios.append({"backend_requested":backend,"backend_used":seq[0]["backend_used"],
                           "operator_error_ratio_32_to_64":seq[0]["operator_relative_Frobenius_error"]/seq[1]["operator_relative_Frobenius_error"],
                           "operator_error_ratio_64_to_128":seq[1]["operator_relative_Frobenius_error"]/seq[2]["operator_relative_Frobenius_error"]})
        convergence.append({"target":name,"L":params.L,"compensate":compensate,"reference":ode,
                            "reference_unitarity_error":reference_unitarity,"resolutions":entries,
                            "fourth_order_ratios":ratios})
    report["convergence"] = convergence

    u=build(omega,phi,64,p)
    negative=build(omega,-phi,64,p)
    report["checks"]["phase_reversal_U_transpose_relative_error"] = float(np.linalg.norm(negative-u.T)/np.sqrt(2*p.L))
    pos_modes=model.floquet_modes(u,omega)
    neg_modes=model.floquet_modes(negative,omega)
    report["checks"]["phase_reversal_outer_D_error"] = abs(float(np.mean(pos_modes["D_site"][outer_indices(pos_modes["quasienergies"])]))-float(np.mean(neg_modes["D_site"][outer_indices(neg_modes["quasienergies"])])))
    chiral_u=u.conj().reshape(p.L,2,p.L,2)[:,::-1,:,::-1].reshape(2*p.L,2*p.L)
    report["checks"]["chiral_particle_hole_U_error"] = float(np.linalg.norm(chiral_u-u)/np.sqrt(2*p.L))

    # At high frequency the Fourier sign can be checked against the exact
    # stroboscopic unitary, rather than just against an algebraic identity.
    sign_checks=[]
    for frequency in [16.,32.]:
        u=build(frequency,phi,128,p)
        kick=expm(-1j*model.kick_operator(frequency,phi,p))
        period=2*np.pi/frequency
        g=model.g_f(frequency,phi,p)
        plus=kick@expm(-1j*period*(own["H0"]+g*own["J2"]))@kick.conj().T
        minus=kick@expm(-1j*period*(own["H0"]-g*own["J2"]))@kick.conj().T
        sign_checks.append({"omega":frequency,"correct_sign_operator_error":float(np.linalg.norm(u-plus)/np.sqrt(2*p.L)),
                            "wrong_sign_operator_error":float(np.linalg.norm(u-minus)/np.sqrt(2*p.L))})
    report["high_frequency_sign_comparison"] = sign_checks
    report["checks"]["high_frequency_correct_sign_better"] = all(x["correct_sign_operator_error"]<x["wrong_sign_operator_error"] for x in sign_checks)

    checks=report["checks"]
    passed=all(v<1e-10 for k,v in checks.items() if not isinstance(v,bool)) and checks["high_frequency_correct_sign_better"]
    passed &= all(row["operator_relative_Frobenius_error"]<1e-6
                  for case in convergence for row in case["resolutions"] if row["steps"]==128)
    passed &= all(row["outer_mean_D_absolute_error"]<1e-5
                  for case in convergence for row in case["resolutions"] if row["steps"]==128)
    passed &= all(10 < row["operator_error_ratio_32_to_64"] < 22 and
                  10 < row["operator_error_ratio_64_to_128"] < 22
                  for case in convergence for row in case["fourth_order_ratios"])
    report["production_M64_representative_passed"] = bool(all(
        row["outer_mean_D_absolute_error"] < 1e-5 and
        row["outer_subspace_mean_overlap"] > .999999 and
        row["outer_membership_changes_after_matching"] == 0
        for case in convergence for row in case["resolutions"] if row["steps"]==64))
    passed &= report["production_M64_representative_passed"]
    report["passed"] = bool(passed)
    target=Path(__file__).resolve().parent/"results"/"solver_validation.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"passed":report["passed"],"checks":checks,"file":str(target)},indent=2),flush=True)
    if not passed:
        raise SystemExit("Validation failed; inspect solver_validation.json")


if __name__=="__main__":
    main()
