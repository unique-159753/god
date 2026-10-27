"""Independent gates, full Hamiltonian, symmetries and ODE validation."""
from __future__ import annotations
import os
for _key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]:os.environ[_key]="1"
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment
from frequency_baselines import parameters_for_V,reflection_spin_operator
from frequency_fast_native import native_library,solve_point
from frequency_solver import solve_point as fft_solve,reconstruct_operator
from model import operators
from validate_frequency_solver import reference_sector_modes,compare_reference
from validate_model import reference_propagator


def main():
    out=Path(__file__).parent/"results/frequency_phase_three_V"
    out.mkdir(parents=True,exist_ok=True)
    t,d,h=.7,.3,.17
    p=parameters_for_V(4);q=reflection_spin_operator(89)
    b=np.array([[-t,d],[-d,t]])
    groups=[]
    edges=(list(range(0,87,2)),[i for i in range(1,87,2) if i!=43],[43],[87,88])
    for group in edges:
        matrix=np.zeros((178,178))
        for i in group:
            j=(i+1)%89
            matrix[2*i:2*i+2,2*j:2*j+2]=b
            matrix[2*j:2*j+2,2*i:2*i+2]=b.T
        groups.append(matrix)
    group_q=[float(np.linalg.norm(x@q-q@x)) for x in groups]
    sum_error=float(np.linalg.norm(sum(groups)-t*operators(p)["T"]-d*operators(p)["S"]))
    u=np.eye(178,dtype=complex)
    native_library().apply_star(u.ctypes.data,89,178,t,d,h)
    star_error=float(np.linalg.norm(u-expm(-1j*h*groups[-1])))
    rows=[];checks=[];arrays={}
    ode_refs=np.load(out/"frequency_solver_validation.npz",allow_pickle=False)
    for v,omega,phi in [(.5,1.,np.pi/3),(2.,1.,np.pi/3),(4.,1.,np.pi/3),
                        (4.,10.,np.pi/3),(.5,10/49,0.),(4.,10/49,np.pi/3)]:
        p=parameters_for_V(v)
        if omega in [1.,10.]:
            tag=f'V{str(v).replace(".","p")}_w{int(omega)}'
            ode=ode_refs[f"{tag}_reference_operator"]
        else:ode,_=reference_propagator(p,omega,phi)
        reference=reference_sector_modes(ode,omega)
        sequence=[256,512,1024,2048] if omega<1 else [32,64,128,256]
        for m in sequence:
            started=perf_counter();a=solve_point(omega,phi,p,m,order=6);elapsed=perf_counter()-started
            rows.append(dict(V=v,omega=omega,phi=phi,steps=m,seconds=elapsed,
                             projection_error=a["sector_projection_error"],unitarity_error=a["unitarity_error"],
                             **compare_reference(a,reference,omega)))
        started=perf_counter();fast=solve_point(omega,phi,p,sequence[-2],order=6);fast_time=perf_counter()-started
        started=perf_counter();fft=fft_solve(omega,phi,p,sequence[-2],order=6);fft_time=perf_counter()-started
        reverse=solve_point(omega,2*np.pi-phi,p,sequence[-2],order=6)
        a_u=reconstruct_operator(fast["sector_operator"]);reverse_u=reconstruct_operator(reverse["sector_operator"])
        overlap=abs(fast["modes"].T@reverse["modes"])**2
        allowed=fast["parity_labels"][:,None]==reverse["parity_labels"][None,:]
        rr,cc=linear_sum_assignment(np.where(allowed,-overlap,10.))
        check=dict(V=v,omega=omega,phi=phi,steps=sequence[-2],fast_seconds=fast_time,fft_seconds=fft_time,
                   speed_ratio=fft_time/fast_time,
                   native_vs_fft_U_difference=float(np.linalg.norm(fast["sector_operator"]-fft["sector_operator"])/np.sqrt(89)),
                   Q_projection_error=fast["sector_projection_error"],
                   phi_transpose_U_error=float(np.linalg.norm(reverse_u-a_u.T)/np.sqrt(178)),
                   phi_matched_Dn_error=float(np.max(abs(fast["Dn"][rr]-reverse["Dn"][cc]))))
        checks.append(check)
        arrays[f"case{len(checks)-1}_reference_U"]=ode
        arrays[f"case{len(checks)-1}_sector_vectors"]=a["sector_vectors"]
        arrays[f"case{len(checks)-1}_reconstruction_order"]=a["reconstruction_order"]
        print(json.dumps(check),flush=True)
    failures=[]
    if max(group_q+[sum_error,star_error])>1e-10:failures.append("gate/Hamiltonian decomposition")
    for c in checks:
        if max(c["Q_projection_error"],c["phi_transpose_U_error"])>1e-9 or c["phi_matched_Dn_error"]>1e-6:
            failures.append(f'symmetry V{c["V"]},omega{c["omega"]}')
    summary=dict(passed=not failures,failures=failures,groups=edges,group_Q_commutator_errors=group_q,
                 total_kinetic_reconstruction_error=sum_error,star_exact_exponential_error=star_error,
                 primary_Dn="-ln(IPR_component)/ln89",order=6,
                 backend_note="Different finite-step spatial factorization from FFT; both converge to the same time-ordered H",
                 native_sha256=hashlib.sha256((Path(__file__).parent/"frequency_fast_native.py").read_bytes()).hexdigest(),
                 checks=checks,convergence_rows=rows)
    (out/"frequency_fast_native_validation.json").write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(out/"frequency_fast_native_validation.npz",**arrays)
    with (out/"frequency_fast_native_validation.csv").open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if failures:raise SystemExit(str(failures))
    print('Reflection-preserving native validation PASS',flush=True)


if __name__=='__main__':main()
