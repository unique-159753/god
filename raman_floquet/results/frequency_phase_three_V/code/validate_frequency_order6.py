"""Independent full-space validation of the production sixth-order method."""
from __future__ import annotations

import os
for _key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_key] = "1"
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from frequency_baselines import parameters_for_V
from frequency_solver import reconstruct_operator, sector_bases, solve_point
from model import _kinetic_rotation, operators
from validate_frequency_solver import compare_reference, reference_sector_modes


def full_space_order6(omega,phi,params,steps):
    """Evolve the full identity independently of the Q+/C reconstruction."""
    dt=2*np.pi/(omega*steps)
    a,b,c=.7845136104775573,.2355732133593581,-1.177679984178871
    coefficients=(a,b,c,1-2*(a+b+c),c,b,a)
    z=operators(params)["z"]
    phases=[np.exp(-.5j*dt*x*z)[:,None] for x in coefficients]
    u=np.eye(178,dtype=complex).reshape(89,2,178)
    for k in range(steps):
        tau=k*dt
        for coefficient,phase in zip(coefficients,phases):
            length=coefficient*dt
            midpoint=tau+length/2
            u[:,0,:]*=phase
            u[:,1,:]*=phase.conj()
            u=_kinetic_rotation(u,length,params.t0+params.A*np.cos(omega*midpoint),
                                params.delta0+params.B*np.cos(omega*midpoint+phi))
            u[:,0,:]*=phase
            u[:,1,:]*=phase.conj()
            tau+=length
    return u.reshape(178,178)


def main():
    out=Path(__file__).parent/"results/frequency_phase_three_V"
    references=np.load(out/"frequency_solver_validation.npz",allow_pickle=False)
    rows=[];checks=[];arrays={};failed=[]
    for v in [.5,2.,4.]:
        p=parameters_for_V(v)
        for omega in [1.,10.]:
            tag=f'V{str(v).replace(".","p")}_w{int(omega)}'
            ref=reference_sector_modes(references[f"{tag}_reference_operator"],omega)
            for steps in [32,64,128,256]:
                solved=solve_point(omega,np.pi/3,p,steps,order=6)
                rows.append(dict(V=v,omega=omega,phi=np.pi/3,steps=steps,order=6,
                                 **compare_reference(solved,ref,omega)))
            # This full-space factorization does not use reconstructed sectors.
            physical=full_space_order6(omega,np.pi/3,p,64)
            forward=solve_point(omega,np.pi/3,p,64,order=6)
            reverse=solve_point(omega,5*np.pi/3,p,64,order=6)
            forward_u=reconstruct_operator(forward["sector_operator"])
            reverse_u=reconstruct_operator(reverse["sector_operator"])
            allowed=forward["parity_labels"][:,None]==reverse["parity_labels"][None,:]
            overlaps=abs(forward["modes"].T@reverse["modes"])**2
            rr,cc=linear_sum_assignment(np.where(allowed,-overlaps,10.))
            check=dict(V=v,omega=omega,order=6,steps=64,
                       full_space_operator_error=float(np.linalg.norm(physical-forward_u)/np.sqrt(178)),
                       phase_reversal_transpose_error=float(np.linalg.norm(reverse_u-forward_u.T)/np.sqrt(178)),
                       phase_reversal_max_Dn_error=float(np.max(abs(forward["Dn"][rr]-reverse["Dn"][cc]))),
                       minimum_phase_matched_overlap=float(overlaps[rr,cc].min()))
            checks.append(check)
            arrays[f"{tag}_full_space_M64"]=physical
            arrays[f"{tag}_sector_M64"]=forward["sector_operator"]
            print(json.dumps(check),flush=True)
            if check["full_space_operator_error"]>1e-9 or check["phase_reversal_transpose_error"]>1e-9 or check["phase_reversal_max_Dn_error"]>1e-6:
                failed.append(f"V{v},omega{omega}")
    summary=dict(scope="production order6: full178 identity propagation, ODE reference and phase reversal",
                 integration_order=6,main_Dn="-ln(IPR_component)/ln89",checks=checks,convergence_rows=rows,
                 frequency_solver_sha256=hashlib.sha256((Path(__file__).parent/"frequency_solver.py").read_bytes()).hexdigest(),
                 failures=failed,passed=not failed)
    (out/"frequency_order6_validation.json").write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(out/"frequency_order6_validation.npz",**arrays)
    with (out/"frequency_order6_validation.csv").open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if failed:raise SystemExit(str(failed))
    print('Sixth-order full-space validation PASS',flush=True)


if __name__=='__main__':main()
