"""Optional reflection-preserving native time-ordered Floquet integrator.

This independent backend leaves frequency_solver.py untouched. Each split
Hamiltonian commutes with Q=reflection tensor sigma_z. The odd-ring closing
two bonds are exponentiated together as an exact three-site star gate.
Compilation requires a local C99 compiler and fails explicitly if unavailable;
this module never silently replaces native results with the FFT backend.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np
from scipy.linalg import schur

from frequency_solver import (reconstruct_modes, sector_bases, spin_swap,
                               symmetry_diagnostics)
from model import Parameters, fold_quasienergies, localization_observables, operators


_SOURCE = r"""
#include <stddef.h>
#include <math.h>
static void bond_group(double *u,size_t L,size_t cols,int group,
                       double p,double q,double r,double s) {
    size_t central=(L-3)/2;
    size_t start=group==2?central:(size_t)group;
    size_t end=group==2?central+1:L-2;
    for(size_t site=start;site<end;site+=2) {
        if(group!=2 && site==central) continue;
        size_t next=site+1;
        double *a=u+2*(2*site)*cols,*c=u+2*(2*site+1)*cols;
        double *b=u+2*(2*next)*cols,*d=u+2*(2*next+1)*cols;
        for(size_t j=0;j<2*cols;j+=2) {
            double ar=a[j],ai=a[j+1],cr=c[j],ci=c[j+1];
            double br=b[j],bi=b[j+1],dr=d[j],di=d[j+1];
            a[j]=p*ar-q*cr-r*bi+s*di;
            a[j+1]=p*ai-q*ci+r*br-s*dr;
            c[j]=p*cr-q*ar-s*bi+r*di;
            c[j+1]=p*ci-q*ai+s*br-r*dr;
            b[j]=p*br+q*dr-r*ai-s*ci;
            b[j+1]=p*bi+q*di+r*ar+s*cr;
            d[j]=p*dr+q*br+s*ai+r*ci;
            d[j+1]=p*di+q*bi-s*ar-r*cr;
        }
    }
}
static void onsite(double*u,size_t L,size_t cols,const double*phase) {
    for(size_t site=0;site<L;site++) {
        double cr=phase[2*site],ci=phase[2*site+1];
        double*a=u+2*(2*site)*cols,*b=u+2*(2*site+1)*cols;
        for(size_t j=0;j<2*cols;j+=2) {
            double ar=a[j],ai=a[j+1],br=b[j],bi=b[j+1];
            a[j]=cr*ar-ci*ai;a[j+1]=cr*ai+ci*ar;
            b[j]=cr*br+ci*bi;b[j+1]=cr*bi-ci*br;
        }
    }
}
static void star(double*u,size_t L,size_t cols,double t,double d,double duration) {
    double sum=t*t+d*d,twotd=2*t*d,difference=t*t-d*d;
    double radius=sqrt(2*sum),alpha,beta;
    if(radius==0) { alpha=duration;beta=-.5*duration*duration; }
    else {alpha=sin(radius*duration)/radius;double z=sin(.5*radius*duration);beta=-2*z*z/(radius*radius);}
    double*left[2]={u+2*(2*(L-2))*cols,u+2*(2*(L-2)+1)*cols};
    double*center[2]={u+2*(2*(L-1))*cols,u+2*(2*(L-1)+1)*cols};
    double*right[2]={u,u+2*cols};
    for(size_t j=0;j<2*cols;j+=2) {
        double x[6][2],h[6][2],h2[6][2];
        for(int z=0;z<2;z++) {
            x[0][z]=left[0][j+z];x[1][z]=left[1][j+z];
            x[2][z]=center[0][j+z];x[3][z]=center[1][j+z];
            x[4][z]=right[0][j+z];x[5][z]=right[1][j+z];
            h[0][z]=-t*x[2][z]+d*x[3][z];
            h[1][z]=-d*x[2][z]+t*x[3][z];
            h[2][z]=-t*x[0][z]-d*x[1][z]-t*x[4][z]+d*x[5][z];
            h[3][z]=d*x[0][z]+t*x[1][z]-d*x[4][z]+t*x[5][z];
            h[4][z]=-t*x[2][z]-d*x[3][z];
            h[5][z]=d*x[2][z]+t*x[3][z];
            h2[0][z]=sum*x[0][z]+twotd*x[1][z]+difference*x[4][z];
            h2[1][z]=twotd*x[0][z]+sum*x[1][z]+difference*x[5][z];
            h2[2][z]=2*sum*x[2][z];h2[3][z]=2*sum*x[3][z];
            h2[4][z]=difference*x[0][z]+sum*x[4][z]-twotd*x[5][z];
            h2[5][z]=difference*x[1][z]-twotd*x[4][z]+sum*x[5][z];
        }
        double*positions[6]={left[0],left[1],center[0],center[1],right[0],right[1]};
        for(int k=0;k<6;k++) {
            positions[k][j]=x[k][0]+alpha*h[k][1]+beta*h2[k][0];
            positions[k][j+1]=x[k][1]-alpha*h[k][0]+beta*h2[k][1];
        }
    }
}
void strang(double*u,size_t L,size_t cols,double t,double d,double duration,
            const double*phase) {
    double ct=cos(.5*t*duration),st=sin(.5*t*duration);
    double cd=cos(.5*d*duration),sd=sin(.5*d*duration);
    double p=ct*cd,q=st*sd,r=st*cd,s=ct*sd;
    bond_group(u,L,cols,0,p,q,r,s);bond_group(u,L,cols,1,p,q,r,s);
    bond_group(u,L,cols,2,p,q,r,s);star(u,L,cols,t,d,.5*duration);
    onsite(u,L,cols,phase);
    star(u,L,cols,t,d,.5*duration);bond_group(u,L,cols,2,p,q,r,s);
    bond_group(u,L,cols,1,p,q,r,s);bond_group(u,L,cols,0,p,q,r,s);
}
void apply_star(double*u,size_t L,size_t cols,double t,double d,double duration) {
    star(u,L,cols,t,d,duration);
}
"""
_TEMP=None
_LIB=None
_LOCK=threading.Lock()


@lru_cache(maxsize=1)
def native_library():
    global _TEMP,_LIB
    with _LOCK:
        if _LIB is not None:return _LIB
        cc=shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if cc is None:raise RuntimeError("Native reflection backend requires a C99 compiler")
        _TEMP=tempfile.TemporaryDirectory(prefix="raman_reflection_native_")
        directory=Path(_TEMP.name)
        (directory/"rotations.c").write_text(_SOURCE)
        process=subprocess.run([cc,"-O3","-std=c99","-fPIC","-shared","-ffp-contract=off",
                                str(directory/"rotations.c"),"-o",str(directory/"rotations.so"),"-lm"],
                               capture_output=True,text=True,timeout=30)
        if process.returncode:raise RuntimeError(process.stderr)
        lib=ctypes.CDLL(str(directory/"rotations.so"))
        lib.strang.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,
                             ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_void_p]
        lib.strang.restype=None
        lib.apply_star.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,
                                 ctypes.c_double,ctypes.c_double,ctypes.c_double]
        lib.apply_star.restype=None
        _LIB=lib
        return lib


def propagate_plus(omega,phi,params,steps,order=6):
    if omega<=0 or not np.isfinite(omega) or steps<1 or int(steps)!=steps:
        raise ValueError("Require omega>0 and positive integer steps")
    if params.L<3 or params.L%2!=1:
        raise ValueError("Reflection-preserving coloring requires an odd PBC ring")
    steps=int(steps)
    if order==6:
        a,b,c=.7845136104775573,.2355732133593581,-1.177679984178871
        weights=(a,b,c,1-2*(a+b+c),c,b,a)
    elif order==4:
        a=1/(2-2**(1/3));weights=(a,1-2*a,a)
    elif order==2:weights=(1.,)
    else:raise ValueError("order must be 2,4,6")
    lib=native_library()
    dt=2*np.pi/(omega*steps)
    z=operators(params)["z"]
    phases=[np.ascontiguousarray(np.exp(-1j*dt*w*z)) for w in weights]
    u=np.array(sector_bases(params.L)[0],dtype=np.complex128,order="C")
    for k in range(steps):
        tau=k*dt
        for weight,phase in zip(weights,phases):
            duration=weight*dt
            mid=tau+duration/2
            t=params.t0+params.A*np.cos(omega*mid)
            d=params.delta0+params.B*np.cos(omega*mid+phi)
            lib.strang(u.ctypes.data,params.L,params.L,t,d,duration,phase.ctypes.data)
            tau+=duration
    return u


def solve_point(omega,phi,params,steps,return_operator=True,order=6):
    """Same observable/raw coefficient contract as frequency_solver.solve_point."""
    started=time.perf_counter()
    sym=symmetry_diagnostics(params)
    if max(sym.values())>1e-10:raise ValueError("Required reflection/chiral symmetry absent")
    plus,_=sector_bases(params.L)
    propagated=propagate_plus(omega,phi,params,steps,order)
    propagation_done=time.perf_counter()
    sector=plus.T@propagated
    projection=float(np.linalg.norm(propagated-plus@sector)/np.sqrt(params.L))
    triangular,vectors=schur(sector,output="complex",check_finite=False)
    values=np.diag(triangular).copy()
    eps=fold_quasienergies(-np.angle(values)*omega/(2*np.pi),omega)
    index=np.argsort(eps,kind="stable");eps,vectors,values=eps[index],vectors[:,index],values[index]
    all_eps=np.r_[eps,fold_quasienergies(-eps,omega)]
    order_index=np.argsort(all_eps,kind="stable")
    modes=reconstruct_modes(vectors,order_index,params.L)
    gram=propagated.conj().T@propagated-np.eye(params.L)
    cross=propagated.conj().T@spin_swap(propagated.conj(),params.L)
    angles=np.sort(np.mod(np.angle(np.r_[values,values.conj()]),2*np.pi))
    gaps=np.diff(np.r_[angles,angles[0]+2*np.pi])
    stats=localization_observables(modes)
    stats["D_component_log2L"]=stats["D_component"]
    stats["D_component"]=stats["D_component_logL"]
    result=dict(omega=float(omega),phi=float(phi),steps=int(steps),order=int(order),
                quasienergies=all_eps[order_index],modes=modes,
                sector_vectors=vectors,sector_operator=sector,sector_quasienergies=eps,
                sector_eigenvalues=values,reconstruction_order=order_index,
                parity_labels=np.r_[np.ones(params.L,int),-np.ones(params.L,int)][order_index],
                sector_parity=np.ones(params.L,int),
                unitarity_error=float(np.sqrt((np.linalg.norm(gram)**2+np.linalg.norm(cross)**2)/params.L)),
                sector_unitarity_error=float(np.linalg.norm(sector.conj().T@sector-np.eye(params.L))/np.sqrt(params.L)),
                eigen_residual=float(np.linalg.norm(propagated@vectors-plus@vectors*values[None,:])/np.sqrt(params.L)),
                orthogonality_error=float(np.linalg.norm(vectors.conj().T@vectors-np.eye(params.L))/np.sqrt(params.L)),
                eigenvalue_modulus_error=float(np.max(abs(abs(values)-1))),
                sector_projection_error=projection,
                schur_offdiagonal_error=float(np.linalg.norm(triangular-np.diag(np.diag(triangular)))/np.sqrt(params.L)),
                min_eigenphase_gap=float(gaps.min()),degenerate_eigenphase_gaps=int(np.count_nonzero(gaps<1e-9)),
                hermiticity_error=max(x for k,x in sym.items() if k.startswith("Hermiticity_")),
                Q_commutator_error=max(x for k,x in sym.items() if k.startswith("Q_commutator_")),
                particle_hole_error=max(x for k,x in sym.items() if k.startswith("C_anticommutator_")),
                propagation_seconds=propagation_done-started,total_seconds=time.perf_counter()-started,
                backend=f"reflection_native_star_Yoshida{order}",**stats)
    result["native_backend"]="reflection_preserving_star_C99"
    result["native_compile_failure_policy"]="raise RuntimeError; no silent FFT fallback"
    result["Dn"]=result["D_component_logL"]
    if return_operator:result["propagated_plus"]=propagated
    return result
