"""Full time-dependent single-particle Floquet calculation in exact symmetry sectors.

The reflection-spin symmetry is only used to choose a reproducible eigenbasis
and reduce dense linear algebra. No high-frequency approximation is used here.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import math,time
from functools import lru_cache
import numpy as np
from scipy import sparse
from scipy.linalg import eigh,schur
PARAMETERS=dict(t0=1.0,Delta0=0.5,V=2.5,W=0.01,A=0.8,B=0.8,theta=0.0,hbar=1.0,boundary='PBC')
METRIC_NAMES=('D_all','D_outer','IPR_all','IPR_outer')

@lru_cache(maxsize=8)
def operators(L,p):
    """Physical sites 1..L; alpha=p/L. Return full operators and Q=R sigma_z sectors."""
    n=2*L
    rows={key:[] for key in ['K','S','J']};cols={key:[] for key in rows};data={key:[] for key in rows}
    def add(key,a,b,v):
        rows[key].extend([a,b]);cols[key].extend([b,a]);data[key].extend([v,np.conjugate(v)])
    for j in range(L):
        k=(j+1)%L;r=(j+2)%L
        add('K',2*j,2*k,-1.);add('K',2*j+1,2*k+1,1.)
        add('S',2*j,2*k+1,1.);add('S',2*j+1,2*k,-1.)
        add('J',2*j,2*r+1,1j);add('J',2*j+1,2*r,1j)
    full={key:sparse.coo_matrix((data[key],(rows[key],cols[key])),shape=(n,n)).tocsr() for key in rows}
    # Exact argument reduction for rational alpha and theta=0. This evaluates
    # the specified cosine while preserving its reflection identity in float64.
    residues=(p*np.arange(1,L+1,dtype=np.int64))%L
    reduced=np.minimum(residues,L-residues)
    z_site=2.5*np.cos(2*np.pi*reduced/L)+0.01
    full['Z']=sparse.diags(np.repeat(z_site,2)*np.tile([1.,-1.],L),format='csr')
    reflection=(-np.arange(L)-2)%L
    rr=np.arange(n);cc=np.repeat(2*reflection,2)+np.tile([0,1],L)
    R=sparse.csr_matrix((np.tile([1.,-1.],L),(rr,cc)),shape=(n,n))
    qs={}
    for parity in [-1,1]:
        qr=[];qc=[];qv=[];column=0
        for j in range(L):
            k=int(reflection[j])
            if j>k:continue
            for spin,sgn in [(0,1),(1,-1)]:
                if j==k:
                    if parity!=sgn:continue
                    qr.append(2*j+spin);qc.append(column);qv.append(1.)
                else:
                    qr.extend([2*j+spin,2*k+spin]);qc.extend([column,column]);qv.extend([1/np.sqrt(2),parity*sgn/np.sqrt(2)])
                column+=1
        assert column==L
        qs[parity]=sparse.csr_matrix((qv,(qr,qc)),shape=(n,L))
    sectors={parity:{key:(q.T@h@q).toarray() for key,h in full.items()} for parity,q in qs.items()}
    def max_sparse(a):return float(np.max(abs(a.data))) if a.nnz else 0.
    diagnostics={
        'symmetry_commutator_max':max(max_sparse(R@h-h@R) for h in full.values()),
        'cross_sector_operator_max':max(max_sparse(qs[-1].T@h@qs[1]) for h in full.values()),
        'basis_orthogonality_max':max(max_sparse(q.T@q-sparse.eye(L)) for q in qs.values()),
        'commutator_sign_max':max_sparse((full['K']@full['S']-full['S']@full['K'])/(2j)-full['J']),
        'raw_cosine_roundoff_max':float(np.max(abs(z_site-(2.5*np.cos(2*np.pi*p/L*np.arange(1,L+1))+.01))))
    }
    assert diagnostics['symmetry_commutator_max']<1e-13
    assert diagnostics['commutator_sign_max']<1e-13
    return full,qs,sectors,diagnostics

def calculate(L,p,omega,phi,slices=32,model='exact',return_matrices=False):
    start=time.perf_counter();full,qs,sectors,op_checks=operators(L,p);period=2*np.pi/omega
    all_e=[];all_v=[];all_lam=[];all_parity=[];sector_U={};residuals=[];unitarity=[];ortho=[];herm=[];radii=[];sector_gaps=[];triangular_errors=[]
    for parity in [-1,1]:
        b=sectors[parity];K=b['K'];S=b['S'];Z=b['Z'];J=b['J']
        if model=='effective':
            h=K+.5*S+Z+(.64/omega*np.sin(phi))*J
            herm.append(float(np.max(abs(h-h.conj().T))))
            energy,v=eigh(h,driver='evd',check_finite=False)
            lam=np.exp(-1j*energy*period);e=-np.angle(lam)/period
            residuals.append(float(np.linalg.norm(h@v-v*energy,'fro')/(np.sqrt(L)*max(1,np.linalg.norm(h,2)))))
            unitarity.append(0.);radii.append(0.);triangular_errors.append(0.)
        elif model=='exact':
            u=np.eye(L,dtype=np.complex128)
            max_hermiticity=0.
            for m in range(slices):
                x=2*np.pi*(m+.5)/slices
                h=(1+.8*np.cos(x))*K+(.5+.8*np.cos(x+phi))*S+Z
                max_hermiticity=max(max_hermiticity,float(np.max(abs(h-h.T))))
                energy,v=eigh(h,driver='evd',check_finite=False)
                # Later times multiply on the left: U_m ... U_1 U_0.
                u=(v*np.exp(-1j*energy*period/slices))@(v.T@u)
            triangular,v=schur(u,output='complex',check_finite=False)
            lam=np.diag(triangular);e=-np.angle(lam)/period
            herm.append(max_hermiticity)
            residuals.append(float(np.linalg.norm(u@v-v*lam,'fro')/np.sqrt(L)))
            triangular_errors.append(float(np.linalg.norm(triangular-np.diag(lam),'fro')/np.sqrt(L)))
            unitarity.append(float(np.linalg.norm(u.conj().T@u-np.eye(L),'fro')/np.sqrt(L)))
            radii.append(float(np.max(abs(abs(lam)-1))))
            if return_matrices:sector_U[parity]=u
        else:raise ValueError(model)
        ortho.append(float(np.linalg.norm(v.conj().T@v-np.eye(L),'fro')/np.sqrt(L)))
        sector_gaps.append(float(np.min(np.diff(np.sort(e)))))
        all_e.append(e);all_lam.append(lam);all_v.append(np.asarray(qs[parity]@v));all_parity.append(np.full(L,parity,dtype=np.int8))
    e=np.concatenate(all_e);v=np.concatenate(all_v,axis=1);lam=np.concatenate(all_lam);parity=np.concatenate(all_parity)
    order=np.argsort(e,kind='stable');e=e[order];v=v[:,order];lam=lam[order];parity=parity[order]
    v/=np.linalg.norm(v,axis=0)
    ipr=np.sum(abs(v)**4,axis=0);D=-np.log(ipr)/np.log(2*L)
    n_outer=math.ceil(.2*2*L);outer=np.argsort(abs(e/omega),kind='stable')[-n_outer:]
    metrics=np.array([np.mean(D),np.mean(D[outer]),np.mean(ipr),np.mean(ipr[outer])])
    result=dict(L=L,alpha_numerator=p,omega=omega,phi=phi,slices=slices,model=model,
                metrics=metrics,quasienergy=e,ipr=ipr,D=D,parity=parity,outer_indices=outer,
                outer_count=n_outer,hermiticity_max=max(herm),unitarity_relative_fro=max(unitarity),
                eigen_residual_relative_fro=max(residuals),schur_offdiagonal_relative_fro=max(triangular_errors),
                eigenvector_orthogonality_relative_fro=max(ortho),eigenvalue_modulus_error=max(radii),
                min_full_quasienergy_gap=float(np.min(np.diff(e))),min_within_sector_quasienergy_gap=min(sector_gaps),
                **op_checks)
    if max(unitarity)>1e-10 or max(residuals)>1e-10 or max(ortho)>1e-11:
        raise ArithmeticError('Numerical residual threshold exceeded')
    if return_matrices:
        result['eigenvectors']=v;result['eigenvalues']=lam
        if model=='exact':result['sector_U']=sector_U
    result['seconds']=time.perf_counter()-start
    return result

def row(result):
    keys=['L','alpha_numerator','omega','phi','slices','model','outer_count','seconds','hermiticity_max','unitarity_relative_fro','eigen_residual_relative_fro','schur_offdiagonal_relative_fro','eigenvector_orthogonality_relative_fro','eigenvalue_modulus_error','min_full_quasienergy_gap','min_within_sector_quasienergy_gap','symmetry_commutator_max','cross_sector_operator_max','basis_orthogonality_max','raw_cosine_roundoff_max']
    return {**{k:result[k] for k in keys},**dict(zip(METRIC_NAMES,result['metrics']))}

if __name__=='__main__':
    r=calculate(89,55,8.,np.pi/2,32)
    import json
    print(json.dumps(row(r),indent=2))
