"""Four explicitly separated Floquet comparison conventions, in one basis."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import math
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from floquet_core import operators,calculate
MODEL_NAMES=('exact','effective_unfolded_selection','effective_folded_selection','effective_folded_kick')
METRIC_NAMES=('D_all','D_outer','IPR_all','IPR_outer')
L=89;DIM=178;NOUT=math.ceil(.2*DIM)
QBIN_EDGES=np.linspace(0,.5,6)

def fold(E,w):return np.remainder(E+w/2,w)-w/2

def outer_indices(q):return np.argsort(abs(q),kind='stable')[-NOUT:]

def central_indices(q):return np.argsort(abs(q),kind='stable')[:NOUT]

def vector_metrics(v):
    norms=np.linalg.norm(v,axis=0)
    err=float(np.max(abs(norms-1)))
    assert err<1e-11
    ipr=np.sum(abs(v/norms)**4,axis=0)
    return ipr,-np.log(ipr)/np.log(DIM),err

def metrics(ipr,D,idx):return np.array([D.mean(),D[idx].mean(),ipr.mean(),ipr[idx].mean()])

def range_info(q,selection_coordinate,idx):
    return dict(outer_count=len(idx),selected_q_min=float(q[idx].min()),selected_q_max=float(q[idx].max()),selected_absq_min=float(abs(q[idx]).min()),selected_absq_max=float(abs(q[idx]).max()),selection_coordinate_abs_min=float(abs(selection_coordinate[idx]).min()),selection_coordinate_abs_max=float(abs(selection_coordinate[idx]).max()))

@lru_cache(maxsize=1)
def static_data():
    full,qs,blocks,diag=operators(L,55);Es=[];Vs=[];Ps=[];Seig={};parts={}
    for p in [-1,1]:
        h=blocks[p]['K']+.5*blocks[p]['S']+blocks[p]['Z'];e,v=eigh(h,driver='evd',check_finite=False)
        Es.append(e);Vs.append(np.asarray(qs[p]@v));Ps.append(np.full(L,p,dtype=np.int8));Seig[p]=eigh(blocks[p]['S'],driver='evd',check_finite=False);parts[p]=(e,v)
    E=np.concatenate(Es);V=np.concatenate(Vs,axis=1);P=np.concatenate(Ps);idx=np.argsort(E,kind='stable');E=E[idx];V=V[:,idx];P=P[idx]
    ipr,D,normerr=vector_metrics(V)
    return dict(energy=E,vectors=V,parity=P,ipr=ipr,D=D,parts=parts,Seig=Seig,omega_fold=2*float(abs(E).max()),Emaxabs=float(abs(E).max()),norm_error=normerr)

def match_by_overlap(Vleft,Vright,parity_left,parity_right,return_participation=False):
    assignment=np.empty(DIM,dtype=int);assigned=np.empty(DIM);pr=np.empty(DIM)
    for p in [-1,1]:
        a=np.flatnonzero(parity_left==p);b=np.flatnonzero(parity_right==p)
        O=abs(Vleft[:,a].conj().T@Vright[:,b])**2
        rr,cc=linear_sum_assignment(-O);assignment[a[rr]]=b[cc];assigned[a[rr]]=O[rr,cc]
        pr[a]=1/np.sum(O**2,axis=1)
    return assignment,assigned,pr

def effective(w,phi,controls=False,need_vectors=False):
    _,qs,blocks,_=operators(L,55);st=static_data();g=.64/w*np.sin(phi);gamma=.8/w*np.sin(phi)
    Es=[];bare=[];kick=[];Ps=[];zero=[]
    for p in [-1,1]:
        b=blocks[p];h=b['K']+.5*b['S']+b['Z']+g*b['J'];E,V=eigh(h,driver='evd',check_finite=False)
        se,sq=st['Seig'][p];Uk=(sq*np.exp(-1j*gamma*se))@sq.T
        Es.append(E);bare.append(np.asarray(qs[p]@V));kick.append(np.asarray(qs[p]@(Uk@V)));Ps.append(np.full(L,p,dtype=np.int8))
        if controls:zero.append(np.asarray(qs[p]@(Uk@st['parts'][p][1])))
    E=np.concatenate(Es);V=np.concatenate(bare,axis=1);Vk=np.concatenate(kick,axis=1);P=np.concatenate(Ps)
    order=np.argsort(E,kind='stable');E=E[order];V=V[:,order];Vk=Vk[:,order];P=P[order];q=fold(E,w)/w
    ipr,D,nerr=vector_metrics(V);iprk,Dk,nerrk=vector_metrics(Vk)
    unfolded=outer_indices(E/w);folded=outer_indices(q);central=central_indices(q)
    out=dict(E=E,q=q,D_bare=D,D_kick=Dk,ipr_bare=ipr,ipr_kick=iprk,parity=P,unfolded_indices=unfolded,folded_indices=folded,central_indices=central,
             metrics=np.stack([metrics(ipr,D,unfolded),metrics(ipr,D,folded),metrics(iprk,Dk,folded)]),g_F=float(g),kick_coefficient=float(gamma),normalization_error=max(nerr,nerrk),
             folded_state_count=int(np.count_nonzero(abs(E)>w/2+1e-12)),bare_vs_fold_replaced=int(NOUT-len(np.intersect1d(unfolded,folded))),
             ranges=[range_info(q,E/w,unfolded),range_info(q,q,folded),range_info(q,q,folded)])
    if controls:
        # Static states in the same raw-energy ordering as the cached H0 basis.
        Ez=np.concatenate([st['parts'][p][0] for p in [-1,1]]);Pz=np.concatenate([np.full(L,p,dtype=np.int8) for p in [-1,1]])
        Vz=np.concatenate(zero,axis=1);oz=np.argsort(Ez,kind='stable');Ez=Ez[oz];Pz=Pz[oz];Vz=Vz[:,oz]
        iprz,Dz,_=vector_metrics(Vz);qz=fold(Ez,w)/w;iz=outer_indices(qz);cz=central_indices(qz)
        matching,O,_=match_by_overlap(Vk,Vz,P,Pz)
        out.update(g0_D_outer=float(Dz[iz].mean()),g0_D_central=float(Dz[cz].mean()),J_dynamic_D_outer=float(Dk[folded].mean()-Dz[iz].mean()),J_dynamic_D_central=float(Dk[central].mean()-Dz[cz].mean()),J_matched_D_outer=float(np.mean(Dk[folded]-Dz[matching[folded]])),J_matched_D_central=float(np.mean(Dk[central]-Dz[matching[central]])),J_control_overlap_outer=float(O[folded].mean()),g0_D=Dz,g0_q=qz)
    if need_vectors:out.update(vectors_bare=V,vectors_kick=Vk)
    return out

def combined_metrics(exact_q,exact_ipr,eff):
    D=-np.log(exact_ipr)/np.log(DIM);idx=outer_indices(exact_q)
    return np.vstack([metrics(exact_ipr,D,idx),eff['metrics']]),D,idx

def binned_D(q,D):
    means=np.full(5,np.nan);counts=np.zeros(5,dtype=int)
    for k in range(5):
        mask=(abs(q)>=QBIN_EDGES[k]) & (abs(q)<QBIN_EDGES[k+1]+(1e-12 if k==4 else 0))
        counts[k]=int(mask.sum())
        if counts[k]:means[k]=D[mask].mean()
    return means,counts
