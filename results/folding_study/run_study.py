"""Run folding, kick, state-selection and symmetry-resolved branch diagnostics.

Default: reuses only the independently revalidated exact128 grid from the
previous same-Hamiltonian run. All new comparisons and branch data are computed.
Use --fresh-exact-grid to recompute that grid rather than reuse the cache.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,csv,json,time,hashlib,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from floquet_core import calculate,PARAMETERS
from analysis_core import effective,static_data,combined_metrics,outer_indices,central_indices,range_info,binned_D,match_by_overlap,MODEL_NAMES,METRIC_NAMES,QBIN_EDGES
from branch_tracking import align,unwrap_q
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'rerun_02'
N=178;K=36

def csvsave(name,data):
    if not data:return
    with (ROOT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)

def grid_point(args):
    w,ph,M,cq,ci=args
    if cq is None:
        x=calculate(89,55,w,ph,M,'exact');cq=x['quasienergy']/w;ci=x['ipr'];source='new exact midpoint evolution'
    else:source='validated previous exact128 grid'
    e=effective(w,ph);met,D,idx=combined_metrics(cq,ci,e)
    ranges=[range_info(cq,cq,idx)]+e['ranges']
    rows=[dict(omega=w,phi=ph,model=model,**dict(zip(METRIC_NAMES,met[i])),**ranges[i],folded_effective_state_count=e['folded_state_count'],effective_outer_replaced=e['bare_vs_fold_replaced'],exact_slices=M,exact_source=source,kick_norm_error=e['normalization_error']) for i,model in enumerate(MODEL_NAMES)]
    return (w,ph),dict(metrics=met,q_exact=cq,ipr_exact=ci,D_exact=D,q_eff=e['q'],E_eff=e['E'],D_eff_bare=e['D_bare'],D_eff_kick=e['D_kick'],ipr_eff_bare=e['ipr_bare'],ipr_eff_kick=e['ipr_kick'],outer_exact=idx,outer_unfolded=e['unfolded_indices'],outer_folded=e['folded_indices'],rows=rows)

def run_grid(size,M,cache=None):
    tic=time.perf_counter();ws=np.linspace(6,10,size);ps=np.linspace(0,np.pi,size);jobs=[]
    if cache is not None:
        cache_e=cache['quasienergy'][0];cache_ipr=cache['ipr'][0];cache_phi=cache['phi'];cache_omega=cache['omega']
    for ph in ps:
        for w in ws:
            if cache is None:cq=ci=None
            else:
                iy=int(np.argmin(abs(cache_phi-ph)));ix=int(np.argmin(abs(cache_omega-w)));assert abs(cache_phi[iy]-ph)<1e-12 and abs(cache_omega[ix]-w)<1e-12
                cq=cache_e[iy,ix]/w;ci=cache_ipr[iy,ix]
            jobs.append((float(w),float(ph),M,cq,ci))
    out={}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for count,f in enumerate(as_completed([pool.submit(grid_point,a) for a in jobs]),1):
            key,r=f.result();out[key]=r
            if count%100==0 or count==len(jobs):print(f'four-model grid {size}: {count}/{len(jobs)}, {time.perf_counter()-tic:.1f}s',flush=True)
    arr=dict(omega=ws,phi=ps,slices=np.array(M),model_names=np.array(MODEL_NAMES),metric_names=np.array(METRIC_NAMES),metrics=np.empty((4,size,size,4)))
    for key in ['q_exact','ipr_exact','D_exact','q_eff','E_eff','D_eff_bare','D_eff_kick','ipr_eff_bare','ipr_eff_kick']:arr[key]=np.empty((size,size,N))
    for key in ['outer_exact','outer_unfolded','outer_folded']:arr[key]=np.empty((size,size,K),dtype=np.int16)
    rows=[]
    for iy,ph in enumerate(ps):
        for ix,w in enumerate(ws):
            r=out[(float(w),float(ph))];arr['metrics'][:,iy,ix]=r['metrics'];rows+=r['rows']
            for key in arr:
                if key in r and key!='metrics':arr[key][iy,ix]=r[key]
    np.savez_compressed(ROOT/f'grid_{size}.npz',**arr);csvsave(f'grid_{size}.csv',rows)
    return time.perf_counter()-tic

def phase_curve(args):
    phi,tag=args;tic=time.perf_counter()
    ws=np.unique(np.round(np.concatenate([np.linspace(5.8,6.6,161),np.linspace(6.6,8.,141),np.linspace(8.,10.,101)]),12));n=len(ws);anchor=int(np.flatnonzero(ws==8.)[0]);data={}
    for key in ['q_exact','q_unwrapped','energy_unwrapped','D_exact','ipr_exact','assigned_overlap','margin','D_eff_matched','D_kick_matched','overlap_eff_matched','overlap_kick_matched','participation_effbasis','participation_kickbasis','q_eff','E_eff','D_eff_bare','D_eff_kick']:
        data[key]=np.empty((n,N))
    for key in ['ambiguous','rank_swap','near_gap','outer_exact_mask','central_exact_mask']:data[key]=np.zeros((n,N),dtype=bool)
    data['metrics']=np.empty((n,4,4));data['qbin_D']=np.full((n,3,5),np.nan);data['qbin_counts']=np.zeros((n,3,5),dtype=np.int16)
    rows=[];state_rows=[];diagnostics=[];parents=np.full(n,-1,dtype=int)
    def evaluate(index,previous=None):
        w=float(ws[index]);x=calculate(89,55,w,phi,128,'exact',return_matrices=True);e=effective(w,phi,controls=True,need_vectors=True)
        v=x['eigenvectors'];q=x['quasienergy']/w;P=x['parity'];D=x['D'];ipr=x['ipr']
        if previous is None:
            perm=np.arange(N);ov=np.ones(N);margin=np.ones(N);flags=np.zeros(N,dtype=bool);rs=flags.copy();ng=flags.copy();qlift=q.copy()
        else:
            prev_index,pv,pq,pqlift,pP=previous;al=align(pv,v,pP,P,pq,q);perm=al['permutation'];ov=al['overlaps'];margin=al['margins'];flags=al['ambiguous'];rs=al['rank_swap'];ng=al['near_gap'];qlift=unwrap_q(pqlift,q[perm]);parents[index]=prev_index
        v=v[:,perm];q=q[perm];P=P[perm];D=D[perm];ipr=ipr[perm]
        im,ob,prb=match_by_overlap(v,e['vectors_bare'],P,e['parity']);ik,ok,prk=match_by_overlap(v,e['vectors_kick'],P,e['parity'])
        exactidx=outer_indices(q);central=central_indices(q);met=np.vstack([np.array([D.mean(),D[exactidx].mean(),ipr.mean(),ipr[exactidx].mean()]),e['metrics']]);data['metrics'][index]=met
        data['outer_exact_mask'][index,exactidx]=True;data['central_exact_mask'][index,central]=True
        values=dict(q_exact=q,q_unwrapped=qlift,energy_unwrapped=w*qlift,D_exact=D,ipr_exact=ipr,assigned_overlap=ov,margin=margin,ambiguous=flags,rank_swap=rs,near_gap=ng,D_eff_matched=e['D_bare'][im],D_kick_matched=e['D_kick'][ik],overlap_eff_matched=ob,overlap_kick_matched=ok,participation_effbasis=prb,participation_kickbasis=prk,q_eff=e['q'],E_eff=e['E'],D_eff_bare=e['D_bare'],D_eff_kick=e['D_kick'])
        for key,value in values.items():data[key][index]=value
        for model,(qb,Db) in enumerate([(q,D),(e['q'],e['D_bare']),(e['q'],e['D_kick'])]):
            data['qbin_D'][index,model],data['qbin_counts'][index,model]=binned_D(qb,Db)
        ranges=[range_info(q,q,exactidx)]+e['ranges']
        r=dict(omega=w,phi=phi,exact_D_central=float(D[central].mean()),exact_IPR_central=float(ipr[central].mean()),eff_fold_D_central=float(e['D_bare'][e['central_indices']].mean()),eff_kick_D_central=float(e['D_kick'][e['central_indices']].mean()),g_F=e['g_F'],folded_effective_state_count=e['folded_state_count'],effective_outer_replaced=e['bare_vs_fold_replaced'],mean_assigned_overlap=float(ov.mean()),min_assigned_overlap=float(ov.min()),low_overlap_count=int((ov<.8).sum()),ambiguous_count=int(flags.sum()),rank_swap_count=int(rs.sum()),near_gap_count=int(ng.sum()),effbasis_participation_outer=float(prb[exactidx].mean()),kickbasis_participation_outer=float(prk[exactidx].mean()),kickbasis_participation_central=float(prk[central].mean()),matched_kick_overlap_outer=float(ok[exactidx].mean()),g0_D_outer=e['g0_D_outer'],g0_D_central=e['g0_D_central'],J_dynamic_D_outer=e['J_dynamic_D_outer'],J_dynamic_D_central=e['J_dynamic_D_central'],J_matched_D_outer=e['J_matched_D_outer'],J_matched_D_central=e['J_matched_D_central'],J_control_overlap_outer=e['J_control_overlap_outer'])
        for model,name in enumerate(MODEL_NAMES):
            for k,metric in enumerate(METRIC_NAMES):r[name+'_'+metric]=float(met[model,k])
            for key,value in ranges[model].items():r[name+'_'+key]=value
        rows.append(r);diagnostics.append(dict(omega=w,phi=phi,**{key:x[key] for key in ['hermiticity_max','unitarity_relative_fro','eigen_residual_relative_fro','eigenvector_orthogonality_relative_fro','min_full_quasienergy_gap','min_within_sector_quasienergy_gap']}))
        return index,v,q,qlift,P
    anchor_state=evaluate(anchor);parity=anchor_state[-1];data['parity']=parity;data['reference_outer_mask']=data['outer_exact_mask'][anchor].copy()
    prev=anchor_state
    for count,index in enumerate(range(anchor-1,-1,-1),1):
        prev=evaluate(index,prev)
        if count%75==0:print(f'curve phi/pi={phi/np.pi:g}, descending {count}/{anchor}, wall {time.perf_counter()-tic:.1f}s',flush=True)
    prev=anchor_state
    for index in range(anchor+1,n):prev=evaluate(index,prev)
    ref=data['reference_outer_mask'];data['fixed_reference_D_outer']=data['D_exact'][:,ref].mean(axis=1)
    data['omega']=ws;data['phi']=np.array(phi);data['anchor_omega']=np.array(8.);data['parent_index']=parents;data['qbin_edges']=QBIN_EDGES
    # Positive normalized-energy targets avoid plotting symmetry-duplicate +/-IPRs.
    representative=[]
    for target in [.05,.20,.32,.40]:
        order=np.argsort(abs(data['q_exact'][anchor]-target))
        representative.append(next(int(j) for j in order if int(j) not in representative))
    data['representative_branch_ids']=np.array(representative,dtype=np.int16)
    turnover=np.zeros(n,dtype=np.int16);intrinsic=np.zeros(n);membership=np.zeros(n)
    for i,parent in enumerate(parents):
        if parent<0:continue
        before=data['outer_exact_mask'][parent];after=data['outer_exact_mask'][i];turnover[i]=K-int(np.sum(before&after))
        intrinsic[i]=float(np.mean(data['D_exact'][i,before]-data['D_exact'][parent,before]));membership[i]=float(np.mean(data['D_exact'][i,after])-np.mean(data['D_exact'][i,before]))
        assert abs((data['metrics'][i,0,1]-data['metrics'][parent,0,1])-(intrinsic[i]+membership[i]))<1e-12
    data['selection_turnover_count']=turnover;data['intrinsic_D_change']=intrinsic;data['membership_D_change']=membership
    np.savez_compressed(ROOT/f'curves_phi_{tag}.npz',**data)
    byomega={r['omega']:r for r in rows}
    for i,w in enumerate(ws):
        r=byomega[float(w)];r['fixed_reference_D_outer']=float(data['fixed_reference_D_outer'][i]);r['selection_turnover_count']=int(turnover[i]);r['intrinsic_D_change']=float(intrinsic[i]);r['membership_D_change']=float(membership[i])
    csvsave(f'curves_phi_{tag}.csv',[byomega[float(w)] for w in ws]);csvsave(f'curve_diagnostics_phi_{tag}.csv',sorted(diagnostics,key=lambda r:r['omega']))
    # All actual tracked state diagnostics, not just representative curves.
    for i,w in enumerate(ws):
        for b in range(N):state_rows.append(dict(omega=float(w),phi=phi,branch=b,parity=int(parity[b]),q=float(data['q_exact'][i,b]),q_lift=float(data['q_unwrapped'][i,b]),energy_lift=float(data['energy_unwrapped'][i,b]),IPR=float(data['ipr_exact'][i,b]),D=float(data['D_exact'][i,b]),overlap=float(data['assigned_overlap'][i,b]),ambiguous=bool(data['ambiguous'][i,b]),rank_swap=bool(data['rank_swap'][i,b]),near_gap=bool(data['near_gap'][i,b]),outer=bool(data['outer_exact_mask'][i,b]),effbasis_PR=float(data['participation_effbasis'][i,b]),kickbasis_PR=float(data['participation_kickbasis'][i,b])))
    csvsave(f'tracked_states_phi_{tag}.csv',state_rows)
    result=dict(phi=phi,tag=tag,points=n,wall_seconds=time.perf_counter()-tic,branch_labels=N,representative_branches=representative,low_overlap_links=int(np.count_nonzero(data['assigned_overlap']<.8)),ambiguous_links=int(data['ambiguous'].sum()))
    print('curve completed '+json.dumps(result),flush=True);return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fresh-exact-grid',action='store_true');args=parser.parse_args();start=time.perf_counter()
    st=static_data();np.savez_compressed(ROOT/'static_spectrum.npz',energy=st['energy'],D=st['D'],ipr=st['ipr'],parity=st['parity'],omega_fold=st['omega_fold'],Emaxabs=st['Emaxabs'])
    csvsave('static_spectrum.csv',[dict(state=i,energy=float(e),D=float(st['D'][i]),IPR=float(st['ipr'][i]),parity=int(st['parity'][i])) for i,e in enumerate(st['energy'])])
    cachepath=ROOT/'input_exact_grid128.npz';source=PREVIOUS/'grid_41.npz';provenance=dict(parameters=PARAMETERS,source_exact_grid=str(source),static_Emaxabs=st['Emaxabs'],static_omega_fold=st['omega_fold'],grid_21_fresh_M64=True,final_grid_M128=True,exact_grid_reused=not args.fresh_exact_grid)
    cache=None;checks=[]
    if not args.fresh_exact_grid:
        if not cachepath.exists():shutil.copy2(source,cachepath)
        provenance['source_sha256']=hashlib.sha256(cachepath.read_bytes()).hexdigest();cache=np.load(cachepath)
        assert int(cache['slices'])==128
        for w,ph in [(6.,0.),(6.2,.325*np.pi),(8.,np.pi/2)]:
            r=calculate(89,55,w,ph,128,'exact');iy=int(np.argmin(abs(cache['phi']-ph)));ix=int(np.argmin(abs(cache['omega']-w)))
            qe=float(np.max(abs(r['quasienergy']-cache['quasienergy'][0,iy,ix])));ie=float(np.max(abs(r['ipr']-cache['ipr'][0,iy,ix])));assert qe<1e-10 and ie<1e-8
            checks.append(dict(omega=w,phi=ph,max_quasienergy_difference=qe,max_IPR_difference=ie))
        provenance['independent_recompute_cache_checks']=checks
    (ROOT/'provenance.json').write_text(json.dumps(provenance,indent=2));timing={}
    timing['fresh_21x21_M64_wall_s']=run_grid(21,64)
    timing['41x41_models_with_exact128_wall_s']=run_grid(41,128,cache)
    print('Grid comparisons complete',flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(phase_curve,[(0.,'0'),(np.pi/4,'pi4'),(np.pi/2,'pi2')]))
    timing['phase_curves']=results;timing['total_wall_s']=time.perf_counter()-start
    (ROOT/'run_summary.json').write_text(json.dumps(dict(timing=timing,provenance=provenance),indent=2));print(json.dumps(timing,indent=2),flush=True)
if __name__=='__main__':main()
