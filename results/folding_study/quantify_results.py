"""Compute reported statistics from the saved real numerical data only."""
from pathlib import Path
import json,csv
import numpy as np
ROOT=Path(__file__).resolve().parent

def save_csv(name,rows):
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def load_rows(name):
    with (ROOT/name).open() as f:return list(csv.DictReader(f))

def main():
    with np.load(ROOT/'grid_41.npz') as z:
        m=z['metrics'];omega=z['omega'];phi=z['phi'];regions={};errors={}
        for name,mask in [('all',np.ones(len(omega),bool)),('low_6_to_6p6',omega<=6.6+1e-12),('high_7_to_10',omega>=7-1e-12)]:
            region={}
            for im,label in [(1,'unfolded'),(2,'folded'),(3,'kick')]:
                err=m[0,:,mask,1]-m[im,:,mask,1]
                region[label]=dict(rms=float(np.sqrt(np.mean(err**2))),mae=float(abs(err).mean()),max_abs=float(abs(err).max()),mean_signed=float(err.mean()))
            d2=abs(m[0,:,mask,1]-m[2,:,mask,1]);d3=abs(m[0,:,mask,1]-m[3,:,mask,1]);region['kick_rms_reduction_fraction']=1-region['kick']['rms']/region['folded']['rms'];region['fraction_kick_improved']=float(np.mean(d3<d2-1e-12));regions[name]=region
        old_fold_error=old_exact_error=None
        if (ROOT/'input_exact_grid128.npz').exists():
            old=np.load(ROOT/'input_exact_grid128.npz');old_fold_error=float(np.max(abs(m[2]-old['metrics'][1])))
            old_exact_error=float(np.max(abs(m[0]-old['metrics'][0])))
        q=z['q_exact'];D=z['D_exact'];s=omega>=7;out_response=m[0,20,s,1]-m[0,0,s,1]
        central=[]
        for iy in [0,20]:
            means=[]
            for ix in np.flatnonzero(s):means.append(float(D[iy,ix,np.argsort(abs(q[iy,ix]),kind='stable')[:36]].mean()))
            central.append(np.array(means))
        center_response=central[1]-central[0];sel=out_response-center_response
        response_rows=[dict(omega=float(w),delta_D_outer=float(a),delta_D_central=float(b),selectivity_outer_minus_central=float(c)) for w,a,b,c in zip(omega[s],out_response,center_response,sel)]
        save_csv('phase_response_7_to_10.csv',response_rows)
        response=dict(outer_min=float(out_response.min()),outer_max=float(out_response.max()),central_min=float(center_response.min()),central_max=float(center_response.max()),selectivity_min=float(sel.min()),selectivity_max=float(sel.max()),effective_folded_states_w_ge_7=int(np.sum(abs(z['E_eff'][:,s])>omega[s][None,:,None]/2)))
    key=[];selections=[];fixed_proof={};Jstats={};basisstats={}
    for tag in ['0','pi4','pi2']:
        rows=load_rows(f'curves_phi_{tag}.csv');curve=np.load(ROOT/f'curves_phi_{tag}.npz');lookup={round(float(r['omega']),10):r for r in rows}
        for w in [6.,6.3,6.5,7.,8.,10.]:
            r=lookup[w];x=float(r['exact_D_outer']);bare=float(r['effective_unfolded_selection_D_outer']);fold=float(r['effective_folded_selection_D_outer']);kick=float(r['effective_folded_kick_D_outer'])
            key.append(dict(omega=w,phi=float(r['phi']),phi_over_pi=float(r['phi'])/np.pi,D_exact=x,D_unfolded=bare,D_folded=fold,D_folded_kick=kick,delta_bare=x-bare,delta_fold=x-fold,delta_kick=x-kick,effective_outer_replaced=int(r['effective_outer_replaced']),effective_folded_states=int(r['folded_effective_state_count']),matched_kick_overlap_outer=float(r['matched_kick_overlap_outer']),effective_basis_participation_outer=float(r['kickbasis_participation_outer']),fixed_reference_D_outer=float(r['fixed_reference_D_outer'])))
        for i,w in enumerate(curve['omega']):
            q=curve['q_exact'][i];qlift=curve['q_unwrapped'][i];folded=np.argsort(abs(q),kind='stable')[-36:];unfolded=np.argsort(abs(qlift),kind='stable')[-36:];fold_mask=np.zeros(178,dtype=bool);fold_mask[folded]=True;unfold_mask=np.zeros(178,dtype=bool);unfold_mask[unfolded]=True
            selections.append(dict(phi=float(curve['phi']),omega=float(w),exact_fold_vs_lift_outer_replaced=int(36-len(np.intersect1d(folded,unfolded))),exact_nonzero_photon_lift_count=int(np.sum(abs(qlift-q)>.5)),effective_unfold_vs_fold_replaced=int(rows[i]['effective_outer_replaced']),instantaneous_exact_turnover_vs_neighbor=int(curve['selection_turnover_count'][i]),selected_count=36,fold_q_abs_min=float(abs(q[folded]).min()),fold_q_abs_max=float(abs(q[folded]).max()),lift_selected_folded_q_abs_min=float(abs(q[unfolded]).min()),lift_selected_folded_q_abs_max=float(abs(q[unfolded]).max())))
        lo=next(r for r in rows if abs(float(r['omega'])-6.3)<1e-10);hi=next(r for r in rows if abs(float(r['omega'])-8.)<1e-10)
        region=(curve['omega']>=6.3)&(curve['omega']<=8);ref=curve['reference_outer_mask'];fixed_proof[tag]=dict(D_reference_outer_w6p3=float(lo['fixed_reference_D_outer']),D_reference_outer_w8=float(hi['fixed_reference_D_outer']),change=float(lo['fixed_reference_D_outer'])-float(hi['fixed_reference_D_outer']),min_neighbor_overlap_reference_set=float(curve['assigned_overlap'][region][:,ref].min()),ambiguous_links_in_reference_range=int(curve['ambiguous'][region][:,ref].sum()))
        high=[r for r in rows if 7<=float(r['omega'])<=10]
        Jstats[tag]={name:dict(min=float(min(float(r[name]) for r in high)),max=float(max(float(r[name]) for r in high))) for name in ['J_dynamic_D_outer','J_dynamic_D_central','J_matched_D_outer','J_matched_D_central','J_control_overlap_outer']}
        basisstats[tag]=dict(w6p3_participation=float(lo['kickbasis_participation_outer']),w6p3_matched_overlap=float(lo['matched_kick_overlap_outer']),w8_participation=float(hi['kickbasis_participation_outer']))
    save_csv('key_frequency_comparison.csv',key);save_csv('selection_changes.csv',selections)
    sample=next(r for r in key if r['omega']==8 and abs(r['phi']-np.pi/2)<1e-12)
    summary=dict(grid_error_regions=regions,old_folded_grid_reproduced_max_difference=old_fold_error,old_exact_grid_reuse_max_difference=old_exact_error,phase_response=response,key_w8_pi2=sample,key_w8_pi2_kick_abs_error_reduction=1-abs(sample['delta_kick'])/abs(sample['delta_fold']),fixed_reference_branch_evidence=fixed_proof,J_control_7_to_10=Jstats,effective_basis_mixing=basisstats)
    (ROOT/'quantitative_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
