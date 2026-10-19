"""Audit actual saved data/figures; no replacement or synthetic physics data."""
from pathlib import Path
import csv,json,re,hashlib
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent

def rows(name):
    with (ROOT/name).open() as f:return list(csv.DictReader(f))

def assert_close(a,b,tol=1e-11):
    err=float(np.max(abs(np.asarray(a)-np.asarray(b))));assert err<tol,(err,tol);return err

def main():
    report={}
    for size,M in [(21,64),(41,128)]:
        with np.load(ROOT/f'grid_{size}.npz') as f:z={k:f[k] for k in f.files}
        assert z['metrics'].shape==(4,size,size,4) and int(z['slices'])==M
        assert np.isfinite(z['metrics']).all();rebuilt=[]
        for model,ipr,D,indices in [(0,z['ipr_exact'],z['D_exact'],z['outer_exact']),(1,z['ipr_eff_bare'],z['D_eff_bare'],z['outer_unfolded']),(2,z['ipr_eff_bare'],z['D_eff_bare'],z['outer_folded']),(3,z['ipr_eff_kick'],z['D_eff_kick'],z['outer_folded'])]:
            assert ipr.min()>=1/178-1e-12 and ipr.max()<=1+1e-12;assert_close(D,-np.log(ipr)/np.log(178))
            v=np.stack([D.mean(-1),np.take_along_axis(D,indices,axis=-1).mean(-1),ipr.mean(-1),np.take_along_axis(ipr,indices,axis=-1).mean(-1)],axis=-1);rebuilt.append(v)
        metric_error=assert_close(np.stack(rebuilt),z['metrics']);assert_close(z['metrics'][1,...,[0,2]],z['metrics'][2,...,[0,2]])
        assert np.max(abs(z['q_exact']))<=.5+1e-12;assert np.max(abs(z['q_eff']))<=.5+1e-12
        selected=np.argsort(abs(z['q_eff']),axis=-1,kind='stable')[...,-36:]
        # ±ties may swap members but their D/IPR are symmetry-identical; require metrics.
        assert_close(np.take_along_axis(z['D_eff_bare'],selected,-1).mean(-1),z['metrics'][2,...,1])
        data=rows(f'grid_{size}.csv');assert len(data)==4*size*size;assert all(int(r['outer_count'])==36 for r in data)
        report[f'grid_{size}']=dict(csv_rows=len(data),slices=M,metrics_rebuilt_max_error=metric_error,kick_norm_max_error=max(float(r['kick_norm_error']) for r in data))
    report['curves']={}
    for tag in ['0','pi4','pi2']:
        with np.load(ROOT/f'curves_phi_{tag}.npz') as f:c={k:f[k] for k in f.files}
        n=len(c['omega']);assert n==401 and c['D_exact'].shape==(n,178)
        assert_close(c['D_exact'],-np.log(c['ipr_exact'])/np.log(178));assert_close(c['energy_unwrapped'],c['omega'][:,None]*c['q_unwrapped'])
        assert_close(c['q_unwrapped']-c['q_exact'],np.rint(c['q_unwrapped']-c['q_exact']))
        assert np.all(c['outer_exact_mask'].sum(-1)==36) and np.all(c['central_exact_mask'].sum(-1)==36)
        dynamic=np.sum(c['D_exact']*c['outer_exact_mask'],axis=-1)/36;assert_close(dynamic,c['metrics'][:,0,1])
        assert_close(c['D_exact'][:,c['reference_outer_mask']].mean(-1),c['fixed_reference_D_outer'])
        expected_parent_delta=np.zeros(n)
        for i,parent in enumerate(c['parent_index']):
            if parent>=0:expected_parent_delta[i]=dynamic[i]-dynamic[parent]
        decomposition_error=assert_close(expected_parent_delta,c['intrinsic_D_change']+c['membership_D_change'])
        assert np.all(c['participation_kickbasis']>=1-1e-10) and np.all(c['participation_kickbasis']<=89+1e-8)
        counts=c['qbin_counts'];means=c['qbin_D'];assert np.all(counts.sum(-1)==178)
        assert np.array_equal(np.isnan(means),counts==0)
        data=rows(f'tracked_states_phi_{tag}.csv');assert len(data)==n*178
        report['curves'][tag]=dict(points=n,tracked_states=len(data),ambiguous_links=int(c['ambiguous'].sum()),selection_identity_error=decomposition_error,maximum_basis_participation=float(c['participation_kickbasis'].max()))
    provenance=json.loads((ROOT/'provenance.json').read_text())
    if provenance['exact_grid_reused']:
        assert hashlib.sha256((ROOT/'input_exact_grid128.npz').read_bytes()).hexdigest()==provenance['source_sha256']
        report['reuse_provenance_verified']=True
    else:report['reuse_provenance_verified']='fresh exact grid, no reuse'
    if (ROOT/'input_exact_grid128.npz').exists():
      with np.load(ROOT/'input_exact_grid128.npz') as old,np.load(ROOT/'grid_41.npz') as new:
        report['prior_folded_model_reproduced_error']=assert_close(old['metrics'][1],new['metrics'][2]);report['exact_cache_metrics_reproduced_error']=assert_close(old['metrics'][0],new['metrics'][0])
    validation=json.loads((ROOT/'validation_summary.json').read_text());assert validation['folded_and_kick_selection_identical'];assert validation['phi_zero_fold_and_kick_identical'];assert validation['maximum_actual_eigen_residual']<1e-10
    refinement=json.loads((ROOT/'refinement_summary.json').read_text());assert refinement['assignment_grid_stable_interval_count']==9;report['remaining_branch_ambiguity_intervals']=refinement['unresolved_interval_count']
    names=['fig1_quasienergy_folding','fig2_corrected_Douter_comparison','fig3_eigenstate_branch_tracking','fig4_effective_model_errors','figS1_selectivity_controls'];report['figures']={}
    for name in names:
        with Image.open(ROOT/(name+'.png')) as im:
            dpi=im.info.get('dpi',(0,0));assert min(dpi)>300;dimensions=im.size;im.verify()
        pdf=(ROOT/(name+'.pdf')).read_bytes();embedded=len(re.findall(rb'/Subtype\s*/Image',pdf));assert embedded==0
        report['figures'][name]=dict(png_pixels=dimensions,dpi=dpi,pdf_embedded_images=embedded)
    report['status']='passed';(ROOT/'artifact_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
