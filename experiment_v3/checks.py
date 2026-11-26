"""Numerical equivalence, frozen grids and saved-artifact isolation checks."""
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,fbeta_score,brier_score_loss
from .core import BRB,active,fuse,threshold,feature_audit,CONFIG,write_json,specs
from .statistics import bootstrap_metrics,KEYS

def checks():
    rng=np.random.default_rng(7)
    x=rng.random((70,9)); x[:,7]=rng.integers(0,2,70)
    a=np.tile(np.array([0,.5,1.])[:,None],(1,4)); b=rng.random(3**4); w=rng.uniform(.01,1,3**4)
    ids,acts,starts=active(x[:20,:4],a); md_vec,mu_vec=fuse((ids,acts,starts),b,w)
    # Independent sequential Dempster combination check
    ends=np.r_[starts[1:],len(ids)]
    md_seq,mu_seq=np.zeros(20),np.zeros(20)
    for i,(s,e) in enumerate(zip(starts,ends)):
        md,mn,mu=0.0,0.0,1.0
        for k,ak_raw in zip(ids[s:e],acts[s:e]):
            ak=ak_raw*w[k]; kd=ak*b[k]; kn=ak*(1.0-b[k]); ku=1.0-ak
            denom=max(1.0-(md*kn+mn*kd),1e-12)
            md,mn,mu=(md*kd+md*ku+mu*kd)/denom,(mn*kn+mn*ku+mu*kn)/denom,(mu*ku)/denom
        md_seq[i],mu_seq[i]=md,mu
    np.testing.assert_allclose(md_vec,md_seq,atol=1e-12)
    np.testing.assert_allclose(mu_vec,mu_seq,atol=1e-12)
    y=np.array([0,1,0,0,1,0,1,0,0,1]); p=np.array([0,.1,.1,.3,.5,.5,.9,.9,1,1])
    weights=rng.multinomial(len(y),np.ones(len(y))/len(y),size=12)
    result=bootstrap_metrics(y,p,.5,.4,weights)
    for i,multiplicity in enumerate(weights):
        ids=np.repeat(np.arange(len(y)),multiplicity)
        yy,pp=y[ids],p[ids]
        if len(np.unique(yy))==2: np.testing.assert_allclose(result['ROC_AUC'][i],roc_auc_score(yy,pp),atol=1e-12)
        np.testing.assert_allclose(result['AP'][i],average_precision_score(yy,pp),atol=1e-12)
        np.testing.assert_allclose(result['Brier'][i],brier_score_loss(yy,pp),atol=1e-12)
        np.testing.assert_allclose(result['F2'][i],fbeta_score(yy,pp>=.5,beta=2),atol=1e-12)
    tau=threshold(y,p); assert .05<=tau<=.95 and abs(tau*100-round(tau*100))<1e-8
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as name:
        folder=Path(name)
        feature_audit(['RIDAGEYR','ALQ130_missing'],folder)
        try: feature_audit(['DPQ030'],folder)
        except AssertionError: pass
        else: raise AssertionError('PHQ-9 item not rejected')
        # Exercise the complete group-bootstrap/GLM/random-triage report, not only metric primitives.
        from .statistics import uncertainty_analysis
        modeldir=folder/'models/Causal_BRB'; modeldir.mkdir(parents=True)
        nn=150; yy=rng.binomial(1,.2,nn); prob=rng.uniform(.01,.3,nn); unc=rng.uniform(.1,.8,nn)
        frame=pd.DataFrame({'SEQN':np.arange(nn),'y_true':yy,'calibrated_prob':prob,'raw_prob':prob,'m_Theta':unc})
        frame.to_csv(modeldir/'test_predictions.csv',index=False); frame.to_csv(modeldir/'val_predictions.csv',index=False)
        uncertainty_analysis(folder,42)
        assert len(pd.read_csv(folder/'statistics/uncertainty_group_CI.csv'))==3
        assert len(pd.read_csv(folder/'statistics/uncertainty_incremental_GLM.csv'))==6
        from .run import worker
        resume=folder/'resume'; (resume/'seed_42').mkdir(parents=True)
        write_json(resume/'seed_42/status.json',{'state':'complete','reason':'中文路径下的状态读取测试'})
        worker(resume,42)
    m=BRB(x,[list(range(9))],flat=True)
    assert m.nr==19683 and m.n_params==39366
    matched=specs([[0],[1,2],[4,5]])['V1_MatchedInputs']['groups'][0]
    assert matched==[0,1,2,4,5,6,7,8], 'Matched control retained an excluded pathway'
    z=m.repair(rng.random(m.n_params)); np.testing.assert_allclose(m.repair(z),z)
    assert len(CONFIG['split_seeds'])==5 and CONFIG['class_weight'] is None
    from causallearn.utils.cit import CIT
    from .fast_gsq import CachedGSquare,install_exact_skeleton_shortcut
    discrete=rng.integers(0,4,(250,8)); discrete[:,6]=discrete[:,0]
    original=CIT(discrete,'gsq'); accelerated=CachedGSquare(discrete)
    for _ in range(300):
        x,y=rng.choice(8,2,replace=False); pool=[k for k in range(8) if k not in [x,y]]
        ss=rng.choice(pool,size=rng.integers(0,6),replace=False).tolist()
        np.testing.assert_allclose(original(int(x),int(y),ss),accelerated(int(x),int(y),ss),atol=2e-9,rtol=1e-8)
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.PCUtils import SkeletonDiscovery
    base_skeleton=SkeletonDiscovery.skeleton_discovery
    install_exact_skeleton_shortcut(); cached_skeleton=SkeletonDiscovery.skeleton_discovery
    for _ in range(4):
        dd=rng.integers(0,3,(180,7)); dd[:,5]=dd[:,0]; dd[:,6]=(dd[:,1]+dd[:,2])%3
        SkeletonDiscovery.skeleton_discovery=base_skeleton
        aa=pc(dd,alpha=.1,indep_test='gsq',stable=True,show_progress=False)
        SkeletonDiscovery.skeleton_discovery=cached_skeleton
        bb=pc(dd,alpha=.1,indep_test='cached_exact_gsq',stable=True,show_progress=False)
        np.testing.assert_array_equal(aa.G.graph,bb.G.graph)
    print('PASS: packed ER equivalence (9-D), tied-score participant-bootstrap metrics, threshold grid, leakage guard, V1 matched dimensions')

def audit_run(out):
    from .statistics import read
    allrows=[]
    for seed in CONFIG['split_seeds']:
        folder=out/f'seed_{seed}'; splits=read(folder/'splits.csv')
        assert not splits.SEQN.duplicated().any()
        ids={s:set(splits.loc[splits.split==s,'SEQN']) for s in ['fit','cal','val','test']}
        assert len(set.union(*ids.values()))==len(splits)
        assert json.loads((folder/'feature_audit.json').read_text(encoding='utf-8'))['passed']
        table=read(folder/'metrics.csv'); assert len(table)==21
        for name in table.Model:
            modeldir=folder/'models'/name; tau=json.loads((modeldir/'threshold.json').read_text(encoding='utf-8'))['tau']
            assert .05<=tau<=.95 and abs(tau*100-round(tau*100))<1e-8
            for s in ['cal','val','test']:
                d=read(modeldir/f'{s}_predictions.csv')
                assert not d.SEQN.duplicated().any() and set(d.SEQN)==ids[s]
                assert np.isfinite(d[['raw_prob','calibrated_prob']].to_numpy()).all()
                np.testing.assert_array_equal(d.prediction,(d.calibrated_prob>=tau).astype(int))
            saved=table.loc[table.Model==name].iloc[0]
            for key,v in [('ROC_AUC',roc_auc_score(d.y_true,d.calibrated_prob)),
                ('AP',average_precision_score(d.y_true,d.calibrated_prob)),
                ('Brier',brier_score_loss(d.y_true,d.calibrated_prob)),
                ('F2',fbeta_score(d.y_true,d.prediction,beta=2))]: np.testing.assert_allclose(saved[key],v,atol=1e-12)
        ci=read(folder/'statistics/metric_CI.csv'); assert len(ci)==21*len(KEYS)
        assert set(ci.Model)==set(table.Model) and ci.Valid_Bootstraps.between(1,1000).all()
        pairs=read(folder/'statistics/paired_differences.csv'); assert len(pairs)==4*len(KEYS)
        structural=0
        if seed in CONFIG['causal_bootstrap_split_seeds']:
            status=json.loads((folder/'stability/status.json').read_text(encoding='utf-8')); assert status['successful']==100,status
            structural=100
        allrows.append({'Split_Seed':seed,'Models':len(table),'Bootstrap_Draws':1000,'Structure_Bootstraps':structural,'Audit':'passed'})
    write_json(out/'audit.json',{'state':'passed','checks':allrows,
        'scope':'identity, frozen thresholds, metric consistency, CIs, structural completion; not causal identification or clinical validation'})

if __name__=='__main__': checks()
