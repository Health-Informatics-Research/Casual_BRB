"""Fit-only structural bootstrap. No BRB retraining or Test use."""
import contextlib
from collections import Counter
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .core import legacy, PATHS, CONFIG, Processor, write_json

def discover(processed,y):
    from .fast_gsq import install_exact_skeleton_shortcut
    install_exact_skeleton_shortcut()
    module=legacy('nhanes_step1_5_causal_discovery')
    module.gsq='cached_exact_gsq'
    reps=[v for vv in module.PATHWAY_REPRESENTATIVES.values() for v in vv if v in processed]
    d=processed[reps].reset_index(drop=True).copy(); d['depression_label']=np.asarray(y)
    _,edges=module.run_pc_with_bk(d,list(d),'depression_label')
    _,_,roles=module.build_dag_and_roles(list(d),edges)
    mapping,excluded=module.cluster_pathways_weighted(d,reps,roles)
    return edges,mapping,excluded,list(d)

def run(out,sets,prepared,seed):
    dest=out/'stability'; dest.mkdir(exist_ok=True)
    fit=sets['fit']; processed=prepared['pipe'].processor.transform(fit)
    rng=np.random.default_rng(seed+89013); results=[]
    for b in range(CONFIG['causal_bootstraps_per_split']):
        path=dest/f'bootstrap_{b:03d}.json'
        idx=rng.integers(0,len(fit),len(fit))
        if path.exists(): results.append(json.loads(path.read_text(encoding='utf-8'))); continue
        # Advance RNG even for cached draws; deterministic resume.
        try:
            x=Processor().fit(fit.iloc[idx]).transform(fit.iloc[idx]) if CONFIG['causal_bootstrap_refit_imputer'] else processed.iloc[idx]
            with (dest/f'bootstrap_{b:03d}.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
                edges,mapping,excluded,nodes=discover(x,fit.depression_label.to_numpy()[idx])
            result={'replicate':b,'state':'complete','edges':edges,'mapping':mapping,'excluded':excluded,'nodes':nodes}
        except Exception as exc: result={'replicate':b,'state':'failed','error':repr(exc)}
        write_json(path,result); results.append(result)
        if (b+1)%10==0: print(f'structure bootstrap {seed}: {b+1}/100',flush=True)
    good=[r for r in results if r['state']=='complete']; edgecounts=Counter(); groups=Counter()
    nodes=sorted(set(n for r in good for n in r['nodes']))
    for r in good:
        edgecounts.update(tuple(e) for e in r['edges'])
        for p in PATHS:
            group=next((k for k,v in r['mapping'].items() if p in v),'Excluded')
            groups[p,group]+=1
    denominator=len(good)
    pd.DataFrame([{'Source':a,'Target':b,'Count':edgecounts[a,b],'Frequency':edgecounts[a,b]/denominator if denominator else np.nan,
        'Successful_Replicates':denominator,'Requested':100} for a in nodes for b in nodes if a!=b]).to_csv(dest/'edge_frequencies.csv',index=False)
    pd.DataFrame([{'Pathway':p,'Group':g,'Count':groups[p,g],'Frequency':groups[p,g]/denominator if denominator else np.nan,
        'Successful_Replicates':denominator} for p in PATHS for g in ['Sub1','Sub2','Sub3','Excluded']]).to_csv(dest/'group_frequencies.csv',index=False)
    write_json(dest/'status.json',{'state':'complete' if denominator==100 else 'partial_failure',
        'requested':100,'successful':denominator,'failures':100-denominator,
        'preprocessing':'refit per bootstrap' if CONFIG['causal_bootstrap_refit_imputer'] else 'conditional on full-Fit preprocessing'})
