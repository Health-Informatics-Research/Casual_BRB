"""Full upstream retraining for PHQ-9 sensitivity and learning curves.

Uses fixed participant partitions from the core run. Each condition learns its
own Fit preprocessing, causal structure, OOF paths and nested path CV caches.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import traceback
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import crossfit, legacy, PATHS, optimize, evaluate, specs, write_json, CONFIG

def prepare_condition(sets,folder):
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'prepared.joblib'
    if path.exists(): return joblib.load(path)
    fit=sets['fit']
    scores_fit,pipe=crossfit(fit)
    scores={'fit':scores_fit,**{s:pipe.predict(sets[s]) for s in ['cal','val','test']}}
    processed=pipe.processor.transform(fit)
    module=legacy('nhanes_step1_5_causal_discovery')
    reps=[v for vv in module.PATHWAY_REPRESENTATIVES.values() for v in vv if v in processed]
    causal_data=processed[reps].copy(); causal_data['depression_label']=fit.depression_label.to_numpy()
    with (folder/'causal.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
        _,edges=module.run_pc_with_bk(causal_data,list(causal_data),'depression_label')
        _,_,roles=module.build_dag_and_roles(list(causal_data),edges)
        mapping,excluded=module.cluster_pathways_weighted(causal_data,reps,roles)
    groups=[[PATHS.index(p) for p in mapping[k]] for k in ['Sub1','Sub2','Sub3']]
    write_json(folder/'hierarchy.json',{'mapping':mapping,'groups':groups,'excluded':excluded,'edges':edges})
    folds=[]
    for fold,(ti,vi) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(fit,fit.depression_label)):
        print(f'  internal fold {fold+1}/3',flush=True)
        tr,fold_pipe=crossfit(fit.iloc[ti].reset_index(drop=True))
        folds.append((tr,fold_pipe.predict(fit.iloc[vi]),fit.depression_label.to_numpy()[vi]))
    bundle={'scores':scores,'pipe':pipe,'groups':groups,'folds':folds}
    joblib.dump(bundle,path)
    pd.DataFrame({'SEQN':fit.SEQN}).to_csv(folder/'fit_participants.csv',index=False)
    write_json(folder/'features.json',{'raw_columns':list(processed),'raw_dimension':processed.shape[1]})
    return bundle

def run_one(sets,folder,prepared=None,random_graph=False):
    folder.mkdir(parents=True,exist_ok=True)
    prepared=prepare_condition(sets,folder) if prepared is None else prepared
    configs=specs(prepared['groups']); names=['Causal_BRB']+(['V4_Random_42'] if random_graph else [])
    rows=[]
    for name in names:
        bundle=optimize(name,configs[name],prepared,folder)
        details={s:bundle['model'].predict(prepared['scores'][s],bundle['params'],details=True) for s in ['cal','val','test']}
        row=evaluate(name,{s:v[0] for s,v in details.items()},sets,folder,details)
        rows.append(row)
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    pd.concat([d.assign(split=s) for s,d in sets.items()],ignore_index=True).drop(columns='split').to_csv(folder/'raw_unimputed.csv',index=False)
    pd.concat([d[['SEQN']].assign(split=s) for s,d in sets.items()],ignore_index=True).to_csv(folder/'splits.csv',index=False)
    from .statistics import metric_inference
    metric_inference(folder,42)
    return rows

def _run(base_out,mode):
    out=base_out/'seed_42'
    dest=base_out/'robustness'
    dest.mkdir(exist_ok=True)
    source=out/'extension_source_hash.json'
    hashes={'extension':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'core':hashlib.sha256((Path(__file__).parent/'core.py').read_bytes()).hexdigest()}
    if source.exists() and json.loads(source.read_text(encoding='utf-8'))!=hashes:
        raise ValueError('Extension sources changed; use a new run')
    write_json(source,hashes)
    raw=pd.read_csv(out/'raw_unimputed.csv').merge(pd.read_csv(out/'splits.csv'),on='SEQN',validate='one_to_one')
    base={s:raw[raw.split==s].drop(columns='split').reset_index(drop=True) for s in ['fit','cal','val','test']}
    status=out/f'extensions_{mode}_status.json'
    write_json(status,{'state':'running','mode':mode})
    try:
        if mode in ['phq','all']:
            rows=[]
            for cutoff in [5,8,10,12,15]:
                print(f'PHQ sensitivity cutoff={cutoff}',flush=True)
                sets={s:d.assign(depression_label=(d.PHQ9_score>=cutoff).astype(int)) for s,d in base.items()}
                folder=dest/'phq'/str(cutoff)
                if cutoff==10:
                    prepared=joblib.load(out/'prepared.joblib')
                    main_metric=out/'models/Causal_BRB/metrics.json'
                    if main_metric.exists(): result=[json.loads(main_metric.read_text(encoding='utf-8'))]
                    else: result=run_one(sets,folder,prepared)
                else: result=run_one(sets,folder)
                rows.extend([{**r,'PHQ_Cutoff':cutoff,'Positive_Rate':sets['test'].depression_label.mean()} for r in result])
                pd.DataFrame(rows).to_csv(dest/'phq_sensitivity.csv',index=False)
            pd.DataFrame(rows).plot(x='PHQ_Cutoff',y=['ROC_AUC','AP','Brier','F2'],marker='o')
            plt.tight_layout(); plt.savefig(dest/'phq_sensitivity.png',dpi=180); plt.close()
        if mode in ['learning','all']:
            rows=[]
            for fraction in [.2,.4,.6,.8,1.]:
                for repeat in range(3):
                    print(f'Learning curve Fit fraction={fraction}, repeat={repeat}',flush=True)
                    if fraction==1:
                        fit=base['fit'].copy(); prepared=joblib.load(out/'prepared.joblib')
                        folder=out
                        result=[json.loads((out/'models'/n/'metrics.json').read_text(encoding='utf-8')) for n in ['Causal_BRB','V4_Random_42']]
                    else:
                        ids,_=train_test_split(np.arange(len(base['fit'])),train_size=fraction,
                            stratify=base['fit'].depression_label,random_state=42+repeat)
                        fit=base['fit'].iloc[ids].reset_index(drop=True)
                        sets={**base,'fit':fit}
                        folder=dest/'learning'/f'fraction_{fraction:.1f}_repeat_{repeat}'
                        result=run_one(sets,folder,random_graph=True)
                    rows.extend([{**r,'Fraction':fraction,'Repeat':repeat,'Fit_N':len(fit),
                        'Independent_Repeat':fraction<1} for r in result])
                    pd.DataFrame(rows).to_csv(dest/'learning_curve.csv',index=False)
            data=pd.DataFrame(rows)
            summary=data.groupby(['Model','Fraction'])[['ROC_AUC','AP','Brier','F2']].agg(['mean','std'])
            summary.to_csv(dest/'learning_curve_summary.csv')
            fig,ax=plt.subplots(figsize=(8,5))
            for model,d in data.groupby('Model'):
                g=d.groupby('Fraction').ROC_AUC.agg(['mean','std'])
                ax.plot(g.index,g['mean'],marker='o',label=model)
                ax.fill_between(g.index,g['mean']-g['std'].fillna(0),g['mean']+g['std'].fillna(0),alpha=.15)
            ax.set(xlabel='Fraction of Fit',ylabel='Test ROC-AUC'); ax.legend()
            fig.tight_layout(); fig.savefig(dest/'learning_curve.png',dpi=180); plt.close(fig)
        write_json(status,{'state':'complete','mode':mode,'note':'100% learning condition reuses identical full-Fit runs; no artificial repeat variability'})
    except Exception:
        write_json(status,{'state':'failed','traceback':traceback.format_exc()}); raise

def run(base_out):
    _run(base_out,'all')
