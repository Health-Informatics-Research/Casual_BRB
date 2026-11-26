"""Frozen repeated-split orchestration with per-model checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from .core import CONFIG, ROOT, Processor, prepare, run_models, feature_audit, write_json

def freeze(out):
    out.mkdir(parents=True,exist_ok=True)
    sources=[p for p in (ROOT/'experiment_v3').glob('*') if p.suffix in ['.py','.json','.md']]
    sources += [ROOT/f'{n}.py' for n in ['nhanes_step1_data_processing','nhanes_step1_5_causal_discovery','nhanes_step2_feature_engineering']]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest=out/'frozen_source_hashes.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8'))!=hashes: raise ValueError('Frozen sources changed; use new output directory')
    else:
        write_json(manifest,hashes); write_json(out/'frozen_config.json',CONFIG)
        snap=out/'source_snapshot'
        for p in sources:
            dest=snap/p.relative_to(ROOT); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
        write_json(out/'environment.json',{'python':sys.version,'packages':{p:importlib.metadata.version(p)
            for p in ['numpy','pandas','scipy','scikit-learn','causal-learn','lightgbm','statsmodels','joblib','filelock']}})
    raw=out/'raw_unimputed.csv'
    if not raw.exists():
        # This artifact contains deterministic cleaning only, never prior imputed features or model predictions.
        shutil.copy2(ROOT/'runs/protocol_v2/raw_unimputed.csv',raw)
    write_json(out/'raw_input_hash.json',{'sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
        'source':'v2 raw_unimputed.csv (deterministic XPT merge/filter/code mapping only)'})

def worker(out,seed):
    from .statistics import analyze
    from .report import generate_seed
    folder=out/f'seed_{seed}'; folder.mkdir(exist_ok=True)
    if (folder/'status.json').exists() and json.loads((folder/'status.json').read_text(encoding='utf-8'))['state'] in ['complete','models_complete']:
        if seed not in CONFIG['causal_bootstrap_split_seeds']:
            write_json(folder/'status.json',{'state':'complete','seed':seed,'structure_bootstrap':'not_requested'})
        return
    shutil.copy2(out/'raw_unimputed.csv',folder/'raw_unimputed.csv')
    write_json(folder/'config.json',{**CONFIG,'active_split_seed':seed})
    try:
        write_json(folder/'status.json',{'state':'preparing','seed':seed})
        prepared,sets=prepare(folder,seed)
        write_json(folder/'status.json',{'state':'training','seed':seed})
        run_models(prepared,sets,folder)
        write_json(folder/'status.json',{'state':'statistics','seed':seed})
        analyze(folder,seed)
        generate_seed(folder,prepared,sets,seed)
        requested=seed in CONFIG['causal_bootstrap_split_seeds']
        write_json(folder/'status.json',{'state':'models_complete' if requested else 'complete',
            'seed':seed,'structure_bootstrap':'pending' if requested else 'not_requested'})
    except Exception:
        write_json(folder/'status.json',{'state':'failed','seed':seed,'traceback':traceback.format_exc()}); raise

def _structure_worker(out,seed):
    import pandas as pd
    import joblib
    from .stability import run as stability
    folder=out/f'seed_{seed}'
    raw=pd.read_csv(folder/'raw_unimputed.csv').merge(pd.read_csv(folder/'splits.csv'),on='SEQN',validate='one_to_one')
    sets={s:raw[raw.split==s].drop(columns='split').reset_index(drop=True) for s in ['fit','cal','val','test']}
    try:
        write_json(folder/'status.json',{'state':'structural_stability','seed':seed})
        stability(folder,sets,joblib.load(folder/'prepared.joblib'),seed)
        write_json(folder/'status.json',{'state':'complete','seed':seed})
    except Exception:
        write_json(folder/'status.json',{'state':'failed','seed':seed,'traceback':traceback.format_exc()}); raise

def structure_worker(out,seed):
    from filelock import FileLock
    folder=out/f'seed_{seed}'
    with FileLock(str(folder/'structure.lock')):
        status=folder/'stability/status.json'
        if status.exists() and json.loads(status.read_text(encoding='utf-8')).get('successful')==100:
            write_json(folder/'status.json',{'state':'complete','seed':seed}); return
        _structure_worker(out,seed)

def main(out):
    from .report import aggregate
    freeze(out)
    # Audit actual Fit schemas for all splits before any model fit.
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    raw=pd.read_csv(out/'raw_unimputed.csv'); audits=[]
    for seed in CONFIG['split_seeds']:
        train,_=train_test_split(np.arange(len(raw)),test_size=.2,stratify=raw.depression_label,random_state=seed)
        fit,_=train_test_split(train,test_size=.4,stratify=raw.iloc[train].depression_label,random_state=seed)
        proc=Processor().fit(raw.iloc[fit])
        folder=out/f'seed_{seed}'; folder.mkdir(exist_ok=True)
        feature_audit(proc.transform(raw.iloc[fit]).columns,folder)
        audits.append({'Split_Seed':seed,'Raw_Dimension':len(proc.cols)+len(proc.indicators),'Feature_Audit':'passed'})
    pd.DataFrame(audits).to_csv(out/'preflight_feature_audits.csv',index=False)
    write_json(out/'status.json',{'state':'running','split_seeds':CONFIG['split_seeds']})
    def launch(seed,structure=False):
        env=os.environ.copy()
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: env[key]='1'
        with (out/f'seed_{seed}{"_structure" if structure else ""}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u','-m','experiment_v3.run','--output',str(out),
                '--structure-only' if structure else '--worker',str(seed)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        return seed,result.returncode
    failed=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(launch,s) for s in CONFIG['split_seeds']]
        for future in as_completed(jobs):
            seed,code=future.result(); print(f'Split {seed} finished: exit={code}',flush=True)
            if code: failed.append(seed)
    if failed:
        write_json(out/'status.json',{'state':'failed','failed_seeds':failed}); return
    aggregate(out,include_stability=False)
    write_json(out/'status.json',{'state':'robustness_extensions'})
    from .extensions import run as extensions
    extensions(out)
    write_json(out/'status.json',{'state':'structural_stability','primary_results':'RESULTS.md'})
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(launch,s,True) for s in CONFIG['causal_bootstrap_split_seeds']]):
            seed,code=future.result(); print(f'Structural bootstrap {seed}: exit={code}',flush=True)
            if code: failed.append(seed)
    if failed:
        write_json(out/'status.json',{'state':'failed','failed_structure_seeds':failed}); return
    aggregate(out)
    from .checks import audit_run
    audit_run(out)
    write_json(out/'status.json',{'state':'complete','seeds':CONFIG['split_seeds'],'external_validation':False,'audit':'passed'})
    print('PROTOCOL V3 COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='runs/protocol_v3_r6')
    parser.add_argument('--worker',type=int); parser.add_argument('--structure-only',type=int); args=parser.parse_args(); out=Path(args.output).resolve()
    if args.structure_only is not None: structure_worker(out,args.structure_only)
    elif args.worker is None:
        try: main(out)
        except Exception:
            out.mkdir(parents=True,exist_ok=True); write_json(out/'status.json',{'state':'failed','traceback':traceback.format_exc()}); raise
    else: worker(out,args.worker)
