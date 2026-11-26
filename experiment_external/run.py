"""Audit first; training is explicit. python -m experiment_external.run --help"""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from filelock import FileLock
from experiment_v3 import core
from .data import ROOT, PROTOCOL, LABEL, load_data, profile, source_files
from .engine import split_data, prepare, run_models, inference, PositivePlatt


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def freeze(out, dataset, mode):
    out.mkdir(parents=True, exist_ok=True)
    sources = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).with_name('protocol.json')]
    sources += list((ROOT/'experiment_v3').glob('*.py')) + [ROOT/'experiment_v3/config.json']
    sources += [ROOT/f'{name}.py' for name in ['nhanes_step1_data_processing','nhanes_step2_feature_engineering','nhanes_step1_5_causal_discovery']]
    manifest = {'dataset':dataset,'mode':mode,'protocol':PROTOCOL,
                'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sources},
                'data_sha256':{str(p):digest(p) for p in source_files(dataset)},
                'profile':profile(dataset),
                'environment': {'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in
                    ['numpy','pandas','scipy','scikit-learn','causal-learn','lightgbm','joblib']}}}
    if mode=='frozen_transport':
        source=ROOT/PROTOCOL['frozen_transport']['source_run']
        final=ROOT/PROTOCOL['frozen_transport']['final_run']
        artifacts=[final/f'{s}_decision.json' for s in ['calibration','de_budget']]
        for seed in PROTOCOL['split_seeds']:
            old=source/f'seed_{seed}'
            artifacts += [old/'prepared.joblib',old/'features.json',old/'fit_pathways.csv']
            for parent in [old,final/f'seed_{seed}']:
                for pattern in ['models/*/model.joblib','models/*/cal_predictions.csv','models/*/val_predictions.csv']:
                    artifacts.extend(parent.glob(pattern))
        manifest['development_artifacts_sha256']={str(p):digest(p) for p in artifacts}
    path = out/'frozen_manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != manifest:
            raise RuntimeError('Frozen source/data/environment changed. Use a new output directory.')
    else:
        # Do not adopt untracked old outputs as resumable results.
        if any(out.glob('seed_*')):
            raise RuntimeError('Existing seed outputs without manifest')
        core.write_json(path, manifest)
        for p in sources:
            target = out/'source_snapshot'/p.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target)


def transport_audit(raw, fields, out):
    source = ROOT/PROTOCOL['frozen_transport']['source_run']
    rows=[]
    for seed in PROTOCOL['split_seeds']:
        required = json.loads((source/f'seed_{seed}/features.json').read_text(encoding='utf-8'))['raw_variables']
        for cycle in sorted(raw.CYCLE.unique()):
            present = set(fields.loc[fields.cycle==cycle,'field'])
            for c in required:
                available = c in present and c in raw and raw.loc[raw.CYCLE==cycle,c].notna().any()
                rows.append({'seed':seed,'cycle':cycle,'field':c,'compatible':available,
                             'reason':'native field present' if available else 'absent or entirely missing; no automatic substitution'})
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'frozen_transport_compatibility.csv',index=False)
    result={'eligible':bool(frame.compatible.all()),'missing_by_cycle':
            frame.loc[~frame.compatible].groupby('cycle').field.unique().apply(list).to_dict(),
            'decision':'No external predictions unless every original input is compatible in every cycle'}
    core.write_json(out/'frozen_transport_gate.json',result)
    return result


def audit(raw, fields, cohort, dataset, out):
    out.mkdir(parents=True,exist_ok=True)
    raw.to_csv(out/'raw_unimputed.csv',index=False)
    fields.to_csv(out/'source_fields.csv',index=False)
    core.write_json(out/'cohort.json',cohort)
    core.write_json(out/'dataset_profile.json',profile(dataset))
    summaries=[]
    for seed in PROTOCOL['split_seeds']:
        sets, assignment=split_data(raw,seed)
        index=raw[[c for c in ['SEQN','family_id'] if c in raw]].copy()
        index['split']=assignment
        index.to_csv(out/f'splits_seed_{seed}.csv',index=False)
        for part,d in sets.items():
            summaries.append({'seed':seed,'split':part,'n':len(d),'positives':int(d[LABEL].sum()),
                              'positive_rate':d[LABEL].mean(),'families':d.family_id.nunique() if 'family_id' in d else None})
    pd.DataFrame(summaries).to_csv(out/'split_summary.csv',index=False)
    if dataset=='nhanes':
        transport_audit(raw,fields,out)


def summarize(out):
    frames=[pd.read_csv(out/f'seed_{s}/metrics.csv') for s in PROTOCOL['split_seeds'] if (out/f'seed_{s}/metrics.csv').exists()]
    if frames:
        merged=pd.concat(frames,ignore_index=True)
        merged.to_csv(out/'all_split_metrics.csv',index=False)
        merged.groupby('Model')[['ROC_AUC','AP','Brier','F2','Sensitivity','Specificity','PPV','ECE','Brier_Skill']].agg(['mean','std','count']).to_csv(out/'summary.csv')
        table=[]
        for name,group in merged.groupby('Model'):
            row={'Model':name,'Completed_Splits':group.seed.nunique()}
            for metric in ['ROC_AUC','AP','Brier','F2']:
                digits=4 if metric=='Brier' else 3
                row[metric]=f'{group[metric].mean():.{digits}f} ± {group[metric].std(ddof=1):.{digits}f}'
            table.append(row)
        pd.DataFrame(table).to_csv(out/'paper_table.csv',index=False,encoding='utf-8-sig')
        core.write_json(out/'summary_status.json',{'completed_seeds':sorted(merged.seed.unique().tolist()),
                        'expected_seeds':PROTOCOL['split_seeds'],'complete':len(frames)==len(PROTOCOL['split_seeds']),
                        'interpretation':'overlapping repeated partitions; SD is not an independent-replication SE'})


def train(raw, dataset, out, seed):
    folder=out/f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/'run.lock'), timeout=0):
        sets, assignment=split_data(raw,seed)
        raw[['SEQN']].assign(split=assignment).to_csv(folder/'splits.csv',index=False)
        core.write_json(folder/'status.json',{'state':'running','experiment':'independent retraining'})
        prepared=prepare(sets,profile(dataset),folder)
        run_models(prepared,sets,profile(dataset),folder)
        inference(folder,seed,float(sets['fit'][LABEL].mean()))
        core.write_json(folder/'status.json',{'state':'complete','experiment':'independent retraining'})
    summarize(out)


def transport(raw, fields, out):
    if not transport_audit(raw,fields,out)['eligible']:
        raise RuntimeError('Frozen-model transport blocked by incompatible native fields. See frozen_transport_gate.json. No training or external prediction was performed.')
    # Deliberately has no MIDUS entry point and no all-missing-column substitution.
    source=ROOT/PROTOCOL['frozen_transport']['source_run']
    final=ROOT/PROTOCOL['frozen_transport']['final_run']
    decisions=[json.loads((final/f'{stage}_decision.json').read_text())['choice'] for stage in ['calibration','de_budget']]
    if decisions!=['positive_slope_platt',240]:
        raise RuntimeError('Final development choices differ from this external protocol')
    for seed in PROTOCOL['split_seeds']:
        folder=out/f'seed_{seed}';folder.mkdir(exist_ok=True)
        old=source/f'seed_{seed}'
        prepared=joblib.load(old/'prepared.joblib')
        # The original pipeline and original models are used without fitting on external data.
        scores=prepared['pipe'].predict(raw)
        specs=core.specs(prepared['groups'])
        joblib.dump({'groups':prepared['groups']},folder/'prepared.joblib')
        names=['V0_Mean']+list(specs)+[f'{m}_{f}' for f in ['Pathway9','Raw'] for m in ['LR','RF','LightGBM']]
        previous=folder/'development_artifacts_sha256.json'
        used_hashes=json.loads(previous.read_text(encoding='utf-8')) if previous.exists() else {}
        used_hashes[str(old/'prepared.joblib')]=digest(old/'prepared.joblib')
        for name in names:
            dest=folder/'models'/name;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'complete.json').exists():
                continue
            src=(final/f'seed_{seed}' if name in specs else old)/'models'/name
            cal_frame=pd.read_csv(src/'cal_predictions.csv')
            val_frame=pd.read_csv(src/'val_predictions.csv')
            cal=PositivePlatt().fit(cal_frame.raw_prob,cal_frame.y_true)
            tau=core.threshold(val_frame.y_true.to_numpy(),cal.predict(val_frame.raw_prob.to_numpy()))
            joblib.dump(cal,dest/'calibrator.joblib')
            core.write_json(dest/'threshold.json',{'tau':tau,'selected_on':'original development Val','calibrator_fit_on':'original development Cal'})
            for f in ['cal_predictions.csv','val_predictions.csv']:
                used_hashes[str(src/f)]=digest(src/f)
            bel=mass=None
            if name in specs:
                bundle=joblib.load(src/'model.joblib')
                p,bel,mass=bundle['model'].predict(scores,bundle['params'],details=True)
            elif name=='V0_Mean':
                p=scores[:,:6].mean(axis=1)
            else:
                model,scaler=joblib.load(src/'model.joblib')
                x=scores if name.endswith('Pathway9') else prepared['pipe'].processor.transform(raw).to_numpy()
                p=model.predict_proba(scaler.transform(x) if scaler else x)[:,1]
            if name!='V0_Mean':
                used_hashes[str(src/'model.joblib')]=digest(src/'model.joblib')
            calibrated=cal.predict(p)
            frame=pd.DataFrame({'SEQN':raw.SEQN,'y_true':raw[LABEL],'raw_prob':p,'calibrated_prob':calibrated,'prediction':(calibrated>=tau).astype(int)})
            if bel is not None:
                frame['Bel_D'],frame['Pl_D'],frame['m_Theta']=bel,bel+mass,mass
            frame.to_csv(dest/'test_predictions.csv',index=False)
            core.write_json(dest/'complete.json',{'state':'complete','external_retraining':False})
        core.write_json(folder/'development_artifacts_sha256.json',used_hashes)
        fit=pd.read_csv(old/'fit_pathways.csv')
        inference(folder,seed,float(fit.label.mean()))
    summarize(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['audit','train','transport'])
    parser.add_argument('--dataset',choices=['nhanes','midus'],required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--seed',type=int,choices=PROTOCOL['split_seeds'])
    args=parser.parse_args()
    if args.stage=='transport' and args.dataset!='nhanes':
        parser.error('MIDUS native variables/outcome are not a transport validation of the NHANES model')
    mode='frozen_transport' if args.stage=='transport' else 'independent_retraining'
    out=args.out or ROOT/'runs'/f'external_v1_{args.dataset}_{mode}'
    if args.stage=='audit':
        out=args.out or ROOT/'runs'/f'external_v1_{args.dataset}_audit'
    out=out.resolve()
    if not out.name.startswith('external_'):
        parser.error('Output directory name must start with external_ to protect historical runs')
    # Audit is read-only with respect to source data and historical runs.
    if args.stage!='audit':
        freeze(out,args.dataset,mode)
    raw,fields,cohort=load_data(args.dataset)
    audit(raw,fields,cohort,args.dataset,out)
    if args.stage=='train':
        for seed in [args.seed] if args.seed else PROTOCOL['split_seeds']:
            train(raw,args.dataset,out,seed)
    elif args.stage=='transport':
        transport(raw,fields,out)
    print(f'{args.stage} finished: {out}',flush=True)


if __name__=='__main__':
    main()
