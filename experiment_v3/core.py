"""Frozen v3: repeated splits, no imbalance weighting, matched-input control."""
from pathlib import Path
import contextlib
import importlib.util
import itertools
import json
import random
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, fbeta_score, confusion_matrix
from scipy.optimize import differential_evolution
from sklearn.model_selection import ParameterGrid

ROOT = Path(__file__).resolve().parents[1]
PATHS = ['P1_Socioeconomic', 'P2_Sleep', 'P3_HealthStatus', 'P4_FoodSecurity', 'P5_Clinical', 'P6_Substance']
DEMOS = ['RIDAGEYR', 'RIAGENDR', 'DMDMARTL']
META = ['SEQN','CYCLE','depression_label','PHQ9_score','WTINT2YR','WTMEC2YR','WTMEC4YR','SDMVPSU','SDMVSTRA']
CONFIG = json.loads((Path(__file__).parent/'config.json').read_text(encoding='utf-8'))

def legacy(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')

def feature_audit(columns, out):
    definitions=legacy('nhanes_step2_feature_engineering').PATHWAY_VARIABLES
    rows=[]
    forbidden={c.upper() for c in META}|{'LABEL','SPLIT'}
    for col in columns:
        base=col.removesuffix('_missing')
        safe=base.upper() not in forbidden and not base.upper().startswith('DPQ')
        paths=[p for p,vs in definitions.items() if base in vs]
        rows.append({'Field':col,'Base_Field':base,'Paths':';'.join(paths) or 'Raw_ML_only',
            'Missing_Indicator':col.endswith('_missing'),'Direct_Demographic_Input':base in DEMOS,
            'Forbidden_Outcome_or_Metadata':not safe})
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'feature_audit.csv',index=False)
    assert not frame.Forbidden_Outcome_or_Metadata.any(), 'Outcome/metadata leakage in actual model inputs'
    write_json(out/'feature_audit.json',{'passed':True,'dimension':len(rows),
        'forbidden_prefix':'DPQ','forbidden_names':sorted(forbidden),'pathway_choice':'historically prespecified; not newly selected on Val/Test'})

class Processor:
    def fit(self, raw):
        candidates = [c for c in raw if c not in META and not c.endswith('_missing')]
        rates = raw[candidates].isna().mean()
        self.cols = rates[rates <= .4].index.tolist()
        self.indicators = rates[(rates >= .2) & (rates <= .4)].index.tolist()
        self.dropped = rates[rates > .4].index.tolist()
        self.imputer = IterativeImputer(random_state=42, max_iter=20, skip_complete=True)
        self.imputer.fit(raw[self.cols])
        return self

    def transform(self, raw):
        x = pd.DataFrame(np.maximum(self.imputer.transform(raw[self.cols]), 0), columns=self.cols, index=raw.index)
        for col in self.indicators:
            x[col+'_missing'] = raw[col].isna().astype(int)
        return x

class PathwayPipeline:
    def fit(self, raw, y):
        self.processor = Processor().fit(raw)
        x = self.processor.transform(raw)
        definitions = legacy('nhanes_step2_feature_engineering').PATHWAY_VARIABLES
        self.models = {}
        for path, variables in definitions.items():
            cols = [c for v in variables for c in [v, v+'_missing'] if c in x]
            if not cols:
                raise ValueError(f'No available variables for {path}')
            scaler = None
            if path in [PATHS[0], PATHS[4]]:
                model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=.05,
                    subsample=.8, min_samples_leaf=20, random_state=42)
                model.fit(x[cols], y)
            else:
                scaler = StandardScaler().fit(x[cols])
                model = LogisticRegression(C=.1, class_weight=None, max_iter=1000, random_state=42)
                model.fit(scaler.transform(x[cols]), y)
            self.models[path] = (cols, scaler, model)
        self.demo_scaler = MinMaxScaler().fit(x[DEMOS])
        return self

    def predict(self, raw):
        x = self.processor.transform(raw)
        scores = []
        for cols, scaler, model in self.models.values():
            xx = scaler.transform(x[cols]) if scaler is not None else x[cols]
            scores.append(model.predict_proba(xx)[:,1])
        return np.column_stack(scores + [self.demo_scaler.transform(x[DEMOS])])

def crossfit(raw, seed=42):
    y = raw.depression_label.to_numpy()
    scores = np.zeros((len(raw),9))
    for fold, (ti, vi) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(raw,y)):
        print(f'  pathway OOF {fold+1}/5, n={len(ti)}', flush=True)
        pipe = PathwayPipeline().fit(raw.iloc[ti], y[ti])
        scores[vi] = pipe.predict(raw.iloc[vi])
    return scores, PathwayPipeline().fit(raw,y)

def prepare(out, split_seed=42):
    raw_path = out/'raw_unimputed.csv'
    if not raw_path.exists():
        mod = legacy('nhanes_step1_data_processing')
        raw = mod.create_combined_weight(mod.clean_variables(mod.initial_filter(mod.load_all_cycles(mod.DATA_DIR))))
        raw.to_csv(raw_path,index=False)
    raw = pd.read_csv(raw_path)
    if raw.SEQN.duplicated().any():
        raise ValueError('Duplicate participant identifiers')
    split_path = out/'splits.csv'
    if not split_path.exists():
        train, test = train_test_split(np.arange(len(raw)), test_size=.2, stratify=raw.depression_label, random_state=split_seed)
        fit, temp = train_test_split(train, test_size=.4, stratify=raw.iloc[train].depression_label, random_state=split_seed)
        cal, val = train_test_split(temp, test_size=.5, stratify=raw.iloc[temp].depression_label, random_state=split_seed)
        assignments = pd.Series(index=np.arange(len(raw)),dtype='object')
        for name, ids in [('fit',fit),('cal',cal),('val',val),('test',test)]: assignments.loc[ids]=name
        pd.DataFrame({'SEQN':raw.SEQN, 'split':assignments}).to_csv(split_path,index=False)
    splits = pd.read_csv(split_path)
    raw = raw.merge(splits, on='SEQN', validate='one_to_one')
    sets = {s:raw[raw.split==s].drop(columns='split').reset_index(drop=True) for s in ['fit','cal','val','test']}
    write_json(out/'split_summary.json', {s:{'n':len(d),'positives':int(d.depression_label.sum())} for s,d in sets.items()})
    cache_path = out/'prepared.joblib'
    if cache_path.exists(): return joblib.load(cache_path), sets
    fit = sets['fit']
    print('Preparing full-Fit OOF pathways',flush=True)
    fit_scores, pipe = crossfit(fit)
    scores = {'fit':fit_scores, **{s:pipe.predict(sets[s]) for s in ['cal','val','test']}}
    proc_fit = pipe.processor.transform(fit)
    feature_audit(proc_fit.columns, out)
    causal = legacy('nhanes_step1_5_causal_discovery')
    reps = [v for vs in causal.PATHWAY_REPRESENTATIVES.values() for v in vs if v in proc_fit]
    causal_data = proc_fit[reps].copy()
    causal_data['depression_label'] = fit.depression_label.to_numpy()
    with (out/'causal_discovery.log').open('w',encoding='utf-8') as log, contextlib.redirect_stdout(log):
        _, edges = causal.run_pc_with_bk(causal_data, list(causal_data), 'depression_label')
        _, distances, roles = causal.build_dag_and_roles(list(causal_data), edges)
        mapping, excluded = causal.cluster_pathways_weighted(causal_data,reps,roles)
    groups = [[PATHS.index(p) for p in mapping[k]] for k in ['Sub1','Sub2','Sub3']]
    write_json(out/'hierarchy.json', {'mapping':mapping,'groups':groups,'excluded':excluded,'edges':edges,
        'distances':{k:(v if np.isfinite(v) else None) for k,v in distances.items()},
        'scope':'Full Fit only; structure held fixed in internal CV'})
    pd.DataFrame(edges,columns=['source','target']).to_csv(out/'causal_edges.csv',index=False)
    write_json(out/'features.json',{'raw_variables':pipe.processor.cols,'missing_indicators':pipe.processor.indicators,
        'raw_model_columns':list(proc_fit),'raw_dimension':len(proc_fit.columns),'dropped':pipe.processor.dropped,
        'pathway_columns':PATHS+DEMOS})
    for s,d in sets.items():
        frame = pd.DataFrame(scores[s],columns=PATHS+DEMOS)
        frame.insert(0,'SEQN',d.SEQN); frame.insert(1,'label',d.depression_label)
        frame.to_csv(out/f'{s}_pathways.csv',index=False)
    folds = []
    for fold,(ti,vi) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(fit,fit.depression_label)):
        print(f'Preparing BRB outer fold {fold+1}/3',flush=True)
        train_scores, fold_pipe = crossfit(fit.iloc[ti].reset_index(drop=True))
        valid_scores = fold_pipe.predict(fit.iloc[vi])
        folds.append((train_scores,valid_scores,fit.depression_label.to_numpy()[vi]))
        pd.DataFrame({'SEQN':fit.iloc[vi].SEQN,'fold':fold}).to_csv(out/f'cv_fold_{fold}_validation.csv',index=False)
    prepared = {'scores':scores,'pipe':pipe,'groups':groups,'folds':folds}
    joblib.dump(prepared,cache_path)
    return prepared, sets

def anchors(x,levels):
    if x.shape[1]==0: return np.empty((levels,0))
    a = np.percentile(x,[16,50,84] if levels==3 else [16,84],axis=0)
    for i in range(1,levels): a[i] = np.maximum(a[i],a[i-1]+1e-5)
    return a

def active(x,a):
    """At most two nonzero memberships per input; retain only active rules."""
    n,d=x.shape; levels=len(a)
    indices=np.zeros((n,1),dtype=np.int64); weights=np.ones((n,1))
    for j in range(d):
        hi=np.searchsorted(a[:,j],x[:,j],side='right').clip(1,levels-1)
        lo=hi-1
        t=((x[:,j]-a[lo,j])/(a[hi,j]-a[lo,j])).clip(0,1)
        indices=np.concatenate([indices*levels+lo[:,None],indices*levels+hi[:,None]],axis=1)
        weights=np.concatenate([weights*(1-t[:,None]),weights*t[:,None]],axis=1)
    # Pack nonzero activations only; mathematically identical, important for 9-D flat BRB.
    keep=weights>0
    starts=np.r_[0,np.cumsum(keep.sum(axis=1))[:-1]]
    return indices[keep],weights[keep],starts

def fuse(activity,b,w):
    """Closed form of normalized conjunctive binary DS fusion."""
    ids,act,starts=activity; tw=act*w[ids]; beta=b[ids]
    ad=np.multiply.reduceat(1-tw*(1-beta),starts)
    an=np.multiply.reduceat(1-tw*beta,starts)
    at=np.multiply.reduceat(1-tw,starts)
    denom=np.maximum(ad+an-at,1e-12)
    return np.clip((ad-at)/denom,0,1),np.clip(at/denom,0,1)

class BRB:
    def __init__(self, fit, groups, weighted=True, uncertainty=True, flat=False, levels=3, monotonic=True):
        self.groups=groups; self.weighted=weighted; self.uncertainty=uncertainty
        self.flat=flat; self.levels=levels; self.monotonic=monotonic
        dims=[len(g) for g in groups]+([] if flat else [4 if uncertainty else 3])
        self.dims=dims; self.sizes=[levels**d for d in dims]; self.nr=sum(self.sizes)
        self.n_params=self.nr*(2 if weighted else 1)
        self.a=[anchors(fit[:,g],levels) for g in groups]
        # Fixed semantic probability anchors avoid dummy-0.5 degeneracy and parameter-dependent references.
        self.main_a=np.tile(np.array([0.,.5,1.] if levels==3 else [0.,1.])[:,None],(1,dims[-1])) if not flat else None

    def repair(self,p):
        b=np.clip(p[:self.nr],0,1).copy(); offset=0
        if self.monotonic:
            for dim,size in zip(self.dims,self.sizes):
                if dim:
                    grid=b[offset:offset+size].reshape((self.levels,)*dim)
                    for axis in range(dim): grid=np.maximum.accumulate(grid,axis=axis)
                    b[offset:offset+size]=grid.ravel()
                offset+=size
        if self.weighted: return np.r_[b,np.clip(p[self.nr:],.01,1)]
        return b

    def activity(self,x): return [active(x[:,g],a) for g,a in zip(self.groups,self.a)]

    def predict(self,x,p,activity=None,details=False):
        p=self.repair(p); b=p[:self.nr]; w=p[self.nr:] if self.weighted else np.ones(self.nr)
        acts=self.activity(x) if activity is None else activity
        mids=[]; uncs=[]; offset=0
        for act,size in zip(acts,self.sizes):
            m,t=fuse(act,b[offset:offset+size],w[offset:offset+size]); offset+=size
            mids.append(m+.5*t); uncs.append(t)
        if not self.flat:
            main=np.column_stack(mids+([np.mean(uncs,axis=0)] if self.uncertainty else []))
            m,t=fuse(active(main,self.main_a),b[offset:],w[offset:])
        prob=np.clip(m+.5*t,0,1)
        return (prob,m,t) if details else prob

def metrics(y,p,tau):
    pred=p>=tau; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'ROC_AUC':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,
        'AP':float(average_precision_score(y,p)) if np.sum(y) else None,
        'Brier':float(brier_score_loss(y,p)),'F2':float(fbeta_score(y,pred,beta=2,zero_division=0)),
        'Tau':float(tau),'Sensitivity':float(tp/max(tp+fn,1)),'Specificity':float(tn/max(tn+fp,1)),
        'PPV':float(tp/(tp+fp)) if tp+fp else None,
        'N':len(y),'Positives':int(np.sum(y))}

def threshold(y,p):
    # All distinct decision sets; ties resolved toward the higher, more specific threshold.
    candidates=np.round(np.arange(5,96)/100,2)
    values=np.array([fbeta_score(y,p>=t,beta=2,zero_division=0) for t in candidates])
    return float(candidates[np.flatnonzero(values==values.max())[-1]])

def optimize(name,config,prepared,out,maxiter=120,population=28):
    folder=out/'models'/name; folder.mkdir(parents=True,exist_ok=True)
    artifact=folder/'model.joblib'
    if artifact.exists(): return joblib.load(artifact)
    cache=[]
    for tr,va,y in prepared['folds']:
        model=BRB(tr,**config); cache.append((model,va,y,model.activity(va)))
    final=BRB(prepared['scores']['fit'],**config)
    rng=np.random.default_rng(42)
    base=np.concatenate([np.mean(np.array(list(itertools.product(range(final.levels),repeat=d))),axis=1)/(final.levels-1)
        if d else np.array([.5]) for d in final.dims])
    if final.weighted: base=np.r_[base,np.ones(final.nr)]
    init=[final.repair(base)]
    for i in range(population-1):
        p=base+rng.normal(0,.06 if i<9 else .15,final.n_params) if i<18 else rng.random(final.n_params)
        init.append(final.repair(p))
    def objective(p):
        losses=[]
        for model,x,y,act in cache:
            pr=model.predict(x,p,act)
            losses.append(.45*(1-roc_auc_score(y,pr))+.25*(1-average_precision_score(y,pr))+.3*brier_score_loss(y,pr))
        return float(np.mean(losses))
    rows=[]; start=time.time()
    def callback(intermediate_result):
        rows.append({'generation':len(rows)+1,'loss':float(intermediate_result.fun),'seconds':time.time()-start})
        pd.DataFrame(rows).to_csv(folder/'convergence.csv',index=False)
        if len(rows)%10==0 or len(rows)==1: print(f'{name}: generation={len(rows)} loss={intermediate_result.fun:.6f}',flush=True)
    result=differential_evolution(objective,[(0.,1.)]*final.n_params,init=np.array(init),maxiter=maxiter,
        strategy='best1bin',mutation=(.5,1.),recombination=.7,tol=1e-5,seed=42,polish=False,workers=1,callback=callback)
    bundle={'model':final,'params':final.repair(result.x),'config':config}
    write_json(folder/'optimization.json',{'rules':final.nr,'parameters':final.n_params,'objective':result.fun,
        'success':result.success,'message':str(result.message),'generations':result.nit,'evaluations':result.nfev,
        'seconds':time.time()-start,'scope':'Fit only; full-Fit fixed causal structure; fold-local pathway cross-fitting'})
    joblib.dump(bundle,artifact)
    return bundle

def evaluate(name,preds,sets,out,details=None):
    folder=out/'models'/name; folder.mkdir(parents=True,exist_ok=True)
    result_path=folder/'metrics.json'
    if result_path.exists(): return json.loads(result_path.read_text(encoding='utf-8'))
    iso=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1).fit(preds['cal'],sets['cal'].depression_label)
    calibrated={s:iso.predict(preds[s]) for s in ['cal','val','test']}
    tau=threshold(sets['val'].depression_label.to_numpy(),calibrated['val'])
    joblib.dump(iso,folder/'calibrator.joblib')
    write_json(folder/'threshold.json',{'tau':tau,'selected_on':'val','criterion':'F2','tie_break':'highest threshold'})
    for s in ['cal','val','test']:
        frame=pd.DataFrame({'SEQN':sets[s].SEQN,'y_true':sets[s].depression_label,
            'raw_prob':preds[s],'calibrated_prob':calibrated[s],'prediction':(calibrated[s]>=tau).astype(int)})
        if details is not None:
            _,m,t=details[s]; frame['Bel_D']=m; frame['Pl_D']=m+t; frame['m_Theta']=t
        frame.to_csv(folder/f'{s}_predictions.csv',index=False)
    result={'Model':name,**metrics(sets['test'].depression_label.to_numpy(),calibrated['test'],tau)}
    # Raw threshold also uses Val; raw results are supplementary, never used for selection.
    raw_tau=threshold(sets['val'].depression_label.to_numpy(),preds['val'])
    result.update({'Raw_'+k:v for k,v in metrics(sets['test'].depression_label.to_numpy(),preds['test'],raw_tau).items()})
    write_json(result_path,result)
    return result

def specs(groups):
    g=[groups[0]+[6,7,8],groups[1],groups[2]]
    result={'Causal_BRB':dict(groups=g), 'V1_Flat':dict(groups=[list(range(6))],flat=True),
        'V1_MatchedInputs':dict(groups=[sorted(set(i for group in g for i in group))],flat=True),
        'V2_Unweighted':dict(groups=g,weighted=False),'V3_NoUnc':dict(groups=g,uncertainty=False),
        'A2_Basic':dict(groups=g,weighted=False,uncertainty=False),
        'HBRB_b':dict(groups=[[0,6,7,8,3],[1,5],[2,4]],uncertainty=False,monotonic=False),
        'BRB_DR':dict(groups=[list(range(6))],flat=True,levels=2,monotonic=False),
        'SBRB_I':dict(groups=[list(range(6))],flat=True,levels=2,weighted=False,monotonic=False)}
    included=[i for group in groups for i in group]
    for seed in [42,99,123,2026,888]:
        shuffled=included.copy(); random.Random(seed).shuffle(shuffled)
        n1,n2=map(len,groups[:2])
        result[f'V4_Random_{seed}']=dict(groups=[shuffled[:n1]+[6,7,8],shuffled[n1:n1+n2],shuffled[n1+n2:]])
    return result

def make_ml(name, params):
    from lightgbm import LGBMClassifier
    if name=='LR': return LogisticRegression(class_weight=None,max_iter=2000,random_state=42,**params)
    if name=='RF': return RandomForestClassifier(class_weight=None,random_state=42,n_jobs=2,**params)
    return LGBMClassifier(class_weight=None,random_state=42,verbosity=-1,n_jobs=2,
        subsample=.8,colsample_bytree=.8,**params)

def ml_cv_cache(sets, out):
    path=out/'ml_cv_cache.joblib'
    if path.exists(): return joblib.load(path)
    fit=sets['fit']; y=fit.depression_label.to_numpy(); cache={'Pathway9':[],'Raw':[]}
    for fold,(ti,vi) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(fit,y)):
        print(f'ML selection fold {fold+1}/5: independently cross-fitting upstream paths',flush=True)
        train=fit.iloc[ti].reset_index(drop=True); valid=fit.iloc[vi].reset_index(drop=True)
        tr,pipe=crossfit(train)
        cache['Pathway9'].append((tr,pipe.predict(valid),y[ti],y[vi]))
        cache['Raw'].append((pipe.processor.transform(train).to_numpy(),pipe.processor.transform(valid).to_numpy(),y[ti],y[vi]))
    joblib.dump(cache,path)
    return cache

def fit_ml_models(prepared,sets,out):
    cache=ml_cv_cache(sets,out)
    scores=prepared['scores']; proc=prepared['pipe'].processor
    features={'Pathway9':scores,'Raw':{s:proc.transform(d).to_numpy() for s,d in sets.items()}}
    results=[]; selections=[]
    for feature_name,xx in features.items():
        for model_name in ['LR','RF','LightGBM']:
            name=f'{model_name}_{feature_name}'; folder=out/'models'/name; folder.mkdir(parents=True,exist_ok=True)
            path=folder/'model.joblib'
            if path.exists():
                model,scaler=joblib.load(path)
                selected=json.loads((folder/'selection.json').read_text(encoding='utf-8'))
            else:
                trials=[]
                for candidate in ParameterGrid(CONFIG['ml_grid'][model_name]):
                    values=[]
                    for tr,va,yt,yv in cache[feature_name]:
                        scaler=StandardScaler().fit(tr) if model_name=='LR' else None
                        model=make_ml(model_name,candidate)
                        model.fit(scaler.transform(tr) if scaler is not None else tr,yt)
                        p=model.predict_proba(scaler.transform(va) if scaler is not None else va)[:,1]
                        values.append(average_precision_score(yv,p))
                    trials.append({'parameters':candidate,'mean_AP':float(np.mean(values)),'fold_AP':values})
                best=max(range(len(trials)),key=lambda i:trials[i]['mean_AP'])
                selected={'model':name,'selected_on':'Fit five-fold CV only','scoring':'AP',
                    'best_parameters':trials[best]['parameters'],'best_CV_AP':trials[best]['mean_AP'],'trials':trials}
                write_json(folder/'selection.json',selected)
                scaler=StandardScaler().fit(xx['fit']) if model_name=='LR' else None
                model=make_ml(model_name,selected['best_parameters'])
                model.fit(scaler.transform(xx['fit']) if scaler is not None else xx['fit'],sets['fit'].depression_label)
                joblib.dump((model,scaler),path)
            preds={s:model.predict_proba(scaler.transform(xx[s]) if scaler is not None else xx[s])[:,1] for s in ['cal','val','test']}
            results.append(evaluate(name,preds,sets,out))
            selections.append({'Model':name,'CV_AP':selected['best_CV_AP'],**selected['best_parameters']})
    pd.DataFrame(selections).to_csv(out/'ml_selected_parameters.csv',index=False)
    return results

def run_models(prepared,sets,out,maxiter=120,population=28):
    scores=prepared['scores']; rows=[]
    preds={s:scores[s][:,:6].mean(axis=1) for s in ['cal','val','test']}
    rows.append(evaluate('V0_Mean',preds,sets,out))
    for name,config in specs(prepared['groups']).items():
        print(f'Training {name}',flush=True)
        bundle=optimize(name,config,prepared,out,maxiter,population)
        details={s:bundle['model'].predict(scores[s],bundle['params'],details=True) for s in ['cal','val','test']}
        rows.append(evaluate(name,{s:v[0] for s,v in details.items()},sets,out,details))
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    rows.extend(fit_ml_models(prepared,sets,out))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    return rows
