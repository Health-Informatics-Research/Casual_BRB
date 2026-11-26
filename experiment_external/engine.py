"""Reuse the frozen BRB inference/DE; adapt only cohort inputs and evaluation protocol."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer, OrdinalEncoder
from sklearn.metrics import average_precision_score
from experiment_v3 import core
from experiment_v3.statistics import bootstrap_metrics, KEYS
from .data import LABEL, META, PROTOCOL


def assert_separate(a, b):
    assert set(a.SEQN).isdisjoint(b.SEQN), 'Participant overlap'
    if 'family_id' in a:
        assert set(a.family_id).isdisjoint(b.family_id), 'Family overlap'


def cv_indices(raw, folds, seed=42):
    y = raw[LABEL].to_numpy()
    if np.unique(y).size != 2:
        raise ValueError('Both classes are required')
    if 'family_id' in raw:
        iterator = StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(raw, y, raw.family_id)
    else:
        iterator = StratifiedKFold(folds, shuffle=True, random_state=seed).split(raw, y)
    for tr, va in iterator:
        assert_separate(raw.iloc[tr], raw.iloc[va])
        if any(raw.iloc[idx][LABEL].nunique() != 2 for idx in [tr, va]):
            raise ValueError('One-class fold; stop rather than change the frozen split')
        yield tr, va


def split_data(raw, seed):
    if 'family_id' in raw:
        assignments = np.empty(len(raw), dtype=object)
        for fold, (_, vi) in enumerate(cv_indices(raw, 25, seed)):
            assignments[vi] = 'test' if fold < 5 else 'cal' if fold < 9 else 'val' if fold < 13 else 'fit'
    else:
        y = raw[LABEL]
        train, test = train_test_split(np.arange(len(raw)), test_size=.2, stratify=y, random_state=seed)
        fit, temp = train_test_split(train, test_size=.4, stratify=y.iloc[train], random_state=seed)
        cal, val = train_test_split(temp, test_size=.5, stratify=y.iloc[temp], random_state=seed)
        assignments = np.empty(len(raw), dtype=object)
        for name, ix in [('fit', fit), ('cal', cal), ('val', val), ('test', test)]:
            assignments[ix] = name
    sets = {s: raw.loc[assignments == s].reset_index(drop=True) for s in ['fit','cal','val','test']}
    for i, s in enumerate(sets):
        for t in list(sets)[i+1:]:
            assert_separate(sets[s], sets[t])
    return sets, assignments


class Processor(core.Processor):
    def fit(self, raw):
        return super().fit(raw.drop(columns=[c for c in META if c in raw]))


class PathwayPipeline:
    def __init__(self, profile):
        self.profile = profile

    def fit(self, raw, y):
        self.processor = Processor().fit(raw)
        x = self.processor.transform(raw)
        self.models = []
        for i, variables in enumerate(self.profile['paths']):
            cols = [c for v in variables for c in [v, v+'_missing'] if c in x]
            if not cols:
                raise ValueError(f'No Fit-eligible variables in pathway P{i+1}; no invented fallback')
            if i in [0, 4]:
                scaler = None
                model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=.05,
                         subsample=.8, min_samples_leaf=20, random_state=42)
                model.fit(x[cols], y)
            else:
                scaler = StandardScaler().fit(x[cols])
                model = LogisticRegression(C=.1, class_weight=None, max_iter=1000, random_state=42)
                model.fit(scaler.transform(x[cols]), y)
            self.models.append((cols, scaler, model))
        self.demo_scaler = MinMaxScaler().fit(x[self.profile['demos']])
        return self

    def predict(self, raw):
        x = self.processor.transform(raw)
        values = [m.predict_proba(sc.transform(x[c]) if sc is not None else x[c])[:, 1]
                  for c, sc, m in self.models]
        return np.column_stack(values + [self.demo_scaler.transform(x[self.profile['demos']])])


def crossfit(raw, profile):
    scores = np.zeros((len(raw), 9))
    for ti, vi in cv_indices(raw, 5):
        pipe = PathwayPipeline(profile).fit(raw.iloc[ti], raw.iloc[ti][LABEL])
        scores[vi] = pipe.predict(raw.iloc[vi])
    return scores, PathwayPipeline(profile).fit(raw, raw[LABEL])


def discover(processed, y, profile):
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import gsq
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    reps = [v for group in profile['representatives'] for v in group if v in processed]
    data = processed[reps].copy()
    data[LABEL] = np.asarray(y)
    encoded = np.zeros(data.shape)
    for i, col in enumerate(data):
        est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile') if data[col].nunique() > 10 else OrdinalEncoder()
        encoded[:, i] = est.fit_transform(data[[col]]).ravel()
    names = list(data)
    nodes = [GraphNode(n) for n in names]
    tiers = {**profile['tiers'], LABEL: 4}
    bk = BackgroundKnowledge()
    for i, n in enumerate(names):
        for j, m in enumerate(names):
            if tiers.get(n, 99) > tiers.get(m, 99):
                bk.add_forbidden_by_node(nodes[i], nodes[j])
    graph = pc(encoded, alpha=.10, indep_test=gsq, stable=True, node_names=names,
               background_knowledge=bk, show_progress=False)
    adj = graph.G.graph
    edges = [(names[i], names[j]) for i in range(len(names)) for j in range(len(names))
             if adj[i,j] == -1 and adj[j,i] == 1]
    legacy = core.legacy('nhanes_step1_5_causal_discovery')
    legacy.PATHWAY_REPRESENTATIVES = {f'P{i+1}': vs for i, vs in enumerate(profile['representatives'])}
    _, distances, roles = legacy.build_dag_and_roles(names, edges)
    mapping, excluded = legacy.cluster_pathways_weighted(data, reps, roles)
    groups = [[int(p[1:])-1 for p in mapping[name]] for name in ['Sub1','Sub2','Sub3']]
    return groups, dict(mapping=mapping, excluded=excluded, edges=edges, roles=roles,
                        tiers=tiers, representatives=reps,
                        distances={k: float(v) if np.isfinite(v) else None for k,v in distances.items()},
                        scope='complete Fit only, held fixed in internal CV, not fully nested')


def prepare(sets, profile, out):
    cache = out/'prepared.joblib'
    if cache.exists():
        return joblib.load(cache)
    fit = sets['fit']
    fit_scores, pipe = crossfit(fit, profile)
    scores = {'fit': fit_scores, **{s: pipe.predict(sets[s]) for s in ['cal','val']}}
    x = pipe.processor.transform(fit)
    rows = []
    for c in x:
        base = c.removesuffix('_missing')
        assert base not in META and not base.startswith('DPQ'), f'Forbidden input {c}'
        rows.append({'Field': c, 'Base_Field': base, 'Paths': ';'.join(f'P{i+1}' for i, vs in enumerate(profile['paths']) if base in vs),
                     'Missing_Indicator': c.endswith('_missing')})
    pd.DataFrame(rows).to_csv(out/'feature_audit.csv', index=False)
    groups, hierarchy = discover(x, fit[LABEL], profile)
    core.write_json(out/'hierarchy.json', hierarchy)
    folds = []
    for f, (ti, vi) in enumerate(cv_indices(fit, 3)):
        train_scores, fold_pipe = crossfit(fit.iloc[ti].reset_index(drop=True), profile)
        folds.append((train_scores, fold_pipe.predict(fit.iloc[vi]), fit.iloc[vi][LABEL].to_numpy()))
        fit.iloc[vi][[c for c in ['SEQN','family_id'] if c in fit]].to_csv(out/f'cv_{f}_validation.csv', index=False)
    prepared = dict(scores=scores, pipe=pipe, groups=groups, folds=folds)
    for s, values in scores.items():
        frame=pd.DataFrame(values,columns=[f'P{i+1}' for i in range(6)]+profile['demos'])
        frame.insert(0,'SEQN',sets[s].SEQN)
        frame.insert(1,'label',sets[s][LABEL])
        frame.to_csv(out/f'{s}_pathways.csv',index=False)
    core.write_json(out/'features.json',{'raw_variables':pipe.processor.cols,
                    'missing_indicators':pipe.processor.indicators,'dropped':pipe.processor.dropped,
                    'raw_dimension':len(x.columns),'profile':profile})
    joblib.dump(prepared, cache)
    return prepared


class PositivePlatt:
    def fit(self, p, y):
        z = logit(np.clip(np.asarray(p), 1e-6, 1-1e-6))
        y = np.asarray(y)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError('Calibration requires two classes')
        def loss(params):
            s = params[0] + params[1]*z
            error = expit(s)-y
            return np.mean(np.logaddexp(0, s)-y*s), np.array([error.mean(), (error*z).mean()])
        fit = minimize(loss, [0., 1.], jac=True, bounds=[(None,None),(1e-8,None)], method='L-BFGS-B')
        if not fit.success:
            raise RuntimeError(fit.message)
        self.params = fit.x
        return self

    def predict(self, p):
        return expit(self.params[0] + self.params[1]*logit(np.clip(p, 1e-6, 1-1e-6)))


def ml_cache(fit, profile, out):
    path = out/'ml_cv_cache.joblib'
    if path.exists():
        return joblib.load(path)
    cache = {'Pathway9': [], 'Raw': []}
    for ti, vi in cv_indices(fit, 5):
        tr, va = fit.iloc[ti].reset_index(drop=True), fit.iloc[vi]
        sc, pipe = crossfit(tr, profile)
        cache['Pathway9'].append((sc, pipe.predict(va), tr[LABEL].to_numpy(), va[LABEL].to_numpy()))
        cache['Raw'].append((pipe.processor.transform(tr).to_numpy(), pipe.processor.transform(va).to_numpy(),
                             tr[LABEL].to_numpy(), va[LABEL].to_numpy()))
    joblib.dump(cache, path)
    return cache


def fit_ml(name, feature, prepared, sets, profile, out):
    folder = out/'models'/f'{name}_{feature}'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'model.joblib'
    if path.exists():
        return joblib.load(path)
    cache = ml_cache(sets['fit'], profile, out)[feature]
    trials = []
    for params in ParameterGrid(core.CONFIG['ml_grid'][name]):
        ap = []
        for tr, va, yt, yv in cache:
            scaler = StandardScaler().fit(tr) if name == 'LR' else None
            model = core.make_ml(name, params)
            model.fit(scaler.transform(tr) if scaler else tr, yt)
            p = model.predict_proba(scaler.transform(va) if scaler else va)[:,1]
            ap.append(average_precision_score(yv, p))
        trials.append({'parameters': params, 'mean_AP': float(np.mean(ap)), 'fold_AP': ap})
    selected = max(trials, key=lambda t:t['mean_AP'])
    core.write_json(folder/'selection.json', {'selected_on':'Fit CV only','trials':trials,'selected':selected})
    x = prepared['scores']['fit'] if feature == 'Pathway9' else prepared['pipe'].processor.transform(sets['fit']).to_numpy()
    scaler = StandardScaler().fit(x) if name == 'LR' else None
    model = core.make_ml(name, selected['parameters'])
    model.fit(scaler.transform(x) if scaler else x, sets['fit'][LABEL])
    joblib.dump((model, scaler), path)
    return model, scaler


def evaluate(name, predict, sets, out):
    folder = out/'models'/name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'complete.json').exists():
        return
    # Freeze calibration and threshold BEFORE accessing Test predictions/labels.
    preds = {s: predict(s) for s in ['cal','val']}
    cal = PositivePlatt().fit(preds['cal'][0], sets['cal'][LABEL])
    tau = core.threshold(sets['val'][LABEL].to_numpy(), cal.predict(preds['val'][0]))
    joblib.dump(cal, folder/'calibrator.joblib')
    core.write_json(folder/'threshold.json', {'tau': tau, 'selected_on': 'Val', 'calibrator_fit_on':'Cal'})
    preds['test'] = predict('test')
    for s, (raw, bel, mass) in preds.items():
        p = cal.predict(raw)
        frame = pd.DataFrame({'SEQN':sets[s].SEQN, 'y_true':sets[s][LABEL], 'raw_prob':raw,
                              'calibrated_prob':p, 'prediction':(p >= tau).astype(int)})
        if 'family_id' in sets[s]:
            frame['family_id'] = sets[s].family_id
        if bel is not None:
            frame['Bel_D'], frame['Pl_D'], frame['m_Theta'] = bel, bel+mass, mass
        frame.to_csv(folder/f'{s}_predictions.csv', index=False)
    core.write_json(folder/'complete.json', {'state':'complete','post_test_selection':False})


def run_models(prepared, sets, profile, out):
    scores = prepared['scores'].copy()
    def get_scores(s):
        if s not in scores:
            scores[s] = prepared['pipe'].predict(sets[s])
        return scores[s]
    evaluate('V0_Mean', lambda s:(get_scores(s)[:,:6].mean(axis=1),None,None), sets, out)
    for name, spec in core.specs(prepared['groups']).items():
        if (out/'models'/name/'complete.json').exists():
            continue
        bundle = core.optimize(name, spec, prepared, out, PROTOCOL['de_generations'], PROTOCOL['de_population'])
        evaluate(name, lambda s:bundle['model'].predict(get_scores(s), bundle['params'], details=True), sets, out)
    for feature in ['Pathway9','Raw']:
        for name in ['LR','RF','LightGBM']:
            full_name = f'{name}_{feature}'
            if (out/'models'/full_name/'complete.json').exists():
                continue
            model, scaler = fit_ml(name, feature, prepared, sets, profile, out)
            def predict(s):
                x = get_scores(s) if feature == 'Pathway9' else prepared['pipe'].processor.transform(sets[s]).to_numpy()
                return model.predict_proba(scaler.transform(x) if scaler else x)[:,1], None, None
            evaluate(full_name, predict, sets, out)


def inference(out, seed, reference):
    names = ['V0_Mean'] + list(core.specs(joblib.load(out/'prepared.joblib')['groups'])) + [f'{m}_{f}' for f in ['Pathway9','Raw'] for m in ['LR','RF','LightGBM']]
    data = {name:pd.read_csv(out/'models'/name/'test_predictions.csv').sort_values('SEQN') for name in names}
    first = data[names[0]]
    n = len(first)
    rng = np.random.default_rng(PROTOCOL['bootstrap_seed']+seed)
    if 'family_id' in first:
        _, inverse = np.unique(first.family_id, return_inverse=True)
        k = inverse.max()+1
        weights = rng.multinomial(k, np.full(k,1/k),size=PROTOCOL['test_bootstraps'])[:,inverse]
        unit = 'family cluster'
    else:
        weights = rng.multinomial(n, np.full(n,1/n),size=PROTOCOL['test_bootstraps'])
        unit = 'participant'
    rows, summaries, draws, points = [], [], {}, {}
    for name, d in data.items():
        np.testing.assert_array_equal(d.SEQN, first.SEQN)
        np.testing.assert_array_equal(d.y_true, first.y_true)
        tau = json.loads((out/'models'/name/'threshold.json').read_text())['tau']
        draws[name] = bootstrap_metrics(d.y_true, d.calibrated_prob, tau, reference, weights)
        points[name] = bootstrap_metrics(d.y_true,d.calibrated_prob,tau,reference,np.ones((1,n)))
        summaries.append({'seed':seed,'Model':name,'Threshold':tau,
                          'Constant_Brier':float(np.mean((d.y_true-reference)**2)),
                          **{k:v[0] for k,v in points[name].items()}})
        for key in KEYS:
            values = draws[name][key]; valid = values[np.isfinite(values)]
            lo,hi = np.quantile(valid,[.025,.975]) if len(valid) else [np.nan,np.nan]
            rows.append({'seed':seed,'Model':name,'Metric':key,'Estimate':points[name][key][0],
                         'CI_Low':lo,'CI_High':hi,'Valid_Bootstraps':len(valid),'Bootstrap_Unit':unit})
    pd.DataFrame(summaries).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'metric_CI.csv',index=False)
    paired=[]
    for other in PROTOCOL['paired_comparators']:
        for key in KEYS:
            v=draws['Causal_BRB'][key]-draws[other][key]
            v=v[np.isfinite(v)]
            lo,hi=np.quantile(v,[.025,.975]) if len(v) else [np.nan,np.nan]
            paired.append({'seed':seed,'Comparator':other,'Metric':key,
                           'Difference':points['Causal_BRB'][key][0]-points[other][key][0],
                           'CI_Low':lo,'CI_High':hi,'Valid_Bootstraps':len(v),'Bootstrap_Unit':unit})
    pd.DataFrame(paired).to_csv(out/'paired_differences.csv',index=False)
    core.write_json(out/'inference_protocol.json',{'bootstrap_unit':unit,'fit_prevalence':reference,
                    'paired_draws':True,'scope':'conditional on fitted model; splits are overlapping, not independent replications'})
