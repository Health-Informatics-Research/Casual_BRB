"""Execute the frozen sequential tuning protocol."""
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from .core import metrics, optimize, specs, threshold, write_json

ROOT = Path(__file__).resolve().parent.parent
CONFIG = Path(__file__).with_name('tuning_frozen_config.json')
OUT = ROOT / 'runs' / 'exploratory_tuning_v1'


def calibrator(method, frame):
    x, y = frame.raw_prob.to_numpy(), frame.y_true.to_numpy()
    if method == 'isotonic':
        return IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(x, y).predict
    z = logit(np.clip(x, 1e-6, 1 - 1e-6))

    def loss(params):
        score = params[0] + params[1] * z
        error = expit(score) - y
        return np.mean(np.logaddexp(0, score) - y * score), [error.mean(), (error * z).mean()]

    fit = minimize(loss, [0., 1.], jac=True, bounds=[(None, None), (1e-8, None)], method='L-BFGS-B')
    if not fit.success:
        raise RuntimeError(fit.message)
    return lambda x: expit(fit.x[0] + fit.x[1] * logit(np.clip(x, 1e-6, 1 - 1e-6)))


def scores(cal, val, method):
    p = calibrator(method, cal)(val.raw_prob.to_numpy())
    return average_precision_score(val.y_true, p), brier_score_loss(val.y_true, p)


def select(rows, stage, reference, candidate, rules):
    frame = pd.DataFrame(rows, columns=['seed', 'model', 'candidate', 'AP', 'Brier'])
    frame.to_csv(OUT / f'{stage}_val.csv', index=False)
    means = frame.groupby(['candidate', 'model'])[['AP', 'Brier']].mean().groupby('candidate').mean()
    delta = means.loc[candidate] - means.loc[reference]
    choice = candidate if (delta.AP >= rules['minimum_mean_ap_gain']
                           and delta.Brier <= rules['maximum_mean_brier_increase']) else reference
    write_json(OUT / f'{stage}_decision.json', {'choice': choice, 'delta': delta.to_dict(),
                                                'means': means.to_dict(orient='index')})
    print(stage, choice, delta.to_dict(), flush=True)
    return choice


def main():
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    ref = ROOT / config['reference_run']
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = OUT / CONFIG.name
    if frozen.exists() and frozen.read_bytes() != CONFIG.read_bytes():
        raise RuntimeError('Frozen configuration differs')
    shutil.copyfile(CONFIG, frozen)
    write_json(OUT / 'provenance.json', {'config_sha256': hashlib.sha256(frozen.read_bytes()).hexdigest()})
    pairs = {}
    rows = []
    for seed in config['split_seeds']:
        for folder in sorted((ref / f'seed_{seed}' / 'models').iterdir()):
            if not folder.is_dir():
                continue
            cal, val = [pd.read_csv(folder / f'{part}_predictions.csv') for part in ('cal', 'val')]
            pairs[seed, folder.name] = (cal, val)
            for method in ('isotonic', 'positive_slope_platt'):
                rows.append((seed, folder.name, method, *scores(cal, val, method)))
    if len(pairs) != 105:
        raise RuntimeError('Expected 105 reference configurations')
    method = select(rows, 'calibration', 'isotonic', 'positive_slope_platt', config['selection'])
    rows = []
    candidates = {}
    for seed in config['split_seeds']:
        source = ref / f'seed_{seed}'
        prepared = joblib.load(source / 'prepared.joblib')
        target = OUT / f'seed_{seed}'
        for model, spec in specs(prepared['groups']).items():
            cal, val = pairs[seed, model]
            historical = source / 'models' / model
            folder = target / 'models' / model
            print(f'DE 240 seed={seed} model={model}', flush=True)
            if (folder / 'model.joblib').exists():
                bundle = joblib.load(folder / 'model.joblib')
            else:
                bundle = optimize(model, spec, prepared, target, 240, 28)
            old = pd.read_csv(historical / 'convergence.csv').loss.to_numpy()
            new = pd.read_csv(folder / 'convergence.csv').loss.to_numpy()
            if len(old) != 120 or len(new) < 120 or not np.array_equal(old, new[:120]):
                raise RuntimeError(f'DE trajectory mismatch: {seed}/{model}')
            candidates[seed, model] = folder
            rows.append((seed, model, 120, *scores(cal, val, method)))
            updated = []
            for part, frame in [('cal', cal), ('val', val)]:
                copy = frame.copy()
                copy['raw_prob'] = bundle['model'].predict(prepared['scores'][part], bundle['params'])
                updated.append(copy)
                copy.to_csv(folder / f'{part}_predictions.csv', index=False)
            rows.append((seed, model, 240, *scores(*updated, method)))
    budget = select(rows, 'de_budget', 120, 240, config['selection'])
    final = OUT / 'exploratory_test_metrics.csv'
    if final.exists():
        return
    results = []
    for (seed, model), (cal, val) in pairs.items():
        source = ref / f'seed_{seed}'
        test = pd.read_csv(source / 'models' / model / 'test_predictions.csv')
        candidate = candidates.get((seed, model)) if budget == 240 else None
        if candidate:
            cal, val = [pd.read_csv(candidate / f'{part}_predictions.csv') for part in ('cal', 'val')]
            bundle = joblib.load(candidate / 'model.joblib')
            pathways = pd.read_csv(source / 'test_pathways.csv')
            if not np.array_equal(pathways.SEQN, test.SEQN):
                raise RuntimeError('Test participant order mismatch')
            test['raw_prob'] = bundle['model'].predict(pathways.iloc[:, 2:].to_numpy(), bundle['params'])
        calibrate = calibrator(method, cal)
        tau = threshold(val.y_true.to_numpy(), calibrate(val.raw_prob.to_numpy()))
        test['calibrated_prob'] = calibrate(test.raw_prob.to_numpy())
        test['prediction'] = (test.calibrated_prob >= tau).astype(int)
        folder = OUT / f'seed_{seed}' / 'final_predictions'
        folder.mkdir(parents=True, exist_ok=True)
        test[['SEQN', 'y_true', 'raw_prob', 'calibrated_prob', 'prediction']].to_csv(folder / f'{model}.csv', index=False)
        results.append({'seed': seed, 'model': model, **metrics(test.y_true.to_numpy(), test.calibrated_prob.to_numpy(), tau)})
    frame = pd.DataFrame(results)
    frame.to_csv(final, index=False)
    frame.groupby('model')[['ROC_AUC', 'AP', 'Brier', 'F2']].agg(['mean', 'std']).to_csv(OUT / 'exploratory_test_summary.csv')
    print('Exploratory re-evaluation complete', flush=True)


if __name__ == '__main__':
    main()
