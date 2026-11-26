"""Development-only integration check. No Test evaluation; not a scientific experiment."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from experiment_v3 import core
from .data import ROOT, LABEL, profile, load_data
from .engine import split_data, PathwayPipeline, discover, PositivePlatt


def main():
    out=ROOT/'runs/external_integration_check'
    out.mkdir(exist_ok=True)
    checks=[]
    for dataset in ['nhanes','midus']:
        raw,_,_=load_data(dataset)
        sets,_=split_data(raw,42)
        pipe=PathwayPipeline(profile(dataset)).fit(sets['fit'],sets['fit'][LABEL])
        scores=pipe.predict(sets['cal'])
        assert scores.shape==(len(sets['cal']),9) and np.isfinite(scores).all()
        path=out/f'{dataset}_processor.joblib'
        joblib.dump(pipe,path)
        np.testing.assert_array_equal(scores,joblib.load(path).predict(sets['cal']))
        groups,hierarchy=discover(pipe.processor.transform(sets['fit']),sets['fit'][LABEL],profile(dataset))
        core.write_json(out/f'{dataset}_structure.json',hierarchy)
        # Training-only exercise of the unchanged BRB implementation and DE callback interface.
        # Synthetic inputs avoid treating in-sample pathway scores as official OOF data.
        rng=np.random.default_rng(42)
        x=rng.random((120,9));y=np.tile([0,1],60)
        prepared={'scores':{'fit':x},'folds':[(x[:80],x[80:],y[80:])]*3}
        spec=core.specs(groups)['Causal_BRB']
        target=out/dataset
        bundle=core.optimize('SMOKE_ONLY',spec,prepared,target,maxiter=1,population=5)
        prediction,belief,mass=bundle['model'].predict(scores,bundle['params'],details=True)
        assert np.isfinite(prediction).all()
        assert np.allclose(prediction,belief+.5*mass)
        cal=PositivePlatt().fit(prediction,sets['cal'][LABEL])
        assert np.isfinite(cal.predict(prediction)).all()
        checks.append({'dataset':dataset,'raw_features':len(pipe.processor.cols),'indicators':len(pipe.processor.indicators),
                       'pathway_columns':[c for c,_,_ in pipe.models], 'groups':groups,
                       'rules':bundle['model'].nr,'parameters':bundle['model'].n_params,
                       'serialization_exact':True,'test_accessed':False,'DE_generations':1})
    core.write_json(out/'checks.json',{'purpose':'integration only, NOT final results','checks':checks})
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
