"""Focused regression checks for cohort cleaning, family isolation, and Cal/Val isolation."""
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import MIDUS_FIELDS, clean_midus, profile, ROOT, LABEL
from .engine import split_data, cv_indices, Processor, PositivePlatt, evaluate, PathwayPipeline, inference
from experiment_v3 import core


class ExternalChecks(unittest.TestCase):
    def test_native_fields_and_special_codes(self):
        rows={c:[max(lo,1)]*4 for c,(_,_,lo,hi,codes,page) in MIDUS_FIELDS.items()}
        rows.update(M2ID=[1,2,3,4],M2FAMNUM=[11,12,13,14],B1PDEPDX=[0,1,98,1],
                    B1PAGE_M2=[40]*4,B1SA1=[8,8,8,-1],B1STINC1=[9999998,100000,0,0],
                    B1SA57A=[98,8,8,8],B1SALCOH=[8,1,0,0],B1SG7=[8,3,1,1])
        raw,_,report=clean_midus(pd.DataFrame(rows))
        self.assertEqual(len(raw),2)
        self.assertEqual(report['excluded_union_n'],2)
        self.assertTrue(raw.loc[0,['B1STINC1','B1SA57A','B1SALCOH','B1SG7']].isna().all())
        self.assertEqual(raw.loc[1,'B1SA57A'],8)  # 8 hours is valid, not a blanket missing code
        self.assertNotIn('INDFMPIR',raw)
        self.assertNotIn('WTMEC4YR',raw)
        self.assertNotIn('B1SCHRON',sum(profile('midus')['paths'],[]))

    def test_family_split_and_nested_folds(self):
        raw=pd.DataFrame({'SEQN':range(500),'family_id':np.repeat(range(250),2),LABEL:np.tile([0,1],250)})
        for seed in [42,99,123,2026,888]:
            sets,assign=split_data(raw,seed)
            self.assertEqual(len(assign),len(raw))
            self.assertEqual(sum(map(len,sets.values())),500)
            for tr,va in cv_indices(sets['fit'],3):
                for ti,vi in cv_indices(sets['fit'].iloc[tr],5):
                    self.assertTrue(len(ti)>0 and len(vi)>0)

    def test_processor_only_fits_training_and_excludes_identifiers(self):
        fit=pd.DataFrame({'SEQN':range(20),'family_id':range(20),LABEL:[0,1]*10,
                          'x':np.arange(20.),'z':[np.nan]*12+list(range(8))})
        proc=Processor().fit(fit)
        self.assertEqual(proc.cols,['x'])
        before=proc.imputer.initial_imputer_.statistics_.copy()
        test=fit.copy();test['x']=1e8
        proc.transform(test)
        np.testing.assert_array_equal(before,proc.imputer.initial_imputer_.statistics_)

    def test_platt_monotonic_and_finite(self):
        p=np.array([0.,1e-300,.1,.3,.7,.9,1-1e-15,1.])
        model=PositivePlatt().fit(p,[0,0,0,1,0,1,1,1])
        out=model.predict(p)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue((np.diff(out)>=0).all())
        self.assertGreaterEqual(model.params[1],1e-8)

    def test_test_labels_cannot_choose_calibration_or_threshold(self):
        temp=ROOT/'runs'/'external_checks_tmp';temp.mkdir(parents=True,exist_ok=True)
        y=np.tile([0,1],20)
        sets={s:pd.DataFrame({'SEQN':np.arange(40)+i*100,LABEL:y}) for i,s in enumerate(['cal','val','test'])}
        p=np.linspace(.01,.99,40)
        with tempfile.TemporaryDirectory(dir=temp) as td:
            out=Path(td)
            def predict(s):
                if s=='test':
                    self.assertTrue((out/'models/A/threshold.json').exists())
                return p,None,None
            evaluate('A',predict,sets,out)
            a=joblib.load(out/'models/A/calibrator.joblib').params
            tau=(out/'models/A/threshold.json').read_bytes()
            sets['test'][LABEL]=1-sets['test'][LABEL]
            evaluate('B',lambda s:(p,None,None),sets,out)
            np.testing.assert_array_equal(a,joblib.load(out/'models/B/calibrator.joblib').params)
            self.assertEqual(tau,(out/'models/B/threshold.json').read_bytes())

    def test_paired_family_bootstrap_and_complete_model_scope(self):
        temp=ROOT/'runs'/'external_checks_tmp';temp.mkdir(parents=True,exist_ok=True)
        groups=[[0],[1,2],[3,4,5]]
        names=['V0_Mean']+list(core.specs(groups))+[f'{m}_{f}' for f in ['Pathway9','Raw'] for m in ['LR','RF','LightGBM']]
        self.assertEqual(len(names),21)
        frame=pd.DataFrame({'SEQN':range(40),'family_id':np.repeat(range(20),2),
                            'y_true':np.tile([0,1],20),'calibrated_prob':np.linspace(.01,.9,40)})
        with tempfile.TemporaryDirectory(dir=temp) as td:
            out=Path(td)
            joblib.dump({'groups':groups},out/'prepared.joblib')
            for name in names:
                d=out/'models'/name;d.mkdir(parents=True)
                frame.to_csv(d/'test_predictions.csv',index=False)
                core.write_json(d/'threshold.json',{'tau':.3})
            inference(out,42,.5)
            paired=pd.read_csv(out/'paired_differences.csv')
            self.assertTrue((paired.Bootstrap_Unit=='family cluster').all())
            np.testing.assert_allclose(paired[['Difference','CI_Low','CI_High']],0)


if __name__=='__main__':
    unittest.main()
