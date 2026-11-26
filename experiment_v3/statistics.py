"""Inference on frozen predictions; no Test-driven model or threshold selection."""
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from .core import CONFIG, write_json

KEYS=['ROC_AUC','AP','Brier','F2','Sensitivity','Specificity','PPV','ECE','Brier_Skill']

def read(path): return pd.read_csv(path,float_precision='round_trip')

def bootstrap_metrics(y,p,tau,reference,weights):
    """Exact tied-score AUC/AP with participant multiplicities; vectorized draws."""
    y=np.asarray(y,dtype=float); p=np.asarray(p); pred=p>=tau
    w=np.asarray(weights,dtype=float); total=w.sum(axis=1); pos=w@y; neg=total-pos
    tp=w@(y*pred); fp=w@((1-y)*pred); fn=pos-tp; tn=neg-fp
    def div(a,b): return np.divide(a,b,out=np.full_like(a,np.nan,dtype=float),where=b!=0)
    order=np.argsort(p,kind='stable'); starts=np.r_[0,np.flatnonzero(np.diff(p[order]))+1]
    pp=np.add.reduceat(w[:,order]*y[order],starts,axis=1)
    nn=np.add.reduceat(w[:,order]*(1-y[order]),starts,axis=1)
    auc=div(np.sum(pp*(np.cumsum(nn,axis=1)-.5*nn),axis=1),pos*neg)
    pp=pp[:,::-1]; nn=nn[:,::-1]
    precision=div(np.cumsum(pp,axis=1),np.cumsum(pp+nn,axis=1))
    ap=div(np.nansum(pp*precision,axis=1),pos)
    br=div(w@((y-p)**2),total); base=div(w@((y-reference)**2),total)
    bins=np.minimum((p*10).astype(int),9); ece=np.zeros(len(w))
    for b in range(10):
        mask=bins==b
        if mask.any(): ece+=np.abs(w[:,mask]@(y[mask]-p[mask]))/total
    return {'ROC_AUC':auc,'AP':ap,'Brier':br,'F2':div(5*tp,5*tp+4*fn+fp),
        'Sensitivity':div(tp,pos),'Specificity':div(tn,neg),'PPV':div(tp,tp+fp),
        'ECE':ece,'Brier_Skill':1-div(br,base)}

def metric_inference(out,seed):
    dest=out/'statistics'; dest.mkdir(exist_ok=True)
    if (dest/'metric_CI.csv').exists() and (dest/'paired_differences.csv').exists(): return
    splits=read(out/'splits.csv'); raw=read(out/'raw_unimputed.csv')
    fit_ids=set(splits.loc[splits.split=='fit','SEQN'])
    reference=raw.loc[raw.SEQN.isin(fit_ids),'depression_label'].mean()
    names=read(out/'metrics.csv').Model.tolist()
    data={name:read(out/'models'/name/'test_predictions.csv').sort_values('SEQN').reset_index(drop=True) for name in names}
    first=data[names[0]]; n=len(first)
    rng=np.random.default_rng(CONFIG['bootstrap_seed']+seed)
    weights=rng.multinomial(n,np.full(n,1/n),size=CONFIG['test_bootstraps'])
    draws={}; points={}; rows=[]
    for name,d in data.items():
        np.testing.assert_array_equal(first.SEQN,d.SEQN); np.testing.assert_array_equal(first.y_true,d.y_true)
        tau=json.loads((out/'models'/name/'threshold.json').read_text(encoding='utf-8'))['tau']
        vals=bootstrap_metrics(d.y_true,d.calibrated_prob,tau,reference,weights)
        point=bootstrap_metrics(d.y_true,d.calibrated_prob,tau,reference,np.ones((1,n)))
        draws[name]=vals; points[name]=point
        for key in KEYS:
            valid=vals[key][np.isfinite(vals[key])]
            lo,hi=np.quantile(valid,[.025,.975]) if len(valid) else (np.nan,np.nan)
            rows.append({'Split_Seed':seed,'Model':name,'Metric':key,'Estimate':point[key][0],
                'CI_Low':lo,'CI_High':hi,'Valid_Bootstraps':len(valid),'Fit_Constant':reference})
    pd.DataFrame(rows).to_csv(dest/'metric_CI.csv',index=False)
    paired=[]
    for comparator in CONFIG['paired_comparators']:
        if comparator not in draws: continue
        for key in KEYS:
            delta=draws['Causal_BRB'][key]-draws[comparator][key]
            valid=delta[np.isfinite(delta)]
            lo,hi=np.quantile(valid,[.025,.975]) if len(valid) else (np.nan,np.nan)
            paired.append({'Split_Seed':seed,'Main':'Causal_BRB','Comparator':comparator,'Metric':key,
                'Difference_Main_Minus_Comparator':points['Causal_BRB'][key][0]-points[comparator][key][0],
                'CI_Low':lo,'CI_High':hi,'Valid_Bootstraps':len(valid),
                'Favorable_Direction':'negative' if key in ['Brier','ECE'] else 'positive'})
    pd.DataFrame(paired).to_csv(dest/'paired_differences.csv',index=False)
    write_json(dest/'bootstrap_protocol.json',{'resamples':len(weights),'sampling':'ordinary participant bootstrap',
        'paired_indices':True,'fit_constant':reference,'thresholds':'frozen Val threshold per model',
        'scope':'conditional on trained models; no retraining; overlapping repeated splits are not independent cohorts'})

def uncertainty_analysis(out,seed):
    import statsmodels.api as sm
    dest=out/'statistics'; dest.mkdir(exist_ok=True)
    test=read(out/'models/Causal_BRB/test_predictions.csv')
    val=read(out/'models/Causal_BRB/val_predictions.csv')
    q1,q2=val.m_Theta.quantile([1/3,2/3]).to_numpy()
    groups=np.where(test.m_Theta<=q1,'Low',np.where(test.m_Theta<=q2,'Medium','High'))
    rng=np.random.default_rng(CONFIG['bootstrap_seed']+seed+1)
    w=rng.multinomial(len(test),np.full(len(test),1/len(test)),size=1000)
    rows=[]; rates={}
    for g in ['Low','Medium','High']:
        mask=groups==g; den=w@mask.astype(float); num=w@(mask*test.y_true.to_numpy())
        rate=np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=den>0); rates[g]=rate
        good=rate[np.isfinite(rate)]; lo,hi=np.quantile(good,[.025,.975]) if len(good) else (np.nan,np.nan)
        rows.append({'Group':g,'N':int(mask.sum()),'Positives':int(test.loc[mask,'y_true'].sum()),
            'Positive_Rate':test.loc[mask,'y_true'].mean(),'CI_Low':lo,'CI_High':hi,'Valid_Bootstraps':len(good)})
    pd.DataFrame(rows).to_csv(dest/'uncertainty_group_CI.csv',index=False)
    ratio=np.divide(rates['High'],rates['Low'],out=np.full(1000,np.nan),where=rates['Low']>0)
    good=ratio[np.isfinite(ratio)]; low=rows[0]['Positive_Rate']; high=rows[2]['Positive_Rate']
    write_json(dest/'uncertainty_rate_ratio.json',{'high_vs_low':high/low if low>0 else None,
        'CI_Low':float(np.quantile(good,.025)) if len(good) else None,
        'CI_High':float(np.quantile(good,.975)) if len(good) else None,'valid_bootstraps':len(good)})
    regressions=[]
    for prob_col in ['calibrated_prob','raw_prob']:
        x=sm.add_constant(test[[prob_col,'m_Theta']],has_constant='add')
        try:
            with warnings.catch_warnings(record=True) as caught:
                result=sm.GLM(test.y_true,x,family=sm.families.Binomial()).fit(cov_type='HC3')
            ci=result.conf_int()
            for term in x.columns:
                regressions.append({'Probability':prob_col,'Term':term,'Coefficient':result.params[term],
                    'CI_Low':ci.loc[term,0],'CI_High':ci.loc[term,1],'P_Value':result.pvalues[term],
                    'OR_per_0_1':np.exp(.1*result.params[term]),'Covariance':'HC3',
                    'Scope':'exploratory Test association, not a new deployed predictor',
                    'Warnings':';'.join(str(c.message) for c in caught)})
        except Exception as exc:
            regressions.append({'Probability':prob_col,'Term':'fit_failed','Warnings':str(exc)})
    pd.DataFrame(regressions).to_csv(dest/'uncertainty_incremental_GLM.csv',index=False)
    strata=[]
    for b in range(10):
        lower,upper=b/10,(b+1)/10
        mask=(test.calibrated_prob>=lower)&((test.calibrated_prob<upper) if b<9 else (test.calibrated_prob<=upper))
        vm=(val.calibrated_prob>=lower)&((val.calibrated_prob<upper) if b<9 else (val.calibrated_prob<=upper))
        cut=float(val.loc[vm,'m_Theta'].median()) if vm.any() else float(val.m_Theta.median())
        for name,sel in [('Lower_unc',mask&(test.m_Theta<=cut)),('Higher_unc',mask&(test.m_Theta>cut))]:
            n=int(sel.sum()); pos=int(test.loc[sel,'y_true'].sum())
            strata.append({'Probability_Low':lower,'Probability_High':upper,'Uncertainty':name,'Val_Cut':cut,
                'N':n,'Positives':pos,'Positive_Rate':pos/n if n else np.nan})
    pd.DataFrame(strata).to_csv(dest/'probability_matched_uncertainty.csv',index=False)
    p=test.calibrated_prob.to_numpy(); y=test.y_true.to_numpy()
    gray=(p>=.05)&(p<.15); extra=gray&(test.m_Theta.to_numpy()>q2)
    pool=np.flatnonzero(gray); k=int(extra.sum()); caught=int(y[extra].sum())
    rng=np.random.default_rng(CONFIG['triage']['seed']+seed)
    draws=np.array([y[rng.choice(pool,size=k,replace=False)].sum() for _ in range(1000)]) if k else np.zeros(1000)
    pd.DataFrame({'Draw':np.arange(1000),'Extra_Positives':draws}).to_csv(dest/'triage_random_draws.csv',index=False)
    write_json(dest/'triage_random_comparison.json',{'gray_pool':len(pool),'extra_reviews':k,'extra_positives':caught,
        'extra_positive_rate':caught/k if k else None,'random_mean_positives':float(draws.mean()),
        'random_mean_positive_rate':float(draws.mean()/k) if k else None,
        'random_positives_95_interval':np.quantile(draws,[.025,.975]).tolist(),
        'empirical_randomization_tail_probability':float((1+np.sum(draws>=caught))/(len(draws)+1)),
        'interpretation':'exploratory; same number sampled uniformly without replacement from same gray area'})

def dca(out):
    rows=[]
    for name in CONFIG['dca_models']:
        d=read(out/'models'/name/'test_predictions.csv'); p=d.calibrated_prob.to_numpy(); y=d.y_true.to_numpy()
        for t in np.linspace(.01,.4,100):
            rows.append({'Model':name,'Threshold':t,'Net_Benefit':np.mean((p>=t)&(y==1))-np.mean((p>=t)&(y==0))*t/(1-t),
                'Treat_All':y.mean()-(1-y.mean())*t/(1-t),'Treat_None':0.})
    pd.DataFrame(rows).to_csv(out/'statistics'/'dca_all.csv',index=False)

def analyze(out,seed):
    metric_inference(out,seed)
    uncertainty_analysis(out,seed)
    dca(out)
