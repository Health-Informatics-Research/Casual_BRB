"""All plots derive from saved predictions or genuine optimizer callbacks."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from .core import metrics, write_json
from .core import CONFIG, PATHS

def read_csv(path):
    return pd.read_csv(path, float_precision='round_trip')

def _basic_generate(out):
    out=Path(out); dest=out/'reports'; dest.mkdir(exist_ok=True)
    table=read_csv(out/'metrics.csv')
    table.to_csv(dest/'model_comparison.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(15,12))
    for ax,col in zip(axes.ravel(),['ROC_AUC','AP','Brier','F2']):
        table.plot.barh(x='Model',y=col,ax=ax,legend=False); ax.set_title(col)
    fig.tight_layout(); fig.savefig(dest/'performance.png',dpi=180); plt.close(fig)
    random_rows=table[table.Model.str.startswith('V4_')]
    random_rows[['ROC_AUC','AP','Brier','F2']].agg(['mean','std']).to_csv(dest/'random_graph_summary.csv')
    fig,ax=plt.subplots(figsize=(9,5)); convergence=[]
    for p in (out/'models').glob('*/convergence.csv'):
        d=read_csv(p); d['Model']=p.parent.name; convergence.append(d)
        ax.plot(d.generation,d.loss,label=p.parent.name)
    if convergence: pd.concat(convergence).to_csv(dest/'convergence_source.csv',index=False)
    ax.set(xlabel='Generation',ylabel='Fit CV objective'); ax.legend(fontsize=7,ncol=2)
    fig.tight_layout(); fig.savefig(dest/'convergence.png',dpi=180); plt.close(fig)
    main=out/'models'/'Causal_BRB'
    test=read_csv(main/'test_predictions.csv'); val=read_csv(main/'val_predictions.csv')
    tau=json.loads((main/'threshold.json').read_text(encoding='utf-8'))['tau']
    q1,q2=val.m_Theta.quantile([1/3,2/3]).to_numpy()
    test['Uncertainty_Group']=np.where(test.m_Theta<=q1,'Low',np.where(test.m_Theta<=q2,'Medium','High'))
    groups=[]
    for group,d in test.groupby('Uncertainty_Group'):
        groups.append({'Group':group,'Positive_Rate':d.y_true.mean(),'Mean_Uncertainty':d.m_Theta.mean(),
            **metrics(d.y_true.to_numpy(),d.calibrated_prob.to_numpy(),tau)})
    grouped=pd.DataFrame(groups); grouped.to_csv(dest/'uncertainty_groups.csv',index=False)
    test.to_csv(dest/'belief_interval_samples.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,col in zip(axes,['Positive_Rate','Brier']): grouped.set_index('Group').reindex(['Low','Medium','High'])[col].plot.bar(ax=ax,title=col)
    fig.tight_layout(); fig.savefig(dest/'uncertainty.png',dpi=180); plt.close(fig)
    calibration=[]
    for name in table.Model:
        d=read_csv(out/'models'/name/'test_predictions.csv')
        bins=pd.cut(d.calibrated_prob,np.linspace(0,1,11),include_lowest=True)
        for interval,g in d.groupby(bins,observed=True):
            calibration.append({'Model':name,'Bin':str(interval),'N':len(g),'Mean_Predicted':g.calibrated_prob.mean(),'Observed':g.y_true.mean()})
    pd.DataFrame(calibration).to_csv(dest/'calibration_bins.csv',index=False)
    raw=read_csv(out/'raw_unimputed.csv'); d=test.merge(raw,on='SEQN',validate='one_to_one')
    masks={'Female':d.RIAGENDR==2,'Male':d.RIAGENDR==1,'Age_lt40':d.RIDAGEYR<40,
        'Age_40_59':d.RIDAGEYR.between(40,59),'Age_ge60':d.RIDAGEYR>=60,
        'White':d.RIDRETH3==3,'Black':d.RIDRETH3==4,'Hispanic':d.RIDRETH3.isin([1,2])}
    fairness=[{'Group':name,**metrics(d.loc[mask,'y_true'].to_numpy(),d.loc[mask,'calibrated_prob'].to_numpy(),tau)} for name,mask in masks.items() if mask.any()]
    pd.DataFrame(fairness).to_csv(dest/'fairness.csv',index=False)
    dca=[]; y=test.y_true.to_numpy(); p=test.calibrated_prob.to_numpy()
    for t in np.linspace(.01,.4,100):
        nb=np.mean((p>=t)&(y==1))-np.mean((p>=t)&(y==0))*t/(1-t)
        dca.append({'Threshold':t,'Causal_BRB':nb,'Treat_All':y.mean()-(1-y.mean())*t/(1-t),'Treat_None':0.})
    dca=pd.DataFrame(dca); dca.to_csv(dest/'dca.csv',index=False)
    dca.plot(x='Threshold',y=['Causal_BRB','Treat_All','Treat_None'],ylim=(-.03,max(dca.Causal_BRB)*1.2))
    plt.tight_layout(); plt.savefig(dest/'dca.png',dpi=180); plt.close()
    triage=[]
    for name,mask in [('All',np.ones(len(p),dtype=bool)),('Probability',p>=.15),
            ('Uncertainty_Gray',(p>=.15)|((p>=.05)&(p<.15)&(test.m_Theta.to_numpy()>q2)))]:
        triage.append({'Strategy':name,'Reviewed':int(mask.sum()),'Detected_Positives':int(y[mask].sum()),
            'Sensitivity':float(y[mask].sum()/max(y.sum(),1)), 'Cost':float(mask.sum()+(0 if name=='All' else .1*len(y)))})
    pd.DataFrame(triage).to_csv(dest/'triage.csv',index=False)
    weight_rows=[]
    valid=d.WTMEC4YR.notna()&(d.WTMEC4YR>0)
    for name,w in [('Unweighted',None),('Survey_Weighted',d.loc[valid,'WTMEC4YR'].to_numpy())]:
        yy=d.loc[valid,'y_true']; pp=d.loc[valid,'calibrated_prob']
        weight_rows.append({'Method':name,'ROC_AUC':roc_auc_score(yy,pp,sample_weight=w),
            'AP':average_precision_score(yy,pp,sample_weight=w),'Brier':brier_score_loss(yy,pp,sample_weight=w)})
    pd.DataFrame(weight_rows).to_csv(dest/'survey_weight_sensitivity.csv',index=False)
    intervention_report(out,dest)
    write_json(dest/'provenance.json',{'comparison':'../metrics.csv','predictions':'../models/*/test_predictions.csv',
        'convergence':'../models/*/convergence.csv','uncertainty_cutpoints_source':'Causal_BRB Val',
        'threshold_source':'Causal_BRB Val','intervals':'raw DS belief/plausibility, not calibrated confidence intervals',
        'causal_claim':'Input interventions are predictive simulations, not identified causal ATEs'})

def intervention_report(out,dest):
    prepared=joblib.load(out/'prepared.joblib'); bundle=joblib.load(out/'models/Causal_BRB/model.joblib')
    iso=joblib.load(out/'models/Causal_BRB/calibrator.joblib')
    x=prepared['scores']['test']; fit=prepared['scores']['fit']
    baseline=read_csv(out/'models/Causal_BRB/test_predictions.csv')
    rows=[]; samples=[]; rng=np.random.default_rng(42)
    for name,cols in [('Sleep',[1]),('Food',[3]),('Clinical',[4]),('Substance',[5]),('Combined',[1,3,4,5])]:
        cf=x.copy()
        for c in cols: cf[:,c]=np.minimum(cf[:,c],np.quantile(fit[:,c],.16))
        pred=iso.predict(bundle['model'].predict(cf,bundle['params']))
        delta=baseline.calibrated_prob.to_numpy()-pred
        boots=[delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(300)]
        lo,hi=np.quantile(boots,[.025,.975])
        rows.append({'Intervention':name,'Mean_Predicted_Risk_Reduction':delta.mean(),'CI_Low':lo,'CI_High':hi,
            'Relative_Reduction':delta.mean()/max(baseline.calibrated_prob.mean(),1e-12),'Interpretation':'predictive input intervention; not identified causal effect'})
        samples.append(pd.DataFrame({'SEQN':baseline.SEQN,'Intervention':name,'Factual':baseline.calibrated_prob,'Counterfactual':pred,'Reduction':delta}))
    result=pd.DataFrame(rows); result.to_csv(dest/'intervention_simulation.csv',index=False)
    pd.concat(samples).to_csv(dest/'intervention_samples.csv',index=False)
    plt.figure(figsize=(8,5)); plt.bar(result.Intervention,result.Mean_Predicted_Risk_Reduction)
    plt.errorbar(result.Intervention,result.Mean_Predicted_Risk_Reduction,
        yerr=np.array([result.Mean_Predicted_Risk_Reduction-result.CI_Low,result.CI_High-result.Mean_Predicted_Risk_Reduction]),fmt='none',color='black')
    plt.ylabel('Mean predicted risk reduction'); plt.tight_layout(); plt.savefig(dest/'interventions.png',dpi=180); plt.close()

def intervention_sensitivity(out,prepared):
    model=joblib.load(out/'models/Causal_BRB/model.joblib')
    iso=joblib.load(out/'models/Causal_BRB/calibrator.joblib')
    original=read_csv(out/'models/Causal_BRB/test_predictions.csv')
    fit=prepared['scores']['fit']; test=prepared['scores']['test']; rows=[]; samples=[]
    for quantile in CONFIG['intervention_quantiles']:
        rng=np.random.default_rng(42)
        for name,cols in [('Sleep',[1]),('Food',[3]),('Clinical',[4]),('Substance',[5]),('Combined',[1,3,4,5])]:
            cf=test.copy()
            for c in cols: cf[:,c]=np.minimum(cf[:,c],np.quantile(fit[:,c],quantile))
            p=iso.predict(model['model'].predict(cf,model['params']))
            delta=original.calibrated_prob.to_numpy()-p
            draws=[delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(300)]
            lo,hi=np.quantile(draws,[.025,.975])
            rows.append({'Target_Quantile':quantile,'Intervention':name,'Mean_Predicted_Risk_Reduction':delta.mean(),
                'CI_Low':lo,'CI_High':hi,'Relative_Reduction':delta.mean()/max(original.calibrated_prob.mean(),1e-12)})
            samples.append(pd.DataFrame({'SEQN':original.SEQN,'Target_Quantile':quantile,'Intervention':name,
                'Factual':original.calibrated_prob,'Counterfactual':p,'Reduction':delta}))
    frame=pd.DataFrame(rows)
    frame['Rank_Within_Target']=frame.groupby('Target_Quantile').Mean_Predicted_Risk_Reduction.rank(ascending=False,method='min')
    frame.to_csv(out/'statistics'/'intervention_sensitivity.csv',index=False)
    pd.concat(samples).to_csv(out/'statistics'/'intervention_sensitivity_samples.csv',index=False)

def generate_seed(out,prepared,sets,seed):
    _basic_generate(out)
    intervention_sensitivity(out,prepared)
    dca=read_csv(out/'statistics/dca_all.csv')
    fig,ax=plt.subplots(figsize=(9,5))
    for name,g in dca.groupby('Model'): ax.plot(g.Threshold,g.Net_Benefit,label=name)
    g=dca[dca.Model=='Causal_BRB']; ax.plot(g.Threshold,g.Treat_All,'--',label='Treat All'); ax.axhline(0,color='black',label='Treat None')
    ax.set(xlabel='Threshold probability',ylabel='Net benefit',ylim=(-.03,max(dca.Net_Benefit)*1.2)); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'reports/dca_with_baselines.png',dpi=180); plt.close(fig)
    ci=read_csv(out/'statistics/metric_CI.csv')
    ece=ci[ci.Metric=='ECE'][['Model','Estimate','CI_Low','CI_High']]
    ece.to_csv(out/'reports/ece.csv',index=False)
    bins=read_csv(out/'reports/calibration_bins.csv')
    fig,ax=plt.subplots(figsize=(7,6))
    for name in CONFIG['dca_models']:
        d=bins[bins.Model==name]; ax.plot(d.Mean_Predicted,d.Observed,marker='o',label=name)
    ax.plot([0,1],[0,1],'--',color='gray'); ax.set(xlabel='Mean calibrated probability',ylabel='Observed positive rate'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'reports/calibration_with_baselines.png',dpi=180); plt.close(fig)

def aggregate(out,include_stability=True):
    dest=out/'summary'; dest.mkdir(exist_ok=True)
    cis=[]; paired=[]; glm=[]; triage=[]; groups=[]; interventions=[]; selected=[]; structures=[]; counts=[]
    for seed in CONFIG['split_seeds']:
        folder=out/f'seed_{seed}'
        for target,file in [(cis,'statistics/metric_CI.csv'),(paired,'statistics/paired_differences.csv'),
            (glm,'statistics/uncertainty_incremental_GLM.csv'),(groups,'statistics/uncertainty_group_CI.csv'),
            (interventions,'statistics/intervention_sensitivity.csv'),(selected,'ml_selected_parameters.csv')]:
            target.append(read_csv(folder/file).assign(Split_Seed=seed))
        triage.append({'Split_Seed':seed,**json.loads((folder/'statistics/triage_random_comparison.json').read_text(encoding='utf-8'))})
        structures.append({'Split_Seed':seed,**json.loads((folder/'hierarchy.json').read_text(encoding='utf-8'))})
        for model_path in (folder/'models').glob('*/optimization.json'):
            d=json.loads(model_path.read_text(encoding='utf-8')); counts.append({'Split_Seed':seed,'Model':model_path.parent.name,**d})
    allci=pd.concat(cis); allci.to_csv(dest/'per_split_metric_CI.csv',index=False)
    summary=allci.groupby(['Model','Metric'],as_index=False).agg(Mean=('Estimate','mean'),SD=('Estimate','std'),Splits=('Split_Seed','nunique'))
    summary.to_csv(dest/'metrics_mean_std.csv',index=False)
    pd.concat(paired).to_csv(dest/'paired_differences_by_split.csv',index=False)
    pd.concat(glm).to_csv(dest/'uncertainty_incremental_by_split.csv',index=False)
    pd.DataFrame(triage).to_csv(dest/'triage_random_by_split.csv',index=False)
    pd.concat(groups).to_csv(dest/'uncertainty_groups_by_split.csv',index=False)
    pd.concat(interventions).to_csv(dest/'intervention_sensitivity_by_split.csv',index=False)
    pd.concat(selected).to_csv(dest/'Table2_selected_ML_parameters.csv',index=False)
    pd.DataFrame(counts).to_csv(dest/'actual_model_sizes_and_optimization.csv',index=False)
    write_json(dest/'structures_by_split.json',structures)
    from collections import Counter
    ec=Counter(); gc=Counter(); allnodes=set()
    for s in structures:
        ec.update(tuple(e) for e in s['edges'])
        allnodes.update(n for edge in s['edges'] for n in edge)
        for p in PATHS: gc[p,next((k for k,v in s['mapping'].items() if p in v),'Excluded')]+=1
    pd.DataFrame([{'Source':a,'Target':b,'Splits_Present':ec[a,b],'Frequency':ec[a,b]/5} for a in sorted(allnodes) for b in sorted(allnodes) if a!=b]).to_csv(dest/'edge_frequency_across_splits.csv',index=False)
    pd.DataFrame([{'Pathway':p,'Group':g,'Splits_Present':gc[p,g],'Frequency':gc[p,g]/5} for p in PATHS for g in ['Sub1','Sub2','Sub3','Excluded']]).to_csv(dest/'group_frequency_across_splits.csv',index=False)
    # Bootstrap structural frequencies are reported per split, not pooled as 500 independent datasets.
    if include_stability:
        for filename in ['edge_frequencies.csv','group_frequencies.csv']:
            pd.concat([read_csv(out/f'seed_{s}/stability'/filename).assign(Split_Seed=s) for s in CONFIG['causal_bootstrap_split_seeds']]).to_csv(dest/f'bootstrap_{filename}',index=False)
    fig,axes=plt.subplots(2,2,figsize=(15,12))
    for ax,key in zip(axes.ravel(),['ROC_AUC','AP','Brier','F2']):
        d=summary[summary.Metric==key]; ax.barh(d.Model,d.Mean,xerr=d.SD.fillna(0)); ax.set_title(key+' (mean +/- split SD)')
    fig.tight_layout(); fig.savefig(dest/'repeated_split_performance.png',dpi=180); plt.close(fig)
    lines=['# protocol_v3 五划分冻结协议结果','',
        '状态：全部分析完成。' if include_stability else '状态：五划分主比较与统计已完成；稳健性重跑及结构bootstrap尚未全部完成。',
        '本轮配置在运行前冻结；v2测试结果此前已被查看，属于冻结协议下重复划分再分析，不是全新盲测。',
        '全部模型不加类别或样本权重。路径算法沿用历史预设，未依据本轮Val/Test选择。',
        '主比较运行五个划分；PHQ和学习曲线保留原分析方法，在代表性种子42完整重跑，未扩大为五种子稳健性实验。',
        '外部验证未运行，论文正文未编辑。','',
        '| Model | AUC mean±SD | AP mean±SD | Brier mean±SD | F2 mean±SD |','|---|---|---|---|---|']
    for name in summary.Model.unique():
        vals=[]
        for key in ['ROC_AUC','AP','Brier','F2']:
            r=summary[(summary.Model==name)&(summary.Metric==key)].iloc[0]; vals.append(f'{r.Mean:.4f} ± {r.SD:.4f}')
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    lines += ['', '各划分1000次bootstrap CI与配对差值：summary/per_split_metric_CI.csv、paired_differences_by_split.csv。',
        '不确定性增量GLM：uncertainty_incremental_by_split.csv；同概率分层结果见各seed/statistics目录。',
        '分诊新增复查阳性率与同人数随机复查：triage_random_by_split.csv。',
        '结构稳定性：五个完整Fit的结构频率，另在代表性种子42的Fit进行100次bootstrap，每次重拟合插补器并重新学习结构；按用户确认的范围执行。',
        '主层锚点仍为[0,0.5,1]，保留v2实现；规则数随Fit发现分组而变化，实际规模逐模型记录。',
        'DE达到上限时如实记录，不写成已收敛；无240代对照。',
        'Brier技能分相对Fit阳性率常数预测；ECE为10个等宽箱。',
        '重复划分测试参与者重叠，不能将五个CI当作独立队列作显著性合并。',
        'GLM和分诊检验为探索性关联/增量分析，未作多重比较校正。',
        '输入干预是固定模型输入模拟，Bootstrap不包含模型重训不确定性，不称ATE或因果效应。']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
