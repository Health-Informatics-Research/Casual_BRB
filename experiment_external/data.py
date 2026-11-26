"""Deterministic adapters. No learned cleaning and no outcome-proxy renaming."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from experiment_v3 import core

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads(Path(__file__).with_name('protocol.json').read_text(encoding='utf-8'))
LABEL = 'depression_label'
META = set(core.META + ['family_id'])

# Native names, explicit valid domains, and PDF page numbers in 04652-0001-Codebook-ddi.pdf.
# No chronic-condition aggregate: the aggregate includes the anxiety/depression item.
MIDUS_FIELDS = {
    'B1PAGE_M2': ('P1', 'Age at phone interview', 28, 84, [98], 8),
    'B1PRSEX': ('P1', 'Sex', 1, 2, [], 11),
    'B1PB19': ('P1', 'Current marital status', 1, 5, [7, 8, 9], 212),
    'B1STINC1': ('P1', 'Household total income in dollars', 0, 300000, [-1, 9999998], 1001),
    'B1PB1': ('P1', 'Education level', 1, 12, [97, 98, 99], 174),
    'B1SA57A': ('P2', 'Hours of sleep on workdays (hours component)', 0, 12, [-1, 98], 602),
    'B1SA1': ('P3', 'Current health rating, higher is better', 0, 10, [-1, 98], 434),
    'B1SBMI': ('P3', 'BMI', 0, 97.999, [-1, 98], 560),
    'B1SG7': ('P4', 'Difficulty paying monthly bills, higher is less difficulty', 1, 4, [-1, 8], 975),
    'B1SG1': ('P4', 'Financial situation, higher is better', 0, 10, [-1, 98], 972),
    'B1SA11D': ('P5', 'Arthritis/rheumatism/bone or joint disease, past 12 months', 1, 2, [-1], 456),
    'B1SA11S': ('P5', 'Hypertension, past 12 months', 1, 2, [-1], 463),
    'B1SA11X': ('P5', 'Diabetes/high blood sugar, past 12 months', 1, 2, [-1], 465),
    'B1SALCOH': ('P6', 'Alcohol problem, past 12 months', 0, 1, [-1, 8], 619),
    'B1PA39': ('P6', 'Currently regularly smokes; inapplicable is missing', 1, 2, [7, 8, 9], 118),
}


def clean_midus(frame):
    frame = frame.copy()
    frame.columns = frame.columns.str.upper()
    required = list(MIDUS_FIELDS) + ['M2ID', 'M2FAMNUM', 'B1PDEPDX']
    if set(required) - set(frame):
        raise ValueError(f'Missing MIDUS source columns: {sorted(set(required)-set(frame))}')
    x = frame[required].apply(pd.to_numeric, errors='coerce')
    valid_label = x.B1PDEPDX.isin([0, 1])
    valid_age = x.B1PAGE_M2.between(28, 84)
    saq = x.B1SA1.notna() & x.B1SA1.ne(-1)
    include = valid_label & valid_age & saq
    report = {'source_n': len(x), 'invalid_label_n': int((~valid_label).sum()),
              'invalid_age_n': int((~valid_age).sum()), 'no_saq_n': int((~saq).sum()),
              'excluded_union_n': int((~include).sum())}
    x = x.loc[include].copy()
    out = pd.DataFrame({'SEQN': x.M2ID, 'family_id': x.M2FAMNUM,
                        'CYCLE': 'MIDUS2', LABEL: x.B1PDEPDX.astype(int)})
    rows = []
    for col, (path, meaning, lo, hi, codes, page) in MIDUS_FIELDS.items():
        values = x[col]
        bad = values.isin(codes) | ~values.between(lo, hi)
        out[col] = values.mask(bad)
        rows.append({'field': col, 'path': path, 'meaning': meaning, 'codebook_pdf_page': page,
                     'missing_codes': json.dumps(codes), 'missing_after_cleaning': int(out[col].isna().sum()),
                     'invalidated_nonmissing': int((bad & values.notna()).sum())})
    if out[['SEQN', 'family_id']].isna().any().any() or out.SEQN.duplicated().any():
        raise ValueError('Missing or duplicated MIDUS identifiers')
    report.update(n=len(out), positives=int(out[LABEL].sum()), families=out.family_id.nunique())
    return out.reset_index(drop=True), pd.DataFrame(rows), report


def profile(dataset):
    if dataset == 'midus':
        paths = [[c for c, info in MIDUS_FIELDS.items() if info[0] == f'P{i}'] for i in range(1, 7)]
        reps = [paths[0][3:], paths[1], paths[2], paths[3], paths[4], paths[5]]
        tiers = {v: tier for variables, tier in zip(reps, [1, 2, 3, 2, 3, 2]) for v in variables}
        return dict(dataset=dataset, paths=paths, demos=paths[0][:3], representatives=reps, tiers=tiers,
                    path_names=['Socioeconomic', 'Sleep duration', 'General health', 'Financial strain',
                                'Physical conditions', 'Substance use'])
    definitions = core.legacy('nhanes_step2_feature_engineering').PATHWAY_VARIABLES
    paths = [list(v) for v in definitions.values()]
    paths[1] = ['SLD010H', 'SLQ050']  # native earlier-cycle questions; no equivalence claim
    causal = core.legacy('nhanes_step1_5_causal_discovery')
    reps = [list(v) for v in causal.PATHWAY_REPRESENTATIVES.values()]
    reps[1] = ['SLD010H', 'SLQ050']
    tiers = {v: tier for variables, tier in zip(reps, [1, 2, 3, 2, 3, 2]) for v in variables}
    tiers['DIQ010'] = 99  # preserve the published implementation; do not silently repair the architecture
    return dict(dataset=dataset, paths=paths, demos=core.DEMOS, representatives=reps, tiers=tiers,
                path_names=list(definitions))


def source_files(dataset):
    directory = (ROOT / PROTOCOL[dataset]['directory']).resolve()
    if dataset == 'midus':
        return [directory / '04652-0001-Data.tsv', directory / '04652-0001-Codebook-ddi.pdf',
                directory / '04652-0001-Documentation-psychosocial_constructs.pdf']
    return [directory / f'{module}_{cycle}.xpt' for cycle in ['G', 'H']
            for module in ['DEMO','DPQ','SLQ','HUQ','FSQ','MCQ','SMQ','ALQ','DUQ','PAQ','BMX','DIQ','BPQ','WHQ','OCQ']]


def load_data(dataset):
    files = source_files(dataset)
    for f in files:
        if not f.is_file():
            raise FileNotFoundError(f)
    if dataset == 'midus':
        return clean_midus(pd.read_csv(files[0], sep='\t', low_memory=False))
    mod = core.legacy('nhanes_step1_data_processing')
    mod.KEEP_VARS_BASE = list(mod.KEEP_VARS_BASE) + ['SLD010H']
    frames, audit = [], []
    for cycle, suffix in [('2011-2012','G'), ('2013-2014','H')]:
        selected = [p for p in files if p.stem.endswith('_'+suffix)]
        table = {p.stem.split('_')[0]: pd.read_sas(p, format='xport') for p in selected}
        raw = table['DEMO'].merge(table['DPQ'], on='SEQN', validate='one_to_one')
        for name, part in table.items():
            if name not in ['DEMO', 'DPQ']:
                columns = ['SEQN'] + [c for c in part if c not in raw]
                raw = raw.merge(part[columns], on='SEQN', how='left', validate='one_to_one')
        raw['CYCLE'] = cycle
        for c in raw:
            audit.append({'cycle': cycle, 'field': c, 'present': True})
        cleaned = mod.create_combined_weight(mod.clean_variables(mod.initial_filter(raw)))
        cleaned['SLD010H'] = cleaned.SLD010H.replace({77: np.nan, 99: np.nan})
        frames.append(cleaned)
    out = pd.concat(frames, ignore_index=True)
    # Retain only declared predictors and metadata; do not accidentally include PHQ items.
    p = profile(dataset)
    allowed = set(sum(p['paths'], [])) | set(core.DEMOS) | {'OCD150','OCD390G'} | set(core.META)
    out = out[[c for c in out if c in allowed]]
    if out.SEQN.duplicated().any():
        raise ValueError('Repeated NHANES participant IDs')
    return out, pd.DataFrame(audit), {'n': len(out), 'positives': int(out[LABEL].sum()),
                                     'cycles': out.groupby('CYCLE').size().to_dict()}
