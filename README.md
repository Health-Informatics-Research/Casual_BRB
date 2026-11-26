# Causal-BRB: Structure-Guided Belief Rule Base for Depression Classification and Input Scenario Simulation

Official reproduction code for the manuscript:
**"Causal-BRB for Structure-Guided Depression Classification and Input Scenario Simulation"**

## Overview

`Causal-BRB` is an inspectable hierarchical belief rule base (BRB) framework combining:
1. **Structure-guided hierarchical grouping**: PC causal discovery under explicit temporal tiers with mutual-information-weighted role assignment, compressing a matched-input flat BRB by 97.3%–99.0% (180–198 vs. 6,561–19,683 rules).
2. **Reliability-discounted evidential fusion & monotone rule-table repair**: Explicit main-layer residual mass reporting (`m_Theta`) with within-layer reliability monotonicity and idempotent axis-wise cumulative-maximum repair (zero adjacent-rule violations).
3. **Frozen-model input scenario simulation**: Assumption-conditional single-path and joint risk-reduction evaluation across matched models (Causal-BRB, V0, and 9-input Logistic Regression).
4. **Multi-cohort evaluation**: Five stratified splits on NHANES 2015–2018 (main cohort, *N* = 10,202), plus independent within-cohort retraining on NHANES 2011–2014 (*N* = 10,297) and MIDUS-2 (*N* = 4,041 across 3,032 families).

## Repository Structure

```text
.
├── experiment_v3/                      # Main NHANES 2015-2018 five-split pipeline
│   ├── core.py                         # BRB inference, monotone repair, DE optimization, baselines
│   ├── run.py                          # Five-split main benchmark runner (21 configurations)
│   ├── tuning.py                       # Positive-slope Platt vs. isotonic & 120 vs. 240 DE budget protocol
│   ├── statistics.py                   # Bootstrap CIs, paired differences, residual-mass & DCA analysis
│   ├── extensions.py                   # PHQ-9 cutoff sensitivity & survey-weighted robustness
│   ├── stability.py                    # 100x Fit bootstrap structural stability analysis
│   ├── fast_gsq.py                     # Cached G-square conditional independence helper
│   ├── report.py                       # Automated summary table & figure generation
│   ├── checks.py                       # Self-contained numerical & leakage unit checks
│   ├── config.json                     # Frozen main experiment configuration
│   └── tuning_frozen_config.json       # Frozen calibration & DE budget selection configuration
├── experiment_external/                # Cross-cohort independent retraining (NHANES 2011-2014 & MIDUS-2)
│   ├── data.py                         # Cohort loaders (NHANES XPT & MIDUS-2 ICPSR)
│   ├── engine.py                       # Within-cohort pathway OOF, PC grouping, DE & Platt calibration
│   ├── run.py                          # Audit, train, and strict transport CLI entry point
│   ├── checks.py                       # External pipeline unit checks
│   ├── smoke.py                        # Lightweight integration check
│   └── protocol.json                   # Frozen external applicability protocol
├── nhanes_step1_data_processing.py     # NHANES 2015-2018 XPT loader & eligibility preprocessing
├── nhanes_step1_5_causal_discovery.py  # PC algorithm with temporal forbidden-edge matrix & role mapping
├── nhanes_step2_feature_engineering.py # Pathway variable definitions
├── requirements.txt                    # Python package dependencies
└── .gitignore
```

## Installation

Python 3.10–3.12 is recommended.

```bash
pip install -r requirements.txt
```

## Data Availability & Directory Layout

Raw survey datasets are **not** redistributed in this repository to respect archive data-use agreements (especially ICPSR for MIDUS-2). Please download the official files from their original sources:

1. **NHANES 2015–2018** (2015–2016 and 2017–2018 cycles):
   - Source: U.S. CDC — https://wwwn.cdc.gov/nchs/nhanes/
   - Place the `.XPT` files under `data/nhanes_2015_2018/` (or set the environment variable `NHANES_2015_2018_DIR`).
2. **NHANES 2011–2014** (2011–2012 and 2013–2014 cycles):
   - Source: U.S. CDC — https://wwwn.cdc.gov/nchs/nhanes/
   - Place the `.XPT` files under `data/nhanes_2011_2014/`.
3. **MIDUS-2** (ICPSR 04652, v8):
   - Source: ICPSR — https://doi.org/10.3886/ICPSR04652.v8 (subject to ICPSR data-use agreement)
   - Place the DS0001 data files under `data/midus2/ICPSR_04652/DS0001/`.

## Quick Verification (No External Data Required)

Run the built-in unit tests to verify evidential fusion equivalence, bootstrap metrics, leakage guards, and external pipeline checks:

```bash
python -m experiment_v3.checks
python -m experiment_external.checks
```

## Reproducing the Experiments

### 1. Main Cohort (NHANES 2015–2018)

```bash
# Run the five-split baseline protocol (seeds: 42, 99, 123, 2026, 888)
python -u -m experiment_v3.run

# Run the calibration (Platt vs. isotonic) and DE budget (120 vs. 240) protocol
python -u -m experiment_v3.tuning
```

### 2. Cross-Cohort Independent Retraining (NHANES 2011–2014 & MIDUS-2)

```bash
# Audit cohort construction and split summaries
python -u -m experiment_external.run --stage audit --dataset nhanes
python -u -m experiment_external.run --stage audit --dataset midus

# Run independent five-split retraining (240 DE generations, positive-slope Platt scaling)
python -u -m experiment_external.run --stage train --dataset nhanes
python -u -m experiment_external.run --stage train --dataset midus
```

Outputs are written under `runs/` (ignored by Git).
