# =============================================================================
# nhanes_step1_data_processing.py  【修复版】
# 修复点：
# 1. 移除插补前的 StandardScaler，直接在原始尺度上插补
# 2. 插补后对所有物理意义上非负的变量执行 .clip(lower=0)
# 3. 对缺失率在 20%~40% 的变量生成额外的缺失指示变量 (_missing)
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 全局配置
# =============================================================================
DATA_DIR = r"D:\AAA研究生学习\Causal-BRB\跨数据集验证\date"
OUTPUT_CLEAN = "nhanes_clean_original.csv"
PHQ9_CUTOFF = 10
RANDOM_SEED = 42
MISSING_THRESHOLD = 0.40          # 缺失率超过此值则剔除原变量
MISSING_INDICATOR_THRESHOLD = 0.20 # 缺失率在此阈值之上的生成指示列
MIN_AGE = 18                      # 分析人群最小年龄

# 需要最终保留的变量（原始分析变量 + 元数据 + 权重）
KEEP_VARS_BASE = [
    "SEQN", "CYCLE",
    "WTINT2YR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA",
    "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "DMDMARTL",
    "INDFMPIR", "INDFMIN2", "INDHHIN2",
    "SLD012", "SLD013", "SLQ050", "SLQ120", "SLQ040",
    "HUQ010", "HUQ090", "HUQ051",
    "BMXBMI", "BMXWAIST",
    "FSDAD", "FSDHH", "FSD032A", "FSD032B", "FSD032C", "FSQ165", "FSD151",
    "MCQ520", "DIQ010", "BPQ020", "MCQ160A", "MCQ080",
    "SMQ020", "ALQ111", "ALQ151", "ALQ130", "ALQ121", "DUQ200", "DUQ240",
    "OCD270", "OCD150", "OCD390G",
    "PHQ9_score", "depression_label"
]

DPQ_COLS = [f"DPQ0{i}0" for i in range(1, 10)]

# =============================================================================
# 1-3. 数据加载与清洗 (保持原逻辑不变)
# =============================================================================
CYCLE_FILES = {
    "2011-2012": {
        "DEMO": "DEMO_G.XPT", "DPQ": "DPQ_G.XPT", "SLQ": "SLQ_G.XPT",
        "HUQ": "HUQ_G.XPT", "FSQ": "FSQ_G.XPT", "MCQ": "MCQ_G.XPT",
        "SMQ": "SMQ_G.XPT", "ALQ": "ALQ_G.XPT", "DUQ": "DUQ_G.XPT",
        "PAQ": "PAQ_G.XPT", "BMX": "BMX_G.XPT", "DIQ": "DIQ_G.XPT",
        "BPQ": "BPQ_G.XPT", "WHQ": "WHQ_G.XPT", "OCQ": "OCQ_G.XPT",
    },
    "2013-2014": {
        "DEMO": "DEMO_H.XPT", "DPQ": "DPQ_H.XPT", "SLQ": "SLQ_H.XPT",
        "HUQ": "HUQ_H.XPT", "FSQ": "FSQ_H.XPT", "MCQ": "MCQ_H.XPT",
        "SMQ": "SMQ_H.XPT", "ALQ": "ALQ_H.XPT", "DUQ": "DUQ_H.XPT",
        "PAQ": "PAQ_H.XPT", "BMX": "BMX_H.XPT", "DIQ": "DIQ_H.XPT",
        "BPQ": "BPQ_H.XPT", "WHQ": "WHQ_H.XPT", "OCQ": "OCQ_H.XPT",
    },
}

def read_xpt(data_dir, filename):
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path): return None
    try: return pd.read_sas(path, encoding="utf-8")
    except: return pd.read_sas(path, encoding="latin-1")

def load_one_cycle(data_dir, cycle, files):
    print(f"\n加载周期: {cycle}")
    df_demo = read_xpt(data_dir, files["DEMO"])
    df_dpq = read_xpt(data_dir, files["DPQ"])
    if df_demo is None or df_dpq is None: return None
    df = df_demo.merge(df_dpq, on="SEQN", how="inner")
    for mod, fname in files.items():
        if mod in ("DEMO", "DPQ"): continue
        tmp = read_xpt(data_dir, fname)
        if tmp is not None and "SEQN" in tmp.columns:
            col_overlap = [c for c in tmp.columns if c not in df.columns or c == "SEQN"]
            df = df.merge(tmp[col_overlap], on="SEQN", how="left")
    df["CYCLE"] = cycle
    return df

def load_all_cycles(data_dir):
    dfs = [load_one_cycle(data_dir, c, f) for c, f in CYCLE_FILES.items()]
    dfs = [d for d in dfs if d is not None]
    return pd.concat(dfs, ignore_index=True) if dfs else None

def initial_filter(df):
    df = df[df["RIDAGEYR"] >= MIN_AGE].copy()
    print(f"[过滤] 成年人: {len(df)} 条")
    for col in DPQ_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].where(~df[col].isin([7, 9]), np.nan)
    df["PHQ9_score"] = df[DPQ_COLS].sum(axis=1, min_count=9)
    df = df.dropna(subset=["PHQ9_score"]).copy()
    df["depression_label"] = (df["PHQ9_score"] >= PHQ9_CUTOFF).astype(int)
    print(f"[标签] 有效 PHQ-9 样本: {len(df)}，抑郁阳性率: {df['depression_label'].mean():.2%}")
    return df

def clean_variables(df):
    present_vars = [v for v in KEEP_VARS_BASE if v in df.columns]
    df = df[present_vars].copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mask = (df[col] != 0) & (df[col].abs() < 1e-50)
        if mask.any(): df.loc[mask, col] = 0.0

    missing_codes_map = {
        "DMDEDUC2": [7, 9], "DMDMARTL": [77, 99],
        "INDFMIN2": [77, 99], "INDHHIN2": [77, 99],
        "SLQ050": [7, 9], "SLQ120": [7, 9], "SLQ040": [7, 9],
        "HUQ010": [7, 9], "HUQ090": [7, 9], "HUQ051": [7, 9, 77, 99],
        "FSD032A": [7, 9], "FSD032B": [7, 9], "FSD032C": [7, 9],
        "FSQ165": [7, 9], "FSD151": [7, 9],
        "MCQ520": [7, 9], "MCQ160A": [7, 9], "MCQ080": [7, 9],
        "SMQ020": [7, 9], "ALQ111": [7, 9], "ALQ151": [7, 9],
        "ALQ130": [777, 999], "ALQ121": [777, 999],
        "DUQ200": [7, 9], "DUQ240": [7, 9],
        "OCD150": [7, 9], "OCD390G": [7, 9, 99],
        "OCD270": [77777, 99999],
        "DIQ010": [7, 9], "BPQ020": [7, 9],
    }
    for var, codes in missing_codes_map.items():
        if var in df.columns:
            df[var] = df[var].replace(codes, np.nan)

    income_map = {
        1: 2500, 2: 7500, 3: 12500, 4: 17500, 5: 22500,
        6: 30000, 7: 42500, 8: 55000, 9: 65000, 10: 75000,
        12: 87500, 13: 125000, 14: 37500, 15: 10000,
    }
    for income_var in ["INDFMIN2", "INDHHIN2"]:
        if income_var in df.columns:
            df[income_var] = df[income_var].map(income_map)

    return df

# =============================================================================
# 4. 缺失处理 (核心修复区)
# =============================================================================
def handle_missing(df):
    label_weight_cols = ["SEQN", "CYCLE", "depression_label", "PHQ9_score",
                         "WTINT2YR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"]
    feature_cols = [c for c in df.columns if c not in label_weight_cols]

    missing_rates = df[feature_cols].isnull().mean()
    
    # 修复 2：生成中等缺失率变量的 missing indicator 列
    indicator_cols = pd.DataFrame(index=df.index)
    generated_indicators = []
    for col in feature_cols:
        rate = missing_rates[col]
        if MISSING_INDICATOR_THRESHOLD <= rate <= MISSING_THRESHOLD:
            indicator_name = f"{col}_missing"
            indicator_cols[indicator_name] = df[col].isnull().astype(int)
            generated_indicators.append(indicator_name)
    if generated_indicators:
        print(f"\n[特征工程] 生成缺失指示列 ({len(generated_indicators)}个): {generated_indicators}")

    high_miss = missing_rates[missing_rates > MISSING_THRESHOLD].index.tolist()
    keep_features = [c for c in feature_cols if c not in high_miss]

    print(f"[缺失处理] 剔除高缺失变量 ({MISSING_THRESHOLD:.0%}): {high_miss}")
    print(f"[缺失处理] 保留需插补特征数: {len(keep_features)}")

    # 修复 1：去掉 StandardScaler，直接插补以避免反转换造成的偏差
    imputer = IterativeImputer(random_state=RANDOM_SEED, max_iter=20, skip_complete=True, verbose=0)
    
    df_keep = df[keep_features].copy()
    idx = df_keep.index
    imputed = imputer.fit_transform(df_keep)
    df_imputed = pd.DataFrame(imputed, columns=keep_features, index=idx)

    # 修复 1 补充：强制 clip 非负变量 (NHANES 数据集中极少有合理负值)
    # 大部分问卷得分、次数、生理指标都必须 >= 0
    print("[缺失处理] 正在对非负变量进行 Clip (下界=0.0) 截断...")
    for v in df_imputed.columns:
        df_imputed[v] = df_imputed[v].clip(lower=0.0)

    # 合并标签、权重、插补后的特征以及缺失指示列
    df_out = df[label_weight_cols].copy()
    df_out = df_out.reset_index(drop=True)
    df_imputed = df_imputed.reset_index(drop=True)
    indicator_cols = indicator_cols.reset_index(drop=True)
    
    df_out = pd.concat([df_out, df_imputed, indicator_cols], axis=1)
    df_out["SEQN"] = df["SEQN"].values
    
    return df_out, high_miss, keep_features, generated_indicators

# =============================================================================
# 5. 权重与报告
# =============================================================================
def create_combined_weight(df):
    df["WTMEC4YR"] = df["WTMEC2YR"] / 2.0
    return df

def report_quality(df, high_miss, keep_features, generated_indicators):
    print("\n" + "="*50)
    print("最终数据集质量报告")
    print("="*50)
    print(f"样本量: {len(df)}")
    print(f"被剔除的高缺失变量: {high_miss}")
    print(f"新增的缺失指示变量: {generated_indicators}")
    print("核心连续变量描述 (检查最小值是否均为非负):")
    # 抽取部分易出负值的特征进行展示
    check_vars = [v for v in ['INDFMPIR', 'SLD012', 'ALQ130', 'SLQ040'] if v in df.columns]
    if check_vars:
        print(df[check_vars].describe().round(2).loc[['min', 'max', 'mean']].to_string())

# =============================================================================
# 主程序
# =============================================================================
def main():
    print("="*60)
    print("NHANES 数据预处理 (异常值+信号修复版)")
    print("="*60)
    df_raw = load_all_cycles(DATA_DIR)
    df = initial_filter(df_raw)
    df = clean_variables(df)
    df, high_miss, keep_features, gen_inds = handle_missing(df)
    df = create_combined_weight(df)
    report_quality(df, high_miss, keep_features, gen_inds)
    df.to_csv(OUTPUT_CLEAN, index=False, encoding="utf-8-sig")
    print(f"\n✅ 数据已保存: {OUTPUT_CLEAN}")

if __name__ == "__main__":
    main()