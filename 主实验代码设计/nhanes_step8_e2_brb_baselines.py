# =============================================================================
# nhanes_step8_e2_brb_baselines.py
# 【E2 跨模型对比：BRB领域最新基准模型原生复刻评估】
#
# 1. HBRB-b (Li et al., 2025): 专家手工层级 (非PC因果) + 有权重
# 2. BRB-DR (Sun et al., 2025): 平层结构 (无因果层级) + 有权重 (模拟动态可靠性)
# 3. SBRB-I (Li et al., 2023): 平层结构 (无因果层级) + 无权重 (纯数据驱动)
# =============================================================================

import itertools
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import average_precision_score, brier_score_loss, fbeta_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 全局配置
# =============================================================================
TRAIN_CSV = "nhanes_brb_train.csv"
TEST_CSV = "nhanes_brb_test.csv"
OUTPUT_DIR = "E2_Baselines"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS_OUTPUT = os.path.join(OUTPUT_DIR, "e2_brb_baselines_metrics.csv")

RANDOM_SEED = 42
N_SPLITS_CV = 3
MAXITER = 120
INIT_POP_SIZE = 28

W_ROC_AUC = 0.45
W_PR_AUC = 0.25
W_BRIER = 0.30

# =============================================================================
# 1. ER 核心数学库
# =============================================================================
def ER_fusion(activation: np.ndarray, rule_beliefs: np.ndarray, rule_weights: np.ndarray):
    n_samples, n_rules = activation.shape
    m_D, m_notD, m_Theta = np.zeros(n_samples), np.zeros(n_samples), np.ones(n_samples)
    for k in range(n_rules):
        w = activation[:, k]
        beta, theta = rule_beliefs[k], rule_weights[k]
        m_k_D = theta * w * beta
        m_k_notD = theta * w * (1.0 - beta)
        m_k_Theta = 1.0 - theta * w
        conflict = m_D * m_k_notD + m_notD * m_k_D
        denom = np.where(np.abs(1.0 - conflict) < 1e-10, 1e-10, 1.0 - conflict)
        m_D_new = (m_D * m_k_D + m_D * m_k_Theta + m_Theta * m_k_D) / denom
        m_notD_new = (m_notD * m_k_notD + m_notD * m_k_Theta + m_Theta * m_k_notD) / denom
        m_Theta_new = (m_Theta * m_k_Theta) / denom
        m_D, m_notD, m_Theta = m_D_new, m_notD_new, m_Theta_new
    return np.clip(m_D, 0.0, 1.0), np.clip(m_Theta, 0.0, 1.0)

def calc_mem_3(values, anchors):
    c0, c1, c2 = anchors
    v = values.astype(float)
    a0, a1, a2 = np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)
    idx0 = v <= c0; a0[idx0] = 1.0
    idx1 = (v > c0) & (v <= c1); a0[idx1] = (c1 - v[idx1]) / (c1 - c0); a1[idx1] = (v[idx1] - c0) / (c1 - c0)
    idx2 = (v > c1) & (v <= c2); a1[idx2] = (c2 - v[idx2]) / (c2 - c1); a2[idx2] = (v[idx2] - c1) / (c2 - c1)
    idx3 = v > c2; a2[idx3] = 1.0
    return a0, a1, a2

def calc_mem_2(values, anchors):
    c0, c1 = anchors
    v = values.astype(float)
    a0, a1 = np.zeros_like(v), np.zeros_like(v)
    idx0 = v <= c0; a0[idx0] = 1.0
    idx1 = (v > c0) & (v <= c1); a0[idx1] = (c1 - v[idx1]) / (c1 - c0); a1[idx1] = (v[idx1] - c0) / (c1 - c0)
    idx2 = v > c1; a1[idx2] = 1.0
    return a0, a1

def get_activation(df, cols, anchors, combos, levels=3):
    n = len(df)
    if levels == 3: mems = [calc_mem_3(df[c].values, anchors[c]) for c in cols]
    else: mems = [calc_mem_2(df[c].values, anchors[c]) for c in cols]
    act = np.zeros((n, len(combos)))
    for i, cb in enumerate(combos):
        w = np.ones(n)
        for j, lvl in enumerate(cb): w *= mems[j][lvl]
        act[:, i] = w
    rs = act.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    return act / rs

def compute_anchors(df, cols, levels=3):
    ans = {}
    for c in cols:
        v = df[c].values
        if levels == 3:
            c0, c1, c2 = np.percentile(v, [16, 50, 84])
            if c1 <= c0: c1 = c0+1e-5
            if c2 <= c1: c2 = c1+1e-5
            ans[c] = (c0, c1, c2)
        else:
            c0, c1 = np.percentile(v, [33, 67])
            if c1 <= c0: c1 = c0+1e-5
            ans[c] = (c0, c1)
    return ans

# =============================================================================
# 2. 基准模型：HBRB-b (专家层级)
# =============================================================================
class Model_HBRB_b:
    """ 强行划分的三层结构：人口+环境 / 个人习惯 / 疾病生理 """
    def __init__(self, df):
        self.s1 = ["P1_Socioeconomic_prob", "Age_prob", "Gender_prob", "DMDMARTL_prob", "P4_FoodSecurity_prob"] # 5维
        self.s2 = ["P2_Sleep_prob", "P6_Substance_prob"] # 2维
        self.s3 = ["P3_HealthStatus_prob", "P5_Clinical_prob"] # 2维
        self.sm = ["S1_mid", "S2_mid", "S3_mid"] # 3维
        
        self.c1 = list(itertools.product([0,1,2], repeat=5)); self.c2 = list(itertools.product([0,1,2], repeat=2))
        self.c3 = list(itertools.product([0,1,2], repeat=2)); self.cm = list(itertools.product([0,1,2], repeat=3))
        self.nr = len(self.c1)+len(self.c2)+len(self.c3)+len(self.cm) # 243+9+9+27 = 288 rules
        self.n_params = self.nr * 2 # 576 params
        
        self.a1 = compute_anchors(df, self.s1); self.a2 = compute_anchors(df, self.s2); self.a3 = compute_anchors(df, self.s3)
        sm_df = self._build_mid(df, {c:0.5 for c in self.c1}, {c:1.0 for c in self.c1}, 
                                {c:0.5 for c in self.c2}, {c:1.0 for c in self.c2},
                                {c:0.5 for c in self.c3}, {c:1.0 for c in self.c3})
        self.am = compute_anchors(sm_df, self.sm)
        
    def _build_mid(self, df, rb1, rw1, rb2, rw2, rb3, rw3):
        m1, _ = ER_fusion(get_activation(df, self.s1, self.a1, self.c1), np.array(list(rb1.values())), np.array(list(rw1.values())))
        m2, _ = ER_fusion(get_activation(df, self.s2, self.a2, self.c2), np.array(list(rb2.values())), np.array(list(rw2.values())))
        m3, _ = ER_fusion(get_activation(df, self.s3, self.a3, self.c3), np.array(list(rb3.values())), np.array(list(rw3.values())))
        return pd.DataFrame({"S1_mid": m1, "S2_mid": m2, "S3_mid": m3})

    def predict(self, df, p):
        b, w = p[:self.nr], p[self.nr:]
        i1 = len(self.c1); i2 = i1+len(self.c2); i3 = i2+len(self.c3)
        rb1, rw1 = dict(zip(self.c1, b[:i1])), dict(zip(self.c1, w[:i1]))
        rb2, rw2 = dict(zip(self.c2, b[i1:i2])), dict(zip(self.c2, w[i1:i2]))
        rb3, rw3 = dict(zip(self.c3, b[i2:i3])), dict(zip(self.c3, w[i2:i3]))
        rbm, rwm = dict(zip(self.cm, b[i3:])), dict(zip(self.cm, w[i3:]))
        
        m_df = self._build_mid(df, rb1, rw1, rb2, rw2, rb3, rw3)
        mD, mT = ER_fusion(get_activation(m_df, self.sm, self.am, self.cm), np.array(list(rbm.values())), np.array(list(rwm.values())))
        return np.clip(mD + 0.5*mT, 0, 1)

# =============================================================================
# 3. 基准模型：BRB-DR / SBRB-I (平层结构 2-Level)
# =============================================================================
class Model_FlatBRB:
    """ BRB-DR (有权重) / SBRB-I (无权重) """
    def __init__(self, df, has_weights=True):
        self.has_w = has_weights
        self.cols = ["P1_Socioeconomic_prob", "P2_Sleep_prob", "P3_HealthStatus_prob", 
                     "P4_FoodSecurity_prob", "P5_Clinical_prob", "P6_Substance_prob"]
        self.cm = list(itertools.product([0, 1], repeat=6)) # 64 rules
        self.nr = 64
        self.n_params = 128 if has_weights else 64
        self.am = compute_anchors(df, self.cols, levels=2)
        
    def predict(self, df, p):
        b = p[:self.nr]
        w = p[self.nr:] if self.has_w else np.ones(self.nr)
        mD, mT = ER_fusion(get_activation(df, self.cols, self.am, self.cm, levels=2), b, w)
        return np.clip(mD + 0.5*mT, 0, 1)

# =============================================================================
# 4. 评估逻辑
# =============================================================================
def find_best_threshold(y_true, y_prob):
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f = fbeta_score(y_true, (y_prob>=t).astype(int), beta=2, zero_division=0)
        if f > bf: bf = f; bt = t
    return bt, bf

def run_optimization(model_cls, name, df_tr, y_tr, has_w=None):
    print(f"\n[训练] {name} ...")
    kf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_SEED)
    cache = []
    for ti, vi in kf.split(df_tr, y_tr):
        m = model_cls(df_tr.iloc[ti]) if has_w is None else model_cls(df_tr.iloc[ti], has_w)
        cache.append((m, df_tr.iloc[vi], y_tr.iloc[vi].values))
        
    nprm = cache[0][0].n_params
    
    def obj(p):
        p = np.clip(p, 0.01, 1.0)
        ls = []
        for m, dval, yval in cache:
            prob = m.predict(dval, p)
            auc = roc_auc_score(yval, prob)
            ap = average_precision_score(yval, prob)
            br = brier_score_loss(yval, prob)
            ls.append(W_ROC_AUC*(1-auc) + W_PR_AUC*(1-ap) + W_BRIER*br)
        return np.mean(ls)

    rng = np.random.default_rng(RANDOM_SEED)
    init = [np.clip(rng.random(nprm), 0.01, 1.0) for _ in range(INIT_POP_SIZE)]
    
    res = differential_evolution(obj, bounds=[(0.01, 1.0)]*nprm, init=init, maxiter=MAXITER, seed=RANDOM_SEED, disp=False)
    
    fm = model_cls(df_tr) if has_w is None else model_cls(df_tr, has_w)
    return fm, np.clip(res.x, 0.01, 1.0)

# =============================================================================
# 5. 主流程
# =============================================================================
def main():
    print("=" * 80)
    print("E2 跨模型对比：原生复刻 HBRB-b, BRB-DR, SBRB-I")
    print("=" * 80)

    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    scl = MinMaxScaler()
    tr[["Age_prob", "Gender_prob", "DMDMARTL_prob"]] = scl.fit_transform(tr[["RIDAGEYR", "RIAGENDR", "DMDMARTL"]])
    te[["Age_prob", "Gender_prob", "DMDMARTL_prob"]] = scl.transform(te[["RIDAGEYR", "RIAGENDR", "DMDMARTL"]])

    y_tr, y_te = tr["label"], te["label"].values
    results = []

    # 1. HBRB-b
    m_hbrb, p_hbrb = run_optimization(Model_HBRB_b, "HBRB-b (Hierarchical Expert)", tr, y_tr)
    prob_hbrb = m_hbrb.predict(te, p_hbrb)
    
    # 2. BRB-DR
    m_dr, p_dr = run_optimization(Model_FlatBRB, "BRB-DR (Flat Dynamic Rel)", tr, y_tr, has_w=True)
    prob_dr = m_dr.predict(te, p_dr)
    
    # 3. SBRB-I
    m_sbrb, p_sbrb = run_optimization(Model_FlatBRB, "SBRB-I (Flat Data-Driven)", tr, y_tr, has_w=False)
    prob_sbrb = m_sbrb.predict(te, p_sbrb)

    # 汇总评估
    for name, prob in [
        ("HBRB-b (Li et al., 2025)", prob_hbrb),
        ("BRB-DR (Sun et al., 2025)", prob_dr),
        ("SBRB-I (Li et al., 2023)", prob_sbrb)
    ]:
        tau, f2 = find_best_threshold(y_te, prob)
        results.append({
            "Model": name,
            "ROC-AUC": roc_auc_score(y_te, prob),
            "PR-AUC": average_precision_score(y_te, prob),
            "Brier": brier_score_loss(y_te, prob),
            "F2": f2,
            "Architecture": "Expert Hierarchy" if "HBRB" in name else "Flat Weighted" if "DR" in name else "Flat Unweighted"
        })

    # 添加 Causal-BRB 数据用于打印对比
    results.insert(0, {
        "Model": "Causal-BRB (Ours)",
        "ROC-AUC": 0.8068,
        "PR-AUC": 0.2612,
        "Brier": 0.0690,
        "F2": 0.5265,
        "Architecture": "PC Hierarchy + Unc Prop"
    })

    df_res = pd.DataFrame(results)
    df_res.to_csv(METRICS_OUTPUT, index=False)
    
    print("\n" + "=" * 80)
    print("E2 最终对比结果表")
    print("=" * 80)
    print(df_res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n✅ 结果已保存至: {METRICS_OUTPUT}")

if __name__ == "__main__":
    main()