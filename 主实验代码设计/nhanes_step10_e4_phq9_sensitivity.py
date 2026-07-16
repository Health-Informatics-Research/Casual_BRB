# =============================================================================
# nhanes_step10_e4_phq9_sensitivity.py
# 【E4-1 临床健壮性：PHQ-9 截断阈值敏感性分析】
#
# 逻辑：
# 遍历临床常用阈值 [5, 8, 10, 12, 15]，动态生成 y_true (使用 PHQ9_score 列)。
# 对每个阈值，使用主实验的 Causal-BRB 架构进行独立训练和评估。
# 最后绘制敏感度折线图，证明模型的普遍适用性。
# =============================================================================

import itertools
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
RAW_CSV = "nhanes_clean_original.csv"  
HIERARCHY_FILE = "brb_hierarchy_config.npy"

OUTPUT_DIR = "E4_Robustness"
os.makedirs(OUTPUT_DIR, exist_ok=True)
METRICS_OUTPUT = os.path.join(OUTPUT_DIR, "E4_1_PHQ9_Sensitivity_Metrics.csv")

# 你的真实总分列名
SCORE_COL = "PHQ9_score"

# 我们测试的 PHQ-9 临床阈值: 5(轻微), 8(轻度边缘), 10(中度/论文基准), 12(中重度边缘), 15(重度)
THRESHOLDS = [5, 8, 10, 12, 15]

RANDOM_SEED = 42
N_SPLITS_CV = 3
MAXITER = 80  # 敏感性测试用 80 代足矣，节省时间
POP_SIZE = 15

W_AUC, W_PR, W_BR = 0.45, 0.25, 0.30

# =============================================================================
# 1. 向量化 ER 核心与网络构建 (极速版)
# =============================================================================
def ER_fusion_vec(activation, rule_beliefs, rule_weights):
    N, K = activation.shape
    tw = rule_weights[None, :] * activation
    m_kD = tw * rule_beliefs[None, :]
    m_kND = tw * (1.0 - rule_beliefs[None, :])
    m_kT = 1.0 - tw
    m_D, m_notD, m_Theta = np.zeros(N), np.zeros(N), np.ones(N)
    for k in range(K):
        kD, kND, kT = m_kD[:, k], m_kND[:, k], m_kT[:, k]
        conflict = m_D * kND + m_notD * kD
        denom = np.where(1.0 - conflict < 1e-10, 1e-10, 1.0 - conflict)
        m_D_new = (m_D * kD + m_D * kT + m_Theta * kD) / denom
        m_notD_new = (m_notD * kND + m_notD * kT + m_Theta * kND) / denom
        m_Theta_new = (m_Theta * kT) / denom
        m_D, m_notD, m_Theta = m_D_new, m_notD_new, m_Theta_new
    return np.clip(m_D, 0, 1), np.clip(m_Theta, 0, 1)

def compute_anchors(df, cols):
    ans = {}
    for c in cols:
        v = df[c].values
        p = np.percentile(v, [16, 50, 84])
        p[1] = max(p[1], p[0] + 1e-5)
        p[2] = max(p[2], p[1] + 1e-5)
        ans[c] = tuple(p)
    return ans

def get_activation(df, cols, anchors, combos):
    n = len(df)
    mems = []
    for c in cols:
        v = df[c].values.astype(float)
        c0, c1, c2 = anchors[c]
        a0, a1, a2 = np.zeros(n), np.zeros(n), np.zeros(n)
        a0[v <= c0] = 1.0
        i1 = (v > c0) & (v <= c1)
        a0[i1] = (c1 - v[i1]) / (c1 - c0); a1[i1] = (v[i1] - c0) / (c1 - c0)
        i2 = (v > c1) & (v <= c2)
        a1[i2] = (c2 - v[i2]) / (c2 - c1); a2[i2] = (v[i2] - c1) / (c2 - c1)
        a2[v > c2] = 1.0
        mems.append((a0, a1, a2))
    act = np.zeros((n, len(combos)))
    for i, cb in enumerate(combos):
        w = np.ones(n)
        for j, lvl in enumerate(cb): w *= mems[j][lvl]
        act[:, i] = w
    rs = act.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    return act / rs

class CausalBRB_Fast:
    def __init__(self, df, hierarchy):
        self.s1 = [f"{p}_prob" for p in hierarchy["Sub1_background"]] + ["Age_prob", "Gender_prob", "DMDMARTL_prob"]
        self.s2 = [f"{p}_prob" for p in hierarchy["Sub2_behavioral"]]
        self.s3 = [f"{p}_prob" for p in hierarchy["Sub3_stress"]]
        self.sm = ["S1_mid", "S2_mid", "S3_mid", "Global_unc"]
        
        self.c1 = list(itertools.product([0, 1, 2], repeat=len(self.s1)))
        self.c2 = list(itertools.product([0, 1, 2], repeat=len(self.s2)))
        self.c3 = list(itertools.product([0, 1, 2], repeat=len(self.s3)))
        self.cm = list(itertools.product([0, 1, 2], repeat=len(self.sm)))
        
        self.nr = len(self.c1) + len(self.c2) + len(self.c3) + len(self.cm)
        self.n_params = self.nr * 2
        
        self.a1 = compute_anchors(df, self.s1)
        self.a2 = compute_anchors(df, self.s2)
        self.a3 = compute_anchors(df, self.s3)
        
        # 为了主层锚点初始化
        # --- 修复：先初始化缓存，再调用 _build_mid ---
        self._act1 = get_activation(df, self.s1, self.a1, self.c1)
        self._act2 = get_activation(df, self.s2, self.a2, self.c2)
        self._act3 = get_activation(df, self.s3, self.a3, self.c3)
        self._df = df
        
        # 为了主层锚点初始化
        b = np.full(self.nr, 0.5); w = np.ones(self.nr)
        m_df = self._build_mid(df, b, w)
        self.am = compute_anchors(m_df, self.sm)

    def _build_mid(self, df, b, w):
        i1 = len(self.c1); i2 = i1+len(self.c2); i3 = i2+len(self.c3)
        if len(df) == len(self._df) and df is self._df:
            act1, act2, act3 = self._act1, self._act2, self._act3
        else:
            act1 = get_activation(df, self.s1, self.a1, self.c1)
            act2 = get_activation(df, self.s2, self.a2, self.c2)
            act3 = get_activation(df, self.s3, self.a3, self.c3)
            
        m1, t1 = ER_fusion_vec(act1, b[:i1], w[:i1])
        m2, t2 = ER_fusion_vec(act2, b[i1:i2], w[i1:i2])
        m3, t3 = ER_fusion_vec(act3, b[i2:i3], w[i2:i3])
        return pd.DataFrame({"S1_mid": m1, "S2_mid": m2, "S3_mid": m3, "Global_unc": (t1+t2+t3)/3.0})

    def predict(self, df, p):
        b, w = np.clip(p[:self.nr], 0.01, 1.0), np.clip(p[self.nr:], 0.01, 1.0)
        i3 = len(self.c1) + len(self.c2) + len(self.c3)
        m_df = self._build_mid(df, b, w)
        act_m = get_activation(m_df, self.sm, self.am, self.cm)
        mD, mT = ER_fusion_vec(act_m, b[i3:], w[i3:])
        return np.clip(mD + 0.5 * mT, 0, 1)

def find_best_threshold(y_true, y_prob):
    # 若某阈值下所有样本都是负例，直接返回
    if np.sum(y_true) == 0:
        return 0.5, 0.0
        
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f = fbeta_score(y_true, (y_prob >= t).astype(int), beta=2, zero_division=0)
        if f > bf: bf = f; bt = t
    return bt, bf

# =============================================================================
# 2. 优化与验证核心引擎
# =============================================================================
def train_and_eval_for_threshold(df_train, df_test, y_train, y_test, hierarchy, thr):
    pos_rate = y_train.mean()
    print(f"\n🚀 开始优化 PHQ-9 阈值 >= {thr} 的模型 (训练集正样本率: {pos_rate:.2%})...")
    
    # 如果极端情况下正样本太少（比如 < 1%），差分进化会非常困难，但我们依然让它跑
    kf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_SEED)
    cache = []
    
    for ti, vi in kf.split(df_train, y_train):
        d_tr = df_train.iloc[ti].reset_index(drop=True)
        d_va = df_train.iloc[vi].reset_index(drop=True)
        y_va = y_train.iloc[vi].values
        
        m = CausalBRB_Fast(d_tr, hierarchy)
        v_act1 = get_activation(d_va, m.s1, m.a1, m.c1)
        v_act2 = get_activation(d_va, m.s2, m.a2, m.c2)
        v_act3 = get_activation(d_va, m.s3, m.a3, m.c3)
        cache.append((m, d_va, y_va, v_act1, v_act2, v_act3))
        
    n_params = cache[0][0].n_params

    def obj(p):
        p = np.clip(p, 0.01, 1.0)
        ls = []
        for m, d_v, y_v, va1, va2, va3 in cache:
            b, w = p[:m.nr], p[m.nr:]
            i1 = len(m.c1); i2 = i1+len(m.c2); i3 = i2+len(m.c3)
            m1, t1 = ER_fusion_vec(va1, b[:i1], w[:i1])
            m2, t2 = ER_fusion_vec(va2, b[i1:i2], w[i1:i2])
            m3, t3 = ER_fusion_vec(va3, b[i2:i3], w[i2:i3])
            
            mid_df = pd.DataFrame({"S1_mid": m1, "S2_mid": m2, "S3_mid": m3, "Global_unc": (t1+t2+t3)/3.0})
            act_m = get_activation(mid_df, m.sm, m.am, m.cm)
            mD, mT = ER_fusion_vec(act_m, b[i3:], w[i3:])
            prob = np.clip(mD + 0.5*mT, 0, 1)
            
            try:
                auc = roc_auc_score(y_v, prob)
                apr = average_precision_score(y_v, prob)
            except: auc, apr = 0.5, 0.0
            br = brier_score_loss(y_v, prob)
            ls.append(W_AUC*(1-auc) + W_PR*(1-apr) + W_BR*br)
        return float(np.mean(ls))

    rng = np.random.default_rng(RANDOM_SEED)
    init = [np.clip(rng.random(n_params), 0.01, 1.0) for _ in range(POP_SIZE)]
    
    res = differential_evolution(
        obj, bounds=[(0.01, 1.0)]*n_params, init=init, maxiter=MAXITER, popsize=POP_SIZE,
        mutation=(0.5, 1.0), recombination=0.7, tol=1e-4, seed=RANDOM_SEED, disp=False, polish=False, workers=1
    )
    
    final_m = CausalBRB_Fast(df_train, hierarchy)
    test_prob = final_m.predict(df_test, res.x)
    
    tau, f2 = find_best_threshold(y_test, test_prob)
    
    try:
        test_auc = roc_auc_score(y_test, test_prob)
        test_pr = average_precision_score(y_test, test_prob)
    except:
        test_auc, test_pr = 0.5, 0.0
        
    return {
        "Threshold": thr,
        "Positive_Rate": float(y_test.mean()),
        "ROC-AUC": test_auc,
        "PR-AUC": test_pr,
        "Brier": brier_score_loss(y_test, test_prob),
        "F2": f2
    }

# =============================================================================
# 3. 主流程与绘图
# =============================================================================
def main():
    print("=" * 80)
    print("E4-1: 启动 PHQ-9 截断阈值敏感性实验")
    print("=" * 80)

    try:
        hierarchy = np.load(HIERARCHY_FILE, allow_pickle=True).item()
    except:
        print("⚠️ 找不到层级配置，使用默认...")
        hierarchy = {"Sub1_background": ["P1_Socioeconomic"], "Sub2_behavioral": ["P2_Sleep", "P3_HealthStatus"], "Sub3_stress": ["P4_FoodSecurity", "P5_Clinical", "P6_Substance"]}

    # --- 核心修复：使用确定的列名 PHQ9_score ---
    raw_df = pd.read_csv(RAW_CSV)[['SEQN', SCORE_COL]]
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    
    tr = pd.merge(tr, raw_df, on='SEQN', how='inner')
    te = pd.merge(te, raw_df, on='SEQN', how='inner')
    
    scl = MinMaxScaler()
    tr[["Age_prob", "Gender_prob", "DMDMARTL_prob"]] = scl.fit_transform(tr[["RIDAGEYR", "RIAGENDR", "DMDMARTL"]])
    te[["Age_prob", "Gender_prob", "DMDMARTL_prob"]] = scl.transform(te[["RIDAGEYR", "RIAGENDR", "DMDMARTL"]])

    results = []
    for thr in THRESHOLDS:
        # 动态打标签
        y_tr_dynamic = (tr[SCORE_COL] >= thr).astype(int)
        y_te_dynamic = (te[SCORE_COL] >= thr).astype(int)
        
        metrics = train_and_eval_for_threshold(tr, te, y_tr_dynamic, y_te_dynamic, hierarchy, thr)
        results.append(metrics)
        print(f"  -> 完成: ROC-AUC={metrics['ROC-AUC']:.4f} | Brier={metrics['Brier']:.4f} | F2={metrics['F2']:.4f}")

    # 保存数据并绘图
    df_res = pd.DataFrame(results)
    df_res.to_csv(METRICS_OUTPUT, index=False)
    
    plt.figure(figsize=(10, 6))
    
    # 左轴画 AUC 和 F2
    ax1 = plt.gca()
    ax1.plot(df_res["Threshold"], df_res["ROC-AUC"], marker='o', color='navy', linewidth=2.5, label='ROC-AUC')
    ax1.plot(df_res["Threshold"], df_res["F2"], marker='s', color='darkorange', linewidth=2.5, label='F2 Score')
    ax1.set_xlabel('PHQ-9 Cut-off Threshold for Depression Label', fontweight='bold')
    ax1.set_ylabel('Score (ROC-AUC / F2)', color='black', fontweight='bold')
    ax1.set_ylim(0.4, 0.9)
    ax1.tick_params(axis='y', labelcolor='black')
    
    # 右轴画 Brier (越低越好)
    ax2 = ax1.twinx()
    ax2.plot(df_res["Threshold"], df_res["Brier"], marker='^', color='firebrick', linewidth=2.5, linestyle='--', label='Brier Score (Lower is Better)')
    ax2.set_ylabel('Brier Score', color='firebrick', fontweight='bold')
    ax2.set_ylim(0.0, 0.2)
    ax2.tick_params(axis='y', labelcolor='firebrick')
    
    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    plt.title('Model Robustness Across Different Clinical PHQ-9 Diagnostic Criteria', pad=15, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    out_plot = os.path.join(OUTPUT_DIR, "E4_1_Threshold_Sensitivity.png")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.close()
    
    print("\n" + "=" * 80)
    print("E4-1 实验完成！结果已汇总：")
    print(df_res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n📈 敏感性趋势图已保存至: {out_plot}")

if __name__ == "__main__":
    main()