# =============================================================================
# nhanes_step4_e1_v4_random_graph_5seeds.py
# 【E1-V4 消融实验：5 种子随机因果图均值化验证】
#
# 目的：
#   采用 5 个不同的随机种子打乱 P1-P6 的层级归属，
#   严格保持 1-2-3 (N_PARAMS=396) 的参数结构，并计算核心指标的均值和标准差，
#   以提供严谨的统计学证据，证明 PC 算法提取的图结构优于随机结构。
# =============================================================================

import itertools
import warnings
import os
import random
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
OUTPUT_DIR = "E1_Variants"

# ★ 实验选用的 5 个随机种子 ★
GRAPH_SEEDS = [42, 99, 123, 2026, 888]

N_SPLITS_CV = 3
MAXITER = 120
INIT_POP_SIZE = 28
POLISH = False

W_ROC_AUC = 0.45
W_PR_AUC = 0.25
W_BRIER = 0.30

# =============================================================================
# 核心业务逻辑 (封装为类，方便多种子循环)
# =============================================================================
def compute_anchors(df: pd.DataFrame, cols: List[str]) -> Dict[str, Tuple[float, float, float]]:
    anchors = {}
    for col in cols:
        vals = df[col].astype(float).values
        c0, c1, c2 = np.percentile(vals, [16, 50, 84])
        if c1 <= c0: c1 = c0 + 1e-5
        if c2 <= c1: c2 = c1 + 1e-5
        anchors[col] = (float(c0), float(c1), float(c2))
    return anchors

def calc_memberships(values: np.ndarray, anchors: Tuple[float, float, float]):
    c0, c1, c2 = anchors
    values = values.astype(float)
    a0, a1, a2 = np.zeros_like(values), np.zeros_like(values), np.zeros_like(values)
    idx0 = values <= c0; a0[idx0] = 1.0
    idx1 = (values > c0) & (values <= c1)
    a0[idx1] = (c1 - values[idx1]) / (c1 - c0)
    a1[idx1] = (values[idx1] - c0) / (c1 - c0)
    idx2 = (values > c1) & (values <= c2)
    a1[idx2] = (c2 - values[idx2]) / (c2 - c1)
    a2[idx2] = (values[idx2] - c1) / (c2 - c1)
    idx3 = values > c2; a2[idx3] = 1.0
    return a0, a1, a2

def ER_fusion(activation: np.ndarray, rule_beliefs: np.ndarray, rule_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_samples, n_rules = activation.shape
    m_D = np.zeros(n_samples, dtype=float)
    m_notD = np.zeros(n_samples, dtype=float)
    m_Theta = np.ones(n_samples, dtype=float)
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

class EvolvableER_BRB_Random:
    def __init__(self, df_train: pd.DataFrame, hierarchy_config: dict, dim_config: dict):
        self.H = hierarchy_config
        self.D = dim_config
        self.anchors_sub1 = compute_anchors(df_train, self.H['SUB1_INPUTS'])
        self.anchors_sub2 = compute_anchors(df_train, self.H['SUB2_INPUTS'])
        self.anchors_sub3 = compute_anchors(df_train, self.H['SUB3_INPUTS'])

        dummy_rb1 = {c: 0.5 for c in self.D['COMBOS1']}; dummy_rw1 = {c: 1.0 for c in self.D['COMBOS1']}
        s1_m, s1_u = self._infer_sub(df_train, self.H['SUB1_INPUTS'], self.anchors_sub1, self.D['COMBOS1'], dummy_rb1, dummy_rw1)
        
        dummy_rb2 = {c: 0.5 for c in self.D['COMBOS2']}; dummy_rw2 = {c: 1.0 for c in self.D['COMBOS2']}
        s2_m, s2_u = self._infer_sub(df_train, self.H['SUB2_INPUTS'], self.anchors_sub2, self.D['COMBOS2'], dummy_rb2, dummy_rw2)
        
        dummy_rb3 = {c: 0.5 for c in self.D['COMBOS3']}; dummy_rw3 = {c: 1.0 for c in self.D['COMBOS3']}
        s3_m, s3_u = self._infer_sub(df_train, self.H['SUB3_INPUTS'], self.anchors_sub3, self.D['COMBOS3'], dummy_rb3, dummy_rw3)

        main_df = pd.DataFrame({"Sub1_mid": s1_m, "Sub2_mid": s2_m, "Sub3_mid": s3_m, "Global_unc": (s1_u + s2_u + s3_u) / 3.0})
        self.anchors_main = compute_anchors(main_df, self.H['MAIN_INPUTS'])

    def _activation_matrix(self, df: pd.DataFrame, input_cols: List[str], anchors: Dict, combos: List[Tuple]) -> np.ndarray:
        n_samples = len(df)
        memberships = [calc_memberships(df[col].values, anchors[col]) for col in input_cols]
        activation = np.zeros((n_samples, len(combos)), dtype=float)
        for i, combo in enumerate(combos):
            w = np.ones(n_samples, dtype=float)
            for j, level in enumerate(combo): w *= memberships[j][level]
            activation[:, i] = w
        row_sum = activation.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return activation / row_sum

    def _infer_sub(self, df, input_cols, anchors, combos, rb, rw):
        activation = self._activation_matrix(df, input_cols, anchors, combos)
        m_D, m_Theta = ER_fusion(activation, np.array([rb[c] for c in combos]), np.array([rw[c] for c in combos]))
        return np.clip(m_D + 0.5 * m_Theta, 0.0, 1.0), np.clip(m_Theta, 0.0, 1.0)

    def split_params(self, params: np.ndarray):
        b, w = params[:self.D['N_RULES']], params[self.D['N_RULES']:]
        idx = 0
        rb1 = {c: b[idx+i] for i,c in enumerate(self.D['COMBOS1'])}; rw1 = {c: w[idx+i] for i,c in enumerate(self.D['COMBOS1'])}; idx += len(self.D['COMBOS1'])
        rb2 = {c: b[idx+i] for i,c in enumerate(self.D['COMBOS2'])}; rw2 = {c: w[idx+i] for i,c in enumerate(self.D['COMBOS2'])}; idx += len(self.D['COMBOS2'])
        rb3 = {c: b[idx+i] for i,c in enumerate(self.D['COMBOS3'])}; rw3 = {c: w[idx+i] for i,c in enumerate(self.D['COMBOS3'])}; idx += len(self.D['COMBOS3'])
        rb_m = {c: b[idx+i] for i,c in enumerate(self.D['COMBOS_M'])}; rw_m = {c: w[idx+i] for i,c in enumerate(self.D['COMBOS_M'])}
        return rb1, rw1, rb2, rw2, rb3, rw3, rb_m, rw_m

    def predict(self, df: pd.DataFrame, params: np.ndarray):
        rb1, rw1, rb2, rw2, rb3, rw3, rb_m, rw_m = self.split_params(params)
        s1_m, s1_u = self._infer_sub(df, self.H['SUB1_INPUTS'], self.anchors_sub1, self.D['COMBOS1'], rb1, rw1)
        s2_m, s2_u = self._infer_sub(df, self.H['SUB2_INPUTS'], self.anchors_sub2, self.D['COMBOS2'], rb2, rw2)
        s3_m, s3_u = self._infer_sub(df, self.H['SUB3_INPUTS'], self.anchors_sub3, self.D['COMBOS3'], rb3, rw3)
        main_df = pd.DataFrame({"Sub1_mid": s1_m, "Sub2_mid": s2_m, "Sub3_mid": s3_m, "Global_unc": (s1_u + s2_u + s3_u)/3.0})
        act = self._activation_matrix(main_df, self.H['MAIN_INPUTS'], self.anchors_main, self.D['COMBOS_M'])
        m_D, m_Theta = ER_fusion(act, np.array([rb_m[c] for c in self.D['COMBOS_M']]), np.array([rw_m[c] for c in self.D['COMBOS_M']]))
        return np.clip(m_D + 0.5 * m_Theta, 0.0, 1.0)

# =============================================================================
# 工具函数 (修复参数、评估)
# =============================================================================
def enforce_monotonicity(rb: Dict[Tuple[int, ...], float], combos: List[Tuple[int, ...]]) -> Dict[Tuple[int, ...], float]:
    rb = rb.copy()
    if not combos: return rb
    n_dims = len(combos[0])
    for _ in range(max(n_dims, 3)):
        changed = False
        for combo in combos:
            for i in range(n_dims):
                if combo[i] > 0:
                    lesser = list(combo)
                    lesser[i] -= 1
                    if rb[combo] < rb[tuple(lesser)]:
                        rb[combo] = rb[tuple(lesser)]
                        changed = True
        if not changed: break
    return rb

def repair_parameters(params: np.ndarray, dim_config: dict) -> np.ndarray:
    params = np.clip(np.asarray(params, dtype=float), 0.0, 1.0)
    b, w = params[:dim_config['N_RULES']].copy(), params[dim_config['N_RULES']:].copy()
    
    idx = 0
    rb1 = {c: b[idx+i] for i,c in enumerate(dim_config['COMBOS1'])}; idx += len(dim_config['COMBOS1'])
    rb2 = {c: b[idx+i] for i,c in enumerate(dim_config['COMBOS2'])}; idx += len(dim_config['COMBOS2'])
    rb3 = {c: b[idx+i] for i,c in enumerate(dim_config['COMBOS3'])}; idx += len(dim_config['COMBOS3'])
    rb_m = {c: b[idx+i] for i,c in enumerate(dim_config['COMBOS_M'])}

    rb1 = enforce_monotonicity(rb1, dim_config['COMBOS1'])
    rb2 = enforce_monotonicity(rb2, dim_config['COMBOS2'])
    rb3 = enforce_monotonicity(rb3, dim_config['COMBOS3'])
    rb_m = enforce_monotonicity(rb_m, dim_config['COMBOS_M'])

    repaired = []
    for c in dim_config['COMBOS1']: repaired.append(rb1[c])
    for c in dim_config['COMBOS2']: repaired.append(rb2[c])
    for c in dim_config['COMBOS3']: repaired.append(rb3[c])
    for c in dim_config['COMBOS_M']: repaired.append(rb_m[c])
    return np.concatenate([np.clip(repaired, 0.0, 1.0), np.clip(w, 0.01, 1.0)])

def build_initial_pop(seed, dim_config: dict) -> np.ndarray:
    rng = np.random.default_rng(seed)
    beliefs = []
    for combos in [dim_config['COMBOS1'], dim_config['COMBOS2'], dim_config['COMBOS3'], dim_config['COMBOS_M']]:
        for c in combos:
            beliefs.append(0.0 if len(c)==0 else sum(c)/(2.0*len(c)))
    base = repair_parameters(np.concatenate([np.array(beliefs), np.ones(dim_config['N_RULES'])]), dim_config)
    
    pop = [base]
    for _ in range(9): pop.append(repair_parameters(base + rng.normal(0, 0.06, dim_config['N_PARAMS']), dim_config))
    for _ in range(9): pop.append(repair_parameters(base + rng.normal(0, 0.15, dim_config['N_PARAMS']), dim_config))
    while len(pop) < INIT_POP_SIZE: pop.append(repair_parameters(rng.random(dim_config['N_PARAMS']), dim_config))
    return np.array(pop)

def find_best_threshold(y_true, y_prob):
    best_tau, best_f2 = 0.5, -1.0
    for tau in np.linspace(0.05, 0.95, 91):
        f2 = fbeta_score(y_true, (y_prob >= tau).astype(int), beta=2, zero_division=0)
        if f2 > best_f2: best_f2 = f2; best_tau = tau
    return best_tau, best_f2

# =============================================================================
# 主程序：5 种子循环执行
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 80)
    print("E1-V4 消融实验：随机层级图 (5 Seeds 连续验证)")
    print("=" * 80)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    scaler = MinMaxScaler()
    cols = ["RIDAGEYR", "RIAGENDR", "DMDMARTL"]
    train_df[["Age_prob", "Gender_prob", "DMDMARTL_prob"]] = scaler.fit_transform(train_df[cols])
    test_df[["Age_prob", "Gender_prob", "DMDMARTL_prob"]] = scaler.transform(test_df[cols])
    
    y_train, y_test = train_df["label"], test_df["label"].values
    all_metrics = []

    for idx, current_seed in enumerate(GRAPH_SEEDS):
        print(f"\n[{idx+1}/5] 正在运行随机种子: {current_seed} ...")
        
        # 1. 生成特定的随机图
        paths = ["P1_Socioeconomic", "P2_Sleep", "P3_HealthStatus", "P4_FoodSecurity", "P5_Clinical", "P6_Substance"]
        random.seed(current_seed)
        shuffled = paths.copy()
        random.shuffle(shuffled)
        
        hierarchy = {
            "SUB1_INPUTS": [f"{p}_prob" for p in shuffled[0:1]] + ["Age_prob", "Gender_prob", "DMDMARTL_prob"],
            "SUB2_INPUTS": [f"{p}_prob" for p in shuffled[1:3]],
            "SUB3_INPUTS": [f"{p}_prob" for p in shuffled[3:6]],
            "MAIN_INPUTS": ["Sub1_mid", "Sub2_mid", "Sub3_mid", "Global_unc"]
        }
        
        dim_config = {
            "COMBOS1": list(itertools.product([0, 1, 2], repeat=len(hierarchy['SUB1_INPUTS']))),
            "COMBOS2": list(itertools.product([0, 1, 2], repeat=len(hierarchy['SUB2_INPUTS']))),
            "COMBOS3": list(itertools.product([0, 1, 2], repeat=len(hierarchy['SUB3_INPUTS']))),
            "COMBOS_M": list(itertools.product([0, 1, 2], repeat=4))
        }
        dim_config['N_RULES'] = sum(len(c) for c in [dim_config['COMBOS1'], dim_config['COMBOS2'], dim_config['COMBOS3'], dim_config['COMBOS_M']])
        dim_config['N_PARAMS'] = dim_config['N_RULES'] * 2 # 确保还是 396 维

        # 2. 准备 CV 缓存
        cv_cache = []
        kf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)
        for tr_idx, val_idx in kf.split(train_df, y_train):
            model = EvolvableER_BRB_Random(train_df.iloc[tr_idx].copy(), hierarchy, dim_config)
            cv_cache.append((model, train_df.iloc[val_idx].copy(), y_train.iloc[val_idx].values))

        # 3. 目标函数
        def objective(params: np.ndarray) -> float:
            params = repair_parameters(params, dim_config)
            losses = []
            for model, df_val, y_val in cv_cache:
                probs = model.predict(df_val, params)
                brier = brier_score_loss(y_val, probs)
                try: auc = roc_auc_score(y_val, probs)
                except: auc = 0.5
                try: ap = average_precision_score(y_val, probs)
                except: ap = float(np.mean(y_val))
                losses.append(W_ROC_AUC * (1.0 - auc) + W_PR_AUC * (1.0 - ap) + W_BRIER * brier)
            return float(np.mean(losses))

        # 4. 优化
        init_pop = build_initial_pop(42, dim_config)
        result = differential_evolution(
            objective, bounds=[(0.0, 1.0)] * dim_config['N_PARAMS'], init=init_pop,
            maxiter=MAXITER, mutation=(0.5, 1.0), recombination=0.7, tol=1e-5,
            seed=42, disp=False, workers=1, updating="immediate"
        )

        # 5. 测试集评估
        best_params = repair_parameters(result.x, dim_config)
        final_model = EvolvableER_BRB_Random(train_df, hierarchy, dim_config)
        test_probs = final_model.predict(test_df, best_params)
        
        metrics = {
            "Seed": current_seed,
            "ROC_AUC": roc_auc_score(y_test, test_probs),
            "PR_AUC": average_precision_score(y_test, test_probs),
            "Brier": brier_score_loss(y_test, test_probs),
            "F2": find_best_threshold(y_test, test_probs)[1]
        }
        all_metrics.append(metrics)
        print(f"完成 -> ROC-AUC: {metrics['ROC_AUC']:.4f} | Brier: {metrics['Brier']:.4f}")

    # =========================================================================
    # 最终汇总统计
    # =========================================================================
    print("\n" + "=" * 80)
    print("E1-V4 (Random Graph) 5次独立实验结果汇总")
    print("=" * 80)
    df_metrics = pd.DataFrame(all_metrics)
    summary = df_metrics.agg(['mean', 'std']).T
    
    print(df_metrics.to_string(index=False))
    print("\n最终填入论文表格的数据 (Mean ± SD):")
    print(f"ROC-AUC : {summary.loc['ROC_AUC', 'mean']:.3f} ± {summary.loc['ROC_AUC', 'std']:.3f}")
    print(f"PR-AUC  : {summary.loc['PR_AUC', 'mean']:.3f} ± {summary.loc['PR_AUC', 'std']:.3f}")
    print(f"Brier   : {summary.loc['Brier', 'mean']:.3f} ± {summary.loc['Brier', 'std']:.3f}")
    print(f"F2      : {summary.loc['F2', 'mean']:.3f} ± {summary.loc['F2', 'std']:.3f}")
    
    csv_path = os.path.join(OUTPUT_DIR, "step4_e1_v4_5seeds_summary.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"\n结果已保存至: {csv_path}")

if __name__ == "__main__":
    main()