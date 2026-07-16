# =============================================================================
# nhanes_step4_e1_v2_unweighted.py
# 【E1-V2 消融实验：无规则权重 (Unweighted ER)】
#
# 目的：
#   关闭规则权重，所有规则的 theta_k 强制设为 1.0 (退化为原始 ER 公式)。
#   N_PARAMS 减半为 198 (只优化 beta)。
#   借此证明引入规则可信度权重对于处理复杂医疗数据的贡献。
# =============================================================================

import itertools
import warnings
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 全局配置 (★ 引入专属输出文件夹 ★)
# =============================================================================
TRAIN_CSV = "nhanes_brb_train.csv"
TEST_CSV = "nhanes_brb_test.csv"
HIERARCHY_FILE = "brb_hierarchy_config.npy"

OUTPUT_DIR = "E1_Variants"

# ★ E1-V2 专属输出文件 ★
PARAMS_OUTPUT = os.path.join(OUTPUT_DIR, "e1_v2_params.npy")
DIM_INFO_OUTPUT = os.path.join(OUTPUT_DIR, "e1_v2_dim_info.npy")
METRICS_OUTPUT = os.path.join(OUTPUT_DIR, "step4_e1_v2_test_metrics.csv")

RANDOM_SEED = 42
N_SPLITS_CV = 3
MAXITER = 120
INIT_POP_SIZE = 28
POLISH = False

W_ROC_AUC = 0.45
W_PR_AUC = 0.25
W_BRIER = 0.30

# =============================================================================
# 1. 数据加载与人口学特征扩充
# =============================================================================
def load_and_augment_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    required_cols = ["RIDAGEYR", "RIAGENDR", "DMDMARTL", "label"]
    scaler = MinMaxScaler()
    cols_to_scale = ["RIDAGEYR", "RIAGENDR", "DMDMARTL"]

    train_df[["Age_prob", "Gender_prob", "DMDMARTL_prob"]] = scaler.fit_transform(train_df[cols_to_scale])
    test_df[["Age_prob", "Gender_prob", "DMDMARTL_prob"]] = scaler.transform(test_df[cols_to_scale])

    return train_df, test_df

# =============================================================================
# 2. 因果层级配置 (带回 Global_unc)
# =============================================================================
def load_hierarchy() -> Dict[str, List[str]]:
    try:
        hierarchy = np.load(HIERARCHY_FILE, allow_pickle=True).item()
    except FileNotFoundError:
        hierarchy = {
            "Sub1_background": ["P1_Socioeconomic"],
            "Sub2_behavioral": ["P2_Sleep", "P3_HealthStatus"],
            "Sub3_stress": ["P4_FoodSecurity", "P5_Clinical", "P6_Substance"],
        }
    return hierarchy

HIERARCHY = load_hierarchy()

SUB1_INPUTS = [f"{p}_prob" for p in HIERARCHY["Sub1_background"]] + ["Age_prob", "Gender_prob", "DMDMARTL_prob"]
SUB2_INPUTS = [f"{p}_prob" for p in HIERARCHY["Sub2_behavioral"]]
SUB3_INPUTS = [f"{p}_prob" for p in HIERARCHY["Sub3_stress"]]

MAIN_INPUTS = ["Sub1_mid", "Sub2_mid", "Sub3_mid", "Global_unc"]

DIM1 = len(SUB1_INPUTS)
DIM2 = len(SUB2_INPUTS)
DIM3 = len(SUB3_INPUTS)
DIM_M = len(MAIN_INPUTS)

COMBOS1 = list(itertools.product([0, 1, 2], repeat=DIM1))
COMBOS2 = list(itertools.product([0, 1, 2], repeat=DIM2))
COMBOS3 = list(itertools.product([0, 1, 2], repeat=DIM3))
COMBOS_M = list(itertools.product([0, 1, 2], repeat=DIM_M))

N_RULES = len(COMBOS1) + len(COMBOS2) + len(COMBOS3) + len(COMBOS_M)
# ★ E1-V2 核心修改：参数量减半，只优化 beta，不优化 theta ★
N_PARAMS = N_RULES 

# =============================================================================
# 3. ER-BRB 基础函数
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
    a0 = np.zeros_like(values, dtype=float)
    a1 = np.zeros_like(values, dtype=float)
    a2 = np.zeros_like(values, dtype=float)

    idx0 = values <= c0
    a0[idx0] = 1.0

    idx1 = (values > c0) & (values <= c1)
    a0[idx1] = (c1 - values[idx1]) / (c1 - c0)
    a1[idx1] = (values[idx1] - c0) / (c1 - c0)

    idx2 = (values > c1) & (values <= c2)
    a1[idx2] = (c2 - values[idx2]) / (c2 - c1)
    a2[idx2] = (values[idx2] - c1) / (c2 - c1)

    idx3 = values > c2
    a2[idx3] = 1.0
    return a0, a1, a2

def ER_fusion(activation: np.ndarray, rule_beliefs: np.ndarray, rule_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_samples, n_rules = activation.shape
    m_D = np.zeros(n_samples, dtype=float)
    m_notD = np.zeros(n_samples, dtype=float)
    m_Theta = np.ones(n_samples, dtype=float)

    for k in range(n_rules):
        w = activation[:, k]
        beta = rule_beliefs[k]
        theta = rule_weights[k]

        m_k_D = theta * w * beta
        m_k_notD = theta * w * (1.0 - beta)
        m_k_Theta = 1.0 - theta * w

        conflict = m_D * m_k_notD + m_notD * m_k_D
        denom = 1.0 - conflict
        denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)

        m_D_new = (m_D * m_k_D + m_D * m_k_Theta + m_Theta * m_k_D) / denom
        m_notD_new = (m_notD * m_k_notD + m_notD * m_k_Theta + m_Theta * m_k_notD) / denom
        m_Theta_new = (m_Theta * m_k_Theta) / denom

        m_D, m_notD, m_Theta = m_D_new, m_notD_new, m_Theta_new

    m_D = np.clip(m_D, 0.0, 1.0)
    m_Theta = np.clip(m_Theta, 0.0, 1.0)
    return m_D, m_Theta

# =============================================================================
# 4. 可进化 ER-BRB 模型
# =============================================================================
class EvolvableER_BRB:
    def __init__(self, df_train: pd.DataFrame):
        self.anchors_sub1 = compute_anchors(df_train, SUB1_INPUTS)
        self.anchors_sub2 = compute_anchors(df_train, SUB2_INPUTS)
        self.anchors_sub3 = compute_anchors(df_train, SUB3_INPUTS)

        dummy_rb1 = {c: 0.5 for c in COMBOS1}
        dummy_rw1 = {c: 1.0 for c in COMBOS1}
        s1_mid, s1_unc = self._infer_sub(df_train, SUB1_INPUTS, self.anchors_sub1, COMBOS1, dummy_rb1, dummy_rw1)

        dummy_rb2 = {c: 0.5 for c in COMBOS2}
        dummy_rw2 = {c: 1.0 for c in COMBOS2}
        s2_mid, s2_unc = self._infer_sub(df_train, SUB2_INPUTS, self.anchors_sub2, COMBOS2, dummy_rb2, dummy_rw2)

        dummy_rb3 = {c: 0.5 for c in COMBOS3}
        dummy_rw3 = {c: 1.0 for c in COMBOS3}
        s3_mid, s3_unc = self._infer_sub(df_train, SUB3_INPUTS, self.anchors_sub3, COMBOS3, dummy_rb3, dummy_rw3)

        main_df = self._build_main_df(s1_mid, s2_mid, s3_mid, s1_unc, s2_unc, s3_unc)
        self.anchors_main = compute_anchors(main_df, MAIN_INPUTS)

    @staticmethod
    def _build_main_df(s1_mid, s2_mid, s3_mid, s1_unc, s2_unc, s3_unc) -> pd.DataFrame:
        global_unc = (s1_unc + s2_unc + s3_unc) / 3.0
        return pd.DataFrame({
            "Sub1_mid": s1_mid,
            "Sub2_mid": s2_mid,
            "Sub3_mid": s3_mid,
            "Global_unc": global_unc,
        })

    def _activation_matrix(self, df: pd.DataFrame, input_cols: List[str], anchors: Dict[str, Tuple[float, float, float]], combos: List[Tuple[int, ...]]) -> np.ndarray:
        n_samples = len(df)
        memberships = [calc_memberships(df[col].values, anchors[col]) for col in input_cols]
        activation = np.zeros((n_samples, len(combos)), dtype=float)
        for i, combo in enumerate(combos):
            w = np.ones(n_samples, dtype=float)
            for j, level in enumerate(combo):
                w *= memberships[j][level]
            activation[:, i] = w
        row_sum = activation.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return activation / row_sum

    def _infer_sub(self, df: pd.DataFrame, input_cols: List[str], anchors: Dict[str, Tuple[float, float, float]], combos: List[Tuple[int, ...]], rb: Dict[Tuple[int, ...], float], rw: Dict[Tuple[int, ...], float]) -> Tuple[np.ndarray, np.ndarray]:
        activation = self._activation_matrix(df, input_cols, anchors, combos)
        beliefs = np.array([rb[c] for c in combos], dtype=float)
        weights = np.array([rw[c] for c in combos], dtype=float)
        m_D, m_Theta = ER_fusion(activation, beliefs, weights)
        sub_mid = m_D + 0.5 * m_Theta
        sub_unc = m_Theta
        return np.clip(sub_mid, 0.0, 1.0), np.clip(sub_unc, 0.0, 1.0)

    def _infer_main(self, main_df: pd.DataFrame, rb_m: Dict[Tuple[int, ...], float], rw_m: Dict[Tuple[int, ...], float]) -> Tuple[np.ndarray, np.ndarray]:
        activation = self._activation_matrix(main_df, MAIN_INPUTS, self.anchors_main, COMBOS_M)
        beliefs = np.array([rb_m[c] for c in COMBOS_M], dtype=float)
        weights = np.array([rw_m[c] for c in COMBOS_M], dtype=float)
        return ER_fusion(activation, beliefs, weights)

    def split_params(self, params: np.ndarray):
        # ★ E1-V2 核心修改：只切分 beliefs，全部 weights 强制赋予 1.0 ★
        beliefs = params[:N_RULES]
        
        idx = 0
        rb1 = {c: beliefs[idx + i] for i, c in enumerate(COMBOS1)}
        rw1 = {c: 1.0 for c in COMBOS1}
        idx += len(COMBOS1)
        
        rb2 = {c: beliefs[idx + i] for i, c in enumerate(COMBOS2)}
        rw2 = {c: 1.0 for c in COMBOS2}
        idx += len(COMBOS2)
        
        rb3 = {c: beliefs[idx + i] for i, c in enumerate(COMBOS3)}
        rw3 = {c: 1.0 for c in COMBOS3}
        idx += len(COMBOS3)
        
        rb_m = {c: beliefs[idx + i] for i, c in enumerate(COMBOS_M)}
        rw_m = {c: 1.0 for c in COMBOS_M}
        
        return rb1, rw1, rb2, rw2, rb3, rw3, rb_m, rw_m

    def predict(self, df: pd.DataFrame, params: np.ndarray, return_uncertainty: bool = False):
        rb1, rw1, rb2, rw2, rb3, rw3, rb_m, rw_m = self.split_params(params)
        s1_mid, s1_unc = self._infer_sub(df, SUB1_INPUTS, self.anchors_sub1, COMBOS1, rb1, rw1)
        s2_mid, s2_unc = self._infer_sub(df, SUB2_INPUTS, self.anchors_sub2, COMBOS2, rb2, rw2)
        s3_mid, s3_unc = self._infer_sub(df, SUB3_INPUTS, self.anchors_sub3, COMBOS3, rb3, rw3)
        
        main_df = self._build_main_df(s1_mid, s2_mid, s3_mid, s1_unc, s2_unc, s3_unc)
        m_D, m_Theta = self._infer_main(main_df, rb_m, rw_m)
        
        if return_uncertainty:
            return m_D, m_Theta
        prob = m_D + 0.5 * m_Theta
        return np.clip(prob, 0.0, 1.0)

# =============================================================================
# 5. 参数修复：单调性 + 边界 
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
                    lesser = tuple(lesser)
                    if rb[combo] < rb[lesser]:
                        rb[combo] = rb[lesser]
                        changed = True
        if not changed: break
    return rb

def repair_parameters(params: np.ndarray) -> np.ndarray:
    # ★ E1-V2 核心修改：只处理 N_RULES 长度的 beliefs ★
    params = np.asarray(params, dtype=float)
    beliefs = np.clip(params[:N_RULES], 0.0, 1.0)
    
    idx = 0
    rb1 = {c: beliefs[idx + i] for i, c in enumerate(COMBOS1)}; idx += len(COMBOS1)
    rb2 = {c: beliefs[idx + i] for i, c in enumerate(COMBOS2)}; idx += len(COMBOS2)
    rb3 = {c: beliefs[idx + i] for i, c in enumerate(COMBOS3)}; idx += len(COMBOS3)
    rb_m = {c: beliefs[idx + i] for i, c in enumerate(COMBOS_M)}

    rb1 = enforce_monotonicity(rb1, COMBOS1)
    rb2 = enforce_monotonicity(rb2, COMBOS2)
    rb3 = enforce_monotonicity(rb3, COMBOS3)
    rb_m = enforce_monotonicity(rb_m, COMBOS_M)

    repaired_beliefs = []
    for c in COMBOS1: repaired_beliefs.append(rb1[c])
    for c in COMBOS2: repaired_beliefs.append(rb2[c])
    for c in COMBOS3: repaired_beliefs.append(rb3[c])
    for c in COMBOS_M: repaired_beliefs.append(rb_m[c])

    return np.clip(np.array(repaired_beliefs, dtype=float), 0.0, 1.0)

def linear_expert_params() -> np.ndarray:
    beliefs = []
    for combos in [COMBOS1, COMBOS2, COMBOS3, COMBOS_M]:
        for c in combos:
            if len(c) == 0: beliefs.append(0.0)
            else: beliefs.append(sum(c) / (2.0 * len(c)))
    # ★ E1-V2 核心修改：只返回 beliefs ★
    return repair_parameters(np.array(beliefs))

def build_initial_population() -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    base = linear_expert_params()
    population = [base]
    for _ in range(9): population.append(repair_parameters(base + rng.normal(0, 0.06, N_PARAMS)))
    for _ in range(9): population.append(repair_parameters(base + rng.normal(0, 0.15, N_PARAMS)))
    while len(population) < INIT_POP_SIZE: population.append(repair_parameters(rng.random(N_PARAMS)))
    return np.array(population, dtype=float)

# =============================================================================
# 6. CV 缓存与目标函数
# =============================================================================
CV_CACHE = []

def prepare_cv_cache(train_df: pd.DataFrame, y_train: pd.Series):
    global CV_CACHE
    CV_CACHE = []
    kf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_SEED)
    for tr_idx, val_idx in kf.split(train_df, y_train):
        df_tr = train_df.iloc[tr_idx].copy()
        df_val = train_df.iloc[val_idx].copy()
        y_val = y_train.iloc[val_idx].values
        model = EvolvableER_BRB(df_tr)
        CV_CACHE.append((model, df_val, y_val))
    print(f"CV 缓存完成：{N_SPLITS_CV} 折。")

BEST_LOSS = np.inf
N_EVALS = 0

def objective_cv(params: np.ndarray) -> float:
    global BEST_LOSS, N_EVALS
    N_EVALS += 1
    params = repair_parameters(params)
    fold_losses, fold_aucs, fold_aps, fold_briers = [], [], [], []

    for model, df_val, y_val in CV_CACHE:
        probs = model.predict(df_val, params)
        brier = brier_score_loss(y_val, probs)
        try: auc = roc_auc_score(y_val, probs)
        except ValueError: auc = 0.5
        try: ap = average_precision_score(y_val, probs)
        except ValueError: ap = float(np.mean(y_val))

        loss = W_ROC_AUC * (1.0 - auc) + W_PR_AUC * (1.0 - ap) + W_BRIER * brier
        fold_losses.append(loss); fold_aucs.append(auc); fold_aps.append(ap); fold_briers.append(brier)

    mean_loss = float(np.mean(fold_losses))
    if mean_loss < BEST_LOSS:
        BEST_LOSS = mean_loss
        print(f"[Best Update] eval={N_EVALS:05d} loss={BEST_LOSS:.6f} cv_auc={np.mean(fold_aucs):.4f} cv_pr_auc={np.mean(fold_aps):.4f} cv_brier={np.mean(fold_briers):.4f}")
    return mean_loss

# =============================================================================
# 7. 评估与主程序
# =============================================================================
def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    best_tau, best_f2 = 0.5, -1.0
    for tau in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= tau).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        if f2 > best_f2: best_f2 = f2; best_tau = float(tau)
    return best_tau, float(best_f2)

def evaluate_on_test(model: EvolvableER_BRB, test_df: pd.DataFrame, y_test: np.ndarray, params: np.ndarray):
    test_probs = model.predict(test_df, params)
    roc = roc_auc_score(y_test, test_probs)
    pr_auc = average_precision_score(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)
    best_tau, best_f2 = find_best_threshold(y_test, test_probs)
    m_D, m_Theta = model.predict(test_df, params, return_uncertainty=True)
    return {
        "ROC_AUC": roc, "PR_AUC": pr_auc, "Brier": brier, "F2": best_f2, "Tau": best_tau,
        "Mean_Uncertainty": float(np.mean(m_Theta)), "Median_Uncertainty": float(np.median(m_Theta)),
    }

def main():
    print("=" * 84)
    print("E1-V2 消融实验：无规则权重 (Unweighted ER)")
    print("=" * 84)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出结果将保存至目录: {OUTPUT_DIR}/")

    train_df, test_df = load_and_augment_data()
    y_train = train_df["label"]
    y_test = test_df["label"].values

    print("\n层级输入配置 (E1-V2)：")
    print(f"  Main inputs: {MAIN_INPUTS}")
    print(f"  Total rules: {N_RULES}")
    print(f"  Total params: {N_PARAMS} (仅优化 beta)")

    prepare_cv_cache(train_df, y_train)
    init_pop = build_initial_population()

    print("\n开始差分进化优化 (E1-V2) ...")
    result = differential_evolution(
        objective_cv, bounds=[(0.0, 1.0)] * N_PARAMS, init=init_pop,
        maxiter=MAXITER, mutation=(0.5, 1.0), recombination=0.7, tol=1e-5,
        seed=RANDOM_SEED, disp=True, polish=POLISH, workers=1, updating="immediate",
    )

    best_params = repair_parameters(result.x)
    final_model = EvolvableER_BRB(train_df)
    metrics = evaluate_on_test(final_model, test_df, y_test, best_params)

    print("\n" + "=" * 84)
    print("E1-V2 测试集性能 (无规则权重)")
    print("=" * 84)
    print(f"  ROC-AUC : {metrics['ROC_AUC']:.4f}")
    print(f"  PR-AUC  : {metrics['PR_AUC']:.4f}")
    print(f"  Brier   : {metrics['Brier']:.4f}")
    print(f"  F2      : {metrics['F2']:.4f}  (tau={metrics['Tau']:.3f})")
    
    np.save(PARAMS_OUTPUT, best_params)
    dim_info = {
        "version": "E1-V2 Unweighted ER",
        "MAIN_INPUTS": MAIN_INPUTS, "N_RULES": N_RULES, "N_PARAMS": N_PARAMS
    }
    np.save(DIM_INFO_OUTPUT, np.array(dim_info, dtype=object))
    pd.DataFrame([metrics]).to_csv(METRICS_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"\n✅ E1-V2 运行完毕。参数已保存至: {PARAMS_OUTPUT}")

if __name__ == "__main__":
    main()