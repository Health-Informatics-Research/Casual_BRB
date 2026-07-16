# =============================================================================
# nhanes_step7_e3_ablation_missing.py
# 【E3 消融实验 A2：基础因果 BRB (有层级 + 无权重 + 无不确定性传播)】
#
# 目的：填补消融实验大表的 A2 环节。
# 维度：主层无 Global_unc (3维输入) -> 144 条规则。
#       无权重 (theta全为1.0) -> 仅优化 144 维 beta。
# =============================================================================

import itertools
import warnings
import os
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
HIERARCHY_FILE = "brb_hierarchy_config.npy"

OUTPUT_DIR = "E3_Ablation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARAMS_OUTPUT = os.path.join(OUTPUT_DIR, "e3_a2_params.npy")
METRICS_OUTPUT = os.path.join(OUTPUT_DIR, "e3_a2_metrics.csv")

RANDOM_SEED = 42
N_SPLITS_CV = 3
MAXITER = 120
INIT_POP_SIZE = 28
POLISH = False

W_ROC_AUC = 0.45
W_PR_AUC = 0.25
W_BRIER = 0.30

# =============================================================================
# 1. 层级配置 (无 Global_unc)
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

MAIN_INPUTS = ["Sub1_mid", "Sub2_mid", "Sub3_mid"]

COMBOS1 = list(itertools.product([0, 1, 2], repeat=len(SUB1_INPUTS)))
COMBOS2 = list(itertools.product([0, 1, 2], repeat=len(SUB2_INPUTS)))
COMBOS3 = list(itertools.product([0, 1, 2], repeat=len(SUB3_INPUTS)))
COMBOS_M = list(itertools.product([0, 1, 2], repeat=len(MAIN_INPUTS)))

N_RULES = len(COMBOS1) + len(COMBOS2) + len(COMBOS3) + len(COMBOS_M) # 81 + 9 + 27 + 27 = 144
N_PARAMS = N_RULES  # 只有 beta，无 theta

# =============================================================================
# 2. ER 推理函数
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
    a0[idx1] = (c1 - values[idx1]) / (c1 - c0); a1[idx1] = (values[idx1] - c0) / (c1 - c0)
    idx2 = (values > c1) & (values <= c2)
    a1[idx2] = (c2 - values[idx2]) / (c2 - c1); a2[idx2] = (values[idx2] - c1) / (c2 - c1)
    idx3 = values > c2; a2[idx3] = 1.0
    return a0, a1, a2

def ER_fusion(activation: np.ndarray, rule_beliefs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_samples, n_rules = activation.shape
    m_D, m_notD, m_Theta = np.zeros(n_samples), np.zeros(n_samples), np.ones(n_samples)

    for k in range(n_rules):
        w = activation[:, k]
        beta = rule_beliefs[k]
        theta = 1.0  # 强制无权重

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

class EvolvableER_BRB_A2:
    def __init__(self, df_train: pd.DataFrame):
        self.anchors_sub1 = compute_anchors(df_train, SUB1_INPUTS)
        self.anchors_sub2 = compute_anchors(df_train, SUB2_INPUTS)
        self.anchors_sub3 = compute_anchors(df_train, SUB3_INPUTS)

        dummy_rb1 = {c: 0.5 for c in COMBOS1}; s1_m, _ = self._infer_sub(df_train, SUB1_INPUTS, self.anchors_sub1, COMBOS1, dummy_rb1)
        dummy_rb2 = {c: 0.5 for c in COMBOS2}; s2_m, _ = self._infer_sub(df_train, SUB2_INPUTS, self.anchors_sub2, COMBOS2, dummy_rb2)
        dummy_rb3 = {c: 0.5 for c in COMBOS3}; s3_m, _ = self._infer_sub(df_train, SUB3_INPUTS, self.anchors_sub3, COMBOS3, dummy_rb3)

        main_df = pd.DataFrame({"Sub1_mid": s1_m, "Sub2_mid": s2_m, "Sub3_mid": s3_m})
        self.anchors_main = compute_anchors(main_df, MAIN_INPUTS)

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

    def _infer_sub(self, df, input_cols, anchors, combos, rb):
        act = self._activation_matrix(df, input_cols, anchors, combos)
        m_D, m_Theta = ER_fusion(act, np.array([rb[c] for c in combos]))
        return np.clip(m_D + 0.5 * m_Theta, 0.0, 1.0), np.clip(m_Theta, 0.0, 1.0)

    def split_params(self, params: np.ndarray):
        b = params[:N_RULES]
        idx = 0
        rb1 = {c: b[idx+i] for i,c in enumerate(COMBOS1)}; idx += len(COMBOS1)
        rb2 = {c: b[idx+i] for i,c in enumerate(COMBOS2)}; idx += len(COMBOS2)
        rb3 = {c: b[idx+i] for i,c in enumerate(COMBOS3)}; idx += len(COMBOS3)
        rb_m = {c: b[idx+i] for i,c in enumerate(COMBOS_M)}
        return rb1, rb2, rb3, rb_m

    def predict(self, df: pd.DataFrame, params: np.ndarray):
        rb1, rb2, rb3, rb_m = self.split_params(params)
        s1_m, _ = self._infer_sub(df, SUB1_INPUTS, self.anchors_sub1, COMBOS1, rb1)
        s2_m, _ = self._infer_sub(df, SUB2_INPUTS, self.anchors_sub2, COMBOS2, rb2)
        s3_m, _ = self._infer_sub(df, SUB3_INPUTS, self.anchors_sub3, COMBOS3, rb3)
        main_df = pd.DataFrame({"Sub1_mid": s1_m, "Sub2_mid": s2_m, "Sub3_mid": s3_m})
        act = self._activation_matrix(main_df, MAIN_INPUTS, self.anchors_main, COMBOS_M)
        m_D, m_Theta = ER_fusion(act, np.array([rb_m[c] for c in COMBOS_M]))
        return np.clip(m_D + 0.5 * m_Theta, 0.0, 1.0)

# =============================================================================
# 3. 修复、目标与主流程
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

def repair_parameters(params: np.ndarray) -> np.ndarray:
    b = np.clip(np.asarray(params, dtype=float), 0.0, 1.0)
    idx = 0
    rb1 = {c: b[idx+i] for i,c in enumerate(COMBOS1)}; idx += len(COMBOS1)
    rb2 = {c: b[idx+i] for i,c in enumerate(COMBOS2)}; idx += len(COMBOS2)
    rb3 = {c: b[idx+i] for i,c in enumerate(COMBOS3)}; idx += len(COMBOS3)
    rb_m = {c: b[idx+i] for i,c in enumerate(COMBOS_M)}

    rb1 = enforce_monotonicity(rb1, COMBOS1)
    rb2 = enforce_monotonicity(rb2, COMBOS2)
    rb3 = enforce_monotonicity(rb3, COMBOS3)
    rb_m = enforce_monotonicity(rb_m, COMBOS_M)

    rep = []
    for c in COMBOS1: rep.append(rb1[c])
    for c in COMBOS2: rep.append(rb2[c])
    for c in COMBOS3: rep.append(rb3[c])
    for c in COMBOS_M: rep.append(rb_m[c])
    return np.clip(rep, 0.0, 1.0)

CV_CACHE = []
def prepare_cv_cache(train_df: pd.DataFrame, y_train: pd.Series):
    global CV_CACHE
    kf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_SEED)
    for tr_idx, val_idx in kf.split(train_df, y_train):
        model = EvolvableER_BRB_A2(train_df.iloc[tr_idx].copy())
        CV_CACHE.append((model, train_df.iloc[val_idx].copy(), y_train.iloc[val_idx].values))

def objective_cv(params: np.ndarray) -> float:
    params = repair_parameters(params)
    losses = []
    for model, df_val, y_val in CV_CACHE:
        probs = model.predict(df_val, params)
        brier = brier_score_loss(y_val, probs)
        try: auc = roc_auc_score(y_val, probs)
        except: auc = 0.5
        try: ap = average_precision_score(y_val, probs)
        except: ap = float(np.mean(y_val))
        losses.append(W_ROC_AUC * (1.0 - auc) + W_PR_AUC * (1.0 - ap) + W_BRIER * brier)
    return float(np.mean(losses))

def find_best_threshold(y_true, y_prob):
    best_tau, best_f2 = 0.5, -1.0
    for tau in np.linspace(0.05, 0.95, 91):
        f2 = fbeta_score(y_true, (y_prob >= tau).astype(int), beta=2, zero_division=0)
        if f2 > best_f2: best_f2 = f2; best_tau = tau
    return best_tau, best_f2

def get_A0_metrics(train_df, test_df):
    path_cols = ["P1_Socioeconomic_prob", "P2_Sleep_prob", "P3_HealthStatus_prob", 
                 "P4_FoodSecurity_prob", "P5_Clinical_prob", "P6_Substance_prob"]
    y_test = test_df["label"].values
    probs = test_df[path_cols].mean(axis=1).values
    tau, f2 = find_best_threshold(y_test, probs)
    return {
        "ROC_AUC": roc_auc_score(y_test, probs),
        "PR_AUC": average_precision_score(y_test, probs),
        "Brier": brier_score_loss(y_test, probs),
        "F2": f2, "Tau": tau
    }

def main():
    print("=" * 70)
    print("E3 消融实验 A2：基础因果 BRB (仅144维参数)")
    print("=" * 70)
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    scaler = MinMaxScaler()
    cols = ["RIDAGEYR", "RIAGENDR", "DMDMARTL"]
    train_df[["Age_prob", "Gender_prob", "DMDMARTL_prob"]] = scaler.fit_transform(train_df[cols])
    test_df[["Age_prob", "Gender_prob", "DMDMARTL_prob"]] = scaler.transform(test_df[cols])

    # 先计算 A0 (基线平均)
    a0_metrics = get_A0_metrics(train_df, test_df)
    print("\n[A0] 路径概率硬平均 (基线):")
    print(f"ROC-AUC: {a0_metrics['ROC_AUC']:.4f}, Brier: {a0_metrics['Brier']:.4f}")

    prepare_cv_cache(train_df, train_df["label"])

    rng = np.random.default_rng(RANDOM_SEED)
    init_pop = [repair_parameters(np.full(N_PARAMS, 0.5))]
    while len(init_pop) < INIT_POP_SIZE: init_pop.append(repair_parameters(rng.random(N_PARAMS)))

    print(f"\n开始差分进化 (参数维度: {N_PARAMS}) ...")
    result = differential_evolution(
        objective_cv, bounds=[(0.0, 1.0)] * N_PARAMS, init=init_pop,
        maxiter=MAXITER, mutation=(0.5, 1.0), recombination=0.7, tol=1e-5,
        seed=RANDOM_SEED, disp=True, workers=1, updating="immediate",
        polish=False
    )

    best_params = repair_parameters(result.x)
    final_model = EvolvableER_BRB_A2(train_df)
    test_probs = final_model.predict(test_df, best_params)
    
    y_test = test_df["label"].values
    tau, f2 = find_best_threshold(y_test, test_probs)
    
    metrics = {
        "ROC_AUC": roc_auc_score(y_test, test_probs),
        "PR_AUC": average_precision_score(y_test, test_probs),
        "Brier": brier_score_loss(y_test, test_probs),
        "F2": f2, "Tau": tau
    }

    print("\n[A2] 基础因果 BRB (无权重, 无传播) 测试结果:")
    print(f"ROC-AUC : {metrics['ROC_AUC']:.4f}")
    print(f"PR-AUC  : {metrics['PR_AUC']:.4f}")
    print(f"Brier   : {metrics['Brier']:.4f}")
    print(f"F2      : {metrics['F2']:.4f}")

    np.save(PARAMS_OUTPUT, best_params)
    pd.DataFrame([metrics]).to_csv(METRICS_OUTPUT, index=False)

if __name__ == "__main__":
    main()