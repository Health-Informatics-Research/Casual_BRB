# =============================================================================
# nhanes_step5_ml_baselines.py
# 【最终修正版：适配 Step 4 Global_unc 396维参数 + 校准防泄露 + Anchor一致性修复V2】
#
# 关键修复：
#   Step 4 最终测试时使用完整 train_df 构建 BRB 锚点；
#   所以 Step 5 中 BRB 也必须使用完整 base_train 构建锚点：
#
#       brb = ER_BRB_Predictor(base_train, best_params)
#
#   不能使用：
#
#       brb = ER_BRB_Predictor(df_fit, best_params)
#
#   否则 Sub/Main 层锚点改变，隶属度与规则激活分布都会改变，
#   会导致 Step 5 的 BRB AUC 从 Step 4 的 0.80 附近异常下降。
#
# 使用前提：
#   1. 已运行 Step 2，生成：
#        nhanes_brb_train.csv
#        nhanes_brb_test.csv
#   2. 已运行最终版 Step 4，生成：
#        best_brb_er_params.npy
#        brb_main_dim_info.npy
#
# 本脚本输出：
#   model_comparison_results.csv
#   brb_belief_interval_samples.csv
#   brb_belief_interval_group_analysis.csv
#   brb_calibration_comparison.csv
# =============================================================================

import itertools
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


# =============================================================================
# 0. 全局参数
# =============================================================================
PARAMS_FILE = "best_brb_er_params.npy"
DIM_INFO_FILE = "brb_main_dim_info.npy"
HIERARCHY_FILE = "brb_hierarchy_config.npy"

TRAIN_CSV = "nhanes_brb_train.csv"
TEST_CSV = "nhanes_brb_test.csv"
RAW_CSV = "nhanes_clean_original.csv"

RANDOM_SEED = 42

RESULTS_OUTPUT = "model_comparison_results.csv"
BI_SAMPLE_OUTPUT = "brb_belief_interval_samples.csv"
BI_GROUP_OUTPUT = "brb_belief_interval_group_analysis.csv"
CALIBRATION_OUTPUT = "brb_calibration_comparison.csv"

FEATURE_COLS_BRB = [
    "P1_Socioeconomic_prob",
    "P2_Sleep_prob",
    "P3_HealthStatus_prob",
    "P4_FoodSecurity_prob",
    "P5_Clinical_prob",
    "P6_Substance_prob",
    "Age_prob",
    "Gender_prob",
    "DMDMARTL_prob",
]


# =============================================================================
# 1. 层级结构与规则空间：必须与 Step 4 Global_unc 版本一致
# =============================================================================
def load_hierarchy() -> Dict[str, List[str]]:
    try:
        hierarchy = np.load(HIERARCHY_FILE, allow_pickle=True).item()
        print("加载因果层级配置:", hierarchy)
    except FileNotFoundError:
        hierarchy = {
            "Sub1_background": ["P1_Socioeconomic"],
            "Sub2_behavioral": ["P2_Sleep", "P3_HealthStatus"],
            "Sub3_stress": ["P4_FoodSecurity", "P5_Clinical", "P6_Substance"],
        }
        print("未找到 brb_hierarchy_config.npy，使用默认层级配置:", hierarchy)
    return hierarchy


HIERARCHY = load_hierarchy()

SUB1_INPUTS = [f"{p}_prob" for p in HIERARCHY["Sub1_background"]] + [
    "Age_prob",
    "Gender_prob",
    "DMDMARTL_prob",
]
SUB2_INPUTS = [f"{p}_prob" for p in HIERARCHY["Sub2_behavioral"]]
SUB3_INPUTS = [f"{p}_prob" for p in HIERARCHY["Sub3_stress"]]

# Step 4 最终论文推荐版：4维主层输入
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
N_PARAMS = N_RULES * 2


# =============================================================================
# 2. 数据加载
# =============================================================================
def load_and_augment_brb_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("加载 BRB 路径特征，并扩充 Age/Gender/DMDMARTL ...")

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    required_cols = ["RIDAGEYR", "RIAGENDR", "DMDMARTL", "label"]
    for name, df in [("train", train_df), ("test", test_df)]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} 数据缺少必要列: {missing}")

    scaler = MinMaxScaler()
    scale_cols = ["RIDAGEYR", "RIAGENDR", "DMDMARTL"]

    train_df[["Age_prob", "Gender_prob", "DMDMARTL_prob"]] = scaler.fit_transform(
        train_df[scale_cols]
    )
    test_df[["Age_prob", "Gender_prob", "DMDMARTL_prob"]] = scaler.transform(
        test_df[scale_cols]
    )

    missing_brb = [c for c in FEATURE_COLS_BRB if c not in train_df.columns]
    if missing_brb:
        raise ValueError(f"BRB 训练特征缺失: {missing_brb}")

    return train_df, test_df


def load_raw_ml_data(base_train: pd.DataFrame, base_test: pd.DataFrame):
    """
    构造黑盒模型所需数据：
      1. BRB Pathways (9-Dim)
      2. Raw Features
    """
    if not os.path.exists(RAW_CSV):
        print(f"未找到 {RAW_CSV}，将只比较 BRB Pathways (9-Dim) 特征。")
        return base_train.copy(), base_test.copy(), []

    raw_df = pd.read_csv(RAW_CSV)

    exclude_cols = [
        "SEQN",
        "CYCLE",
        "depression_label",
        "PHQ9_score",
        "WTINT2YR",
        "WTMEC2YR",
        "WTMEC4YR",
        "SDMVPSU",
        "SDMVSTRA",
        "label",
    ]

    raw_features = [c for c in raw_df.columns if c not in exclude_cols]

    keep_cols = ["SEQN", "label"] + [c for c in base_train.columns if c.endswith("_prob")]

    ml_train_df = pd.merge(base_train[keep_cols], raw_df, on="SEQN", how="left")
    ml_test_df = pd.merge(base_test[keep_cols], raw_df, on="SEQN", how="left")

    for col in raw_features:
        if col in ml_train_df.columns:
            med = ml_train_df[col].median()
            ml_train_df[col] = ml_train_df[col].fillna(med)
            ml_test_df[col] = ml_test_df[col].fillna(med)

    return ml_train_df, ml_test_df, raw_features


# =============================================================================
# 3. ER-BRB 推理核心：与 Step 4 Global_unc 版本保持一致
# =============================================================================
def compute_anchors(df: pd.DataFrame, cols: List[str]) -> Dict[str, Tuple[float, float, float]]:
    anchors = {}

    for col in cols:
        if col not in df.columns:
            raise ValueError(f"缺少输入列: {col}")

        vals = df[col].astype(float).values
        c0, c1, c2 = np.percentile(vals, [16, 50, 84])

        if c1 <= c0:
            c1 = c0 + 1e-5
        if c2 <= c1:
            c2 = c1 + 1e-5

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


def ER_fusion(
    activation: np.ndarray,
    rule_beliefs: np.ndarray,
    rule_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
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
        m_notD_new = (
            m_notD * m_k_notD + m_notD * m_k_Theta + m_Theta * m_k_notD
        ) / denom
        m_Theta_new = (m_Theta * m_k_Theta) / denom

        m_D, m_notD, m_Theta = m_D_new, m_notD_new, m_Theta_new

    return np.clip(m_D, 0.0, 1.0), np.clip(m_Theta, 0.0, 1.0)


class ER_BRB_Predictor:
    def __init__(self, df_train_for_anchors: pd.DataFrame, best_params: np.ndarray):
        if len(best_params) != N_PARAMS:
            raise ValueError(
                f"参数维度不匹配：当前 Step 5 期望 {N_PARAMS} 维，"
                f"但 best_brb_er_params.npy 是 {len(best_params)} 维。\n"
                f"请确认你运行的是 Step 4 Global_unc 最终版，而不是旧的 288维或1692维版本。"
            )

        self.best_params = best_params

        self.anchors_sub1 = compute_anchors(df_train_for_anchors, SUB1_INPUTS)
        self.anchors_sub2 = compute_anchors(df_train_for_anchors, SUB2_INPUTS)
        self.anchors_sub3 = compute_anchors(df_train_for_anchors, SUB3_INPUTS)

        (
            self.rb1,
            self.rw1,
            self.rb2,
            self.rw2,
            self.rb3,
            self.rw3,
            self.rb_m,
            self.rw_m,
        ) = self._split_params(best_params)

        # ============================================================
        # 【关键修复 V2】
        # Step 4 中 EvolvableER_BRB.__init__ 计算 Main 层 anchors 时，
        # 使用的是 dummy 参数：
        #   beta = 0.5
        #   theta = 1.0
        # 而不是优化后的 best_params。
        #
        # 如果 Step 5 这里用 best_params 计算 Main anchors，
        # 即使 base_train 一致，也会导致主层隶属度空间改变，
        # 进而出现 Step 4 AUC≈0.80、Step 5 AUC≈0.69 的崩溃。
        #
        # 因此这里必须严格复刻 Step 4：
        #   Sub anchors: base_train 分位点
        #   Main anchors: dummy beta/theta 前向传播后再取分位点
        # ============================================================
        dummy_rb1 = {c: 0.5 for c in COMBOS1}
        dummy_rw1 = {c: 1.0 for c in COMBOS1}
        s1_mid, s1_unc = self._infer_sub(
            df_train_for_anchors, SUB1_INPUTS, self.anchors_sub1, COMBOS1, dummy_rb1, dummy_rw1
        )

        dummy_rb2 = {c: 0.5 for c in COMBOS2}
        dummy_rw2 = {c: 1.0 for c in COMBOS2}
        s2_mid, s2_unc = self._infer_sub(
            df_train_for_anchors, SUB2_INPUTS, self.anchors_sub2, COMBOS2, dummy_rb2, dummy_rw2
        )

        dummy_rb3 = {c: 0.5 for c in COMBOS3}
        dummy_rw3 = {c: 1.0 for c in COMBOS3}
        s3_mid, s3_unc = self._infer_sub(
            df_train_for_anchors, SUB3_INPUTS, self.anchors_sub3, COMBOS3, dummy_rb3, dummy_rw3
        )

        main_df = self._build_main_df(s1_mid, s2_mid, s3_mid, s1_unc, s2_unc, s3_unc)
        self.anchors_main = compute_anchors(main_df, MAIN_INPUTS)

    @staticmethod
    def _build_main_df(
        s1_mid: np.ndarray,
        s2_mid: np.ndarray,
        s3_mid: np.ndarray,
        s1_unc: np.ndarray,
        s2_unc: np.ndarray,
        s3_unc: np.ndarray,
    ) -> pd.DataFrame:
        global_unc = (s1_unc + s2_unc + s3_unc) / 3.0

        return pd.DataFrame(
            {
                "Sub1_mid": s1_mid,
                "Sub2_mid": s2_mid,
                "Sub3_mid": s3_mid,
                "Global_unc": global_unc,
            }
        )

    def _split_params(self, params: np.ndarray):
        beliefs = params[:N_RULES]
        weights = params[N_RULES:]

        idx = 0

        rb1 = {c: beliefs[idx + i] for i, c in enumerate(COMBOS1)}
        rw1 = {c: weights[idx + i] for i, c in enumerate(COMBOS1)}
        idx += len(COMBOS1)

        rb2 = {c: beliefs[idx + i] for i, c in enumerate(COMBOS2)}
        rw2 = {c: weights[idx + i] for i, c in enumerate(COMBOS2)}
        idx += len(COMBOS2)

        rb3 = {c: beliefs[idx + i] for i, c in enumerate(COMBOS3)}
        rw3 = {c: weights[idx + i] for i, c in enumerate(COMBOS3)}
        idx += len(COMBOS3)

        rb_m = {c: beliefs[idx + i] for i, c in enumerate(COMBOS_M)}
        rw_m = {c: weights[idx + i] for i, c in enumerate(COMBOS_M)}

        return rb1, rw1, rb2, rw2, rb3, rw3, rb_m, rw_m

    def _activation_matrix(
        self,
        df: pd.DataFrame,
        input_cols: List[str],
        anchors: Dict[str, Tuple[float, float, float]],
        combos: List[Tuple[int, ...]],
    ) -> np.ndarray:
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

    def _infer_sub(
        self,
        df: pd.DataFrame,
        input_cols: List[str],
        anchors: Dict[str, Tuple[float, float, float]],
        combos: List[Tuple[int, ...]],
        rb: Dict[Tuple[int, ...], float],
        rw: Dict[Tuple[int, ...], float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        activation = self._activation_matrix(df, input_cols, anchors, combos)
        beliefs = np.array([rb[c] for c in combos], dtype=float)
        weights = np.array([rw[c] for c in combos], dtype=float)

        m_D, m_Theta = ER_fusion(activation, beliefs, weights)

        sub_mid = m_D + 0.5 * m_Theta
        sub_unc = m_Theta

        return np.clip(sub_mid, 0.0, 1.0), np.clip(sub_unc, 0.0, 1.0)

    def _infer_main(self, main_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        activation = self._activation_matrix(main_df, MAIN_INPUTS, self.anchors_main, COMBOS_M)
        beliefs = np.array([self.rb_m[c] for c in COMBOS_M], dtype=float)
        weights = np.array([self.rw_m[c] for c in COMBOS_M], dtype=float)

        return ER_fusion(activation, beliefs, weights)

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        m_D, m_Theta = self.predict_with_uncertainty(df)
        return np.clip(m_D + 0.5 * m_Theta, 0.0, 1.0)

    def predict_with_uncertainty(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        s1_mid, s1_unc = self._infer_sub(
            df, SUB1_INPUTS, self.anchors_sub1, COMBOS1, self.rb1, self.rw1
        )
        s2_mid, s2_unc = self._infer_sub(
            df, SUB2_INPUTS, self.anchors_sub2, COMBOS2, self.rb2, self.rw2
        )
        s3_mid, s3_unc = self._infer_sub(
            df, SUB3_INPUTS, self.anchors_sub3, COMBOS3, self.rb3, self.rw3
        )

        main_df = self._build_main_df(s1_mid, s2_mid, s3_mid, s1_unc, s2_unc, s3_unc)
        m_D, m_Theta = self._infer_main(main_df)

        return m_D, m_Theta


# =============================================================================
# 4. 指标、阈值、校准
# =============================================================================
def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return np.nan


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return np.nan


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 2.0):
    best_tau = 0.5
    best_f = -1.0

    for tau in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= tau).astype(int)
        f = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

        if f > best_f:
            best_f = float(f)
            best_tau = float(tau)

    return best_tau, best_f


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
    feature_name: str,
    note: str = "",
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    return {
        "Model": model_name,
        "Features": feature_name,
        "Note": note,
        "Threshold": float(threshold),
        "ROC-AUC": safe_auc(y_true, y_prob),
        "PR-AUC": safe_ap(y_true, y_prob),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "F2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Recall/Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(specificity),
        "Precision/PPV": float(precision_score(y_true, y_pred, zero_division=0)),
        "NPV": float(npv),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "N": int(len(y_true)),
        "Positive_Rate": float(np.mean(y_true)),
        "Predicted_Positive_Rate": float(np.mean(y_pred)),
    }


def fit_platt_calibrator(raw_prob_cal: np.ndarray, y_cal: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(C=1e10, solver="lbfgs", random_state=RANDOM_SEED, max_iter=1000)
    lr.fit(raw_prob_cal.reshape(-1, 1), y_cal)
    return lr


def apply_platt(calibrator: LogisticRegression, raw_prob: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(raw_prob.reshape(-1, 1))[:, 1]


def fit_isotonic_calibrator(raw_prob_cal: np.ndarray, y_cal: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_prob_cal, y_cal)
    return iso


def apply_isotonic(calibrator: IsotonicRegression, raw_prob: np.ndarray) -> np.ndarray:
    return np.clip(calibrator.predict(raw_prob), 0.0, 1.0)


# =============================================================================
# 5. Belief Interval 分析
# =============================================================================
def belief_interval_analysis(
    brb: ER_BRB_Predictor,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    raw_prob: np.ndarray,
    calibrated_prob: np.ndarray,
):
    m_D, m_Theta = brb.predict_with_uncertainty(test_df)

    bel = m_D
    pl = np.clip(m_D + m_Theta, 0.0, 1.0)
    width = pl - bel

    sample_df = pd.DataFrame(
        {
            "SEQN": test_df["SEQN"].values if "SEQN" in test_df.columns else np.arange(len(test_df)),
            "y_true": y_test,
            "BRB_raw_prob": raw_prob,
            "BRB_calibrated_prob": calibrated_prob,
            "Bel_D": bel,
            "Pl_D": pl,
            "Interval_Width": width,
            "m_Theta": m_Theta,
        }
    )

    q_low, q_high = np.percentile(width, [33.33, 66.67])

    def group_name(x):
        if x <= q_low:
            return "Low uncertainty"
        if x <= q_high:
            return "Medium uncertainty"
        return "High uncertainty"

    sample_df["Uncertainty_Group"] = sample_df["Interval_Width"].apply(group_name)

    group_rows = []
    for g in ["Low uncertainty", "Medium uncertainty", "High uncertainty"]:
        sub = sample_df[sample_df["Uncertainty_Group"] == g].copy()
        yt = sub["y_true"].values
        pr_raw = sub["BRB_raw_prob"].values
        pr_cal = sub["BRB_calibrated_prob"].values

        if len(np.unique(yt)) > 1:
            tau_g, f2_g = find_best_threshold(yt, pr_cal)
            raw_auc = safe_auc(yt, pr_raw)
            cal_auc = safe_auc(yt, pr_cal)
            cal_ap = safe_ap(yt, pr_cal)
        else:
            tau_g, f2_g = np.nan, np.nan
            raw_auc, cal_auc, cal_ap = np.nan, np.nan, np.nan

        group_rows.append(
            {
                "Group": g,
                "N": len(sub),
                "Positive_Rate": float(sub["y_true"].mean()) if len(sub) else np.nan,
                "Mean_Bel": float(sub["Bel_D"].mean()) if len(sub) else np.nan,
                "Mean_Pl": float(sub["Pl_D"].mean()) if len(sub) else np.nan,
                "Mean_Interval_Width": float(sub["Interval_Width"].mean()) if len(sub) else np.nan,
                "Median_Interval_Width": float(sub["Interval_Width"].median()) if len(sub) else np.nan,
                "Raw_ROC_AUC": raw_auc,
                "Cal_ROC_AUC": cal_auc,
                "Cal_PR_AUC": cal_ap,
                "Cal_Brier": float(brier_score_loss(yt, pr_cal)) if len(sub) else np.nan,
                "Cal_F2_Best": f2_g,
                "Cal_Tau_Best": tau_g,
            }
        )

    group_df = pd.DataFrame(group_rows)

    sample_df.to_csv(BI_SAMPLE_OUTPUT, index=False, encoding="utf-8-sig")
    group_df.to_csv(BI_GROUP_OUTPUT, index=False, encoding="utf-8-sig")

    return sample_df, group_df


def calibration_bins(y_true: np.ndarray, raw_prob: np.ndarray, platt_prob: np.ndarray, iso_prob: np.ndarray):
    rows = []

    for name, prob in [
        ("Raw", raw_prob),
        ("Platt", platt_prob),
        ("Isotonic", iso_prob),
    ]:
        df = pd.DataFrame({"y": y_true, "p": prob})
        df["bin"] = pd.qcut(df["p"], q=10, duplicates="drop")

        for b, sub in df.groupby("bin", observed=True):
            rows.append(
                {
                    "Calibration": name,
                    "Bin": str(b),
                    "N": len(sub),
                    "Mean_Predicted_Prob": float(sub["p"].mean()),
                    "Observed_Positive_Rate": float(sub["y"].mean()),
                }
            )

    cal_df = pd.DataFrame(rows)
    cal_df.to_csv(CALIBRATION_OUTPUT, index=False, encoding="utf-8-sig")
    return cal_df


# =============================================================================
# 6. 主流程
# =============================================================================
def main():
    print("=" * 90)
    print("Step 5: 模型对比【Global_unc ER-BRB + 校准防泄露 + Anchor一致性修复V2】")
    print("=" * 90)

    if not os.path.exists(PARAMS_FILE):
        raise FileNotFoundError(f"未找到 {PARAMS_FILE}，请先运行 Step 4。")

    best_params = np.load(PARAMS_FILE)

    print("\nStep 5 规则配置：")
    print(f"  Main inputs : {MAIN_INPUTS}")
    print(f"  N_RULES     : {N_RULES}")
    print(f"  N_PARAMS    : {N_PARAMS}")
    print(f"  params file : {len(best_params)} dims")

    if os.path.exists(DIM_INFO_FILE):
        dim_info = np.load(DIM_INFO_FILE, allow_pickle=True).item()
        print("  Step 4 dim info:", dim_info)

    if len(best_params) != N_PARAMS:
        raise ValueError(
            f"参数维度错误：Step 5 期望 {N_PARAMS} 维，但读取到 {len(best_params)} 维。\n"
            "请重新运行最终版 Step 4，确保主层是 [Sub1_mid, Sub2_mid, Sub3_mid, Global_unc]。"
        )

    base_train, base_test = load_and_augment_brb_data()
    y_test = base_test["label"].values

    # 三路划分：
    # df_cal 只训练校准器；
    # df_val 只选阈值；
    # base_test 只做最终测试。
    # 注意：BRB anchor 不用 df_fit，而用完整 base_train，与 Step 4 保持一致。
    df_fit, df_temp = train_test_split(
        base_train,
        test_size=0.4,
        stratify=base_train["label"],
        random_state=RANDOM_SEED,
    )
    df_cal, df_val = train_test_split(
        df_temp,
        test_size=0.5,
        stratify=df_temp["label"],
        random_state=RANDOM_SEED,
    )

    print("\n三路划分：")
    print(f"  Fit set: {len(df_fit)}")
    print(f"  Cal set: {len(df_cal)}")
    print(f"  Val set: {len(df_val)}")
    print(f"  Test set: {len(base_test)}")

    results = []

    # =========================================================================
    # A. ER-BRB Raw / Platt / Isotonic
    # =========================================================================
    print("\n[A] 评估 ER-BRB Raw / Platt / Isotonic ...")

    # ================================
    # 【最关键修复】
    # 用完整 base_train 构建 BRB 锚点，与 Step 4 最终测试保持一致。
    # 不要改成 df_fit。
    # ================================
    brb = ER_BRB_Predictor(base_train, best_params)

    raw_cal_prob = brb.predict_raw(df_cal)
    raw_val_prob = brb.predict_raw(df_val)
    raw_test_prob = brb.predict_raw(base_test)

    raw_tau, _ = find_best_threshold(df_val["label"].values, raw_val_prob)
    results.append(
        compute_metrics(
            y_test,
            raw_test_prob,
            raw_tau,
            "ER Causal-BRB (Raw)",
            "BRB Pathways + Global_unc",
            "Anchor uses full training set; threshold selected on validation set",
        )
    )

    # Platt 校准：只在 cal 上训练，val 上找阈值，test 上评估
    platt = fit_platt_calibrator(raw_cal_prob, df_cal["label"].values)
    platt_val_prob = apply_platt(platt, raw_val_prob)
    platt_test_prob = apply_platt(platt, raw_test_prob)

    platt_tau, _ = find_best_threshold(df_val["label"].values, platt_val_prob)
    results.append(
        compute_metrics(
            y_test,
            platt_test_prob,
            platt_tau,
            "ER Causal-BRB (Platt calibrated)",
            "BRB Pathways + Global_unc",
            "Anchor uses full training set; calibration trained on independent calibration set",
        )
    )

    # Isotonic 校准：只在 cal 上训练，val 上找阈值，test 上评估
    iso = fit_isotonic_calibrator(raw_cal_prob, df_cal["label"].values)
    iso_val_prob = apply_isotonic(iso, raw_val_prob)
    iso_test_prob = apply_isotonic(iso, raw_test_prob)

    iso_tau, _ = find_best_threshold(df_val["label"].values, iso_val_prob)
    results.append(
        compute_metrics(
            y_test,
            iso_test_prob,
            iso_tau,
            "ER Causal-BRB (Isotonic calibrated)",
            "BRB Pathways + Global_unc",
            "Anchor uses full training set; calibration trained on independent calibration set",
        )
    )

    calibration_bins(y_test, raw_test_prob, platt_test_prob, iso_test_prob)

    # 默认把 Platt 作为主校准结果，用于 Belief Interval 分析
    _, bi_group_df = belief_interval_analysis(
        brb=brb,
        test_df=base_test,
        y_test=y_test,
        raw_prob=raw_test_prob,
        calibrated_prob=platt_test_prob,
    )

    # =========================================================================
    # B. 黑盒模型对比
    # =========================================================================
    print("\n[B] 评估黑盒基线模型 ...")

    ml_train_df, ml_test_df, raw_features = load_raw_ml_data(base_train, base_test)

    # 黑盒模型继续用 fit/val/test 划分，这是合理的；
    # 只有 BRB anchor 需要与 Step 4 保持一致。
    fit_seqn = set(df_fit["SEQN"].values) if "SEQN" in df_fit.columns else None
    val_seqn = set(df_val["SEQN"].values) if "SEQN" in df_val.columns else None

    if fit_seqn is not None and "SEQN" in ml_train_df.columns:
        ml_fit_df = ml_train_df[ml_train_df["SEQN"].isin(fit_seqn)].copy()
        ml_val_df = ml_train_df[ml_train_df["SEQN"].isin(val_seqn)].copy()
    else:
        y_train_ml = ml_train_df["label"].values
        ml_fit_df, ml_val_df = train_test_split(
            ml_train_df,
            test_size=0.2,
            stratify=y_train_ml,
            random_state=RANDOM_SEED,
        )

    baselines = [
        (
            "Logistic Regression",
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        ),
    ]

    if HAS_LIGHTGBM:
        baselines.append(
            (
                "LightGBM",
                LGBMClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.03,
                    num_leaves=15,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    verbose=-1,
                ),
            )
        )
    else:
        print("未检测到 lightgbm，跳过 LightGBM。")

    feature_sets = [("BRB Pathways (9-Dim)", FEATURE_COLS_BRB)]
    if raw_features:
        feature_sets.append(("Raw Features", raw_features))

    for feat_name, cols in feature_sets:
        available_cols = [c for c in cols if c in ml_fit_df.columns and c in ml_test_df.columns]
        if not available_cols:
            print(f"跳过 {feat_name}：没有可用特征。")
            continue

        for model_name, model in baselines:
            m = clone(model)

            X_fit = ml_fit_df[available_cols].copy()
            y_fit = ml_fit_df["label"].values
            X_val = ml_val_df[available_cols].copy()
            y_val = ml_val_df["label"].values
            X_test = ml_test_df[available_cols].copy()

            med = X_fit.median(numeric_only=True)
            X_fit = X_fit.fillna(med)
            X_val = X_val.fillna(med)
            X_test = X_test.fillna(med)

            m.fit(X_fit, y_fit)

            val_prob = m.predict_proba(X_val)[:, 1]
            test_prob = m.predict_proba(X_test)[:, 1]

            tau, _ = find_best_threshold(y_val, val_prob)

            results.append(
                compute_metrics(
                    y_test,
                    test_prob,
                    tau,
                    model_name,
                    feat_name,
                    "Baseline threshold selected on validation set",
                )
            )

    # =========================================================================
    # C. 保存和打印结果
    # =========================================================================
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=["ROC-AUC", "PR-AUC", "Brier"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    results_df.to_csv(RESULTS_OUTPUT, index=False, encoding="utf-8-sig")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    print("\n" + "=" * 90)
    print("模型对比结果")
    print("=" * 90)
    show_cols = [
        "Model",
        "Features",
        "ROC-AUC",
        "PR-AUC",
        "Brier",
        "F2",
        "Recall/Sensitivity",
        "Specificity",
        "Precision/PPV",
        "Threshold",
    ]
    print(results_df[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 90)
    print("BRB Belief Interval 不确定性分组分析")
    print("=" * 90)
    print(bi_group_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n保存文件：")
    print(f"  ✅ {RESULTS_OUTPUT}")
    print(f"  ✅ {BI_SAMPLE_OUTPUT}")
    print(f"  ✅ {BI_GROUP_OUTPUT}")
    print(f"  ✅ {CALIBRATION_OUTPUT}")

    print("\n优秀门槛检查重点看：")
    print("  1) ER Causal-BRB (Platt calibrated)")
    print("  2) ER Causal-BRB (Isotonic calibrated)")
    print("  门槛：ROC-AUC >= 0.80, PR-AUC >= 0.30, F2 >= 0.50, Brier <= 0.08")


if __name__ == "__main__":
    main()
