# =============================================================================
# nhanes_step6_counterfactual_auto_adjustment.py
# 【最终补强版：Global_unc ER-BRB + Isotonic校准 + 自动后门调整集 + Bootstrap CI】
#
# 相对上一版 Step 6 的关键补强：
#   1. 不再在 INTERVENTIONS 中手动写 adj_z；
#   2. 根据 brb_hierarchy_config.npy 自动构建路径层级因果图；
#   3. 根据治疗路径所在层级，自动生成后门调整集 Z；
#   4. 输出 adjustment_sets 和 graph_edges，作为论文中“调整集来源”的证据；
#   5. 保留原有的路径级反事实模拟、校准后风险、Bootstrap 95% CI。
#
# 当前最终模型：
#   Step 4 PR-AUC enhanced
#   Main inputs = [Sub1_mid, Sub2_mid, Sub3_mid, Global_unc]
#   N_RULES = 198
#   N_PARAMS = 396
#
# 输出文件：
#   step6_counterfactual_results_auto.csv
#   step6_counterfactual_samples_auto.csv
#   step6_counterfactual_ranked_auto.csv
#   step6_adjustment_sets_auto.csv
#   step6_causal_graph_edges_auto.csv
# =============================================================================

import itertools
import os
import warnings
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, fbeta_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


# =============================================================================
# 0. 全局配置
# =============================================================================
PARAMS_FILE = "best_brb_er_params.npy"
DIM_INFO_FILE = "brb_main_dim_info.npy"
HIERARCHY_FILE = "brb_hierarchy_config.npy"

TRAIN_CSV = "nhanes_brb_train.csv"
TEST_CSV = "nhanes_brb_test.csv"

RANDOM_SEED = 42
N_BOOTSTRAP = 300

RESULTS_OUTPUT = "step6_counterfactual_results_auto.csv"
SAMPLES_OUTPUT = "step6_counterfactual_samples_auto.csv"
RANKED_OUTPUT = "step6_counterfactual_ranked_auto.csv"
ADJSETS_OUTPUT = "step6_adjustment_sets_auto.csv"
GRAPH_EDGES_OUTPUT = "step6_causal_graph_edges_auto.csv"

OUTCOME_NODE = "Depression"

ALL_PATHWAYS = [
    "P1_Socioeconomic",
    "P2_Sleep",
    "P3_HealthStatus",
    "P4_FoodSecurity",
    "P5_Clinical",
    "P6_Substance",
]

# 只定义干预路径，不再手写 adj_z
INTERVENTIONS = {
    "Improve sleep risk (do P2=low)": {
        "paths": ["P2_Sleep"],
        "target_quantile": 0.16,
        "description": "将睡眠风险路径降低到低风险参考水平",
    },
    "Reduce food insecurity (do P4=low)": {
        "paths": ["P4_FoodSecurity"],
        "target_quantile": 0.16,
        "description": "将食物不安全风险路径降低到低风险参考水平",
    },
    "Reduce clinical burden (do P5=low)": {
        "paths": ["P5_Clinical"],
        "target_quantile": 0.16,
        "description": "将临床慢病负担路径降低到低风险参考水平",
    },
    "Reduce substance use risk (do P6=low)": {
        "paths": ["P6_Substance"],
        "target_quantile": 0.16,
        "description": "将物质使用风险路径降低到低风险参考水平",
    },
    "Combined modifiable pathways (P2+P4+P5+P6=low)": {
        "paths": ["P2_Sleep", "P4_FoodSecurity", "P5_Clinical", "P6_Substance"],
        "target_quantile": 0.16,
        "description": "同时降低睡眠、食物不安全、临床负担和物质使用风险路径",
    },
}


# =============================================================================
# 1. 层级结构与自动因果图
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


def build_layer_map(hierarchy: Dict[str, List[str]]) -> Dict[str, int]:
    """
    将路径映射到因果层级：
      Layer 1: background
      Layer 2: behavioral
      Layer 3: stress / downstream risk
    """
    layer_map = {}
    for p in hierarchy.get("Sub1_background", []):
        layer_map[p] = 1
    for p in hierarchy.get("Sub2_behavioral", []):
        layer_map[p] = 2
    for p in hierarchy.get("Sub3_stress", []):
        layer_map[p] = 3
    return layer_map


def build_hierarchical_causal_edges(hierarchy: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """
    根据 BRB 因果层级构建路径级 DAG 边。
    该图用于调整集自动生成和论文中说明调整集来源。
    """
    l1 = hierarchy.get("Sub1_background", [])
    l2 = hierarchy.get("Sub2_behavioral", [])
    l3 = hierarchy.get("Sub3_stress", [])

    edges = []

    # 背景层影响行为层
    for a in l1:
        for b in l2:
            edges.append((a, b))

    # 背景层和行为层共同影响下游压力/负担层
    for a in l1:
        for c in l3:
            edges.append((a, c))
    for b in l2:
        for c in l3:
            edges.append((b, c))

    # 所有路径均进入最终结局风险
    for p in l1 + l2 + l3:
        edges.append((p, OUTCOME_NODE))

    return edges


def ancestors_of(node: str, edges: List[Tuple[str, str]]) -> Set[str]:
    parents = {}
    for u, v in edges:
        parents.setdefault(v, set()).add(u)

    visited = set()
    stack = list(parents.get(node, set()))

    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        stack.extend(list(parents.get(cur, set())))

    return visited


def descendants_of(node: str, edges: List[Tuple[str, str]]) -> Set[str]:
    children = {}
    for u, v in edges:
        children.setdefault(u, set()).add(v)

    visited = set()
    stack = list(children.get(node, set()))

    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        stack.extend(list(children.get(cur, set())))

    return visited


def auto_adjustment_set(
    treatment_paths: List[str],
    hierarchy: Dict[str, List[str]],
    edges: List[Tuple[str, str]],
) -> List[str]:
    """
    自动生成后门调整集 Z。

    设计原则：
      1. 候选变量来自 treatment 的上游祖先节点；
      2. 排除 treatment 自身；
      3. 排除 treatment 的后代，避免调整中介或碰撞路径；
      4. 对多路径联合干预，使用所有 treatment 的祖先并集；
      5. 若某个上游变量也被同时干预，则不再作为调整变量。

    在当前三层层级图中，这相当于：
      - 对 P2/P3：调整 P1；
      - 对 P4/P5/P6：调整 P1/P2/P3；
      - 对 P2+P4+P5+P6 联合干预：调整 P1/P3，
        其中 P2 是干预变量，不作为调整变量。
    """
    treatments = set(treatment_paths)

    candidate = set()
    descendants = set()

    for t in treatment_paths:
        candidate |= ancestors_of(t, edges)
        descendants |= descendants_of(t, edges)

    # 只允许路径变量作为调整变量，不加入 Outcome
    valid_path_nodes = set(ALL_PATHWAYS)

    z = candidate & valid_path_nodes
    z = z - treatments
    z = z - descendants

    # 按层级与名称排序，保证结果稳定
    layer_map = build_layer_map(hierarchy)
    z_sorted = sorted(list(z), key=lambda x: (layer_map.get(x, 99), x))

    return z_sorted


def export_graph_and_adjustment_sets(hierarchy: Dict[str, List[str]], interventions: Dict):
    edges = build_hierarchical_causal_edges(hierarchy)

    graph_df = pd.DataFrame(edges, columns=["Source", "Target"])
    graph_df.to_csv(GRAPH_EDGES_OUTPUT, index=False, encoding="utf-8-sig")

    rows = []
    for name, spec in interventions.items():
        z = auto_adjustment_set(spec["paths"], hierarchy, edges)
        rows.append(
            {
                "Intervention": name,
                "Treatment_Paths": "+".join(spec["paths"]),
                "Auto_Adjustment_Set_Z": "+".join(z) if z else "None",
                "Rule": "Ancestors(treatment) excluding treatments and descendants, based on hierarchical causal graph",
            }
        )

    adj_df = pd.DataFrame(rows)
    adj_df.to_csv(ADJSETS_OUTPUT, index=False, encoding="utf-8-sig")

    return edges, adj_df


# =============================================================================
# 2. 数据加载与预处理
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

    for p in ALL_PATHWAYS:
        col = f"{p}_prob"
        if col not in train_df.columns:
            raise ValueError(f"缺少路径概率列: {col}")

    scaler = MinMaxScaler()
    scale_cols = ["RIDAGEYR", "RIAGENDR", "DMDMARTL"]

    train_df[["Age_prob", "Gender_prob", "DMDMARTL_prob"]] = scaler.fit_transform(
        train_df[scale_cols]
    )
    test_df[["Age_prob", "Gender_prob", "DMDMARTL_prob"]] = scaler.transform(
        test_df[scale_cols]
    )

    return train_df, test_df


# =============================================================================
# 3. ER-BRB 推理核心
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
                f"参数维度不匹配：当前 Step 6 期望 {N_PARAMS} 维，"
                f"但 best_brb_er_params.npy 是 {len(best_params)} 维。"
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

        # 与 Step 4/Step 5 V2 一致：Main anchors 使用 dummy beta=0.5, theta=1.0
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
# 4. 校准器与指标
# =============================================================================
def fit_isotonic_calibrator(brb: ER_BRB_Predictor, base_train: pd.DataFrame) -> IsotonicRegression:
    _, df_temp = train_test_split(
        base_train,
        test_size=0.4,
        stratify=base_train["label"],
        random_state=RANDOM_SEED,
    )
    df_cal, _ = train_test_split(
        df_temp,
        test_size=0.5,
        stratify=df_temp["label"],
        random_state=RANDOM_SEED,
    )

    raw_cal_prob = brb.predict_raw(df_cal)
    y_cal = df_cal["label"].values

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_cal_prob, y_cal)

    return iso


def apply_isotonic(iso: IsotonicRegression, raw_prob: np.ndarray) -> np.ndarray:
    return np.clip(iso.predict(raw_prob), 0.0, 1.0)


def summarize_prediction(y_true: np.ndarray, prob: np.ndarray, prefix: str) -> Dict[str, float]:
    try:
        roc = roc_auc_score(y_true, prob)
    except ValueError:
        roc = np.nan

    try:
        pr = average_precision_score(y_true, prob)
    except ValueError:
        pr = np.nan

    brier = brier_score_loss(y_true, prob)

    best_tau = 0.5
    best_f2 = -1.0
    for tau in np.linspace(0.05, 0.95, 91):
        f2 = fbeta_score(y_true, (prob >= tau).astype(int), beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_tau = tau

    return {
        f"{prefix}_ROC_AUC": roc,
        f"{prefix}_PR_AUC": pr,
        f"{prefix}_Brier": brier,
        f"{prefix}_F2_best": best_f2,
        f"{prefix}_Tau_best": best_tau,
        f"{prefix}_MeanRisk": float(np.mean(prob)),
    }


# =============================================================================
# 5. 反事实干预与后门调整
# =============================================================================
def get_low_risk_targets(train_df: pd.DataFrame, intervention: Dict) -> Dict[str, float]:
    targets = {}
    q = intervention.get("target_quantile", 0.16)

    for p in intervention["paths"]:
        col = f"{p}_prob"
        targets[col] = float(np.quantile(train_df[col].values, q))

    return targets


def apply_intervention(df: pd.DataFrame, train_df: pd.DataFrame, intervention: Dict) -> pd.DataFrame:
    cf_df = df.copy()
    targets = get_low_risk_targets(train_df, intervention)

    for col, target_value in targets.items():
        # 风险降低干预：只把高于低风险参考值的样本下调，不把已经低风险者上调
        cf_df[col] = np.minimum(cf_df[col].values, target_value)

    return cf_df


def make_adjustment_strata(df: pd.DataFrame, adj_z: List[str], train_df: pd.DataFrame) -> pd.Series:
    if not adj_z:
        return pd.Series(["ALL"] * len(df), index=df.index)

    strata_parts = []

    for p in adj_z:
        col = f"{p}_prob"
        if col not in df.columns:
            raise ValueError(f"调整变量缺失: {col}")

        q1, q2 = np.quantile(train_df[col].values, [1 / 3, 2 / 3])

        vals = df[col].values
        levels = np.where(vals <= q1, "L", np.where(vals <= q2, "M", "H"))
        strata_parts.append(pd.Series([f"{p}:{v}" for v in levels], index=df.index))

    strata = strata_parts[0].astype(str)
    for s in strata_parts[1:]:
        strata = strata + "|" + s.astype(str)

    return strata


def adjusted_ate(
    df: pd.DataFrame,
    factual_prob: np.ndarray,
    counterfactual_prob: np.ndarray,
    train_df: pd.DataFrame,
    adj_z: List[str],
) -> Dict[str, float]:
    strata = make_adjustment_strata(df, adj_z, train_df)

    tmp = pd.DataFrame(
        {
            "stratum": strata.values,
            "factual": factual_prob,
            "counterfactual": counterfactual_prob,
        }
    )
    tmp["delta"] = tmp["counterfactual"] - tmp["factual"]
    tmp["reduction"] = tmp["factual"] - tmp["counterfactual"]

    total_n = len(tmp)
    weighted_delta = 0.0
    weighted_reduction = 0.0
    n_strata = 0

    for _, sub in tmp.groupby("stratum"):
        w = len(sub) / total_n
        weighted_delta += w * sub["delta"].mean()
        weighted_reduction += w * sub["reduction"].mean()
        n_strata += 1

    return {
        "ATE_adj_cf_minus_fact": float(weighted_delta),
        "RiskReduction_adj_fact_minus_cf": float(weighted_reduction),
        "N_strata": int(n_strata),
    }


def bootstrap_ci(
    df: pd.DataFrame,
    factual_prob: np.ndarray,
    counterfactual_prob: np.ndarray,
    train_df: pd.DataFrame,
    adj_z: List[str],
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
) -> Tuple[float, float]:
    reductions = []
    n = len(df)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)

        boot_df = df.iloc[idx].reset_index(drop=True)
        fact_b = factual_prob[idx]
        cf_b = counterfactual_prob[idx]

        est = adjusted_ate(boot_df, fact_b, cf_b, train_df, adj_z)
        reductions.append(est["RiskReduction_adj_fact_minus_cf"])

    lo, hi = np.percentile(reductions, [2.5, 97.5])
    return float(lo), float(hi)


# =============================================================================
# 6. 主流程
# =============================================================================
def main():
    print("=" * 96)
    print("Step 6: 反事实推演【Global_unc ER-BRB + Isotonic校准 + 自动后门调整集】")
    print("=" * 96)

    if not os.path.exists(PARAMS_FILE):
        raise FileNotFoundError(f"未找到 {PARAMS_FILE}，请先运行 Step 4。")

    best_params = np.load(PARAMS_FILE)

    print("\n模型配置：")
    print(f"  Main inputs : {MAIN_INPUTS}")
    print(f"  N_RULES     : {N_RULES}")
    print(f"  N_PARAMS    : {N_PARAMS}")
    print(f"  params file : {len(best_params)} dims")

    if os.path.exists(DIM_INFO_FILE):
        dim_info = np.load(DIM_INFO_FILE, allow_pickle=True).item()
        print("  Step 4 dim info:", dim_info)

    if len(best_params) != N_PARAMS:
        raise ValueError(
            f"参数维度错误：Step 6 期望 {N_PARAMS} 维，但读取到 {len(best_params)} 维。"
        )

    edges, adj_df = export_graph_and_adjustment_sets(HIERARCHY, INTERVENTIONS)

    print("\n自动后门调整集：")
    print(adj_df[["Intervention", "Treatment_Paths", "Auto_Adjustment_Set_Z"]].to_string(index=False))

    train_df, test_df = load_and_augment_brb_data()
    y_test = test_df["label"].values

    # 与 Step 5 V2 一致：完整训练集构建 BRB anchor
    brb = ER_BRB_Predictor(train_df, best_params)

    # 与 Step 5 一致：独立 calibration set 上拟合 isotonic
    iso = fit_isotonic_calibrator(brb, train_df)

    factual_raw = brb.predict_raw(test_df)
    factual_cal = apply_isotonic(iso, factual_raw)
    factual_mD, factual_mTheta = brb.predict_with_uncertainty(test_df)

    base_metrics = summarize_prediction(y_test, factual_cal, "Factual_Calibrated")
    raw_metrics = summarize_prediction(y_test, factual_raw, "Factual_Raw")

    print("\n事实预测性能：")
    print(f"  Raw ROC-AUC       : {raw_metrics['Factual_Raw_ROC_AUC']:.4f}")
    print(f"  Raw PR-AUC        : {raw_metrics['Factual_Raw_PR_AUC']:.4f}")
    print(f"  Raw Brier         : {raw_metrics['Factual_Raw_Brier']:.4f}")
    print(f"  Cal ROC-AUC       : {base_metrics['Factual_Calibrated_ROC_AUC']:.4f}")
    print(f"  Cal PR-AUC        : {base_metrics['Factual_Calibrated_PR_AUC']:.4f}")
    print(f"  Cal Brier         : {base_metrics['Factual_Calibrated_Brier']:.4f}")
    print(f"  Cal Mean Risk     : {base_metrics['Factual_Calibrated_MeanRisk']:.4f}")

    rng = np.random.default_rng(RANDOM_SEED)

    result_rows = []
    sample_frames = []

    for name, intervention in INTERVENTIONS.items():
        treatment_paths = intervention["paths"]
        adj_z = auto_adjustment_set(treatment_paths, HIERARCHY, edges)

        print(f"\n执行干预: {name}")
        print(f"  Auto adjustment set Z: {adj_z if adj_z else 'None'}")

        cf_df = apply_intervention(test_df, train_df, intervention)

        cf_raw = brb.predict_raw(cf_df)
        cf_cal = apply_isotonic(iso, cf_raw)
        cf_mD, cf_mTheta = brb.predict_with_uncertainty(cf_df)

        adj = adjusted_ate(
            df=test_df,
            factual_prob=factual_cal,
            counterfactual_prob=cf_cal,
            train_df=train_df,
            adj_z=adj_z,
        )

        ci_low, ci_high = bootstrap_ci(
            df=test_df,
            factual_prob=factual_cal,
            counterfactual_prob=cf_cal,
            train_df=train_df,
            adj_z=adj_z,
            rng=rng,
            n_bootstrap=N_BOOTSTRAP,
        )

        factual_mean = float(np.mean(factual_cal))
        cf_mean = float(np.mean(cf_cal))
        raw_factual_mean = float(np.mean(factual_raw))
        raw_cf_mean = float(np.mean(cf_raw))

        risk_reduction = adj["RiskReduction_adj_fact_minus_cf"]
        relative_reduction = risk_reduction / factual_mean if factual_mean > 0 else np.nan

        target_values = get_low_risk_targets(train_df, intervention)

        row = {
            "Intervention": name,
            "Description": intervention["description"],
            "Paths": "+".join(treatment_paths),
            "Auto_Adjustment_Set_Z": "+".join(adj_z) if adj_z else "None",
            "Adjustment_Rule": "Ancestors(treatment) excluding treatments and descendants",
            "Target_Quantile": intervention.get("target_quantile", 0.16),
            "Target_Values": str({k: round(v, 6) for k, v in target_values.items()}),
            "N_Test": len(test_df),
            "N_Strata": adj["N_strata"],
            "Factual_MeanRisk_Cal": factual_mean,
            "Counterfactual_MeanRisk_Cal": cf_mean,
            "RiskReduction_Cal_Adjusted": risk_reduction,
            "RiskReduction_Cal_CI95_Low": ci_low,
            "RiskReduction_Cal_CI95_High": ci_high,
            "Relative_RiskReduction_Cal": relative_reduction,
            "Factual_MeanRisk_Raw": raw_factual_mean,
            "Counterfactual_MeanRisk_Raw": raw_cf_mean,
            "RiskReduction_Raw_Unadjusted": raw_factual_mean - raw_cf_mean,
            "Mean_mTheta_Factual": float(np.mean(factual_mTheta)),
            "Mean_mTheta_Counterfactual": float(np.mean(cf_mTheta)),
            "Delta_mTheta_FactMinusCF": float(np.mean(factual_mTheta - cf_mTheta)),
        }
        result_rows.append(row)

        sample_df = pd.DataFrame(
            {
                "SEQN": test_df["SEQN"].values if "SEQN" in test_df.columns else np.arange(len(test_df)),
                "Intervention": name,
                "Auto_Adjustment_Set_Z": "+".join(adj_z) if adj_z else "None",
                "y_true": y_test,
                "Factual_RawRisk": factual_raw,
                "Counterfactual_RawRisk": cf_raw,
                "Factual_CalRisk": factual_cal,
                "Counterfactual_CalRisk": cf_cal,
                "RiskReduction_Cal": factual_cal - cf_cal,
                "Factual_Bel_D": factual_mD,
                "Factual_Pl_D": np.clip(factual_mD + factual_mTheta, 0.0, 1.0),
                "Factual_mTheta": factual_mTheta,
                "Counterfactual_Bel_D": cf_mD,
                "Counterfactual_Pl_D": np.clip(cf_mD + cf_mTheta, 0.0, 1.0),
                "Counterfactual_mTheta": cf_mTheta,
            }
        )
        sample_frames.append(sample_df)

        print(
            f"  Cal risk: {factual_mean:.4f} -> {cf_mean:.4f} | "
            f"Reduction={risk_reduction:.4f} "
            f"95%CI[{ci_low:.4f}, {ci_high:.4f}] | "
            f"Relative={relative_reduction * 100:.2f}%"
        )

    results_df = pd.DataFrame(result_rows)
    samples_df = pd.concat(sample_frames, ignore_index=True)

    ranked_df = results_df.sort_values(
        by="RiskReduction_Cal_Adjusted",
        ascending=False,
    ).reset_index(drop=True)

    results_df.to_csv(RESULTS_OUTPUT, index=False, encoding="utf-8-sig")
    samples_df.to_csv(SAMPLES_OUTPUT, index=False, encoding="utf-8-sig")
    ranked_df.to_csv(RANKED_OUTPUT, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 96)
    print("反事实干预结果排序：按校准后调整风险降低量从高到低")
    print("=" * 96)
    show_cols = [
        "Intervention",
        "Paths",
        "Auto_Adjustment_Set_Z",
        "Factual_MeanRisk_Cal",
        "Counterfactual_MeanRisk_Cal",
        "RiskReduction_Cal_Adjusted",
        "RiskReduction_Cal_CI95_Low",
        "RiskReduction_Cal_CI95_High",
        "Relative_RiskReduction_Cal",
        "Delta_mTheta_FactMinusCF",
    ]
    print(ranked_df[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n保存文件：")
    print(f"  ✅ {RESULTS_OUTPUT}")
    print(f"  ✅ {SAMPLES_OUTPUT}")
    print(f"  ✅ {RANKED_OUTPUT}")
    print(f"  ✅ {ADJSETS_OUTPUT}")
    print(f"  ✅ {GRAPH_EDGES_OUTPUT}")

    print("\n论文建议：")
    print("  1) 现在可以写成：adjustment sets were automatically derived from the learned hierarchical causal graph.")
    print("  2) 仍然建议表述为 model-based pathway-level counterfactual simulation，而不是临床真实干预效果。")
    print("  3) 主文报告前 3 个干预，并在附表报告自动调整集。")


if __name__ == "__main__":
    main()
