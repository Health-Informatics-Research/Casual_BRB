# =============================================================================
# nhanes_step2_feature_engineering.py  【GBM增强版】
#
# 相对上一版本的三处修改：
#
# 修改 A —— P1 / P5 路径改用 GradientBoostingClassifier
#   原版 LR (C=0.1) 对 P1（收入/教育/年龄多维非线性）和 P5（慢性病共病）
#   过于简单，OOF Train/Test AUC 差距达 0.05（P1: 0.664 vs 0.614），
#   说明线性模型欠拟合。换用 GBM 后可捕获非线性交互，预计 P1/P5 的
#   Test AUC 分别提升 0.04~0.07。
#
# 修改 B —— 新增 PathwayScorerGBM 类，与原 PathwayScorerSupervised 接口统一
#   两者都实现 fit() / predict_prob()，可以无缝接入 fit_and_predict_oof()。
#
# 修改 C —— 报告各路径的 Train/Test AUC 差距，便于论文直接引用
#   输出"泛化差距 (Gap)"列，差距 > 0.04 时标注警告。
# =============================================================================
 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import warnings
 
warnings.filterwarnings("ignore")
 
# =============================================================================
# 0. 全局配置
# =============================================================================
INPUT_CSV    = "nhanes_clean_original.csv"
TRAIN_OUTPUT = "nhanes_brb_train.csv"
TEST_OUTPUT  = "nhanes_brb_test.csv"
RANDOM_SEED  = 42
TEST_SIZE    = 0.2
N_OOF_FOLDS = 5
 
# 路径变量定义（与上游 Step 1.5 因果图对齐）
PATHWAY_VARIABLES = {
    "P1_Socioeconomic": [
        "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2",
        "DMDMARTL", "INDFMPIR", "INDFMIN2", "INDHHIN2"
    ],
    "P2_Sleep": ["SLD012", "SLQ050", "SLQ120", "SLQ040"],
    "P3_HealthStatus": ["HUQ010", "HUQ090", "HUQ051", "BMXBMI", "BMXWAIST"],
    "P4_FoodSecurity": [
        "FSDAD", "FSDHH", "FSD032A", "FSD032B",
        "FSD032C", "FSQ165", "FSD151"
    ],
    "P5_Clinical": ["DIQ010", "BPQ020", "MCQ160A", "MCQ080"],
    "P6_Substance": ["SMQ020", "ALQ151", "ALQ130", "DUQ200", "DUQ240"],
}
 
# 修改 A：指定哪些路径使用 GBM（非线性关系更强的路径）
GBM_PATHWAYS = {"P1_Socioeconomic", "P5_Clinical"}
 
# =============================================================================
# 1. 路径子模型定义
# =============================================================================
 
class PathwayScorerLR:
    """
    线性路径模型（LR）：适用于 P2/P3/P4/P6 这类变量相对稀疏、
    样本量足够让 L2 正则化充分发挥作用的路径。
    """
    def __init__(self, name, variables):
        self.name = name
        self.vars_available = variables
        self.scaler = StandardScaler()
        self.model  = LogisticRegression(
            penalty='l2', C=0.1,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            max_iter=1000,
            solver='lbfgs'
        )
 
    def fit(self, X_train, y_train):
        X_sub    = X_train[self.vars_available].copy()
        X_scaled = self.scaler.fit_transform(X_sub)
        self.model.fit(X_scaled, y_train)
        return self
 
    def predict_prob(self, X):
        X_sub    = X[self.vars_available].copy()
        X_scaled = self.scaler.transform(X_sub)
        return self.model.predict_proba(X_scaled)[:, 1]
 
 
class PathwayScorerGBM:
    """
    修改 A：梯度提升路径模型（GBM）。
    用于 P1（收入/教育/年龄高度非线性）和 P5（慢性病共病交互）。
 
    超参选择原则：
      - n_estimators=200, max_depth=3：防止对小阳性类过拟合（阳性率 8.6%）
      - learning_rate=0.05：小步长 + 多树，泛化更稳定
      - subsample=0.8：Stochastic GBM，降低方差
      - min_samples_leaf=20：对 n~8000 的训练集，叶节点至少 20 样本
    无需 StandardScaler（树模型对尺度不敏感）。
    """
    def __init__(self, name, variables):
        self.name = name
        self.vars_available = variables
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=RANDOM_SEED
        )
 
    def fit(self, X_train, y_train):
        X_sub = X_train[self.vars_available].copy()
        # GBM 内部处理类不平衡：通过 sample_weight 等效 class_weight='balanced'
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        ratio = n_neg / max(n_pos, 1)
        sample_weight = np.where(y_train == 1, ratio, 1.0)
        self.model.fit(X_sub, y_train, sample_weight=sample_weight)
        return self
 
    def predict_prob(self, X):
        X_sub = X[self.vars_available].copy()
        return self.model.predict_proba(X_sub)[:, 1]
 
 
def make_scorer(path_name, variables):
    """
    工厂函数：根据路径名称选择合适的子模型。
    GBM_PATHWAYS 中的路径使用 GBM，其余使用 LR。
    """
    if path_name in GBM_PATHWAYS:
        return PathwayScorerGBM(path_name, variables)
    return PathwayScorerLR(path_name, variables)
 
 
# =============================================================================
# 2. Out-of-Fold 预测（防标签泄露）
# =============================================================================
 
def fit_and_predict_oof(scorer_template, df_train, y_train, df_test,
                         n_folds=N_OOF_FOLDS):
    """
    OOF 逻辑：
      1. 5 折 CV 生成训练集无偏概率（每个样本由没见过它的子模型预测）
      2. 全量训练集重新 fit 一次，用于测试集预测
    这确保下游 BRB 优化器接收到的训练集概率分布与测试集一致。
    """
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof_probs = np.zeros(len(df_train))
 
    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(df_train, y_train)):
        # 每折独立创建子模型，避免 scaler / 树共享状态
        fold_scorer = make_scorer(scorer_template.name,
                                  scorer_template.vars_available)
        fold_scorer.fit(df_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof_probs[val_idx] = fold_scorer.predict_prob(df_train.iloc[val_idx])
 
    # 全量 fit → 测试集打分
    final_scorer = make_scorer(scorer_template.name,
                               scorer_template.vars_available)
    final_scorer.fit(df_train, y_train)
    test_probs = final_scorer.predict_prob(df_test)
 
    return oof_probs, test_probs
 
 
# =============================================================================
# 3. 主流程
# =============================================================================
 
def main():
    print("=" * 65)
    print("NHANES Step 2：路径特征构建  【GBM 增强 + OOF 防泄漏版】")
    print("=" * 65)
    print(f"GBM 路径: {sorted(GBM_PATHWAYS)}")
    print(f"LR  路径: {sorted(set(PATHWAY_VARIABLES) - GBM_PATHWAYS)}")
    print()
 
    df = pd.read_csv(INPUT_CSV)
 
    # 传递给下游的元数据列（Step 4/5/6 均需要 RIDAGEYR/RIAGENDR/DMDMARTL 做归一化）
    meta_cols = [
        "SEQN", "depression_label", "WTMEC4YR",
        "SDMVPSU", "SDMVSTRA",
        "RIDAGEYR", "RIAGENDR", "DMDMARTL"
    ]
    meta_present = [c for c in meta_cols if c in df.columns]
 
    # 全局一次切分，保证 Step 2~6 使用完全相同的 train/test 分割
    y = df["depression_label"]
    idx_all = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx_all, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_SEED
    )
 
    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_test  = df.iloc[test_idx].copy().reset_index(drop=True)
    y_train  = y.iloc[train_idx].reset_index(drop=True)
    y_test   = y.iloc[test_idx].reset_index(drop=True)
 
    train_out = df_train[meta_present].copy()
    test_out  = df_test[meta_present].copy()
 
    print(f"{'路径':<25} {'模型':<6} {'变量数':>5}  "
          f"{'OOF Train AUC':>14}  {'Test AUC':>9}  {'Gap':>6}  {'警告':>6}")
    print("-" * 80)
 
    for path_name, var_list in PATHWAY_VARIABLES.items():
        # 自动纳入 _missing 指示列（Step 1 生成）
        available = []
        for v in var_list:
            if v in df.columns:
                available.append(v)
            if f"{v}_missing" in df.columns:
                available.append(f"{v}_missing")
 
        if not available:
            print(f"[警告] {path_name} 无可用变量，已跳过。")
            continue
 
        # 创建模板（仅用于传递 name 和 vars，实际 fit 在 OOF 内部完成）
        template = make_scorer(path_name, available)
        model_tag = "GBM" if path_name in GBM_PATHWAYS else "LR"
 
        train_probs, test_probs = fit_and_predict_oof(
            template, df_train, y_train, df_test
        )
 
        train_auc = roc_auc_score(y_train, train_probs)
        test_auc  = roc_auc_score(y_test,  test_probs)
        gap       = train_auc - test_auc
        warn      = "⚠️" if gap > 0.04 else "✅"
 
        print(f"{path_name:<25} {model_tag:<6} {len(available):>5}  "
              f"{train_auc:>14.4f}  {test_auc:>9.4f}  {gap:>6.4f}  {warn:>6}")
 
        train_out[f"{path_name}_prob"] = train_probs
        test_out[f"{path_name}_prob"]  = test_probs
 
    train_out.rename(columns={"depression_label": "label"}, inplace=True)
    test_out.rename(columns={"depression_label": "label"},  inplace=True)
 
    train_out.to_csv(TRAIN_OUTPUT, index=False, encoding="utf-8-sig")
    test_out.to_csv(TEST_OUTPUT,   index=False, encoding="utf-8-sig")
 
    print()
    print("=" * 65)
    print(f"✅ 路径特征已保存：")
    print(f"   训练集 → {TRAIN_OUTPUT}  ({len(train_out)} 行)")
    print(f"   测试集 → {TEST_OUTPUT}   ({len(test_out)} 行)")
    print("=" * 65)
 
 
if __name__ == "__main__":
    main()