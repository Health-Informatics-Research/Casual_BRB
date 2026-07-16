# =============================================================================
# nhanes_step3_brb_baseline.py 【重构版】
# NHANES 因果层级 BRB 基线 —— 三角隶属度软激活 + 加权融合
#
# 改进：
#   1. 使用 Step 1.5 学到的因果层级动态构建 Sub1/Sub2/Sub3
#   2. 隶属度锚点由训练集分位数驱动（数据驱动）
#   3. 三种初始化方式比较，最终选择 Monotonic 作为后续优化起点
#   4. 输出仅用于对比，不调参
# =============================================================================

import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score

# =============================================================================
# 0. 加载因果层级配置
# =============================================================================
try:
    HIERARCHY = np.load("brb_hierarchy_config.npy", allow_pickle=True).item()
    print("加载因果层级配置：")
    for k, v in HIERARCHY.items():
        print(f"  {k}: {v}")
except FileNotFoundError:
    # 若文件不存在，使用默认配置（按之前运行结果）
    HIERARCHY = {
        "Sub1_background": ["P1_Socioeconomic"],
        "Sub2_behavioral": ["P2_Sleep", "P3_HealthStatus"],
        "Sub3_stress": ["P4_FoodSecurity", "P5_Clinical", "P6_Substance"]
    }
    print("未找到 brb_hierarchy_config.npy，使用默认层级。")

# 构建每个子规则库的输入路径
SUB1_INPUTS = [f"{p}_prob" for p in HIERARCHY["Sub1_background"]]
SUB2_INPUTS = [f"{p}_prob" for p in HIERARCHY["Sub2_behavioral"]]
SUB3_INPUTS = [f"{p}_prob" for p in HIERARCHY["Sub3_stress"]]

# =============================================================================
# 1. ER-BRB 节点（三角隶属度 + 加权融合）
# =============================================================================
class FuzzyBRB:
    def __init__(self, name, input_cols, init_method="linear", alpha=5.0):
        self.name = name
        self.input_cols = input_cols
        self.num_inputs = len(input_cols)
        self.init_method = init_method
        self.alpha = alpha
        self.rule_base = {}
        self.ref_values = {}      # {col: (c0, c1, c2)}

    def fit(self, df_train, y_train, discrete_cols=None):
        # 1. 隶属度锚点：训练集 16%, 50%, 84% 分位数
        for col in self.input_cols:
            probs = df_train[col].values
            c0 = np.percentile(probs, 16.0)
            c1 = np.percentile(probs, 50.0)
            c2 = np.percentile(probs, 84.0)
            if c1 <= c0: c1 = c0 + 1e-5
            if c2 <= c1: c2 = c1 + 1e-5
            self.ref_values[col] = (c0, c1, c2)

        all_combos = list(itertools.product([0,1,2], repeat=self.num_inputs))
        pos_rate = float(y_train.mean())

        if self.init_method == "linear":
            max_sum = self.num_inputs * 2.0
            self.rule_base = {combo: sum(combo) / max_sum for combo in all_combos}

        elif self.init_method in ["empirical", "monotonic"]:
            self.rule_base = {}
            for combo in all_combos:
                mask = np.ones(len(df_train), dtype=bool)
                for i, dcol in enumerate(discrete_cols):
                    mask &= (df_train[dcol].values == combo[i])
                n_rule = mask.sum()
                n_pos = y_train[mask].sum() if n_rule > 0 else 0
                belief = (n_pos + self.alpha * pos_rate) / (n_rule + self.alpha)
                self.rule_base[combo] = float(belief)

            if self.init_method == "monotonic":
                self.rule_base = self._enforce_monotonicity(self.rule_base)

        else:
            raise ValueError(f"未知初始化方法: {self.init_method}")

        return self

    def _enforce_monotonicity(self, rule_base):
        rb = rule_base.copy()
        sorted_keys = sorted(rb.keys(), key=lambda x: sum(x))
        for k in sorted_keys:
            for i in range(len(k)):
                if k[i] > 0:
                    lesser = list(k)
                    lesser[i] -= 1
                    lesser = tuple(lesser)
                    if rb[k] < rb[lesser]:
                        rb[k] = rb[lesser]
        return rb

    def _calc_matching(self, x, col):
        c0, c1, c2 = self.ref_values[col]
        a0 = np.zeros_like(x)
        a1 = np.zeros_like(x)
        a2 = np.zeros_like(x)

        idx0 = x <= c0
        a0[idx0] = 1.0
        idx1 = (x > c0) & (x <= c1)
        a0[idx1] = (c1 - x[idx1]) / (c1 - c0)
        a1[idx1] = (x[idx1] - c0) / (c1 - c0)
        idx2 = (x > c1) & (x <= c2)
        a1[idx2] = (c2 - x[idx2]) / (c2 - c1)
        a2[idx2] = (x[idx2] - c1) / (c2 - c1)
        idx3 = x > c2
        a2[idx3] = 1.0
        return a0, a1, a2

    def predict(self, df):
        N = len(df)
        combos = list(itertools.product([0,1,2], repeat=self.num_inputs))
        # 匹配度
        matchings = [self._calc_matching(df[col].values, col) for col in self.input_cols]

        activation = np.zeros((N, len(combos)))
        for i, combo in enumerate(combos):
            w = np.ones(N)
            for j, lvl in enumerate(combo):
                w *= matchings[j][lvl]
            activation[:, i] = w

        w_sum = activation.sum(axis=1, keepdims=True)
        w_sum[w_sum == 0] = 1.0
        norm_w = activation / w_sum

        final = np.zeros(N)
        for i, combo in enumerate(combos):
            final += norm_w[:, i] * self.rule_base[combo]
        return final

# =============================================================================
# 2. 评估函数
# =============================================================================
def find_best_threshold(y_true, y_probs):
    best_tau, best_f2 = 0.5, -1
    for tau in np.linspace(0.05, 0.95, 91):
        y_pred = (y_probs >= tau).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_tau = tau
    return best_tau, best_f2

# =============================================================================
# 3. 管道：给定初始化方法，训练并评估
# =============================================================================
def run_pipeline(init_method, df_fit, y_fit, df_val, y_val, df_test, y_test):
    # 构建子规则库
    sub1 = FuzzyBRB("Sub1", SUB1_INPUTS, init_method)
    sub2 = FuzzyBRB("Sub2", SUB2_INPUTS, init_method)
    sub3 = FuzzyBRB("Sub3", SUB3_INPUTS, init_method)

    # 获取离散等级列用于 empirical 统计
    dcol1 = [c.replace("_prob", "_BRB_level") for c in SUB1_INPUTS]
    dcol2 = [c.replace("_prob", "_BRB_level") for c in SUB2_INPUTS]
    dcol3 = [c.replace("_prob", "_BRB_level") for c in SUB3_INPUTS]

    sub1.fit(df_fit, y_fit, dcol1)
    sub2.fit(df_fit, y_fit, dcol2)
    sub3.fit(df_fit, y_fit, dcol3)

    # 生成中间层得分
    for df_ in [df_fit, df_val, df_test]:
        df_["Sub1_prob"] = sub1.predict(df_)
        df_["Sub2_prob"] = sub2.predict(df_)
        df_["Sub3_prob"] = sub3.predict(df_)

    # 主规则库（接收三个子层输出）
    main_brb = FuzzyBRB("Main", ["Sub1_prob", "Sub2_prob", "Sub3_prob"], init_method)
    # 创建临时离散等级（便于 empirical 统计，只用于初始化，推理用连续值）
    for df_ in [df_fit, df_val, df_test]:
        df_["Sub1_fake_level"] = np.where(df_["Sub1_prob"] < 0.33, 0, np.where(df_["Sub1_prob"] < 0.67, 1, 2))
        df_["Sub2_fake_level"] = np.where(df_["Sub2_prob"] < 0.33, 0, np.where(df_["Sub2_prob"] < 0.67, 1, 2))
        df_["Sub3_fake_level"] = np.where(df_["Sub3_prob"] < 0.33, 0, np.where(df_["Sub3_prob"] < 0.67, 1, 2))
    main_discrete = ["Sub1_fake_level", "Sub2_fake_level", "Sub3_fake_level"]
    main_brb.fit(df_fit, y_fit, main_discrete)

    val_probs = main_brb.predict(df_val)
    best_tau, _ = find_best_threshold(y_val, val_probs)

    test_probs = main_brb.predict(df_test)
    y_pred = (test_probs >= best_tau).astype(int)

    roc = roc_auc_score(y_test, test_probs)
    pr  = average_precision_score(y_test, test_probs)
    f2  = fbeta_score(y_test, y_pred, beta=2, zero_division=0)

    return {
        "Method": init_method.capitalize(),
        "Best_Tau": best_tau,
        "ROC-AUC": roc,
        "PR-AUC": pr,
        "F2": f2
    }

# =============================================================================
# 主程序
# =============================================================================
def main():
    print("=" * 60)
    print("NHANES 因果层级 BRB 基线 (模糊规则激活融合)")
    print("=" * 60)

    train_df = pd.read_csv("nhanes_brb_train.csv")
    test_df  = pd.read_csv("nhanes_brb_test.csv")
    y_train_full = train_df["label"].values

    # 划分训练/验证（与后续优化一致）
    df_fit, df_val, y_fit, y_val = train_test_split(
        train_df, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )
    y_test = test_df["label"].values

    results = []
    for method in ["linear", "empirical", "monotonic"]:
        res = run_pipeline(method,
                           df_fit.copy(), y_fit,
                           df_val.copy(), y_val,
                           test_df.copy(), y_test)
        results.append(res)

    res_df = pd.DataFrame(results).set_index("Method")
    print("\n测试集基线表现：")
    print(res_df.round(4).to_string())

    # 选择单调版本参数保存，供后续优化热启动
    mono_res = [r for r in results if r["Method"] == "Monotonic"][0]
    print(f"\nMonotonic 基线 F2: {mono_res['F2']:.4f}, ROC-AUC: {mono_res['ROC-AUC']:.4f}")

    print("=" * 60)
    print("Step 3 完成。下一步：差分进化优化 (Step 4)")

if __name__ == "__main__":
    main()