# =============================================================================
# nhanes_step11_e4_learning_curve.py
# 【E4-2 学习曲线：Causal-BRB vs. 随机图架构】
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ★ 复用 E4-1 中验证过的极其稳定的核心引擎 ★
from nhanes_step10_e4_phq9_sensitivity import train_and_eval_for_threshold

# =============================================================================
# 0. 全局配置
# =============================================================================
TRAIN_CSV = "nhanes_brb_train.csv"
TEST_CSV = "nhanes_brb_test.csv"
OUTPUT_DIR = "E4_Robustness"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 规模设置与重复次数
TRAIN_FRACS = [0.2, 0.4, 0.6, 0.8, 1.0]
REPEATS = 3  # 每个比例重复3次，以便画出漂亮的误差带

def run_learning_curve():
    print("=" * 80)
    print("E4-2: 启动学习曲线对比实验 (Causal vs Random)")
    print("=" * 80)
    
    # 1. 加载数据
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    
    # 2. 拟合全局 Scaler (这是修复报错的核心)
    scl = MinMaxScaler()
    cols_to_scale = ["RIDAGEYR", "RIAGENDR", "DMDMARTL"]
    scl.fit(tr[cols_to_scale])
    
    # ★ 必须同时为测试集转换特征，否则 predict 时会报 KeyError
    te[["Age_prob", "Gender_prob", "DMDMARTL_prob"]] = scl.transform(te[cols_to_scale])
    
    # 3. 加载因果层级配置
    try:
        hierarchy = np.load("brb_hierarchy_config.npy", allow_pickle=True).item()
    except FileNotFoundError:
        print("⚠️ 找不到因果层级配置，使用默认...")
        hierarchy = {
            "Sub1_background": ["P1_Socioeconomic"], 
            "Sub2_behavioral": ["P2_Sleep", "P3_HealthStatus"], 
            "Sub3_stress": ["P4_FoodSecurity", "P5_Clinical", "P6_Substance"]
        }
    
    results = []
    
    # 4. 核心循环：不同比例 * 多次重复
    for frac in TRAIN_FRACS:
        for seed in range(REPEATS):
            print(f"\n📏 规模: {frac*100:.0f}% | 重复: {seed+1}/{REPEATS}")
            
            # 独立采样并复制，避免警告
            sub_tr = tr.sample(frac=frac, random_state=42 + seed).copy()
            
            # 为采样后的训练子集转换特征
            sub_tr[["Age_prob", "Gender_prob", "DMDMARTL_prob"]] = scl.transform(sub_tr[cols_to_scale])
            
            # 统一使用默认标签 (>=10)
            y_tr = sub_tr["label"]
            y_te = te["label"]
            
            # ---------------------------------------------------------
            # 模型 A: Causal-BRB (使用真实的因果层级)
            # ---------------------------------------------------------
            res_causal = train_and_eval_for_threshold(sub_tr, te, y_tr, y_te, hierarchy, 10)
            res_causal["Architecture"] = "Causal-BRB (Ours)"
            res_causal["Fraction"] = frac
            results.append(res_causal)
            
            # ---------------------------------------------------------
            # 模型 B: Random-Graph (打乱的层级，代表无因果先验)
            # ---------------------------------------------------------
            random_h = {
                "Sub1_background": ["P2_Sleep"], 
                "Sub2_behavioral": ["P1_Socioeconomic", "P4_FoodSecurity"],
                "Sub3_stress": ["P3_HealthStatus", "P5_Clinical", "P6_Substance"]
            }
            res_random = train_and_eval_for_threshold(sub_tr, te, y_tr, y_te, random_h, 10)
            res_random["Architecture"] = "Random-Graph (Baseline)"
            res_random["Fraction"] = frac
            results.append(res_random)

    # 5. 保存数据
    df_res = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "E4_2_Learning_Curve.csv")
    df_res.to_csv(csv_path, index=False)
    
    # 6. 学术绘图
    plt.figure(figsize=(9, 6))
    
    # lineplot 会自动计算多个 seed 的均值并画出阴影区（置信区间/误差带）
    sns.lineplot(
        data=df_res, 
        x="Fraction", 
        y="ROC-AUC", 
        hue="Architecture", 
        style="Architecture",
        palette=["navy", "firebrick"],
        markers=True, 
        dashes=True, 
        linewidth=2.5, 
        markersize=9
    )
    
    plt.title("Data Efficiency: Causal Prior vs. Random Architecture", pad=15, fontweight='bold')
    plt.xlabel("Fraction of Training Data", fontweight='bold')
    plt.ylabel("Testing ROC-AUC Score", fontweight='bold')
    plt.xticks(TRAIN_FRACS, [f"{int(f*100)}%" for f in TRAIN_FRACS])
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(title="Model Architecture", loc="lower right")
    
    out_plot = os.path.join(OUTPUT_DIR, "E4_2_Learning_Curve.png")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.close()
    
    print("\n" + "=" * 80)
    print("🎉 E4-2 学习曲线实验已完成！")
    print(f"📊 数据已保存至: {csv_path}")
    print(f"📈 学习曲线图已保存至: {out_plot}")

if __name__ == "__main__":
    run_learning_curve()