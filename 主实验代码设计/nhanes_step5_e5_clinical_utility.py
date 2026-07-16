# =============================================================================
# nhanes_step5_e5_clinical_utility.py
# 【E5 临床实用性分析：DCA + 人口公平性 + 临床分诊成本效益模拟】
#
# 前提：必须已经运行 Step 5，生成了 brb_belief_interval_samples.csv
#
# 包含内容（严格对齐实验设计文档）：
#   E5-1: 决策曲线分析 (DCA)
#   E5-2: 人口公平性分析 (Fairness Analysis)
#   E5-3: 信念间隔临床分诊价值 (Cost-Benefit Triage Simulation)
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, fbeta_score, roc_auc_score

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 全局配置
# =============================================================================
BI_SAMPLES_CSV = "brb_belief_interval_samples.csv"
RAW_CSV = "nhanes_clean_original.csv"

OUTPUT_DIR = "E5_Clinical_Utility"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 学术风绘图设置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def safe_auc(y_true, y_prob):
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return np.nan

def find_best_threshold(y_true, y_prob):
    best_tau, best_f2 = 0.5, -1.0
    for tau in np.linspace(0.05, 0.95, 91):
        f2 = fbeta_score(y_true, (y_prob >= tau).astype(int), beta=2, zero_division=0)
        if f2 > best_f2: best_f2 = f2; best_tau = tau
    return best_tau

# =============================================================================
# E5-1: 决策曲线分析 (Decision Curve Analysis, DCA)
# =============================================================================
def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefits = []
    n = len(y_true)
    for t in thresholds:
        tp = np.sum((y_prob >= t) & (y_true == 1))
        fp = np.sum((y_prob >= t) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (t / (1 - t + 1e-10))
        net_benefits.append(nb)
    return np.array(net_benefits)

def run_e5_1_dca(df):
    print("\n" + "="*70)
    print("E5-1: 执行决策曲线分析 (DCA)")
    print("="*70)
    
    y_true = df["y_true"].values
    brb_prob = df["BRB_calibrated_prob"].values
    
    thresholds = np.linspace(0.01, 0.40, 100)
    prevalence = np.mean(y_true)
    
    nb_brb = calculate_net_benefit(y_true, brb_prob, thresholds)
    nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds + 1e-10))
    nb_none = np.zeros_like(thresholds)
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, nb_brb, label='Causal-BRB', color='firebrick', linewidth=2.5)
    plt.plot(thresholds, nb_all, label='Treat All', color='gray', linestyle='--')
    plt.plot(thresholds, nb_none, label='Treat None', color='black')
    
    plt.title('Decision Curve Analysis (DCA) for Depression Risk', pad=15)
    plt.xlabel('Threshold Probability ($p_t$)')
    plt.ylabel('Net Benefit')
    plt.ylim(-0.02, max(nb_brb) * 1.2)
    plt.xlim(0.01, 0.40)
    plt.legend(loc='upper right')
    
    out_file = os.path.join(OUTPUT_DIR, "E5_1_Decision_Curve_Analysis.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✅ DCA 曲线已保存至: {out_file}")

# =============================================================================
# E5-2: 人口公平性分析 (Fairness Analysis)
# =============================================================================
def run_e5_2_fairness(df):
    print("\n" + "="*70)
    print("E5-2: 执行人口公平性分析 (Fairness Analysis)")
    print("="*70)
    
    # 严格遵循设计文档的 Subgroups 定义
    SUBGROUPS = {
        "Female": df["RIAGENDR"] == 2,
        "Male": df["RIAGENDR"] == 1,
        "Age < 40": df["RIDAGEYR"] < 40,
        "Age 40-60": (df["RIDAGEYR"] >= 40) & (df["RIDAGEYR"] < 60),
        "Age > 60": df["RIDAGEYR"] >= 60,
        "Non-Hispanic White": df["RIDRETH3"] == 3,
        "Non-Hispanic Black": df["RIDRETH3"] == 4,
        "Hispanic": df["RIDRETH3"].isin([1, 2]),
    }
    
    best_tau = find_best_threshold(df["y_true"].values, df["BRB_calibrated_prob"].values)
    
    fairness_rows = []
    for group_name, mask in SUBGROUPS.items():
        sub_df = df[mask]
        if len(sub_df) == 0:
            continue
            
        sub_y = sub_df["y_true"].values
        sub_prob = sub_df["BRB_calibrated_prob"].values
        
        fairness_rows.append({
            "Group": group_name,
            "N": int(mask.sum()),
            "Positive_Rate": float(sub_y.mean()),
            "ROC-AUC": safe_auc(sub_y, sub_prob),
            "Brier": float(brier_score_loss(sub_y, sub_prob)),
            "F2": float(fbeta_score(sub_y, (sub_prob >= best_tau).astype(int), beta=2, zero_division=0))
        })
        
    fair_df = pd.DataFrame(fairness_rows)
    out_file = os.path.join(OUTPUT_DIR, "E5_2_Fairness_Analysis.csv")
    fair_df.to_csv(out_file, index=False, encoding='utf-8-sig')
    
    print(fair_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n✅ 公平性数据已保存至: {out_file}")

# =============================================================================
# E5-3: 信念间隔临床分诊价值 (Cost-Benefit Triage Simulation) - 灰色地带修正版
# =============================================================================
def run_e5_3_triage_simulation(df):
    print("\n" + "="*70)
    print("E5-3: 信念间隔临床分诊价值 (Triage Cost-Benefit) [灰色地带修正版]")
    print("="*70)
    
    y_true = df["y_true"].values
    prob = df["BRB_calibrated_prob"].values
    m_theta = df["m_Theta"].values
    
    total_pop = len(df)
    total_positives = int(np.sum(y_true))
    
    COST_FULL_EVAL = 1.0  # 全量 PHQ-9 评估成本
    COST_BRB_SCREEN = 0.1 # BRB 初筛成本
    
    # 策略 0: Treat All
    cost_treat_all = total_pop * COST_FULL_EVAL
    
    # 定义阈值区间
    tau_high = 0.15  # 高风险直接复查
    tau_low = 0.05   # 低于此值绝对安全
    q67 = np.percentile(m_theta, 66.67) # 高不确定性门槛
    
    # ==================================================
    # 策略 A: 传统黑盒单点概率筛查 (仅看 prob >= 0.15)
    # ==================================================
    screened_mask_A = prob >= tau_high
    n_screened_A = np.sum(screened_mask_A)
    caught_A = np.sum(screened_mask_A & (y_true == 1))
    cost_A = (total_pop * COST_BRB_SCREEN) + (n_screened_A * COST_FULL_EVAL)
    
    # ==================================================
    # 策略 B: 灰色地带精准打击 (prob >= 0.15  OR  [0.05 <= prob < 0.15 AND m_Theta > P67])
    # ==================================================
    gray_area_mask = (prob >= tau_low) & (prob < tau_high)
    high_unc_mask = m_theta > q67
    
    # 仅仅在“灰色地带”中，挑出那些高不确定性的人额外复查
    extra_screened_mask = gray_area_mask & high_unc_mask
    n_extra_screened = np.sum(extra_screened_mask)
    n_extra_caught = np.sum(extra_screened_mask & (y_true == 1))
    
    screened_mask_B = screened_mask_A | extra_screened_mask
    n_screened_B = np.sum(screened_mask_B)
    caught_B = np.sum(screened_mask_B & (y_true == 1))
    cost_B = (total_pop * COST_BRB_SCREEN) + (n_screened_B * COST_FULL_EVAL)
    
    print(f"[基础数据]")
    print(f"总测试人数: {total_pop}, 真实抑郁阳性数: {total_positives}")
    print(f"灰色地带(0.05 <= Prob < 0.15)人数: {np.sum(gray_area_mask)}")
    
    print(f"\n[策略 A: 传统概率筛查 (Prob >= {tau_high})]")
    print(f"复查人数: {n_screened_A} ({n_screened_A/total_pop*100:.1f}%) | 筛出阳性: {caught_A} ({caught_A/total_positives*100:.1f}%)")
    
    print(f"\n[策略 B: 灰色地带不确定性兜底 (Prob >= {tau_high} 或 [灰色地带且 m_Theta > P67])]")
    print(f"复查人数: {n_screened_B} ({n_screened_B/total_pop*100:.1f}%) | 筛出阳性: {caught_B} ({caught_B/total_positives*100:.1f}%)")
    
    print(f"\n[🚀 核心临床价值]")
    print(f"在 {np.sum(gray_area_mask)} 名处于'灰色地带'的疑似人群中，我们利用高不确定性精准锁定了 {n_extra_screened} 人进行复查。")
    print(f" -> 结论: 仅额外增加 {(n_extra_screened/total_pop)*100:.1f}% 的总体筛查成本，就成功挽救了 {(n_extra_caught/total_positives)*100:.1f}% 原本会被确诊漏掉的隐匿性阳性病例！")
    
    df_rescued = df[extra_screened_mask]
    out_file = os.path.join(OUTPUT_DIR, "E5_3_Gray_Area_Rescued_Cases.csv")
    df_rescued.to_csv(out_file, index=False, encoding='utf-8-sig')
    
    # 保存被捞回来的特殊病例画像
    df_rescued = df[extra_screened_mask]
    out_file = os.path.join(OUTPUT_DIR, "E5_3_High_Uncertainty_Rescued_Cases.csv")
    df_rescued.to_csv(out_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 额外捞回的特殊病例数据已保存至: {out_file} (可用于5.2节病理案例分析)")

# =============================================================================
# 主程序
# =============================================================================
def main():
    if not os.path.exists(BI_SAMPLES_CSV):
        raise FileNotFoundError(f"找不到 {BI_SAMPLES_CSV}！请确保已运行 Step 5。")
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"找不到 {RAW_CSV}！公平性分析需要提取原始人口学特征。")
        
    print("正在合并特征以支持 E5 分析...")
    bi_df = pd.read_csv(BI_SAMPLES_CSV)
    raw_df = pd.read_csv(RAW_CSV)
    
    # 根据 SEQN 提取公平性分析所需的人口统计学原始特征
    keep_cols = ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3']
    raw_sub = raw_df[[c for c in keep_cols if c in raw_df.columns]]
    
    df = pd.merge(bi_df, raw_sub, on="SEQN", how="left")
    
    # 依次执行设计文档中的三个分析
    run_e5_1_dca(df)
    run_e5_2_fairness(df)
    run_e5_3_triage_simulation(df)
    
    print("\n🎉 E5 临床实用性分析全部执行完毕！")

if __name__ == "__main__":
    main()