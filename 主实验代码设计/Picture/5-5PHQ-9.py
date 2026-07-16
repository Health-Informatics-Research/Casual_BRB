import matplotlib.pyplot as plt
import numpy as np

# 数据 (来自日志)
thresholds = [5, 8, 10, 12, 15]
auc = [0.7525, 0.7507, 0.8026, 0.8185, 0.8262]
brier = [0.1691, 0.1303, 0.0842, 0.0792, 0.0832]

plt.rcParams['font.family'] = 'Times New Roman'
fig, ax1 = plt.subplots(figsize=(8, 6))

# 画高危区间阴影 (阈值 10-15)
ax1.axvspan(9.8, 15.2, color='#E8F5E9', alpha=0.6, lw=0, label='Moderate-Severe Zone')

# 左轴 AUC
color1 = '#283593'
ax1.set_xlabel('PHQ-9 Diagnostic Cut-off Threshold', fontsize=13, fontweight='bold')
ax1.set_ylabel('ROC-AUC Score (Higher is Better)', color=color1, fontsize=13, fontweight='bold')
line1 = ax1.plot(thresholds, auc, marker='o', markersize=8, color=color1, linewidth=2.5, label='ROC-AUC')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0.70, 0.85)

# 右轴 Brier
ax2 = ax1.twinx()
color2 = '#C62828'
ax2.set_ylabel('Brier Score (Lower is Better)', color=color2, fontsize=13, fontweight='bold')
line2 = ax2.plot(thresholds, brier, marker='^', markersize=8, linestyle='--', color=color2, linewidth=2.5, label='Brier Score')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0.06, 0.18)

# 添加垂直虚线标注论文基准阈值
ax1.axvline(x=10, color='gray', linestyle=':', lw=2)
ax1.text(10.2, 0.71, 'Paper Baseline\n(Threshold=10)', color='gray', fontweight='bold')

# 图例合并
lines = line1 + line2 + [plt.Rectangle((0,0),1,1, color='#E8F5E9', alpha=0.6)]
labels = ['ROC-AUC', 'Brier Score', 'Target Robustness Zone']
ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(0.05, 0.5), fontsize=11)


# =========================================================
# 保存为 PNG 和 PDF 到 Picture 文件夹
# =========================================================
png_path = 'Picture/Figure_5_5_Threshold_Sensitivity.png'
pdf_path = 'Picture/Figure_5_5_Threshold_Sensitivity.pdf'

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')

print(f"✅ 图片已成功生成并保存为 PNG 和 PDF 格式于 Picture 文件夹：")
print(f"   - {png_path}")
print(f"   - {pdf_path}")

plt.show()