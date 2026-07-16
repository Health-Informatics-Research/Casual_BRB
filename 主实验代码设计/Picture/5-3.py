import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# =========================================================
# 数据：五个干预策略
# =========================================================
interventions = [
    'Comprehensive\n(Sleep+Food+Clinical+Substance)',
    'Sleep Quality\nImprovement',
    'Food Security\nGuarantee',
    'Substance Use\nIntervention',
    'Clinical Burden\nRelief',
]

# 后门调整后的 ATE 点估计（相对风险降低率 %）
ate_point  = [67.04, 54.28, 34.65, 17.71, 8.23]

# 95% CI 下界和上界（相对风险降低率 %）
ci_low     = [61.81, 49.69, 31.67, 15.42, 6.32]
ci_high    = [71.97, 58.31, 37.57, 19.77, 10.24]

# 朴素遮蔽法估计（仅睡眠干预有对比，其余不标）
naive_sleep = 73.0   # 索引1（Sleep Quality）对应

# 颜色：按干预效果强弱渐变
bar_colors = ['#C62828', '#E53935', '#EF9A9A', '#BDBDBD', '#9E9E9E']

# =========================================================
# 画布设置
# =========================================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.axisbelow'] = True

fig, ax = plt.subplots(figsize=(11, 6.5))

y      = np.arange(len(interventions))
height = 0.52

# ---------- 水平条形图 ----------
bars = ax.barh(y, ate_point, height,
               color=bar_colors, edgecolor='black', linewidth=0.9,
               zorder=3, label='ATE (Backdoor Adjusted)')

# ---------- 95% CI 误差条 ----------
xerr_low  = [ate_point[i] - ci_low[i]  for i in range(len(ate_point))]
xerr_high = [ci_high[i]  - ate_point[i] for i in range(len(ate_point))]
ax.errorbar(ate_point, y,
            xerr=[xerr_low, xerr_high],
            fmt='none', color='black', capsize=5, capthick=1.5,
            elinewidth=1.5, zorder=4)

# ---------- 朴素遮蔽法虚线标注（仅 Sleep，索引 1）----------
sleep_idx = 1
ax.axvline(x=naive_sleep, ymin=0, ymax=1,
           color='#1565C0', linewidth=1.4, linestyle='--', alpha=0.5, zorder=2)
ax.annotate('Naive Blocking: 73.0%\n(Unadjusted)',
            xy=(naive_sleep, sleep_idx),
            xytext=(naive_sleep + 1.5, sleep_idx + 0.55),
            fontsize=9.5, color='#1565C0', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.2))

# 18.7pp 偏差标注（在 Sleep 条右侧）
ax.annotate('',
            xy=(naive_sleep, sleep_idx),
            xytext=(ate_point[sleep_idx], sleep_idx),
            arrowprops=dict(arrowstyle='<->', color='#E53935', lw=1.6))
ax.text((naive_sleep + ate_point[sleep_idx]) / 2, sleep_idx - 0.34,
        '18.7pp bias\neliminated',
        ha='center', va='top', fontsize=9, color='#B71C1C', fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='#B71C1C',
                  linewidth=0.9, alpha=0.9, boxstyle='round,pad=0.2'))

# ---------- 条形末端数值标签 ----------
for i, (v, lo, hi) in enumerate(zip(ate_point, ci_low, ci_high)):
    ax.text(hi + 0.8, i, f'{v:.2f}%\n[{lo:.2f}, {hi:.2f}]',
            va='center', ha='left', fontsize=9, color='#212121')

# ---------- 轴与网格 ----------
ax.set_xlabel('Relative Risk Reduction — ATE (%)', fontsize=12, fontweight='bold')

ax.set_yticks(y)
ax.set_yticklabels(interventions, fontsize=10.5, fontweight='bold')
ax.set_xlim(0, 82)
ax.invert_yaxis()   # 效果最大的在最上方
ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------- 图例 ----------
patch_ate   = mpatches.Patch(color='#E53935', label='ATE — Backdoor Adjusted (95% CI)')
line_naive  = plt.Line2D([0], [0], color='#1565C0', linewidth=1.4,
                         linestyle='--', label='Naive Blocking Estimate (Sleep only)')
ax.legend(handles=[patch_ate, line_naive],
          loc='lower right', fontsize=9.5, framealpha=0.9)

plt.tight_layout()

# =========================================================
# 保存为 PNG 和 PDF 到 Picture 文件夹
# =========================================================
png_path = 'Picture/Figure_5_3_ATE_Horizontal.png'
pdf_path = 'Picture/Figure_5_3_ATE_Horizontal.pdf'

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')

print(f"✅ 图片已成功生成并保存为 PNG 和 PDF 格式于 Picture 文件夹：")
print(f"   - {png_path}")
print(f"   - {pdf_path}")

plt.show()