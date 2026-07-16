import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 数据定义
groups = [
    'Low Uncertainty\n$(m_\\Theta < P_{33})$',
    'Medium Uncertainty\n$(P_{33} \\leq m_\\Theta < P_{67})$',
    'High Uncertainty\n$(m_\\Theta \\geq P_{67})$'
]
positive_rates = [2.57, 9.29, 20.91]
brier_scores   = [0.025, 0.087, 0.155]
sample_sizes   = [1168, 323, 550]
colors         = ['#81C784', '#FFB74D', '#E53935']  # 绿色、橙色、红色
brier_color    = '#1565C0'  # 蓝色

# 设置字体和网格
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.axisbelow'] = True  # 网格在最下层

fig, ax1 = plt.subplots(figsize=(10, 6.5))  # 稍微调整画布大小
ax2 = ax1.twinx()  # 创建双Y轴
x, width = np.arange(len(groups)), 0.52

# ==============================================================================
# 左轴：阳性率柱状图
# ==============================================================================
bars = ax1.bar(x, positive_rates, width,
               color=colors, edgecolor='black', linewidth=1.1, zorder=3,
               label='Positive Rate (%)')

# 设置左轴
ax1.set_ylabel('Actual Depression Positive Rate (%)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 27)  # 留出空间给标题和标注
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
ax1.tick_params(axis='y', labelsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(groups, fontsize=11, fontweight='bold')

# 网格
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# 核心修改：将柱状图上方文字移至左侧，避免被折线遮挡
# Low Uncertainty 文字移到左侧空白处
ax1.text(-0.3, positive_rates[0], f'{positive_rates[0]:.2f}%\n(N={sample_sizes[0]})',
         ha='right', va='center', fontsize=10, fontweight='bold', color='black', zorder=7)

# Medium Uncertainty 文字移到左侧空白处
ax1.text(x[1]-0.3, positive_rates[1], f'{positive_rates[1]:.2f}%\n(N={sample_sizes[1]})',
         ha='right', va='center', fontsize=10, fontweight='bold', color='black', zorder=7)

# High Uncertainty 文字移到左侧空白处
ax1.text(x[2]-0.3, positive_rates[2], f'{positive_rates[2]:.2f}%\n(N={sample_sizes[2]})',
         ha='right', va='center', fontsize=10, fontweight='bold', color='black', zorder=7)

# ==============================================================================
# 右轴：Brier Score 折线图
# ==============================================================================
ax2.plot(x, brier_scores, color=brier_color, linewidth=2.2,
         marker='D', markersize=8, markerfacecolor='white',
         markeredgecolor=brier_color, markeredgewidth=2,
         zorder=4, label='Brier Score')

# 为折线添加数据标签，并确保不被柱子遮挡
for i, val in enumerate(brier_scores):
    # 将标签放在标记下方，并增加 zorder 确保在柱子之上
    ax2.text(x[i], val - 0.015, f'{val:.3f}',
             ha='center', va='top', fontsize=10, color=brier_color, fontweight='bold', zorder=7)

# 设置右轴
ax2.set_ylabel('Brier Score (↓ Lower is Better)',
               fontsize=12, fontweight='bold', color=brier_color)
ax2.set_ylim(0.000, 0.22)  # 设定范围
ax2.tick_params(axis='y', labelcolor=brier_color, labelsize=11)
ax2.spines['right'].set_edgecolor(brier_color)  # 右侧边框颜色与右轴一致

# ==============================================================================
# 标注和图例
# ==============================================================================
# 8.1× Significant Gap 标注
ax1.annotate('', xy=(2, 22.0), xytext=(0, 3.2),
             arrowprops=dict(arrowstyle='<->', color='#616161', lw=1.5, ls='dashed'), zorder=5)
ax1.text(1, 14.0, '8.1× Significant Gap\n($p < 0.001$)',
         ha='center', va='center', fontsize=11, fontweight='bold', color='#B71C1C',
         bbox=dict(facecolor='white', edgecolor='#B71C1C', linewidth=1.2,
                   alpha=0.9, boxstyle='round,pad=0.3'), zorder=6)

# 图例（修改处：分离 legend 的生成和 zorder 的设置）
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
leg = ax1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=10, framealpha=0.9)
leg.set_zorder(8)


# X轴范围
ax1.set_xlim(-0.6, 2.6)  # 留出空间给左侧文字

plt.tight_layout()

# ==============================================================================
# 保存为 PNG 和 PDF 到 Picture 文件夹
# ==============================================================================
png_path = 'Picture/Figure_5_2_Uncertainty_DualAxis.png'
pdf_path = 'Picture/Figure_5_2_Uncertainty_DualAxis.pdf'

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')

print(f"✅ 图片已成功生成并保存为 PNG 和 PDF 格式于 Picture 文件夹：")
print(f"   - {png_path}")
print(f"   - {pdf_path}")

plt.show()