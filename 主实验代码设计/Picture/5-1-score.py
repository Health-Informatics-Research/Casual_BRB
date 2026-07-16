import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# 1. 实验数据准备 (严格对应表 5-3 的数据)
# =========================================================
labels = ['A0\n(Baseline)', 'V1\n(Flat BRB)', 'V4\n(Random)', 'V2\n(Unweighted)', 'V3\n(No Unc)', 'Ours\n(Causal-BRB)']

# 填入提取的数据 (A0 的 PR-AUC 和 F2 设为 0，因为基线通常不计算/不适用这些特定指标)
roc_auc = [0.8392, 0.8102, 0.8121, 0.7977, 0.7889, 0.8068]
pr_auc  = [0.0000, 0.2951, 0.2774, 0.3017, 0.2689, 0.2612] 
brier   = [0.1909, 0.2185, 0.1817, 0.1490, 0.0917, 0.0690]
f2      = [0.0000, 0.4965, 0.4791, 0.4946, 0.5091, 0.5265]

# 配色方案：基线(灰), 变体(蓝), Ours(红)
colors = ['#B0BEC5', '#5C6BC0', '#5C6BC0', '#5C6BC0', '#5C6BC0', '#E53935']
# 仅给 Ours 加上阴影，凸显地位
hatches = ['', '', '', '', '', '//']

# =========================================================
# 2. 全局画图样式设置 (顶刊风)
# =========================================================
plt.rcParams['font.family'] = 'Times New Roman'  # 学术标准字体
plt.rcParams['axes.axisbelow'] = True            # 让网格线在柱子下方
plt.rcParams['font.size'] = 12

fig, axs = plt.subplots(2, 2, figsize=(14, 11))

x = np.arange(len(labels))
width = 0.65

def plot_bar(ax, data, title, ylabel, y_limits, is_lower_better=False):
    """绘制单个子图的通用函数"""
    bars = ax.bar(x, data, width, color=colors, edgecolor='black', linewidth=1.2)
    
    # 为 Ours 添加阴影纹理
    for i, bar in enumerate(bars):
        if hatches[i]:
            bar.set_hatch(hatches[i])
            
    # 设置标题和轴
    if is_lower_better:
        ax.set_title(title + ' (Lower is Better) ↓', fontsize=14, fontweight='bold', color='#D84315')
    else:
        ax.set_title(title + ' (Higher is Better) ↑', fontsize=14, fontweight='bold')
        
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=11)
    ax.set_ylim(y_limits)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 在柱子上方添加数据标签
    for i, bar in enumerate(bars):
        val = data[i]
        if val == 0.0:  # 针对 A0 缺失的指标打印 N/A
            ax.text(bar.get_x() + bar.get_width()/2., y_limits[0] + (y_limits[1]-y_limits[0])*0.05,
                    'N/A', ha='center', va='bottom', fontsize=11, color='gray', fontweight='bold')
        else:
            # Brier 分数使用 4 位小数，其他使用 4 位
            ax.text(bar.get_x() + bar.get_width()/2., val + (y_limits[1]-y_limits[0])*0.015,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# =========================================================
# 3. 绘制 2x2 矩阵的四个子图
# =========================================================

# 图 (a) ROC-AUC
plot_bar(axs[0, 0], roc_auc, '(a) ROC-AUC', 'Area Under Curve', [0.75, 0.86])

# 图 (b) PR-AUC
plot_bar(axs[0, 1], pr_auc, '(b) PR-AUC', 'Area Under Curve', [0.20, 0.32])

# 图 (c) Brier Score (重点高亮，向下箭头)
plot_bar(axs[1, 0], brier, '(c) Brier Score', 'Mean Squared Error', [0.0, 0.25], is_lower_better=True)

# 图 (d) F2 Score
plot_bar(axs[1, 1], f2, '(d) F2 Score', 'F2 Value', [0.45, 0.55])

# =========================================================
# 4. 布局调整与双格式保存 (PNG + PDF)
# =========================================================
plt.tight_layout(rect=[0, 0, 1, 0.94])  # 留出顶部大标题空间

# 保存路径（假设 Picture 文件夹已存在）
png_path = 'Picture/Figure_5_1_Ablation_Study.png'
pdf_path = 'Picture/Figure_5_1_Ablation_Study.pdf'

# 保存 PNG（高分辨率位图）
plt.savefig(png_path, dpi=300, bbox_inches='tight')
# 保存 PDF（矢量格式，适合论文出版）
plt.savefig(pdf_path, bbox_inches='tight')

print(f"✅ 图片已成功生成并保存为 PNG 和 PDF 格式于 Picture 文件夹：")
print(f"   - {png_path}")
print(f"   - {pdf_path}")

plt.show()