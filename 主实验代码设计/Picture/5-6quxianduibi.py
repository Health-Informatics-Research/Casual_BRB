import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.axisbelow'] = True

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# =========================================================
# 左栏：数据效率学习曲线
# =========================================================
fracs        = np.array([20, 40, 60, 80, 100])
causal_auc   = np.array([0.767, 0.785, 0.795, 0.802, 0.807])
random_auc   = np.array([0.680, 0.705, 0.720, 0.730, 0.739])

x_smooth     = np.linspace(20, 100, 300)
causal_spl   = make_interp_spline(fracs, causal_auc)(x_smooth)
random_spl   = make_interp_spline(fracs, random_auc)(x_smooth)

ax1.plot(x_smooth, causal_spl, color='#C62828', linewidth=2.5,
         label='Causal-BRB (Causal Prior)')
ax1.fill_between(x_smooth, causal_spl - 0.008, causal_spl + 0.008,
                 color='#C62828', alpha=0.15)

ax1.plot(x_smooth, random_spl, color='#1565C0', linewidth=2.2,
         linestyle='--', label='Random Graph (Baseline)')
ax1.fill_between(x_smooth, random_spl - 0.012, random_spl + 0.012,
                 color='#1565C0', alpha=0.10)

# 锚点
ax1.scatter([20],  [0.767], color='#C62828', s=80, zorder=5,
            edgecolor='black', linewidth=0.8)
ax1.scatter([100], [0.739], color='#1565C0', s=80, zorder=5,
            edgecolor='black', linewidth=0.8)

# 100% baseline 参考线
ax1.axhline(y=0.739, color='gray', linestyle=':', linewidth=1.3, alpha=0.8)

# 数据效率标注
ax1.annotate('Data Efficiency:\n20% Causal > 100% Baseline',
             xy=(20, 0.767), xytext=(38, 0.747),
             fontsize=9.5, fontweight='bold', color='#C62828',
             arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.3),
             bbox=dict(boxstyle='round,pad=0.3', fc='white',
                       ec='#C62828', lw=1.1, alpha=0.92))

ax1.set_xlabel('Fraction of Training Data (%)', fontsize=11, fontweight='bold')
ax1.set_ylabel('ROC-AUC Score', fontsize=11, fontweight='bold')
ax1.set_title('(a) Data Efficiency & Few-shot Learning', fontsize=12,
              fontweight='bold', pad=10)
ax1.set_xticks(fracs)
ax1.set_xlim(15, 105)
ax1.set_ylim(0.655, 0.825)
ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax1.grid(linestyle='--', alpha=0.45)
ax1.tick_params(labelsize=10)

# =========================================================
# 右栏：DE 收敛轨迹对比（叠加 V4 随机初始化）
# =========================================================
gens   = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80,  90,  100, 110, 120])

# Causal-BRB：三段式初始化，收敛快且损失低
loss_causal = np.array([0.4352, 0.2811, 0.1804, 0.1523, 0.1190,
                         0.1001, 0.0950, 0.0880, 0.0820, 0.0781,
                         0.0750, 0.0732, 0.0732])

# V4 随机图变体：随机初始化，起点更高、收敛慢、最终损失更高
loss_v4     = np.array([0.4780, 0.3650, 0.2810, 0.2390, 0.1980,
                         0.1720, 0.1560, 0.1430, 0.1340, 0.1270,
                         0.1220, 0.1195, 0.1190])

x_s = np.linspace(gens.min(), gens.max(), 300)

spl_causal = make_interp_spline(gens, loss_causal, k=3)
y_causal   = np.maximum(spl_causal(x_s), 0.0732)

spl_v4     = make_interp_spline(gens, loss_v4, k=3)
y_v4       = np.maximum(spl_v4(x_s), 0.1190)

# Causal-BRB 收敛曲线
ax2.plot(x_s, y_causal, color='#00695C', linewidth=2.5,
         label='Causal-BRB (3-stage Init)')
ax2.fill_between(x_s, y_causal, 0.47, color='#00695C', alpha=0.08)

# V4 随机初始化收敛曲线
ax2.plot(x_s, y_v4, color='#E65100', linewidth=2.2, linestyle='--',
         label='Random Graph V4 (Random Init)')

# 收敛平台线
ax2.axhline(y=0.0732, color='#00695C', linestyle=':',
            linewidth=1.5, alpha=0.8)
ax2.axhline(y=0.1190, color='#E65100', linestyle=':',
            linewidth=1.5, alpha=0.8)

# 最终损失标注
ax2.text(122, 0.0732, '0.0732', va='center', ha='left',
         fontsize=9, color='#00695C', fontweight='bold')
ax2.text(122, 0.1190, '0.1190', va='center', ha='left',
         fontsize=9, color='#E65100', fontweight='bold')

# 阶段标注
ax2.text(18, 0.30, 'Phase 1:\nRapid Escape',
         fontsize=9, color='#004D40', fontweight='bold', ha='center')
ax2.text(85, 0.165, 'Phase 2:\nStable Convergence',
         fontsize=9, color='#BF360C', fontweight='bold', ha='center')

# 优势差距标注（右侧终点）
ax2.annotate('',
             xy=(118, 0.0732), xytext=(118, 0.1190),
             arrowprops=dict(arrowstyle='<->', color='#424242', lw=1.4))
ax2.text(106, 0.096, 'Δ=0.046\nlower loss',
         ha='center', va='center', fontsize=9, color='#212121',
         bbox=dict(facecolor='white', edgecolor='#424242',
                   linewidth=0.8, alpha=0.9, boxstyle='round,pad=0.25'))

ax2.set_xlabel('Generation (DE Algorithm Iterations)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Composite Objective Loss', fontsize=11, fontweight='bold')
ax2.set_title('(b) Convergence Trajectory in 396-D Parameter Space',
              fontsize=12, fontweight='bold', pad=10)
ax2.set_xlim(0, 128)
ax2.set_ylim(0.04, 0.47)
ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax2.grid(linestyle='--', alpha=0.45)
ax2.tick_params(labelsize=10)

# =========================================================
# 总标题与保存
# =========================================================

plt.tight_layout()

# =========================================================
# 保存为 PNG 和 PDF 到 Picture 文件夹
# =========================================================
png_path = 'Picture/Figure_5_6_Dual_Panel.png'
pdf_path = 'Picture/Figure_5_6_Dual_Panel.pdf'

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')

print(f"✅ 图片已成功生成并保存为 PNG 和 PDF 格式于 Picture 文件夹：")
print(f"   - {png_path}")
print(f"   - {pdf_path}")

plt.show()