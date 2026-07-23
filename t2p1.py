# -*- coding: utf-8 -*-
"""
t2p1.py — 问题2流程图：递进式血糖预测建模技术路线
根据问题2思路.pdf 绘制
输出: figures/问题2_技术路线图.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.path import Path
import os

_chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'WenQuanYi Micro Hei',
                  'Noto Sans CJK SC', 'PingFang SC', 'Heiti SC', 'DengXian']
_chosen = None
for f in _chinese_fonts:
    try:
        fm.findfont(f, fallback_to_default=False)
        _chosen = f
        break
    except Exception:
        continue
if _chosen:
    plt.rcParams['font.sans-serif'] = [_chosen] + plt.rcParams['font.sans-serif']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

FIGURE_DIR = './figures'
os.makedirs(FIGURE_DIR, exist_ok=True)

C = {
    'bg':       '#F7F9FC',
    'box_top':  '#1a5276',
    'box_mid':  '#2E86AB',
    'box_grp':  '#D4E6F1',
    'box_grp_b':'#A9CCE3',
    'box_mlr':  '#85C1E9',
    'box_ridge':'#52BE80',
    'box_rf':   '#F39C12',
    'box_result':'#E74C3C',
    'arrow':    '#5D6D7E',
    'white':    '#FFFFFF',
}

def draw_box(ax, cx, cy, w, h, fc, text='', fs=11, fw='normal', fc_text='black', ec=None):
    if ec is None: ec = fc
    rect = mpatches.FancyBboxPatch((cx-w/2, cy-h/2), w, h,
        boxstyle="round,pad=0.12", facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, fontweight=fw, color=fc_text, zorder=5)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=2), zorder=3)

# ── 画布 ──
fig, ax = plt.subplots(1, 1, figsize=(9, 11))
ax.set_xlim(0, 10); ax.set_ylim(0, 12)
ax.axis('off'); ax.set_facecolor(C['bg']); fig.patch.set_facecolor(C['bg'])

ax.text(5, 11.3, '问题2：递进式血糖预测建模技术路线', ha='center', va='center',
        fontsize=17, fontweight='bold', color='#1a1a2e')

# ── 1. 顶部 ──
draw_box(ax, 5, 9.8, 6, 0.7, C['box_top'], text='问题1筛选出的主要变量（7个）', fs=12, fw='bold', fc_text=C['white'])
draw_arrow(ax, 5, 9.45, 5, 8.6)

draw_box(ax, 5, 7.8, 4.5, 0.6, C['box_mid'], text='血糖预测建模', fs=12, fw='bold', fc_text=C['white'])
draw_arrow(ax, 5, 7.5, 5, 6.8)

# ── 2. 分组框 ──
grp1 = mpatches.FancyBboxPatch((0.5, 3.4), 4.2, 3.2,
    boxstyle="round,pad=0.15", facecolor=C['box_grp'], edgecolor=C['box_grp_b'], linewidth=2)
ax.add_patch(grp1)
ax.text(2.6, 6.4, '统计模型组', ha='center', va='center', fontsize=13, fontweight='bold')

grp2 = mpatches.FancyBboxPatch((5.3, 4.0), 4.2, 2.6,
    boxstyle="round,pad=0.15", facecolor=C['box_grp'], edgecolor=C['box_grp_b'], linewidth=2)
ax.add_patch(grp2)
ax.text(7.4, 6.4, '非线性模型组', ha='center', va='center', fontsize=13, fontweight='bold')

# ── 3. 分支箭头 ──
for target_x, target_y in [(2.6, 6.1), (7.4, 6.1)]:
    path = Path([(5, 6.8), (5, 6.3), (target_x, 6.3), (target_x, 6.1)],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.LINETO])
    ax.add_patch(mpatches.PathPatch(path, facecolor='none', edgecolor=C['arrow'], lw=2))
    ax.annotate('', xy=(target_x, 6.1), xytext=(target_x, 6.05),
                arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=2))

# ── 4. 三个模型框 ──
# MLR（基础）
draw_box(ax, 2.6, 5.1, 3.2, 0.65, C['box_mlr'], text='多元线性回归 (MLR)\n[基础模型]', fs=11, fw='bold', fc_text='#1C2833')
draw_arrow(ax, 2.6, 5.75, 2.6, 5.45)

# Ridge箭头（递进关系）
ax.annotate('', xy=(2.6, 4.0), xytext=(2.6, 4.4),
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.5), zorder=3)
ax.text(3.05, 4.2, 'L2正则化改进', ha='left', va='center', fontsize=9,
        color='#E74C3C', fontstyle='italic', fontweight='bold')

# Ridge（改进）
draw_box(ax, 2.6, 3.65, 3.2, 0.65, C['box_ridge'],
         text='岭回归 (Ridge)\n[MLR+L2改进]', fs=11, fw='bold', fc_text=C['white'])

# RF（非线性）
draw_box(ax, 7.4, 5.1, 3.2, 0.65, C['box_rf'], text='随机森林回归 (RF)\n[非线性模型]', fs=11, fw='bold', fc_text=C['white'])
draw_arrow(ax, 7.4, 5.75, 7.4, 5.45)

# ── 5. 汇合箭头 ──
ax.annotate('', xy=(5, 2.6), xytext=(2.6, 3.0),
            arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=2))
ax.annotate('', xy=(5, 2.6), xytext=(5.15, 3.0),
            arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=2))
# 第二个从RF的汇合需要单独画
path_rf = Path([(7.4, 4.0), (7.4, 3.3), (5.15, 2.8), (5, 2.6)],
               [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.LINETO])
ax.add_patch(mpatches.PathPatch(path_rf, facecolor='none', edgecolor=C['arrow'], lw=2))
ax.annotate('', xy=(5, 2.6), xytext=(5, 2.55),
            arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=2))

# 汇合点
ax.plot(5, 2.6, 'o', color=C['arrow'], markersize=8, zorder=4)

# ── 6. 对比框 ──
draw_box(ax, 5, 2.0, 5.5, 0.6, C['box_mid'],
         text=r'模型对比：岭回归(改进线性) vs 随机森林(非线性)', fs=11, fw='bold', fc_text=C['white'])
draw_arrow(ax, 5, 2.55, 5, 2.3)

# ── 7. 指标框 ──
draw_arrow(ax, 5, 1.7, 5, 1.4)
draw_box(ax, 5, 0.9, 4, 0.55, C['box_result'],
         text=r'最优预测模型 ($R^2$/RMSE/MAE)', fs=12, fw='bold', fc_text=C['white'])

# ── 8. 底部说明 ──
ax.text(5, 0.2, '基于附件1全部数据整体建模（不划分训练/测试集，不使用附件2）',
        ha='center', va='center', fontsize=9, color='#7F8C8D', fontstyle='italic')

plt.tight_layout()
outpath = f'{FIGURE_DIR}/问题2_技术路线图.png'
plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor=C['bg'])
plt.close()
print(f"流程图已生成: {outpath}")
