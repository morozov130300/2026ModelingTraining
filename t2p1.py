# -*- coding: utf-8 -*-
import os, matplotlib, numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager as fm
from matplotlib.path import Path

_CN_FP = None
for _fp in ["C:/Windows/Fonts/msyh.ttc","C:/Windows/Fonts/simhei.ttf"]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp); _CN_FP = fm.FontProperties(fname=_fp); _CN_FP.set_size(8); break
if not _CN_FP:
    for _f in fm.fontManager.ttflist:
        if any(k in _f.name for k in ["YaHei","SimHei","PingFang"]):
            _CN_FP = fm.FontProperties(family=_f.name); _CN_FP.set_size(8); break

def fp(sz):
    f = fm.FontProperties(fname=_CN_FP.get_file()) if _CN_FP.get_file() else fm.FontProperties(family=_CN_FP.get_name())
    f.set_size(sz); return f

ROOT = "体检指标\n(39项自变量)"
CATS = [
    ("人口学特征\n(2项)", "#E74C3C", ["性别","年龄"]),
    ("糖脂代谢\n(4项)", "#3498DB", ["甘油三酯","总胆固醇","高密度脂蛋白\n胆固醇","低密度脂蛋白\n胆固醇"]),
    ("肝功能\n(8项)", "#2ECC71", ["天门冬氨酸\n氨基转换酶","丙氨酸\n氨基转换酶","碱性磷酸酶","r-谷氨酰基\n转换酶",
                              "总蛋白","白蛋白","球蛋白","白球比例"]),
    ("肾功能\n(3项)", "#F39C12", ["尿素","肌酐","尿酸"]),
    ("血常规\n(17项)", "#9B59B6", ["白细胞计数","红细胞计数","血红蛋白","红细胞压积","红细胞\n平均体积",
                                "红细胞平均\n血红蛋白量","红细胞平均\n血红蛋白浓度","红细胞体积\n分布宽度",
                                "血小板计数","血小板\n平均体积","血小板体积\n分布宽度","血小板比积",
                                "中性粒细胞%","淋巴细胞%","单核细胞%","嗜酸细胞%","嗜碱细胞%"]),
    ("感染免疫\n(5项)", "#1ABC9C", ["乙肝表面抗原","乙肝表面抗体","乙肝e抗原","乙肝e抗体","乙肝核心抗体"]),
]
FIG_W, FIG_H = 22, 14
N = len(CATS)
cat_x = np.linspace(1.8, FIG_W - 1.8, N)
root_x, root_y = FIG_W / 2, FIG_H - 0.6
fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H), facecolor="#FAFAFA")
ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H); ax.axis("off")

def round_rect(ax, cx, cy, w, h, fc, ec="#333", lw=1, alpha=1, z=3, r=None):
    if r is None: r = min(w, h) * 0.15
    p = mpatches.FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                                boxstyle=mpatches.BoxStyle("Round", pad=r*0.3),
                                facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=z)
    ax.add_patch(p)

def round_rect_shadow(ax, cx, cy, w, h, z=2):
    p = mpatches.FancyBboxPatch((cx-w/2+0.06, cy-h/2-0.06), w, h,
                                boxstyle=mpatches.BoxStyle("Round", pad=min(w,h)*0.05),
                                facecolor="none", edgecolor="none", alpha=0.15, zorder=z, fc="#000")
    ax.add_patch(p)

def cbezier(ax, x1, y1, x2, y2, color="#999", lw=1, alpha=0.5, z=1):
    mid = (y1 + y2) / 2
    verts = [(x1, y1), (x1, mid), (x2, mid), (x2, y2)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    p = Path(verts, codes); patch = mpatches.PathPatch(p, facecolor="none", edgecolor=color, linewidth=lw, alpha=alpha, zorder=z)
    ax.add_patch(patch)

root_w, root_h = 3.0, 0.8
round_rect_shadow(ax, root_x, root_y, root_w, root_h, z=4)
round_rect(ax, root_x, root_y, root_w, root_h, "#2C3E50", "#1A252F", lw=2.5, z=5)
ax.text(root_x, root_y, ROOT, fontproperties=fp(12), ha="center", va="center", color="white", fontweight="bold", zorder=6)
for i in range(50):
    ax.plot(np.random.uniform(0, FIG_W), np.random.uniform(0, FIG_H),
            'o', ms=1, alpha=0.1, color="#3498DB", zorder=0)
cat_y = FIG_H - 2.5
for ci, (name, color, vars_list) in enumerate(CATS):
    cx = cat_x[ci]
    nv = len(vars_list)
    cbezier(ax, root_x, root_y - root_h/2, cx, cat_y, color, lw=1.8, alpha=0.6)
    cat_w, cat_h = 2.6, 0.65
    round_rect_shadow(ax, cx, cat_y, cat_w, cat_h)
    round_rect(ax, cx, cat_y, cat_w, cat_h, color, "white", lw=2, z=10)
    ax.text(cx, cat_y, name, fontproperties=fp(10), ha="center", va="center", color="white", fontweight="bold", zorder=11)
    bg_w = cat_w * 0.7
    bg_h = nv * 0.36 + 0.25
    bg_y = cat_y - cat_h/2 - bg_h/2 - 0.1
    round_rect(ax, cx, bg_y, bg_w, bg_h, color, color, lw=0.5, alpha=0.08, z=2)
    for vi, vname in enumerate(vars_list):
        vy = cat_y - 0.55 - vi * 0.36
        branch_x = cx - cat_w * 0.25
        cbezier(ax, cx, vy, branch_x, vy, color, lw=0.8, alpha=0.3, z=3)
        ax.plot(branch_x, vy, 'o', ms=3, color=color, alpha=0.5, zorder=4)
        card_w, card_h = 1.6, 0.30
        card_cx = branch_x - card_w/2
        round_rect(ax, card_cx - card_w/2, vy, 0.04, card_h, color, "none", lw=0, z=8)
        round_rect(ax, card_cx, vy, card_w, card_h, "#FFFFFF", color, lw=0.8, z=8)
        ax.text(card_cx, vy, vname, fontproperties=fp(6), ha="center", va="center", color="#2C3E50", zorder=9)

leg_y = 0.5
for ci, (name, color, _) in enumerate(CATS):
    lx = ci * 1.1 + 1.5
    ax.plot(lx, leg_y, 's', ms=12, color=color, alpha=0.85, zorder=20)
    ax.text(lx + 0.25, leg_y, name.replace("\n"," "), fontproperties=fp(6.5),
            ha="left", va="center", color="#333", zorder=20)
ax.text(FIG_W/2, 0.1, "颜色代表不同临床类别 | 卡片左侧色条标识类别归属",
        fontproperties=fp(7), ha="center", va="center", color="#999", zorder=20, style="italic")
ax.text(FIG_W/2, FIG_H - 0.05, "体检指标临床分类流程图",
        fontproperties=fp(16), ha="center", va="center", color="#2C3E50", fontweight="bold", zorder=20)
ax.text(FIG_W/2, FIG_H - 0.5, "39项自变量 → 6大临床类别",
        fontproperties=fp(8), ha="center", va="center", color="#7F8C8D", zorder=20)
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "问题2")
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "问题2_变量分类流程图.png"), dpi=350, bbox_inches="tight",
            facecolor="#FAFAFA", edgecolor="none")
plt.close()
print("-> 输出/问题2/问题2_变量分类流程图.png")
