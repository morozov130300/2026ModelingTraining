# -*- coding: utf-8 -*-
"""
t4p1.py — 问题4补充可视化：个体风险预测结果深度分析
====================================================
基于 问题4_个体风险预测结果.csv 的数据可视化。
不依赖模型，仅从CSV读取结果进行展示。
"""
import warnings, os, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# ── 字体 ──
_CN_FP = None
for _fp in ["C:/Windows/Fonts/msyh.ttc","C:/Windows/Fonts/simhei.ttf"]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp); _CN_FP = fm.FontProperties(fname=_fp); break
if not _CN_FP:
    for _f in fm.fontManager.ttflist:
        if any(k in _f.name for k in ["YaHei","SimHei","PingFang"]):
            _CN_FP = fm.FontProperties(family=_f.name); break
plt.rcParams["font.family"] = "sans-serif"; plt.rcParams["axes.unicode_minus"] = False

# ── 路径 ──
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CUR_DIR, "output", "问题4")
CSV_PATH = os.path.join(OUTPUT_DIR, "问题4_个体风险预测结果.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RISK_LABELS = ["极低风险","低风险","中等风险","高风险","极高风险"]
RISK_COLORS = ["#2ECC71","#58D68D","#F39C12","#E67E22","#E74C3C"]
RISK_LEVEL_ORDER = {lbl: i for i, lbl in enumerate(RISK_LABELS)}

SEP = "=" * 70
SEP2 = "-" * 60


def load_data():
    """加载CSV"""
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df["风险等级排序"] = df["风险等级"].map(RISK_LEVEL_ORDER)
    df = df.sort_values("预测糖尿病风险概率", ascending=False).reset_index(drop=True)
    df["排名"] = df.index + 1
    print("  加载 %d 条个体风险记录" % len(df))
    print("  概率范围: %.4f ~ %.4f, 均值=%.4f" %
          (df["预测糖尿病风险概率"].min(), df["预测糖尿病风险概率"].max(),
           df["预测糖尿病风险概率"].mean()))
    return df


# ══════════════════════════════════════════════════════════
#  图1: 个体风险排序阶梯图（141人的风险全貌）
# ══════════════════════════════════════════════════════════

def plot_risk_ladder(df):
    """
    个体风险排序阶梯图：
    - X轴：按风险从高到低排序的个体排名
    - Y轴：预测风险概率
    - 每个条颜色标记风险等级
    - 标注Top20%阈值线
    - 标注关键个体（最高风险/阈值附近/最低风险）
    """
    n = len(df)
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = [RISK_COLORS[RISK_LEVEL_ORDER.get(r, 2)] for r in df["风险等级"]]
    x = df["排名"].values
    y = df["预测糖尿病风险概率"].values

    # 阶梯条形图
    bars = ax.bar(x, y, width=0.7, color=colors, alpha=0.85, edgecolor="white", linewidth=0.3)

    # Top20%阈值线
    thr_20 = np.percentile(y, 80)
    ax.axhline(thr_20, color="#E74C3C", ls="--", lw=2, alpha=0.7,
               label="Top20%%阈值 = %.4f (n=%d)" % (thr_20, int(n * 0.2)))

    # 风险等级色块标注条
    legend_patches = []
    for i, (lbl, clr) in enumerate(zip(RISK_LABELS, RISK_COLORS)):
        n_lvl = (df["风险等级"] == lbl).sum()
        legend_patches.append(mpatches.Patch(color=clr, alpha=0.7,
                                             label="%s (n=%d)" % (lbl, n_lvl)))

    # 标注特殊个体
    # 最高风险
    ax.annotate("最高风险\nID=%d\nP=%.4f" % (df.loc[0, "id"], y[0]),
                xy=(1, y[0]), xytext=(15, y[0]+0.08),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                fontproperties=_CN_FP, fontsize=8, color="#E74C3C",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E74C3C", lw=0.5))

    # 最低风险
    ax.annotate("最低风险\nID=%d\nP=%.4f" % (df.loc[n-1, "id"], y[-1]),
                xy=(n, y[-1]), xytext=(n-55, y[-1]-0.08),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                fontproperties=_CN_FP, fontsize=8, color="#2ECC71",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ECC71", lw=0.5))

    # 阈值附近标注
    thr_idx = np.searchsorted(-y, -thr_20)  # 第一个低于阈值的索引
    if thr_idx < n:
        ax.annotate("阈值边界\nID=%d\nP=%.4f" % (df.loc[thr_idx, "id"], y[thr_idx]),
                    xy=(thr_idx+1, y[thr_idx]), xytext=(thr_idx+15, y[thr_idx]-0.1),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                    fontproperties=_CN_FP, fontsize=8, color="#E67E22",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E67E22", lw=0.5))

    # 图例
    ax.legend(handles=legend_patches, loc="upper right", prop=_CN_FP, fontsize=8,
              ncol=2, framealpha=0.9)

    ax.set_xlabel("风险排名 (1=最高风险)", fontproperties=_CN_FP)
    ax.set_ylabel("预测糖尿病风险概率", fontproperties=_CN_FP)
    ax.set_title("附件2个体糖尿病风险排序阶梯图（n=141）", fontproperties=_CN_FP, fontweight="bold")
    ax.set_xlim([0, n+1])
    ax.set_ylim([0, 1.0])

    # 背景色块分高风险/低风险区
    ax.axvspan(0.5, 29.5, alpha=0.05, color="#E74C3C", label="Top20%高分区")
    ax.axvspan(29.5, n+0.5, alpha=0.03, color="#3498DB", label="其余区")

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题4p1_个体风险排序阶梯图.png"), dpi=300)
    plt.close()
    print("  -> 个体风险排序阶梯图.png")


# ══════════════════════════════════════════════════════════
#  图2: 风险等级概率分布箱线图
# ══════════════════════════════════════════════════════════

def plot_risk_box(df):
    """
    各风险等级内的预测概率分布箱线图。
    展示各等级的离散程度和阈值区间。
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    data_groups = []
    positions = []
    labels_show = []
    colors_show = []
    for i, lbl in enumerate(RISK_LABELS):
        subset = df[df["风险等级"] == lbl]["预测糖尿病风险概率"].values
        if len(subset) > 0:
            data_groups.append(subset)
            positions.append(i)
            labels_show.append("%s\n(n=%d)" % (lbl, len(subset)))
            colors_show.append(RISK_COLORS[i])

    bp = ax.boxplot(data_groups, positions=positions, patch_artist=True,
                     widths=0.5, showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="white",
                                    markeredgecolor="black", markersize=6))
    for patch, color in zip(bp["boxes"], colors_show):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # 散点叠加（jitter）
    for i, data in enumerate(data_groups):
        jitter = np.random.normal(i, 0.04, size=len(data))
        ax.scatter(jitter, data, alpha=0.4, s=15, color=colors_show[i],
                   edgecolors="white", linewidth=0.3, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_show, fontproperties=_CN_FP, fontsize=9)
    ax.set_ylabel("预测糖尿病风险概率", fontproperties=_CN_FP)
    ax.set_title("各风险等级的概率分布箱线图", fontproperties=_CN_FP, fontweight="bold")
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题4p1_风险等级箱线图.png"), dpi=300)
    plt.close()
    print("  -> 风险等级箱线图.png")


# ══════════════════════════════════════════════════════════
#  图3: 累积风险覆盖曲线
# ══════════════════════════════════════════════════════════

def plot_cumulative_risk(df):
    """
    累积风险覆盖曲线：
    - X轴：按风险排序的累积人数占比
    - Y轴：累积风险（概率之和的占比）
    - 展示"少数人承担了大部分风险"
    """
    y = df["预测糖尿病风险概率"].values
    cum_risk = np.cumsum(y) / np.sum(y)
    pct = np.arange(1, len(y)+1) / len(y) * 100

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(pct, cum_risk, "#E74C3C", lw=2.5, label="累积风险曲线")
    ax.fill_between(pct, cum_risk, alpha=0.12, color="#E74C3C")
    # 对角线（均匀分布参考线）
    ax.plot(pct, pct/100, "gray", ls="--", lw=1.2, label="均匀分布参考线")

    # 标注关键点
    for label_pct in [10, 20, 30, 50]:
        idx = int(label_pct / 100 * len(y)) - 1
        if idx >= 0:
            cum_val = cum_risk[idx] * 100
            ax.plot([label_pct, label_pct], [0, cum_val/100],
                    "gray", ls=":", lw=0.8, alpha=0.5)
            ax.plot(label_pct, cum_val/100, "o", color="#E74C3C", ms=6)
            ax.annotate("Top%.0f%%=%.1f%%" % (label_pct, cum_val),
                        xy=(label_pct, cum_val/100),
                        xytext=(label_pct+5, cum_val/100+0.03),
                        fontproperties=_CN_FP, fontsize=9,
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xlabel("累积人群占比 (%)", fontproperties=_CN_FP)
    ax.set_ylabel("累积风险占比", fontproperties=_CN_FP)
    ax.set_title("累积风险覆盖曲线", fontproperties=_CN_FP, fontweight="bold")
    ax.legend(loc="lower right", prop=_CN_FP)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题4p1_累积风险覆盖曲线.png"), dpi=300)
    plt.close()
    print("  -> 累积风险覆盖曲线.png")


# ══════════════════════════════════════════════════════════
#  图4: 高风险 vs 非高风险 提琴图+棒棒糖图
# ══════════════════════════════════════════════════════════

def plot_high_vs_low(df):
    """
    高风险组 vs 非高风险组的对比图：
    - 左侧：提琴图展示两组概率分布
    - 右侧：棒棒糖图展示每个个体的概率（分组显示）
    """
    high = df[df["是否高风险(Top20%)"] == "是"]["预测糖尿病风险概率"].values
    low = df[df["是否高风险(Top20%)"] == "否"]["预测糖尿病风险概率"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # 左图：提琴图
    parts = ax1.violinplot([low, high], positions=[0, 1], showmedians=True,
                           showmeans=True, widths=0.5)
    for pc, color in zip(parts["bodies"], ["#3498DB", "#E74C3C"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
    for partname in ["cbars", "cmins", "cmaxes", "cmedians", "cmeans"]:
        if partname in parts:
            parts[partname].set_color("black")
            parts[partname].set_linewidth(1.5)

    # 散点叠加
    ax1.scatter(np.random.normal(0, 0.04, len(low)), low, alpha=0.4, s=12,
                color="#3498DB", edgecolors="white", linewidth=0.3)
    ax1.scatter(np.random.normal(1, 0.04, len(high)), high, alpha=0.6, s=12,
                color="#E74C3C", edgecolors="white", linewidth=0.3)

    # 统计标注
    ax1.text(0, low.mean(), "均值=%.4f" % low.mean(), ha="center", va="bottom",
             fontproperties=_CN_FP, fontsize=8, color="#3498DB")
    ax1.text(1, high.mean(), "均值=%.4f" % high.mean(), ha="center", va="bottom",
             fontproperties=_CN_FP, fontsize=8, color="#E74C3C")

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["非高风险\n(n=%d)" % len(low),
                         "高风险Top20%%\n(n=%d)" % len(high)], fontproperties=_CN_FP)
    ax1.set_ylabel("预测糖尿病风险概率", fontproperties=_CN_FP)
    ax1.set_title("高风险 vs 非高风险 — 概率分布", fontproperties=_CN_FP, fontweight="bold")
    ax1.set_ylim([-0.02, 1.02])

    # 右图：棒棒糖图
    n_high, n_low = len(high), len(low)
    y_all = np.concatenate([low[::-1], high[::-1]])
    x_all = np.arange(len(y_all))
    colors_all = ["#3498DB"] * n_low + ["#E74C3C"] * n_high
    ax2.stem(x_all, y_all, linefmt="gray", markerfmt=" ", basefmt=" ")
    ax2.scatter(x_all, y_all, c=colors_all, s=8, alpha=0.6, edgecolors="none")

    ax2.axhline(y=0.6340, color="black", ls="--", lw=1.5, alpha=0.5,
                label="Thr=0.634 (Top20%)")
    ax2.set_xlabel("个体序号 (左=低风险, 右=高风险)", fontproperties=_CN_FP)
    ax2.set_ylabel("预测糖尿病风险概率", fontproperties=_CN_FP)
    ax2.set_title("各风险概率棒棒糖图", fontproperties=_CN_FP, fontweight="bold")
    ax2.legend(prop=_CN_FP, fontsize=8)
    ax2.set_ylim([-0.02, 1.02])
    ax2.set_xlim([-1, len(y_all)])
    ax2.axvline(n_low-0.5, color="gray", ls=":", lw=1, alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题4p1_高风险对比图.png"), dpi=300)
    plt.close()
    print("  -> 高风险对比图.png")


# ══════════════════════════════════════════════════════════
#  图5: 风险等级转换矩阵热力图（年龄×风险等级）
# ══════════════════════════════════════════════════════════

def plot_risk_confusion_heatmap(df):
    """
    将原始ID排序与风险等级映射为热力条形图。
    每行一个个体，颜色=风险等级，展示风险在人群中的分布模式。
    """
    df_sorted = df.sort_values("预测糖尿病风险概率", ascending=False).reset_index(drop=True)
    n = len(df_sorted)

    fig, ax = plt.subplots(figsize=(14, 5))

    # 创建热力条
    color_matrix = np.array([RISK_COLORS[RISK_LEVEL_ORDER[r]] for r in df_sorted["风险等级"]])
    # 绘制为1×n的像素条
    for i in range(n):
        ax.bar(i, 1, width=1, color=color_matrix[i], edgecolor="none", alpha=0.85)

    # 阈值分割线
    thr_idx = int(n * 0.2)
    ax.axvline(thr_idx - 0.5, color="black", ls="--", lw=2, alpha=0.7,
               label="Top20%%分割线 (n=%d)" % thr_idx)

    # 图例
    legend_patches = []
    for lbl, clr in zip(RISK_LABELS, RISK_COLORS):
        n_lvl = (df_sorted["风险等级"] == lbl).sum()
        legend_patches.append(mpatches.Patch(color=clr, alpha=0.85,
                                             label="%s (n=%d)" % (lbl, n_lvl)))
    legend_patches.append(mpatches.Patch(color="none", label=" "))
    legend_patches.append(mpatches.Patch(color="none",
                                         label="高风险概率范围: %.4f ~ %.4f" %
                                         (df_sorted["预测糖尿病风险概率"].min(),
                                          df_sorted["预测糖尿病风险概率"].max())))
    ax.legend(handles=legend_patches, loc="upper right", prop=_CN_FP, fontsize=8,
              framealpha=0.9)

    ax.set_xlim([-1, n])
    ax.set_ylim([0, 1.5])
    ax.set_xlabel("个体排名 (1=最高风险 → %d=最低风险)" % n, fontproperties=_CN_FP)
    ax.set_yticks([])
    ax.set_title("附件2个体风险等级全景图（按风险排序）", fontproperties=_CN_FP, fontweight="bold")

    # 添加风险概率刻度
    ax2 = ax.twinx()
    # 在底部标记关键ID
    tick_positions = []
    tick_labels = []
    for pos in [0, thr_idx-1, n//2, n-1]:
        if 0 <= pos < n:
            tick_positions.append(pos)
            tick_labels.append("ID=%d\nP=%.3f" %
                               (df_sorted.loc[pos, "id"], df_sorted.loc[pos, "预测糖尿病风险概率"]))
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontproperties=_CN_FP, fontsize=7, ha="center")
    ax2.set_xlim([-1, n])

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题4p1_风险等级全景图.png"), dpi=300)
    plt.close()
    print("  -> 风险等级全景图.png")


# ══════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════

def main():
    print(SEP)
    print("  问题4补充可视化 — 个体风险预测深度分析")
    print(SEP)

    df = load_data()

    print("\n" + SEP2)
    print("  [生成可视化]")
    print(SEP2)

    plot_risk_ladder(df)
    plot_risk_box(df)
    plot_cumulative_risk(df)
    plot_high_vs_low(df)
    plot_risk_confusion_heatmap(df)

    print("\n" + SEP)
    print("  生成完成")
    print(SEP)
    print("\n  输出文件:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if "p1" in f:
            print("    " + f)


if __name__ == "__main__":
    main()
