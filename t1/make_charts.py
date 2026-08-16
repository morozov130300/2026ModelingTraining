#!/usr/bin/env python3
"""生成分析报告唯一保留的区域任务类型结构图。

从表 3（区域 × 类型任务数）和表 4（区域 × 类型 GPU-hour）读取数据，
生成按任务数与按 GPU-hour 的双面 100% 堆叠柱状图。

依赖：numpy, pandas, matplotlib（运行 t1.py 时已安装）
运行：python make_charts.py
输出：t1/output/charts/region_task_structure_count_vs_gpuh.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STATISTICS = HERE / "output" / "statistics"
CHARTS = HERE / "output" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
TYPES = ["RealTimeInference", "BatchInference", "AITraining"]
TYPE_LABELS = {"RealTimeInference": "实时推理", "BatchInference": "批量推理", "AITraining": "AI训练"}
TYPE_COLORS = {"RealTimeInference": "#D62728", "BatchInference": "#FF9F1C", "AITraining": "#2A9D8F"}


def configure_chinese_font() -> None:
    """优先使用常见中文字体；未发现时由 matplotlib 自动回退。"""
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Arial Unicode MS"]
    available = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_as_percentage(filename: str) -> pd.DataFrame:
    data = pd.read_csv(STATISTICS / filename, index_col=0)
    data = data.drop(index="All", errors="ignore").drop(columns="All", errors="ignore")
    data = data.reindex(index=REGIONS, columns=TYPES)
    return data.div(data.sum(axis=1), axis=0) * 100


def add_percentage_labels(ax, values: pd.DataFrame) -> None:
    bottom = np.zeros(len(REGIONS))
    for task_type in TYPES:
        segment = values[task_type].to_numpy()
        for pos, value, start in zip(np.arange(len(REGIONS)), segment, bottom):
            if value >= 4:
                ax.text(pos, start + value / 2, f"{value:.1f}%", ha="center", va="center", fontsize=8, color="#1F2937")
        bottom += segment


def draw_stacked_axis(ax, values: pd.DataFrame, title: str) -> None:
    bottom = np.zeros(len(REGIONS))
    for task_type in TYPES:
        segment = values[task_type].to_numpy()
        ax.bar(REGIONS, segment, bottom=bottom, color=TYPE_COLORS[task_type], width=0.72, label=TYPE_LABELS[task_type])
        bottom += segment
    add_percentage_labels(ax, values)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("占比 (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#94A3B8", alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=0)


def main() -> None:
    configure_chinese_font()
    count_share = load_as_percentage("region_type_task_count.csv")
    gpuh_share = load_as_percentage("region_type_gpuh.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)
    draw_stacked_axis(axes[0], count_share, "按任务数")
    draw_stacked_axis(axes[1], gpuh_share, "按 GPU-hour")
    axes[1].set_ylabel("")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    output = CHARTS / "region_task_structure_count_vs_gpuh.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"图表已生成：{output.resolve()}")


if __name__ == "__main__":
    main()
