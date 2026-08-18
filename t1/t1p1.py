#!/usr/bin/env python3
"""问题一综合可视化：末 24 小时调度甘特图、区域利用率曲线与预测对比。

运行：
    python t1p1.py
    python t1p1.py --output-dir output --data-dir ../题目

输入为 t1.py 已生成的 output 目录；若提供 data-dir，则尝试叠加原始基线 GPU 利用率。
输出：output/plots/t1p1_visualization.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
TASK_TYPES = ["RealTimeInference", "BatchInference", "AITraining"]
TYPE_LABELS = {
    "RealTimeInference": "实时推理",
    "BatchInference": "批量推理",
    "AITraining": "AI训练",
}
TYPE_COLORS = {
    "RealTimeInference": "#E45756",
    "BatchInference": "#F2A541",
    "AITraining": "#2A9D8F",
}
REGION_COLORS = ["#264653", "#287271", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]
START_HOUR, END_HOUR = 2376, 2400


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=here / "output")
    parser.add_argument("--data-dir", type=Path, default=here.parent / "题目")
    parser.add_argument("--output-name", default="t1p1_visualization.png")
    return parser.parse_args()


def configure_fonts() -> None:
    """中文优先使用宋体，英文字母和数字使用 Times New Roman。"""
    available = {font.name for font in fm.fontManager.ttflist}
    chinese = next((name for name in ("SimSun", "宋体", "Noto Serif CJK SC") if name in available), "DejaVu Serif")
    english = "Times New Roman" if "Times New Roman" in available else "DejaVu Serif"
    plt.rcParams.update({
        "font.family": [english, chinese],
        "axes.unicode_minus": False,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "text.color": "#000000",
        "axes.titleweight": "bold",
    })


def load_inputs(output_dir: Path, data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    schedule = pd.read_csv(output_dir / "schedule" / "schedule.csv")
    usage = pd.read_csv(output_dir / "schedule" / "scheduled_hourly_usage.csv")
    predictions = pd.read_csv(output_dir / "forecast" / "predictions_2376_2399.csv")
    baseline = None
    source = data_dir / "region_time_data.xlsx"
    if source.exists():
        try:
            baseline = pd.read_excel(source, sheet_name="region_time_data")
        except (OSError, ValueError):
            baseline = None
    return schedule, usage, predictions, baseline


def style_axis(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#CBD5E1", alpha=0.42, linewidth=0.8)
    ax.set_axisbelow(True)


def draw_gantt(ax, schedule: pd.DataFrame) -> None:
    part = schedule[
        (schedule["ArrivalHour"].between(START_HOUR, END_HOUR - 1))
        & (schedule["FinishHour"] > START_HOUR)
    ].copy()
    part["left"] = part["StartHour"].clip(lower=START_HOUR)
    part["right"] = part["FinishHour"].clip(upper=END_HOUR)
    part = part[part["right"] > part["left"]].sort_values(["ExecRegion", "StartHour", "TaskID"])

    row_by_region = {region: i for i, region in enumerate(REGIONS)}
    counts = part.groupby("ExecRegion")["TaskID"].count().to_dict()
    for region, group in part.groupby("ExecRegion", sort=False):
        base = row_by_region.get(region, 0)
        for offset, (_, row) in enumerate(group.iterrows()):
            y = base + (offset % 3) * 0.22 - 0.22
            width = row["right"] - row["left"]
            ax.barh(y, width, left=row["left"], height=0.17,
                    color=TYPE_COLORS.get(row["TaskType"], "#64748B"),
                    edgecolor="white", linewidth=0.35, alpha=0.94)

    ax.set_yticks(range(len(REGIONS)))
    ax.set_yticklabels([f"{r}  ·  {counts.get(r, 0):,}项" for r in REGIONS], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(START_HOUR, END_HOUR)
    ax.set_ylabel("执行区域", fontsize=12)
    ax.set_title("末 24 小时任务排布 · 区域 × 时间 × 类型", loc="left", fontsize=16, pad=30, color="#000000")
    ax.axvspan(START_HOUR, END_HOUR, color="#EAF2FF", alpha=0.35, zorder=-2)
    ax.axvline(START_HOUR, color="#1D4ED8", linestyle="--", linewidth=1.1)
    ax.axvline(END_HOUR, color="#1D4ED8", linestyle="--", linewidth=1.1)
    ax.set_xticks(np.arange(START_HOUR, END_HOUR + 1, 4))
    ax.set_xticklabels([str(x) for x in np.arange(START_HOUR, END_HOUR + 1, 4)], fontsize=11)
    ax.set_xlabel("小时（Hour）", fontsize=12)
    ax.grid(axis="x", color="#CBD5E1", alpha=0.5, linewidth=0.75)
    ax.grid(axis="y", visible=False)
    style_axis(ax)

    handles = [patches.Patch(facecolor=TYPE_COLORS[t], label=TYPE_LABELS[t]) for t in TASK_TYPES]
    ax.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              frameon=False, fontsize=11, borderaxespad=0.0)


def draw_utilization(ax, usage: pd.DataFrame, baseline: Optional[pd.DataFrame]) -> None:
    part = usage[usage["Hour"].between(START_HOUR, END_HOUR - 1)].copy()
    for color, region in zip(REGION_COLORS, REGIONS):
        group = part[part["Region"].eq(region)].sort_values("Hour")
        ax.plot(group["Hour"], group["GPU_Utilization_Percent"], color=color, linewidth=2.0,
                marker="o", markersize=2.8, label=region)

    if baseline is not None and {"Hour", "Region", "GPU_Utilization_Percent"}.issubset(baseline.columns):
        base = baseline[baseline["Hour"].between(START_HOUR, END_HOUR - 1)]
        for color, region in zip(REGION_COLORS, REGIONS):
            group = base[base["Region"].eq(region)].sort_values("Hour")
            if not group.empty:
                ax.plot(group["Hour"], group["GPU_Utilization_Percent"], color=color,
                        linewidth=0.9, linestyle="--", alpha=0.38)

    ax.axvline(START_HOUR, color="#1D4ED8", linestyle="--", linewidth=1.1)
    ax.axvline(END_HOUR, color="#1D4ED8", linestyle="--", linewidth=1.1)
    ax.set_xlim(START_HOUR, END_HOUR)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("GPU 利用率 (%)", fontsize=12)
    ax.set_title("六区域 GPU 利用率曲线 · 共享时间轴", loc="left", fontsize=16, pad=12, color="#000000")
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=10)
    ax.set_xticks(np.arange(START_HOUR, END_HOUR + 1, 4))
    ax.set_xlabel("小时（Hour）", fontsize=12)
    style_axis(ax)


def draw_forecast(ax, predictions: pd.DataFrame) -> None:
    group = predictions[predictions["Model"].eq("GBDT")].groupby("Hour")[["Actual_GPU_h", "Predicted_GPU_h"]].sum()
    hours = group.index.to_numpy()
    actual = group["Actual_GPU_h"].to_numpy()
    predicted = group["Predicted_GPU_h"].to_numpy()
    actual_smooth = group["Actual_GPU_h"].rolling(window=3, center=True, min_periods=1).mean().to_numpy()
    predicted_smooth = group["Predicted_GPU_h"].rolling(window=3, center=True, min_periods=1).mean().to_numpy()

    ax.plot(hours, actual, color="#94A3B8", linewidth=0.9, alpha=0.38, label="实际值（逐时）")
    ax.plot(hours, predicted, color="#93C5FD", linewidth=0.9, alpha=0.42,
            linestyle="--", label="预测值（逐时）")
    ax.plot(hours, actual_smooth, color="#111827", linewidth=2.5, marker="o", markersize=3.4,
            label="实际趋势（3小时均值）")
    ax.plot(hours, predicted_smooth, color="#2563EB", linewidth=2.3, marker="s", markersize=3.0,
            linestyle="--", label="预测趋势（3小时均值）")
    ax.axvline(START_HOUR, color="#1D4ED8", linestyle="--", linewidth=1.1)
    ax.axvline(END_HOUR, color="#1D4ED8", linestyle="--", linewidth=1.1)
    ax.set_xlim(START_HOUR, END_HOUR)
    ax.set_ylabel("GPU-hour", fontsize=12)
    ax.set_xlabel("小时（Hour）", fontsize=12)
    ax.set_title("2376–2399 预测 vs 实际 · 逐时数据与 3 小时趋势", loc="left", fontsize=16, pad=12, color="#000000")
    ax.set_xticks(np.arange(START_HOUR, END_HOUR + 1, 4))
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=10)
    style_axis(ax)


def save_single_plot(drawer, output: Path, figsize: Tuple[float, float], *args) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    drawer(ax, *args)
    fig.tight_layout()
    fig.savefig(output, dpi=260, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_fonts()
    schedule, usage, predictions, baseline = load_inputs(args.output_dir, args.data_dir)
    output_dir = args.output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_single_plot(
        draw_gantt,
        output_dir / "gantt_2376_2400.png",
        (16, 7.6),
        schedule,
    )
    save_single_plot(
        draw_utilization,
        output_dir / "gpu_utilization_2376_2399.png",
        (16, 6.2),
        usage,
        baseline,
    )
    save_single_plot(
        draw_forecast,
        output_dir / "forecast_vs_actual_2376_2399.png",
        (16, 6.2),
        predictions,
    )
    print(f"三张独立图已生成：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
