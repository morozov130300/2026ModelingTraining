#!/usr/bin/env python3
"""从 t1/output 已有结果生成分析报告所需图表（PNG）。

依赖：numpy, pandas, matplotlib（运行 t1.py 时已安装）
运行：python make_charts.py
输出：t1/output/charts/*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
CHART = OUT / "charts"
CHART.mkdir(parents=True, exist_ok=True)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "DejaVu Sans"

TYPES = ["RealTimeInference", "BatchInference", "AITraining"]
TYPE_LABEL = {"RealTimeInference": "RealTime", "BatchInference": "Batch", "AITraining": "Training"}
TYPE_COLOR = {"RealTimeInference": "#4C78A8", "BatchInference": "#F58518", "AITraining": "#54A24B"}
REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]


def load_pivot(name: str) -> pd.DataFrame:
    df = pd.read_csv(OUT / "statistics" / name, index_col=0)
    df = df.drop(index="All", errors="ignore")
    df = df.drop(columns="All", errors="ignore")
    return df.reindex(columns=TYPES, index=REGIONS)


def chart_task_type_shares():
    df = pd.read_csv(OUT / "statistics" / "task_type_summary.csv", index_col=0)
    df = df.reindex(TYPES)
    labels = [TYPE_LABEL[t] for t in TYPES]
    x = np.arange(len(TYPES))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = [
        (df["TaskCount_Share"] * 100, "Task count", "#4C78A8"),
        (df["GPU_h_Share"] * 100, "GPU-hour", "#F58518"),
        (df["IT_MWh_Share"] * 100, "IT energy", "#54A24B"),
    ]
    for i, (vals, name, color) in enumerate(bars):
        ax.bar(x + (i - 1) * width, vals, width, label=name, color=color)
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Task type shares: count vs GPU-hour vs IT energy")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHART / "task_type_shares.png", dpi=180)
    plt.close(fig)


def chart_region_type_gpuh():
    df = load_pivot("region_type_gpuh.csv")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(REGIONS))
    for t in TYPES:
        ax.bar(REGIONS, df[t], bottom=bottom, label=TYPE_LABEL[t], color=TYPE_COLOR[t])
        bottom += df[t].to_numpy()
    ax.set_ylabel("GPU-hour")
    ax.set_title("GPU-hour by region and task type")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHART / "region_type_gpuh.png", dpi=180)
    plt.close(fig)


def chart_region_type_energy():
    df = load_pivot("region_type_it_energy.csv")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(REGIONS))
    for t in TYPES:
        ax.bar(REGIONS, df[t], bottom=bottom, label=TYPE_LABEL[t], color=TYPE_COLOR[t])
        bottom += df[t].to_numpy()
    ax.set_ylabel("IT energy (MWh)")
    ax.set_title("IT energy by region and task type")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHART / "region_type_energy.png", dpi=180)
    plt.close(fig)


def chart_region_structure():
    df = pd.read_csv(OUT / "statistics" / "region_task_type_share.csv", index_col=0)
    df = df.reindex(REGIONS)[TYPES]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(REGIONS))
    for t in TYPES:
        ax.bar(REGIONS, df[t] * 100, bottom=bottom, label=TYPE_LABEL[t], color=TYPE_COLOR[t])
        bottom += df[t].to_numpy() * 100
    ax.set_ylabel("Share (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Task type structure by region")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHART / "region_structure.png", dpi=180)
    plt.close(fig)


def chart_power_margin():
    df = pd.read_csv(OUT / "statistics" / "it_power_margin_summary.csv", index_col=0)
    df = df.reindex(REGIONS)
    x = np.arange(len(REGIONS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (col, name, color) in enumerate([("min", "Min", "#C0392B"), ("mean", "Mean", "#F39C12"), ("max", "Max", "#27AE60")]):
        ax.bar(x + (i - 1) * width, df[col], width, label=name, color=color)
        for xi, v in zip(x + (i - 1) * width, df[col]):
            ax.text(xi, v + 5, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(REGIONS)
    ax.set_ylabel("IT power margin (MW)")
    ax.set_title("IT power margin by region (min / mean / max)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHART / "power_margin.png", dpi=180)
    plt.close(fig)


def chart_forecast_total():
    df = pd.read_csv(OUT / "forecast" / "predictions_2376_2399.csv")
    gbdt = df[df["Model"].eq("GBDT")].groupby("Hour")[["Actual_GPU_h", "Predicted_GPU_h"]].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gbdt.index, gbdt["Actual_GPU_h"], marker="o", ms=3, label="Actual", color="#C0392B")
    ax.plot(gbdt.index, gbdt["Predicted_GPU_h"], marker="s", ms=3, label="Predicted (GBDT)", color="#2471A3")
    ax.set_xlabel("Hour")
    ax.set_ylabel("GPU-hour")
    ax.set_title("Forecast vs actual (all regions and task types, 2376-2399)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHART / "forecast_total.png", dpi=180)
    plt.close(fig)


def main():
    chart_task_type_shares()
    chart_region_type_gpuh()
    chart_region_type_energy()
    chart_region_structure()
    chart_power_margin()
    chart_forecast_total()
    print(f"图表已生成到 {CHART.resolve()}")


if __name__ == "__main__":
    main()
