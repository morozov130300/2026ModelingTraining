#!/usr/bin/env python3
"""使用 t2/output_merged 中已有 CSV 重新生成问题二图表。

运行：
    python3 t2/dataToImg.py

显式指定其他结果目录：
    python3 t2/dataToImg.py --output-dir t2/output_gpu
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="根据问题二合并输出 CSV 重新生成图表")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "output_merged",
        help="结果目录，默认 t2/output_merged",
    )
    return parser.parse_args()


def require_columns(name: str, frame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} 缺少字段: {missing}")


def make_plots(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    metrics_path = output_dir / "reports" / "scenario_comparison.csv"
    schedule_path = output_dir / "schedule" / "scheme_b_carbon_aware_schedule.csv"
    usage_path = output_dir / "schedule" / "scheme_b_hourly_usage.csv"

    metrics = pd.read_csv(metrics_path)
    scheme_b = pd.read_csv(schedule_path)
    usage = pd.read_csv(usage_path)

    require_columns(
        str(metrics_path),
        metrics,
        ["Scenario", "OperatingCost_CNY", "CarbonEmission_tCO2"],
    )
    require_columns(
        str(schedule_path),
        scheme_b,
        ["SourceRegion", "ExecRegion", "GPU_h"],
    )
    require_columns(
        str(usage_path),
        usage,
        ["Hour", "Region", "GPU_Utilization_Percent"],
    )

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "SimSun"],
            "axes.unicode_minus": False,
        }
    )

    plotted = metrics.dropna(subset=["OperatingCost_CNY"])
    ax = plotted.set_index("Scenario")[["OperatingCost_CNY", "CarbonEmission_tCO2"]].plot(
        kind="bar",
        secondary_y="CarbonEmission_tCO2",
        figsize=(10, 5),
    )
    ax.set_ylabel("Operating cost (CNY)")
    ax.right_ax.set_ylabel("Carbon emissions (tCO2)")
    plt.tight_layout()
    plt.savefig(plot_dir / "scenario_comparison.png", dpi=180)
    plt.close()

    migration = scheme_b.groupby(
        ["SourceRegion", "ExecRegion"], observed=False
    )["GPU_h"].sum().unstack(fill_value=0)
    ax = migration.plot(
        kind="bar",
        stacked=True,
        figsize=(11, 5),
        color=[
            "#3E73B5",
            "#DD6E29",
            "#4DAF4A",
            "#F1B744",
            "#9467BD",
            "#5BC0BE",
        ][: len(migration.columns)],
    )
    ax.set_ylabel("Migrated / executed GPU-hour")
    plt.tight_layout()
    plt.savefig(plot_dir / "migration_flow_by_source.png", dpi=180)
    plt.close()

    regions = usage["Region"].drop_duplicates().tolist()
    if len(regions) != 6:
        raise ValueError(f"{usage_path} 应包含 6 个区域，实际为 {len(regions)} 个: {regions}")
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for axis, region in zip(axes.flat, regions):
        part = usage[(usage["Region"] == region) & usage["Hour"].between(2376, 2405)]
        axis.plot(part["Hour"], part["GPU_Utilization_Percent"])
        axis.set_title(region)
        axis.set_ylabel("GPU utilization (%)")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "gpu_utilization_2376_2405.png", dpi=180)
    plt.close(fig)

    print(f"完成。图表目录：{plot_dir.resolve()}")


def main() -> None:
    args = parse_args()
    make_plots(args.output_dir)


if __name__ == "__main__":
    main()
