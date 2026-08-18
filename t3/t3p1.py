#!/usr/bin/env python3
"""问题三论文可视化：六区域储能时序与储能敏感性曲线。

数据源默认使用 t3/output 下已经生成的 LP 结果，不重新求解模型：
    python t3/t3p1.py

输出：
    t3/output/plots/soc_charge_discharge_96h.png
    t3/output/plots/sensitivity_cost_grid.png
"""

from __future__ import annotations

import argparse
from pathlib import Path


FONT_CANDIDATES = {
    "zh": [
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/simsun.ttf"),
        Path("/system/fonts/NotoSerifCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
    ],
    "en": [
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/timesbd.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/times.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    ],
}

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
REGION_CN = {region: f"区域{region[-1]}" for region in REGIONS}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="生成问题三储能行为和敏感性可视化")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "output",
        help="问题三结果目录，默认 t3/output",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=96,
        help="储能时序窗口长度，默认 96 小时",
    )
    return parser.parse_args()


def configure_fonts(matplotlib):
    from matplotlib.font_manager import FontProperties, fontManager

    selected = {}
    for language, candidates in FONT_CANDIDATES.items():
        font_path = next((path for path in candidates if path.exists()), None)
        if font_path is None:
            label = "中文" if language == "zh" else "英文"
            raise FileNotFoundError(f"未找到{label}绘图字体，请安装宋体或 Times New Roman")
        fontManager.addfont(str(font_path))
        selected[language] = FontProperties(fname=str(font_path))
    matplotlib.rcParams.update(
        {
            "font.family": selected["zh"].get_name(),
            "font.serif": [selected["zh"].get_name()],
            "axes.unicode_minus": False,
        }
    )
    return selected["zh"], selected["en"]


def require_columns(name: str, frame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} 缺少字段: {missing}")


def apply_tick_font(axis, font_properties) -> None:
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontproperties(font_properties)


def load_inputs(output_dir: Path):
    import pandas as pd

    energy_path = output_dir / "energy" / "scheme2_storage_lp_energy_balance.csv"
    baseline_path = output_dir / "energy" / "scheme0_attachment_baseline_energy_balance.csv"
    sensitivity_path = output_dir / "reports" / "sensitivity_analysis.csv"
    energy = pd.read_csv(energy_path)
    baseline = pd.read_csv(baseline_path)
    sensitivity = pd.read_csv(sensitivity_path)
    require_columns(
        str(energy_path),
        energy,
        [
            "Hour", "Region", "RenewableCharge_MW", "GridCharge_MW",
            "DischargePower_MW", "SOC_MWh", "NetGridImport_MW",
        ],
    )
    require_columns(str(baseline_path), baseline, ["Hour", "Region", "NetGridImport_MW"])
    require_columns(
        str(sensitivity_path),
        sensitivity,
        [
            "Dimension", "CapacityFactor", "RenewableAlpha", "Feasible",
            "OperatingCost_CNY", "GridPurchase_MWh",
        ],
    )
    energy["Region"] = energy["Region"].astype(str)
    baseline["Region"] = baseline["Region"].astype(str)
    return energy, baseline, sensitivity


def select_window(energy, window_hours: int) -> tuple[int, int]:
    if window_hours < 1:
        raise ValueError("window-hours 必须为正整数")
    hours = sorted(energy["Hour"].astype(int).unique())
    if len(hours) < window_hours:
        raise ValueError("能源时序长度小于指定窗口")
    activity = (
        energy.assign(
            StorageActivity_MW=(
                energy["RenewableCharge_MW"].abs()
                + energy["GridCharge_MW"].abs()
                + energy["DischargePower_MW"].abs()
            )
        )
        .groupby("Hour", observed=False)["StorageActivity_MW"]
        .sum()
        .reindex(range(min(hours), max(hours) + 1), fill_value=0.0)
    )
    rolling = activity.rolling(window_hours, min_periods=window_hours).sum()
    end_hour = int(rolling.idxmax())
    start_hour = end_hour - window_hours + 1
    return start_hour, end_hour


def make_storage_timeseries_plot(
    energy,
    baseline,
    output_dir: Path,
    window_hours: int,
    plt,
    zh_font,
    en_font,
) -> None:
    import numpy as np

    start, end = select_window(energy, window_hours)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True, facecolor="#FFFFFF")
    for axis in axes.flat:
        axis.set_facecolor("#FFFFFF")
    colors = {
        "soc": "#2F5597",
        "charge": "#59A14F",
        "discharge": "#E15759",
        "net": "#6C757D",
        "baseline": "#A78BFA",
    }

    for axis, region in zip(axes.flat, REGIONS):
        part = energy[
            (energy["Region"] == region)
            & energy["Hour"].between(start, end)
        ].sort_values("Hour")
        base = baseline[
            (baseline["Region"] == region)
            & baseline["Hour"].between(start, end)
        ].sort_values("Hour")
        if part.empty:
            raise ValueError(f"scheme2 能源结果缺少 {region} 的窗口数据")

        charge = part["RenewableCharge_MW"] + part["GridCharge_MW"]
        discharge = -part["DischargePower_MW"]
        axis.bar(
            part["Hour"], charge, width=0.82, color=colors["charge"],
            alpha=0.62, label="充电功率",
        )
        axis.bar(
            part["Hour"], discharge, width=0.82, color=colors["discharge"],
            alpha=0.62, label="放电功率",
        )
        axis.axhline(0, color="#777777", linewidth=0.7)
        axis.set_title(REGION_CN[region], fontsize=13, fontweight="bold", color="#000000", fontproperties=zh_font)
        axis.set_ylabel("充/放功率（MW）", fontproperties=zh_font)
        axis.grid(axis="y", alpha=0.22)

        soc_axis = axis.twinx()
        soc_axis.plot(
            part["Hour"], part["SOC_MWh"], color=colors["soc"], linewidth=2.0,
            label="SOC",
        )
        soc_axis.set_ylabel("SOC（MWh）", color=colors["soc"], fontproperties=zh_font)
        soc_axis.tick_params(axis="y", colors=colors["soc"])
        soc_axis.spines["top"].set_visible(False)

        net_axis = axis.twinx()
        net_axis.spines["right"].set_position(("axes", 1.10))
        net_axis.plot(
            part["Hour"], part["NetGridImport_MW"], color=colors["net"],
            linewidth=1.0, linestyle="--", alpha=0.78, label="优化净购电",
        )
        if not base.empty:
            net_axis.plot(
                base["Hour"], base["NetGridImport_MW"], color=colors["baseline"],
                linewidth=0.9, alpha=0.65, label="附件基准净购电",
            )
        net_axis.set_ylabel("净购电（MW）", color=colors["net"], fontproperties=zh_font)
        net_axis.tick_params(axis="y", colors=colors["net"])
        net_axis.spines["top"].set_visible(False)

        handles = [
            axis.containers[0], axis.containers[1],
            soc_axis.lines[0], net_axis.lines[0],
        ]
        labels = ["充电功率", "放电功率", "SOC", "优化净购电"]
        if not base.empty:
            handles.append(net_axis.lines[1])
            labels.append("附件基准净购电")
        if axis is axes.flat[0]:
            axis.legend(
                handles, labels, loc="upper left", fontsize=8, frameon=False,
                prop=zh_font,
            )
        axis.set_xlim(start - 1, end + 1)
        apply_tick_font(axis, en_font)
        apply_tick_font(soc_axis, en_font)
        apply_tick_font(net_axis, en_font)

    axes.flat[-1].set_xlabel("小时", fontproperties=zh_font)
    axes.flat[-2].set_xlabel("小时", fontproperties=zh_font)
    fig.suptitle(
        f"问题三：六区域储能行为时序（{start}—{end}，{window_hours} h）",
        fontsize=20, fontweight="bold", color="#000000", y=0.985,
        fontproperties=zh_font,
    )
    fig.text(
        0.5, 0.955,
        "正柱为充电、负柱为放电；蓝线为 SOC；虚线比较优化与附件基准净购电",
        ha="center", fontsize=11, color="#636E72", fontproperties=zh_font,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.935])
    fig.savefig(
        plot_dir / "soc_charge_discharge_96h.png",
        dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def make_sensitivity_plot(sensitivity, output_dir: Path, plt, zh_font, en_font) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    feasible = sensitivity[sensitivity["Feasible"].astype(bool)].copy()
    if feasible.empty:
        raise ValueError("sensitivity_analysis.csv 没有可行场景")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), facecolor="#FFFFFF")
    for axis in axes:
        axis.set_facecolor("#FFFFFF")
    colors = {"cost": "#2F5597", "purchase": "#E15759"}
    panels = [
        (
            axes[0], "renewable_alpha", "RenewableAlpha", "可再生消纳上限 α",
            "α", "α 越收紧，储能价值与购电压力快速上升",
        ),
        (
            axes[1], "capacity_factor", "CapacityFactor", "储能容量系数",
            "容量系数 k", "容量增加后的边际收益逐步递减",
        ),
    ]
    for axis, dimension, field, xlabel, x_label, subtitle in panels:
        part = feasible[feasible["Dimension"] == dimension].sort_values(field)
        if part.empty:
            raise ValueError(f"敏感性结果缺少 {dimension} 场景")
        cost_axis = axis
        purchase_axis = axis.twinx()
        cost_line = cost_axis.plot(
            part[field], part["OperatingCost_CNY"] / 1_000_000,
            color=colors["cost"], marker="o", linewidth=2.0, label="运行成本",
        )[0]
        purchase_line = purchase_axis.plot(
            part[field], part["GridPurchase_MWh"] / 1_000,
            color=colors["purchase"], marker="s", linewidth=2.0, label="购电量",
        )[0]
        cost_axis.set_xlabel(xlabel, fontproperties=zh_font)
        cost_axis.set_ylabel("运行成本（百万元）", color=colors["cost"], fontproperties=zh_font)
        purchase_axis.set_ylabel("购电量（千 MWh）", color=colors["purchase"], fontproperties=zh_font)
        cost_axis.tick_params(axis="y", colors=colors["cost"])
        purchase_axis.tick_params(axis="y", colors=colors["purchase"])
        cost_axis.set_title(subtitle, fontsize=15, fontweight="bold", pad=14, color="#000000", fontproperties=zh_font)
        cost_axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.28)
        cost_axis.spines["top"].set_visible(False)
        purchase_axis.spines["top"].set_visible(False)
        cost_axis.legend(
            [cost_line, purchase_line], ["运行成本", "购电量"],
            loc="best", frameon=False, prop=zh_font,
        )
        apply_tick_font(cost_axis, en_font)
        apply_tick_font(purchase_axis, en_font)

    fig.suptitle(
        "问题三：储能敏感性曲线——成本与购电量的边际响应",
        fontsize=20, fontweight="bold", color="#000000", y=0.98,
        fontproperties=zh_font,
    )
    fig.text(
        0.5, 0.935,
        "左：α 收紧时价值陡增；右：容量扩大后收益趋于平缓，为问题四极端场景提供伏笔",
        ha="center", fontsize=11, color="#636E72", fontproperties=zh_font,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(
        plot_dir / "sensitivity_cost_grid.png",
        dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.window_hours < 1:
        raise ValueError("window-hours 必须为正整数")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    energy, baseline, sensitivity = load_inputs(args.output_dir)
    zh_font, en_font = configure_fonts(matplotlib)
    make_storage_timeseries_plot(
        energy, baseline, args.output_dir, args.window_hours,
        plt, zh_font, en_font,
    )
    make_sensitivity_plot(sensitivity, args.output_dir, plt, zh_font, en_font)
    print(f"完成。图表目录：{(args.output_dir / 'plots').resolve()}")


if __name__ == "__main__":
    main()
