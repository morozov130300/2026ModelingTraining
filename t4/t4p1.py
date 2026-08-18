#!/usr/bin/env python3
"""问题四论文可视化：消融瀑布、反馈 incumbent、alpha 压力传导。

默认读取 t4/output 下已有 CSV，不重新求解模型：
    python t4/t4p1.py

输出：
    t4/output/plots/ablation_waterfall_A0_A4.png
    t4/output/plots/feedback_incumbent_convergence.png
    t4/output/plots/alpha_pressure_transmission.png
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

def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="生成问题四论文图表")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "output",
        help="问题四结果目录，默认 t4/output",
    )
    return parser.parse_args()


def configure_fonts(matplotlib):
    from matplotlib.font_manager import FontProperties, fontManager

    selected = {}
    for language, candidates in FONT_CANDIDATES.items():
        font_path = next((path for path in candidates if path.exists()), None)
        label = "中文" if language == "zh" else "英文"
        if font_path is None:
            raise FileNotFoundError(f"未找到{label}绘图字体，请安装宋体或 Times New Roman")
        fontManager.addfont(str(font_path))
        selected[language] = FontProperties(fname=str(font_path))

    matplotlib.rcParams.update(
        {
            "font.family": selected["zh"].get_name(),
            "font.serif": [selected["zh"].get_name()],
            "axes.unicode_minus": False,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#B8C2CC",
            "axes.labelcolor": "#000000",
            "axes.titlecolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "text.color": "#000000",
        }
    )
    return selected["zh"], selected["en"]


def require_columns(name: str, frame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} 缺少字段: {missing}")


def apply_axis_fonts(axis, x_font, y_font) -> None:
    for label in axis.get_xticklabels():
        label.set_fontproperties(x_font)
    for label in axis.get_yticklabels():
        label.set_fontproperties(y_font)
    x_offset = axis.xaxis.get_offset_text()
    y_offset = axis.yaxis.get_offset_text()
    if x_offset is not None:
        x_offset.set_fontproperties(x_font)
    if y_offset is not None:
        y_offset.set_fontproperties(y_font)


def load_inputs(output_dir: Path):
    import pandas as pd

    reports = output_dir / "reports"
    ablation = pd.read_csv(reports / "ablation_A0_A4.csv", encoding="utf-8-sig")
    convergence = pd.read_csv(reports / "convergence.csv", encoding="utf-8-sig")
    sensitivity = pd.read_csv(reports / "scenario_sensitivity.csv", encoding="utf-8-sig")

    require_columns(
        "ablation_A0_A4.csv",
        ablation,
        ["Scenario", "Objective_CNY"],
    )
    require_columns(
        "convergence.csv",
        convergence,
        ["Iteration", "Objective_CNY", "BestObjective_CNY"],
    )
    require_columns(
        "scenario_sensitivity.csv",
        sensitivity,
        [
            "Scenario",
            "Parameter",
            "MigratedTasks",
            "Objective_CNY",
            "GridPurchase_MWh",
        ],
    )
    return ablation, convergence, sensitivity


def format_million(value: float, digits: int = 2) -> str:
    return f"{value / 1_000_000:.{digits}f}M"


def make_ablation_waterfall(ablation, plot_dir: Path, plt, zh_font, en_font) -> None:
    from matplotlib.patches import Patch

    order = ["A0", "A1", "A2", "A3", "A4"]
    data = ablation.set_index("Scenario").loc[order].copy()
    objective = data["Objective_CNY"].astype(float)
    improvements = -objective.diff().fillna(0.0)
    step_labels = [
        "A0",
        "A0→A1\n迁移",
        "A1→A2\n调时",
        "A2→A3\n储能",
        "A3→A4\n反馈",
    ]

    fig, ax = plt.subplots(figsize=(13.8, 8.2), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x_values = list(range(len(order)))
    base_million = -float(objective.iloc[0]) / 1_000_000
    running = base_million
    bar_bottoms = [0.0]
    bar_heights = [base_million]
    colors = ["#5B8DEF"]

    for scenario in order[1:]:
        delta_million = float(improvements.loc[scenario]) / 1_000_000
        if abs(delta_million) < 0.005:
            bar_bottoms.append(running - 0.06)
            bar_heights.append(0.12)
            colors.append("#8A949E")
        else:
            bar_bottoms.append(running)
            bar_heights.append(delta_million)
            colors.append("#1FA971" if scenario == "A3" else "#64B96A")
            running += delta_million

    bars = ax.bar(
        x_values,
        bar_heights,
        bottom=bar_bottoms,
        width=0.56,
        color=colors,
        edgecolor="#1F2933",
        linewidth=1.0,
        zorder=3,
    )

    cumulative = [-float(value) / 1_000_000 for value in objective]
    for left, right, level in zip(x_values[:-1], x_values[1:], cumulative[:-1]):
        ax.plot(
            [left + 0.28, right - 0.28],
            [level, level],
            color="#4A5568",
            linewidth=1.1,
            linestyle="--",
            alpha=0.65,
            zorder=2,
        )

    for index, (bar, scenario) in enumerate(zip(bars, order)):
        if index == 0:
            label = format_million(-float(objective.loc[scenario]))
            y = bar.get_height() + 2.0
            font_prop = en_font
        else:
            delta = float(improvements.loc[scenario])
            label = "0" if abs(delta) < 1 else f"+{format_million(delta)}"
            y = bar.get_y() + bar.get_height() + 1.5
            font_prop = en_font if index > 0 else zh_font
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=12.5,
            fontweight="bold",
            fontproperties=font_prop,
            color="#000000",
        )

    storage_index = order.index("A3")
    ax.annotate(
        "储能是主导收益",
        xy=(storage_index, cumulative[storage_index]),
        xytext=(storage_index + 0.86, cumulative[storage_index] + 13),
        arrowprops={"arrowstyle": "->", "color": "#0B7A53", "linewidth": 1.6},
        fontsize=15,
        fontweight="bold",
        fontproperties=zh_font,
        color="#0B7A53",
    )
    ax.annotate(
        "A4 零增量：采用 incumbent，反馈无额外收益",
        xy=(4, cumulative[4]),
        xytext=(2.75, cumulative[4] - 20),
        arrowprops={"arrowstyle": "->", "color": "#5F6B7A", "linewidth": 1.5},
        fontsize=12.5,
        fontproperties=zh_font,
        color="#000000",
    )

    ax.set_xticks(x_values)
    ax.set_xticklabels(step_labels, fontproperties=zh_font, fontsize=13)
    ax.set_ylabel("目标值改善累计（百万元）", fontproperties=zh_font, fontsize=14)
    ax.set_title(
        "A0→A4 消融瀑布：四步改善链条与反馈零增量",
        fontproperties=zh_font,
        fontsize=18,
        fontweight="bold",
        color="#000000",
        pad=24,
    )
    ax.set_ylim(-5, max(cumulative) + 26)
    ax.grid(axis="y", color="#D9E2EC", linewidth=0.8, alpha=0.85, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    apply_axis_fonts(ax, zh_font, en_font)
    ax.legend(
        handles=[
            Patch(facecolor="#5B8DEF", edgecolor="#1F2933", label="A0 基准绝对水平"),
            Patch(facecolor="#64B96A", edgecolor="#1F2933", label="相对上一步改善"),
            Patch(facecolor="#1FA971", edgecolor="#1F2933", label="储能主导改善"),
            Patch(facecolor="#8A949E", edgecolor="#1F2933", label="零增量"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
        prop=zh_font,
        fontsize=11.5,
    )
    fig.subplots_adjust(bottom=0.19, top=0.90)
    fig.savefig(plot_dir / "ablation_waterfall_A0_A4.png", dpi=220, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)


def make_feedback_incumbent(convergence, plot_dir: Path, plt, zh_font, en_font) -> None:
    data = convergence.sort_values("Iteration").copy()
    data["Objective_Million"] = data["Objective_CNY"].astype(float) / 1_000_000
    data["BestObjective_Million"] = data["BestObjective_CNY"].astype(float) / 1_000_000

    fig, ax = plt.subplots(figsize=(11.2, 6.6), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    ax.plot(
        data["Iteration"],
        data["Objective_Million"],
        marker="o",
        markersize=8,
        linewidth=2.4,
        color="#D95F02",
        label="当轮目标值",
    )
    ax.plot(
        data["Iteration"],
        data["BestObjective_Million"],
        marker="s",
        markersize=7,
        linewidth=2.4,
        color="#1B6CA8",
        label="历史最优（incumbent）",
    )

    if len(data) >= 2:
        second = data.iloc[1]
        best = float(second["BestObjective_Million"])
        current = float(second["Objective_Million"])
        gap_cny = float(second["Objective_CNY"] - second["BestObjective_CNY"])
        ax.annotate(
            f"劣化 {gap_cny:,.2f} 元\n采用历史最优",
            xy=(float(second["Iteration"]), current),
            xytext=(float(second["Iteration"]) - 0.42, current + 0.032),
            arrowprops={"arrowstyle": "->", "color": "#C2410C", "linewidth": 1.5},
            fontsize=13,
            fontproperties=zh_font,
            color="#000000",
            ha="right",
        )
        ax.vlines(
            float(second["Iteration"]),
            ymin=min(best, current),
            ymax=max(best, current),
            color="#C2410C",
            linestyle=":",
            linewidth=1.5,
        )

    ax.set_xticks(data["Iteration"].astype(int).tolist())
    ax.set_xlabel("反馈轮次", fontproperties=zh_font, fontsize=14)
    ax.set_ylabel("目标值（百万元）", fontproperties=zh_font, fontsize=14)
    ax.set_title(
        "反馈收敛与 incumbent：停止条件不等于最优",
        fontproperties=zh_font,
        fontsize=18,
        fontweight="bold",
        color="#000000",
        pad=16,
    )
    ax.grid(color="#D9E2EC", linewidth=0.8, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    apply_axis_fonts(ax, zh_font, en_font)
    ax.legend(loc="best", frameon=False, prop=zh_font, fontsize=12)
    fig.tight_layout()
    fig.savefig(plot_dir / "feedback_incumbent_convergence.png", dpi=220, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)


def make_alpha_pressure(sensitivity, plot_dir: Path, plt, zh_font, en_font) -> None:
    alpha = sensitivity.loc[sensitivity["Scenario"].eq("alpha")].copy()
    alpha = alpha.sort_values("Parameter")
    required_alpha = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha = alpha[alpha["Parameter"].round(6).isin(required_alpha)]
    if len(alpha) != len(required_alpha):
        raise ValueError("scenario_sensitivity.csv 缺少 alpha=0.5--1.0 的完整结果")

    pressure = sensitivity.loc[sensitivity["Scenario"].eq("renewable_minus_20_percent")].copy()
    pressure_point = pressure.iloc[0] if not pressure.empty else None

    fig, (top_ax, bottom_ax) = plt.subplots(
        2,
        1,
        figsize=(12.8, 9.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.16},
        facecolor="#FFFFFF",
    )
    top_ax.set_facecolor("#FFFFFF")
    bottom_ax.set_facecolor("#FFFFFF")

    x = alpha["Parameter"].astype(float)
    grid_purchase = alpha["GridPurchase_MWh"].astype(float)
    objective = alpha["Objective_CNY"].astype(float) / 1_000_000
    migrations = alpha["MigratedTasks"].astype(int)

    top_ax.plot(
        x,
        grid_purchase,
        marker="o",
        markersize=7,
        linewidth=2.3,
        color="#1B6CA8",
        label="购电量",
    )
    top_ax.set_ylabel("购电量（MWh）", fontproperties=zh_font, fontsize=14, color="#1B6CA8")
    top_ax.tick_params(axis="y", labelcolor="#1B6CA8")
    top_ax.grid(color="#D9E2EC", linewidth=0.8, alpha=0.85)
    top_ax.spines[["top"]].set_visible(False)

    objective_ax = top_ax.twinx()
    objective_ax.plot(
        x,
        objective,
        marker="s",
        markersize=6.5,
        linewidth=2.3,
        color="#C2410C",
        label="目标值",
    )
    objective_ax.set_ylabel("目标值（百万元）", fontproperties=zh_font, fontsize=14, color="#C2410C")
    objective_ax.tick_params(axis="y", labelcolor="#C2410C")
    objective_ax.spines[["top"]].set_visible(False)

    top_ax.annotate(
        "α=0.5：购电约105万MWh\n目标值转正",
        xy=(0.5, float(grid_purchase.iloc[0])),
        xytext=(0.58, float(grid_purchase.iloc[0]) * 0.84),
        arrowprops={"arrowstyle": "->", "color": "#1B6CA8", "linewidth": 1.4},
        fontsize=13,
        fontproperties=zh_font,
        color="#000000",
    )

    if pressure_point is not None:
        pressure_x = float(pressure_point["Parameter"])
        pressure_grid = float(pressure_point["GridPurchase_MWh"])
        pressure_obj = float(pressure_point["Objective_CNY"]) / 1_000_000
        top_ax.scatter(
            [pressure_x],
            [pressure_grid],
            marker="D",
            s=105,
            facecolors="none",
            edgecolors="#7C3AED",
            linewidths=2.0,
            zorder=5,
            label="新能源下移 20%",
        )
        objective_ax.scatter(
            [pressure_x],
            [pressure_obj],
            marker="D",
            s=95,
            facecolors="none",
            edgecolors="#7C3AED",
            linewidths=1.8,
            zorder=5,
        )
        top_ax.annotate(
            "新能源下移20%\n与 α=0.8 等价",
            xy=(pressure_x, pressure_grid),
            xytext=(0.66, max(grid_purchase) * 0.27),
            arrowprops={"arrowstyle": "->", "color": "#7C3AED", "linewidth": 1.3},
            fontsize=12.5,
            fontproperties=zh_font,
            color="#000000",
        )

    bars = bottom_ax.bar(
        x,
        migrations,
        width=0.055,
        color="#3D9970",
        edgecolor="#1F2933",
        linewidth=0.9,
        zorder=3,
    )
    for bar, count in zip(bars, migrations):
        bottom_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontproperties=zh_font,
            color="#000000",
        )
    bottom_ax.text(
        0.5,
        0.90,
        "触发与否取决于区域—时间边际信号差异",
        transform=bottom_ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        fontproperties=zh_font,
        color="#000000",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#FFFFFF", "ec": "#A0AEC0", "lw": 0.9},
    )
    bottom_ax.set_ylabel("迁移任务数", fontproperties=zh_font, fontsize=14)
    bottom_ax.set_xlabel("可再生能源利用上限 α", fontproperties=zh_font, fontsize=14)
    bottom_ax.set_xticks(required_alpha)
    bottom_ax.set_ylim(0, max(migrations.max() * 1.28, 100))
    bottom_ax.grid(axis="y", color="#D9E2EC", linewidth=0.8, alpha=0.85, zorder=0)
    bottom_ax.spines[["top", "right"]].set_visible(False)

    handles_1, labels_1 = top_ax.get_legend_handles_labels()
    handles_2, labels_2 = objective_ax.get_legend_handles_labels()
    top_ax.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="upper right",
        frameon=False,
        prop=zh_font,
        fontsize=12,
    )
    top_ax.set_title(
        "α 敏感性压力传导：购电、目标值与迁移触发不同步",
        fontproperties=zh_font,
        fontsize=18,
        fontweight="bold",
        color="#000000",
        pad=14,
    )
    apply_axis_fonts(top_ax, zh_font, en_font)
    apply_axis_fonts(objective_ax, zh_font, en_font)
    apply_axis_fonts(bottom_ax, zh_font, en_font)

    fig.tight_layout()
    fig.savefig(plot_dir / "alpha_pressure_transmission.png", dpi=220, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    zh_font, en_font = configure_fonts(matplotlib)
    ablation, convergence, sensitivity = load_inputs(args.output_dir)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    make_ablation_waterfall(ablation, plot_dir, plt, zh_font, en_font)
    make_feedback_incumbent(convergence, plot_dir, plt, zh_font, en_font)
    make_alpha_pressure(sensitivity, plot_dir, plt, zh_font, en_font)

    print(f"已生成问题四论文图表：{plot_dir}")


if __name__ == "__main__":
    main()
