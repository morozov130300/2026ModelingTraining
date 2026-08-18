#!/usr/bin/env python3
"""使用 t2/output_merged 中合并后的 CSV 重新生成问题二图表。

默认只读取合并结果目录，并将所有 PNG 写回该目录的 plots 子目录：
    python3 t2/dataToImg.py

如需指定其他“合并结果目录”：
    python3 t2/dataToImg.py --merged-dir t2/output_merged
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
        }
    )
    return selected["zh"], selected["en"]


def apply_tick_font(axis, font_properties) -> None:
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontproperties(font_properties)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="根据问题二合并输出 CSV 重新生成图表")
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=here / "output_merged",
        help="合并后的结果目录，默认 t2/output_merged；数据和图表均使用此目录",
    )
    return parser.parse_args()


def require_columns(name: str, frame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} 缺少字段: {missing}")


def _format_gpuh(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value / 1_000:.0f}k"


def _draw_flow(ax, x0: float, x1: float, y0: float, y1: float,
               width: float, color: str, alpha: float = 0.62) -> None:
    """绘制一条带宽度的平滑流带；带宽与 GPU-hour 成正比。"""
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    bend = (x1 - x0) * 0.42
    half = width / 2
    path = MplPath(
        [
            (x0, y0 - half),
            (x0 + bend, y0 - half),
            (x1 - bend, y1 - half),
            (x1, y1 - half),
            (x1, y1 + half),
            (x1 - bend, y1 + half),
            (x0 + bend, y0 + half),
            (x0, y0 + half),
            (x0, y0 - half),
        ],
        [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ],
    )
    ax.add_patch(PathPatch(path, facecolor=color, edgecolor="none", alpha=alpha, zorder=1))


def make_migration_combo_plot(scheme_b, plot_dir, plt, pd, zh_font, en_font) -> None:
    """生成“迁移桑基图 + 富集气泡图”组合图。

    桑基图只展示跨区流动，流带宽度严格按 GPU-hour 缩放；气泡图用同一口径呈现
    各源区到执行区的迁移富集程度。
    """
    import numpy as np
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Rectangle

    regions = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
    region_cn = {
        "RegionA": "区域A", "RegionB": "区域B", "RegionC": "区域C",
        "RegionD": "区域D", "RegionE": "区域E", "RegionF": "区域F",
    }
    colors = {
        "RegionA": "#4C78A8", "RegionB": "#72B7B2", "RegionC": "#59A14F",
        "RegionD": "#F28E2B", "RegionE": "#E15759", "RegionF": "#B279A2",
    }

    flow = (
        scheme_b.groupby(["SourceRegion", "ExecRegion"], observed=False)["GPU_h"]
        .sum()
        .reset_index()
    )
    flow = flow[flow["SourceRegion"] != flow["ExecRegion"]].copy()
    flow = flow[flow["GPU_h"] > 0].sort_values("GPU_h", ascending=False)
    matrix = (
        flow.pivot_table(
            index="SourceRegion", columns="ExecRegion", values="GPU_h",
            aggfunc="sum", fill_value=0, observed=False,
        )
        .reindex(index=regions, columns=regions, fill_value=0)
    )
    row_total = matrix.sum(axis=1)
    migration_total = flow.groupby("SourceRegion")["GPU_h"].sum()
    source_share = flow["GPU_h"] / flow["SourceRegion"].map(migration_total)
    flow["source_share"] = source_share.fillna(0)

    fig = plt.figure(figsize=(19, 9.8), facecolor="#FFFFFF")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.18, 1], wspace=0.18)
    sankey_ax = fig.add_subplot(grid[0, 0])
    bubble_ax = fig.add_subplot(grid[0, 1])
    for ax in (sankey_ax, bubble_ax):
        ax.set_facecolor("#FFFFFF")

    # 左：源区域—执行区域迁移桑基图。两列节点分开，避免自循环遮挡方向。
    y_pos = {region: len(regions) - 1 - i for i, region in enumerate(regions)}
    max_flow = float(flow["GPU_h"].max()) if len(flow) else 1.0
    min_width, max_width = 0.006, 0.075
    scale = (max_width - min_width) / max_flow
    for source in regions:
        part = flow[flow["SourceRegion"] == source].sort_values("ExecRegion")
        offset = 0.0
        total = max(float(part["GPU_h"].sum()), 1.0)
        for row in part.itertuples(index=False):
            width = min_width + float(row.GPU_h) * scale
            y0 = y_pos[row.SourceRegion] + 0.34 - offset - width / 2
            y1 = y_pos[row.ExecRegion] + 0.34 - (float(row.GPU_h) / total) * 0.45
            _draw_flow(sankey_ax, 0.19, 0.80, y0, y1, width, colors[row.SourceRegion])
            offset += width * 1.06

    for i, region in enumerate(regions):
        y = y_pos[region]
        sankey_ax.add_patch(Rectangle((0.12, y), 0.07, 0.68, color=colors[region], alpha=0.95, zorder=3))
        sankey_ax.add_patch(Rectangle((0.80, y), 0.07, 0.68, color=colors[region], alpha=0.95, zorder=3))
        sankey_ax.text(0.105, y + 0.34, region_cn[region], ha="right", va="center", fontsize=13, color="#2D3436", fontproperties=zh_font)
        sankey_ax.text(0.885, y + 0.34, region_cn[region], ha="left", va="center", fontsize=13, color="#2D3436", fontproperties=zh_font)
    sankey_ax.text(0.155, 6.05, "来源区域", ha="center", va="bottom", fontsize=20, fontweight="bold", color="#000000", fontproperties=zh_font)
    sankey_ax.text(0.835, 6.05, "执行区域", ha="center", va="bottom", fontsize=20, fontweight="bold", color="#000000", fontproperties=zh_font)
    # 分标题使用图级坐标，避免关闭坐标轴后被子图布局压缩。
    sankey_ax.set_title("")
    sankey_ax.text(0.50, -0.52, f"跨区迁移 GPU-hour：{flow['GPU_h'].sum():,.0f}；实时任务保持本地，未绘入跨区流带", ha="center", va="top", fontsize=11.5, color="#636E72", fontproperties=zh_font)
    sankey_ax.set_xlim(0, 1)
    sankey_ax.set_ylim(-0.8, 6.8)
    sankey_ax.axis("off")

    # 右：跨区迁移富集气泡矩阵。气泡面积 ∝ GPU-hour，颜色表示该流量占源区域迁移总量的比例。
    values = matrix.to_numpy(float)
    positive = values[values > 0]
    bubble_sizes = np.where(values > 0, 80 + 1900 * np.sqrt(values / positive.max()), 0)
    x, y = np.meshgrid(np.arange(len(regions)), np.arange(len(regions)))
    norm = Normalize(vmin=0, vmax=max(float(flow["source_share"].max()), 0.01))
    cmap = plt.get_cmap("YlOrRd")
    colors_bubble = cmap(norm(np.divide(values, row_total.to_numpy()[:, None], out=np.zeros_like(values), where=row_total.to_numpy()[:, None] > 0)))
    colors_bubble[values == 0, 3] = 0
    bubble_ax.scatter(x.ravel(), y.ravel(), s=bubble_sizes.ravel(), c=colors_bubble.reshape(-1, 4), edgecolors="#FFFFFF", linewidths=0.9, zorder=3)
    for i in range(len(regions)):
        for j in range(len(regions)):
            if values[i, j] > 0:
                bubble_ax.text(j, i, _format_gpuh(values[i, j]), ha="center", va="center", fontsize=10, color="#2D3436", zorder=4)
    bubble_ax.set_xticks(np.arange(len(regions)), [region_cn[r] for r in regions], fontsize=12, fontproperties=zh_font)
    bubble_ax.set_yticks(np.arange(len(regions)), [region_cn[r] for r in regions], fontsize=12, fontproperties=zh_font)
    bubble_ax.set_xlabel("执行区域", fontsize=18, labelpad=13, fontproperties=zh_font)
    bubble_ax.set_ylabel("来源区域", fontsize=18, labelpad=13, fontproperties=zh_font)
    bubble_ax.set_title("")
    bubble_ax.set_xlim(-0.6, 5.6)
    bubble_ax.set_ylim(5.6, -0.6)
    bubble_ax.set_aspect("equal")
    bubble_ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    bubble_ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    bubble_ax.grid(which="minor", color="#D9D4C8", linewidth=0.8)
    bubble_ax.grid(which="major", visible=False)
    bubble_ax.tick_params(which="minor", bottom=False, left=False)
    bubble_ax.text(0.02, -0.17, "气泡面积 ∝ GPU-hour；颜色越深表示该源区迁移流量占比越高；仅展示跨区流动", transform=bubble_ax.transAxes, fontsize=10.5, color="#636E72", fontproperties=zh_font)
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=bubble_ax, fraction=0.046, pad=0.04)
    colorbar.set_label("占源区域迁移总量比例", fontsize=10.5, fontproperties=zh_font)
    colorbar.ax.tick_params(labelsize=9)

    fig.subplots_adjust(top=0.80, bottom=0.12)
    sankey_bounds = sankey_ax.get_position()
    bubble_bounds = bubble_ax.get_position()
    panel_title_y = max(sankey_bounds.y1, bubble_bounds.y1) + 0.035

    fig.suptitle("问题二：迁移方向与 GPU-hour 富集结构", fontsize=23, fontweight="bold", color="#000000", y=0.965, ha="center", fontproperties=zh_font)
    fig.text((sankey_bounds.x0 + sankey_bounds.x1) / 2, panel_title_y, "跨区迁移流向（流宽 ∝ GPU-hour）", ha="center", va="center", fontsize=18, fontweight="bold", color="#000000", fontproperties=zh_font)
    fig.text((bubble_bounds.x0 + bubble_bounds.x1) / 2, panel_title_y, "迁移富集气泡矩阵", ha="center", va="center", fontsize=18, fontweight="bold", color="#000000", fontproperties=zh_font)
    fig.text(0.5, 0.915, "方案 B · 跨区流动揭示 D/E/F 双向迁移与能源信号退化", ha="center", fontsize=12.5, color="#636E72", fontproperties=zh_font)
    fig.savefig(plot_dir / "迁移桑基与富集气泡图.png", dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_scheme_tradeoff_plot(
    metrics, scheme_a, scheme_b, energy_a, energy_b, plot_dir, plt, pd, zh_font, en_font
) -> None:
    """用双面板同时展示表 11 总体权衡和区域级代价形状。"""
    labels = {
        "SchemeA_RenewableFirst_NoMigration": "方案 A",
        "SchemeB_CarbonAware": "方案 B",
    }
    plotted = metrics[metrics["Scenario"].isin(labels)].copy()
    if len(plotted) != 2:
        raise ValueError("scenario_comparison.csv 必须同时包含方案 A 与方案 B")
    plotted["方案"] = plotted["Scenario"].map(labels)
    plotted = plotted.set_index("方案").loc[["方案 A", "方案 B"]]

    colors = {"方案 A": "#4C78A8", "方案 B": "#E15759"}
    regions = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
    region_cn = {region: f"区域{region[-1]}" for region in regions}

    latency_a = float(plotted.loc["方案 A", "WeightedAverageLatency_ms"])
    latency_b = float(plotted.loc["方案 B", "WeightedAverageLatency_ms"])
    cost_a = float(plotted.loc["方案 A", "OperatingCost_CNY"])
    cost_b = float(plotted.loc["方案 B", "OperatingCost_CNY"])
    latency_delta = latency_b - latency_a
    latency_ratio = latency_b / latency_a
    cost_delta = cost_b - cost_a
    cost_improvement = -cost_delta
    cost_change_rate = abs(cost_delta) / abs(cost_a) * 100

    fig, (global_ax, regional_ax) = plt.subplots(
        1, 2, figsize=(21.5, 10.2), facecolor="#FFFFFF",
        gridspec_kw={"width_ratios": [1.0, 1.2]},
    )
    for ax in (global_ax, regional_ax):
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 左面板：严格使用表 11 的总体指标，每个方案一个点。
    global_x = [latency_a, latency_b]
    global_y = [cost_a / 1_000_000, cost_b / 1_000_000]
    global_ax.annotate(
        "", xy=(global_x[1], global_y[1]), xytext=(global_x[0], global_y[0]),
        arrowprops={"arrowstyle": "-|>", "color": "#8D8A83", "lw": 2.0,
                    "shrinkA": 12, "shrinkB": 12, "connectionstyle": "arc3,rad=-0.04"},
        zorder=2,
    )
    global_ax.scatter(global_x, global_y, s=520, c=[colors["方案 A"], colors["方案 B"]],
                      edgecolors="white", linewidths=2.2, zorder=4)
    for scheme, x, y in zip(["方案 A", "方案 B"], global_x, global_y):
        global_ax.annotate(
            f"{scheme}\n{x:.2f} ms\n{y:.3f} 百万元",
            (x, y), xytext=((16, 24) if scheme == "方案 A" else (-16, -66)),
            textcoords="offset points", ha=("left" if scheme == "方案 A" else "right"),
            fontsize=14, fontweight="bold", color=colors[scheme], fontproperties=zh_font,
        )
    global_ax.annotate(
        f"时延 +{latency_delta:.2f} ms（{latency_ratio:.2f} 倍）\n成本改善 {cost_improvement:,.2f} 元（{cost_change_rate:.3f}%）",
        ((latency_a + latency_b) / 2, sum(global_y) / 2), xytext=(0, 82),
        textcoords="offset points", ha="center", va="bottom", fontsize=15.5,
        color="#111111", fontproperties=zh_font,
        bbox={"boxstyle": "round,pad=0.65", "fc": "#FFFFFF", "ec": "#A99F90", "lw": 1.2},
    )
    x_pad = max(latency_delta * 0.22, 3.0)
    y_pad = max(abs(global_y[1] - global_y[0]) * 2.2, 0.11)
    global_ax.set_xlim(latency_a - x_pad, latency_b + x_pad)
    global_ax.set_ylim(min(global_y) - y_pad, max(global_y) + y_pad)
    global_ax.set_xlabel("GPU-hour 加权平均时延（ms）", fontsize=16, labelpad=16, fontproperties=zh_font)
    global_ax.set_ylabel("总运行成本（百万元）", fontsize=16, labelpad=18, fontproperties=zh_font)
    global_ax.set_title("表 11：总体方案权衡", fontsize=22, fontweight="bold", pad=34, color="#000000", fontproperties=zh_font)
    global_ax.text(0.5, 1.17, "B 的成本几乎不变，但时延扩大至约 7.5 倍",
                   transform=global_ax.transAxes, ha="center", va="bottom",
                   fontsize=14.8, color="#111111", fontproperties=zh_font)

    # 右面板：区域级分解，展示“代价形状”，不与总体点混用坐标。
    energy_cost = {
        "方案 A": energy_a.groupby("Region", observed=False)["Cost_CNY"].sum(),
        "方案 B": energy_b.groupby("Region", observed=False)["Cost_CNY"].sum(),
    }
    regional = {}
    for scheme, schedule in [("方案 A", scheme_a), ("方案 B", scheme_b)]:
        rows = []
        for region, part in schedule.groupby("ExecRegion", observed=False):
            gpuh = float(part["GPU_h"].sum())
            if gpuh <= 0:
                continue
            rows.append({
                "Region": region,
                "Latency": float((part["ActualLatency_ms"] * part["GPU_h"]).sum() / gpuh),
                "Cost": float(energy_cost[scheme].get(region, 0.0)) / 1_000_000,
                "GPU_h": gpuh,
            })
        regional[scheme] = pd.DataFrame(rows).set_index("Region")

    for region in regions:
        if region not in regional["方案 A"].index or region not in regional["方案 B"].index:
            continue
        a = regional["方案 A"].loc[region]
        b = regional["方案 B"].loc[region]
        regional_ax.plot([a["Latency"], b["Latency"]], [a["Cost"], b["Cost"]],
                         color="#8F8578", linewidth=1.15, alpha=0.9, zorder=1)
        regional_ax.annotate(region_cn[region], (b["Latency"], b["Cost"]),
                             xytext=(8, 6), textcoords="offset points", fontsize=12,
                             color="#222222", fontproperties=zh_font)

    for scheme, marker in [("方案 A", "o"), ("方案 B", "D")]:
        part = regional[scheme]
        regional_ax.scatter(part["Latency"], part["Cost"], marker=marker,
                            s=95 + 165 * (part["GPU_h"] / max(part["GPU_h"].max(), 1)) ** 0.5,
                            color=colors[scheme], edgecolors="white", linewidths=1.2,
                            alpha=0.92, label=scheme, zorder=3)
    regional_ax.set_xlabel("区域 GPU-hour 加权平均时延（ms）", fontsize=16, labelpad=16, fontproperties=zh_font)
    regional_ax.set_ylabel("区域运行成本（百万元）", fontsize=16, labelpad=18, fontproperties=zh_font)
    regional_ax.set_title("区域级代价形状：A → B", fontsize=22, fontweight="bold", pad=34, color="#000000", fontproperties=zh_font)
    regional_ax.legend(prop=zh_font, frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.01), borderaxespad=0)
    regional_ax.text(0.02, 0.995, "圆点/菱形：区域 A/B；连线：同一区域迁移后的变化；点大小 ∝ GPU-hour",
                     transform=regional_ax.transAxes, fontsize=14.2, color="#111111", fontproperties=zh_font, va="top")

    for ax in (global_ax, regional_ax):
        apply_tick_font(ax, en_font)
        ax.tick_params(axis="both", labelsize=14, colors="#111111")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.80, bottom=0.12, wspace=0.28)
    fig.suptitle("方案 A/B 的时延—成本权衡：总体结论与区域代价形状",
                 fontsize=26, fontweight="bold", color="#000000", y=0.965, fontproperties=zh_font)
    fig.tight_layout(rect=[0.02, 0.03, 0.985, 0.80])
    fig.savefig(plot_dir / "方案甲乙时延成本权衡图.png", dpi=220,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_plots(merged_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    zh_font, en_font = configure_fonts(matplotlib)
    metrics_path = merged_dir / "reports" / "scenario_comparison.csv"
    schedule_a_path = merged_dir / "schedule" / "scheme_a_no_migration_schedule.csv"
    schedule_path = merged_dir / "schedule" / "scheme_b_carbon_aware_schedule.csv"
    usage_path = merged_dir / "schedule" / "scheme_b_hourly_usage.csv"
    energy_a_path = merged_dir / "energy" / "scheme_a_energy_balance.csv"
    energy_b_path = merged_dir / "energy" / "scheme_b_energy_balance.csv"

    metrics = pd.read_csv(metrics_path)
    scheme_a = pd.read_csv(schedule_a_path)
    scheme_b = pd.read_csv(schedule_path)
    usage = pd.read_csv(usage_path)
    energy_a = pd.read_csv(energy_a_path)
    energy_b = pd.read_csv(energy_b_path)

    require_columns(
        str(metrics_path),
        metrics,
        ["Scenario", "OperatingCost_CNY", "CarbonEmission_tCO2", "WeightedAverageLatency_ms"],
    )
    require_columns(
        str(schedule_a_path),
        scheme_a,
        ["ExecRegion", "GPU_h", "ActualLatency_ms"],
    )
    require_columns(
        str(schedule_path),
        scheme_b,
        ["SourceRegion", "ExecRegion", "GPU_h", "ActualLatency_ms"],
    )
    require_columns(
        str(usage_path),
        usage,
        ["Hour", "Region", "GPU_Utilization_Percent"],
    )
    require_columns(str(energy_a_path), energy_a, ["Region", "Cost_CNY"])
    require_columns(str(energy_b_path), energy_b, ["Region", "Cost_CNY"])
    plot_dir = merged_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.titlecolor": "#000000",
        }
    )

    plotted = metrics.dropna(subset=["OperatingCost_CNY"])
    ax = plotted.set_index("Scenario")[["OperatingCost_CNY", "CarbonEmission_tCO2"]].plot(
        kind="bar", secondary_y="CarbonEmission_tCO2", figsize=(10, 5),
    )
    ax.set_ylabel("运行成本（元）", fontproperties=zh_font)
    ax.right_ax.set_ylabel("碳排放（tCO2）", fontproperties=zh_font)
    apply_tick_font(ax, en_font)
    apply_tick_font(ax.right_ax, en_font)
    plt.tight_layout()
    plt.savefig(plot_dir / "方案成本碳排对比图.png", dpi=180)
    plt.close()

    migration = scheme_b.groupby(["SourceRegion", "ExecRegion"], observed=False)["GPU_h"].sum().unstack(fill_value=0)
    ax = migration.plot(kind="bar", stacked=True, figsize=(11, 5))
    ax.set_ylabel("执行 GPU-hour", fontproperties=zh_font)
    apply_tick_font(ax, en_font)
    plt.tight_layout()
    plt.savefig(plot_dir / "迁移流向按来源区域汇总图.png", dpi=180)
    plt.close()

    make_migration_combo_plot(scheme_b, plot_dir, plt, pd, zh_font, en_font)
    make_scheme_tradeoff_plot(
        metrics, scheme_a, scheme_b, energy_a, energy_b,
        plot_dir, plt, pd, zh_font, en_font,
    )

    regions = usage["Region"].drop_duplicates().tolist()
    if len(regions) != 6:
        raise ValueError(f"{usage_path} 应包含 6 个区域，实际为 {len(regions)} 个: {regions}")
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True, facecolor="#FFFFFF")
    for axis, region in zip(axes.flat, regions):
        part = usage[(usage["Region"] == region) & usage["Hour"].between(2376, 2405)]
        axis.plot(part["Hour"], part["GPU_Utilization_Percent"])
        axis.set_title(region, color="#000000", fontproperties=en_font)
        axis.set_ylabel("GPU 利用率（%）", fontproperties=zh_font)
        apply_tick_font(axis, en_font)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "图形处理器利用率图_2376至2405.png", dpi=180)
    plt.close(fig)

    print(f"完成。图表目录：{plot_dir.resolve()}")


def main() -> None:
    args = parse_args()
    make_plots(args.merged_dir)


if __name__ == "__main__":
    main()
