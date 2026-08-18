#!/usr/bin/env python3
"""基于问题二输出数据生成“相关性矩阵—方法—决策输出”综合关系图。

运行：
    python3 t2/make_relationship_diagram.py

显式指定结果目录：
    python3 t2/make_relationship_diagram.py --output-dir t2/output
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape


INDICATORS = [
    ("AI IT负荷", "AI_IT_Load_MW"),
    ("非AI IT负荷", "NonAI_IT_Load_MW"),
    ("设施总负荷", "Total_Load_MW"),
    ("可用新能源", "AvailableRenewable_MW"),
    ("购电价格", "ElectricityPrice_CNY_per_MWh"),
    ("碳排强度", "CarbonIntensity_tCO2_per_MWh"),
    ("新能源余量", "RenewableSurplus_MW"),
    ("GPU利用率", "GPU_Utilization_Percent"),
    ("IT功率余量", "IT_Power_Margin_MW"),
]

METHODS = [
    ("多源数据融合", "逐时负荷、能源与算力状态对齐"),
    ("Pearson相关分析", "识别指标同向与反向关系"),
    ("SLA可达域筛选", "按20/80/150 ms约束预剪枝"),
    ("两阶段碳感知调度", "区域粗指派与EDF时刻细化"),
    ("情景对比与归因", "Q1、方案A、方案B及λ灵敏度"),
]

OUTPUTS = [
    ("任务迁移方案", "执行区域与迁移流向"),
    ("开工时刻方案", "弹性任务等待与新能源匹配"),
    ("经济与碳效益", "成本、购电量与碳排放"),
    ("新能源消纳评价", "直接消纳、外送与弃电"),
    ("服务质量评价", "时延、容量与截止约束"),
]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="生成问题二综合关系可视化图")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "output",
        help="问题二结果目录，默认 t2/output",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        default=True,
        help="同时用 matplotlib 生成 PNG（默认开启）",
    )
    parser.add_argument(
        "--no-png",
        action="store_false",
        dest="png",
        help="跳过 PNG，只生成 SVG",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def aligned_indicator_rows(output_dir: Path) -> list[list[float]]:
    energy_path = output_dir / "energy" / "scheme_b_energy_balance.csv"
    usage_path = output_dir / "schedule" / "scheme_b_hourly_usage.csv"
    if not energy_path.exists() or not usage_path.exists():
        raise FileNotFoundError(
            f"缺少问题二结果文件：{energy_path} 或 {usage_path}"
        )

    usage = {
        (row["Hour"], row["Region"]): row
        for row in read_rows(usage_path)
    }
    values: list[list[float]] = []
    for row in read_rows(energy_path):
        key = (row["Hour"], row["Region"])
        usage_row = usage.get(key)
        if usage_row is None:
            continue
        merged = dict(row)
        merged.update(usage_row)
        merged["RenewableSurplus_MW"] = str(
            float(row["AvailableRenewable_MW"]) - float(row["Total_Load_MW"])
        )
        values.append([float(merged[column]) for _, column in INDICATORS])
    if not values:
        raise ValueError("能源与算力逐时数据无法按 Hour、Region 对齐")
    return values


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    variance_x = sum((a - mean_x) ** 2 for a in x)
    variance_y = sum((b - mean_y) ** 2 for b in y)
    denominator = math.sqrt(variance_x * variance_y)
    return covariance / denominator if denominator > 0 else 0.0


def correlation_matrix(rows: list[list[float]]) -> list[list[float]]:
    columns = [list(column) for column in zip(*rows)]
    return [[pearson(x, y) for y in columns] for x in columns]


def significance_stars(r: float, sample_size: int) -> str:
    if sample_size <= 3 or abs(r) >= 1.0:
        return "***" if abs(r) >= 1.0 else ""
    z_score = abs(math.atanh(r)) * math.sqrt(sample_size - 3)
    p_value = math.erfc(z_score / math.sqrt(2.0))
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def mix_color(r: float) -> str:
    white = (248, 248, 246)
    target = (220, 92, 49) if r >= 0 else (49, 112, 171)
    weight = min(1.0, abs(r)) ** 0.72
    rgb = tuple(round(white[i] * (1 - weight) + target[i] * weight) for i in range(3))
    return "#%02x%02x%02x" % rgb


def svg_text(x: float, y: float, text: str, size: int = 20, **attrs: str) -> str:
    attributes = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" {attributes}>{escape(text)}</text>'


def curved_path(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    width: float,
    opacity: float = 0.58,
    route_offset: float = 0.0,
) -> str:
    span = x2 - x1
    control_1_x = x1 + span * 0.30
    control_2_x = x2 - span * 0.30
    control_1_y = y1 + route_offset
    control_2_y = y2 - route_offset
    return (
        f'<path d="M {x1:.1f},{y1:.1f} C {control_1_x:.1f},{control_1_y:.1f} '
        f'{control_2_x:.1f},{control_2_y:.1f} {x2:.1f},{y2:.1f}" fill="none" '
        f'stroke="{color}" stroke-width="{width:.1f}" opacity="{opacity}" '
        f'stroke-linecap="round"/>'
    )


def layout_data(rows: list[list[float]]):
    correlations = correlation_matrix(rows)
    width, height = 1900, 1120
    matrix_x, matrix_y, cell = 300, 210, 72
    method_x, method_w = 1060, 300
    output_x, output_w = 1545, 285
    method_centers = [235 + index * 155 + 47 for index in range(len(METHODS))]
    output_centers = [235 + index * 155 + 47 for index in range(len(OUTPUTS))]
    factor_links = [0, 0, 1, 1, 1, 1, 1, 3, 3]
    factor_colors = ["#9b9b9b", "#3f7fb9", "#df7a35", "#3f7fb9", "#df7a35", "#3f7fb9", "#df7a35", "#3f7fb9", "#df7a35"]
    method_to_outputs = {
        0: [(0, "#999999"), (2, "#999999"), (3, "#999999")],
        1: [(2, "#3f7fb9"), (3, "#df7a35")],
        2: [(0, "#3f7fb9"), (4, "#df7a35")],
        3: [(0, "#df7a35"), (1, "#3f7fb9"), (2, "#df7a35"), (3, "#3f7fb9"), (4, "#999999")],
        4: [(2, "#3f7fb9"), (3, "#df7a35"), (4, "#999999")],
    }
    return {
        "correlations": correlations,
        "width": width,
        "height": height,
        "matrix_x": matrix_x,
        "matrix_y": matrix_y,
        "cell": cell,
        "method_x": method_x,
        "method_w": method_w,
        "output_x": output_x,
        "output_w": output_w,
        "method_centers": method_centers,
        "output_centers": output_centers,
        "factor_links": factor_links,
        "factor_colors": factor_colors,
        "method_to_outputs": method_to_outputs,
    }


def generate_svg(output_dir: Path, rows: list[list[float]], layout: dict) -> Path:
    correlations = layout["correlations"]
    width, height = layout["width"], layout["height"]
    matrix_x, matrix_y, cell = layout["matrix_x"], layout["matrix_y"], layout["cell"]
    method_x, method_w = layout["method_x"], layout["method_w"]
    output_x, output_w = layout["output_x"], layout["output_w"]
    method_centers = layout["method_centers"]
    output_centers = layout["output_centers"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei","SimHei",sans-serif;fill:#000000}</style>',
        svg_text(70, 70, "问题二：碳感知调度的指标关联、建模方法与决策输出", 34, font_weight="700"),
        svg_text(70, 108, f"基于方案B的 {len(rows):,} 条区域—小时记录；方格为 Pearson r，星号表示显著性", 18, fill="#666666"),
        svg_text(matrix_x + len(INDICATORS) * cell / 2, 160, "逐时指标相关性矩阵", 25, font_weight="700", text_anchor="middle"),
        svg_text(method_x + method_w / 2, 160, "分析与优化方法", 25, font_weight="700", text_anchor="middle"),
        svg_text(output_x + output_w / 2, 160, "核心决策输出", 25, font_weight="700", text_anchor="middle"),
    ]

    label_y = matrix_y + len(INDICATORS) * cell + 28
    for index, (label, _) in enumerate(INDICATORS):
        x = matrix_x + index * cell + cell / 2
        y = matrix_y + index * cell + cell / 2
        parts.append(svg_text(x, label_y, label, 16, text_anchor="end", transform=f"rotate(45 {x:.1f} {label_y:.1f})"))
        parts.append(svg_text(matrix_x - 20, y + 6, label, 17, text_anchor="end"))

    for row_index in range(len(INDICATORS)):
        for column_index in range(row_index + 1):
            r = correlations[row_index][column_index]
            x = matrix_x + column_index * cell
            y = matrix_y + row_index * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{mix_color(r)}" stroke="#ffffff" stroke-width="2" rx="3"/>'
            )
            text_color = "#ffffff" if abs(r) > 0.55 else "#303030"
            parts.append(svg_text(x + cell / 2, y + 31, f"{r:+.2f}", 15, text_anchor="middle", fill=text_color, font_weight="600"))
            parts.append(svg_text(x + cell / 2, y + 53, significance_stars(r, len(rows)), 16, text_anchor="middle", fill=text_color))

    for index, (title, subtitle) in enumerate(METHODS):
        y = 235 + index * 155
        parts.extend([
            f'<rect x="{method_x}" y="{y}" width="{method_w}" height="94" rx="18" fill="#ffffff" stroke="#c8c8c8" stroke-width="2"/>',
            f'<circle cx="{method_x}" cy="{y + 47}" r="7" fill="#d43d2f"/>',
            svg_text(method_x + method_w / 2, y + 38, title, 21, text_anchor="middle", font_weight="700"),
            svg_text(method_x + method_w / 2, y + 68, subtitle, 15, text_anchor="middle", fill="#666666"),
        ])

    for index, (title, subtitle) in enumerate(OUTPUTS):
        y = 235 + index * 155
        parts.extend([
            f'<rect x="{output_x}" y="{y}" width="{output_w}" height="94" rx="18" fill="#FFFFFF" stroke="#c8c8c8" stroke-width="2"/>',
            f'<circle cx="{output_x}" cy="{y + 47}" r="7" fill="#d43d2f"/>',
            svg_text(output_x + output_w / 2, y + 38, title, 21, text_anchor="middle", font_weight="700"),
            svg_text(output_x + output_w / 2, y + 68, subtitle, 15, text_anchor="middle", fill="#666666"),
        ])

    for index, target in enumerate(layout["factor_links"]):
        start_x = matrix_x + (index + 1) * cell + 8
        start_y = matrix_y + index * cell + cell / 2
        parts.append(f'<circle cx="{start_x}" cy="{start_y}" r="5" fill="#d43d2f"/>')
        parts.append(curved_path(start_x, start_y, method_x, method_centers[target], layout["factor_colors"][index], 2.2))

    for method_index, targets in layout["method_to_outputs"].items():
        for route_index, (output_index, color) in enumerate(targets):
            source_offset = (route_index - (len(targets) - 1) / 2) * 8
            incoming_methods = [
                index
                for index, routes in layout["method_to_outputs"].items()
                if any(target == output_index for target, _ in routes)
            ]
            target_offset = (
                incoming_methods.index(method_index) - (len(incoming_methods) - 1) / 2
            ) * 8
            route_offset = (output_index - method_index) * 12
            parts.append(
                curved_path(
                    method_x + method_w,
                    method_centers[method_index] + source_offset,
                    output_x,
                    output_centers[output_index] + target_offset,
                    color,
                    2.4,
                    route_offset=route_offset,
                )
            )

    legend_y = 1010
    parts.extend([
        svg_text(70, legend_y, "图例", 19, font_weight="700"),
        f'<rect x="130" y="{legend_y - 18}" width="28" height="18" fill="#dc5c31"/>',
        svg_text(168, legend_y - 2, "正相关", 16),
        f'<rect x="255" y="{legend_y - 18}" width="28" height="18" fill="#3978ad"/>',
        svg_text(293, legend_y - 2, "负相关", 16),
        svg_text(390, legend_y - 2, "* p<0.05   ** p<0.01   *** p<0.001", 16, fill="#555555"),
        f'<line x1="755" y1="{legend_y - 9}" x2="815" y2="{legend_y - 9}" stroke="#df7a35" stroke-width="3"/>',
        svg_text(825, legend_y - 2, "能源/经济路径", 16),
        f'<line x1="980" y1="{legend_y - 9}" x2="1040" y2="{legend_y - 9}" stroke="#3f7fb9" stroke-width="3"/>',
        svg_text(1050, legend_y - 2, "算力/服务路径", 16),
        f'<line x1="1200" y1="{legend_y - 9}" x2="1260" y2="{legend_y - 9}" stroke="#999999" stroke-width="3"/>',
        svg_text(1270, legend_y - 2, "数据/评价路径", 16),
        svg_text(70, 1060, "说明：连线表示问题二建模链条中的作用关系；统计显著性仅针对左侧逐时指标相关矩阵。", 16, fill="#666666"),
        "</svg>",
    ])

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / "q2_relationship_framework.svg"
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path


def generate_png(output_dir: Path, rows: list[list[float]], layout: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle

    correlations = layout["correlations"]
    width, height = layout["width"], layout["height"]
    matrix_x, matrix_y, cell = layout["matrix_x"], layout["matrix_y"], layout["cell"]
    method_x, method_w = layout["method_x"], layout["method_w"]
    output_x, output_w = layout["output_x"], layout["output_w"]
    method_centers = layout["method_centers"]
    output_centers = layout["output_centers"]

    # 中文用宋体，英文/数字用 Times New Roman。
    # 直接按文件路径加载字体，绕开 matplotlib 字体缓存（TTC 集合常不被缓存识别）。
    from matplotlib import font_manager

    font_candidates = [
        r"C:\Windows\Fonts\simsun.ttc",   # 宋体
        r"C:\Windows\Fonts\times.ttf",    # Times New Roman
    ]
    for font_path in font_candidates:
        try:
            font_manager.fontManager.addfont(font_path)
        except Exception:
            pass
    # 全局默认用宋体，保证中文不出现方块；纯英文/数字文本单独指定 Times New Roman。
    plt.rcParams["font.family"] = "SimSun"
    plt.rcParams["axes.unicode_minus"] = False

    def zh(**kwargs):
        kwargs["fontfamily"] = "SimSun"
        return kwargs

    def en(**kwargs):
        kwargs["fontfamily"] = "Times New Roman"
        return kwargs

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), width, height, facecolor="#FFFFFF", edgecolor="none"))
    ax.set_facecolor("#FFFFFF")

    ax.text(70, 70, "问题二：碳感知调度的指标关联、建模方法与决策输出", fontsize=34, fontweight="bold", color="#000000")
    ax.text(70, 108, f"基于方案B的 {len(rows):,} 条区域—小时记录；方格为 Pearson r，星号表示显著性", fontsize=18, color="#666666")
    ax.text(matrix_x + len(INDICATORS) * cell / 2, 160, "逐时指标相关性矩阵", fontsize=25, fontweight="bold", ha="center", color="#000000")
    ax.text(method_x + method_w / 2, 160, "分析与优化方法", fontsize=25, fontweight="bold", ha="center", color="#000000")
    ax.text(output_x + output_w / 2, 160, "核心决策输出", fontsize=25, fontweight="bold", ha="center", color="#000000")

    label_y = matrix_y + len(INDICATORS) * cell + 28
    for index, (label, _) in enumerate(INDICATORS):
        x = matrix_x + index * cell + cell / 2
        y = matrix_y + index * cell + cell / 2
        ax.text(x, label_y, label, fontsize=16, ha="right", va="top", rotation=45, rotation_mode="anchor")
        ax.text(matrix_x - 20, y + 6, label, fontsize=17, ha="right")

    for row_index in range(len(INDICATORS)):
        for column_index in range(row_index + 1):
            r = correlations[row_index][column_index]
            x = matrix_x + column_index * cell
            y = matrix_y + row_index * cell
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=mix_color(r), edgecolor="#ffffff", linewidth=2))
            text_color = "#ffffff" if abs(r) > 0.55 else "#303030"
            ax.text(x + cell / 2, y + 31, f"{r:+.2f}", fontsize=15, ha="center", color=text_color, fontweight="bold", **en())
            ax.text(x + cell / 2, y + 53, significance_stars(r, len(rows)), fontsize=16, ha="center", color=text_color, **en())

    for index, (title, subtitle) in enumerate(METHODS):
        y = 235 + index * 155
        box = FancyBboxPatch((method_x, y), method_w, 94, boxstyle="round,pad=0,rounding_size=18",
                             facecolor="#ffffff", edgecolor="#c8c8c8", linewidth=2)
        ax.add_patch(box)
        ax.add_patch(plt.Circle((method_x, y + 47), 7, color="#d43d2f"))
        ax.text(method_x + method_w / 2, y + 38, title, fontsize=21, ha="center", fontweight="bold", color="#000000")
        ax.text(method_x + method_w / 2, y + 68, subtitle, fontsize=15, ha="center", color="#666666")

    for index, (title, subtitle) in enumerate(OUTPUTS):
        y = 235 + index * 155
        box = FancyBboxPatch((output_x, y), output_w, 94, boxstyle="round,pad=0,rounding_size=18",
                             facecolor="#FFFFFF", edgecolor="#c8c8c8", linewidth=2)
        ax.add_patch(box)
        ax.add_patch(plt.Circle((output_x, y + 47), 7, color="#d43d2f"))
        ax.text(output_x + output_w / 2, y + 38, title, fontsize=21, ha="center", fontweight="bold", color="#000000")
        ax.text(output_x + output_w / 2, y + 68, subtitle, fontsize=15, ha="center", color="#666666")

    def draw_curve(x1, y1, x2, y2, color, lw, alpha=0.58, route_offset=0.0):
        span = x2 - x1
        vertices = [
            (x1, y1),
            (x1 + span * 0.30, y1 + route_offset),
            (x2 - span * 0.30, y2 - route_offset),
            (x2, y2),
        ]
        path = MplPath(
            vertices,
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
        )
        ax.add_patch(
            PathPatch(
                path,
                facecolor="none",
                edgecolor=color,
                linewidth=lw,
                alpha=alpha,
                capstyle="round",
            )
        )

    for index, target in enumerate(layout["factor_links"]):
        start_x = matrix_x + (index + 1) * cell + 8
        start_y = matrix_y + index * cell + cell / 2
        ax.add_patch(plt.Circle((start_x, start_y), 5, color="#d43d2f"))
        draw_curve(start_x, start_y, method_x, method_centers[target], layout["factor_colors"][index], 2.2)

    for method_index, targets in layout["method_to_outputs"].items():
        for route_index, (output_index, color) in enumerate(targets):
            source_offset = (route_index - (len(targets) - 1) / 2) * 8
            incoming_methods = [
                index
                for index, routes in layout["method_to_outputs"].items()
                if any(target == output_index for target, _ in routes)
            ]
            target_offset = (
                incoming_methods.index(method_index) - (len(incoming_methods) - 1) / 2
            ) * 8
            route_offset = (output_index - method_index) * 12
            draw_curve(
                method_x + method_w,
                method_centers[method_index] + source_offset,
                output_x,
                output_centers[output_index] + target_offset,
                color,
                2.4,
                route_offset=route_offset,
            )

    legend_y = 1010
    ax.text(70, legend_y, "图例", fontsize=19, fontweight="bold", color="#000000")
    ax.add_patch(Rectangle((130, legend_y - 18), 28, 18, facecolor="#dc5c31"))
    ax.text(168, legend_y - 2, "正相关", fontsize=16)
    ax.add_patch(Rectangle((255, legend_y - 18), 28, 18, facecolor="#3978ad"))
    ax.text(293, legend_y - 2, "负相关", fontsize=16)
    ax.text(390, legend_y - 2, "* p<0.05   ** p<0.01   *** p<0.001", fontsize=16, color="#555555", **en())
    ax.plot([755, 815], [legend_y - 9, legend_y - 9], color="#df7a35", lw=3)
    ax.text(825, legend_y - 2, "能源/经济路径", fontsize=16)
    ax.plot([980, 1040], [legend_y - 9, legend_y - 9], color="#3f7fb9", lw=3)
    ax.text(1050, legend_y - 2, "算力/服务路径", fontsize=16)
    ax.plot([1200, 1260], [legend_y - 9, legend_y - 9], color="#999999", lw=3)
    ax.text(1270, legend_y - 2, "数据/评价路径", fontsize=16)
    ax.text(70, 1060, "说明：连线表示问题二建模链条中的作用关系；统计显著性仅针对左侧逐时指标相关矩阵。", fontsize=16, color="#666666")

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / "q2_relationship_framework.png"
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    rows = aligned_indicator_rows(args.output_dir)
    layout = layout_data(rows)
    svg_path = generate_svg(args.output_dir, rows, layout)
    print(f"完成：{svg_path.resolve()}")
    if args.png:
        png_path = generate_png(args.output_dir, rows, layout)
        print(f"完成：{png_path.resolve()}")


if __name__ == "__main__":
    main()
