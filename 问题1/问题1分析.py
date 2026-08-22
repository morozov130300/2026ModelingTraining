from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact, spearmanr, rankdata, gaussian_kde
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

warnings.filterwarnings("ignore")
SEED = 2024
np.random.seed(SEED)
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "数据预处理结果" / "data_merged.csv"
OUTPUT_DIR = BASE_DIR / "问题1" / "问题1结果"
FIGURE_DIR = OUTPUT_DIR / "附图"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 图形字体规范：中文使用微软雅黑，英文和数字使用Times New Roman。
FONT_CN = FontProperties(family="Microsoft YaHei")
FONT_EN = FontProperties(family="Times New Roman")
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

AGE = "年龄"
SEX = "性别"
BLOOD = ["WBC", "N", "L", "M", "RBC", "HB", "PLT", "RDW"]
ALL_CONTINUOUS = [AGE] + BLOOD
FUNCTION_GROUPS = {
    "氧气运输与贫血诊断": ["RBC", "HB", "RDW"],
    "免疫防御与感染鉴别": ["WBC", "N", "L", "M"],
    "凝血与止血": ["PLT"],
}
GROUP_ORDER = {v: i for i, v in enumerate(FUNCTION_GROUPS)}

# 参考区间来源：WS/T 405-2012《血细胞分析参考区间》（国家卫生健康委员会发布）。
# 适用人群：中国健康成人，按性别/年龄分层。本脚本使用合并口径（不分层）作为暂定区间。
# 局限：不同实验室、检测系统、人群可能导致参考区间变化；M 无 WS/T 405-2012 分层数据，沿用文献常用范围。
REFERENCE_INTERVALS = {
    "WBC": (3.5, 9.5), "N": (1.8, 6.3), "L": (1.1, 3.2),
    "M": (0.1, 0.6), "RBC": (4.3, 5.8), "HB": (120.0, 160.0),
    "PLT": (125.0, 350.0), "RDW": (11.5, 15.0),
}
REFERENCE_SOURCE = "WS/T 405-2012《血细胞分析参考区间》（国家卫生健康委员会发布，基于中国健康人群大规模多中心数据）"


def apply_font_rules(fig):
    """按文本内容设置字体：含中文使用微软雅黑，纯英文/数字使用Times New Roman。"""
    for text in fig.findobj(plt.Text):
        value = text.get_text()
        text.set_fontproperties(FONT_CN if any("\u4e00" <= char <= "\u9fff" for char in value) else FONT_EN)


def save_figure(fig, path):
    apply_font_rules(fig)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def read_data():
    data = pd.read_csv(INPUT_PATH)
    data["label"] = pd.to_numeric(data["label"], errors="raise").astype(int)
    for col in ALL_CONTINUOUS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[SEX] = data[SEX].astype(str)
    return data


def cliff_delta(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ranks = rankdata(np.concatenate([x, y]))
    nx, ny = len(x), len(y)
    u = ranks[:nx].sum() - nx * (nx + 1) / 2
    return float(2 * u / (nx * ny) - 1)


def iqr(x):
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def auc_abs_direction(x, y):
    """返回以 label=1 为阳性类的 AUC；方向不足 0.5 时取 1-AUC 仅用于强度排序。"""
    values = np.concatenate([x, y])
    labels = np.r_[np.ones(len(x)), np.zeros(len(y))]
    auc = roc_auc_score(labels, values)
    return float(auc), float(max(auc, 1 - auc))


def bh(p):
    """使用NumPy实现Benjamini-Hochberg FDR校正。"""
    p = np.asarray(p, dtype=float)
    result = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    values = p[valid]
    if values.size == 0:
        return result
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    result[valid] = restored
    return result


def descriptive_table(data):
    rows = []
    for group_label, group_name in [(1, "流感A组"), (0, "健康组")]:
        group = data.loc[data.label == group_label]
        for col in ALL_CONTINUOUS:
            values = group[col].dropna().to_numpy(float)
            rows.append({
                "组": group_name, "label": group_label, "变量": col,
                "n": len(values), "中位数": np.median(values),
                "Q1": np.percentile(values, 25), "Q3": np.percentile(values, 75),
                "IQR": iqr(values),
                "偏度": pd.Series(values).skew(),
                "均值": np.mean(values), "标准差": np.std(values, ddof=1),
            })
    return pd.DataFrame(rows)


def difference_table(data):
    rows = []
    for col in ALL_CONTINUOUS:
        x = data.loc[data.label == 1, col].dropna().to_numpy(float)
        y = data.loc[data.label == 0, col].dropna().to_numpy(float)
        u, p = mannwhitneyu(x, y, alternative="two-sided")
        delta = cliff_delta(x, y)
        auc_raw, auc_strength = auc_abs_direction(x, y)
        med_x, med_y = np.median(x), np.median(y)
        rows.append({
            "变量": col, "所属功能组": next(g for g, cols in FUNCTION_GROUPS.items() if col in cols) if col != AGE else "基本信息",
            "流感组中位数": med_x, "健康组中位数": med_y,
            "流感组IQR": iqr(x), "健康组IQR": iqr(y),
            "Mann-Whitney U": u, "原始p值": p,
            "Cliff's delta": delta, "单变量AUC": auc_raw,
            "AUC强度": auc_strength, "方向": "↑" if med_x > med_y else "↓" if med_x < med_y else "="
        })
    result = pd.DataFrame(rows)
    result["组内FDR校正p值"] = np.nan
    for group_name in result["所属功能组"].unique():
        mask = result["所属功能组"] == group_name
        result.loc[mask, "组内FDR校正p值"] = bh(result.loc[mask, "原始p值"])
    result["全局BH校正p值"] = bh(result["原始p值"])
    result["潜力评级"] = np.select(
        [(result["全局BH校正p值"] < .05) & (result["Cliff's delta"].abs() >= .5) & (result["AUC强度"] >= .8),
         (result["全局BH校正p值"] < .05) & (result["Cliff's delta"].abs() >= .3),
         (result["全局BH校正p值"] < .05)],
        ["高", "中", "低"], default="低")
    return result


def categorical_difference(data):
    table = pd.crosstab(data[SEX], data["label"]).reindex(columns=[0, 1], fill_value=0)
    chi2, p, dof, expected = chi2_contingency(table)
    result = table.reset_index().rename(columns={0: "健康组", 1: "流感A组"})
    result.to_csv(OUTPUT_DIR / "性别构成.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"检验": "Pearson卡方", "统计量": chi2, "自由度": dof, "p值": p}]).to_csv(OUTPUT_DIR / "性别差异检验.csv", index=False, encoding="utf-8-sig")


def plot_group_violin(ax, data, column):
    groups = [data.loc[data["label"] == label, column].dropna().to_numpy(float) for label in [0, 1]]
    parts = ax.violinplot(groups, positions=[0, 1], showmeans=False, showmedians=True, showextrema=True)
    colors = ["#4575b4", "#d73027"]
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.65)
    for key in ("cmins", "cmaxes", "cbars", "cmedians"):
        if key in parts:
            parts[key].set_color("#333333")
            parts[key].set_linewidth(0.8)
    ax.set_xticks([0, 1], ["健康组", "流感A组"])


def plot_group_hist(ax, data, column, bins=30):
    for label, name, color in [(0, "健康组", "#4575b4"), (1, "流感A组", "#d73027")]:
        values = data.loc[data["label"] == label, column].dropna().to_numpy(float)
        ax.hist(values, bins=bins, density=True, alpha=0.45, label=name, color=color)
    ax.legend()


def plot_log_density(ax, data, column):
    for label, name, color in [(0, "健康组", "#4575b4"), (1, "流感A组", "#d73027")]:
        values = data.loc[data["label"] == label, column].dropna().to_numpy(float)
        if len(values) > 1 and np.ptp(values) > 0:
            grid = np.linspace(values.min(), values.max(), 300)
            ax.plot(grid, gaussian_kde(values)(grid), label=name, color=color)
        else:
            ax.axvline(values[0], label=name, color=color)
    ax.legend()


def save_distribution_figures(data, descriptive):
    for group_name, columns in FUNCTION_GROUPS.items():
        fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4), squeeze=False)
        for ax, col in zip(axes[0], columns):
            plot_group_violin(ax, data, col)
            ax.set_title(col)
            ax.set_xlabel("组别")
        fig.suptitle(f"问题1：{group_name}分布比较")
        fig.tight_layout()
        save_figure(fig, FIGURE_DIR / f"分布_{group_name}.png")
    for col in ["WBC", "N", "L", "M", "PLT"]:
        transform_col = f"{col}_变换后"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plot_group_hist(axes[0], data, col)
        plot_group_hist(axes[1], data, transform_col)
        axes[0].set_title(f"{col}原始刻度")
        axes[1].set_title(f"{col}Box-Cox变换后")
        fig.tight_layout()
        save_figure(fig, FIGURE_DIR / f"变换对比_{col}.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_log_density(ax, data, "NLR")
    ax.set_xscale("log")
    ax.set_title("NLR按组密度图（横轴为log刻度）")
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "NLR按组密度图.png")


def relation_analysis(data):
    columns = ALL_CONTINUOUS
    corr_all = data[columns].corr(method="spearman")
    corr_all.to_csv(OUTPUT_DIR / "全样本Spearman相关矩阵.csv", encoding="utf-8-sig")
    for label, name in [(1, "流感A组"), (0, "健康组")]:
        data.loc[data.label == label, columns].corr(method="spearman").to_csv(OUTPUT_DIR / f"{name}Spearman相关矩阵.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr_all.to_numpy(float), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(image, ax=ax, label="Spearman相关系数")
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            value = corr_all.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("全样本Spearman相关矩阵")
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "全样本相关热图.png")
    validation = []
    for pair, hypothesis in [(("RBC", "HB"), "H1/H6氧气运输结构"), (("WBC", "N"), "H6免疫结构")]:
        row = {"指标1": pair[0], "指标2": pair[1], "假设": hypothesis}
        for label, name in [(1, "流感A组"), (0, "健康组"), (None, "全样本")]:
            subset = data if label is None else data.loc[data.label == label]
            r, p = spearmanr(subset[pair[0]], subset[pair[1]], nan_policy="omit")
            row[f"{name}Spearman_r"] = r; row[f"{name}p值"] = p
        row["是否支持|r|>0.6"] = bool(abs(row["全样本Spearman_r"]) > .6)
        validation.append(row)
    pd.DataFrame(validation).to_csv(OUTPUT_DIR / "组内相关验证.csv", index=False, encoding="utf-8-sig")


def vif_table(data):
    rows = []
    matrix = data[ALL_CONTINUOUS].dropna().to_numpy(float)
    for i, col in enumerate(ALL_CONTINUOUS):
        y = matrix[:, i]; x = np.delete(matrix, i, axis=1)
        design = np.c_[np.ones(len(x)), x]
        pred = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append({"变量": col, "VIF": np.inf if r2 >= 1 else 1 / (1 - r2)})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "问题1按组VIF.csv", index=False, encoding="utf-8-sig")


def combination_metrics(data):
    data = data.copy()
    data["NLR"] = data["N"] / data["L"].replace(0, np.nan)
    data["MLR"] = data["M"] / data["L"].replace(0, np.nan)
    data["PLR"] = data["PLT"] / data["L"].replace(0, np.nan)
    rows = []
    for col in ["N", "L", "M", "NLR", "MLR", "PLR"]:
        x = data.loc[data.label == 1, col].dropna().to_numpy(float)
        y = data.loc[data.label == 0, col].dropna().to_numpy(float)
        u, p = mannwhitneyu(x, y, alternative="two-sided")
        raw_auc, strength = auc_abs_direction(x, y)
        rows.append({"指标": col, "Mann-Whitney U": u, "p值": p, "Cliff's delta": cliff_delta(x, y), "单变量AUC": raw_auc, "AUC强度": strength})
    result = pd.DataFrame(rows); result["BH校正p值"] = bh(result["p值"])
    result.to_csv(OUTPUT_DIR / "组合指标评估.csv", index=False, encoding="utf-8-sig")
    data[["序号", "label", "NLR", "MLR", "PLR"]].to_csv(OUTPUT_DIR / "组合指标数据.csv", index=False, encoding="utf-8-sig")
    # 比值放大诊断：统计L极小值数量及其对组合指标的影响
    l_vals = data["L"].dropna()
    l_small = (l_vals < 0.3).sum()
    l_medium = ((l_vals >= 0.3) & (l_vals < 0.5)).sum()
    mlr_max = data["MLR"].dropna().max()
    nlr_max = data["NLR"].dropna().max()
    pd.DataFrame([{
        "诊断项": "L极小值(<0.3)样本数", "值": int(l_small),
    }, {
        "诊断项": "L偏小(0.3-0.5)样本数", "值": int(l_medium),
    }, {
        "诊断项": "MLR最大值", "值": float(mlr_max),
    }, {
        "诊断项": "NLR最大值", "值": float(nlr_max),
    }, {
        "诊断项": "比值放大说明",
        "值": f"MLR/NLR/PLR的高AUC部分来自L极小值导致的比值放大效应（L<0.3共{l_small}例），属病例-对照谱系下的预期内虚高，未经独立验证",
    }]).to_csv(OUTPUT_DIR / "比值放大诊断.csv", index=False, encoding="utf-8-sig")
    return result


def effect_heatmap(diff):
    ordered = diff.copy()
    ordered["order"] = ordered["所属功能组"].map({"基本信息": -1, **GROUP_ORDER})
    ordered = ordered.sort_values(["order", "变量"])
    fig, ax = plt.subplots(figsize=(7, 7))
    values = ordered["Cliff's delta"].to_numpy(float).reshape(-1, 1)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(image, ax=ax, label="Cliff's delta")
    ax.set_yticks(range(len(ordered)), ordered["变量"])
    ax.set_xticks([0], ["Cliff's delta（流感A相对健康）"])
    for i, value in enumerate(values[:, 0]):
        ax.text(0, i, f"{value:.2f}", ha="center", va="center")
    ax.set_title("效应量热图")
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "效应量热图.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_data = ordered.sort_values("AUC强度", ascending=True)
    colors = plot_data["所属功能组"].map({"基本信息": "#777777", "氧气运输与贫血诊断": "#1b9e77", "免疫防御与感染鉴别": "#d95f02", "凝血与止血": "#7570b3"})
    ax.barh(plot_data["变量"], plot_data["AUC强度"], color=colors)
    ax.axvline(.5, color="black", ls="--", lw=1)
    ax.set_xlim(.5, 1); ax.set_xlabel("AUC强度（方向折叠后）"); ax.set_title("单变量AUC排序")
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "单变量AUC排序.png")


def reference_abnormal(data):
    rows, matrix = [], pd.DataFrame(index=data.index)
    for col, (lower, upper) in REFERENCE_INTERVALS.items():
        status = np.select([data[col] < lower, data[col] > upper], ["偏低", "偏高"], default="正常")
        matrix[col] = status
        for label, name in [(0, "健康组"), (1, "流感A组")]:
            subset = status[data.label.to_numpy() == label]
            abnormal = np.sum(subset != "正常"); total = len(subset)
            table = np.array([[abnormal, total - abnormal], [0, 0]])
            # 两组比例比较；小计数时保留Fisher作为备用说明。
            other = status[data.label.to_numpy() != label]
            other_abnormal = np.sum(other != "正常")
            contingency = np.array([[abnormal, total - abnormal], [other_abnormal, len(other) - other_abnormal]])
            if np.all(contingency >= 0):
                try: p = fisher_exact(contingency)[1] if np.min(contingency) < 5 else chi2_contingency(contingency)[1]
                except ValueError: p = np.nan
            else: p = np.nan
            rows.append({"变量": col, "参考下限": lower, "参考上限": upper, "组": name, "异常数": int(abnormal), "样本数": total, "异常比例": abnormal / total, "与另一组比较p值": p})
    matrix["异常指标数"] = (matrix[BLOOD] != "正常").sum(axis=1)
    count_rows = []
    for label, name in [(0, "健康组"), (1, "流感A组")]:
        count_rows.append({"组": name, "n": int((data.label == label).sum()), "异常指标数中位数": matrix.loc[data.label == label, "异常指标数"].median(), "异常指标数IQR": iqr(matrix.loc[data.label == label, "异常指标数"])})
    x = matrix.loc[data.label == 1, "异常指标数"]; y = matrix.loc[data.label == 0, "异常指标数"]
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "参考区间异常比例.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(count_rows + [{"组": "两组比较", "n": "", "异常指标数中位数": u, "异常指标数IQR": p}]).to_csv(OUTPUT_DIR / "异常计数比较.csv", index=False, encoding="utf-8-sig")
    patterns = matrix[BLOOD].apply(lambda row: " + ".join(f"{c}{s[0]}" for c, s in row.items() if s != "正常") or "全部正常", axis=1)
    pattern_table = pd.crosstab(patterns, data["label"]).rename(columns={0: "健康组", 1: "流感A组"})
    pattern_table = pattern_table.reindex(columns=["健康组", "流感A组"], fill_value=0)
    pattern_table["合计"] = pattern_table.sum(axis=1)
    pattern_table = pattern_table.sort_values("合计", ascending=False).drop(columns="合计")
    pattern_table.head(20).to_csv(OUTPUT_DIR / "异常模式频次.csv", encoding="utf-8-sig")


def clustering_analysis(data):
    # 仅用 L/M/N 聚类（最具区分力的三个变量），评估是否值得保留对照框架
    cols = [f"{c}_变换后" for c in ["L", "M", "N"]]
    x = data[cols].copy()
    x = pd.DataFrame(StandardScaler().fit_transform(x), columns=cols, index=data.index)
    scores = []
    for k in range(2, 7):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
        scores.append({"K": k, "轮廓系数": silhouette_score(x, labels), "模型": "Ward层次聚类"})
    score_df = pd.DataFrame(scores); score_df.to_csv(OUTPUT_DIR / "聚类K选择_LMN.csv", index=False, encoding="utf-8-sig")
    best_k = int(score_df.loc[score_df["轮廓系数"].idxmax(), "K"])
    best_silhouette = float(score_df.loc[score_df["轮廓系数"].idxmax(), "轮廓系数"])
    cluster = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(x)
    cross = pd.crosstab(cluster, data["label"]).rename(columns={0: "健康组", 1: "流感A组"})
    cross["簇样本数"] = cross.sum(axis=1)
    cross["流感组比例"] = cross["流感A组"] / cross["簇样本数"]
    cross.to_csv(OUTPUT_DIR / "聚类簇组别交叉表_LMN.csv", encoding="utf-8-sig")
    profile = pd.DataFrame(x).assign(簇=cluster).groupby("簇").mean()
    profile.to_csv(OUTPUT_DIR / "聚类簇剖面_LMN.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(profile.to_numpy(float), cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
    fig.colorbar(image, ax=ax, label="标准化均值")
    ax.set_xticks(range(len(profile.columns)), profile.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(profile.index)), [f"簇{i}" for i in profile.index])
    for i in range(profile.shape[0]):
        for j in range(profile.shape[1]):
            ax.text(j, i, f"{profile.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Ward层次聚类簇剖面（L/M/N，K={best_k}，轮廓系数={best_silhouette:.3f}）")
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "聚类簇剖面_LMN.png")
    # 同时尝试全变量聚类作为对比
    all_cols = [f"{c}_变换后" for c in BLOOD]
    x_all = pd.DataFrame(StandardScaler().fit_transform(data[all_cols]), columns=all_cols, index=data.index)
    all_scores = []
    for k in range(2, 7):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x_all)
        all_scores.append({"K": k, "轮廓系数": silhouette_score(x_all, labels), "模型": "Ward层次聚类"})
    all_score_df = pd.DataFrame(all_scores)
    all_best_silhouette = float(all_score_df.loc[all_score_df["轮廓系数"].idxmax(), "轮廓系数"])
    # 输出诊断：LMN聚类 vs 全变量聚类的轮廓系数对比
    pd.DataFrame([
        {"方案": "L/M/N三变量聚类", "最佳K": best_k, "最佳轮廓系数": best_silhouette},
        {"方案": "全变量聚类", "最佳K": int(all_score_df.loc[all_score_df["轮廓系数"].idxmax(), "K"]), "最佳轮廓系数": all_best_silhouette},
        {"方案": "判定", "最佳K": "", "最佳轮廓系数": 1 if best_silhouette >= 0.25 else 0,
         "说明": "轮廓系数≥0.25保留对照框架；<0.25建议删除第8节"},
    ]).to_csv(OUTPUT_DIR / "聚类方案对比.csv", index=False, encoding="utf-8-sig")


def _group_summary_rows(diff, combo, descriptive, data):
    """按方案 3.1/7.3 生成功能组级摘要行。"""
    combo_lookup = combo.set_index("指标")[["Cliff's delta", "单变量AUC", "AUC强度"]].to_dict("index")
    rows = []
    for group, cols in FUNCTION_GROUPS.items():
        subset = diff[diff["变量"].isin(cols)]
        best = subset.sort_values("AUC强度", ascending=False).iloc[0]
        rows.append({
            "功能组": group,
            "组内指标": ", ".join(cols),
            "组内指标效应量绝对值均值": float(subset["Cliff's delta"].abs().mean()),
            "组内单变量AUC强度中位数": float(subset["AUC强度"].median()),
            "组内最强差异变量": best["变量"],
            "组内最强差异方向": best["方向"],
            "组内最强差异AUC强度": float(best["AUC强度"]),
            "组内最强差异Cliff's delta": float(best["Cliff's delta"]),
            "组内最强差异全局BH校正p值": float(best["全局BH校正p值"]),
        })
    # 免疫组补充组合指标摘要
    immune = [r for r in rows if r["功能组"] == "免疫防御与感染鉴别"][0]
    for name in ["NLR", "MLR", "PLR"]:
        if name in combo_lookup:
            row = combo_lookup[name]
            immune[f"{name}Cliff's delta"] = float(row["Cliff's delta"])
            immune[f"{name}AUC强度"] = float(row["AUC强度"])
            immune[f"{name}单变量AUC"] = float(row["单变量AUC"])
    return pd.DataFrame(rows)


def _group_level_conclusions(rows):
    """按方案 7.3 生成功能组层面结论句。"""
    ordered = rows.sort_values("组内单变量AUC强度中位数", ascending=False)
    lines = []
    for _, r in ordered.iterrows():
        auc = r["组内单变量AUC强度中位数"]
        delta = r["组内指标效应量绝对值均值"]
        best_auc = r["组内最强差异AUC强度"]
        best_p = r["组内最强差异全局BH校正p值"]
        lines.append(
            f"{r['功能组']}：组内效应量绝对值均值 {delta:.3f}，"
            f"组内 AUC 强度中位数 {auc:.3f}，"
            f"最强差异变量为 {r['组内最强差异变量']}（方向 {r['组内最强差异方向']}，"
            f"AUC 强度 {best_auc:.3f}，全局 BH 校正 p={best_p:.2e}）。"
        )
    return lines


def _variable_level_conclusions(diff, combo):
    """按方案 7.3 生成变量层面结论句。"""
    lines = []
    for _, r in diff.sort_values("AUC强度", ascending=False).iterrows():
        delta = r["Cliff's delta"]
        auc = r["AUC强度"]
        p = r["全局BH校正p值"]
        lines.append(
            f"{r['变量']}（{r['所属功能组']}）：Cliff's delta={delta:.3f}，"
            f"AUC 强度={auc:.3f}，全局 BH 校正 p={p:.2e}，方向 {r['方向']}，"
            f"潜力评级 {r['潜力评级']}。"
        )
    for _, r in combo[combo["指标"].isin(["NLR", "MLR", "PLR"])].sort_values("AUC强度", ascending=False).iterrows():
        delta = r["Cliff's delta"]
        auc = r["AUC强度"]
        p = r["BH校正p值"]
        lines.append(
            f"{r['指标']}（免疫组合指标）：Cliff's delta={delta:.3f}，"
            f"AUC 强度={auc:.3f}，全局 BH 校正 p={p:.2e}。"
        )
    return lines


def _reference_abnormal_summary(data):
    """按方案 8.1-8.5 生成辅视角摘要表。"""
    rows, matrix = [], pd.DataFrame(index=data.index)
    for col, (lower, upper) in REFERENCE_INTERVALS.items():
        status = np.select([data[col] < lower, data[col] > upper], ["偏低", "偏高"], default="正常")
        matrix[col] = status
        for label, name in [(0, "健康组"), (1, "流感A组")]:
            subset = status[data.label.to_numpy() == label]
            abnormal = int(np.sum(subset != "正常")); total = int(len(subset))
            other = status[data.label.to_numpy() != label]
            other_abnormal = int(np.sum(other != "正常"))
            contingency = np.array([[abnormal, total - abnormal], [other_abnormal, len(other) - other_abnormal]])
            try:
                p = fisher_exact(contingency)[1] if np.min(contingency) < 5 else chi2_contingency(contingency)[1]
            except ValueError:
                p = np.nan
            rows.append({
                "变量": col, "参考下限": lower, "参考上限": upper, "组": name,
                "异常数": abnormal, "样本数": total, "异常比例": abnormal / total,
                "与另一组比较p值": p,
            })
    matrix["异常指标数"] = (matrix[BLOOD] != "正常").sum(axis=1)
    count_rows = []
    for label, name in [(0, "健康组"), (1, "流感A组")]:
        count_rows.append({
            "组": name, "n": int((data.label == label).sum()),
            "异常指标数中位数": float(matrix.loc[data.label == label, "异常指标数"].median()),
            "异常指标数IQR": float(iqr(matrix.loc[data.label == label, "异常指标数"])),
        })
    x = matrix.loc[data.label == 1, "异常指标数"]
    y = matrix.loc[data.label == 0, "异常指标数"]
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    count_rows.append({"组": "两组比较", "n": "", "异常指标数中位数": float(u), "异常指标数IQR": float(p)})
    patterns = matrix[BLOOD].apply(lambda row: " + ".join(f"{c}{s[0]}" for c, s in row.items() if s != "正常") or "全部正常", axis=1)
    pattern_table = pd.crosstab(patterns, data["label"]).rename(columns={0: "健康组", 1: "流感A组"})
    pattern_table = pattern_table.reindex(columns=["健康组", "流感A组"], fill_value=0)
    pattern_table["合计"] = pattern_table.sum(axis=1)
    pattern_table = pattern_table.sort_values("合计", ascending=False).drop(columns="合计")
    return pd.DataFrame(rows), pd.DataFrame(count_rows), pattern_table.head(20)


def _clustering_summary(data):
    """按方案 9.1-9.5 生成对照框架摘要（仅用 L/M/N）。"""
    cols = [f"{c}_变换后" for c in ["L", "M", "N"]]
    x = pd.DataFrame(StandardScaler().fit_transform(data[cols]), columns=cols, index=data.index)
    scores = []
    for k in range(2, 7):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
        scores.append({"K": k, "轮廓系数": float(silhouette_score(x, labels)), "模型": "Ward层次聚类"})
    score_df = pd.DataFrame(scores)
    best_k = int(score_df.loc[score_df["轮廓系数"].idxmax(), "K"])
    best_silhouette = float(score_df.loc[score_df["轮廓系数"].idxmax(), "轮廓系数"])
    cluster = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(x)
    cross = pd.crosstab(cluster, data["label"]).rename(columns={0: "健康组", 1: "流感A组"})
    cross["簇样本数"] = cross.sum(axis=1)
    cross["流感组比例"] = cross["流感A组"] / cross["簇样本数"]
    profile = pd.DataFrame(x).assign(簇=cluster).groupby("簇").mean()
    return score_df, cross, profile, best_k, best_silhouette


def write_summary(diff, combo, data):
    descriptive = descriptive_table(data)
    group_rows = _group_summary_rows(diff, combo, descriptive, data)
    group_rows.to_csv(OUTPUT_DIR / "功能组综合比较.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"功能组层面结论": _group_level_conclusions(group_rows)}).to_csv(
        OUTPUT_DIR / "功能组层面结论.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"变量层面结论": _variable_level_conclusions(diff, combo)}).to_csv(
        OUTPUT_DIR / "变量层面结论.csv", index=False, encoding="utf-8-sig"
    )
    abnormal_df, count_df, pattern_df = _reference_abnormal_summary(data)
    abnormal_df.to_csv(OUTPUT_DIR / "参考区间异常比例.csv", index=False, encoding="utf-8-sig")
    count_df.to_csv(OUTPUT_DIR / "异常计数比较.csv", index=False, encoding="utf-8-sig")
    pattern_df.to_csv(OUTPUT_DIR / "异常模式频次.csv", encoding="utf-8-sig")
    score_df, cross_df, profile_df, best_k, best_silhouette = _clustering_summary(data)
    score_df.to_csv(OUTPUT_DIR / "聚类K选择_LMN.csv", index=False, encoding="utf-8-sig")
    cross_df.to_csv(OUTPUT_DIR / "聚类簇组别交叉表_LMN.csv", encoding="utf-8-sig")
    profile_df.to_csv(OUTPUT_DIR / "聚类簇剖面_LMN.csv", encoding="utf-8-sig")
    # 读取预处理阶段的bootstrap CI，输出问题1专用的CI摘要表
    bootstrap_path = BASE_DIR / "数据预处理结果" / "bootstrap置信区间.csv"
    if bootstrap_path.exists():
        bootstrap_df = pd.read_csv(bootstrap_path)
        bootstrap_df.to_csv(OUTPUT_DIR / "bootstrap置信区间摘要.csv", index=False, encoding="utf-8-sig")
    with open(OUTPUT_DIR / "运行摘要.txt", "w", encoding="utf-8") as f:
        f.write(f"随机种子={SEED}\n样本量={len(data)}；流感A={int((data.label == 1).sum())}；健康={int((data.label == 0).sum())}\n")
        f.write("本脚本仅进行问题1统计分析，不训练分类器。\n")
        f.write("章节对应：3.1按功能组展开，7为辅视角，8为对照框架。\n")
        f.write("数据修正：WBC单位混用已修正（÷1000），RDW-SD混入已剔除。\n")


def main():
    data = read_data()
    data["NLR"] = data["N"] / data["L"].replace(0, np.nan)
    descriptive = descriptive_table(data)
    descriptive.to_csv(OUTPUT_DIR / "按组描述性统计.csv", index=False, encoding="utf-8-sig")
    diff = difference_table(data)
    diff.to_csv(OUTPUT_DIR / "差异分析表.csv", index=False, encoding="utf-8-sig")
    categorical_difference(data)
    save_distribution_figures(data, descriptive)
    relation_analysis(data)
    vif_table(data)
    combo = combination_metrics(data)
    effect_heatmap(diff)
    reference_abnormal(data)
    clustering_analysis(data)
    write_summary(diff, combo, data)
    evidence = diff[["变量", "所属功能组", "全局BH校正p值", "Cliff's delta", "单变量AUC", "潜力评级"]].copy()
    combo2 = combo[["指标", "BH校正p值", "Cliff's delta", "单变量AUC"]].rename(columns={"指标": "变量", "BH校正p值": "全局BH校正p值"})
    evidence = pd.concat([evidence, combo2.assign(所属功能组="免疫组合指标", 潜力评级="补充")], ignore_index=True)
    evidence.to_csv(OUTPUT_DIR / "综合证据表.csv", index=False, encoding="utf-8-sig")
    # 按方案 3.1 输出按功能组拆分的差异子表，便于报告逐组展开
    for group_name, cols in FUNCTION_GROUPS.items():
        subset = diff[diff["变量"].isin(cols)].copy()
        subset.to_csv(OUTPUT_DIR / f"差异分析表_{group_name}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
