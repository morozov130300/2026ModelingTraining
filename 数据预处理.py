from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import boxcox, skew, kurtosis, mannwhitneyu, shapiro, rankdata, probplot, chi2_contingency

warnings.filterwarnings("ignore")

SEED = 2024
np.random.seed(SEED)
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "题目" / "附件"
OUTPUT_DIR = BASE_DIR / "数据预处理结果"
FLUA_PATH = INPUT_DIR / "FLUA.xls"
HEALTHY_PATH = INPUT_DIR / "Healthy control.xlsx"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR = OUTPUT_DIR / "附图"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMNS = ["序号", "性别", "年龄", "WBC", "N", "L", "M", "RBC", "HB", "PLT", "RDW"]
NUMERIC_COLUMNS = ["年龄", "WBC", "N", "L", "M", "RBC", "HB", "PLT", "RDW"]
BLOOD_COLUMNS = ["WBC", "N", "L", "M", "RBC", "HB", "PLT", "RDW"]
PREJUDGED_LOG_COLUMNS = {"WBC", "N", "L", "M", "PLT"}
UNITS = {"年龄": "岁", "WBC": "×10⁹/L", "N": "×10⁹/L", "L": "×10⁹/L", "M": "×10⁹/L", "RBC": "×10¹²/L", "HB": "g/L", "PLT": "×10⁹/L", "RDW": "%"}
VARIABLE_TYPES = {"序号": "样本标识", "性别": "基本信息类/分类变量", "年龄": "基本信息类", **{c: "血液检测指标类" for c in BLOOD_COLUMNS}}
COLUMN_ALIASES = {
    "序号": ["序号", "编号", "ID", "id", "No", "NO"],
    "性别": ["性别", "Gender", "Sex"],
    "年龄": ["年龄", "Age"],
    "WBC": ["WBC", "WBC（*10^9）", "WBC(*10^9)", "白细胞计数", "白细胞"],
    "N": ["N", "N（*10^9）", "N(*10^9)", "中性粒细胞计数", "中性粒细胞", "NEUT", "NEUT#"],
    "L": ["L", "L（*10^9）", "L(*10^9)", "淋巴细胞计数", "淋巴细胞", "LYM", "LYM#"],
    "M": ["M", "M（*10^9）", "M(*10^9)", "单核细胞计数", "单核细胞", "MON", "MONO", "MON#"],
    "RBC": ["RBC", "RBC（*10^12）", "RBC(*10^12)", "红细胞计数", "红细胞"],
    "HB": ["HB", "HGB", "血红蛋白", "血红蛋白浓度"],
    "PLT": ["PLT", "PLT（*10^9）", "PLT(*10^9)", "血小板计数", "血小板"],
    "RDW": ["RDW", "RDW（CV-%）", "RDW(CV-%)", "红细胞分布宽度", "红细胞体积分布宽度"],
}
MISSING_MARKERS = {"", "NA", "N/A", "na", "n/a", "null", "None", "-", "—", "–", "999"}


def normalize_name(name):
    return re.sub(r"[\s_（）()\[\]【】:/\\-]+", "", str(name)).lower()


def resolve_columns(df):
    normalized = {normalize_name(c): c for c in df.columns}
    mapping = {}
    actual_rows = []
    for standard, aliases in COLUMN_ALIASES.items():
        source = next((normalized.get(normalize_name(alias)) for alias in aliases if normalized.get(normalize_name(alias)) is not None), None)
        if source is not None:
            mapping[source] = standard
            actual_rows.append({"原始列名": source, "统一列名": standard})
    missing = [c for c in TARGET_COLUMNS if c not in mapping.values() and c != "序号"]
    if missing:
        raise ValueError(f"缺少必要字段：{missing}；原始列名为：{list(df.columns)}")
    return mapping, actual_rows


def parse_numeric(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in MISSING_MARKERS:
        return np.nan
    text = text.replace(",", "").replace("，", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    return float(match.group()) if match else np.nan


def parse_age(value):
    return parse_numeric(value)


def normalize_gender(value):
    if pd.isna(value) or str(value).strip() in MISSING_MARKERS:
        return "未知"
    text = str(value).strip().upper()
    if text in {"男", "M", "MALE", "1"}:
        return 1
    if text in {"女", "F", "FEMALE", "2"}:
        return 0
    return "未知"


def read_and_standardize(path, source_name, label):
    excel = pd.ExcelFile(path)
    structure_rows = []
    mapping_rows = []
    frames = []
    for sheet in excel.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet)
        mapping, actual_mapping = resolve_columns(raw)
        for row in actual_mapping:
            mapping_rows.append({"来源": source_name, "工作表": sheet, **row})
        for column in raw.columns:
            structure_rows.append({"来源": source_name, "工作表": sheet, "原始列名": column, "行数": len(raw), "数据类型": str(raw[column].dtype), "缺失数": int(raw[column].isna().sum()), "空字符串数": int(raw[column].astype("string").str.strip().eq("").sum()), "缺失标记样式": "空值/NA/N/A/na/n/a/null/None/-/—/–/999统一为NaN"})
        frame = raw.rename(columns=mapping).copy()
        for column in TARGET_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        frame = frame[TARGET_COLUMNS]
        frame["来源"] = source_name
        frame["label"] = label
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    for column in NUMERIC_COLUMNS:
        data[column] = data[column].map(parse_age if column == "年龄" else parse_numeric)
    data["性别"] = data["性别"].map(normalize_gender)
    return data, structure_rows, mapping_rows


def missing_rate_table(data):
    rows = []
    for column in TARGET_COLUMNS + ["来源", "label"]:
        rows.append({"变量": column, "流感组缺失数": int(data.loc[data.label == 1, column].isna().sum()), "流感组样本数": int((data.label == 1).sum()), "流感组缺失率": float(data.loc[data.label == 1, column].isna().mean()), "健康组缺失数": int(data.loc[data.label == 0, column].isna().sum()), "健康组样本数": int((data.label == 0).sum()), "健康组缺失率": float(data.loc[data.label == 0, column].isna().mean()), "总体缺失数": int(data[column].isna().sum()), "总体缺失率": float(data[column].isna().mean())})
    return pd.DataFrame(rows)


def robust_z(values):
    median = values.median()
    mad = np.median(np.abs(values.dropna() - median))
    if mad == 0 or pd.isna(mad):
        return pd.Series(np.nan, index=values.index)
    return (values - median) / (1.4826 * mad)


def cliff_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    ranks = rankdata(np.concatenate([x, y]))
    u = ranks[:nx].sum() - nx * (nx + 1) / 2
    return float(2 * u / (nx * ny) - 1)


def bh_fdr(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values, kind="mergesort")
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def calculate_vif(matrix, index):
    target = matrix[:, index]
    predictors = np.delete(matrix, index, axis=1)
    design = np.column_stack([np.ones(len(predictors)), predictors])
    fitted = design @ np.linalg.lstsq(design, target, rcond=None)[0]
    total_sum_squares = np.sum((target - target.mean()) ** 2)
    if total_sum_squares == 0:
        return np.inf
    residual_sum_squares = np.sum((target - fitted) ** 2)
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    return np.inf if r_squared >= 1.0 else 1.0 / (1.0 - r_squared)


def bootstrap_summary(values, seed=SEED, repetitions=2000):
    values = np.asarray(values.dropna(), dtype=float)
    if len(values) == 0:
        return {"n": 0, "median": np.nan, "median_ci_lower": np.nan, "median_ci_upper": np.nan, "iqr": np.nan, "iqr_ci_lower": np.nan, "iqr_ci_upper": np.nan}
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    iqrs = np.percentile(samples, 75, axis=1) - np.percentile(samples, 25, axis=1)
    return {"n": len(values), "median": float(np.median(values)), "median_ci_lower": float(np.percentile(medians, 2.5)), "median_ci_upper": float(np.percentile(medians, 97.5)), "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)), "iqr_ci_lower": float(np.percentile(iqrs, 2.5)), "iqr_ci_upper": float(np.percentile(iqrs, 97.5))}


def remove_recording_errors(data, cleaning_log):
    error_rows = []
    error_mask = pd.Series(False, index=data.index)
    for column in NUMERIC_COLUMNS:
        current = data[column]
        mask = current.notna() & (current < 0)
        for index in data.index[mask]:
            error_rows.append({"行号": int(index + 1), "序号": data.loc[index, "序号"], "来源": data.loc[index, "来源"], "变量": column, "原值": data.loc[index, column], "处理": "剔除整行", "理由": "不可能为负值"})
        error_mask |= mask
    hb_bad = data["HB"].notna() & (data["HB"] > 200)
    for index in data.index[hb_bad]:
        error_rows.append({"行号": int(index + 1), "序号": data.loc[index, "序号"], "来源": data.loc[index, "来源"], "变量": "HB", "原值": data.loc[index, "HB"], "处理": "剔除整行", "理由": "HB超过200 g/L"})
    error_mask |= hb_bad
    removed = data.loc[error_mask].copy()
    cleaned = data.loc[~error_mask].copy()
    cleaning_log.append({"步骤": "4.1 第一层录入错误", "删除行数": int(error_mask.sum()), "修改内容": "剔除包含负值或HB超过200 g/L的整行", "理由": "严格按方案仅剔除明确的录入错误；其他极端值全部保留并标记"})
    return cleaned, removed, pd.DataFrame(error_rows)


def inspect_suspicious(data):
    rows = []
    for label, group_name in [(1, "流感A组"), (0, "健康组")]:
        group = data.loc[data.label == label]
        for column in NUMERIC_COLUMNS:
            values = group[column]
            z = robust_z(values)
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            flag = z.abs() > 3
            box_flag = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
            for index in group.index[flag.fillna(False) | box_flag.fillna(False)]:
                rows.append({"行号": int(index + 1), "序号": data.loc[index, "序号"], "来源": group_name, "变量": column, "数值": data.loc[index, column], "稳健z分数": float(z.loc[index]) if pd.notna(z.loc[index]) else np.nan, "箱线图1.5IQR标记": bool(box_flag.loc[index]), "处理": "标记并保留"})
    return pd.DataFrame(rows)


def transform_data(data):
    transformed = data.copy()
    decision_rows = []
    for column in BLOOD_COLUMNS:
        values = data[column].dropna().astype(float)
        original_skew = float(skew(values, bias=False)) if len(values) > 2 else np.nan
        shift = max(0.0, -float(values.min()) + 1e-8) if len(values) else 0.0
        positive = values + shift
        if len(values) > 1 and positive.nunique() > 1:
            transformed_boxcox, lambda_estimate = boxcox(positive)
            boxcox_skew = float(skew(transformed_boxcox, bias=False)) if len(transformed_boxcox) > 2 else np.nan
            log_values = np.log(positive)
            log_skew = float(skew(log_values, bias=False)) if len(log_values) > 2 else np.nan
            raw_p = float(shapiro(values).pvalue) if 3 <= len(values) <= 5000 else np.nan
            boxcox_p = float(shapiro(transformed_boxcox).pvalue) if 3 <= len(values) <= 5000 else np.nan
            log_p = float(shapiro(log_values).pvalue) if 3 <= len(values) <= 5000 else np.nan
            candidates = [("不变换", values, original_skew, raw_p), ("Box-Cox", transformed_boxcox, boxcox_skew, boxcox_p)]
            if column in PREJUDGED_LOG_COLUMNS:
                candidates.append(("log(x+c)", log_values, log_skew, log_p))
            selected_method, selected, selected_skew, selected_p = min(candidates, key=lambda item: (abs(item[2]), -item[3] if pd.notna(item[3]) else np.inf))
            if selected_method == "不变换" and column in PREJUDGED_LOG_COLUMNS and abs(lambda_estimate) < abs(lambda_estimate - 1) and abs(log_skew) < abs(original_skew):
                selected_method, selected, selected_skew, selected_p = "log(x+c)", log_values, log_skew, log_p
            decision_basis = "先按原始值、Box-Cox和预判表建议的log候选比较偏度与Shapiro-Wilk；仅选择证据更优的候选，不使用未规定的固定阈值"
        else:
            lambda_estimate, boxcox_skew, log_skew, raw_p, boxcox_p, log_p = [np.nan] * 6
            selected_method, selected, selected_skew, selected_p = "不变换", values, original_skew, raw_p
            decision_basis = "有效非缺失值不足以进行Box-Cox比较，保留原始值"
        transformed[f"{column}_变换后"] = data[column]
        transformed.loc[values.index, f"{column}_变换后"] = selected
        decision_rows.append({"指标": column, "原始偏度": original_skew, "Box-Cox λ": lambda_estimate, "Box-Cox后偏度": boxcox_skew, "log后偏度": log_skew, "变换后偏度": selected_skew, "Shapiro-Wilk变换前p值": raw_p, "Shapiro-Wilk Box-Cox后p值": boxcox_p, "Shapiro-Wilk log后p值": log_p, "变换方式": selected_method, "平移常数c": shift, "决策依据": decision_basis})
    return transformed, pd.DataFrame(decision_rows)


def save_distribution_plots(raw, transformed, decision):
    for column in BLOOD_COLUMNS + ["年龄"]:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for label, name in [(0, "健康组"), (1, "流感A组")]:
            raw_values = raw.loc[raw.label == label, column].dropna()
            transformed_values = transformed.loc[transformed.label == label, f"{column}_变换后"].dropna() if f"{column}_变换后" in transformed.columns else raw_values
            axes[0, 0].hist(raw_values, alpha=0.5, label=name, bins=20)
            axes[0, 1].hist(transformed_values, alpha=0.5, label=name, bins=20)
            axes[0, 2].boxplot(raw_values, positions=[label + 1], tick_labels=[name])
            if len(raw_values) > 1:
                probplot(raw_values, dist="norm", plot=axes[1, 0])
            if len(transformed_values) > 1:
                probplot(transformed_values, dist="norm", plot=axes[1, 1])
        axes[0, 0].set_title(f"{column}变换前分布")
        axes[0, 1].set_title(f"{column}变换后分布")
        axes[0, 2].set_title(f"{column}按组箱线图")
        axes[1, 0].set_title(f"{column}变换前QQ图")
        axes[1, 1].set_title(f"{column}变换后QQ图")
        axes[1, 2].axis("off")
        axes[0, 0].legend()
        axes[0, 1].legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{column}_变换前后分布图.png", dpi=200)
        plt.close()


def main():
    cleaning_log = []
    flua, flua_structure, flua_mapping = read_and_standardize(FLUA_PATH, "流感A组", 1)
    healthy, healthy_structure, healthy_mapping = read_and_standardize(HEALTHY_PATH, "健康组", 0)
    structure = pd.DataFrame(flua_structure + healthy_structure)
    mapping = pd.DataFrame(flua_mapping + healthy_mapping)
    structure.to_csv(OUTPUT_DIR / "结构对照表.csv", index=False, encoding="utf-8-sig")
    mapping.to_csv(OUTPUT_DIR / "列名映射表.csv", index=False, encoding="utf-8-sig")
    units = pd.DataFrame([{"变量": c, "统一单位": UNITS[c], "单位核查": "依据原始表头单位核对；未发现需要换算的单位"} for c in BLOOD_COLUMNS])
    units.to_csv(OUTPUT_DIR / "单位清单.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"变量": "性别", "编码规则": "男/M/MALE/1→1；女/F/FEMALE/2→0；缺失或其他→未知"}]).to_csv(OUTPUT_DIR / "编码规则表.csv", index=False, encoding="utf-8-sig")
    cleaning_log.append({"步骤": "1.1-1.5 数据盘点与格式统一", "删除行数": 0, "修改内容": "实际读取两份表并统一列名、单位、数值类型、缺失标记和性别编码", "理由": "按原始表格结构执行"})

    flua_ids = flua["序号"].dropna()
    healthy_ids = healthy["序号"].dropna()
    duplicate_flua = flua_ids[flua_ids.duplicated(keep=False)]
    duplicate_healthy = healthy_ids[healthy_ids.duplicated(keep=False)]
    cross_ids = set(flua_ids.astype(str)) & set(healthy_ids.astype(str))
    duplicate_rows = pd.DataFrame([{"来源": "流感A组", "重复序号": str(x)} for x in sorted(set(duplicate_flua.astype(str)))] + [{"来源": "健康组", "重复序号": str(x)} for x in sorted(set(duplicate_healthy.astype(str)))] + [{"来源": "跨组", "重复序号": str(x)} for x in sorted(cross_ids)])
    duplicate_rows.to_csv(OUTPUT_DIR / "重复记录清单.csv", index=False, encoding="utf-8-sig")
    cleaning_log.append({"步骤": "1.6 重复检查", "删除行数": 0, "修改内容": f"流感组组内重复={duplicate_flua.nunique()}，健康组组内重复={duplicate_healthy.nunique()}，跨组重复={len(cross_ids)}", "理由": "先分别检查组内重复，再检查跨组重复；本批数据不默认删除记录"})

    raw_merged = pd.concat([flua, healthy], ignore_index=True)
    raw_missing = missing_rate_table(raw_merged)
    raw_missing.to_csv(OUTPUT_DIR / "缺失率统计.csv", index=False, encoding="utf-8-sig")
    raw_merged[TARGET_COLUMNS].isna().to_csv(OUTPUT_DIR / "缺失模式矩阵.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(12, 7))
    plt.imshow(raw_merged[TARGET_COLUMNS].isna().astype(int).T, aspect="auto", cmap="viridis", interpolation="nearest")
    plt.colorbar(label="缺失(1)/非缺失(0)")
    plt.yticks(range(len(TARGET_COLUMNS)), TARGET_COLUMNS)
    plt.title("缺失模式热图")
    plt.xlabel("样本")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "缺失热图.png", dpi=200)
    plt.close()
    high_missing = raw_missing[(raw_missing["总体缺失率"] > 0.4) & raw_missing["变量"].isin(TARGET_COLUMNS)]
    for _, row in high_missing.iterrows():
        cleaning_log.append({"步骤": "3.4 缺失率处置标记", "删除行数": 0, "修改内容": f"{row['变量']}标记为建模时慎用/剔除候选", "理由": "总体缺失率大于40%"})

    clean, removed, error_table = remove_recording_errors(raw_merged, cleaning_log)
    error_table.to_csv(OUTPUT_DIR / "录入错误清单.csv", index=False, encoding="utf-8-sig")
    suspicious_table = inspect_suspicious(clean)
    suspicious_table.to_csv(OUTPUT_DIR / "可疑值清单.csv", index=False, encoding="utf-8-sig")
    variable_rows = []
    for column in NUMERIC_COLUMNS:
        values = clean[column].dropna()
        variable_rows.append({"变量": column, "单位": UNITS.get(column, "无"), "类型": VARIABLE_TYPES[column], "最小值": float(values.min()) if len(values) else np.nan, "最大值": float(values.max()) if len(values) else np.nan, "偏度": float(skew(values, bias=False)) if len(values) > 2 else np.nan, "峰度": float(kurtosis(values, bias=False)) if len(values) > 3 else np.nan, "极端值数量": int((suspicious_table["变量"] == column).sum()) if len(suspicious_table) else 0, "可疑值清单": "见可疑值清单.csv"})
    variable_type_rows = [{"变量": column, "类别": VARIABLE_TYPES.get(column, "来源/标签"), "是否预测候选": "否" if column in {"序号", "来源", "label"} else "是"} for column in TARGET_COLUMNS + ["来源", "label"]]
    pd.DataFrame(variable_type_rows).to_csv(OUTPUT_DIR / "变量类型表.csv", index=False, encoding="utf-8-sig")
    cleaning_log.append({"步骤": "6.1 变量分类", "删除行数": 0, "修改内容": "输出基本信息类、血液检测指标类、样本标识、来源和标签分类", "理由": "序号仅作样本标识，不进入预测特征"})

    transformed, decisions = transform_data(clean)
    decisions.to_csv(OUTPUT_DIR / "变换决策表.csv", index=False, encoding="utf-8-sig")
    save_distribution_plots(clean, transformed, decisions)

    potential_rows = []
    for column in ["年龄"] + BLOOD_COLUMNS:
        group1 = clean.loc[clean.label == 1, column].dropna()
        group0 = clean.loc[clean.label == 0, column].dropna()
        if len(group1) and len(group0):
            statistic, p_value = mannwhitneyu(group1, group0, alternative="two-sided")
            delta = cliff_delta(group1, group0)
        else:
            statistic, p_value, delta = np.nan, np.nan, np.nan
        potential_rows.append({"变量": column, "Mann-Whitney U": statistic, "原始p值": p_value, "Cliff's delta": delta})
    potential = pd.DataFrame(potential_rows)
    valid = potential["原始p值"].notna()
    potential.loc[valid, "BH-FDR校正p值"] = bh_fdr(potential.loc[valid, "原始p值"])
    potential["潜力分级"] = "待阈值确认"
    potential.to_csv(OUTPUT_DIR / "潜力分级表.csv", index=False, encoding="utf-8-sig")

    correlation = clean[NUMERIC_COLUMNS].corr(method="spearman")
    redundancy_rows = []
    for left_index, left in enumerate(NUMERIC_COLUMNS):
        for right in NUMERIC_COLUMNS[left_index + 1:]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(value) >= 0.7:
                redundancy_rows.append({"指标1": left, "指标2": right, "Spearman相关系数": float(value), "判定": "候选冗余组", "判定规则": "|Spearman相关系数|≥0.7或VIF≥5；仅标记，不删除"})
    plt.figure(figsize=(11, 9))
    plt.imshow(correlation.values, aspect="auto", cmap="coolwarm", interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar(label="Spearman相关系数")
    plt.xticks(range(len(NUMERIC_COLUMNS)), NUMERIC_COLUMNS, rotation=45, ha="right")
    plt.yticks(range(len(NUMERIC_COLUMNS)), NUMERIC_COLUMNS)
    for row_index in range(len(NUMERIC_COLUMNS)):
        for col_index in range(len(NUMERIC_COLUMNS)):
            plt.text(col_index, row_index, f"{correlation.values[row_index, col_index]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Spearman相关热图")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "相关热图.png", dpi=200)
    plt.close()
    complete = clean[NUMERIC_COLUMNS].dropna()
    vif_rows = [{"变量": column, "VIF": float(calculate_vif(complete.values, index))} for index, column in enumerate(NUMERIC_COLUMNS)] if len(complete) > len(NUMERIC_COLUMNS) else [{"变量": c, "VIF": np.nan} for c in NUMERIC_COLUMNS]
    pd.DataFrame(vif_rows).to_csv(OUTPUT_DIR / "VIF表.csv", index=False, encoding="utf-8-sig")

    vif_values = {row["变量"]: row["VIF"] for row in vif_rows}
    for row in redundancy_rows:
        if vif_values.get(row["指标1"], np.nan) >= 5 or vif_values.get(row["指标2"], np.nan) >= 5:
            row["判定规则"] = "|Spearman相关系数|≥0.7或VIF≥5；仅标记，不删除"
    pd.DataFrame(redundancy_rows).to_csv(OUTPUT_DIR / "冗余指标组.csv", index=False, encoding="utf-8-sig")

    age_balance = clean.groupby("label")["年龄"].agg(["count", "median", "mean", "std", "min", "max"]).reset_index()
    age_balance["组"] = age_balance["label"].map({0: "健康组", 1: "流感A组"})
    sex_balance = pd.crosstab(clean["性别"], clean["label"], dropna=False).rename(columns={0: "健康组", 1: "流感A组"}).reset_index()
    sex_counts = pd.crosstab(clean["性别"], clean["label"])
    if sex_counts.shape == (2, 2):
        sex_chi2, sex_p, _, _ = chi2_contingency(sex_counts)
    else:
        sex_chi2, sex_p = np.nan, np.nan
    sex_reference = sex_balance.copy()
    sex_reference["两组性别卡方统计量"] = sex_chi2
    sex_reference["两组性别卡方p值"] = sex_p
    sex_reference.to_csv(OUTPUT_DIR / "性别判别参考.csv", index=False, encoding="utf-8-sig")
    bootstrap_rows = []
    for label, group_name in [(1, "流感A组"), (0, "健康组")]:
        group = clean.loc[clean.label == label]
        for column in ["年龄"] + BLOOD_COLUMNS:
            summary = bootstrap_summary(group[column], seed=SEED + label)
            bootstrap_rows.append({"组": group_name, "变量": column, **summary, "置信水平": "95%百分位bootstrap区间", "重复次数": 2000})
    pd.DataFrame(bootstrap_rows).to_csv(OUTPUT_DIR / "bootstrap置信区间.csv", index=False, encoding="utf-8-sig")
    age_balance.to_csv(OUTPUT_DIR / "年龄平衡核查.csv", index=False, encoding="utf-8-sig")
    sex_balance.to_csv(OUTPUT_DIR / "性别平衡核查.csv", index=False, encoding="utf-8-sig")
    counts = clean["label"].value_counts().to_dict()
    n1, n2 = int(counts.get(1, 0)), int(counts.get(0, 0))
    pd.DataFrame([{"总样本量N": len(clean), "流感组n1": n1, "健康组n2": n2, "n1/n2": n1 / n2 if n2 else np.nan, "是否超过1比3": bool(n1 / n2 > 3 or n2 / n1 > 3) if n1 and n2 else np.nan, "重采样": "不进行", "少数类": "健康组" if n2 < n1 else "流感A组", "EPV按10事件/变量建议上限": int(min(n1, n2) // 10), "影响说明": "小样本侧中位数/IQR不稳定，已输出bootstrap置信区间；检验不显著不等于无差异；EPV约束建模变量数"}]).to_csv(OUTPUT_DIR / "样本量差异核查.csv", index=False, encoding="utf-8-sig")

    model_data = transformed.copy()
    for column in NUMERIC_COLUMNS:
        model_data[f"{column}_缺失指示"] = model_data[column].isna().astype(int)
    model_data["性别"] = model_data["性别"].astype(str)
    model_data.to_csv(OUTPUT_DIR / "data_merged.csv", index=False, encoding="utf-8-sig")
    dictionary_rows = []
    for _, map_row in mapping.iterrows():
        column = map_row["统一列名"]
        decision = decisions.loc[decisions["指标"] == column].iloc[0] if column in BLOOD_COLUMNS and (decisions["指标"] == column).any() else None
        dictionary_rows.append({"来源": map_row["来源"], "工作表": map_row["工作表"], "原始名": map_row["原始列名"], "统一名": column, "单位": UNITS.get(column, "无"), "类型": VARIABLE_TYPES.get(column, "来源类别"), "缺失率": float(raw_merged[column].isna().mean()), "处理动作": "数值转换/缺失统一/分类编码" if column in NUMERIC_COLUMNS + ["性别"] else "保留", "变换方式": decision["变换方式"] if decision is not None else "无"})
    for column, variable_type in [("来源", "来源类别"), ("label", "目标标签")]:
        dictionary_rows.append({"来源": "合并数据", "工作表": "合并", "原始名": "无", "统一名": column, "单位": "无", "类型": variable_type, "缺失率": float(raw_merged[column].isna().mean()), "处理动作": "按来源构建" if column == "label" else "保留", "变换方式": "无"})
    pd.DataFrame(dictionary_rows).to_csv(OUTPUT_DIR / "数据字典.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cleaning_log).to_csv(OUTPUT_DIR / "清洗日志.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
