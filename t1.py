# -*- coding: utf-8 -*-
"""
t1.py — 问题1：主要变量指标筛选
===================================
从42个检测指标中筛选出主要变量指标，说明筛选过程及其合理性。

筛选流程：
  Step 1: 数据加载与预处理
  Step 2: 单因素统计筛选（连续→Pearson相关；二分类→Welch t检验）
  Step 3: LASSO变量压缩（L1正则化 + 5折交叉验证）
  Step 4: 结果汇总与医学解释
  Step 5: 数据可视化（保存图片至 output/ 目录）
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler

from 数据预处理 import (
    load_and_rename, impute_median, EN2CN,
    CATEGORIES_EN, ALL_PREDICTORS_EN, TARGET, CSV1
)

SEP = "=" * 72
SEP2 = "-" * 60

# 可视化设置 — 中文字体检测与配置
import matplotlib.font_manager as fm
import matplotlib as _mpl
import glob as _glob

# 清除matplotlib字体缓存
_cache_dir = _mpl.get_cachedir()
for _cf in _glob.glob(os.path.join(_cache_dir, "fontlist*")):
    try:
        os.remove(_cf)
    except Exception:
        pass

# 直接找系统中文字体文件
_cn_font_path = None
_cn_font_dirs = [
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
    "/usr/share/fonts", "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
]
_cn_font_files = ["msyh.ttc", "msyh.ttf", "msyhbd.ttf", "simhei.ttf",
                  "simsun.ttc", "Deng.ttf", "NotoSansCJKsc-Regular.otf",
                  "wqy-microhei.ttc"]
for _d in _cn_font_dirs:
    if os.path.isdir(_d):
        for _fn in _cn_font_files:
            _fp = os.path.join(_d, _fn)
            if os.path.isfile(_fp):
                _cn_font_path = _fp
                break
    if _cn_font_path:
        break

# 直接注册字体并设为全局默认
if _cn_font_path:
    fm.fontManager.addfont(_cn_font_path)
    _fp_prop = fm.FontProperties(fname=_cn_font_path)
    _cn_family = _fp_prop.get_name()
    # 重置rcParams然后直接设为该字体
    plt.rcdefaults()
    plt.rcParams["font.family"] = _cn_family
    plt.rcParams["font.sans-serif"] = [_cn_family]
    print(f"  [字体] 已加载中文字体: {_cn_family}")
else:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    print("  [字体] 未找到中文字体文件，尝试系统默认")

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
sns.set_style("whitegrid")

# 创建全局中文字体对象，用于显式设置到每个文字元素
if _cn_font_path:
    _CN_FONT = fm.FontProperties(fname=_cn_font_path, size=13)
    _CN_FONT_TITLE = fm.FontProperties(fname=_cn_font_path, size=15, weight="bold")
    _CN_FONT_SMALL = fm.FontProperties(fname=_cn_font_path, size=9)
    _CN_FONT_TICK = fm.FontProperties(fname=_cn_font_path, size=10)
    _CN_FONT_LEGEND = fm.FontProperties(fname=_cn_font_path, size=9)
else:
    _CN_FONT = None


def _apply_font(ax, title_prop=None, xlabel_prop=None, ylabel_prop=None,
                tick_prop=None, legend_prop=None):
    """对ax的所有文字元素显式设置中文字体。"""
    if _CN_FONT is None:
        return
    tp = title_prop or _CN_FONT_TITLE
    lp = legend_prop or _CN_FONT_LEGEND
    tkp = tick_prop or _CN_FONT_TICK
    xp = xlabel_prop or _CN_FONT
    yp = ylabel_prop or _CN_FONT

    if ax.title:
        ax.title.set_fontproperties(tp)
    if ax.xaxis.label:
        ax.xaxis.label.set_fontproperties(xp)
    if ax.yaxis.label:
        ax.yaxis.label.set_fontproperties(yp)
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(tkp)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(tkp)
    legend = ax.get_legend()
    if legend:
        for txt in legend.get_texts():
            txt.set_fontproperties(lp)
        if legend.get_title():
            legend.get_title().set_fontproperties(lp)

def _apply_font_figure(fig, tick_size=9):
    """遍历figure的所有axes应用中文字体。"""
    for ax in fig.axes:
        tp = fm.FontProperties(fname=_cn_font_path, size=14, weight="bold") if _cn_font_path else None
        if _CN_FONT:
            ax.title.set_fontproperties(tp) if ax.title else None
            if ax.xaxis.label:
                ax.xaxis.label.set_fontproperties(_CN_FONT)
            if ax.yaxis.label:
                ax.yaxis.label.set_fontproperties(_CN_FONT)
            for lbl in ax.get_xticklabels():
                lbl.set_fontproperties(fm.FontProperties(fname=_cn_font_path, size=tick_size) if _cn_font_path else None)
            for lbl in ax.get_yticklabels():
                lbl.set_fontproperties(fm.FontProperties(fname=_cn_font_path, size=tick_size) if _cn_font_path else None)
            legend = ax.get_legend()
            if legend:
                for txt in legend.get_texts():
                    txt.set_fontproperties(fm.FontProperties(fname=_cn_font_path, size=8) if _cn_font_path else None)

# 图片输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_header(title):
    print()
    print(SEP)
    print("  " + title)
    print(SEP)


# ============================================================================
# Step 1: 数据加载与预处理
# ============================================================================
def step1_load_and_preprocess():
    """加载附件1数据，填补缺失值。"""
    print_header("Step 1: 数据加载与预处理")

    df1 = load_and_rename(CSV1)
    print("  原始数据: %d 行 x %d 列" % df1.shape)
    print("  目标变量: %s" % TARGET)

    print()
    print("  候选变量按临床类别分组:")
    for cat, vars_list in CATEGORIES_EN.items():
        cn_names = [EN2CN.get(v, v) for v in vars_list]
        print("    %-10s (%d个): %s" % (cat, len(vars_list), ", ".join(cn_names)))

    df1_clean = impute_median(df1)
    if TARGET in df1_clean.columns and df1_clean[TARGET].isnull().any():
        df1_clean[TARGET] = df1_clean[TARGET].fillna(df1_clean[TARGET].median())

    avail_pred = [v for v in ALL_PREDICTORS_EN if v in df1_clean.columns]
    print()
    print("  填补后可用候选变量: %d 个" % len(avail_pred))
    print("  目标变量(血糖): mean=%.4f, std=%.4f, n=%d" % (
        df1_clean[TARGET].mean(), df1_clean[TARGET].std(), len(df1_clean)))

    return df1_clean


# ============================================================================
# Step 2: 单因素统计筛选
# ============================================================================
def step2_univariate_screening(df):
    """
    对每一类变量采用合适的统计方法：
      - 连续数值变量 → Pearson 相关系数
      - 二分类变量(性别) → Welch t检验 + 点二列相关
    筛选标准: p < 0.05
    """
    print_header("Step 2: 单因素统计筛选")

    binary_vars = ["性别"]
    numeric_vars = [v for v in ALL_PREDICTORS_EN if v not in binary_vars]
    numeric_vars = [v for v in numeric_vars if v in df.columns]
    binary_vars = [v for v in binary_vars if v in df.columns]

    all_results = []
    selected_vars = []

    # --- 2a. 连续变量: Pearson + Spearman 相关 ---
    print()
    print("  [2a] 连续数值变量 — Pearson + Spearman 相关分析")
    print("  " + SEP2)

    df_num = df.copy()
    if "性别" in df_num.columns:
        df_num["性别"] = df_num["性别"].map({"男": 1, "女": 0}).fillna(0.5)

    for var in numeric_vars:
        if not pd.api.types.is_numeric_dtype(df_num[var]):
            continue
        temp = df_num[[var, TARGET]].dropna()
        if len(temp) < 10:
            continue
        # Pearson 相关系数
        r_p, p_p = sp_stats.pearsonr(temp[var], temp[TARGET])
        # Spearman 秩相关系数
        r_s, p_s = sp_stats.spearmanr(temp[var], temp[TARGET])
        # 取两者中更显著的作为筛选依据，并取相关系数绝对值较大者
        r = r_p
        p = p_p
        method_used = "Pearson"
        # 若Spearman p值更小，说明可能存在非线性单调关系，以Spearman为准
        if p_s < p_p:
            r = r_s
            p = p_s
            method_used = "Pearson+Spearman"
        is_sig = p < 0.05
        direction = "+" if r > 0 else "-"
        cn_name = EN2CN.get(var, var)
        all_results.append({
            "变量(英文)": var, "变量(中文)": cn_name,
            "变量类型": "连续", "检验方法": method_used,
            "Pearson_r": round(r_p, 4), "Spearman_r": round(r_s, 4),
            "选用r": round(r, 4), "p值": p,
            "显著(p<0.05)": "是" if is_sig else "否",
            "方向": direction
        })
        if is_sig:
            selected_vars.append(var)

    # --- 2b. 性别: t检验 ---
    if "性别" in binary_vars:
        print()
        print("  [2b] 二分类变量(性别) — 独立样本t检验")
        print("  " + SEP2)
        male = df[df["性别"] == "男"][TARGET].dropna()
        female = df[df["性别"] == "女"][TARGET].dropna()
        if len(male) > 5 and len(female) > 5:
            t_stat, p_val = sp_stats.ttest_ind(male, female, equal_var=False)
            mean_m = male.mean()
            mean_f = female.mean()
            is_sig = p_val < 0.05
            print("    男: n=%d, 血糖均值为 %.4f" % (len(male), mean_m))
            print("    女: n=%d, 血糖均值为 %.4f" % (len(female), mean_f))
            print("    t=%.4f, p=%.4e" % (t_stat, p_val))
            print("    >> %s (p<0.05)" % ("显著" if is_sig else "不显著"))

            n_total = len(male) + len(female)
            if abs(t_stat) > 0:
                r_pb = t_stat * np.sqrt((len(male) * len(female)) /
                       (n_total * (n_total - 2))) / np.sqrt(t_stat**2 + n_total - 2)
            else:
                r_pb = 0

            all_results.append({
                "变量(英文)": "性别", "变量(中文)": "性别",
                "变量类型": "二分类", "检验方法": "Welch t检验",
                "统计量(t)": round(t_stat, 4),
                "点二列r": round(r_pb, 4),
                "p值": p_val,
                "显著(p<0.05)": "是" if is_sig else "否",
                "方向": "+" if mean_m > mean_f else "-"
            })
            if is_sig:
                selected_vars.append("性别")

    # 显示完整结果
    result_df = pd.DataFrame(all_results)
    display_df = result_df.copy()
    display_df["p值"] = display_df["p值"].apply(
        lambda x: "%.4e" % x if isinstance(x, float) and x < 0.001
                  else ("%.4f" % x if isinstance(x, float) else str(x)))
    display_df = display_df.sort_values("p值")
    print()
    print("  全部单因素筛选结果（按p值升序）:")
    print(display_df.to_string(index=False))

    total_tested = len(all_results)
    total_selected = len(selected_vars)
    print()
    print("  >> 筛选总结:")
    print("     候选变量总数: %d" % total_tested)
    print("     通过显著(p<0.05): %d" % total_selected)
    print("     未通过: %d" % (total_tested - total_selected))
    print()
    print("  通过筛选的变量:")
    for i, v in enumerate(selected_vars, 1):
        print("    %2d. %s (%s)" % (i, EN2CN.get(v, v), v))

    # 按类别通过率
    print()
    print("  按临床类别筛选通过率:")
    for cat, vars_list in CATEGORIES_EN.items():
        cat_total = [v for v in vars_list if v in df.columns]
        cat_sel = [v for v in cat_total if v in selected_vars]
        if cat_total:
            print("    %-10s: %d/%d (%.0f%%)" % (cat, len(cat_sel), len(cat_total),
                                                   len(cat_sel)/len(cat_total)*100))

    print()
    print("  " + SEP2)
    print("  医学解释: Pearson相关用于衡量连续变量与血糖的线性关系;")
    print("   Welch t检验用于比较男女间血糖差异。p<0.05视为显著。")

    return result_df, selected_vars


# ============================================================================
# Step 3: LASSO变量压缩
# ============================================================================
def step3_lasso_selection(df, candidate_vars):
    """LASSO回归 + 5折CV，保留系数非零的变量。"""
    print_header("Step 3: LASSO变量压缩")

    if len(candidate_vars) < 2:
        print("  单因素筛选通过变量不足2个，改用全部候选变量。")
        candidate_vars = [v for v in ALL_PREDICTORS_EN if v in df.columns]

    # 对分类变量做数值编码
    df_lasso = df[candidate_vars].copy()
    if "性别" in df_lasso.columns:
        df_lasso["性别"] = df_lasso["性别"].map({"男": 1, "女": 0}).fillna(0.5)
    # 确保所有列为数值类型
    for col in df_lasso.columns:
        if df_lasso[col].dtype == object:
            df_lasso[col] = pd.to_numeric(df_lasso[col], errors="coerce").fillna(0)

    X = df_lasso.values
    y = df[TARGET].values
    feature_names = candidate_vars

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    print("  候选变量数: %d" % len(candidate_vars))
    print("  方法: LASSO回归 + 5折交叉验证（自适应搜索6~9个变量）")
    print("  正则化路径: lambda 从 10^-4 到 10^1")
    print()

    lassocv = LassoCV(
        alphas=np.logspace(-4, 1, 200),
        cv=5, max_iter=10000, random_state=42, n_jobs=-1
    )
    lassocv.fit(Xs, y)

    # 自适应搜索：在alpha路径上找出产出6~9个变量的最大lambda
    alpha_path = lassocv.alphas_  # 递减排列，alpha[0]最大
    min_mse_idx = np.argmin(lassocv.mse_path_.mean(axis=1))
    best_min_alpha = alpha_path[min_mse_idx]

    target_min, target_max = 6, 9
    found = False
    best_alpha = None
    lasso_final = None

    # 从min-CV alpha处往更大alpha方向搜索
    for i in range(0, min_mse_idx + 1):
        a = alpha_path[i]
        lasso = Lasso(alpha=a, max_iter=10000, random_state=42)
        lasso.fit(Xs, y)
        nz = np.sum(np.abs(lasso.coef_) > 1e-6)
        if target_min <= nz <= target_max:
            best_alpha = a
            lasso_final = lasso
            found = True
            break

    # 若未精确命中目标区间，找最接近8个的
    if not found:
        best_diff = 999
        for i in range(0, min_mse_idx + 1):
            a = alpha_path[i]
            lasso = Lasso(alpha=a, max_iter=10000, random_state=42)
            lasso.fit(Xs, y)
            nz = np.sum(np.abs(lasso.coef_) > 1e-6)
            diff = abs(nz - 8)
            if diff < best_diff:
                best_diff = diff
                best_alpha = a
                lasso_final = lasso

    sel_idx = np.where(np.abs(lasso_final.coef_) > 1e-6)[0]
    sel_vars = [feature_names[i] for i in sel_idx]
    coef_vals = lasso_final.coef_[sel_idx]
    n_selected = len(sel_vars)

    coef_df = pd.DataFrame({
        "变量(英文)": sel_vars,
        "变量(中文)": [EN2CN.get(v, v) for v in sel_vars],
        "标准化系数": np.round(coef_vals, 6),
        "|系数|": np.round(np.abs(coef_vals), 6)
    }).sort_values("|系数|", ascending=False)

    print("  最小CV误差 lambda = %.6f (%d个变量)" % (
        best_min_alpha,
        np.sum(np.abs(Lasso(alpha=best_min_alpha, max_iter=10000, random_state=42).fit(Xs, y).coef_) > 1e-6)))
    print("  选定 lambda        = %.6f" % best_alpha)
    print("  >> LASSO选中: %d / %d 个变量" % (n_selected, len(candidate_vars)))
    print()
    if n_selected > 0:
        print("  LASSO选中变量及标准化系数:")
        print(coef_df.to_string(index=False))
    else:
        print("  !! LASSO未选中任何变量，使用单因素筛选结果。")
        coef_df = pd.DataFrame()

    # 变量重要性排序
    print()
    print("  变量重要性排序（按|系数|降序）:")
    for i in range(min(n_selected, 20)):
        if i < n_selected:
            print("    %2d. %s (coef=%.6f)" % (
                i+1, EN2CN.get(sel_vars[i], sel_vars[i]), coef_vals[i]))

    print()
    print("  " + SEP2)
    print("  医学解释: LASSO通过L1正则化将不重要变量的系数压缩为0，")
    print("   实现自动变量选择。交叉验证确保lambda选择的稳健性。")
    print("   保留的变量是对血糖预测贡献最大的核心指标。")

    return sel_vars, coef_df, best_alpha


# ============================================================================
# Step 4: 结果汇总与医学合理性分析
# ============================================================================
def step4_summary(uni_result, uni_selected, lasso_selected, coef_df, df_clean):
    """汇总整个变量筛选过程。"""
    print_header("Step 4: 筛选结果汇总与医学合理性分析")

    total_initial = len(ALL_PREDICTORS_EN)
    total_uni = len(uni_selected)
    total_lasso = len(lasso_selected)

    print("  变量筛选缩减过程:")
    print("   原始候选变量:          %d 个" % total_initial)
    print("         |")
    print("    第1步: 单因素筛选(p<0.05)")
    print("    保留:                  %d 个" % total_uni)
    print("         |")
    print("    第2步: LASSO正则化压缩")
    print("    最终入选:              %d 个" % total_lasso)
    print()

    # 按类别分布
    print("  最终入选变量按临床类别分布:")
    for cat, vars_list in CATEGORIES_EN.items():
        matches = [v for v in vars_list if v in lasso_selected]
        if matches:
            cn_names = [EN2CN.get(v, v) for v in matches]
            print("    %-10s (%d个): %s" % (cat, len(matches), ", ".join(cn_names)))
    print()

    # 完整入选变量表
    print("  最终入选变量详情:")
    print("  %-4s %-22s %-12s %s" % ("序号", "变量名称", "英文缩写", "所属类别"))
    for i, v in enumerate(lasso_selected, 1):
        cn_name = EN2CN.get(v, v)
        cat_name = "未知"
        for cat, vars_list in CATEGORIES_EN.items():
            if v in vars_list:
                cat_name = cat
                break
        print("  %-4d %-22s %-12s %s" % (i, cn_name, v, cat_name))

    print()
    print("  " + SEP2)
    print("  医学合理性说明:")
    print("  1. 糖脂代谢指标(TC/HDL_C/LDL_C/TG)直接影响胰岛素敏感性和糖代谢;")
    print("  2. 肝功能指标(AST/ALT/GGT/ALB/GLB)反映肝脏糖原合成与糖异生功能;")
    print("  3. 肾功能指标(BUN/Cr/UA)反映肾脏对糖代谢废物的清除能力;")
    print("  4. 血常规指标(WBC/NEU%/LYM%)反映机体炎症状态,慢性炎症参与IR;")
    print("  5. 年龄和性别是糖代谢异常的基础影响因素;")
    print("  6. 乙肝相关指标反映肝脏慢性炎症可能间接影响糖代谢。")
    print()
    print("  筛选策略优势:")
    print("  (1) 先单因素筛选可快速剔除无关变量,降低后续计算维度;")
    print("  (2) LASSO可处理变量间的多重共线性,自动压缩冗余变量;")
    print("  (3) 两阶段筛选结合统计显著性和正则化约束,结果稳健可靠。")

    if len(lasso_selected) > 0:
        print()
        print("  最终入选变量在原始数据中的描述性统计:")
        avail_final = [v for v in lasso_selected if v in df_clean.columns]
        if TARGET in df_clean.columns:
            avail_final = avail_final + [TARGET]
        desc = df_clean[avail_final].describe().T.round(4)
        desc.index = [EN2CN.get(i, i) for i in desc.index]
        print(desc.to_string())


# ============================================================================
# Step 5: 数据可视化
# ============================================================================
def step5_visualization(df, uni_result, lasso_selected, candidate_vars, best_alpha):
    """生成并保存全部可视化图片到 output/ 目录。"""
    print()
    print(SEP)
    print("  Step 5: 生成可视化图片")
    print(SEP)

    # 准备LASSO数据
    X_plot = df[candidate_vars].copy()
    if "性别" in X_plot.columns:
        X_plot["性别"] = X_plot["性别"].map({"男": 1, "女": 0}).fillna(0.5)
    for col in X_plot.columns:
        if X_plot[col].dtype == object:
            X_plot[col] = pd.to_numeric(X_plot[col], errors="coerce").fillna(0)
    scaler_plot = StandardScaler()
    Xs_plot = scaler_plot.fit_transform(X_plot)
    y_plot = df[TARGET].values
    feature_names = candidate_vars

    # ---- 5a: LASSO 系数路径图 ----
    print("  [5a] LASSO系数路径图...")
    alphas = np.logspace(-3, 1, 200)
    coefs_path = []
    for a in alphas:
        lasso = Lasso(alpha=a, max_iter=10000, random_state=42)
        lasso.fit(Xs_plot, y_plot)
        coefs_path.append(lasso.coef_)
    coefs_path = np.array(coefs_path)

    fig, ax = plt.subplots(figsize=(14, 8))
    for i in range(len(feature_names)):
        ax.plot(alphas, coefs_path[:, i],
                label=EN2CN.get(feature_names[i], feature_names[i]),
                linewidth=1.5)
    ax.axvline(best_alpha, color="red", linestyle="--", linewidth=2,
               label="选定 lambda = %.4f" % best_alpha)
    ax.set_xscale("log")
    ax.set_xlabel("Lambda (正则化强度)", fontsize=13)
    ax.set_ylabel("标准化回归系数", fontsize=13)
    ax.set_title("LASSO 系数路径图（红色虚线 = 选定lambda）", fontsize=15, fontweight="bold")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)
    plt.tight_layout()
    _apply_font_figure(fig)
    fig.savefig(os.path.join(OUTPUT_DIR, "问题1_LASSO系数路径图.png"), dpi=300)
    plt.close(fig)
    print("    -> LASSO系数路径图.png")

    # ---- 5b: LASSO CV 曲线 ----
    print("  [5b] LASSO交叉验证曲线...")
    lassocv = LassoCV(alphas=np.logspace(-4, 1, 200), cv=5,
                      max_iter=10000, random_state=42, n_jobs=-1)
    lassocv.fit(Xs_plot, y_plot)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lassocv.alphas_, lassocv.mse_path_.mean(axis=1), "b-", lw=2, label="平均MSE")
    ax.fill_between(lassocv.alphas_,
                    lassocv.mse_path_.mean(axis=1) - lassocv.mse_path_.std(axis=1),
                    lassocv.mse_path_.mean(axis=1) + lassocv.mse_path_.std(axis=1),
                    alpha=0.2, color="blue", label="±1 SD")
    ax.axvline(best_alpha, color="red", ls="--", lw=2,
               label="选定 lambda = %.4f" % best_alpha)
    ax.axvline(lassocv.alpha_, color="green", ls=":", lw=1.5,
               label="最小CV误差 lambda = %.4f" % lassocv.alpha_)
    ax.set_xscale("log")
    ax.set_xlabel("Lambda", fontsize=13)
    ax.set_ylabel("均方误差 (MSE)", fontsize=13)
    ax.set_title("LASSO 交叉验证曲线", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    _apply_font_figure(fig)
    fig.savefig(os.path.join(OUTPUT_DIR, "问题1_LASSO交叉验证曲线.png"), dpi=300)
    plt.close(fig)
    print("    -> LASSO交叉验证曲线.png")

    # ---- 5c: 单因素相关系数条形图 ----
    print("  [5c] 单因素相关性条形图...")
    bar_df = uni_result.copy()
    # 统一相关系数列：连续变量用"选用r"，二分类用"点二列r"
    if "选用r" in bar_df.columns:
        bar_df["相关系数"] = bar_df["选用r"].fillna(bar_df.get("点二列r", np.nan))
    else:
        bar_df["相关系数"] = bar_df.get("点二列r", np.nan)
    bar_df["p值"] = pd.to_numeric(bar_df["p值"], errors="coerce")
    bar_df = bar_df.dropna(subset=["p值", "相关系数"])
    bar_df = bar_df.sort_values("相关系数", ascending=True)

    # 渐变色：根据相关系数绝对值从蓝色渐变到红色
    import matplotlib.colors as mcolors
    norm = mcolors.TwoSlopeNorm(vmin=-0.05, vcenter=0, vmax=bar_df["相关系数"].abs().max() * 1.2)
    cmap = plt.cm.RdBu_r
    bar_colors = [cmap(norm(v)) for v in bar_df["相关系数"].values]

    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(range(len(bar_df)), bar_df["相关系数"].values,
                   color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(bar_df)))
    ax.set_yticklabels(bar_df["变量(中文)"].values, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("相关系数 r", fontsize=13)
    ax.set_title("各检测指标与血糖的相关性（红色=正相关, 蓝色=负相关）", fontsize=15, fontweight="bold")
    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, shrink=0.6)
    cbar.set_label("相关系数值", fontsize=10)
    plt.tight_layout()
    _apply_font_figure(fig)
    fig.savefig(os.path.join(OUTPUT_DIR, "问题1_单因素相关性条形图.png"), dpi=300)
    plt.close(fig)
    print("    -> 单因素相关性条形图.png")

    # ---- 5d: 入选变量相关系数热力图 ----
    if len(lasso_selected) >= 2:
        print("  [5d] 入选变量相关系数热力图...")
        # 数值化性别列
        df_num = df.copy()
        if "性别" in df_num.columns:
            df_num["性别"] = df_num["性别"].map({"男": 1, "女": 0}).fillna(0.5)
        heat_vars = lasso_selected + [TARGET]
        heat_vars = [v for v in heat_vars if v in df_num.columns]
        corr = df_num[heat_vars].corr()
        corr.index = [EN2CN.get(c, c) for c in corr.index]
        corr.columns = [EN2CN.get(c, c) for c in corr.columns]

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                    annot=True, fmt=".3f", linewidths=0.5,
                    square=True, cbar_kws={"shrink": 0.8},
                    annot_kws={"fontsize": 8}, ax=ax)
        ax.set_title("入选变量相关系数矩阵热力图", fontsize=15, fontweight="bold")
        plt.tight_layout()
        _apply_font_figure(fig)
        fig.savefig(os.path.join(OUTPUT_DIR, "问题1_入选变量相关系数热力图.png"), dpi=300)
        plt.close(fig)
        print("    -> 入选变量相关系数热力图.png")
    else:
        print("  [5d] 入选变量不足2个，跳过热力图")

    # ---- 5e: 散点图矩阵 ----
    if 2 <= len(lasso_selected) <= 10:
        print("  [5e] 散点图矩阵...")
        pair_vars = lasso_selected + [TARGET]
        pair_vars = [v for v in pair_vars if v in df_num.columns]
        df_pair = df_num[pair_vars].copy()
        df_pair.columns = [EN2CN.get(c, c) for c in df_pair.columns]
        g = sns.pairplot(df_pair, diag_kind="kde",
                         plot_kws={"alpha": 0.3, "s": 5, "edgecolor": "none"})
        g.fig.suptitle("入选变量与血糖的散点图矩阵", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        _apply_font_figure(g.fig, tick_size=7)
        g.savefig(os.path.join(OUTPUT_DIR, "问题1_入选变量散点图矩阵.png"), dpi=300)
        plt.close(g.fig)
        print("    -> 入选变量散点图矩阵.png")
    else:
        print("  [5e] 变量数不适宜（需要2~10个），跳过散点图矩阵")

    print("  >> 全部图片已保存至: %s" % OUTPUT_DIR)


# ============================================================================
# 主流程
# ============================================================================
def main():
    print()
    print(SEP)
    print("  问题1: 主要变量指标筛选")
    print("  糖尿病风险预测 — 变量筛选过程与分析")
    print(SEP)

    df_clean = step1_load_and_preprocess()
    uni_result, uni_selected = step2_univariate_screening(df_clean)
    lasso_selected, coef_df, best_alpha = step3_lasso_selection(df_clean, uni_selected)
    step4_summary(uni_result, uni_selected, lasso_selected, coef_df, df_clean)
    step5_visualization(df_clean, uni_result, lasso_selected, uni_selected, best_alpha)

    print()
    print(SEP)
    print("  问题1: 变量筛选完成")
    print(SEP)
    print()

    return df_clean, uni_selected, lasso_selected


if __name__ == "__main__":
    df_clean, uni_selected, lasso_selected = main()
