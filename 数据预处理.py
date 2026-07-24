# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
import os

SEP1 = "=" * 70
SEP2 = "-" * 60

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SUBDIR = "2026年武汉理工大学数学建模训练题目1-3"
CSV1 = os.path.join(CURRENT_DIR, DATA_SUBDIR, "附件1：有血糖值的检测数据.csv")
CSV2 = os.path.join(CURRENT_DIR, DATA_SUBDIR, "附件2：无血糖值的检测数据.csv")
TARGET = "血糖"

CN2EN = {
    "id": "id", "体检日期": "体检日期", "性别": "性别", "年龄": "年龄",
    "*r-谷氨酰基转换酶": "GGT", "*丙氨酸氨基转换酶": "ALT",
    "*天门冬氨酸氨基转换酶": "AST", "*总蛋白": "TP", "*球蛋白": "GLB",
    "*碱性磷酸酶": "ALP", "尿素": "BUN", "肌酐": "Cr", "尿酸": "UA",
    "甘油三酯": "TG", "总胆固醇": "TC", "高密度脂蛋白胆固醇": "HDL_C",
    "低密度脂蛋白胆固醇": "LDL_C", "白蛋白": "ALB", "白球比例": "A_G",
    "白细胞计数": "WBC", "红细胞计数": "RBC", "血红蛋白": "HGB",
    "红细胞压积": "HCT", "红细胞平均体积": "MCV", "红细胞平均血红蛋白量": "MCH",
    "红细胞平均血红蛋白浓度": "MCHC", "红细胞体积分布宽度": "RDW",
    "血小板计数": "PLT", "血小板平均体积": "MPV", "血小板体积分布宽度": "PDW",
    "血小板比积": "PCT", "中性粒细胞%": "NEU_pct", "淋巴细胞%": "LYM_pct",
    "单核细胞%": "MON_pct", "嗜酸细胞%": "EOS_pct", "嗜碱细胞%": "BAS_pct",
    "乙肝表面抗原": "HBsAg", "乙肝表面抗体": "HBsAb", "乙肝e抗原": "HBeAg",
    "乙肝e抗体": "HBeAb", "乙肝核心抗体": "HBcAb", "血糖": "血糖",
}
EN2CN = {v: k for k, v in CN2EN.items()}

CATEGORIES_EN = {
    "人口学特征": ["性别", "年龄"],
    "糖脂代谢指标": ["TG", "TC", "HDL_C", "LDL_C"],
    "肝功能指标": ["AST", "ALT", "ALP", "GGT", "TP", "ALB", "GLB", "A_G"],
    "肾功能指标": ["BUN", "Cr", "UA"],
    "血常规指标": ["WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "RDW",
                    "PLT", "MPV", "PDW", "PCT",
                    "NEU_pct", "LYM_pct", "MON_pct", "EOS_pct", "BAS_pct"],
    "感染免疫指标": ["HBsAg", "HBsAb", "HBeAg", "HBeAb", "HBcAb"],
}

ALL_PREDICTORS_EN = []
for v in CATEGORIES_EN.values():
    ALL_PREDICTORS_EN.extend(v)
ALL_PREDICTORS_EN = list(set(ALL_PREDICTORS_EN))


def load_and_rename(filepath):
    df = pd.read_csv(filepath, encoding="gbk")
    df.columns = df.columns.str.strip()
    rename_map = {cn: CN2EN[cn] for cn in df.columns if cn in CN2EN}
    df_renamed = df.rename(columns=rename_map)
    keep_cols = [c for c in df_renamed.columns if c in CN2EN.values()]
    return df_renamed[keep_cols]


def analyze_missing(df, title="数据"):
    miss_count = df.isnull().sum()
    miss_rate = (df.isnull().sum() / len(df) * 100).round(2)
    miss_df = pd.DataFrame({"缺失数": miss_count, "缺失率(%)": miss_rate})
    miss_df = miss_df[miss_df["缺失数"] > 0].sort_values("缺失率(%)", ascending=False)
    print(SEP2)
    print("  [缺失值分析] " + title)
    print(SEP2)
    if len(miss_df) > 0:
        display = miss_df.copy()
        display.index = [EN2CN.get(i, str(i)) for i in display.index]
        print(display.to_string())
        print("  >> 共 %d 个变量存在缺失" % len(miss_df))
    else:
        print("  无缺失值")
    return miss_df


def impute_median(df):
    df_out = df.copy()
    for col in df_out.columns:
        if df_out[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_out[col]):
                df_out[col] = df_out[col].fillna(df_out[col].median())
            else:
                mode_val = df_out[col].mode()
                if len(mode_val) > 0:
                    df_out[col] = df_out[col].fillna(mode_val.iloc[0])
    return df_out


def descriptive_stats(df, title="数据"):
    print()
    print(SEP1)
    print("  [描述性统计] " + title)
    print(SEP1)
    for cat_name, cat_vars in CATEGORIES_EN.items():
        avail = [v for v in cat_vars if v in df.columns]
        if not avail:
            continue
        sub = df[avail]
        desc = sub.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
        desc = desc.round(4)
        print()
        print("--- %s (%d个变量) ---" % (cat_name, len(avail)))
        print(desc.to_string())


def univariate_correlation(df):
    print()
    print(SEP1)
    print("  [单因素相关性] Pearson with " + TARGET)
    print(SEP1)
    results = []
    for var in ALL_PREDICTORS_EN:
        if var not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[var]):
            continue
        temp = df[[var, TARGET]].dropna()
        if len(temp) < 10:
            continue
        r, p = stats.pearsonr(temp[var], temp[TARGET])
        results.append({
            "变量(英文)": var, "变量(中文)": EN2CN.get(var, var),
            "相关系数_r": round(r, 4),
            "p值": "%.4e" % p if p < 0.0001 else "%.4f" % p,
            "显著(p<0.05)": "是" if p < 0.05 else "否",
            "方向": "正相关" if r > 0 else "负相关"
        })
    result_df = pd.DataFrame(results).sort_values("p值")
    selected = [r["变量(英文)"] for _, r in result_df.iterrows()
                if r["显著(p<0.05)"] == "是"]
    print(result_df.to_string(index=False))
    print()
    print("  >> p<0.05显著: %d / %d 个变量" % (len(selected), len(results)))
    return result_df, selected


def lasso_selection(df, candidate_vars):
    print()
    print(SEP1)
    print("  [LASSO回归] 变量筛选")
    print(SEP1)
    df_lasso = df[candidate_vars].copy()
    if "性别" in df_lasso.columns:
        df_lasso["性别"] = df_lasso["性别"].map({"男": 1, "女": 0}).fillna(0.5)
    for col in df_lasso.columns:
        if df_lasso[col].dtype == object:
            df_lasso[col] = pd.to_numeric(df_lasso[col], errors="coerce").fillna(0)
    X = df_lasso.values
    y = df[TARGET].values
    feature_names = candidate_vars
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lassocv = LassoCV(alphas=np.logspace(-4, 1, 200), cv=5,
                      max_iter=10000, random_state=42, n_jobs=-1)
    lassocv.fit(Xs, y)
    alpha_path = lassocv.alphas_
    min_idx = np.argmin(lassocv.mse_path_.mean(axis=1))
    target_min, target_max = 6, 9
    best_a = None
    for i in range(0, min_idx + 1):
        a = alpha_path[i]
        lasso = Lasso(alpha=a, max_iter=10000, random_state=42)
        lasso.fit(Xs, y)
        nz = np.sum(np.abs(lasso.coef_) > 1e-6)
        if target_min <= nz <= target_max:
            best_a = a
            lasso_final = lasso
            break
    if best_a is None:
        best_diff = 999
        for i in range(0, min_idx + 1):
            a = alpha_path[i]
            lasso = Lasso(alpha=a, max_iter=10000, random_state=42)
            lasso.fit(Xs, y)
            nz = np.sum(np.abs(lasso.coef_) > 1e-6)
            diff = abs(nz - 8)
            if diff < best_diff:
                best_diff = diff
                best_a = a
                lasso_final = lasso
    sel_idx = np.where(np.abs(lasso_final.coef_) > 1e-6)[0]
    sel_vars = [feature_names[i] for i in sel_idx]
    coef_vals = lasso_final.coef_[sel_idx]
    coef_df = pd.DataFrame({
        "变量(英文)": sel_vars, "变量(中文)": [EN2CN.get(v, v) for v in sel_vars],
        "标准化系数": np.round(coef_vals, 6),
        "|系数|": np.round(np.abs(coef_vals), 6)
    }).sort_values("|系数|", ascending=False)
    print()
    print("  最优 lambda = %.6f" % best_a)
    print("  >> 选中: %d / %d 个变量" % (len(sel_vars), len(candidate_vars)))
    if len(sel_vars) > 0:
        print(coef_df.to_string(index=False))
    else:
        print("  !! LASSO未选中任何变量")
    return sel_vars, coef_df, best_a


def vif_analysis(df, features, threshold=10):
    print()
    print(SEP1)
    print("  [VIF分析] 多重共线性（阈值=%d）" % threshold)
    print(SEP1)
    if len(features) < 2:
        print("  变量不足2个，跳过")
        return pd.DataFrame(), features
    X = df[features].copy()
    X = X.fillna(X.median())
    remaining = features[:]
    while len(remaining) >= 2:
        X_sub = X[remaining].values
        vif_vals = []
        valid = []
        for i in range(len(remaining)):
            if np.std(X_sub[:, i]) < 1e-10:
                continue
            vif = variance_inflation_factor(X_sub, i)
            vif_vals.append(vif)
            valid.append(remaining[i])
        if not vif_vals:
            break
        vif_series = pd.Series(vif_vals, index=valid)
        max_vif = vif_series.max()
        max_var = vif_series.idxmax()
        if max_vif <= threshold:
            break
        X = X.drop(columns=[max_var])
        remaining.remove(max_var)
        print("  剔除: %s (VIF=%.2f)" % (EN2CN.get(max_var, max_var), max_vif))
    if len(remaining) >= 2:
        Xf = X[remaining].values
        vif_final = pd.DataFrame({
            "变量(英文)": remaining, "变量(中文)": [EN2CN.get(v, v) for v in remaining],
            "VIF": [round(variance_inflation_factor(Xf, i), 4) for i in range(len(remaining))]
        }).sort_values("VIF", ascending=False)
    else:
        vif_final = pd.DataFrame()
    print()
    print("  最终保留: %d 个 (VIF <= %d)" % (len(remaining), threshold))
    if not vif_final.empty:
        print(vif_final.to_string(index=False))
    return vif_final, remaining


def multiple_linear_regression(df, predictors):
    print()
    print(SEP1)
    print("  [多元线性回归] OLS")
    print(SEP1)
    X = df[predictors].copy()
    X = X.fillna(X.median())
    y = df[TARGET].values
    Xc = add_constant(X)
    model = sm.OLS(y, Xc.astype(float)).fit()
    coef_df = pd.DataFrame({
        "变量": model.params.index,
        "变量(中文)": ["截距"] + [EN2CN.get(c, c) for c in list(X.columns)],
        "系数": np.round(model.params.values, 6),
        "标准误": np.round(model.bse.values, 6),
        "t值": np.round(model.tvalues.values, 4),
        "p值": model.pvalues.values,
        "显著性": ["***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                  for p in model.pvalues.values]
    })
    coef_df["p值"] = coef_df["p值"].apply(
        lambda x: "%.4e" % x if x < 0.0001 else "%.4f" % x)
    print()
    print("  回归系数表:")
    print(coef_df.to_string(index=False))
    y_pred = model.predict(Xc)
    print()
    print("  模型拟合指标:")
    print("    R-squared        = %.6f" % model.rsquared)
    print("    Adj. R-squared   = %.6f" % model.rsquared_adj)
    print("    F-statistic      = %.4f  (p = %.4e)" % (model.fvalue, model.f_pvalue))
    print("    AIC              = %.2f" % model.aic)
    print("    BIC              = %.2f" % model.bic)
    print("    Log-Likelihood   = %.4f" % model.llf)
    print("    RMSE             = %.6f" % np.sqrt(mean_squared_error(y, y_pred)))
    print("    Durbin-Watson    = %.4f" % model.durbin_watson())
    return model, coef_df


def main():
    print()
    print(SEP1)
    print("  武汉理工大学数学建模训练 第3题 — 数据预处理与建模分析")
    print(SEP1)
    print(SEP2)
    print("  Step 1: 加载数据")
    print(SEP2)
    df1 = load_and_rename(CSV1)
    df2 = load_and_rename(CSV2)
    print("  训练集: %d 行 x %d 列" % df1.shape)
    print("  测试集: %d 行 x %d 列" % df2.shape)
    print(SEP2)
    print("  Step 2: 缺失值分析")
    print(SEP2)
    analyze_missing(df1, "训练集(填补前)")
    print(SEP2)
    print("  Step 3: 缺失值填补（中位数/众数）")
    print(SEP2)
    df1_clean = impute_median(df1)
    df2_clean = impute_median(df2)
    rem = df1_clean.isnull().sum().sum()
    print("  填补后缺失总数: %d" % rem)
    if TARGET in df1_clean.columns and df1_clean[TARGET].isnull().any():
        df1_clean[TARGET] = df1_clean[TARGET].fillna(df1_clean[TARGET].median())
    descriptive_stats(df1_clean, "训练集(填补后)")
    uni_result, uni_selected = univariate_correlation(df1_clean)
    lasso_candidates = uni_selected if len(uni_selected) > 0 else ALL_PREDICTORS_EN
    lasso_sel, lasso_coefs, best_a = lasso_selection(df1_clean, lasso_candidates)
    final_vars = lasso_sel if len(lasso_sel) > 0 else uni_selected[:min(10, len(uni_selected))]
    if len(final_vars) == 0:
        final_vars = ALL_PREDICTORS_EN[:min(5, len(ALL_PREDICTORS_EN))]
    vif_result, final_vars = vif_analysis(df1_clean, final_vars)
    if len(final_vars) == 0:
        final_vars = lasso_sel[:min(3, len(lasso_sel))]
    if len(final_vars) >= 1:
        model, coef_table = multiple_linear_regression(df1_clean, final_vars)
    else:
        print("  无可用预测变量，跳过回归")
        model = None
    print()
    print(SEP1)
    print("  最终汇总")
    print(SEP1)
    print()
    print("  最终入选变量 (%d 个):" % len(final_vars))
    for i, v in enumerate(final_vars, 1):
        print("    %2d. %s (%s)" % (i, EN2CN.get(v, v), v))
    if model is not None:
        print()
        print("    R-squared       = %.6f" % model.rsquared)
        print("    Adj. R-squared  = %.6f" % model.rsquared_adj)
        print("    AIC             = %.2f" % model.aic)
        print("    BIC             = %.2f" % model.bic)
    print()
    print(SEP1)
    print("  预处理完成")
    print(SEP1)
    return df1_clean, df2_clean, final_vars, model


if __name__ == "__main__":
    df1, df2, selected, model = main()
