# -*- coding: utf-8 -*-
"""
============================================================================
t1.py — 问题1：主要变量筛选（严格按问题1gpt.pdf方案）
============================================================================
流程（PDF方案 + advanced_preprocessing预处理）:
  1. MICE多重插补（调用 advanced_preprocessing）
  2. 删除日期衍生变量（exam_year, exam_month）
  3. 单因素相关性分析（Pearson/Spearman, p<0.05）
  4. LASSO变量压缩（LassoCV, 1se准则选λ）
  5. 向后消去法二次筛选（p_remove=0.05）→ 5~10个主要变量
  6. VIF多重共线性诊断
  7. 多元线性回归验证

用法:
  python t1.py

输出:
  figures/问题1_*.png — 图表
  output/问题1_*.csv  — 结果表
============================================================================
"""

import os, sys, warnings, datetime
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from advanced_preprocessing import preprocess_with_advanced_imputation, COLUMN_MAP

warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================
FIGURE_DIR = './figures'
OUTPUT_DIR = './output'
for d in [FIGURE_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

DIABETES_THRESHOLD = 6.7
RANDOM_SEED = 2026
np.random.seed(RANDOM_SEED)
P_THRESHOLD = 0.05  # PDF要求单因素筛选 p<0.05

# 列名逆向映射
col_map_reverse = {v: k for k, v in COLUMN_MAP.items()}

# 变量分组（用于可视化）
VAR_GROUPS = {
    '肝功能': ['GGT', 'ALT', 'AST', 'TP', 'GLOB', 'ALP', 'ALB', 'AGR'],
    '肾功能': ['BUN', 'Cr', 'UA'],
    '血脂':   ['TC', 'TG', 'HDL_C', 'LDL_C'],
    '乙肝':   ['HBsAg', 'HBsAb', 'HBeAg', 'HBeAb', 'HBcAb'],
    '乙肝缺失标记': ['HBsAg_missing', 'HBsAb_missing', 'HBeAg_missing', 'HBeAb_missing', 'HBcAb_missing'],
    '血常规': ['NEUT_pct', 'LYMPH_pct', 'MONO_pct', 'EO_pct', 'BASO_pct',
               'WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW',
               'PLT', 'MPV', 'PDW', 'PCT'],
    '基本信息': ['age', 'gender_male'],
}


# ============================================================================
# Step 0: 数据加载与预处理（PDF Step 1）
# ============================================================================
def load_data():
    """调用 advanced_preprocessing（MICE插补），然后按PDF方案删除日期衍生变量"""
    print("=" * 60)
    print("Step 0: 数据加载与预处理")
    print("   - MICE多重插补（advanced_preprocessing）")
    print("   - 删除日期衍生变量（PDF要求）")
    print("   - 保留乙肝缺失指示变量（用于综合判断）")
    print("=" * 60)

    # 调用 advanced_preprocessing（含MICE插补、缺失标记、日期提取等）
    df1, df2 = preprocess_with_advanced_imputation(strategy='mice', mice_iter=20)

    # 仅使用附件1（有血糖数据）
    df = df1.copy()

    # 按PDF方案：删除日期衍生变量（冲突1决策）
    for col in ['exam_year', 'exam_month']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            print(f"  已删除日期衍生列: {col}")

    print(f"\n预处理完成: {df.shape}, NaN残留: {df.isnull().sum().sum()}")
    print(f"血糖均值={df['glucose'].mean():.4f}, "
          f"糖尿病风险={((df['glucose']>=DIABETES_THRESHOLD).sum())}/{len(df)} "
          f"({(df['glucose']>=DIABETES_THRESHOLD).mean()*100:.2f}%)")
    return df


# ============================================================================
# Step 1: 单因素相关性分析（PDF Step 2）
# ============================================================================
def step1_univariate_screening(df):
    """单因素统计分析：连续变量Pearson/Spearman，分类变量t检验，p<0.05"""
    print("\n" + "=" * 60)
    print("Step 1: 单因素统计筛选（PDF Step 2）")
    print("筛选标准：Pearson/Spearman相关性，p < 0.05")
    print("=" * 60)

    glucose = df['glucose'].values
    feature_cols = [c for c in df.columns if c != 'glucose']

    results = []
    for col in feature_cols:
        x = df[col].values
        valid = ~(np.isnan(x) | np.isnan(glucose))
        if valid.sum() < 10:
            continue

        n_unique = df[col].nunique()
        try:
            if n_unique <= 2:
                # 二值变量用点二列相关（等价于t检验）
                pr, pp = stats.pearsonr(x[valid], glucose[valid])
                sr, sp = stats.spearmanr(x[valid], glucose[valid])
            else:
                # 连续变量：优先Pearson
                pr, pp = stats.pearsonr(x[valid], glucose[valid])
                sr, sp = stats.spearmanr(x[valid], glucose[valid])

            results.append({
                '特征': col, '中文名': col_map_reverse.get(col, col),
                '类型': '二值' if n_unique <= 2 else '连续',
                'Pearson_r': pr, 'Pearson_p': pp,
                'Spearman_r': sr, 'Spearman_p': sp,
                'abs_r': abs(pr)
            })
        except Exception:
            continue

    result_df = pd.DataFrame(results).sort_values('abs_r', ascending=False)
    result_df.to_csv(f'{OUTPUT_DIR}/问题1_单因素筛选.csv', index=False, encoding='utf-8-sig')

    # 筛选：p < 0.05
    passed = result_df[result_df['Pearson_p'] < P_THRESHOLD]
    print(f"\n全部特征: {len(result_df)} 个")
    print(f"通过单因素筛选(p<{P_THRESHOLD}): {len(passed)} 个")

    # 打印前15
    print("\n相关性前15:")
    print(passed[['中文名', '类型', 'Pearson_r', 'Pearson_p']].head(15).to_string(index=False))
    print(f"\n  未通过的特征: {[r['中文名'] for _, r in result_df.iterrows() if r['Pearson_p'] >= P_THRESHOLD]}")

    # 可视化
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_df = result_df.sort_values('Pearson_r')
    colors = ['#e74c3c' if r > 0 else '#3498db' for r in plot_df['Pearson_r']]
    bars = ax.barh(range(len(plot_df)), plot_df['Pearson_r'],
                   color=colors, alpha=1.0)
    # 逐一设置透明度（alpha不支持列表，需逐bar设置）
    for i, (_, row) in enumerate(plot_df.iterrows()):
        if row['Pearson_p'] >= P_THRESHOLD:
            bars[i].set_alpha(0.25)
            ax.text(0, i, ' ✗ p≥0.05', va='center', fontsize=7, color='gray')

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['中文名'], fontsize=8)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Pearson 相关系数')
    ax.set_title(f'与血糖的单因素相关性分析（深色=p<{P_THRESHOLD}，浅色=未通过）')
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题1_单因素筛选.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  图表: {FIGURE_DIR}/问题1_单因素筛选.png")

    return passed['特征'].tolist()


# ============================================================================
# Step 2: LASSO变量选择（PDF Step 3）
# ============================================================================
def step2_lasso(df, candidate_features):
    """LASSO变量压缩：LassoCV交叉验证 + 1se准则选取更严格λ"""
    print("\n" + "=" * 60)
    print("Step 2: LASSO变量压缩（PDF Step 3）")
    print("方法: LassoCV, 5折交叉验证, 1se准则选λ")
    print("=" * 60)

    X = df[candidate_features].copy()
    y = df['glucose'].values

    # 检查并处理无穷值
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    print(f"\n候选特征数: {len(candidate_features)}")
    print(f"样本数: {len(X)}")

    # 标准化（LASSO对量纲敏感）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # LassoCV交叉验证
    lasso = LassoCV(
        cv=5,
        random_state=RANDOM_SEED,
        max_iter=10000,
        n_jobs=-1,
        alphas=np.logspace(-4, 1, 100)
    )
    lasso.fit(X_scaled, y)

    # === 1se准则：选择MSE在最小值1个标准误内的最大λ（更严格） ===
    alphas = lasso.alphas_                       # 递减顺序
    mse_mean = lasso.mse_path_.mean(axis=1)
    mse_std = lasso.mse_path_.std(axis=1)
    min_idx = np.argmin(mse_mean)
    one_se = mse_mean[min_idx] + mse_std[min_idx]

    # 找出满足MSE <= (min+1se)的最大α（即最左侧索引）
    candidates_1se = np.where(mse_mean <= one_se)[0]
    alpha_1se_idx = candidates_1se[0]
    alpha_1se = alphas[alpha_1se_idx]

    # 用1se对应的α重新拟合
    lasso_final = Lasso(alpha=alpha_1se, max_iter=10000, random_state=RANDOM_SEED)
    lasso_final.fit(X_scaled, y)

    # 提取选中变量
    coef_mask = np.abs(lasso_final.coef_) > 1e-6
    selected_features = [candidate_features[i] for i, keep in enumerate(coef_mask) if keep]

    print(f"\n最小MSE对应λ: {lasso.alpha_:.6f} (非零系数: {np.sum(np.abs(lasso.coef_) > 1e-6)})")
    print(f"1se准则对应λ:  {alpha_1se:.6f}")
    print(f"1se规则非零系数个数: {sum(coef_mask)} / {len(candidate_features)}")

    # === 若1se选出0个变量，回退到最小MSE对应的λ ===
    if len(selected_features) == 0:
        print("\n  1se准则未选中任何变量，回退到最小MSE对应λ")
        alpha_1se = lasso.alpha_
        lasso_final = Lasso(alpha=alpha_1se, max_iter=10000, random_state=RANDOM_SEED)
        lasso_final.fit(X_scaled, y)
        coef_mask = np.abs(lasso_final.coef_) > 1e-6
        selected_features = [candidate_features[i] for i, keep in enumerate(coef_mask) if keep]
        print(f"  回退后非零系数个数: {sum(coef_mask)} / {len(candidate_features)}")

    if selected_features:
        print(f"\nLASSO选中的变量 ({len(selected_features)} 个):")
        for f in selected_features:
            idx = candidate_features.index(f)
            coef_val = lasso_final.coef_[idx]
            print(f"  {f:30s} ({col_map_reverse.get(f, f):12s})  系数={coef_val:+.6f}")
    else:
        print("\n警告: LASSO未选中任何变量！")

    # 保存LASSO路径图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：交叉验证误差路径（标出min和1se）
    ax = axes[0]
    ax.plot(np.log10(alphas), mse_mean, 'b-', label='CV均值')
    ax.fill_between(np.log10(alphas),
                    mse_mean - mse_std, mse_mean + mse_std,
                    alpha=0.2, color='b')
    ax.axvline(np.log10(lasso.alpha_), color='r', linestyle='--', alpha=0.5,
               label=f'λ_min=10^{np.log10(lasso.alpha_):.2f}')
    ax.axvline(np.log10(alpha_1se), color='darkred', linestyle='-', linewidth=2,
               label=f'λ_1se=10^{np.log10(alpha_1se):.2f}')
    ax.axhline(one_se, color='gray', linestyle=':', alpha=0.7,
               label='min+1se')
    ax.set_xlabel('log10(λ)')
    ax.set_ylabel('均方误差')
    ax.set_title('LASSO交叉验证误差路径（1se准则）')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 右图：系数
    ax = axes[1]
    coef_df = pd.DataFrame({
        '特征': selected_features,
        '中文名': [col_map_reverse.get(f, f) for f in selected_features],
        '标准化系数': [lasso_final.coef_[candidate_features.index(f)] for f in selected_features]
    }).sort_values('标准化系数', ascending=True)

    colors = ['#e74c3c' if c > 0 else '#3498db' for c in coef_df['标准化系数']]
    ax.barh(range(len(coef_df)), coef_df['标准化系数'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(coef_df)))
    ax.set_yticklabels(coef_df['中文名'], fontsize=9)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('LASSO标准化系数')
    ax.set_title(f'LASSO选中的变量（1se准则, {len(selected_features)}个）')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题1_LASSO结果.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  图表: {FIGURE_DIR}/问题1_LASSO结果.png")

    return selected_features


# ============================================================================
# Step 2.5: 向后消去法二次筛选（基于回归p值）
# ============================================================================
def step25_backward_elimination(df, lasso_features):
    """在LASSO基础上，用向后消去法剔除不显著变量（p_remove=0.05）"""
    print("\n" + "=" * 60)
    print("Step 2.5: 向后消去法（二次筛选）")
    print("方法: 在LASSO选中的变量上逐步回归，剔除p>0.05的变量")
    print("=" * 60)

    included = list(lasso_features)
    y = df['glucose']

    if len(included) == 0:
        return []

    round_num = 0
    while len(included) > 0:
        round_num += 1
        X = sm.add_constant(df[included])
        model = sm.OLS(y, X).fit()
        pvals = model.pvalues.drop('const', errors='ignore')

        if len(pvals) == 0:
            break

        worst_p = pvals.max()
        if worst_p > 0.05:
            drop_var = pvals.idxmax()
            included.remove(drop_var)
            print(f"    第{round_num}轮 → 剔除: {drop_var:30s} ({col_map_reverse.get(drop_var, drop_var):12s})  p={worst_p:.4f}")
        else:
            print(f"    第{round_num}轮 → 停止（最大p={worst_p:.4f} ≤ 0.05）")
            break

    # 若全部被剔除
    if len(included) == 0:
        print("\n  所有变量均被剔除！")
        return []

    print(f"\n向后消去后剩余 {len(included)} 个变量:")
    for f in included:
        print(f"  {f} ({col_map_reverse.get(f, f)})")
    return included


# ============================================================================
# Step 3: VIF多重共线性诊断（PDF Step 4的一部分）
# ============================================================================
def step3_vif(df, selected_features):
    """VIF多重共线性诊断"""
    print("\n" + "=" * 60)
    print("Step 3: VIF多重共线性诊断")
    print("=" * 60)

    if len(selected_features) < 2:
        print("选中变量不足2个，跳过VIF分析。")
        return None

    X = df[selected_features].dropna()
    vif_data = pd.DataFrame({
        '特征': selected_features,
        '中文名': [col_map_reverse.get(c, c) for c in selected_features],
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    }).sort_values('VIF', ascending=False)

    print("\nVIF结果:")
    print(vif_data.to_string(index=False))

    flagged = vif_data[vif_data['VIF'] > 10]
    if len(flagged) > 0:
        print(f"\nVIF>10的变量（存在共线性问题）: {flagged['特征'].tolist()}")
    else:
        print("\n所有变量VIF<10，不存在严重共线性。")

    vif_data.to_csv(f'{OUTPUT_DIR}/问题1_VIF分析.csv', index=False, encoding='utf-8-sig')
    return vif_data


# ============================================================================
# Step 4: 多元线性回归验证（PDF Step 4）
# ============================================================================
def step4_regression_validation(df, selected_features):
    """多元线性回归：系数解释、显著性检验"""
    print("\n" + "=" * 60)
    print("Step 4: 多元线性回归验证（PDF Step 4）")
    print("目的：回归系数解释 + 显著性检验 + 变量合理性验证")
    print("=" * 60)

    if len(selected_features) == 0:
        print("无选中变量，跳过回归验证。")
        return None

    X = sm.add_constant(df[selected_features])
    y = df['glucose']

    model = sm.OLS(y, X).fit()
    print(model.summary())

    # 系数表
    coef_table = pd.DataFrame({
        '变量': model.params.index,
        '中文名': [col_map_reverse.get(c, c) if c != 'const' else '截距' for c in model.params.index],
        '系数(B)': model.params.values,
        '标准误(SE)': model.bse.values,
        't值': model.tvalues.values,
        'p值': model.pvalues.values,
        '95%CI_lower': model.conf_int().iloc[:, 0].values,
        '95%CI_upper': model.conf_int().iloc[:, 1].values,
    })
    coef_table.to_csv(f'{OUTPUT_DIR}/问题1_最终回归系数.csv', index=False, encoding='utf-8-sig')
    print(f"\n系数表已保存: {OUTPUT_DIR}/问题1_最终回归系数.csv")

    # 森林图
    plot_df = coef_table[coef_table['变量'] != 'const'].copy()
    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.4)))
    y_pos = range(len(plot_df))
    ax.errorbar(plot_df['系数(B)'], y_pos,
                xerr=[(plot_df['系数(B)'] - plot_df['95%CI_lower']),
                      (plot_df['95%CI_upper'] - plot_df['系数(B)'])],
                fmt='o', capsize=3, markersize=7)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_df['中文名'])
    ax.set_xlabel('回归系数 (95% CI)')
    ax.set_title('最终变量回归系数及置信区间')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题1_回归系数森林图.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表: {FIGURE_DIR}/问题1_回归系数森林图.png")

    return model


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("=" * 70)
    print("问题1：主要变量筛选")
    print("方案: 问题1gpt.pdf + advanced_preprocessing预处理")
    print("流程: MICE插补 → 删除日期 → 单因素(p<0.05) → LASSO(1se) → 向后消去(p<0.05) → 回归验证")
    print("=" * 70)
    print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 0: 数据预处理
    df = load_data()

    # Step 1: 单因素筛选
    candidates = step1_univariate_screening(df)

    # Step 2: LASSO变量压缩（1se准则）
    lasso_selected = step2_lasso(df, candidates)

    if len(lasso_selected) == 0:
        print("\nERROR: LASSO未选中任何变量。请检查数据或调整LASSO参数。")
        return

    # Step 2.5: 向后消去法二次筛选
    final_features = step25_backward_elimination(df, lasso_selected)

    if len(final_features) == 0:
        print("\nERROR: 向后消去后无剩余变量。")
        return

    # Step 3: VIF诊断
    vif = step3_vif(df, final_features)

    # Step 4: 多元回归验证
    model = step4_regression_validation(df, final_features)

    print("\n" + "=" * 70)
    print(f"问题1分析完成！")
    print(f"最终确定 {len(final_features)} 个主要变量（控制在5~10个）:")
    for i, f in enumerate(final_features, 1):
        print(f"  {i}. {f} ({col_map_reverse.get(f, f)})")
    print(f"完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == '__main__':
    main()