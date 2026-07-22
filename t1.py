# -*- coding: utf-8 -*-
"""
============================================================================
t1.py — 问题1：主要变量筛选
糖尿病风险预测 — 数学建模C题
============================================================================
流程:
  1. MICE多重插补（调用 advanced_preprocessing）
  2. 单因素相关性分析（Pearson/Spearman）
  3. VIF多重共线性诊断
  4. 医学知识精选候选特征（24个）
  5. 向后消去法逐步回归（p_remove=0.0001）
  6. 多元回归验证 → 最终9个主要变量

用法:
  python t1.py

输出:
  figures/问题1_*.png — 图表
  output/问题1_*.csv  — 结果表
  output/问题1_答案与分析报告.md — 完整报告
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

# 列名逆向映射
col_map_reverse = {v: k for k, v in COLUMN_MAP.items()}

# 变量分组
VAR_GROUPS = {
    '肝功能': ['GGT', 'ALT', 'AST', 'TP', 'GLOB', 'ALP', 'ALB', 'AGR'],
    '肾功能': ['BUN', 'Cr', 'UA'],
    '血脂':   ['TC', 'TG', 'HDL_C', 'LDL_C'],
    '乙肝':   ['HBsAg', 'HBsAb', 'HBeAg', 'HBeAb', 'HBcAb'],
    '血常规': ['NEUT_pct', 'LYMPH_pct', 'MONO_pct', 'EO_pct', 'BASO_pct',
               'WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW',
               'PLT', 'MPV', 'PDW', 'PCT'],
    '基本信息': ['age', 'gender'],
}

# 医学知识精选的候选特征（从每组高相关指标中保留最具代表性者）
CURATED_FEATURES = [
    'age', 'gender_male',
    'TC', 'TG', 'HDL_C',
    'ALT', 'AST', 'GGT', 'ALB',
    'BUN', 'Cr', 'UA',
    'HGB', 'MCV', 'RDW',
    'WBC', 'NEUT_pct', 'LYMPH_pct',
    'PLT', 'MPV',
    'ALP',
    'HBsAg', 'HBsAb',
]


# ============================================================================
# Step 0: 数据加载与预处理
# ============================================================================
def load_data():
    """调用 advanced_preprocessing 加载并插补数据"""
    print("=" * 60)
    print("Step 0: 数据加载与预处理（MICE多重插补）")
    print("=" * 60)
    df1, df2 = preprocess_with_advanced_imputation(strategy='mice', mice_iter=20)
    print(f"\n附件1: {df1.shape}, 附件2: {df2.shape}")
    print(f"血糖均值={df1['glucose'].mean():.4f}, 糖尿病风险={((df1['glucose']>=DIABETES_THRESHOLD).sum())}/{len(df1)} ({(df1['glucose']>=DIABETES_THRESHOLD).mean()*100:.2f}%)")
    return df1


# ============================================================================
# Step 1: 单因素相关性分析
# ============================================================================
def step1_correlation(df):
    """Pearson/Spearman相关性分析 + 筛选"""
    print("\n" + "=" * 60)
    print("Step 1: 单因素相关性分析")
    print("=" * 60)
    
    glucose = df['glucose'].values
    feature_cols = [c for c in df.columns if c != 'glucose']
    
    results = []
    for col in feature_cols:
        x = df[col].values
        valid = ~(np.isnan(x) | np.isnan(glucose))
        if valid.sum() < 10:
            continue
        try:
            pr, pp = stats.pearsonr(x[valid], glucose[valid])
            sr, sp = stats.spearmanr(x[valid], glucose[valid])
            results.append({'特征': col, '中文名': col_map_reverse.get(col, col),
                            'Pearson_r': pr, 'Pearson_p': pp,
                            'Spearman_r': sr, 'Spearman_p': sp, 'abs_r': abs(pr)})
        except:
            continue
    
    result_df = pd.DataFrame(results).sort_values('abs_r', ascending=False)
    
    print("\n相关性前10:")
    print(result_df[['中文名', 'Pearson_r', 'Pearson_p']].head(10).to_string(index=False))
    print(f"\n显著相关（p<0.001）: {(result_df['Pearson_p']<0.001).sum()}/{len(result_df)}")
    
    result_df.to_csv(f'{OUTPUT_DIR}/问题1_单因素筛选.csv', index=False, encoding='utf-8-sig')
    
    # 可视化
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_df = result_df.head(25).sort_values('Pearson_r')
    colors = ['#e74c3c' if r > 0 else '#3498db' for r in plot_df['Pearson_r']]
    ax.barh(range(len(plot_df)), plot_df['Pearson_r'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['中文名'], fontsize=9)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Pearson 相关系数')
    ax.set_title('与血糖显著相关的指标（按|r|排序）')
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题1_单因素筛选.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表: {FIGURE_DIR}/问题1_单因素筛选.png")
    
    return result_df['特征'].tolist()


# ============================================================================
# Step 2: VIF多重共线性诊断
# ============================================================================
def step2_vif(df):
    """计算VIF"""
    print("\n" + "=" * 60)
    print("Step 2: VIF多重共线性诊断")
    print("=" * 60)
    
    num_cols = [c for c in df.columns if c != 'glucose'
                and np.issubdtype(df[c].dtype, np.number) and df[c].nunique() > 2]
    X = df[num_cols].dropna()
    
    vif_data = pd.DataFrame({
        '特征': num_cols,
        '中文名': [col_map_reverse.get(c, c) for c in num_cols],
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    }).sort_values('VIF', ascending=False)
    
    print("\nVIF前10:")
    print(vif_data.head(10).to_string(index=False))
    
    vif_data.to_csv(f'{OUTPUT_DIR}/问题1_VIF分析.csv', index=False, encoding='utf-8-sig')
    return vif_data


# ============================================================================
# Step 3: 向后消去法
# ============================================================================
def step3_backward_elimination(df):
    """向后消去法逐步回归"""
    print("\n" + "=" * 60)
    print("Step 3: 向后消去法逐步回归（p_remove=0.0001）")
    print("=" * 60)
    
    # 候选特征：医学精选 + 乙肝缺失指示变量（仅保留HBsAg_missing）
    hbsag_missing = [f'HBsAg_missing'] if 'HBsAg_missing' in df.columns else []
    candidates = [c for c in CURATED_FEATURES if c in df.columns] + hbsag_missing
    
    print(f"\n候选特征 ({len(candidates)} 个):")
    for c in candidates:
        print(f"  {c} ({col_map_reverse.get(c, c)})")
    
    X = df[candidates].copy()
    y = df['glucose']
    
    # 清洗inf/nan
    X = X.replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1)
    if (~valid).sum() > 0:
        print(f"  删除 {(~valid).sum()} 行无效数据")
        X, y = X[valid], y[valid]
    
    # 向后消去
    included = list(X.columns)
    
    for round_num in range(1, len(candidates) * 2 + 1):
        if len(included) == 0:
            break
        
        X_sub = sm.add_constant(X[included]).replace([np.inf, -np.inf], np.nan)
        valid_rows = X_sub.notna().all(axis=1)
        X_clean, y_clean = X_sub[valid_rows], y[valid_rows]
        
        if len(X_clean) < 100:
            included.pop()
            continue
        
        try:
            model = sm.OLS(y_clean, X_clean).fit()
        except Exception:
            included.pop()
            continue
        
        pvalues = model.pvalues.drop('const', errors='ignore')
        if len(pvalues) == 0:
            break
        
        worst_pval = pvalues.max()
        worst_feature = pvalues.idxmax()
        
        if worst_pval > 0.0001:
            included.remove(worst_feature)
            print(f"    第{round_num}轮 → 剔除: {worst_feature} (p={worst_pval:.6f})")
        else:
            print(f"    第{round_num}轮 → 停止消去（最大p={worst_pval:.6f} ≤ 0.0001）")
            break
    
    print(f"\n最终选中 {len(included)} 个变量:")
    for f in included:
        print(f"  {f} ({col_map_reverse.get(f, f)})")
    
    return included, X, y


# ============================================================================
# Step 4: 多元回归验证
# ============================================================================
def step4_validation(df, selected_features):
    """多元线性回归验证"""
    print("\n" + "=" * 60)
    print("Step 4: 多元线性回归验证")
    print("=" * 60)
    
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
    print("方法: MICE插补 → 单因素分析 → VIF → 向后消去法 → 回归验证")
    print("=" * 70)
    print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    df = load_data()
    candidates = step1_correlation(df)
    vif = step2_vif(df)
    selected, X_clean, y_clean = step3_backward_elimination(df)
    model = step4_validation(df, selected)
    
    print("\n" + "=" * 70)
    print("问题1分析完成！")
    print(f"完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
