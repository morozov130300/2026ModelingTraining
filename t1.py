# -*- coding: utf-8 -*-
"""
============================================================================
t1.py — 问题1：主要变量筛选（严格按问题1gpt.pdf方案）
============================================================================
流程（PDF方案 + advanced_preprocessing预处理）:
  1. MICE多重插补（调用 advanced_preprocessing）
  2. 删除日期衍生变量（exam_year, exam_month）
  3. 单因素相关性分析（Pearson/Spearman, p<0.05）
  4. LASSO变量选择（沿正则化路径自动选取5~10个变量）
  5. 显著性校验 + VIF多重共线性诊断
  6. 多元线性回归验证

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
# 中文字体配置（按优先级尝试多个常见字体）
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
_chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'WenQuanYi Micro Hei',
                   'Noto Sans CJK SC', 'PingFang SC', 'Heiti SC', 'DengXian']
_chosen = None
for f in _chinese_fonts:
    try:
        fm.findfont(f, fallback_to_default=False)
        _chosen = f
        break
    except Exception:
        continue
if _chosen:
    plt.rcParams['font.sans-serif'] = [_chosen] + plt.rcParams['font.sans-serif']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LassoCV, Lasso, lasso_path
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

    # 可视化：单因素相关性分析图
    fig, ax = plt.subplots(figsize=(13, 8))

    # 给每个特征标上颜色（按生理系统分组）
    group_colors = {
        '基本信息': '#E74C3C', '肝功能': '#2ECC71', '肾功能': '#3498DB',
        '血脂': '#F39C12', '乙肝': '#9B59B6', '乙肝缺失标记': '#BDC3C7',
        '血常规': '#1ABC9C',
    }
    # 为每个特征分配组别
    feature_group = {}
    for grp, cols in VAR_GROUPS.items():
        for c in cols:
            feature_group[c] = grp
    # special cols not in VAR_GROUPS
    for c in df.columns:
        if c != 'glucose' and c not in feature_group:
            feature_group[c] = '其他'

    plot_df = result_df.sort_values('Pearson_r')
    bar_colors = []
    for _, row in plot_df.iterrows():
        col = row['特征']
        grp = feature_group.get(col, '其他')
        bar_colors.append(group_colors.get(grp, '#95A5A6'))

    bars = ax.barh(range(len(plot_df)), plot_df['Pearson_r'],
                   color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    # 淡化未通过的特征
    for i, (_, row) in enumerate(plot_df.iterrows()):
        if row['Pearson_p'] >= P_THRESHOLD:
            bars[i].set_alpha(0.2)
            ax.text(0.01, i, '✗ 未通过(p≥0.05)', va='center', fontsize=7,
                    color='#7F8C8D')

    # 在柱状条上标注相关系数
    for i, (_, row) in enumerate(plot_df.iterrows()):
        r = row['Pearson_r']
        label = f"r={r:.3f}"
        ax.text(r + 0.003 if r >= 0 else r - 0.04, i, label,
                va='center', fontsize=7, color='#2C3E50')

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['中文名'], fontsize=9)
    ax.axvline(0, color='#2C3E50', linewidth=1)
    ax.set_xlabel('Pearson 相关系数（r）', fontsize=12)
    ax.set_title('与血糖的单因素相关性分析', fontsize=14, fontweight='bold')
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.85, label=grp)
                       for grp, c in group_colors.items()
                       if any(feature_group.get(c2) == grp for _, c2 in plot_df['特征'].items())]
    # 只显示实际出现的分组
    present_groups = set()
    for _, row in plot_df.iterrows():
        grp = feature_group.get(row['特征'], '其他')
        present_groups.add(grp)
    legend_elements = [Patch(facecolor=c, alpha=0.85, label=grp)
                       for grp, c in group_colors.items()
                       if grp in present_groups]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right',
              title='生理系统', title_fontsize=9)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')

    # 标注显著性区间
    ax.text(0.98, 0.95, f'通过筛选: {len(passed)}/44 个 (p<{P_THRESHOLD})',
            transform=ax.transAxes, fontsize=9, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='#E8F8F5', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题1_单因素相关性分析.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  图表: {FIGURE_DIR}/问题1_单因素相关性分析.png")

    return passed['特征'].tolist()


# ============================================================================
# Step 2: LASSO变量选择（PDF Step 3）
# ============================================================================
def step2_lasso(df, candidate_features):
    """LASSO变量选择：沿正则化路径搜索能筛选出5~10个变量的最优λ"""
    print("\n" + "=" * 60)
    print("Step 2: LASSO变量选择（PDF Step 3）")
    print("方法: 沿正则化路径搜索，自动选取使变量数在5~10个的λ")
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

    # 沿全正则化路径，精细搜索变量数
    alphas_path, coefs_path, _ = lasso_path(
        X_scaled, y,
        alphas=np.logspace(-3, 0.5, 300),  # 0.001 ~ 3.16, 精细300步
        random_state=RANDOM_SEED,
        max_iter=10000
    )
    # alphas_path的索引0=最大α（最简模型），索引-1=最小α（最复杂模型）
    n_nonzero = np.sum(np.abs(coefs_path) > 1e-6, axis=0)

    # === 搜索目标: 5~10个变量 ===
    target_min, target_max = 5, 10
    selected_alpha = None
    selected_count = 0
    selected_idx = 0

    # 从最简模型向复杂搜索：找第一个变量数 >= target_min 的位置
    for i in range(len(alphas_path)):
        cnt = n_nonzero[i]
        if target_min <= cnt <= target_max:
            selected_alpha = alphas_path[i]
            selected_count = cnt
            selected_idx = i
            break
    else:
        # 没有精确落在范围内的：找变量数刚好超过target_max的位置
        for i in range(len(alphas_path)):
            cnt = n_nonzero[i]
            if cnt >= target_min:
                selected_alpha = alphas_path[i]
                selected_count = cnt
                selected_idx = i
                break
        # 兜底：取最简模型中变量数最接近target_max的
        if selected_alpha is None:
            # 从复杂端往回找
            for i in range(len(alphas_path) - 1, -1, -1):
                cnt = n_nonzero[i]
                if cnt > 0:
                    selected_alpha = alphas_path[i + 1] if i + 1 < len(alphas_path) else alphas_path[i]
                    selected_count = n_nonzero[i + 1] if i + 1 < len(alphas_path) else cnt
                    selected_idx = i + 1 if i + 1 < len(alphas_path) else i
                    break

    # 用选中α拟合最终模型
    lasso_final = Lasso(alpha=selected_alpha, max_iter=10000, random_state=RANDOM_SEED)
    lasso_final.fit(X_scaled, y)
    coef_mask = np.abs(lasso_final.coef_) > 1e-6
    selected_features = [candidate_features[i] for i, keep in enumerate(coef_mask) if keep]

    # 同时也用min-CV α做对比（仅用于展示CV误差路径）
    lasso_cv = LassoCV(cv=5, random_state=RANDOM_SEED, max_iter=10000, alphas=np.logspace(-3, 0.5, 100))
    lasso_cv.fit(X_scaled, y)
    alpha_min = lasso_cv.alpha_
    # 提取min-CV的变量数
    lasso_cv_final = Lasso(alpha=alpha_min, max_iter=10000, random_state=RANDOM_SEED)
    lasso_cv_final.fit(X_scaled, y)
    cv_count = np.sum(np.abs(lasso_cv_final.coef_) > 1e-6)

    print(f"\nmin-CV(λ={alpha_min:.6f}) → {cv_count}个变量")
    print(f"选中λ={selected_alpha:.6f} → {selected_count}个变量 ✓")

    if selected_features:
        print(f"\nLASSO选中的变量 ({len(selected_features)} 个):")
        for f in selected_features:
            idx = candidate_features.index(f)
            coef_val = lasso_final.coef_[idx]
            print(f"  {f:30s} ({col_map_reverse.get(f, f):12s})  系数={coef_val:+.6f}")

    # 保存LASSO路径图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ====== 左图：正则化路径 ======
    ax = axes[0]
    # 绘制变量系数路径（显示全部32条）
    cmap = plt.cm.viridis
    for j in range(coefs_path.shape[0]):
        ax.plot(np.log10(alphas_path), coefs_path[j, :],
                color=cmap(j / coefs_path.shape[0]), alpha=0.6, linewidth=0.8)

    # 标注非零系数个数
    ax_twin = ax.twinx()
    ax_twin.plot(np.log10(alphas_path), n_nonzero, 'k-', linewidth=2.5, alpha=0.7)
    ax_twin.fill_between(np.log10(alphas_path), 0, n_nonzero, color='black', alpha=0.08)
    ax_twin.set_ylabel('非零系数个数', fontsize=11, color='black')

    # 目标区间标注
    ax_twin.axhspan(target_min, target_max, xmin=0, xmax=1,
                    alpha=0.15, color='#2ECC71')
    ax_twin.axhline(target_min, color='#2ECC71', linestyle='--', alpha=0.6, linewidth=1)
    ax_twin.axhline(target_max, color='#2ECC71', linestyle='--', alpha=0.6, linewidth=1,
                    label=f'目标区间[{target_min},{target_max}]')
    # 选中点
    ax.axvline(np.log10(selected_alpha), color='#E74C3C', linestyle='-', linewidth=2.5)
    ax.annotate(f'选中: λ={selected_alpha:.4f}\n{selected_count}个变量',
                xy=(np.log10(selected_alpha), 1), fontsize=10,
                color='#E74C3C', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#FDEDEC', alpha=0.9))

    ax.set_xlabel('log10(λ)', fontsize=12)
    ax.set_ylabel('回归系数', fontsize=11)
    ax.set_title('LASSO正则化路径', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax_twin.legend(fontsize=8, loc='upper right')

    # ====== 右图：选中变量的系数 ======
    ax = axes[1]
    coef_df = pd.DataFrame({
        '特征': selected_features,
        '中文名': [col_map_reverse.get(f, f) for f in selected_features],
        '标准化系数': [lasso_final.coef_[candidate_features.index(f)] for f in selected_features]
    }).sort_values('标准化系数', ascending=True)

    colors_bar = ['#E74C3C' if c > 0 else '#3498DB' for c in coef_df['标准化系数']]
    bars = ax.barh(range(len(coef_df)), coef_df['标准化系数'],
                   color=colors_bar, alpha=0.85, edgecolor='white', linewidth=0.8)

    # 在柱上标注数值
    for i, (_, row) in enumerate(coef_df.iterrows()):
        v = row['标准化系数']
        label = f"{v:+.3f}"
        ax.text(v + 0.02 if v >= 0 else v - 0.12, i, label,
                va='center', fontsize=9, color='#2C3E50', fontweight='bold')

    ax.set_yticks(range(len(coef_df)))
    ax.set_yticklabels(coef_df['中文名'], fontsize=10)
    ax.axvline(0, color='#2C3E50', linewidth=1.2)
    ax.set_xlabel('LASSO标准化系数', fontsize=12)
    ax.set_title(f'最终选中变量（{len(selected_features)}个, λ={selected_alpha:.4f}）',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题1_LASSO正则化路径.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  图表: {FIGURE_DIR}/问题1_LASSO正则化路径.png")

    return selected_features


# ============================================================================
# Step 2.5: 向后消去法二次筛选（基于回归p值）
# ============================================================================
def step25_backward_elimination(df, lasso_features):
    """轻量校验：检查LASSO选出的变量是否在回归中保持显著，必要时微调"""
    print("\n" + "=" * 60)
    print("Step 2.5: 显著性校验（轻量）")
    print("说明: LASSO已选出5~10个变量，此步仅验证其回归显著性")
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
    """VIF多重共线性诊断 + 可视化"""
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

    # VIF可视化
    fig, ax = plt.subplots(figsize=(10, max(4, len(vif_data) * 0.35)))

    plot_df = vif_data.copy()
    # 避免极端值影响可视化，VIF截断显示
    plot_df['VIF_display'] = plot_df['VIF'].clip(upper=50)
    plot_df['is_high'] = plot_df['VIF'] > 10

    bar_colors = ['#E74C3C' if h else '#3498DB' for h in plot_df['is_high']]
    bars = ax.barh(range(len(plot_df)), plot_df['VIF_display'],
                   color=bar_colors, alpha=0.8, edgecolor='white', linewidth=0.5)

    ax.axvline(10, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7,
               label='VIF=10（共线性警戒线）')

    # 标注实际VIF值
    for i, (_, row) in enumerate(plot_df.iterrows()):
        v = row['VIF']
        label = f"{v:.1f}"
        if v > 50:
            label += ' (严重)'
        ax.text(min(v, 50) + 0.5, i, label, va='center', fontsize=8, color='#2C3E50')

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['中文名'], fontsize=9)
    ax.set_xlabel('方差膨胀因子（VIF）', fontsize=12)
    ax.set_title('多重共线性诊断（VIF）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')

    # 标注解释
    high_count = plot_df['is_high'].sum()
    ax.text(0.98, 0.95, f'VIF>10: {high_count}/{len(plot_df)} 个',
            transform=ax.transAxes, fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='#FDEDEC', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题1_VIF共线性诊断.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表: {FIGURE_DIR}/问题1_VIF共线性诊断.png")

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

    # 森林图（回归系数及95%置信区间）
    plot_df = coef_table[coef_table['变量'] != 'const'].copy()
    plot_df = plot_df.sort_values('系数(B)')

    fig, ax = plt.subplots(figsize=(11, max(5, len(plot_df) * 0.45)))

    y_pos = range(len(plot_df))
    # 标记显著与否的颜色
    point_colors = ['#2ECC71' if p < 0.05 else '#BDC3C7' for p in plot_df['p值']]

    # 先画误差棒（不带标记）
    ax.errorbar(plot_df['系数(B)'], y_pos,
                xerr=[(plot_df['系数(B)'] - plot_df['95%CI_lower']),
                      (plot_df['95%CI_upper'] - plot_df['系数(B)'])],
                fmt='none', ecolor='#7F8C8D', elinewidth=1.5, capsize=4)
    # 再逐个画标记点（支持不同颜色）
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.plot(row['系数(B)'], i, 'o', markersize=9,
                color='#2C3E50', markerfacecolor=point_colors[i],
                markeredgecolor='#2C3E50', markeredgewidth=1)

    # 标注p值
    for i, (_, row) in enumerate(plot_df.iterrows()):
        p = row['p值']
        if p < 0.001:
            p_label = '***'
        elif p < 0.01:
            p_label = '**'
        elif p < 0.05:
            p_label = '*'
        else:
            p_label = 'n.s.'
        ax.text(row['95%CI_upper'] + 0.02, i, p_label, va='center', fontsize=9,
                color='#E74C3C' if p < 0.05 else '#95A5A6', fontweight='bold')

    ax.axvline(0, color='#2C3E50', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_df['中文名'], fontsize=10)
    ax.set_xlabel('回归系数 (95% 置信区间)', fontsize=12)
    ax.set_title('最终变量回归系数及置信区间', fontsize=14, fontweight='bold')

    # R²标注
    r2_text = f'R² = {model.rsquared:.3f}   调整R² = {model.rsquared_adj:.3f}'
    ax.text(0.98, 0.02, r2_text, transform=ax.transAxes, fontsize=10,
            ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='#EBF5FB', alpha=0.8))

    # 显著性标注说明
    ax.text(0.98, 0.92, '* p<0.05  ** p<0.01  *** p<0.001',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            color='#7F8C8D')

    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题1_最终回归系数森林图.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表: {FIGURE_DIR}/问题1_最终回归系数森林图.png")

    # 模型诊断图（残差分析）
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    residuals = model.resid
    fitted = model.fittedvalues
    standardized_resid = residuals / np.std(residuals)

    # 1. 残差 vs 拟合值
    ax = axes[0, 0]
    ax.scatter(fitted, residuals, alpha=0.3, s=10, color='#3498DB')
    ax.axhline(y=0, color='#E74C3C', linestyle='--', linewidth=1.5)
    # 平滑趋势线
    from scipy.interpolate import UnivariateSpline
    try:
        sort_idx = np.argsort(fitted)
        spline = UnivariateSpline(fitted[sort_idx], residuals[sort_idx], s=len(fitted)*10)
        ax.plot(fitted[sort_idx], spline(fitted[sort_idx]), color='#E74C3C', linewidth=2, alpha=0.7)
    except Exception:
        pass
    ax.set_xlabel('拟合值', fontsize=11)
    ax.set_ylabel('残差', fontsize=11)
    ax.set_title('残差 vs 拟合值', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')

    # 2. Q-Q图
    ax = axes[0, 1]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor('#3498DB')
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[0].set_alpha(0.4)
    ax.get_lines()[1].set_color('#E74C3C')
    ax.get_lines()[1].set_linewidth(2)
    ax.set_title('正态性Q-Q图', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')

    # 3. 残差直方图
    ax = axes[1, 0]
    ax.hist(residuals, bins=40, edgecolor='white', alpha=0.7, color='#3498DB')
    # 叠加正态曲线
    from scipy.stats import norm
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x_range, len(residuals) * np.diff(np.histogram(residuals, bins=40)[1])[0]
            * norm.pdf(x_range, residuals.mean(), residuals.std()),
            color='#E74C3C', linewidth=2, alpha=0.7)
    ax.set_xlabel('残差', fontsize=11)
    ax.set_ylabel('频数', fontsize=11)
    ax.set_title('残差分布', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')

    # 4. 标准化残差 vs 拟合值
    ax = axes[1, 1]
    ax.scatter(fitted, standardized_resid, alpha=0.3, s=10, color='#3498DB')
    ax.axhline(y=0, color='#E74C3C', linestyle='--', linewidth=1.5)
    ax.axhline(y=3, color='#95A5A6', linestyle=':', alpha=0.7, label='±3σ')
    ax.axhline(y=-3, color='#95A5A6', linestyle=':', alpha=0.7)
    ax.set_xlabel('拟合值', fontsize=11)
    ax.set_ylabel('标准化残差', fontsize=11)
    ax.set_title('标准化残差 vs 拟合值', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题1_模型诊断图.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表: {FIGURE_DIR}/问题1_模型诊断图.png")

    return model


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("=" * 70)
    print("问题1：主要变量筛选")
    print("方案: 问题1gpt.pdf + advanced_preprocessing预处理")
    print("流程: MICE插补 → 单因素(p<0.05) → LASSO(5~10变量) → 回归验证 + VIF")
    print("=" * 70)
    print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 0: 数据预处理
    df = load_data()

    # Step 1: 单因素筛选
    candidates = step1_univariate_screening(df)

    # Step 2: LASSO变量选择（正则化路径搜索，自动锁定5~10个变量）
    lasso_selected = step2_lasso(df, candidates)

    if len(lasso_selected) == 0:
        print("\nERROR: LASSO未选中任何变量。请检查数据或调整LASSO参数。")
        return

    # Step 2.5: 显著性校验（LASSO为主，此步仅微调）
    final_features = step25_backward_elimination(df, lasso_selected)

    if len(final_features) == 0:
        print("\nERROR: 向后消去后无剩余变量。")
        return

    # Step 3: VIF诊断
    vif = step3_vif(df, final_features)

    # Step 4: 多元回归验证
    model = step4_regression_validation(df, final_features)

    # 最终变量相关性热力图
    if len(final_features) >= 3:
        print("\n" + "=" * 60)
        print("生成最终变量相关性热力图")
        print("=" * 60)
        corr_cols = final_features + ['glucose']
        corr_data = df[corr_cols].copy()
        corr_matrix = corr_data.corr()

        # 中文标签映射
        corr_labels = [col_map_reverse.get(c, c) for c in corr_matrix.columns]

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                    annot=True, fmt='.2f', square=True,
                    linewidths=0.8, cbar_kws={"shrink": 0.8},
                    xticklabels=corr_labels, yticklabels=corr_labels,
                    ax=ax)

        ax.set_title('最终变量与血糖的相关性热力图', fontsize=14, fontweight='bold', pad=15)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig(f'{FIGURE_DIR}/问题1_最终变量相关性热力图.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  图表: {FIGURE_DIR}/问题1_最终变量相关性热力图.png")

    print("\n" + "=" * 70)
    print(f"问题1分析完成！")
    print(f"最终确定 {len(final_features)} 个主要变量（控制在5~10个）:")
    for i, f in enumerate(final_features, 1):
        print(f"  {i}. {f} ({col_map_reverse.get(f, f)})")
    print(f"完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == '__main__':
    main()