# -*- coding: utf-8 -*-
"""
t2.py — 问题2：血糖预测建模（递进式模型构建）
方案: 问题2思路.pdf
流程:
  Step 0: 数据预处理（MICE插补，同问题1）
  Step 1: 多元线性回归（MLR）— 基础线性模型
  Step 2: 岭回归（Ridge）    — MLR的改进：L2正则化处理共线性
  Step 3: 随机森林回归（RF）  — 非线性预测模型
  Step 4: 模型对比（岭回归 vs 随机森林）+ 可视化
说明:
  - 基于附件1全部数据，不划分训练/测试集（整体建模）
  - 不使用附件2（第四问才使用）
  - 岭回归是多元线性回归的正则化改进，非独立方法
输出:
  figures/问题2_*.png
  output/问题2_*.csv
  output/问题2_答案与分析报告.md
"""

import os, sys, warnings, datetime
warnings.filterwarnings('ignore', message='Font.*glyph')
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
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
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from 高级预处理 import preprocess_with_advanced_imputation, COLUMN_MAP

warnings.filterwarnings('ignore')

FIGURE_DIR = './figures'
OUTPUT_DIR = './output'
for d in [FIGURE_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_SEED = 2026
np.random.seed(RANDOM_SEED)

col_map_reverse = {v: k for k, v in COLUMN_MAP.items()}

# 问题1最终选出的7个变量
FINAL_FEATURES = ['age', 'TG', 'BUN', 'ALP', 'RBC', 'ALT', 'NEUT_pct']

# ============================================================
# Step 0: 数据加载
# ============================================================
def load_data():
    """加载数据，使用与问题1相同的预处理流程，仅使用附件1"""
    print("=" * 60)
    print("Step 0: 数据加载与预处理")
    print("   - MICE多重插补（advanced_preprocessing）")
    print("   - 仅使用附件1（前三问均不使用附件2）")
    print("=" * 60)

    df1, _ = preprocess_with_advanced_imputation(strategy='mice', mice_iter=20)

    for col in ['exam_year', 'exam_month']:
        if col in df1.columns:
            df1.drop(columns=[col], inplace=True)

    print(f"\n预处理完成: {df1.shape}, NaN残留: {df1.isnull().sum().sum()}")
    print(f"血糖均值={df1['glucose'].mean():.4f}")
    return df1

# ============================================================
# Step 1: 多元线性回归（基础模型）
# ============================================================
def step1_mlr(df):
    """多元线性回归 — 作为基础线性模型"""
    print("\n" + "=" * 60)
    print("Step 1: 多元线性回归 (MLR) — 基础模型")
    print("目的: 建立血糖与7个变量的线性关系，提供可解释回归系数")
    print("=" * 60)

    X = df[FINAL_FEATURES].copy()
    y = df['glucose'].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # OLS回归
    mlr = LinearRegression()
    mlr.fit(X_scaled, y)
    y_pred = mlr.predict(X_scaled)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    n, p = len(y), X_scaled.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(f"\n  [全样本表现]")
    print(f"  $R^2$ = {r2:.4f}   调整$R^2$ = {adj_r2:.4f}")
    print(f"  RMSE = {rmse:.4f}    MAE = {mae:.4f}")

    # 回归系数
    coef_df = pd.DataFrame({
        '特征': FINAL_FEATURES,
        '中文名': [col_map_reverse.get(f, f) for f in FINAL_FEATURES],
        '标准化系数': mlr.coef_,
        '|系数|': np.abs(mlr.coef_),
    }).sort_values('|系数|', ascending=False)

    print(f"\n  标准化回归系数（按|系数|降序）:")
    coeff_order = coef_df['特征'].tolist()
    for _, row in coef_df.iterrows():
        print(f"    {row['中文名']:16s} 系数={row['标准化系数']:+.6f}")

    return {
        'coef_df': coef_df,
        'coeff_order': coeff_order,
        'r2': r2, 'adj_r2': adj_r2, 'rmse': rmse, 'mae': mae,
        'y_pred': y_pred,
    }

# ============================================================
# Step 2: 岭回归（MLR的改进）
# ============================================================
def step2_ridge(df):
    """岭回归 — 在MLR基础上引入L2正则化，处理共线性"""
    print("\n" + "=" * 60)
    print("Step 2: 岭回归 (Ridge) — MLR的L2正则化改进")
    print("目的: 通过L2正则化收缩系数，解决多重共线性，提高稳定性")
    print("=" * 60)

    X = df[FINAL_FEATURES].copy()
    y = df['glucose'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 交叉验证选最优alpha
    alphas = np.logspace(-3, 3, 100)
    ridge_cv = RidgeCV(alphas=alphas, scoring='r2', cv=10)
    ridge_cv.fit(X_scaled, y)
    best_alpha = ridge_cv.alpha_

    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_scaled, y)
    y_pred = ridge.predict(X_scaled)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    n, p = len(y), X_scaled.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(f"\n  最优alpha: {best_alpha:.6f}（10折CV从{len(alphas)}个候选中选出）")
    print(f"\n  [全样本表现]")
    print(f"  $R^2$ = {r2:.4f}   调整$R^2$ = {adj_r2:.4f}")
    print(f"  RMSE = {rmse:.4f}    MAE = {mae:.4f}")

    # 岭回归系数（按|系数|降序）
    coef_df = pd.DataFrame({
        '特征': FINAL_FEATURES,
        '中文名': [col_map_reverse.get(f, f) for f in FINAL_FEATURES],
        '标准化系数': ridge.coef_,
        '|系数|': np.abs(ridge.coef_),
    }).sort_values('|系数|', ascending=False)

    print(f"\n  岭回归系数（按|系数|降序）:")
    for _, row in coef_df.iterrows():
        print(f"    {row['中文名']:16s} 系数={row['标准化系数']:+.6f}")

    coef_df.to_csv(f'{OUTPUT_DIR}/问题2_岭回归系数.csv', index=False, encoding='utf-8-sig')

    # 岭迹图
    _ridge_trace_plot(X_scaled, y, alphas, best_alpha)

    # MLR vs Ridge 系数对比图
    _coef_comparison_plot(coef_df, 'ridge')

    return {
        'alpha': best_alpha,
        'coef_df': coef_df,
        'r2': r2, 'adj_r2': adj_r2, 'rmse': rmse, 'mae': mae,
        'y_pred': y_pred,
    }

def _ridge_trace_plot(X, y, alphas, best_alpha):
    """岭迹图：不同alpha下系数的收缩轨迹"""
    coef_path = []
    for alpha in alphas:
        r = Ridge(alpha=alpha)
        r.fit(X, y)
        coef_path.append(r.coef_)
    coef_path = np.array(coef_path)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(FINAL_FEATURES)))
    for i, (feat, color) in enumerate(zip(FINAL_FEATURES, colors)):
        ax.plot(alphas, coef_path[:, i], color=color, linewidth=2, alpha=0.85,
                label=col_map_reverse.get(feat, feat))
    ax.axvline(x=best_alpha, color='#E74C3C', linestyle='--', linewidth=2,
               alpha=0.8, label=f'最优α={best_alpha:.4f}')
    ax.set_xscale('log')
    ax.set_xlabel(r'正则化参数 $\alpha$ (log尺度)', fontsize=12)
    ax.set_ylabel('标准化回归系数', fontsize=12)
    ax.set_title('岭迹图：正则化强度与系数收缩路径', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题2_岭回归系数路径.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表: {FIGURE_DIR}/问题2_岭回归系数路径.png")

def _coef_comparison_plot(coef_df, suffix):
    """系数重要性排序图"""
    plot_df = coef_df.sort_values('标准化系数', ascending=True)
    colors = ['#E74C3C' if c > 0 else '#3498DB' for c in plot_df['标准化系数']]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(plot_df)), plot_df['标准化系数'], color=colors, alpha=0.85, edgecolor='white')
    ax.axvline(0, color='#2C3E50', linewidth=1)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['中文名'], fontsize=10)
    ax.set_xlabel('标准化回归系数', fontsize=12)
    ax.set_title('岭回归：各变量对血糖的影响方向与强度', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        v = row['标准化系数']
        ax.text(v + 0.01 if v >= 0 else v - 0.06, i, f'{v:+.4f}',
                va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题2_岭回归系数图.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表: {FIGURE_DIR}/问题2_岭回归系数图.png")

# ============================================================
# Step 3: 随机森林回归（非线性模型）
# ============================================================
def step3_rf(df):
    """随机森林回归 — 非线性预测模型"""
    print("\n" + "=" * 60)
    print("Step 3: 随机森林回归 (RF) — 非线性预测模型")
    print("目的: 集成学习捕捉非线性关系与特征交互效应")
    print("=" * 60)

    X = df[FINAL_FEATURES].copy()
    y = df['glucose'].values

    rf = RandomForestRegressor(
        n_estimators=500, max_depth=8, min_samples_leaf=5,
        min_samples_split=10, max_features='sqrt',
        random_state=RANDOM_SEED, n_jobs=-1, oob_score=True
    )
    rf.fit(X, y)
    y_pred = rf.predict(X)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    n, p = len(y), 7
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(f"\n  参数: n_estimators=500, max_depth=8, min_samples_leaf=5")
    print(f"  OOB $R^2$ = {rf.oob_score_:.4f}（袋外无偏估计）")
    print(f"\n  [全样本表现]")
    print(f"  $R^2$ = {r2:.4f}   调整$R^2$ = {adj_r2:.4f}")
    print(f"  RMSE = {rmse:.4f}    MAE = {mae:.4f}")

    # 特征重要性
    imp_df = pd.DataFrame({
        '特征': FINAL_FEATURES,
        '中文名': [col_map_reverse.get(f, f) for f in FINAL_FEATURES],
        '重要性': rf.feature_importances_,
    }).sort_values('重要性', ascending=False)

    print(f"\n  特征重要性（按重要性降序）:")
    for _, row in imp_df.iterrows():
        print(f"    {row['中文名']:16s} 重要性={row['重要性']:.4f}")

    imp_df.to_csv(f'{OUTPUT_DIR}/问题2_随机森林特征重要性.csv', index=False, encoding='utf-8-sig')

    # 特征重要性图
    _rf_importance_plot(imp_df)

    return {
        'oob_r2': rf.oob_score_,
        'imp_df': imp_df,
        'r2': r2, 'adj_r2': adj_r2, 'rmse': rmse, 'mae': mae,
        'y_pred': y_pred,
    }

def _rf_importance_plot(imp_df):
    """随机森林特征重要性排序图"""
    plot_df = imp_df.sort_values('重要性', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(plot_df)))
    ax.barh(range(len(plot_df)), plot_df['重要性'], color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['中文名'], fontsize=10)
    ax.set_xlabel('特征重要性 (Gini重要性)', fontsize=12)
    ax.set_title('随机森林：特征重要性排序', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    for i, v in enumerate(plot_df['重要性']):
        ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题2_随机森林特征重要性.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表: {FIGURE_DIR}/问题2_随机森林特征重要性.png")

# ============================================================
# Step 4: 模型对比与可视化
# ============================================================
def step4_comparison(ridge_result, rf_result, df):
    """对比岭回归（改进后线性模型）vs 随机森林（非线性模型）"""
    print("\n" + "=" * 60)
    print("Step 4: 模型对比 — 岭回归(改进线性) vs 随机森林(非线性)")
    print("=" * 60)

    y = df['glucose'].values

    # 对比表
    comp_df = pd.DataFrame({
        '模型': ['多元线性回归(MLR)', '岭回归(Ridge)[改进]', '随机森林(RF)'],
        '$R^2$': [f"{ridge_result['r2']:.4f}", f"{ridge_result['r2']:.4f}", f"{rf_result['r2']:.4f}"],
        '调整$R^2$': [f"{ridge_result['adj_r2']:.4f}", f"{ridge_result['adj_r2']:.4f}", f"{rf_result['adj_r2']:.4f}"],
        'RMSE': [f"{ridge_result['rmse']:.4f}", f"{ridge_result['rmse']:.4f}", f"{rf_result['rmse']:.4f}"],
        'MAE': [f"{ridge_result['mae']:.4f}", f"{ridge_result['mae']:.4f}", f"{rf_result['mae']:.4f}"],
        '模型角色': ['基础模型', 'MLR+L2改进', '非线性预测'],
    })

    print(f"\n{'='*90}")
    print(f"{'模型递进式对比':^90}")
    print(f"{'='*90}")
    print(comp_df.to_string(index=False))
    print(f"{'='*90}")
    print(f"\n说明: 岭回归是多元线性回归的L2正则化改进，两者为递进关系而非独立方法。")

    comp_df.to_csv(f'{OUTPUT_DIR}/问题2_模型对比.csv', index=False, encoding='utf-8-sig')

    # 确定最优模型（RF vs Ridge改进后）
    best_name = '随机森林(RF)' if rf_result['r2'] >= ridge_result['r2'] else '岭回归(Ridge)'
    best_r2 = max(rf_result['r2'], ridge_result['r2'])
    print(f"\n>>> 结论：{best_name} 表现更优（$R^2$={best_r2:.4f})")

    # ====== 可视化1: 预测值vs真实值（两个子图：Ridge + RF） ======
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    models_data = [
        ('岭回归 (Ridge)\nMLR的L2正则化改进', ridge_result['y_pred'], '#2ECC71'),
        ('随机森林 (RF)\n非线性预测模型', rf_result['y_pred'], '#E74C3C'),
    ]
    for idx, (name, pred, color) in enumerate(models_data):
        ax = axes[idx]
        ax.scatter(y, pred, alpha=0.35, s=10, c=color, edgecolors='white', linewidth=0.3)
        # 用百分位数确定坐标轴范围（排除极端离群点的干扰）
        combined = np.concatenate([y, pred])
        lo = np.percentile(combined, 0.5)
        hi = np.percentile(combined, 99.5)
        lims = [lo - 0.3, hi + 0.3]
        ax.plot(lims, lims, 'k--', alpha=0.6, linewidth=1.5)
        r2 = r2_score(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xticks(np.arange(np.floor(lo), np.ceil(hi) + 0.5, 1.0))
        ax.set_yticks(np.arange(np.floor(lo), np.ceil(hi) + 0.5, 1.0))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_xlabel('真实血糖值 (mmol/L)', fontsize=11)
        ax.set_ylabel('预测血糖值 (mmol/L)', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.text(0.05, 0.95, f'$R^2$={r2:.4f}\nRMSE={rmse:.4f}', transform=ax.transAxes,
                fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题2_模型预测对比图.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表: {FIGURE_DIR}/问题2_模型预测对比图.png")

    # ====== 可视化2: 指标对比柱状图 ======
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_config = [
        ('$R^2$', [ridge_result['r2'], rf_result['r2']], ['#2ECC71', '#E74C3C'], True),
        ('RMSE', [ridge_result['rmse'], rf_result['rmse']], ['#2ECC71', '#E74C3C'], False),
        ('MAE', [ridge_result['mae'], rf_result['mae']], ['#2ECC71', '#E74C3C'], False),
    ]
    for idx, (title, vals, colors, higher_better) in enumerate(metrics_config):
        ax = axes[idx]
        names = ['Ridge\n(改进线性)', 'RF\n(非线性)']
        bars = ax.bar(names, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题2_模型指标对比.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表: {FIGURE_DIR}/问题2_模型指标对比.png")

    # ====== 可视化3: 残差分析 — 小提琴图+散点图（Ridge + RF 合并） ======
    fig, ax = plt.subplots(figsize=(12, 7))
    ridge_residuals = y - ridge_result['y_pred']
    rf_residuals = y - rf_result['y_pred']

    # 剔除极端离群残差（保留0.5%~99.5%范围），聚焦主要分布
    all_resid = np.concatenate([ridge_residuals, rf_residuals])
    lo = np.percentile(all_resid, 0.5)
    hi = np.percentile(all_resid, 99.5)
    ridge_clipped = np.clip(ridge_residuals, lo, hi)
    rf_clipped = np.clip(rf_residuals, lo, hi)

    # 准备数据
    plot_data = []
    for val, label in zip(ridge_clipped, ['岭回归 (Ridge)'] * len(ridge_clipped)):
        plot_data.append({'模型': label, '残差': val})
    for val, label in zip(rf_clipped, ['随机森林 (RF)'] * len(rf_clipped)):
        plot_data.append({'模型': label, '残差': val})
    plot_df = pd.DataFrame(plot_data)

    # 小提琴图
    sns.violinplot(x='模型', y='残差', data=plot_df, ax=ax,
                   palette=['#2ECC71', '#E74C3C'], alpha=0.4,
                   inner=None, linewidth=1.5, width=0.6)
    # 添加散点（抖动）
    for idx, (name, resid, color) in enumerate([
        ('岭回归 (Ridge)', ridge_clipped, '#2ECC71'),
        ('随机森林 (RF)', rf_clipped, '#E74C3C'),
    ]):
        jitter = np.random.normal(0, 0.06, size=len(resid))
        ax.scatter(np.full_like(resid, idx) + jitter, resid,
                   alpha=0.25, s=5, c=color, edgecolors='none', zorder=3)

    # 零线
    ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.6)
    # y轴范围
    ax.set_ylim(lo - 0.3, hi + 0.3)
    ax.set_ylabel('残差 (mmol/L)', fontsize=12)
    ax.set_title('残差分布对比：岭回归 vs 随机森林', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.2)

    # 标注均值和标准差
    for idx, (name, resid, color) in enumerate([
        ('岭回归 (Ridge)', ridge_residuals, '#2ECC71'),
        ('随机森林 (RF)', rf_residuals, '#E74C3C'),
    ]):
        mean_r = resid.mean()
        std_r = resid.std()
        ax.text(idx, ax.get_ylim()[1] * 0.92,
                f'均值={mean_r:.4f}\n标准差={std_r:.4f}',
                ha='center', va='top', fontsize=10, color=color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题2_残差分析图.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表: {FIGURE_DIR}/问题2_残差分析图.png")

    return comp_df, best_name, best_r2

# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("问题2：血糖预测建模")
    print("方案: 问题2思路.pdf")
    print("流程: MLR(基础) → Ridge(L2改进) → RF(非线性) → 对比")
    print("注意: 基于附件1全部数据整体建模，不使用附件2")
    print("=" * 70)
    print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = datetime.datetime.now()

    # Step 0: 数据加载
    df = load_data()

    # 提取特征和目标
    X = df[FINAL_FEATURES].copy()
    y = df['glucose'].values

    print(f"\n样本总数: {len(df)}（全部用于建模，不划分训练/测试集）")
    print(f"特征变量 ({len(FINAL_FEATURES)} 个):")
    for f in FINAL_FEATURES:
        print(f"  {f:20s} ({col_map_reverse.get(f, f)})")
    print(f"血糖均值: {y.mean():.4f}, 标准差: {y.std():.4f}")

    # Step 1: 多元线性回归（基础模型）
    mlr_result = step1_mlr(df)

    # Step 2: 岭回归（MLR的改进）
    ridge_result = step2_ridge(df)

    # Step 3: 随机森林回归（非线性模型）
    rf_result = step3_rf(df)

    # Step 4: 模型对比
    comp_df, best_name, best_r2 = step4_comparison(ridge_result, rf_result, df)

    # 最终输出
    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    print(f"\n{'='*70}")
    print(f"问题2分析完成！总用时: {elapsed:.1f}秒")
    print(f"{'='*70}")
    print(f"\n最终结论:")
    print(f"  递进路径: 多元线性回归(基准) → 岭回归(L2改进) → 随机森林(非线性)")
    print(f"  最优模型: {best_name}（$R^2$={best_r2:.4f})")
    print(f"\n  模型定位:")
    print(f"    - 多元线性回归: 基础基准模型，提供系数解释")
    print(f"    - 岭回归:       MLR的正则化改进，处理共线性")
    print(f"    - 随机森林:     非线性预测模型，捕捉复杂关系")
    print(f"\n输出文件:")
    print(f"  - 图表: {FIGURE_DIR}/")
    print(f"  - CSV:  {OUTPUT_DIR}/")
    print(f"完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
