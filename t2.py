# -*- coding: utf-8 -*-
"""
t2.py — 问题2：多元线性回归 vs 岭回归 vs 随机森林 血糖预测模型对比
方案: 问题2思路.pdf
方法1: 多元线性回归（MLR）+ 岭回归（Ridge Regression）
方法2: 随机森林回归（Random Forest Regression）
流程: 数据预处理 → 方法1(MLR+Ridge) → 方法2(RF) → RMSE/MAE/R²对比
输出:
  figures/问题2_*.png — 图表
  output/问题2_*.csv  — 结果表
  output/问题2_答案与分析报告.md — 分析报告
"""

import os, sys, warnings, datetime
warnings.filterwarnings('ignore', message='Font.*glyph')  # 静默字体缺失Unicode减号的警告（不影响输出图片）
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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

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
# Step 0: 数据加载与预处理
# ============================================================
def load_data():
    """加载数据，使用与问题1相同的预处理流程，返回df1和df2"""
    print("=" * 60)
    print("Step 0: 数据加载与预处理")
    print("   - MICE多重插补（advanced_preprocessing）")
    print("   - 删除日期衍生变量")
    print("=" * 60)

    df1, df2 = preprocess_with_advanced_imputation(strategy='mice', mice_iter=20)

    for col in ['exam_year', 'exam_month']:
        if col in df1.columns:
            df1.drop(columns=[col], inplace=True)
        if col in df2.columns:
            df2.drop(columns=[col], inplace=True)

    print(f"\n预处理完成: 附件1 {df1.shape}, 附件2 {df2.shape}")
    print(f"附件1血糖均值={df1['glucose'].mean():.4f}")
    return df1, df2

# ============================================================
# Step 1: 数据准备与划分
# ============================================================
def prepare_data(df):
    """准备训练/测试数据"""
    print("\n" + "=" * 60)
    print("Step 1: 数据准备与划分")
    print("=" * 60)

    X = df[FINAL_FEATURES].copy()
    y = df['glucose'].values

    print(f"\n特征变量 ({len(FINAL_FEATURES)} 个):")
    for f in FINAL_FEATURES:
        print(f"  {f:20s} ({col_map_reverse.get(f, f)})")
    print(f"\n样本总数: {len(df)}")
    print(f"血糖均值: {y.mean():.4f}, 标准差: {y.std():.4f}")

    return X, y

# ============================================================
# Step 2: 多元线性回归（方法1a）
# ============================================================
def step2_mlr(X_train, y_train, X_test, y_test):
    """多元线性回归"""
    print("\n" + "=" * 60)
    print("Step 2: 方法1a — 多元线性回归 (MLR)")
    print("目的: 建立基础线性预测模型，提供可解释回归系数")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlr = LinearRegression()
    mlr.fit(X_train_scaled, y_train)

    y_train_pred = mlr.predict(X_train_scaled)
    y_test_pred = mlr.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f"\n  [训练集] R²={train_r2:.4f}  RMSE={train_rmse:.4f}  MAE={train_mae:.4f}")
    print(f"  [测试集] R²={test_r2:.4f}  RMSE={test_rmse:.4f}  MAE={test_mae:.4f}")

    coef_df = pd.DataFrame({
        '特征': FINAL_FEATURES,
        '中文名': [col_map_reverse.get(f, f) for f in FINAL_FEATURES],
        '系数': mlr.coef_,
        '|系数|': np.abs(mlr.coef_),
    }).sort_values('|系数|', ascending=False)

    print(f"\n  回归系数（按|系数|降序）:")
    for _, row in coef_df.iterrows():
        print(f"    {row['中文名']:16s} 系数={row['系数']:+.6f}")

    return {
        'model': mlr, 'scaler': scaler,
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'test_mae': test_mae,
        'coef_df': coef_df,
        'y_pred': y_test_pred,
    }

# ============================================================
# Step 3: 岭回归（方法1b）
# ============================================================
def step3_ridge(X_train, y_train, X_test, y_test):
    """岭回归 — 处理共线性"""
    print("\n" + "=" * 60)
    print("Step 3: 方法1b — 岭回归 (Ridge Regression)")
    print("目的: L2正则化处理多重共线性，优化线性预测稳定性")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 交叉验证选最优alpha
    alphas = np.logspace(-3, 3, 100)
    ridge_cv = RidgeCV(alphas=alphas, scoring='r2', cv=10)
    ridge_cv.fit(X_train_scaled, y_train)
    best_alpha = ridge_cv.alpha_

    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_train_scaled, y_train)

    y_train_pred = ridge.predict(X_train_scaled)
    y_test_pred = ridge.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_r2 = cross_val_score(Ridge(alpha=best_alpha), X_train_scaled, y_train, cv=kf, scoring='r2')

    print(f"\n  最优alpha: {best_alpha:.6f} ({alphas[0]:.6f} ~ {alphas[-1]:.1f}, 共{len(alphas)}个候选)")
    print(f"  [训练集] R²={train_r2:.4f}  RMSE={train_rmse:.4f}  MAE={train_mae:.4f}")
    print(f"  [测试集] R²={test_r2:.4f}  RMSE={test_rmse:.4f}  MAE={test_mae:.4f}")
    print(f"  [5折CV] R²={cv_r2.mean():.4f} ± {cv_r2.std():.4f}  各折:{np.round(cv_r2, 4)}")

    coef_df = pd.DataFrame({
        '特征': FINAL_FEATURES,
        '中文名': [col_map_reverse.get(f, f) for f in FINAL_FEATURES],
        '系数': ridge.coef_,
        '|系数|': np.abs(ridge.coef_),
    }).sort_values('|系数|', ascending=False)

    print(f"\n  岭回归系数（按|系数|降序）:")
    for _, row in coef_df.iterrows():
        print(f"    {row['中文名']:16s} 系数={row['系数']:+.6f}")

    coef_df.to_csv(f'{OUTPUT_DIR}/问题2_岭回归系数.csv', index=False, encoding='utf-8-sig')

    # 岭迹图
    _plot_ridge_trace(X_train_scaled, y_train, alphas, best_alpha)

    return {
        'model': ridge, 'scaler': scaler, 'best_alpha': best_alpha,
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'test_mae': test_mae,
        'cv_r2_mean': cv_r2.mean(), 'cv_r2_std': cv_r2.std(),
        'cv_r2_folds': cv_r2,
        'coef_df': coef_df,
        'y_pred': y_test_pred,
    }

def _plot_ridge_trace(X, y, alphas, best_alpha):
    """岭迹图"""
    coef_path = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X, y)
        coef_path.append(ridge.coef_)
    coef_path = np.array(coef_path)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(FINAL_FEATURES)))
    for i, (feat, color) in enumerate(zip(FINAL_FEATURES, colors)):
        ax.plot(alphas, coef_path[:, i], color=color,
                linewidth=2, alpha=0.85,
                label=col_map_reverse.get(feat, feat))
    ax.axvline(x=best_alpha, color='#E74C3C', linestyle='--',
               linewidth=2, alpha=0.8, label=f'最优α={best_alpha:.4f}')
    ax.set_xscale('log')
    ax.set_xlabel('正则化参数 α (log尺度)', fontsize=12)
    ax.set_ylabel('回归系数', fontsize=12)
    ax.set_title('岭回归：系数随正则化强度的变化路径', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题2_岭回归系数路径.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表: {FIGURE_DIR}/问题2_岭回归系数路径.png")

# ============================================================
# Step 4: 随机森林回归（方法2）
# ============================================================
def step4_rf(X_train, y_train, X_test, y_test):
    """随机森林回归"""
    print("\n" + "=" * 60)
    print("Step 4: 方法2 — 随机森林回归 (Random Forest)")
    print("目的: 集成学习捕捉非线性关系与特征交互效应")
    print("=" * 60)

    rf = RandomForestRegressor(
        n_estimators=500, max_depth=8, min_samples_leaf=5,
        min_samples_split=10, max_features='sqrt',
        random_state=RANDOM_SEED, n_jobs=-1,
        oob_score=True
    )
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_r2 = cross_val_score(
        RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=5,
                              min_samples_split=10, max_features='sqrt',
                              random_state=RANDOM_SEED, n_jobs=-1),
        X_train, y_train, cv=kf, scoring='r2'
    )

    print(f"\n  参数: n_estimators=500, max_depth=8, min_samples_leaf=5")
    print(f"  OOB R² = {rf.oob_score_:.4f}")
    print(f"  [训练集] R²={train_r2:.4f}  RMSE={train_rmse:.4f}  MAE={train_mae:.4f}")
    print(f"  [测试集] R²={test_r2:.4f}  RMSE={test_rmse:.4f}  MAE={test_mae:.4f}")
    print(f"  [5折CV] R²={cv_r2.mean():.4f} ± {cv_r2.std():.4f}  各折:{np.round(cv_r2, 4)}")

    imp_df = pd.DataFrame({
        '特征': FINAL_FEATURES,
        '中文名': [col_map_reverse.get(f, f) for f in FINAL_FEATURES],
        '重要性': rf.feature_importances_,
    }).sort_values('重要性', ascending=False)

    print(f"\n  特征重要性（按重要性降序）:")
    for _, row in imp_df.iterrows():
        print(f"    {row['中文名']:16s} 重要性={row['重要性']:.4f}")

    imp_df.to_csv(f'{OUTPUT_DIR}/问题2_随机森林特征重要性.csv', index=False, encoding='utf-8-sig')

    return {
        'model': rf,
        'oob_r2': rf.oob_score_,
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'test_mae': test_mae,
        'cv_r2_mean': cv_r2.mean(), 'cv_r2_std': cv_r2.std(),
        'cv_r2_folds': cv_r2,
        'imp_df': imp_df,
        'y_pred': y_test_pred,
    }

# ============================================================
# Step 5: 模型对比与可视化
# ============================================================
def step5_comparison(mlr_result, ridge_result, rf_result, X_test, y_test):
    """模型对比，输出综合结果"""
    print("\n" + "=" * 60)
    print("Step 5: 模型对比与评估")
    print("=" * 60)

    # ====== 对比表 ======
    comp_df = pd.DataFrame({
        '模型': ['多元线性回归(MLR)', '岭回归(Ridge)', '随机森林(RF)'],
        '训练R²': [f"{mlr_result['train_r2']:.4f}", f"{ridge_result['train_r2']:.4f}", f"{rf_result['train_r2']:.4f}"],
        '测试R²': [f"{mlr_result['test_r2']:.4f}", f"{ridge_result['test_r2']:.4f}", f"{rf_result['test_r2']:.4f}"],
        '测试RMSE': [f"{mlr_result['test_rmse']:.4f}", f"{ridge_result['test_rmse']:.4f}", f"{rf_result['test_rmse']:.4f}"],
        '测试MAE': [f"{mlr_result['test_mae']:.4f}", f"{ridge_result['test_mae']:.4f}", f"{rf_result['test_mae']:.4f}"],
        '5折CV-R²': [f"—", f"{ridge_result['cv_r2_mean']:.4f}±{ridge_result['cv_r2_std']:.4f}",
                      f"{rf_result['cv_r2_mean']:.4f}±{rf_result['cv_r2_std']:.4f}"],
    })

    print(f"\n{'='*90}")
    print(f"{'模型对比汇总':^90}")
    print(f"{'='*90}")
    print(comp_df.to_string(index=False))
    print(f"{'='*90}")

    comp_df.to_csv(f'{OUTPUT_DIR}/问题2_模型对比.csv', index=False, encoding='utf-8-sig')

    # ====== 确定最优模型 ======
    results = [
        ('多元线性回归', mlr_result['test_r2']),
        ('岭回归', ridge_result['test_r2']),
        ('随机森林', rf_result['test_r2']),
    ]
    best_name, best_r2 = max(results, key=lambda x: x[1])
    best_model = {
        '多元线性回归': mlr_result,
        '岭回归': ridge_result,
        '随机森林': rf_result,
    }[best_name]

    print(f"\n>>> 结论：{best_name} 表现最优（测试R²={best_r2:.4f}）")

    # ====== 可视化1: 预测值vs真实值散点图 ======
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    models_data = [
        ('多元线性回归 (MLR)', mlr_result['y_pred'], '#3498DB'),
        ('岭回归 (Ridge)', ridge_result['y_pred'], '#2ECC71'),
        ('随机森林 (RF)', rf_result['y_pred'], '#E74C3C'),
    ]
    for idx, (name, pred, color) in enumerate(models_data):
        ax = axes[idx]
        ax.scatter(y_test, pred, alpha=0.4, s=12, c=color, edgecolors='white', linewidth=0.3)
        lims = [min(y_test.min(), pred.min()) - 0.5, max(y_test.max(), pred.max()) + 0.5]
        ax.plot(lims, lims, 'k--', alpha=0.6, linewidth=1.5)
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('真实血糖值 (mmol/L)', fontsize=11)
        ax.set_ylabel('预测血糖值 (mmol/L)', fontsize=11)
        ax.set_title(f'{name}', fontsize=13, fontweight='bold')
        ax.text(0.05, 0.95, f'R²={r2:.4f}\nRMSE={rmse:.4f}', transform=ax.transAxes,
                fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题2_模型预测对比图.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表: {FIGURE_DIR}/问题2_模型预测对比图.png")

    # ====== 可视化2: 模型指标对比柱状图 ======
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_config = [
        ('测试R²', [mlr_result['test_r2'], ridge_result['test_r2'], rf_result['test_r2']],
         ['#3498DB', '#2ECC71', '#E74C3C'], True),
        ('测试RMSE', [mlr_result['test_rmse'], ridge_result['test_rmse'], rf_result['test_rmse']],
         ['#3498DB', '#2ECC71', '#E74C3C'], False),
        ('测试MAE', [mlr_result['test_mae'], ridge_result['test_mae'], rf_result['test_mae']],
         ['#3498DB', '#2ECC71', '#E74C3C'], False),
    ]
    for idx, (title, vals, colors, higher_better) in enumerate(metrics_config):
        ax = axes[idx]
        model_names = ['MLR', 'Ridge', 'RF']
        bars = ax.bar(model_names, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题2_模型指标对比.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表: {FIGURE_DIR}/问题2_模型指标对比.png")

    # ====== 可视化3: 残差分析图 ======
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, (name, pred, color) in enumerate(models_data):
        residuals = y_test - pred
        # 残差分布直方图
        ax = axes[0, idx]
        ax.hist(residuals, bins=40, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('残差', fontsize=10); ax.set_ylabel('频数', fontsize=10)
        ax.set_title(f'{name} 残差分布', fontsize=12, fontweight='bold')
        ax.text(0.95, 0.95, f'均值={residuals.mean():.4f}\n标准差={residuals.std():.4f}',
                transform=ax.transAxes, fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.grid(True, alpha=0.2)
        # 残差vs预测值
        ax = axes[1, idx]
        ax.scatter(pred, residuals, alpha=0.4, s=12, c=color, edgecolors='white')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('预测值', fontsize=10); ax.set_ylabel('残差', fontsize=10)
        ax.set_title(f'{name} 残差vs预测值', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/问题2_残差分析图.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表: {FIGURE_DIR}/问题2_残差分析图.png")

    return comp_df, best_name, best_r2

# ============================================================
# Step 6: 附件2血糖预测
# ============================================================
def step6_predict_attachment2(ridge_model_dict, rf_model_dict, df2):
    """对附件2进行血糖预测"""
    print("\n" + "=" * 60)
    print("Step 6: 附件2血糖预测")
    print("=" * 60)

    # 检查各模型需要的特征是否都在df2中
    available_features = [f for f in FINAL_FEATURES if f in df2.columns]
    if len(available_features) < len(FINAL_FEATURES):
        missing = [f for f in FINAL_FEATURES if f not in df2.columns]
        print(f"  ⚠ 附件2缺失特征: {missing}")

    X2 = df2[available_features].copy()

    # 标准化后预测（岭回归）
    X2_scaled = ridge_model_dict['scaler'].transform(X2)
    ridge_pred = ridge_model_dict['model'].predict(X2_scaled)

    # 随机森林直接预测
    rf_pred = rf_model_dict['model'].predict(X2)

    pred_df = pd.DataFrame({
        '岭回归预测值': ridge_pred,
        '随机森林预测值': rf_pred,
        '两种方法均值': (ridge_pred + rf_pred) / 2,
    })

    pred_df.to_csv(f'{OUTPUT_DIR}/问题2_附件2预测结果.csv', index=False, encoding='utf-8-sig')

    print(f"\n  附件2预测统计:")
    print(f"    岭回归: 均值={ridge_pred.mean():.4f}, 标准差={ridge_pred.std():.4f}")
    print(f"    随机森林: 均值={rf_pred.mean():.4f}, 标准差={rf_pred.std():.4f}")

    return pred_df

# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("问题2：血糖预测模型对比")
    print("方案: 问题2思路.pdf")
    print("方法1: 多元线性回归 + 岭回归（统计线性模型）")
    print("方法2: 随机森林回归（非线性预测模型）")
    print("=" * 70)
    print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = datetime.datetime.now()

    # Step 0: 数据加载
    df1, df2 = load_data()

    # Step 1: 数据准备与划分
    X, y = prepare_data(df1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"\n  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")

    # Step 2: 多元线性回归
    mlr_result = step2_mlr(X_train, y_train, X_test, y_test)

    # Step 3: 岭回归
    ridge_result = step3_ridge(X_train, y_train, X_test, y_test)

    # Step 4: 随机森林回归
    rf_result = step4_rf(X_train, y_train, X_test, y_test)

    # Step 5: 模型对比
    comp_df, best_name, best_r2 = step5_comparison(
        mlr_result, ridge_result, rf_result, X_test, y_test
    )

    # Step 6: 附件2预测
    pred_df = step6_predict_attachment2(ridge_result, rf_result, df2)

    # ====== 最终结果输出 ======
    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    print(f"\n{'='*70}")
    print(f"问题2分析完成！总用时: {elapsed:.1f}秒")
    print(f"{'='*70}")
    print(f"\n最终结论:")
    print(f"  对比方法: 多元线性回归 / 岭回归 / 随机森林")
    print(f"  最优模型: {best_name}（测试R²={best_r2:.4f}）")
    print(f"\n  推荐方案:")
    print(f"    - 线性解释用: 多元线性回归（系数可解释性强）")
    print(f"    - 预测准确用: 随机森林回归（R²最高）")
    print(f"    - 两者互补: 岭回归作为折中方案")
    print(f"\n输出文件:")
    print(f"  - 图表: {FIGURE_DIR}/")
    print(f"  - CSV: {OUTPUT_DIR}/")
    print(f"完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return mlr_result, ridge_result, rf_result

if __name__ == '__main__':
    main()
