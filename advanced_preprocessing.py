# -*- coding: utf-8 -*-
"""
============================================================================
高级数据预处理模块
针对糖尿病数据中的大幅度缺失值问题（乙肝指标缺失76%，生化指标缺失~21%）
实现了多种缺失值插补算法：KNN、MICE、中位数插补
============================================================================

使用方法：
    from advanced_preprocessing import preprocess_with_advanced_imputation
    
    df1, df2 = preprocess_with_advanced_imputation(
        file1='附件1.csv', file2='附件2.csv',
        strategy='mice'       # 可选: 'median', 'knn', 'mice'
    )

库依赖：
    pandas, numpy, scipy, sklearn
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
try:
    from sklearn.impute import KNNImputer
    # IterativeImputer在不同sklearn版本中位置可能不同
    try:
        from sklearn.impute import IterativeImputer
    except ImportError:
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
        except ImportError:
            # sklearn >= 1.9 可能移除了IterativeImputer
            IterativeImputer = None
    _SKLEARN_AVAILABLE = True
except ImportError as e:
    _SKLEARN_AVAILABLE = False
    print(f"⚠ sklearn导入失败: {e}")
    print("  回退到中位数插补")

warnings.filterwarnings('ignore')

# ============================================================================
# 配置常量
# ============================================================================

COLUMN_MAP = {
    'id':                     'id',
    '年龄':                    'age',
    '性别':                    'gender',
    '体检日期':                 'exam_date',
    '*r-谷氨酰基转换酶':       'GGT',
    '*丙氨酸氨基转换酶':        'ALT',
    '*天门冬氨酸氨基转换酶':    'AST',
    '*总蛋白':                 'TP',
    '*球蛋白':                 'GLOB',
    '*碱性磷酸酶':              'ALP',
    '白蛋白':                  'ALB',
    '白球比例':                'AGR',
    '尿素':                    'BUN',
    '肌酐':                    'Cr',
    '尿酸':                    'UA',
    '总胆固醇':                'TC',
    '甘油三酯':                'TG',
    '高密度脂蛋白胆固醇':      'HDL_C',
    '低密度脂蛋白胆固醇':      'LDL_C',
    '中性粒细胞%':             'NEUT_pct',
    '淋巴细胞%':               'LYMPH_pct',
    '单核细胞%':               'MONO_pct',
    '嗜酸细胞%':               'EO_pct',
    '嗜碱细胞%':               'BASO_pct',
    '白细胞计数':              'WBC',
    '红细胞计数':              'RBC',
    '血红蛋白':                'HGB',
    '红细胞压积':              'HCT',
    '红细胞平均体积':          'MCV',
    '红细胞平均血红蛋白量':     'MCH',
    '红细胞平均血红蛋白浓度':  'MCHC',
    '红细胞体积分布宽度':      'RDW',
    '血小板计数':              'PLT',
    '血小板平均体积':          'MPV',
    '血小板体积分布宽度':      'PDW',
    '血小板比积':              'PCT',
    '血糖':                    'glucose',
    '乙肝表面抗原':            'HBsAg',
    '乙肝表面抗体':            'HBsAb',
    '乙肝e抗原':               'HBeAg',
    '乙肝e抗体':               'HBeAb',
    '乙肝核心抗体':            'HBcAb',
}

HEPATITIS_COLS = ['HBsAg', 'HBsAb', 'HBeAg', 'HBeAb', 'HBcAb']


# ============================================================================
# 第1步：基础加载和清洗
# ============================================================================

def load_clean_data(file1='附件1.csv', file2='附件2.csv'):
    """加载两个附件，执行列名映射、日期处理、性别编码"""
    print("=" * 60)
    print("基础数据加载与清洗")
    print("=" * 60)
    
    df1 = pd.read_csv(file1, encoding='gbk')
    df2 = pd.read_csv(file2, encoding='gbk')
    print(f"附件1: {df1.shape}, 附件2: {df2.shape}")
    
    # 重命名列
    rename1 = {k: v for k, v in COLUMN_MAP.items() if k in df1.columns}
    rename2 = {k: v for k, v in COLUMN_MAP.items() if k in df2.columns}
    df1 = df1.rename(columns=rename1)
    df2 = df2.rename(columns=rename2)
    
    # 处理日期
    for df in [df1, df2]:
        if 'exam_date' in df.columns:
            df['exam_date'] = pd.to_datetime(df['exam_date'], format='%d/%m/%Y', errors='coerce')
            df['exam_year'] = df['exam_date'].dt.year
            df['exam_month'] = df['exam_date'].dt.month
    
    # 处理性别
    for df in [df1, df2]:
        if 'gender' in df.columns:
            df['gender_male'] = df['gender'].apply(
                lambda x: 1 if str(x).strip() in ['男', 'M'] 
                else (0 if str(x).strip() in ['女', 'F'] else np.nan)
            )
            # 显式转为标量，消除类型警告
            mode_val = float(df['gender_male'].mode().iloc[0])
            df['gender_male'].fillna(mode_val, inplace=True)
    
    # 剔除无关列
    for df in [df1, df2]:
        for col in ['id', 'gender', 'exam_date']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
    
    print(f"清洗后: 附件1 {df1.shape}, 附件2 {df2.shape}")
    return df1, df2


# ============================================================================
# 第2步：缺失值统计
# ============================================================================

def print_missing_report(df, name='数据'):
    """打印缺失值统计报告"""
    missing = df.isnull().sum()
    missing_pct = missing[missing > 0].sort_values(ascending=False)
    if len(missing_pct) == 0:
        print(f"{name}: 无缺失值")
        return
    print(f"\n{name}缺失情况:")
    print(f"{'列名':<22} {'缺失数':<8} {'缺失率':<8}")
    print("-" * 40)
    for col in missing_pct.index:
        pct = missing_pct[col] / len(df) * 100
        print(f"{col:<22} {missing_pct[col]:<8} {pct:<7.2f}%")


# ============================================================================
# 第3步：中位数插补（基线方法）
# ============================================================================

def median_imputation(df1, df2):
    """
    中位数插补 - 基线方法
    
    原理：用各列中位数填充缺失值。
    适用：缺失率<5%的场景。简单快速，但会压低方差。
    """
    print("\n>>> 执行中位数插补...")
    for name, df in [('附件1', df1), ('附件2', df2)]:
        for col in df.columns:
            if col in ['glucose', 'exam_year', 'exam_month']:
                continue
            na = df[col].isna().sum()
            if na > 0:
                df[col] = df[col].fillna(df[col].median())
                if na > 100:
                    print(f"  {name}.{col}: {na}个缺失 -> 中位数{df[col].median():.4f}")
    return df1, df2


# ============================================================================
# 第4步：KNN插补
# ============================================================================

def knn_imputation(df1, df2, n_neighbors=5, missing_threshold=0.5):
    """KNN插补 - 如果sklearn不可用则回退到中位数插补"""
    if not _SKLEARN_AVAILABLE:
        print("\n>>> sklearn不可用，KNN回退到中位数插补")
        return median_imputation(df1, df2)
    
    print(f"\n>>> 执行KNN插补(K={n_neighbors})...")
    
    for name, df in [('附件1', df1), ('附件2', df2)]:
        has_glucose = 'glucose' in df.columns
        glucose_series = df['glucose'].copy() if has_glucose else None
        
        impute_cols = [c for c in df.columns if c not in ['glucose', 'exam_year', 'exam_month']]
        
        high_missing = [c for c in impute_cols if df[c].isna().sum() / len(df) > missing_threshold]
        if high_missing:
            print(f"  {name}: 缺失率>{missing_threshold*100:.0f}%的列: {high_missing}")
            impute_cols = [c for c in impute_cols if c not in high_missing]
        
        impute_data = df[impute_cols].copy()
        before_na = impute_data.isnull().sum().sum()
        
        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')  # type: ignore
        imputed_array = imputer.fit_transform(impute_data)
        imputed_df = pd.DataFrame(imputed_array, columns=impute_cols, index=df.index)
        
        for col in impute_cols:
            df[col] = imputed_df[col]
        
        if has_glucose:
            df['glucose'] = glucose_series
        
        after_na = df[impute_cols].isnull().sum().sum()
        print(f"  {name}: KNN完成 ({before_na}->{after_na} 个NaN)")
    
    return df1, df2


# ============================================================================
# 第5步：MICE链式方程多重插补
# ============================================================================

def mice_imputation(df1, df2, max_iter=20, random_state=2026, missing_threshold=0.5):
    """MICE插补 - 如果sklearn不可用或IterativeImputer不存在则回退到KNN/中位数"""
    if not _SKLEARN_AVAILABLE or IterativeImputer is None:
        if IterativeImputer is None:
            print("\n>>> IterativeImputer在当前sklearn版本中不可用，回退到KNN插补")
            return knn_imputation(df1, df2, n_neighbors=5, missing_threshold=missing_threshold)
        print("\n>>> sklearn不可用，MICE回退到中位数插补")
        return median_imputation(df1, df2)
    
    print(f"\n>>> 执行MICE插补(max_iter={max_iter})...")
    
    for name, df in [('附件1', df1), ('附件2', df2)]:
        has_glucose = 'glucose' in df.columns
        glucose_series = df['glucose'].copy() if has_glucose else None
        
        impute_cols = [c for c in df.columns if c not in ['glucose', 'exam_year', 'exam_month']]
        
        # 极高缺失率列用中位数填充（不参与迭代建模）
        high_missing = [c for c in impute_cols if df[c].isna().sum() / len(df) > missing_threshold]
        if high_missing:
            print(f"  {name}: 缺失率>{missing_threshold*100:.0f}%直接中位数填充: {high_missing}")
            for c in high_missing:
                df[c] = df[c].fillna(df[c].median())
            impute_cols = [c for c in impute_cols if c not in high_missing]
        
        impute_data = df[impute_cols].copy()
        before_na = impute_data.isnull().sum().sum()
        
        imputer = IterativeImputer(  # type: ignore
            max_iter=max_iter,
            random_state=random_state,
            imputation_order='ascending',
            initial_strategy='median',
            
            
        )
        
        imputed_array = imputer.fit_transform(impute_data)
        imputed_df = pd.DataFrame(imputed_array, columns=impute_cols, index=df.index)
        
        for col in impute_cols:
            df[col] = imputed_df[col]
        
        if has_glucose:
            df['glucose'] = glucose_series
        
        after_na = df[impute_cols].isnull().sum().sum()
        n_iter = getattr(imputer, 'n_iter_', '?')
        print(f"  {name}: MICE完成 (迭代{n_iter}次, {before_na}->{after_na}个NaN)")
    
    return df1, df2


# ============================================================================
# 第6步：统一入口
# ============================================================================

def preprocess_with_advanced_imputation(
    file1='附件1.csv',
    file2='附件2.csv',
    strategy='mice',
    knn_k=5,
    mice_iter=20,
    missing_threshold=0.5,
):
    """
    高级数据预处理统一入口。
    
    参数：
        file1, file2      : CSV文件路径
        strategy           : 插补策略 ('median', 'knn', 'mice')
        knn_k              : KNN的K值
        mice_iter          : MICE最大迭代次数
        missing_threshold  : 大于此缺失率的列不参与插补建模
    
    返回：
        df1, df2: 插补完成的数据框
    """
    print("\n" + "=" * 60)
    print("  高级数据预处理模块")
    strategy_names = {'median': '中位数插补', 'knn': f'KNN(k={knn_k})', 'mice': f'MICE(iter={mice_iter})'}
    print("  缺失值处理策略: " + strategy_names.get(strategy, strategy))
    print("=" * 60)
    
    # Step 1
    df1, df2 = load_clean_data(file1, file2)
    
    # Step 2: 缺失报告
    print("\n--- 插补前缺失诊断 ---")
    print_missing_report(df1, '附件1')
    print_missing_report(df2, '附件2')
    
    # Step 3: 添加乙肝缺失指示变量
    for df in [df1, df2]:
        for col in HEPATITIS_COLS:
            if col in df.columns:
                df[f'{col}_missing'] = df[col].isna().astype(int)
    
    # Step 4: 执行插补
    if strategy == 'median':
        df1, df2 = median_imputation(df1, df2)
    elif strategy == 'knn':
        df1, df2 = knn_imputation(df1, df2, n_neighbors=knn_k, missing_threshold=missing_threshold)
    elif strategy == 'mice':
        df1, df2 = mice_imputation(df1, df2, max_iter=mice_iter, random_state=2026, missing_threshold=missing_threshold)
    else:
        raise ValueError(f"未知策略: {strategy}")
    
    # Step 5: 数值类型检查
    print("\n--- 最终检查 ---")
    for name, df in [('附件1', df1), ('附件2', df2)]:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # type: ignore
            df[col] = df[col].astype(float)  # type: ignore
        remaining = df.isnull().sum().sum()
        print(f"  {name}: {df.shape}, NaN残留: {remaining}")
    
    return df1, df2
