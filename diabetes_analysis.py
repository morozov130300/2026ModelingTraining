# -*- coding: utf-8 -*-
"""
============================================================================
2026年武汉理工大学数学建模训练 - C题：糖尿病风险预测
传统统计模型分析代码（不使用机器学习）
============================================================================

库版本要求：
    pandas==1.5.0
    numpy==1.23.0
    scipy==1.9.0
    statsmodels==0.13.0
    matplotlib==3.6.0
    seaborn==0.12.0

数据文件：
    附件1.csv - 有血糖值的检测数据（5905行 x 42列）
    附件2.csv - 无血糖值的检测数据（141行 x 41列）

问题对应：
    - 问题1: 特征筛选 → problem1_feature_selection()
    - 问题2: 血糖预测 → problem2_glucose_prediction()
    - 问题3: 风险评估 → problem3_risk_assessment()
    - 问题4: 附件2预测 → problem4_predict_attachment2()

作者：数学建模团队 - 代码实现
日期：2026年
============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
# 兼容不同版本的scipy（trapz在1.11+中被重命名为trapezoid）
try:
    from scipy.integrate import trapz
except ImportError:
    from scipy.integrate import trapezoid as trapz
import matplotlib
matplotlib.use('Agg')  # 非交互模式，用于服务器环境
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, normal_ad
from statsmodels.stats.stattools import durbin_watson
import datetime

# 高级数据预处理模块（缺失值插补：MICE/KNN/中位数）
from advanced_preprocessing import preprocess_with_advanced_imputation

# 忽略警告
warnings.filterwarnings('ignore')

# ============================================================================
# 全局配置
# ============================================================================

# 图片输出目录
FIGURE_DIR = './figures'
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

# 结果输出目录
OUTPUT_DIR = './output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 糖尿病诊断阈值（题目标准：血糖值 > 6.7 mmol/L）
DIABETES_THRESHOLD = 6.7

# 随机种子，保证可复现
RANDOM_SEED = 2026
np.random.seed(RANDOM_SEED)

# 列名映射：中文 → 英文（用于statsmodels公式API）
COLUMN_MAP = {
    'id':                           'id',
    '年龄':                          'age',
    '性别':                          'gender',
    '体检日期':                       'exam_date',
    '*r-谷氨酰基转换酶':             'GGT',
    '*丙氨酸氨基转换酶':              'ALT',
    '*天门冬氨酸氨基转换酶':          'AST',
    '*总蛋白':                       'TP',
    '*球蛋白':                       'GLOB',
    '*碱性磷酸酶':                    'ALP',
    '白蛋白':                        'ALB',
    '白球比例':                      'AGR',
    '尿素':                          'BUN',
    '肌酐':                          'Cr',
    '尿酸':                          'UA',
    '总胆固醇':                      'TC',
    '甘油三酯':                      'TG',
    '高密度脂蛋白胆固醇':            'HDL_C',
    '低密度脂蛋白胆固醇':            'LDL_C',
    '中性粒细胞%':                   'NEUT_pct',
    '淋巴细胞%':                     'LYMPH_pct',
    '单核细胞%':                     'MONO_pct',
    '嗜酸细胞%':                     'EO_pct',
    '嗜碱细胞%':                     'BASO_pct',
    '白细胞计数':                    'WBC',
    '红细胞计数':                    'RBC',
    '血红蛋白':                      'HGB',
    '红细胞压积':                    'HCT',
    '红细胞平均体积':                'MCV',
    '红细胞平均血红蛋白量':           'MCH',
    '红细胞平均血红蛋白浓度':        'MCHC',
    '红细胞体积分布宽度':            'RDW',
    '血小板计数':                    'PLT',
    '血小板平均体积':                'MPV',
    '血小板体积分布宽度':            'PDW',
    '血小板比积':                    'PCT',
    '血糖':                          'glucose',
    '乙肝表面抗原':                  'HBsAg',
    '乙肝表面抗体':                  'HBsAb',
    '乙肝e抗原':                     'HBeAg',
    '乙肝e抗体':                     'HBeAb',
    '乙肝核心抗体':                  'HBcAb',
}

# 变量分组（用于医学解释）
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

# 乙肝相关列（需添加缺失指示变量的列）
HEPATITIS_COLS = ['HBsAg', 'HBsAb', 'HBeAg', 'HBeAb', 'HBcAb']


# ============================================================================
# 第1部分：数据加载与预处理
# ============================================================================

def load_and_preprocess():
    """
    加载并预处理两个附件的数据。
    
    处理步骤：
        1. 读取CSV（GBK编码，处理中文列名）
        2. 重命名列为英文
        3. 处理日期列
        4. 处理性别编码
        5. 为乙肝指标添加缺失指示变量
        6. 填充/处理其他缺失值
        7. 输出数据概览
    
    Returns:
        df1 (pd.DataFrame): 预处理后的附件1数据
        df2 (pd.DataFrame): 预处理后的附件2数据
        col_map_reverse (dict): 英文→中文的逆向列名映射
    """
    print("=" * 70)
    print("第1部分：数据加载与预处理")
    print("=" * 70)
    
    # --- 1a. 读取原始数据 ---
    print("\n[1a] 读取原始CSV文件...")
    df1_raw = pd.read_csv('附件1.csv', encoding='gbk')
    df2_raw = pd.read_csv('附件2.csv', encoding='gbk')
    
    print(f"  附件1（有血糖）: {df1_raw.shape[0]} 行 x {df1_raw.shape[1]} 列")
    print(f"  附件2（无血糖）: {df2_raw.shape[0]} 行 x {df2_raw.shape[1]} 列")
    
    # --- 1b. 列名重命名 ---
    print("\n[1b] 列名重命名（中文→英文）...")
    # 只重命名存在于映射中的列
    rename_dict1 = {k: v for k, v in COLUMN_MAP.items() if k in df1_raw.columns}
    rename_dict2 = {k: v for k, v in COLUMN_MAP.items() if k in df2_raw.columns}
    
    df1 = df1_raw.rename(columns=rename_dict1)
    df2 = df2_raw.rename(columns=rename_dict2)
    
    # 逆向映射
    col_map_reverse = {v: k for k, v in COLUMN_MAP.items()}
    
    print(f"  附件1重命名后列数: {len(df1.columns)}")
    print(f"  附件2重命名后列数: {len(df2.columns)}")
    
    # --- 1c. 处理日期列 ---
    print("\n[1c] 处理日期列...")
    for df_name, df in [('附件1', df1), ('附件2', df2)]:
        if 'exam_date' in df.columns:
            # 尝试多种日期格式解析
            df['exam_date'] = pd.to_datetime(df['exam_date'], format='%d/%m/%Y', errors='coerce')
            # 如果解析失败，尝试其他格式
            if df['exam_date'].isna().sum() > 0:
                df['exam_date'] = pd.to_datetime(df['exam_date'], format='%Y/%m/%d', errors='coerce')
            if df['exam_date'].isna().sum() > 0:
                df['exam_date'] = pd.to_datetime(df['exam_date'], errors='coerce')
            
            # 提取年份和月份作为额外特征
            df['exam_year'] = df['exam_date'].dt.year
            df['exam_month'] = df['exam_date'].dt.month
            
            valid_dates = df['exam_date'].notna().sum()
            total = len(df)
            print(f"  {df_name}: 解析成功 {valid_dates}/{total} 条日期")
    
    # --- 1d. 处理性别 ---
    print("\n[1d] 处理性别变量（男=1，女=0）...")
    for df_name, df in [('附件1', df1), ('附件2', df2)]:
        if 'gender' in df.columns:
            # 查看性别唯一值
            unique_genders = df['gender'].unique()
            print(f"  {df_name} 性别原始值: {unique_genders}")
            # 编码：男=1, 女=0
            df['gender_male'] = df['gender'].apply(
                lambda x: 1 if str(x).strip() in ['男', 'M', 'male', '1'] 
                else (0 if str(x).strip() in ['女', 'F', 'female', '0'] else np.nan)
            )
            # 检查是否有无法编码的值
            na_count = df['gender_male'].isna().sum()
            if na_count > 0:
                print(f"  警告：{na_count} 条性别无法编码，已设为缺失")
                df['gender_male'].fillna(df['gender_male'].mode()[0], inplace=True)
            print(f"  编码后：男(1)={df['gender_male'].sum():.0f}, 女(0)={(df['gender_male']==0).sum()}")
    
    # --- 1e. 剔除无关列 ---
    print("\n[1e] 剔除无关列并处理数据类型...")
    cols_to_drop = ['id', 'gender', 'exam_date']  # 保留exam_year/exam_month和gender_male
    for col in cols_to_drop:
        if col in df1.columns:
            df1.drop(columns=[col], inplace=True)
        if col in df2.columns:
            df2.drop(columns=[col], inplace=True)
    
    # 确保数值列为float类型
    numeric_cols = [c for c in df1.columns if c not in ['glucose']]
    # 附件1中没有glucose，目标在附件1中
    for c in numeric_cols:
        if c in df1.columns:
            df1[c] = pd.to_numeric(df1[c], errors='coerce')
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors='coerce')
    
    print(f"  附件1最终列数: {len(df1.columns)}")
    print(f"  列名: {list(df1.columns)}")
    print(f"  附件2最终列数: {len(df2.columns)}")
    print(f"  列名: {list(df2.columns)}")
    
    # --- 1f. 缺失值分析 ---
    print("\n[1f] 缺失值分析...")
    for df_name, df in [('附件1', df1), ('附件2', df2)]:
        missing = df.isnull().sum()
        missing_pct = missing / len(df) * 100
        missing_df = pd.DataFrame({
            '缺失数': missing,
            '缺失率(%)': missing_pct.round(2)
        })
        missing_df = missing_df[missing_df['缺失数'] > 0].sort_values('缺失率(%)', ascending=False)
        print(f"\n  --- {df_name} 缺失情况 ---")
        if len(missing_df) > 0:
            print(f"  {missing_df.to_string()}")
        else:
            print("  (无缺失值)")
    
    # --- 1g. 为乙肝指标添加缺失指示变量 ---
    print("\n[1g] 为乙肝指标添加缺失指示变量...")
    for df_name, df in [('附件1', df1), ('附件2', df2)]:
        for col in HEPATITIS_COLS:
            if col in df.columns:
                missing_mask = df[col].isna()
                df[f'{col}_missing'] = missing_mask.astype(int)
                missing_count = missing_mask.sum()
                print(f"  {df_name}: {col} 缺失 {missing_count}/{len(df)} "
                      f"({missing_count/len(df)*100:.1f}%), 已添加 {col}_missing 指示变量")
    
    # --- 1h. 缺失值填充 ---
    print("\n[1h] 缺失值填充（分组中位数）...")
    for df_name, df in [('附件1', df1), ('附件2', df2)]:
        # 按变量分组分别填充中位数
        for group_name, cols in VAR_GROUPS.items():
            for col in cols:
                if col in df.columns:
                    # 如果该列有缺失，用中位数填充
                    na_count = df[col].isna().sum()
                    if na_count > 0:
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                        print(f"  {df_name}: {col} ({group_name}) 填充 {na_count} 个缺失值 "
                              f"→ 中位数 {median_val:.3f}")
        
        # 处理其他未分组列（如exam_year, exam_month等）的缺失
        other_cols = [c for c in df.columns if c not in sum(VAR_GROUPS.values(), [])
                      and c not in [f'{h}_missing' for h in HEPATITIS_COLS]
                      and c not in ['glucose', 'exam_year', 'exam_month']]
        for col in other_cols:
            na_count = df[col].isna().sum()
            if na_count > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  {df_name}: {col} 填充 {na_count} 个缺失值 → 中位数 {median_val:.3f}")
    
    # --- 1i. 强制转换为浮点类型（确保statsmodels和scipy能正确处理） ---
    print("\n[1i] 强制转换数值列为float类型...")
    for df_name, df in [('附件1', df1), ('附件2', df2)]:
        convert_cols = [c for c in df.columns if c not in ['glucose', 'exam_year', 'exam_month']]
        for c in convert_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
        # 确保glucose也是float
        if 'glucose' in df.columns:
            df['glucose'] = pd.to_numeric(df['glucose'], errors='coerce').astype(float)
        # 检查dtype
        non_float = [c for c in df.columns if df[c].dtype not in ['float64', 'int64', 'int32']]
        if non_float:
            print(f"  警告: {df_name} 中还有非数值列: {non_float}")
        else:
            print(f"  {df_name}: 所有列已转换为数值类型 ✓")
    
    # --- 1j. 数据概览 ---
    print("\n[1i] 预处理后数据概览:")
    print(f"\n  附件1: {df1.shape}")
    print(f"  附件2: {df2.shape}")
    print(f"\n  附件1目标变量（glucose）统计:")
    print(f"  {df1['glucose'].describe().to_string()}")
    
    # 按题目标准统计糖尿病比例
    diabetes_count = (df1['glucose'] >= DIABETES_THRESHOLD).sum()
    print(f"\n  附件1中血糖≥{DIABETES_THRESHOLD}（糖尿病风险）人数: "
          f"{diabetes_count}/{len(df1)} ({diabetes_count/len(df1)*100:.2f}%)")
    
    return df1, df2, col_map_reverse


# ============================================================================
# 第2部分：问题1 - 特征筛选
# ============================================================================

def problem1_feature_selection(df, col_map_reverse):
    """
    问题1：从42个检测指标中筛选出与血糖相关的主要变量。
    
    筛选方法：
        1. 单变量相关性分析（Pearson + Spearman）
        2. 多重共线性诊断（VIF）
        3. 基于AIC的逐步回归（双向筛选）
        4. 医学合理性综合分析
    
    Args:
        df (pd.DataFrame): 预处理后的附件1数据
        col_map_reverse (dict): 英文→中文列名映射
    
    Returns:
        selected_features (list): 筛选出的特征列表
        stepwise_result (DataFrame): 逐步回归结果表
    """
    print("\n" + "=" * 70)
    print("问题1：主要变量筛选")
    print("=" * 70)
    
    # 特征列（排除目标变量glucose）
    feature_cols = [c for c in df.columns if c != 'glucose']
    print(f"\n待筛选特征数量: {len(feature_cols)}")
    
    # --- 2a. 单变量相关性分析 ---
    print("\n" + "-" * 50)
    print("[2a] 单变量相关性分析")
    print("-" * 50)
    
    glucose = df['glucose'].values
    
    correlation_results = []
    for col in feature_cols:
        # 检查列是否存在且为数值类型
        if col not in df.columns:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        
        x = df[col].values
        
        # 处理NaN值：仅对两个变量都非缺失的配对计算相关性
        valid_mask = ~(np.isnan(x) | np.isnan(glucose))
        if valid_mask.sum() < 10:  # 至少10个有效配对才计算
            continue
        
        x_valid = x[valid_mask]
        g_valid = glucose[valid_mask]
        
        try:
            # Pearson相关系数
            pearson_r, pearson_p = stats.pearsonr(x_valid, g_valid)
        except Exception:
            pearson_r, pearson_p = np.nan, np.nan
        
        try:
            # Spearman秩相关系数
            spearman_r, spearman_p = stats.spearmanr(x_valid, g_valid)
        except Exception:
            spearman_r, spearman_p = np.nan, np.nan
        
        cn_name = col_map_reverse.get(col, col)
        correlation_results.append({
                '特征': col,
                '中文名': cn_name,
                'Pearson_r': pearson_r,
                'Pearson_p': pearson_p,
                'Spearman_r': spearman_r,
                'Spearman_p': spearman_p,
                'abs_Pearson': abs(pearson_r),
                'abs_Spearman': abs(spearman_r),
            })
    
    corr_df = pd.DataFrame(correlation_results)
    corr_df = corr_df.sort_values('abs_Pearson', ascending=False)
    
    print("\n与血糖相关性最高的前20个特征（按|Pearson r|排序）:")
    print(corr_df[['中文名', 'Pearson_r', 'Pearson_p', 'Spearman_r', 'Spearman_p']].head(20).to_string(index=False))
    
    # 显著性标记
    corr_df['Pearson_sig'] = corr_df['Pearson_p'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    )
    
    print("\nPearson相关性显著数量:")
    for sig_level in ['*** (p<0.001)', '** (p<0.01)', '* (p<0.05)', 'ns (p>=0.05)']:
        count = (corr_df['Pearson_sig'] == sig_level.split(' ')[0]).sum()
        print(f"  {sig_level}: {count}")
    
    # 保存相关性分析结果
    corr_output = corr_df[['特征', '中文名', 'Pearson_r', 'Pearson_p', 'Pearson_sig',
                           'Spearman_r', 'Spearman_p']]
    corr_output.to_csv(f'{OUTPUT_DIR}/01_correlation_analysis.csv', index=False,
                       encoding='utf-8-sig')
    print(f"\n相关性分析结果已保存至: {OUTPUT_DIR}/01_correlation_analysis.csv")
    
    # --- 2b. 多重共线性检验（VIF）—— 仅作为诊断，不用于过滤 ---
    print("\n" + "-" * 50)
    print("[2b] 多重共线性诊断（VIF）—— 仅作参考，不用于排除特征")
    print("-" * 50)
    
    # 选择数值型特征
    numeric_features = [c for c in feature_cols 
                        if np.issubdtype(df[c].dtype, np.number) 
                        and df[c].nunique() > 2]  # 排除二值变量
    
    # 计算VIF（作为诊断信息）
    X_vif = df[numeric_features].dropna()
    vif_data = pd.DataFrame()
    vif_data['特征'] = numeric_features
    vif_data['中文名'] = vif_data['特征'].map(col_map_reverse)
    vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) 
                       for i in range(X_vif.shape[1])]
    
    vif_data = vif_data.sort_values('VIF', ascending=False)
    
    print("\nVIF值（前20个，仅作诊断参考）:")
    print(vif_data.head(20).to_string(index=False))
    
    # 保存VIF结果
    vif_data.to_csv(f'{OUTPUT_DIR}/02_vif_analysis.csv', index=False, encoding='utf-8-sig')
    print(f"\nVIF分析结果已保存至: {OUTPUT_DIR}/02_vif_analysis.csv")
    
    # --- 2c. 基于医学知识的候选特征精选 + 逐步回归 ---
    print("\n" + "-" * 50)
    print("[2c] 医学知识指导的候选特征精选 + 双向逐步回归")
    print("-" * 50)
    
    # 策略说明：
    #   血常规指标之间存在天然的高度相关性（如RBC-HGB-HCT, MCV-MCH-MCHC等），
    #   如果直接用VIF<10过滤，几乎所有有意义的指标都会被排除。
    #   因此，我们基于医学知识，从每个高度相关的特征组中选取最具有代表性的指标。
    
    # 医学知识指导的特征精选规则：
    #   - 血常规红系：从{RBC, HGB, HCT}中选 HGB（临床最常用）
    #   - 血常规红系形态：从{MCV, MCH, MCHC}中选 MCV（最重要的红细胞参数）
    #   - 血脂：保留 TC, TG, HDL_C（LDL_C与TC高度相关，且通常由公式计算得到，排除LDL_C）
    #   - 肝功能：保留 ALT, AST, GGT（TP/ALB/GLOB/AGR存在完美共线，仅保留GGT和ALB作为代表）
    #        + 保留 ALB（白蛋白，重要的营养/肝功能指标）
    #   - 白细胞分类：从{NEUT_pct, LYMPH_pct, MONO_pct, EO_pct, BASO_pct}中
    #       保留 NEUT_pct 和 LYMPH_pct（最具有临床意义的两种）
    #   - 血小板：保留 PLT, MPV（排除PCT/PDW）
    #   - 肾功：保留 BUN, Cr, UA（三者之间相关性不高）
    #   - 乙肝：保留 HBsAg, HBsAb（最具临床意义的两个），并保留其缺失指示变量
    
    # 注意：不纳入 exam_year 和 exam_month，因为它们反映的是体检批次/时间效应，
    # 而非生理指标，引入它们会掩盖真实的医学因素与血糖的关系。
    curated_features = [
        # 基本信息
        'age', 'gender_male',
        # 血脂（排除LDL_C，因其与TC高度相关）
        'TC', 'TG', 'HDL_C',
        # 肝功能（从多个相关指标中选取代表性指标）
        'ALT', 'AST', 'GGT', 'ALB',
        # 肾功能
        'BUN', 'Cr', 'UA',
        # 血常规-红系（从{RBC, HGB, HCT}中选 HGB）
        'HGB',
        # 血常规-红系形态（从{MCV, MCH, MCHC}中选 MCV）
        'MCV', 'RDW',
        # 血常规-白细胞（从5个分类中选NEUT_pct和LYMPH_pct）
        'WBC', 'NEUT_pct', 'LYMPH_pct',
        # 血常规-血小板
        'PLT', 'MPV',
        # 碱性磷酸酶（独立指标）
        'ALP',
        # 乙肝（最有意义的两项）
        'HBsAg', 'HBsAb',
    ]
    
    # 只保留数据框中实际存在的特征
    curated_features = [c for c in curated_features if c in df.columns]
    
    # 添加乙肝缺失指示变量
    missing_indicators = [f'{h}_missing' for h in HEPATITIS_COLS 
                          if f'{h}_missing' in df.columns]
    
    # 注意：5个乙肝缺失指示变量完全相关（同4493条缺失），只保留HBsAg_missing作代表
    hbsag_missing = [f'{h}_missing' for h in HEPATITIS_COLS if f'{h}_missing' in df.columns]
    if hbsag_missing:
        # 只保留第一个(HBsAg_missing)，其余剔除
        representative_missing = hbsag_missing[:1]
        excluded_redundant = hbsag_missing[1:]
        print(f"\n  注意：{len(hbsag_missing)}个乙肝缺失指示变量完全相关(共缺失4493条)")
        print(f"    保留代表: {representative_missing[0]}")
        print(f"    剔除冗余: {excluded_redundant}")
    else:
        representative_missing = []
    
    candidate_features = curated_features + representative_missing
    
    print(f"\n基于医学知识精选的候选特征 ({len(candidate_features)} 个):")
    for f in candidate_features:
        cn = col_map_reverse.get(f, f)
        print(f"  {f} ({cn})")
    
    # --- 向后消去法（基于p值） ---
    def backward_elimination(X, y, p_remove=0.0001, verbose=True):
        """
        基于p值的向后消去法。
        从全模型开始，每轮剔除p值最大的特征（p > p_remove）。
        自动处理NaN和inf值。
        """
        included = list(X.columns)
        
        for round_num in range(1, len(included) * 2 + 1):
            if len(included) == 0:
                break
            
            # 构建设计矩阵并清洗inf/nan
            X_sub = sm.add_constant(X[included])
            
            # 替换inf为NaN，然后删除含NaN的行
            X_sub = X_sub.replace([np.inf, -np.inf], np.nan)
            valid_rows = X_sub.notna().all(axis=1)
            X_clean = X_sub[valid_rows]
            y_clean = y[valid_rows]
            
            if len(X_clean) < 100:  # 有效数据太少
                if verbose:
                    print(f"    ⚠ 第{round_num}轮有效数据不足({len(X_clean)}行)，剔除: {included[-1]}")
                included.pop()
                continue
            
            try:
                model = sm.OLS(y_clean, X_clean).fit()
            except Exception as e:
                if verbose:
                    print(f"    ⚠ 第{round_num}轮拟合失败({str(e)[:50]})，剔除: {included[-1]}")
                included.pop()
                continue
            
            pvalues = model.pvalues.drop('const', errors='ignore')
            if len(pvalues) == 0:
                break
            
            worst_pval = pvalues.max()
            worst_feature = pvalues.idxmax()
            
            if worst_pval > p_remove:
                included.remove(worst_feature)
                if verbose:
                    print(f"    第{round_num}轮 → 剔除: {worst_feature} (p={worst_pval:.6f})")
            else:
                if verbose:
                    print(f"    第{round_num}轮 → 停止消去（最大p={worst_pval:.6f} ≤ {p_remove}）")
                break
        
        return included
    
    # 准备数据：清洗inf/nan
    X_candidates = df[candidate_features].copy()
    y = df['glucose']
    
    # 检查并报告数据中的inf/nan
    inf_mask = np.isinf(X_candidates.values).any(axis=1)
    nan_mask = np.isnan(X_candidates.values).any(axis=1)
    if inf_mask.sum() > 0 or nan_mask.sum() > 0:
        inf_cols = [c for c in X_candidates.columns if np.isinf(X_candidates[c]).any()]
        nan_cols = [c for c in X_candidates.columns if np.isnan(X_candidates[c]).any()]
        print(f"  数据清洗: {inf_mask.sum()}行含inf({inf_cols}), {nan_mask.sum()}行含NaN({nan_cols})")
    
    # 替换inf为NaN，整行删除
    X_candidates = X_candidates.replace([np.inf, -np.inf], np.nan)
    valid_rows = X_candidates.notna().all(axis=1)
    dropped = (~valid_rows).sum()
    if dropped > 0:
        print(f"  删除 {dropped} 行无效数据，保留 {valid_rows.sum()} 行")
        X_candidates = X_candidates[valid_rows]
        y = y[valid_rows]
    
    print(f"\n执行向后消去法（p_remove=0.0001）...")
    print(f"  起始特征数: {len(candidate_features)}，样本数: {len(X_candidates)}")
    
    selected_features = backward_elimination(
        X_candidates, y,
        p_remove=0.0001,
        verbose=True
    )
    
    print(f"\n逐步回归选出的特征 ({len(selected_features)} 个):")
    for i, feat in enumerate(selected_features, 1):
        cn = col_map_reverse.get(feat, feat)
        print(f"  {i}. {feat} ({cn})")
    
    # --- 2d. 最终模型汇总 ---
    print("\n" + "-" * 50)
    print("[2d] 最终筛选模型汇总")
    print("-" * 50)
    
    X_final = sm.add_constant(X_candidates[selected_features])
    final_model = sm.OLS(y, X_final).fit()
    
    print("\n最终模型摘要:")
    print(final_model.summary())
    
    # 保存逐步回归过程和结果
    stepwise_result = pd.DataFrame({
        '特征': selected_features,
        '中文名': [col_map_reverse.get(f, f) for f in selected_features],
        '系数': final_model.params.drop('const').values if 'const' in final_model.params 
               else final_model.params.values,
        'p值': final_model.pvalues.drop('const').values if 'const' in final_model.pvalues
               else final_model.pvalues.values,
    })
    stepwise_result.to_csv(f'{OUTPUT_DIR}/03_stepwise_selection.csv', 
                           index=False, encoding='utf-8-sig')
    print(f"\n逐步回归结果已保存至: {OUTPUT_DIR}/03_stepwise_selection.csv")
    
    # 打印选中的特征及其医学意义
    print("\n特征分组及医学解释:")
    feature_set = set(selected_features)
    for group_name, group_cols in VAR_GROUPS.items():
        matched = [c for c in group_cols if c in feature_set]
        if matched:
            names = [f"{c}({col_map_reverse.get(c, c)})" for c in matched]
            print(f"  {group_name}: {', '.join(names)}")
    
    # 找出缺失指示变量
    missing_selected = [f for f in selected_features if f.endswith('_missing')]
    if missing_selected:
        print(f"\n  乙肝缺失指示变量被选中: {missing_selected}")
        print(f"  说明这些乙肝指标的缺失模式本身与血糖水平相关")
    
    return selected_features, stepwise_result


# ============================================================================
# 第3部分：问题2 - 血糖值预测模型
# ============================================================================

def problem2_glucose_prediction(df, selected_features, col_map_reverse):
    """
    问题2：根据体检数据建立血糖值的预测模型。
    
    使用逐步回归筛选出的特征，建立OLS线性回归模型，
    并对模型进行完整的诊断和评估。
    
    Args:
        df (pd.DataFrame): 预处理后的附件1数据
        selected_features (list): 筛选出的特征列表
        col_map_reverse (dict): 列名逆向映射
    """
    print("\n" + "=" * 70)
    print("问题2：血糖值预测模型（多元线性回归）")
    print("=" * 70)
    
    # --- 3a. 划分训练集和测试集 ---
    print("\n[3a] 划分训练集（80%）和测试集（20%）...")
    
    np.random.seed(RANDOM_SEED)
    n = len(df)
    indices = np.random.permutation(n)
    split_idx = int(n * 0.8)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    print(f"  训练集: {len(train_df)} 条")
    print(f"  测试集: {len(test_df)} 条")
    
    # --- 3b. 构建OLS回归模型 ---
    print("\n[3b] 构建OLS多元线性回归模型...")
    
    # 清洗nan/inf（使用向后消去法相同策略）
    def _clean_xy(X, y):
        """清洗X和y中的NaN/Inf"""
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        valid = X.notna().all(axis=1) & y.notna()
        return X[valid], y[valid]
    
    X_train_raw = sm.add_constant(train_df[selected_features])
    y_train_raw = train_df['glucose']
    X_test_raw = sm.add_constant(test_df[selected_features])
    y_test_raw = test_df['glucose']
    
    X_train, y_train = _clean_xy(X_train_raw, y_train_raw)
    X_test, y_test = _clean_xy(X_test_raw, y_test_raw)
    
    dropped_train = len(X_train_raw) - len(X_train)
    dropped_test = len(X_test_raw) - len(X_test)
    if dropped_train > 0 or dropped_test > 0:
        print(f"  数据清洗: 训练集删除 {dropped_train}行, 测试集删除 {dropped_test}行")
        print(f"  训练集: {len(X_train)}条, 测试集: {len(X_test)}条")
    
    ols_model = sm.OLS(y_train, X_train).fit()
    
    print("\n模型摘要:")
    print(ols_model.summary())
    
    # 保存回归系数
    coef_table = pd.DataFrame({
        '变量': ols_model.params.index,
        '中文名': [col_map_reverse.get(c, c) if c != 'const' else '截距' for c in ols_model.params.index],
        '系数(B)': ols_model.params.values,
        '标准误(SE)': ols_model.bse.values,
        't值': ols_model.tvalues.values,
        'p值': ols_model.pvalues.values,
        '95%CI_lower': ols_model.conf_int().iloc[:, 0].values,
        '95%CI_upper': ols_model.conf_int().iloc[:, 1].values,
    })
    coef_table.to_csv(f'{OUTPUT_DIR}/04_ols_coefficients.csv', index=False, encoding='utf-8-sig')
    print(f"\n回归系数表已保存至: {OUTPUT_DIR}/04_ols_coefficients.csv")
    
    # --- 3c. 模型诊断 ---
    print("\n[3c] 模型诊断...")
    
    # 残差分析
    residuals = ols_model.resid
    fitted = ols_model.fittedvalues
    
    print(f"\n  残差统计:")
    print(f"    均值: {residuals.mean():.6f}（理想值=0）")
    print(f"    标准差: {residuals.std():.4f}")
    print(f"    偏度: {stats.skew(residuals):.4f}（理想值=0）")
    print(f"    峰度: {stats.kurtosis(residuals):.4f}（理想值=3）")
    
    # 正态性检验
    jarque_bera = stats.jarque_bera(residuals)
    ad_stat, ad_p = normal_ad(residuals)
    shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # Shapiro-Wilk限制样本量
    
    print(f"\n  正态性检验:")
    print(f"    Jarque-Bera: 统计量={jarque_bera[0]:.4f}, p值={jarque_bera[1]:.6f}")
    print(f"    Shapiro-Wilk: 统计量={shapiro_stat:.4f}, p值={shapiro_p:.6f}")
    print(f"    Anderson-Darling: 统计量={ad_stat:.4f}, p值={ad_p:.6f}")
    
    # 异方差检验（Breusch-Pagan）
    bp_test = het_breuschpagan(residuals, X_train)
    print(f"\n  异方差检验（Breusch-Pagan）:")
    print(f"    LM统计量: {bp_test[0]:.4f}")
    print(f"    p值: {bp_test[1]:.6f}")
    print(f"    F统计量: {bp_test[2]:.4f}")
    print(f"    F_p值: {bp_test[3]:.6f}")
    
    # Durbin-Watson检验（自相关）
    dw = durbin_watson(residuals)
    print(f"\n  Durbin-Watson检验（自相关）:")
    print(f"    DW统计量: {dw:.4f}（接近2表示无自相关）")
    
    # --- 3d. 模型评估 ---
    print("\n[3d] 模型评估（测试集）...")
    
    y_pred = ols_model.predict(X_test)
    
    # 计算评估指标
    residuals_test = y_test - y_pred
    rmse = np.sqrt(np.mean(residuals_test ** 2))
    mae = np.mean(np.abs(residuals_test))
    mape = np.mean(np.abs(residuals_test / y_test)) * 100
    r2 = 1 - np.sum(residuals_test ** 2) / np.sum((y_test - y_test.mean()) ** 2)
    
    print(f"\n  测试集评估指标:")
    print(f"    R²: {r2:.4f}")
    print(f"    RMSE: {rmse:.4f} mmol/L")
    print(f"    MAE: {mae:.4f} mmol/L")
    print(f"    MAPE: {mape:.2f}%")
    
    # 训练集评估指标
    r2_train = ols_model.rsquared
    adj_r2_train = ols_model.rsquared_adj
    aic = ols_model.aic
    bic = ols_model.bic
    
    print(f"\n  训练集评估指标:")
    print(f"    R²: {r2_train:.4f}")
    print(f"    调整R²: {adj_r2_train:.4f}")
    print(f"    AIC: {aic:.4f}")
    print(f"    BIC: {bic:.4f}")
    
    # 保存评估结果
    eval_results = pd.DataFrame({
        '指标': ['R²', '调整R²', 'RMSE', 'MAE', 'MAPE(%)', 'AIC', 'BIC'],
        '训练集': [r2_train, adj_r2_train, np.nan, np.nan, np.nan, aic, bic],
        '测试集': [r2, np.nan, rmse, mae, mape, np.nan, np.nan]
    })
    eval_results.to_csv(f'{OUTPUT_DIR}/05_model_evaluation.csv', index=False, encoding='utf-8-sig')
    print(f"\n模型评估结果已保存至: {OUTPUT_DIR}/05_model_evaluation.csv")
    
    # --- 绘制诊断图 ---
    print("\n  生成模型诊断图...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('血糖预测模型 - 诊断图', fontsize=16, fontweight='bold')
    
    # 图1: 残差 vs 拟合值
    axes[0, 0].scatter(fitted, residuals, alpha=0.5, s=10)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('拟合值')
    axes[0, 0].set_ylabel('残差')
    axes[0, 0].set_title('残差 vs 拟合值')
    
    # 图2: Q-Q图
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q图（正态性检验）')
    
    # 图3: 残差直方图
    axes[0, 2].hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0, 2].plot(x_range, stats.norm.pdf(x_range, residuals.mean(), residuals.std()),
                    'r-', linewidth=2)
    axes[0, 2].set_xlabel('残差')
    axes[0, 2].set_ylabel('密度')
    axes[0, 2].set_title('残差分布')
    
    # 图4: 预测值 vs 实际值（测试集）
    axes[1, 0].scatter(y_test, y_pred, alpha=0.5, s=10)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    axes[1, 0].set_xlabel('实际血糖值 (mmol/L)')
    axes[1, 0].set_ylabel('预测血糖值 (mmol/L)')
    axes[1, 0].set_title(f'预测 vs 实际 (测试集, R²={r2:.4f})')
    
    # 图5: 标准化残差
    standardized_residuals = residuals / np.std(residuals)
    axes[1, 1].scatter(range(len(standardized_residuals)), standardized_residuals, alpha=0.5, s=10)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=3, color='orange', linestyle=':', alpha=0.5)
    axes[1, 1].axhline(y=-3, color='orange', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('观测序号')
    axes[1, 1].set_ylabel('标准化残差')
    axes[1, 1].set_title('标准化残差图')
    
    # 图6: 预测误差分布（测试集）
    axes[1, 2].hist(y_test - y_pred, bins=40, alpha=0.7, edgecolor='black')
    axes[1, 2].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlabel('预测误差 (mmol/L)')
    axes[1, 2].set_ylabel('频数')
    axes[1, 2].set_title(f'预测误差分布 (MAE={mae:.3f})')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/02_glucose_model_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  诊断图已保存: {FIGURE_DIR}/02_glucose_model_diagnostics.png")
    
    return ols_model


# ============================================================================
# 第4部分：问题3 - 糖尿病风险评估
# ============================================================================

def problem3_risk_assessment(df, selected_features, col_map_reverse):
    """
    问题3：根据体检数据对糖尿病的风险进行评估。
    
    使用Logistic回归模型，以血糖>=6.7 mmol/L作为糖尿病阳性标签，
    建立风险预测模型，计算各因素的OR值和95%置信区间。
    
    Args:
        df (pd.DataFrame): 预处理后的附件1数据
        selected_features (list): 筛选出的特征列表
        col_map_reverse (dict): 列名逆向映射
    
    Returns:
        logit_model: 拟合的Logistic回归模型
    """
    print("\n" + "=" * 70)
    print("问题3：糖尿病风险评估（Logistic回归）")
    print("=" * 70)
    
    # --- 4a. 定义糖尿病标签 ---
    print("\n[4a] 定义糖尿病风险标签（根据题目标准）...")
    print(f"  诊断标准: 血糖值 >= {DIABETES_THRESHOLD} mmol/L")
    
    df['diabetes'] = (df['glucose'] >= DIABETES_THRESHOLD).astype(int)
    
    diabetes_count = df['diabetes'].sum()
    non_diabetes_count = len(df) - diabetes_count
    print(f"  糖尿病风险组: {diabetes_count} 人 ({diabetes_count/len(df)*100:.2f}%)")
    print(f"  正常组: {non_diabetes_count} 人 ({non_diabetes_count/len(df)*100:.2f}%)")
    
    # --- 4b. 划分训练集和测试集 ---
    print("\n[4b] 划分训练集（80%）和测试集（20%）...")
    
    np.random.seed(RANDOM_SEED)
    n = len(df)
    indices = np.random.permutation(n)
    split_idx = int(n * 0.8)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    print(f"  训练集: {len(train_df)} 条")
    print(f"  测试集: {len(test_df)} 条")
    
    # --- 4c. 构建Logistic回归模型 ---
    print("\n[4c] 构建Logistic回归模型...")
    
    X_train = sm.add_constant(train_df[selected_features])
    y_train = train_df['diabetes']
    X_test = sm.add_constant(test_df[selected_features])
    y_test = test_df['diabetes']
    
    # 清洗inf/nan
    for X_data, name in [(X_train, '训练集'), (X_test, '测试集')]:
        for col in X_data.columns:
            X_data[col] = X_data[col].replace([np.inf, -np.inf], np.nan)
        na_rows = X_data.isna().any(axis=1) | np.isnan(y_train if '训练' in name else y_test)
        na_count = na_rows.sum()
        if na_count > 0:
            print(f"  {name}: 删除 {na_count} 行含NaN/Inf的数据")
    
    # 防止Logit因奇异矩阵失败：去除零方差的列
    variance_mask = X_train.iloc[:, 1:].var() > 1e-10  # 跳过const
    zero_var_cols = variance_mask[~variance_mask].index.tolist()
    if zero_var_cols:
        print(f"  剔除零方差特征: {zero_var_cols}")
        X_train = X_train.drop(columns=zero_var_cols)
        X_test = X_test.drop(columns=zero_var_cols)
    
    logit_model = sm.Logit(y_train, X_train).fit(disp=1, maxiter=100)
    
    print("\nLogistic回归模型摘要:")
    print(logit_model.summary())
    
    # --- 4d. OR值与95%置信区间 ---
    print("\n[4d] 计算OR值及95%置信区间...")
    
    odds_ratios = np.exp(logit_model.params)
    ci_lower = np.exp(logit_model.conf_int().iloc[:, 0])
    ci_upper = np.exp(logit_model.conf_int().iloc[:, 1])
    
    or_table = pd.DataFrame({
        '变量': logit_model.params.index,
        '中文名': [col_map_reverse.get(c, c) if c != 'const' else '截距' 
                   for c in logit_model.params.index],
        '系数(B)': logit_model.params.values,
        '标准误(SE)': logit_model.bse.values,
        'z值': logit_model.tvalues.values,
        'p值': logit_model.pvalues.values,
        'OR值': odds_ratios.values,
        'OR_95%CI_lower': ci_lower.values,
        'OR_95%CI_upper': ci_upper.values,
    })
    
    print("\n危险因素分析（OR值 > 1 为危险因素，< 1 为保护因素）:")
    print("=" * 80)
    print(or_table.to_string(index=False))
    
    # 保存OR表
    or_table.to_csv(f'{OUTPUT_DIR}/06_logistic_odds_ratios.csv', index=False, encoding='utf-8-sig')
    print(f"\nOR值结果已保存至: {OUTPUT_DIR}/06_logistic_odds_ratios.csv")
    
    # --- 4e. 模型评估 ---
    print("\n[4e] 模型评估...")
    
    # 训练集预测
    y_train_pred_prob = logit_model.predict(X_train)
    y_train_pred = (y_train_pred_prob >= 0.5).astype(int)
    
    # 测试集预测
    y_test_pred_prob = logit_model.predict(X_test)
    y_test_pred = (y_test_pred_prob >= 0.5).astype(int)
    
    # 混淆矩阵
    def confusion_matrix_manual(y_true, y_pred):
        """手动计算混淆矩阵"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp, tn, fp, fn
    
    def calc_metrics(tp, tn, fp, fn):
        """计算分类评估指标"""
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 召回率
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        return accuracy, sensitivity, specificity, precision, f1
    
    # 训练集
    tp_tr, tn_tr, fp_tr, fn_tr = confusion_matrix_manual(y_train, y_train_pred)
    acc_tr, sens_tr, spec_tr, prec_tr, f1_tr = calc_metrics(tp_tr, tn_tr, fp_tr, fn_tr)
    
    print(f"\n  训练集混淆矩阵:")
    print(f"                预测正类    预测负类")
    print(f"    实际正类      {tp_tr:6d}     {fn_tr:6d}")
    print(f"    实际负类      {fp_tr:6d}     {tn_tr:6d}")
    print(f"    准确率: {acc_tr:.4f}, 灵敏度: {sens_tr:.4f}, 特异度: {spec_tr:.4f}")
    
    # 测试集
    tp_te, tn_te, fp_te, fn_te = confusion_matrix_manual(y_test, y_test_pred)
    acc_te, sens_te, spec_te, prec_te, f1_te = calc_metrics(tp_te, tn_te, fp_te, fn_te)
    
    print(f"\n  测试集混淆矩阵:")
    print(f"                预测正类    预测负类")
    print(f"    实际正类      {tp_te:6d}     {fn_te:6d}")
    print(f"    实际负类      {fp_te:6d}     {tn_te:6d}")
    print(f"    准确率: {acc_te:.4f}, 灵敏度: {sens_te:.4f}, 特异度: {spec_te:.4f}")
    
    # 保存评估结果
    eval_metrics = pd.DataFrame({
        '数据集': ['训练集', '测试集'],
        '准确率': [acc_tr, acc_te],
        '灵敏度(Sensitivity)': [sens_tr, sens_te],
        '特异度(Specificity)': [spec_tr, spec_te],
        '精确率(Precision)': [prec_tr, prec_te],
        'F1分数': [f1_tr, f1_te],
    })
    eval_metrics.to_csv(f'{OUTPUT_DIR}/07_logistic_evaluation.csv', index=False, encoding='utf-8-sig')
    print(f"\n评估结果已保存至: {OUTPUT_DIR}/07_logistic_evaluation.csv")
    
    # 伪R²
    print(f"\n  模型拟合优度:")
    print(f"    McFadden伪R²: {logit_model.prsquared:.4f}")
    print(f"    Log-Likelihood: {logit_model.llf:.4f}")
    print(f"    AIC: {logit_model.aic:.4f}")
    print(f"    BIC: {logit_model.bic:.4f}")
    
    # --- 4f. 计算AUC ---
    print("\n[4f] 计算ROC曲线和AUC...")
    
    def roc_curve_manual(y_true, y_score, n_thresholds=100):
        """手动计算ROC曲线（阈值从1降到0，确保FPR递增）"""
        thresholds = np.linspace(1, 0, n_thresholds)  # 从高到低，确保FPR从0到1递增
        tpr = []
        fpr = []
        
        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            
            tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        
        return np.array(fpr), np.array(tpr), thresholds
    
    # 测试集ROC
    fpr, tpr, thresholds = roc_curve_manual(y_test.values, y_test_pred_prob.values)
    auc_value = trapz(tpr, fpr)
    
    print(f"  测试集AUC: {auc_value:.4f}")
    print(f"  AUC解释: ", end="")
    if auc_value >= 0.9:
        print("极好（AUC>=0.9）")
    elif auc_value >= 0.8:
        print("良好（0.8<=AUC<0.9）")
    elif auc_value >= 0.7:
        print("中等（0.7<=AUC<0.8）")
    else:
        print("较差（AUC<0.7）")
    
    # 绘制ROC曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 图1: ROC曲线
    axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_value:.4f}')
    axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='随机猜测')
    axes[0].set_xlabel('假阳性率 (1-Specificity)')
    axes[0].set_ylabel('真阳性率 (Sensitivity)')
    axes[0].set_title(f'ROC曲线 (测试集, AUC={auc_value:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 图2: 预测概率分布
    axes[1].hist(y_test_pred_prob[y_test == 0], bins=30, alpha=0.6, 
                 label='正常组', color='green', edgecolor='black')
    axes[1].hist(y_test_pred_prob[y_test == 1], bins=30, alpha=0.6,
                 label='糖尿病风险组', color='red', edgecolor='black')
    axes[1].axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('预测概率')
    axes[1].set_ylabel('频数')
    axes[1].set_title('预测概率分布')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/03_risk_assessment_roc.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ROC图已保存: {FIGURE_DIR}/03_risk_assessment_roc.png")
    
    # --- 4g. OR值森林图 ---
    print("\n  生成OR值森林图...")
    
    # 排除截距
    or_plot = or_table[or_table['变量'] != 'const'].copy()
    or_plot = or_plot.sort_values('OR值', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(or_plot) * 0.4)))
    
    y_pos = range(len(or_plot))
    ax.errorbar(or_plot['OR值'].values, y_pos, 
                xerr=[(or_plot['OR值'] - or_plot['OR_95%CI_lower']).values,
                      (or_plot['OR_95%CI_upper'] - or_plot['OR值']).values],
                fmt='o', capsize=3, capthick=1, elinewidth=1, markersize=6)
    ax.axvline(x=1, color='r', linestyle='--', alpha=0.5, label='OR=1（无影响）')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(or_plot['中文名'])
    ax.set_xlabel('OR值 (95% CI)')
    ax.set_title('糖尿病风险因素森林图')
    ax.grid(True, axis='x', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/04_odds_ratio_forest.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  森林图已保存: {FIGURE_DIR}/04_odds_ratio_forest.png")
    
    return logit_model


# ============================================================================
# 第5部分：问题4 - 附件2预测
# ============================================================================

def problem4_predict_attachment2(df2, ols_model, logit_model, 
                                  selected_features, col_map_reverse):
    """
    问题4：对附件2（无血糖值数据）进行血糖预测和风险评估。
    
    Args:
        df2 (pd.DataFrame): 预处理后的附件2数据
        ols_model: 问题2的OLS回归模型
        logit_model: 问题3的Logistic回归模型
        selected_features (list): 使用的特征列表
        col_map_reverse (dict): 列名逆向映射
    """
    print("\n" + "=" * 70)
    print("问题4：附件2预测（血糖值 + 糖尿病风险评估）")
    print("=" * 70)
    
    # --- 5a. 数据准备 ---
    print("\n[5a] 准备附件2数据...")
    
    # 确保附件2包含所有需要的特征
    missing_features = [f for f in selected_features if f not in df2.columns]
    if missing_features:
        print(f"  警告：附件2缺少以下特征: {missing_features}")
        for f in missing_features:
            df2[f] = 0
            print(f"    已用0填充缺失特征: {f}")
    
    X_pred = sm.add_constant(df2[selected_features])
    print(f"  预测数据准备完成: {len(df2)} 条 x {X_pred.shape[1]} 个特征")
    
    # --- 5b. 血糖值预测 ---
    print("\n[5b] 血糖值预测...")
    
    glucose_pred = ols_model.predict(X_pred)
    glucose_pred_rounded = np.round(glucose_pred, 3)
    
    print(f"\n  预测血糖值统计:")
    print(f"    均值: {glucose_pred.mean():.4f} mmol/L")
    print(f"    标准差: {glucose_pred.std():.4f}")
    print(f"    最小值: {glucose_pred.min():.4f}")
    print(f"    最大值: {glucose_pred.max():.4f}")
    print(f"    中位数: {np.median(glucose_pred):.4f}")
    
    # 按题目标准分类
    high_risk_count = (glucose_pred >= DIABETES_THRESHOLD).sum()
    print(f"\n  预测血糖≥{DIABETES_THRESHOLD}（糖尿病风险）: "
          f"{high_risk_count}/{len(df2)} ({high_risk_count/len(df2)*100:.2f}%)")
    
    normal_count = (glucose_pred < DIABETES_THRESHOLD).sum()
    print(f"  预测血糖<{DIABETES_THRESHOLD}（正常）: "
          f"{normal_count}/{len(df2)} ({normal_count/len(df2)*100:.2f}%)")
    
    # --- 5c. 糖尿病风险评估 ---
    print("\n[5c] 糖尿病风险评估...")
    
    diabetes_prob = logit_model.predict(X_pred)
    diabetes_pred = (diabetes_prob >= 0.5).astype(int)
    
    print(f"\n  预测糖尿病风险概率统计:")
    print(f"    均值: {diabetes_prob.mean():.4f}")
    print(f"    标准差: {diabetes_prob.std():.4f}")
    print(f"    最小值: {diabetes_prob.min():.4f}")
    print(f"    最大值: {diabetes_prob.max():.4f}")
    
    # 风险分级
    low_risk = (diabetes_prob < 0.3).sum()
    medium_risk = ((diabetes_prob >= 0.3) & (diabetes_prob < 0.7)).sum()
    high_risk = (diabetes_prob >= 0.7).sum()
    
    print(f"\n  糖尿病风险分级:")
    print(f"    低风险 (P<0.3): {low_risk} 人 ({low_risk/len(df2)*100:.1f}%)")
    print(f"    中风险 (0.3<=P<0.7): {medium_risk} 人 ({medium_risk/len(df2)*100:.1f}%)")
    print(f"    高风险 (P>=0.7): {high_risk} 人 ({high_risk/len(df2)*100:.1f}%)")
    
    # --- 5d. 输出预测结果 ---
    print("\n[5d] 输出预测结果表...")
    
    # 使用原始ID（如果有的话）关联回原始数据
    original_df2 = pd.read_csv('附件2.csv', encoding='gbk')
    
    results_df = pd.DataFrame({
        'id': original_df2['id'].values if 'id' in original_df2.columns else range(1, len(df2) + 1),
        '预测血糖值(mmol/L)': glucose_pred_rounded,
        '糖尿病风险概率': diabetes_prob.round(4),
        '糖尿病风险分类': ['高风险' if p >= 0.7 
                          else ('中风险' if p >= 0.3 else '低风险') 
                          for p in diabetes_prob],
        '预测血糖分级': ['糖尿病风险' if g >= DIABETES_THRESHOLD else '正常' 
                        for g in glucose_pred],
    })
    
    # 合并原始基本信息
    for col in ['年龄', '性别']:
        if col in original_df2.columns:
            results_df[col] = original_df2[col].values
    
    # 输出前20条预览
    print("\n预测结果预览（前20条）:")
    print(results_df.head(20).to_string(index=False))
    
    # 保存完整结果
    results_df.to_csv(f'{OUTPUT_DIR}/08_attachment2_predictions.csv', 
                      index=False, encoding='utf-8-sig')
    print(f"\n完整预测结果已保存至: {OUTPUT_DIR}/08_attachment2_predictions.csv")
    
    # --- 5e. 可视化预测结果 ---
    print("\n  生成预测结果可视化...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 图1: 预测血糖分布
    axes[0].hist(glucose_pred, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0].axvline(x=DIABETES_THRESHOLD, color='r', linestyle='--', 
                    label=f'阈值={DIABETES_THRESHOLD}')
    axes[0].set_xlabel('预测血糖值 (mmol/L)')
    axes[0].set_ylabel('人数')
    axes[0].set_title('附件2预测血糖分布')
    axes[0].legend()
    
    # 图2: 风险概率分布
    axes[1].hist(diabetes_prob, bins=20, alpha=0.7, edgecolor='black', color='coral')
    axes[1].axvline(x=0.3, color='orange', linestyle=':', alpha=0.7, label='低/中风险边界')
    axes[1].axvline(x=0.7, color='red', linestyle=':', alpha=0.7, label='中/高风险边界')
    axes[1].set_xlabel('糖尿病风险概率')
    axes[1].set_ylabel('人数')
    axes[1].set_title('附件2糖尿病风险概率分布')
    axes[1].legend()
    
    # 图3: 风险等级饼图
    risk_counts = [low_risk, medium_risk, high_risk]
    labels = ['低风险', '中风险', '高风险']
    colors = ['green', 'orange', 'red']
    axes[2].pie(risk_counts, labels=labels, autopct='%1.1f%%', 
                colors=colors, startangle=90)
    axes[2].set_title('附件2风险等级分布')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/05_attachment2_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  预测结果图已保存: {FIGURE_DIR}/05_attachment2_prediction.png")


# ============================================================================
# 第6部分：数据探索性分析（EDA）- 辅助可视化
# ============================================================================

def exploratory_analysis(df, col_map_reverse):
    """
    数据探索性分析，生成论文所需的图表。
    
    Args:
        df (pd.DataFrame): 预处理后的附件1数据
        col_map_reverse (dict): 列名逆向映射
    """
    print("\n" + "=" * 70)
    print("探索性数据分析（EDA）")
    print("=" * 70)
    
    # --- EDA1: 血糖分布图 ---
    print("\n[EDA] 生成血糖分布图...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 直方图 + 核密度
    axes[0].hist(df['glucose'], bins=60, density=True, alpha=0.7, 
                 color='steelblue', edgecolor='black')
    
    # 拟合正态分布
    mu, sigma = df['glucose'].mean(), df['glucose'].std()
    x = np.linspace(df['glucose'].min(), df['glucose'].max(), 200)
    axes[0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                 label=f'正态分布 (mu={mu:.2f}, sigma={sigma:.2f})')
    axes[0].axvline(x=DIABETES_THRESHOLD, color='red', linestyle='--', 
                    linewidth=2, label=f'糖尿病阈值 ({DIABETES_THRESHOLD})')
    axes[0].axvline(x=6.1, color='orange', linestyle=':', linewidth=1.5,
                    label='空腹血糖正常上限(6.1)')
    axes[0].set_xlabel('血糖值 (mmol/L)')
    axes[0].set_ylabel('密度')
    axes[0].set_title(f'血糖分布 (n={len(df)}, 均值={mu:.2f}, 中位数={df["glucose"].median():.2f})')
    axes[0].legend()
    
    # 箱线图
    df['diabetes_group'] = df['glucose'].apply(
        lambda x: '糖尿病风险' if x >= DIABETES_THRESHOLD else '正常')
    colors = {'正常': 'green', '糖尿病风险': 'red'}
    for i, (group, data) in enumerate(df.groupby('diabetes_group')):
        bp = axes[1].boxplot(data['glucose'], positions=[i], widths=0.4,
                             patch_artist=True,
                             boxprops=dict(facecolor=colors[group], alpha=0.6))
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['正常', '糖尿病风险'])
    axes[1].set_ylabel('血糖值 (mmol/L)')
    axes[1].set_title(f'血糖分组箱线图 (阈值={DIABETES_THRESHOLD})')
    axes[1].axhline(y=DIABETES_THRESHOLD, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/01_glucose_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {FIGURE_DIR}/01_glucose_distribution.png")
    
    # --- EDA2: 年龄与血糖的关系 ---
    print("\n[EDA] 生成年龄-血糖关系图...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 散点图 + 回归线
    axes[0].scatter(df['age'], df['glucose'], alpha=0.3, s=5, c='steelblue')
    
    # 按年龄分组计算均值
    age_bins = range(0, 100, 10)
    df['age_group'] = pd.cut(df['age'], bins=age_bins, right=False)
    age_mean = df.groupby('age_group', observed=True)['glucose'].agg(['mean', 'std', 'count'])
    age_centers = [(interval.left + interval.right) / 2 for interval in age_mean.index]
    
    axes[0].errorbar(age_centers, age_mean['mean'], yerr=age_mean['std']/np.sqrt(age_mean['count']),
                     fmt='ro-', capsize=3, markersize=6, linewidth=2, label='年龄组均值 ± SE')
    axes[0].set_xlabel('年龄')
    axes[0].set_ylabel('血糖值 (mmol/L)')
    axes[0].set_title('年龄与血糖关系')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 各年龄组糖尿病比例
    diabetes_rate = df.groupby('age_group', observed=True)['diabetes_group'].apply(
        lambda x: (x == '糖尿病风险').mean() * 100)
    axes[1].bar(range(len(diabetes_rate)), diabetes_rate.values, alpha=0.7, 
                color='coral', edgecolor='black')
    axes[1].set_xticks(range(len(diabetes_rate)))
    axes[1].set_xticklabels([str(int(interval.left)) + '-' + str(int(interval.right)) 
                             for interval in diabetes_rate.index], rotation=45)
    axes[1].set_ylabel('糖尿病风险比例 (%)')
    axes[1].set_title('各年龄组糖尿病风险比例')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/02_age_glucose_relationship.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {FIGURE_DIR}/02_age_glucose_relationship.png")
    
    # --- EDA3: 性别差异 ---
    print("\n[EDA] 生成性别差异图...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    gender_labels = {0: '女', 1: '男'}
    df['gender_label'] = df['gender_male'].map(gender_labels)
    
    # 箱线图
    df.boxplot(column='glucose', by='gender_label', ax=axes[0])
    axes[0].set_title('不同性别的血糖分布')
    axes[0].set_xlabel('性别')
    axes[0].set_ylabel('血糖值 (mmol/L)')
    
    # 性别-糖尿病比例
    gender_diabetes = df.groupby('gender_label')['diabetes_group'].apply(
        lambda x: (x == '糖尿病风险').mean() * 100)
    axes[1].bar(gender_diabetes.index, gender_diabetes.values, 
                color=['pink', 'lightblue'], edgecolor='black', alpha=0.7)
    axes[1].set_ylabel('糖尿病风险比例 (%)')
    axes[1].set_title('不同性别的糖尿病风险比例')
    for i, v in enumerate(gender_diabetes.values):
        axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center')
    
    plt.suptitle('')  # 移除默认标题
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/03_gender_difference.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {FIGURE_DIR}/03_gender_difference.png")
    
    # --- EDA4: 相关性热图 ---
    print("\n[EDA] 生成相关性热图...")
    
    # 选择数值特征 + 血糖
    corr_features = ['glucose', 'age', 'gender_male', 'TC', 'TG', 'HDL_C', 'LDL_C',
                     'BUN', 'Cr', 'UA', 'ALT', 'AST', 'GGT', 'ALP', 'WBC', 'HGB',
                     'PLT', 'ALB', 'TP']
    available_features = [f for f in corr_features if f in df.columns]
    
    corr_matrix = df[available_features].corr(method='pearson')
    
    # 重命名行列方便阅读
    corr_matrix.index = [col_map_reverse.get(c, c) for c in corr_matrix.index]
    corr_matrix.columns = [col_map_reverse.get(c, c) for c in corr_matrix.columns]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0, 
                annot=True, fmt='.2f', square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('关键指标与血糖的Pearson相关性热图', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {FIGURE_DIR}/04_correlation_heatmap.png")
    
    # --- EDA5: 缺失值模式图 ---
    print("\n[EDA] 生成缺失值模式图...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    missing_pct = df.isnull().sum() / len(df) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    if len(missing_pct) > 0:
        cn_labels = [col_map_reverse.get(c, c) for c in missing_pct.index]
        bars = ax.bar(range(len(missing_pct)), missing_pct.values, 
                      color='coral', alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(missing_pct)))
        ax.set_xticklabels(cn_labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('缺失率 (%)')
        ax.set_title('各指标缺失率分布')
        ax.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='20%缺失阈值')
        ax.legend()
        
        for bar, val in zip(bars, missing_pct.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/05_missing_value_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {FIGURE_DIR}/05_missing_value_pattern.png")
    
    # 清理临时列
    df.drop(columns=['diabetes_group', 'age_group', 'gender_label'], 
            inplace=True, errors='ignore')


# ============================================================================
# 主函数入口
# ============================================================================

def main():
    """
    主函数：按顺序执行四个问题的分析流程。
    """
    print("=" * 70)
    print("2026年武汉理工大学数学建模训练 - C题：糖尿病风险预测")
    print("传统统计模型分析（pandas + numpy + scipy + statsmodels）")
    print("=" * 70)
    print(f"\n分析开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"糖尿病诊断阈值: 血糖 >= {DIABETES_THRESHOLD} mmol/L（题目标准）")
    print(f"图片保存目录: {FIGURE_DIR}/")
    print(f"结果保存目录: {OUTPUT_DIR}/")
    
    # ---- 第1步：高级数据预处理（MICE链式方程多重插补） ----
    print("\n" + "=" * 70)
    print("第1步：高级数据预处理 — MICE链式方程多重插补")
    print("=" * 70)
    
    # 使用高级预处理模块（缺失值插补策略：mice / knn / median）
    # 详细算法说明见: output/缺失值插补算法解读报告.md
    df1, df2 = preprocess_with_advanced_imputation(
        file1='附件1.csv',
        file2='附件2.csv',
        strategy='mice',         # 推荐MICE，适合高缺失率数据
        knn_k=5,
        mice_iter=20,
        missing_threshold=0.5,   # >50%缺失的列不参与迭代建模
    )
    
    # 逆向列名映射（用于图表和报表的显示）
    col_map_reverse = {v: k for k, v in COLUMN_MAP.items()}
    
    # 目标变量统计
    print("\n" + "-" * 40)
    print("目标变量（glucose）统计")
    print("-" * 40)
    print(f"  {df1['glucose'].describe().to_string()}")
    diabetes_count = (df1['glucose'] >= DIABETES_THRESHOLD).sum()
    print(f"\n  血糖≥{DIABETES_THRESHOLD}（糖尿病风险）: {diabetes_count}/{len(df1)} "
          f"({diabetes_count/len(df1)*100:.2f}%)")
    
    # ---- 第2步：探索性数据分析 ----
    exploratory_analysis(df1, col_map_reverse)
    
    # ---- 第3步：问题1 - 特征筛选 ----
    selected_features, stepwise_result = problem1_feature_selection(
        df1, col_map_reverse)
    
    # ---- 第4步：问题2 - 血糖值预测模型 ----
    ols_model = problem2_glucose_prediction(
        df1, selected_features, col_map_reverse)
    
    # ---- 第5步：问题3 - 糖尿病风险评估 ----
    logit_model = problem3_risk_assessment(
        df1, selected_features, col_map_reverse)
    
    # ---- 第6步：问题4 - 附件2预测 ----
    problem4_predict_attachment2(
        df2, ols_model, logit_model, selected_features, col_map_reverse)
    
    # ---- 完成 ----
    print("\n" + "=" * 70)
    print("分析完成！")
    print(f"完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出文件汇总:")
    print(f"  图片目录: {FIGURE_DIR}/")
    print(f"    - 01_glucose_distribution.png      (血糖分布)")
    print(f"    - 02_age_glucose_relationship.png   (年龄-血糖关系)")
    print(f"    - 03_gender_difference.png          (性别差异)")
    print(f"    - 04_correlation_heatmap.png        (相关性热图)")
    print(f"    - 05_missing_value_pattern.png      (缺失值模式)")
    print(f"    - 02_glucose_model_diagnostics.png  (模型诊断图)")
    print(f"    - 03_risk_assessment_roc.png        (ROC曲线)")
    print(f"    - 04_odds_ratio_forest.png          (OR森林图)")
    print(f"    - 05_attachment2_prediction.png     (附件2预测结果)")
    print(f"  结果CSV目录: {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    - {f}")
    print("=" * 70)


if __name__ == '__main__':
    main()