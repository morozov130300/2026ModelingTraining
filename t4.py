# -*- coding: utf-8 -*-
"""
t4.py — 问题4：基于Logistic GAM的高风险人群识别与特征分析
============================================================
流程：
  4.1 基于Logistic GAM的附件2风险预测 → 输出个人风险概率
  4.2 高风险人群识别（预测风险Top20%）
  4.3 高风险人群特征分析（描述统计 + 显著性比较）
  4.4 基于GAM的风险因素贡献分析
  4.5 高风险人群健康管理建议
"""
import warnings, os, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy import stats
from sklearn.utils.class_weight import compute_class_weight

# ── 字体 ──
_CN_FP = None
for _fp in ["C:/Windows/Fonts/msyh.ttc","C:/Windows/Fonts/simhei.ttf"]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp); _CN_FP = fm.FontProperties(fname=_fp); break
if not _CN_FP:
    for _f in fm.fontManager.ttflist:
        if any(k in _f.name for k in ["YaHei","SimHei","PingFang"]):
            _CN_FP = fm.FontProperties(family=_f.name); break
plt.rcParams["font.family"] = "sans-serif"; plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "问题4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from 数据预处理 import load_and_rename, impute_median, EN2CN, CN2EN, CSV1, CSV2, TARGET

# ══════════════════════════════════════════════════════════
#  常数（与t3.py一致的模型超参数 + 最终特征列表）
# ══════════════════════════════════════════════════════════
K_SPLINES = 5
LAM = 1.0
MAX_ITER = 5000
TOL = 1e-6

# 从t3.py输出的最终13个特征（英文列名）
FINAL_VARS = ["年龄","TG","RBC","MCHC","HGB","ALT","性别",
              "LDL_C","HCT","WBC","TC","PLT","年龄_TG"]

# 展示用中文名映射
VAR_DISPLAY = {
    "年龄":"年龄","TG":"甘油三酯","RBC":"红细胞计数","MCHC":"MCHC",
    "HGB":"血红蛋白","ALT":"ALT","性别":"性别","LDL_C":"LDL-C",
    "HCT":"红细胞压积","WBC":"白细胞计数","TC":"总胆固醇","PLT":"血小板计数",
    "年龄_TG":"年龄×TG"
}
DISPLAY_VARS = [VAR_DISPLAY[v] for v in FINAL_VARS]

SEP = "=" * 70
SEP2 = "-" * 60

RISK_LABELS = ["极低风险","低风险","中等风险","高风险","极高风险"]
RISK_COLORS = ["#2ECC71","#58D68D","#F39C12","#E67E22","#E74C3C"]


# ══════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════

def _cat_encode(df):
    """编码性别为数值 + 确保所有列为数值型"""
    df = df.copy()
    if "性别" in df.columns:
        if df["性别"].dtype == object:
            df["性别"] = df["性别"].map({"男":1,"女":0}).fillna(0.5)
        else:
            df["性别"] = df["性别"].fillna(0.5)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def _ts(L):
    """pyGAM TermList拼接"""
    if not L: return None
    r = L[0]
    for t in L[1:]: r = r + t
    return r


def _sample_weights(y):
    """平衡样本权重"""
    classes = np.unique(y)
    if len(classes) <= 1:
        return np.ones(len(y))
    cw = compute_class_weight("balanced", classes=classes, y=y)
    return np.array([cw[int(i)] for i in y])


# ══════════════════════════════════════════════════════════
#  4.1 模型训练函数
# ══════════════════════════════════════════════════════════

def prepare_train_data(dm_threshold=7.0):
    """加载附件1，返回含DM标签的特征DataFrame + 特征列表"""
    df = load_and_rename(CSV1)
    df = impute_median(df)
    if TARGET in df.columns and df[TARGET].isnull().any():
        df[TARGET] = df[TARGET].fillna(df[TARGET].median())
    # 二分类标签
    df["DM"] = (df[TARGET] > dm_threshold).astype(int)
    # 性别编码
    df = _cat_encode(df)
    # 特征工程：创建年龄_TG交互
    if "年龄" in df.columns and "TG" in df.columns:
        df["年龄_TG"] = df["年龄"].values * df["TG"].values
    # 确保所有FINAL_VARS存在
    for v in FINAL_VARS:
        if v not in df.columns:
            df[v] = 0
    # 显式确保FINAL_VARS列为数值类型
    for v in FINAL_VARS:
        df[v] = pd.to_numeric(df[v], errors="coerce").fillna(0)
    print("  训练样本: %d, DM阳性率: %.2f%%" % (len(df), df["DM"].mean()*100))
    return df


def prepare_pred_data():
    """加载附件2，返回特征DataFrame（含id列）"""
    df = load_and_rename(CSV2)
    df = impute_median(df)
    df = _cat_encode(df)
    # 特征工程
    if "年龄" in df.columns and "TG" in df.columns:
        df["年龄_TG"] = df["年龄"].values * df["TG"].values
    for v in FINAL_VARS:
        if v not in df.columns:
            df[v] = 0
    # 显式确保FINAL_VARS列为数值类型
    for v in FINAL_VARS:
        df[v] = pd.to_numeric(df[v], errors="coerce").fillna(0)
    print("  预测样本: %d" % len(df))
    return df


def fit_gam(df, preds, sample_weight=None):
    """Logistic GAM训练"""
    from pygam import LogisticGAM, s, f
    X, y = df[preds].values, df["DM"].values
    continuous_idx = [i for i, p in enumerate(preds) if p != "性别"]
    cat_idx = [i for i, p in enumerate(preds) if p == "性别"]
    terms = _ts([s(i, n_splines=K_SPLINES) for i in continuous_idx])
    for ci in cat_idx:
        terms = terms + f(ci)
    # 强制类型转换：确保X为float64（防止object列渗入）
    if X.dtype.kind == "O":
        print("  [调试] X为object类型，强制转换...", end=" ", flush=True)
        X = np.asarray(X, dtype=float)
    gam = LogisticGAM(terms, fit_intercept=True, max_iter=MAX_ITER, lam=LAM, tol=TOL)
    print("  拟合Logistic GAM (k=%d, lam=%.1f, max_iter=%d)..." %
          (K_SPLINES, LAM, MAX_ITER), end=" ", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gam.fit(X, y, weights=sample_weight)
    print("完成")
    # 训练集AUC
    from sklearn.metrics import roc_auc_score
    yp = gam.predict_proba(X)
    auc = roc_auc_score(y, yp) if len(np.unique(y)) > 1 else 0
    print("  训练集AUC: %.4f" % auc)
    try:
        udof = gam.statistics_.get("edof", gam.statistics_.get("edf", 0))
        print("  UBRE=%.4f  EDF=%.2f" % (gam.statistics_.get("UBRE",0), udof))
    except:
        pass
    return gam


# ══════════════════════════════════════════════════════════
#  4.2 高风险人群识别
# ══════════════════════════════════════════════════════════

def identify_high_risk(df, risk_col="风险概率", top_pct=0.2):
    """按预测风险Top pct划分高风险人群"""
    threshold = np.percentile(df[risk_col], (1 - top_pct) * 100)
    df["高风险"] = (df[risk_col] >= threshold).astype(int)
    print("\n  高风险阈值 (Top %.0f%%): %.4f" % (top_pct*100, threshold))
    print("  高风险人数: %d / %d (%.1f%%)" %
          (df["高风险"].sum(), len(df), df["高风险"].mean()*100))
    return df, threshold


# ══════════════════════════════════════════════════════════
#  4.3 高风险人群特征分析
# ══════════════════════════════════════════════════════════

def risk_stratify_5(y_prob):
    """五档风险分层"""
    bins = np.quantile(y_prob, [0.0, 0.1, 0.3, 0.5, 0.8, 1.0])
    bins[0] = 0.0; bins[-1] = 1.0
    bins = sorted(set(np.round(bins, 6)))
    while len(bins) < 6:
        mid = (bins[-2] + bins[-1]) / 2
        bins.insert(-1, mid)
    bins = np.array(bins[:6])
    labels = list(range(len(bins) - 1))
    return pd.cut(y_prob, bins=bins, labels=labels, right=False, include_lowest=True).astype(int), bins


def descriptive_analysis(df, risk_col="风险概率", group_col="高风险"):
    """描述统计 + 显著性比较（高风险组 vs 其余）"""
    from scipy import stats as _stats
    print("\n" + SEP2)
    print("  [4.3 高风险人群特征分析]")
    print(SEP2)

    # 分组
    high = df[df[group_col] == 1]
    low = df[df[group_col] == 0]
    print("\n  高风险组 n=%d | 其余组 n=%d\n" % (len(high), len(low)))
    print("  %-20s %12s %12s %10s %8s" %
          ("变量", "高风险均值", "其余均值", "差异", "p值"))
    print("  " + "-" * 65)

    # 对每个连续变量计算
    cont_vars = [v for v in FINAL_VARS if v != "性别"]
    cat_vars = ["性别"]

    sig_results = {}  # 存显著结果用于报告
    rows_desc = []
    for v in cont_vars:
        h_mean = high[v].mean()
        l_mean = low[v].mean()
        diff = h_mean - l_mean
        # Mann-Whitney U检验（非参数，不假设正态）
        try:
            _, p = _stats.mannwhitneyu(high[v].dropna(), low[v].dropna(), alternative="two-sided")
        except:
            p = 1.0
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        rows_desc.append({"变量": v, "高风险均值": h_mean, "其余均值": l_mean,
                          "差异": diff, "p值": p, "显著性": sig})
        print("  %-20s %12.4f %12.4f %+10.4f %8.4f %s" %
              (VAR_DISPLAY.get(v, v), h_mean, l_mean, diff, p, sig))
        if p < 0.05:
            sig_results[v] = {"diff": diff, "p": p}

    # 性别（分类变量 → 卡方检验）
    for v in cat_vars:
        # 男性比例
        h_male = (high[v] >= 0.5).mean()
        l_male = (low[v] >= 0.5).mean()
        diff = h_male - l_male
        try:
            tbl = pd.crosstab(df[v] >= 0.5, df[group_col])
            _, p, _, _ = _stats.chi2_contingency(tbl)
        except:
            p = 1.0
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        rows_desc.append({"变量": v+"(男比)", "高风险均值": h_male, "其余均值": l_male,
                          "差异": diff, "p值": p, "显著性": sig})
        print("  %-20s %12.4f %12.4f %+10.4f %8.4f %s" %
              ("性别(男比例)", h_male, l_male, diff, p, sig))
        if p < 0.05:
            sig_results[v] = {"diff": diff, "p": p}

    # 年龄分段（>55岁比例）
    age_high = (high["年龄"] > 55).mean()
    age_low = (low["年龄"] > 55).mean()
    print("\n  --- 年龄分段 ---")
    print("  年龄>55岁比例: 高风险组=%.1f%%  其余组=%.1f%%  差值=%+.1f%%" %
          (age_high*100, age_low*100, (age_high - age_low)*100))

    # TG>2.3比例
    tg_high = (high["TG"] > 2.3).mean()
    tg_low = (low["TG"] > 2.3).mean()
    print("  TG>2.3比例: 高风险组=%.1f%%  其余组=%.1f%%  差值=%+.1f%%" %
          (tg_high*100, tg_low*100, (tg_high - tg_low)*100))

    # 高危亚群：年龄>55且TG>2.3
    cluster_high = ((high["年龄"] > 55) & (high["TG"] > 2.3)).sum()
    cluster_low = ((low["年龄"] > 55) & (low["TG"] > 2.3)).sum()
    cluster_high_pct = cluster_high / len(high) * 100 if len(high) > 0 else 0
    cluster_low_pct = cluster_low / len(low) * 100 if len(low) > 0 else 0
    print("  年龄>55且TG>2.3亚群: 高风险组=%d(%.1f%%)  其余组=%d(%.1f%%)" %
          (cluster_high, cluster_high_pct, cluster_low, cluster_low_pct))

    df_desc = pd.DataFrame(rows_desc)
    return df_desc, sig_results


def cross_analysis(df, risk_col="风险概率"):
    """交叉分析：风险等级 × 性别/年龄段"""
    levels, bins = risk_stratify_5(df[risk_col].values)
    df = df.copy()
    df["风险等级"] = levels

    print("\n" + SEP2)
    print("  [交叉分析] 风险等级 × 年龄段")
    print(SEP2)
    age_bins = [0, 30, 45, 55, 65, 100]
    age_labels = ["<30","30-45","45-55","55-65",">65"]
    df["年龄组"] = pd.cut(df["年龄"], bins=age_bins, labels=age_labels, right=False)
    cross_age = pd.crosstab(df["年龄组"], df["风险等级"],
                            normalize="index") * 100
    cross_age.columns = RISK_LABELS[:len(cross_age.columns)]
    print(cross_age.round(1).to_string())
    print()

    # 性别交叉
    df["性别标签"] = df["性别"].apply(lambda x: "男" if x >= 0.5 else "女")
    cross_gender = pd.crosstab(df["性别标签"], df["风险等级"],
                               normalize="index") * 100
    cross_gender.columns = RISK_LABELS[:len(cross_gender.columns)]
    print(cross_gender.round(1).to_string())

    # 高危亚群详细报告
    print("\n  --- 高危亚群识别 ---")
    high_mask = df["风险等级"] >= 3  # 高风险+极高风险
    high_df = df[high_mask]
    print("  高风险+极高风险人群: %d人 (%.1f%%)" % (len(high_df), len(high_df)/len(df)*100))

    # 年龄>55 + TG>2.3
    cluster = ((df["年龄"] > 55) & (df["TG"] > 2.3))
    cluster_in_high = cluster[high_mask].sum()
    cluster_total = cluster.sum()
    print("  年龄>55岁且TG>2.3: 共%d人, 其中%d人(%.1f%%)属于高风险" %
          (cluster_total, cluster_in_high,
           cluster_in_high/cluster_total*100 if cluster_total > 0 else 0))

    # 男性 >55
    male_old = ((df["性别"] >= 0.5) & (df["年龄"] > 55))
    male_old_high = male_old[high_mask].sum()
    male_old_total = male_old.sum()
    print("  男性>55岁: 共%d人, 其中%d人(%.1f%%)属于高风险" %
          (male_old_total, male_old_high,
           male_old_high/male_old_total*100 if male_old_total > 0 else 0))

    return df


# ══════════════════════════════════════════════════════════
#  4.4 GAM风险因素贡献分析
# ══════════════════════════════════════════════════════════

def contribution_analysis(gam, X, preds, z_threshold=2.0):
    """
    基于GAM偏效应的全局变量贡献分析。
    对于每个变量，计算偏效应曲线的"总波动幅度"(range)作为群体层面的贡献大小。
    range = max(偏效应) - min(偏效应) → 越大说明该变量在群体中的风险驱动越强。
    """
    from pygam import s, f
    nv = len(preds)
    x_mean = X.mean(axis=0)
    contrib = {}

    print("\n" + SEP2)
    print("  [4.4 GAM风险因素贡献分析]")
    print(SEP2)
    print("\n  %-20s %10s %10s %8s" % ("变量", "贡献(ΔlogOR)", "排名", "方向"))
    print("  " + "-" * 52)

    for i, p in enumerate(preds):
        # 使用训练数据的实际唯一值（避免pyGAM categorical domain错误）
        uniq_vals = np.sort(np.unique(X[:, i]))
        if p == "性别":
            x_vals = uniq_vals
            X_grid = np.tile(x_mean, (len(x_vals), 1)); X_grid[:, i] = x_vals
            eta = np.array([0.0, 0.0])
            try:
                with np.errstate(divide="ignore"):
                    prob = np.clip(gam.predict_proba(X_grid), 1e-10, 1-1e-10)
                    eta = np.log(prob/(1-prob)); eta -= eta.mean()
                contrib_range = abs(eta[-1] - eta[0]) if len(eta) > 1 else 0
            except Exception:
                contrib_range = 0
            direction = "男>女" if (len(uniq_vals)>1 and eta[-1] > eta[0]) else "女>男" if len(uniq_vals)>1 else "-"
        else:
            xmin, xmax = X[:, i].min(), X[:, i].max()
            if np.isclose(xmin, xmax):
                contrib_range = 0
                direction = "-"
            else:
                xg = np.linspace(xmin, xmax, 100)
                X_grid = np.tile(x_mean, (100, 1)); X_grid[:, i] = xg
                with np.errstate(divide="ignore"):
                    prob = np.clip(gam.predict_proba(X_grid), 1e-10, 1-1e-10)
                    eta = np.log(prob/(1-prob)); eta -= eta.mean()
                contrib_range = eta.max() - eta.min()
                direction = "正向" if eta[-1] > eta[0] else "负向"
        contrib[p] = contrib_range
        print("  %-20s %10.4f %10s %8s" %
              (VAR_DISPLAY.get(p, p), contrib_range,
               "★" if contrib_range >= np.percentile(list(contrib.values()), 75) else "",
               direction))

    # 排序
    sorted_vars = sorted(contrib.items(), key=lambda x: -x[1])
    print("\n  --- 贡献排序（从大到小） ---")
    for rank, (v, c) in enumerate(sorted_vars, 1):
        print("  #%d %-18s ΔlogOR=%.4f" % (rank, VAR_DISPLAY.get(v, v), c))

    # 贡献占比
    total = sum(c for _, c in sorted_vars)
    print("\n  累计贡献分布:")
    cum = 0
    for rank, (v, c) in enumerate(sorted_vars, 1):
        cum += c / total * 100
        print("  Top%d %-18s %.1f%% (累计%.1f%%)" %
              (rank, VAR_DISPLAY.get(v, v), c/total*100, cum))

    return sorted_vars


# ══════════════════════════════════════════════════════════
#  4.5 健康管理建议
# ══════════════════════════════════════════════════════════

def generate_recommendations(df, sig_results, contrib_sorted, desc_df):
    """基于分析结果生成结构化健康管理建议"""
    print("\n" + SEP2)
    print("  [4.5 高风险人群健康管理建议]")
    print(SEP2)

    n_total = len(df)
    n_high = df["高风险"].sum()
    high_ratio = n_high / n_total * 100

    # 高风险人群基本统计
    high_df = df[df["高风险"] == 1]
    avg_age_high = high_df["年龄"].mean()
    male_ratio_high = (high_df["性别"] >= 0.5).mean() * 100
    avg_tg_high = high_df["TG"].mean()

    # 高风险中老年人占比
    old_high = (high_df["年龄"] > 55).sum() / n_high * 100

    print("""
╔══════════════════════════════════════════════════════════════╗
║                糖尿病高风险人群健康管理建议                    ║
╚══════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
一、筛查优先级建议
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")

    # Top-3 贡献变量
    top3 = [v for v, _ in contrib_sorted[:3]]
    top3_names = [VAR_DISPLAY.get(v, v) for v in top3]
    print("""
  基于Logistic GAM模型分析，以下变量对糖尿病风险贡献最大：
  1. %s
  2. %s
  3. %s

  筛查优先级排序：
  - 极高优先：年龄>55岁 且 TG>2.3 mmol/L 的个体（高风险亚群聚集）
  - 高度优先：年龄>55岁 或 TG>2.3 mmol/L 的个体
  - 中度优先：具备上述任一风险因素且有其他代谢异常（如LDL-C偏高、WBC偏高）
  - 建议对以上人群优先安排口服葡萄糖耐量试验（OGTT）确认诊断
""" % (top3_names[0], top3_names[1], top3_names[2]))

    # 高风险人群概况
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("二、高风险人群概况")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("""
  高风险人群（Top 20%%预测风险）特征概要：
  - 人数: %d / %d (%.1f%%)
  - 平均年龄: %.1f岁
  - 男性比例: %.1f%%
  - 平均TG: %.2f mmol/L
  - 年龄>55岁占比: %.1f%%
  - 年龄>55岁且TG>2.3: 重点关注亚群
""" % (n_high, n_total, high_ratio, avg_age_high,
       male_ratio_high, avg_tg_high, old_high))

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("三、干预靶点建议")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("""
  基于对该人群最突出的风险因素分析，提出以下干预靶点：

  血脂管理（核心靶点）：
  - TG偏高是该高风险人群最突出的可干预因素
  - 建议加强血脂管理健康宣教，包括低糖低脂饮食、规律运动
  - 必要时在医生指导下使用降脂药物

  血糖监测：
  - 高风险人群建议每年至少检测1次空腹血糖和糖化血红蛋白
  - 年龄>55岁合并TG>2.3者建议每半年检测1次

  生活方式干预：
  - 建议高风险人群参加系统的糖尿病预防计划
  - 体重管理（如BMI>24）、增加体力活动、减少久坐
""")

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("四、资源分配建议")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # 性别分布
    if "性别(男比例)" in [r["变量"] for r in desc_df.to_dict("records") if "变量" in r]:
        pass
    male_bias = "男性" if (df[df["高风险"]==1]["性别"] >= 0.5).mean() > 0.5 else "女性"

    print("""
  性别分布：%s在高风险人群中占比偏高，建议在%s群体中增加筛查资源投入。
  年龄分布：年龄是最重要的非修饰风险因素，建议对55岁以上人群建立常规筛查机制。
  资源效率：将OGTT等确诊检查资源集中在高危亚群（年龄>55且TG>2.3），
           可在有限资源下最大化检出效率。
""" % (male_bias, male_bias))

    print(SEP)


# ══════════════════════════════════════════════════════════
#  可视化
# ══════════════════════════════════════════════════════════

def plot_risk_distribution(df, risk_col="风险概率"):
    """4.1 风险等级分布（饼图 + 柱状图）"""
    levels, bins = risk_stratify_5(df[risk_col].values)
    counts = [np.sum(levels == i) for i in range(len(bins)-1)]
    pcts = [c/len(df)*100 for c in counts]
    n_levels = len(counts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # 饼图
    wedges, texts, autotexts = ax1.pie(
        counts, labels=None, autopct="%.1f%%",
        colors=RISK_COLORS[:n_levels], startangle=90,
        textprops={"fontproperties":_CN_FP, "fontsize":9})
    ax1.set_title("附件2人群风险等级分布", fontproperties=_CN_FP, fontweight="bold")

    # 柱状图
    x = np.arange(n_levels)
    bars = ax2.bar(x, pcts, color=RISK_COLORS[:n_levels], alpha=0.8, width=0.5)
    ax2.set_xticks(x)
    lvls = [RISK_LABELS[i] + "\n[%.3f,%.3f)" % (bins[i], bins[i+1]) for i in range(n_levels)]
    ax2.set_xticklabels(lvls, fontproperties=_CN_FP, fontsize=7.5)
    ax2.set_ylabel("占比 (%)", fontproperties=_CN_FP)
    ax2.set_title("各风险等级人数占比", fontproperties=_CN_FP, fontweight="bold")
    for bar, p in zip(bars, pcts):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                "%.1f%%" % p, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题4_风险等级分布.png"), dpi=300)
    plt.close()
    print("  -> 风险等级分布.png")


def plot_radar(df, risk_col="风险概率"):
    """个体风险画像雷达图（选取高风险/中风险/低风险典型个体）"""
    # 选取3个典型个体：高风险(>P90)、中风险(P45-P55)、低风险(<P10)
    probs = df[risk_col].values
    def pick_closest(target_pct):
        thr = np.percentile(probs, target_pct)
        idx = np.argmin(np.abs(probs - thr))
        return idx

    radar_indices = [pick_closest(95), pick_closest(50), pick_closest(5)]
    radar_labels_cn = ["高风险(典型)","中等风险(典型)","低风险(典型)"]
    radar_colors = ["#E74C3C","#F39C12","#2ECC71"]

    # 选取数值型变量（排除性别和年龄_TG交互，用可解释的连续变量）
    radar_vars = ["年龄","TG","RBC","MCHC","HGB","ALT","LDL_C","HCT","WBC","TC","PLT"]
    radar_display = [VAR_DISPLAY[v] for v in radar_vars]
    n_radar = len(radar_vars)

    # 归一化到[0,1]（用min-max）
    X_radar = df[radar_vars].values
    x_min, x_max = X_radar.min(axis=0), X_radar.max(axis=0)
    ranges = x_max - x_min
    ranges[ranges == 0] = 1
    X_norm = (X_radar - x_min) / ranges

    angles = np.linspace(0, 2*np.pi, n_radar, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for idx_i, (idx, lbl, clr) in enumerate(zip(radar_indices, radar_labels_cn, radar_colors)):
        values = X_norm[idx].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", lw=2, color=clr, label=lbl + " (风险=%.3f)" % probs[idx])
        ax.fill(angles, values, alpha=0.08, color=clr)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_display, fontproperties=_CN_FP, fontsize=8)
    ax.set_title("个体风险画像雷达图", fontproperties=_CN_FP, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), prop=_CN_FP)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题4_个体风险画像雷达图.png"), dpi=300)
    plt.close()
    print("  -> 个体风险画像雷达图.png")


def plot_interaction_heatmap(df, risk_col="风险概率"):
    """年龄-TG交互效应热力图"""
    age_bins = np.linspace(df["年龄"].min(), df["年龄"].max(), 20)
    tg_bins = np.linspace(df["TG"].min(), df["TG"].max(), 20)
    age_centers = (age_bins[:-1] + age_bins[1:]) / 2
    tg_centers = (tg_bins[:-1] + tg_bins[1:]) / 2

    risk_grid = np.zeros((len(tg_centers), len(age_centers)))
    count_grid = np.zeros_like(risk_grid)
    for i in range(len(age_bins)-1):
        amask = (df["年龄"] >= age_bins[i]) & (df["年龄"] < age_bins[i+1])
        if amask.sum() == 0: continue
        for j in range(len(tg_bins)-1):
            tmask = (df["TG"] >= tg_bins[j]) & (df["TG"] < tg_bins[j+1])
            mask = amask & tmask
            if mask.sum() > 0:
                risk_grid[j, i] = df.loc[mask, risk_col].mean()
                count_grid[j, i] = mask.sum()

    # 只显示有数据的格子
    risk_grid_m = np.ma.masked_where(count_grid == 0, risk_grid)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.pcolormesh(age_bins, tg_bins, risk_grid_m, cmap="RdYlBu_r",
                        shading="auto", vmin=0, vmax=0.6)
    cbar = fig.colorbar(im, ax=ax, label="平均糖尿病风险概率")
    cbar.ax.set_ylabel("平均风险概率", fontproperties=_CN_FP)

    # 标注高风险区域
    ax.axhline(2.3, color="black", ls="--", lw=1.5, alpha=0.6, label="TG=2.3临界")
    ax.axvline(55, color="gray", ls="--", lw=1.5, alpha=0.6, label="年龄=55临界")

    ax.set_xlabel("年龄", fontproperties=_CN_FP)
    ax.set_ylabel("甘油三酯 (TG, mmol/L)", fontproperties=_CN_FP)
    ax.set_title("年龄-TG交互效应: 平均糖尿病风险", fontproperties=_CN_FP, fontweight="bold")
    ax.legend(prop=_CN_FP, loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题4_年龄TG交互热力图.png"), dpi=300)
    plt.close()
    print("  -> 年龄TG交互热力图.png")


def plot_contribution(contrib_sorted, top_n=12):
    """风险因素贡献排序图"""
    vars_sorted, contribs = zip(*contrib_sorted[:top_n])
    names = [VAR_DISPLAY.get(v, v) for v in vars_sorted]
    contribs = np.array(contribs)
    pcts = contribs / contribs.sum() * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    colors_bar = [RISK_COLORS[4] if c >= np.percentile(contribs, 75)
                  else RISK_COLORS[2] if c >= np.percentile(contribs, 50)
                  else RISK_COLORS[0] for c in contribs]
    bars = ax.barh(range(len(names)), pcts, color=colors_bar, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontproperties=_CN_FP, fontsize=9)
    ax.set_xlabel("群体层面贡献占比 (%)", fontproperties=_CN_FP)
    ax.set_title("Logistic GAM风险因素贡献排序", fontproperties=_CN_FP, fontweight="bold")
    ax.invert_yaxis()
    for bar, p in zip(bars, pcts):
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                "%.1f%%" % p, va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题4_风险因素贡献排序.png"), dpi=300)
    plt.close()
    print("  -> 风险因素贡献排序.png")


def plot_risk_histogram(df, risk_col="风险概率"):
    """风险概率分布直方图（叠加临床诊断阈值参考线）"""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # 总体分布
    ax.hist(df[risk_col].values, bins=40, alpha=0.7, color="#3498DB",
            density=True, label="全人群")

    # 阈值参考线
    thresholds = {"Top20%阈值": np.percentile(df[risk_col], 80),
                  "Top10%阈值": np.percentile(df[risk_col], 90)}
    colors_lines = ["#E74C3C", "#F39C12"]
    for (lbl, thr), clr in zip(thresholds.items(), colors_lines):
        ax.axvline(thr, color=clr, ls="--", lw=2, alpha=0.8,
                   label="%s=%.4f" % (lbl, thr))

    ax.set_xlabel("预测糖尿病风险概率", fontproperties=_CN_FP)
    ax.set_ylabel("密度", fontproperties=_CN_FP)
    ax.set_title("附件2人群糖尿病风险概率分布", fontproperties=_CN_FP, fontweight="bold")
    ax.legend(prop=_CN_FP)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题4_风险概率分布直方图.png"), dpi=300)
    plt.close()
    print("  -> 风险概率分布直方图.png")


def plot_age_tg_scatter(df, risk_col="风险概率"):
    """年龄-TG散点图，颜色映射风险概率"""
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(df["年龄"], df["TG"], c=df[risk_col], cmap="RdYlBu_r",
                    s=15, alpha=0.5, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax, label="糖尿病风险概率")
    cbar.ax.set_ylabel("风险概率", fontproperties=_CN_FP)
    ax.axhline(2.3, color="red", ls="--", lw=1.5, alpha=0.6, label="TG=2.3临界")
    ax.axvline(55, color="gray", ls="--", lw=1.5, alpha=0.6, label="年龄=55临界")
    ax.set_xlabel("年龄", fontproperties=_CN_FP)
    ax.set_ylabel("甘油三酯 (TG, mmol/L)", fontproperties=_CN_FP)
    ax.set_title("年龄-TG与糖尿病风险散点图", fontproperties=_CN_FP, fontweight="bold")
    ax.legend(prop=_CN_FP)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题4_年龄TG散点图.png"), dpi=300)
    plt.close()
    print("  -> 年龄TG散点图.png")


def plot_gender_age_risk(df, risk_col="风险概率"):
    """性别-年龄组风险对比图"""
    age_bins = [0, 30, 40, 50, 60, 70, 100]
    age_labels = ["<30","30-40","40-50","50-60","60-70",">70"]
    df["年龄组"] = pd.cut(df["年龄"], bins=age_bins, labels=age_labels, right=False)
    df["性别标签"] = df["性别"].apply(lambda x: "男" if x >= 0.5 else "女")

    # 各年龄组-性别的平均风险
    group_stats = df.groupby(["年龄组","性别标签"])[risk_col].agg(["mean","std","count"]).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(age_labels))
    width = 0.35
    for gi, gender in enumerate(["男","女"]):
        subset = group_stats[group_stats["性别标签"] == gender]
        means = [subset.loc[subset["年龄组"]==lbl, "mean"].values[0] if lbl in subset["年龄组"].values else 0
                 for lbl in age_labels]
        errs = [subset.loc[subset["年龄组"]==lbl, "std"].values[0] if lbl in subset["年龄组"].values else 0
                for lbl in age_labels]
        ax.bar(x + gi*width, means, width, yerr=errs, capsize=3,
               color=["#E74C3C","#3498DB"][gi], alpha=0.75, label=gender)

    ax.set_xlabel("年龄组", fontproperties=_CN_FP)
    ax.set_ylabel("平均糖尿病风险概率", fontproperties=_CN_FP)
    ax.set_title("不同年龄-性别组的平均糖尿病风险", fontproperties=_CN_FP, fontweight="bold")
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(age_labels)
    ax.legend(prop=_CN_FP)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题4_性别年龄风险对比.png"), dpi=300)
    plt.close()
    print("  -> 性别年龄风险对比.png")


# ══════════════════════════════════════════════════════════
#  输出个体风险结果（CSV）
# ══════════════════════════════════════════════════════════

def save_risk_output(df, risk_col="风险概率", group_col="高风险"):
    """保存个体风险预测结果到CSV"""
    levels, bins = risk_stratify_5(df[risk_col].values)
    df_out = pd.DataFrame({
        "id": df["id"].values if "id" in df.columns else range(1, len(df)+1),
        "预测糖尿病风险概率": df[risk_col].values,
        "风险等级": [RISK_LABELS[l] for l in levels],
        "是否高风险(Top20%)": ["是" if g == 1 else "否" for g in df[group_col]]
    })
    outpath = os.path.join(OUTPUT_DIR, "问题4_个体风险预测结果.csv")
    df_out.to_csv(outpath, index=False, encoding="utf-8-sig")
    print("\n  个体风险预测结果已保存: 问题4_个体风险预测结果.csv")
    print("  共 %d 条记录" % len(df_out))
    # 打印统计
    for lvl in RISK_LABELS:
        n = (df_out["风险等级"] == lvl).sum()
        print("    %-8s: %d人 (%.1f%%)" % (lvl, n, n/len(df_out)*100))


# ══════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════

def main():
    print(SEP)
    print("  问题4: 基于Logistic GAM的糖尿病风险预测与高风险人群分析")
    print(SEP)

    # ── 4.1 训练模型 + 预测 ──
    print("\n" + SEP2)
    print("  [4.1] 训练Logistic GAM模型")
    print(SEP2)
    df_train = prepare_train_data(dm_threshold=7.0)
    sw = _sample_weights(df_train["DM"].values)
    gam = fit_gam(df_train, FINAL_VARS, sample_weight=sw)
    X_train = df_train[FINAL_VARS].values

    print("\n" + SEP2)
    print("  [4.1] 附件2风险预测")
    print(SEP2)
    df_pred = prepare_pred_data()
    X_test = np.asarray(df_pred[FINAL_VARS].values, dtype=float)
    risk_prob = gam.predict_proba(X_test)
    df_pred["风险概率"] = risk_prob
    print("  附件2风险预测完成: 最小值=%.4f, 最大值=%.4f, 均值=%.4f" %
          (risk_prob.min(), risk_prob.max(), risk_prob.mean()))

    # ── 4.2 高风险人群识别（必须在save_risk_output之前） ──
    print("\n" + SEP2)
    print("  [4.2] 高风险人群识别")
    print(SEP2)
    df_pred, high_thr = identify_high_risk(df_pred, risk_col="风险概率", top_pct=0.2)

    # ── 输出个人风险概率（含风险等级和是否高风险） ──
    save_risk_output(df_pred)

    # ── 4.3 高风险人群特征分析 ──
    desc_df, sig_results = descriptive_analysis(df_pred)
    df_pred = cross_analysis(df_pred)

    # ── 4.4 GAM风险因素贡献分析 ──
    contrib_sorted = contribution_analysis(gam, X_train, FINAL_VARS)

    # ── 4.5 健康管理建议 ──
    generate_recommendations(df_pred, sig_results, contrib_sorted, desc_df)

    # ── 可视化 ──
    print("\n" + SEP2)
    print("  [可视化]")
    print(SEP2)
    plot_risk_distribution(df_pred)
    plot_radar(df_pred)
    plot_interaction_heatmap(df_pred)
    plot_contribution(contrib_sorted)
    plot_risk_histogram(df_pred)
    plot_age_tg_scatter(df_pred)
    plot_gender_age_risk(df_pred)

    # ── 输出文件列表 ──
    print("\n" + SEP)
    print("  问题4分析完成")
    print(SEP)
    print("\n  输出文件:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print("    " + f)


if __name__ == "__main__":
    main()
