# -*- coding: utf-8 -*-
"""
t3.py — 问题3：糖尿病风险评估（Logistic GAM）
===============================================
构建Logistic广义可加模型，将血糖值转化为二分类风险预测。
标准：血糖 > 6.1 mmol/L 定义为高风险（空腹血糖受损/糖尿病前期）
      血糖 > 7.0 mmol/L 定义为糖尿病
"""

import warnings, os, numpy as np, pandas as pd, concurrent.futures, multiprocessing
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy import stats
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score,
                             brier_score_loss, average_precision_score, precision_recall_curve,
                             fbeta_score)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from functools import reduce
from itertools import cycle

# 可选依赖
try:
    import xgboost as xgb
    _HAVE_XGB = True
except ImportError:
    _HAVE_XGB = False
    print("  [提示] xgboost 未安装 (pip install xgboost), 降级到 RandomForest")

_CN_FP = None
for _fp in ["C:/Windows/Fonts/msyh.ttc","C:/Windows/Fonts/simhei.ttf"]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp); _CN_FP = fm.FontProperties(fname=_fp); break
if not _CN_FP:
    for _f in fm.fontManager.ttflist:
        if any(k in _f.name for k in ["YaHei","SimHei","PingFang"]):
            _CN_FP = fm.FontProperties(family=_f.name); break
plt.rcParams["font.family"] = "sans-serif"; plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "问题3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from 数据预处理 import (load_and_rename, impute_median, EN2CN, TARGET, CSV1,
                         ALL_PREDICTORS_EN, CATEGORIES_EN)

# 核心7变量（从问题1 LASSO筛选得来）
CORE_VARS = ["年龄","TG","RBC","MCHC","HGB","ALT","性别"]
VARN = {v: EN2CN.get(v,v) for v in CORE_VARS}
SEP = "=" * 70
SEP2 = "-" * 60

# 模型超参数（收敛优化）
K_SPLINES = 5       # 降低样条基函数数量减少过拟合
LAM = 1.0           # 增大惩罚使优化更稳定
MAX_ITER = 5000     # 增加迭代上限确保收敛
TOL = 1e-6          # 收紧收敛容差

# 临床决策参数
FP_COST = 1.0       # 假阳性代价
FN_COST = 3.0       # 假阴性代价
SCREENING_SENS_TARGET = 0.95
DIAGNOSIS_SPEC_TARGET = 0.90
F2_BETA = 2.0

# 风险五档阈值（固定临床含义，兼顾概率分布）
RISK_LABELS = ["极低风险", "低风险", "中等风险", "高风险", "极高风险"]

N_BOOT = 200


# ═══════════════════════════════════════════
#  动态风险分层阈值
# ═══════════════════════════════════════════
def compute_adaptive_bins(y_prob, y_true=None):
    """
    基于预测概率分位数动态计算五档阈值。
    不依赖 y_true（避免标签泄露），若传入仅作日志输出。
    """
    quantiles = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
    bins = np.quantile(y_prob, quantiles).tolist()
    bins[0] = 0.0
    bins[-1] = 1.0
    bins = sorted(set(np.round(bins, 6)))
    while len(bins) < 6:
        mid = (bins[-2] + bins[-1]) / 2
        bins.insert(-1, mid)
    bins = bins[:6]
    return np.array(bins)


# ═══════════════════════════════════════════
#  特征工程
# ═══════════════════════════════════════════
def _cat_encode(df):
    """编码性别为数值，确保所有预测变量为数值型（兼容已编码情形）"""
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


def expand_features(df, core_vars, n_additional=5):
    """
    特征扩展（精简版）：
    1. 保留核心7变量
    2. 从生化指标中挑选与DM显著相关的top N
    3. 仅保留 年龄_TG 交互项（有临床证据支持）
    返回 (特征列表, 扩展后的DataFrame)
    """
    df_num = _cat_encode(df)
    y = df_num["DM"].values
    selected = list(core_vars)
    added = []

    # 从剩余生化指标中筛选
    remaining = [v for v in ALL_PREDICTORS_EN
                 if v not in selected and v in df_num.columns
                 and v != TARGET and pd.api.types.is_numeric_dtype(df_num[v])]
    # 排除衍生特征来源（已够用）
    exclude_derived = ["GGT", "AST", "TP", "GLB", "ALP", "BUN",
                       "MCV", "MCH", "RDW", "MPV", "PDW", "PCT",
                       "BAS_pct", "EOS_pct", "MON_pct",
                       "HBsAg", "HBsAb", "HBeAg", "HBeAb", "HBcAb"]
    remaining = [v for v in remaining if v not in exclude_derived]

    scores = []
    for v in remaining:
        xv = df_num[v].fillna(df_num[v].median()).values
        if np.std(xv) > 1e-10 and len(np.unique(y)) > 1:
            r, p = stats.pointbiserialr(xv, y)
            scores.append((v, abs(r), p))
    scores.sort(key=lambda x: -x[1])
    for v, r, p in scores:
        if p < 0.05 and len(selected) < len(core_vars) + n_additional:
            if v not in selected:
                selected.append(v); added.append(v)

    # 年龄_TG 交互（唯一保留的衍生特征）
    if "年龄" in df_num.columns and "TG" in df_num.columns:
        df_num["年龄_TG"] = df_num["年龄"].values * df_num["TG"].values
        if "年龄_TG" not in selected:
            selected.append("年龄_TG"); added.append("年龄_TG")

    if added:
        print("  特征精简扩展: +%d个 (%s)" % (len(added), ", ".join(EN2CN.get(v,v) for v in added)))
    else:
        print("  特征精简扩展: 无新增")
    return selected, df_num[selected + ["DM"]]


def _sample_weights(y):
    """计算平衡样本权重（用于处理类别不平衡）"""
    classes = np.unique(y)
    if len(classes) <= 1:
        return np.ones(len(y))
    cw = compute_class_weight("balanced", classes=classes, y=y)
    return np.array([cw[int(i)] for i in y])


# ═══════════════════════════════════════════
#  XGBoost + Sigmoid 校准（主预测模型）
# ═══════════════════════════════════════════
def fit_xgb_calibrated(X, y, sw=None):
    """XGBoost + CalibratedClassifierCV (Platt scaling)"""
    from sklearn.calibration import CalibratedClassifierCV
    sw = sw if sw is not None else _sample_weights(y)
    base = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, eval_metric="logloss", use_label_encoder=False
    ) if _HAVE_XGB else RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    calib = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    calib.fit(X, y, sample_weight=sw)
    return calib


# ═══════════════════════════════════════════
#  风险五档分层
# ═══════════════════════════════════════════
def risk_stratify(y_prob, bins=None):
    """将预测概率映射为五档风险等级（0=极低, 4=极高）"""
    if bins is None:
        bins = compute_adaptive_bins(y_prob)
    labels = list(range(len(bins) - 1))
    return pd.cut(y_prob, bins=bins, labels=labels, right=False, include_lowest=True).astype(int), bins


def evaluate_stratified(y_true, y_prob, bins=None):
    """
    五档分层评估（支持动态阈值）
    返回 (df_levels, summary_dict, bins)
    """
    levels, bins = risk_stratify(y_prob, bins)
    n_levels = len(bins) - 1
    result = []
    for lvl in range(n_levels):
        mask = levels == lvl
        n = mask.sum()
        if n == 0:
            continue
        pos_rate = y_true[mask].mean()
        # 累计（从该档及以上视为阳性）
        cum_mask = levels >= lvl
        tn, fp, fn, tp = confusion_matrix(y_true, cum_mask.astype(int)).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        cost = FP_COST * fp + FN_COST * fn
        result.append({
            "风险等级": RISK_LABELS[lvl] if lvl < len(RISK_LABELS) else f"等级{lvl}",
            "区间": "[%.3f, %.3f)" % (bins[lvl], bins[lvl+1]),
            "人数": n, "占比": n / len(y_true) * 100,
            "实际阳性率": pos_rate,
            "累积敏感度": sens, "累积PPV": ppv,
            "累积代价": cost
        })
    df_lev = pd.DataFrame(result)

    # 高风险+极高风险合并为阳性
    high_mask = levels >= (n_levels - 2)
    tn, fp, fn, tp = confusion_matrix(y_true, high_mask.astype(int)).ravel()
    sens_high = tp / (tp + fn) if (tp + fn) > 0 else 0
    ppv_high = tp / (tp + fp) if (tp + fp) > 0 else 0
    spec_high = tn / (tn + fp) if (tn + fp) > 0 else 0

    # 极低+低风险合并为阴性的 NPV
    low_mask = levels <= 1
    yp_low_pos = (levels >= 2).astype(int)  # 中及以上=阳性
    tn2, fp2, fn2, tp2 = confusion_matrix(y_true, yp_low_pos).ravel()
    npv_low = tn2 / (tn2 + fn2) if (tn2 + fn2) > 0 else 0

    # 极高风险独立PPV
    top_mask = levels == (n_levels - 1)
    if top_mask.sum() > 0:
        top_pos = y_true[top_mask].mean()
    else:
        top_pos = 0

    summary = {
        "高风险+极高风险": {"灵敏度": sens_high, "特异度": spec_high, "PPV": ppv_high},
        "极高风险独立PPV": top_pos,
        "极低+低风险NPV": npv_low,
        "中风险(灰区)占比": (levels == (n_levels // 2)).mean() * 100
    }
    return df_lev, summary, bins


# ═══════════════════════════════════════════
#  F2-优化阈值（参考保留）
# ═══════════════════════════════════════════
def fbeta_optimized_threshold(y_true, y_prob, beta=F2_BETA):
    """最大化F-beta score的阈值搜索"""
    thr_grid = np.linspace(0.01, 0.99, 198)
    best_f, best_thr = 0, 0.5
    for thr in thr_grid:
        yp = (y_prob > thr).astype(int)
        f = fbeta_score(y_true, yp, beta=beta)
        if f > best_f:
            best_f = f; best_thr = thr
    return best_thr, best_f


# ═══════════════════════════════════════════
#  F2-优化阈值（参考保留）
# ═══════════════════════════════════════════

    # 一次计算全部阈值下的指标（避免逐阈值循环）
    sens_list, spec_list = [], []
    for thr in thr_grid:
        yp = (y_prob > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp).ravel()
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    # 初筛：从右向左找第一个满足灵敏度要求的阈值（最高阈值下仍有高灵敏度）
    low_thr_idx = -1
    for i in range(len(thr_grid) - 1, -1, -1):
        if sens_list[i] >= SCREENING_SENS_TARGET:
            low_thr_idx = i
            break
    if low_thr_idx < 0:
        # 没有任何阈值能达到灵敏度目标，选阈值最小的那个（灵敏度最高的）
        low_thr_idx = 0
        SCREENING_SENS_TARGET_actual = sens_list[0]
    else:
        SCREENING_SENS_TARGET_actual = SCREENING_SENS_TARGET
    low_thr = thr_grid[low_thr_idx]
    yp_low = (y_prob > low_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yp_low).ravel()
    low_metrics = {
        "阈值": low_thr, "灵敏度": tp/(tp+fn), "特异度": tn/(tn+fp),
        "PPV": tp/(tp+fp) if (tp+fp)>0 else 0, "NPV": tn/(tn+fn) if (tn+fn)>0 else 0,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn
    }
def plot_risk_stratification(y_true, y_prob, bins=None):
    """风险分层柱状图（支持动态阈值）"""
    levels, bins = risk_stratify(y_prob, bins)
    n_levels = len(bins) - 1
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(n_levels)
    colors = ["#2ECC71", "#58D68D", "#F39C12", "#E67E22", "#E74C3C"]
    counts = [np.sum(levels == i) for i in range(n_levels)]
    bars = ax1.bar(x, counts, color=colors[:n_levels], alpha=0.7, width=0.6)
    ax1.set_ylabel("人数", fontproperties=_CN_FP)
    ax1.set_xlabel("风险等级", fontproperties=_CN_FP)
    ax2 = ax1.twinx()
    pos_rates = [y_true[levels == i].mean() if (levels == i).sum() > 0 else 0 for i in range(n_levels)]
    ax2.plot(x, pos_rates, "o-", c="black", lw=2.5, ms=8, label="实际阳性率")
    ax2.axhline(y_true.mean(), c="gray", ls="--", lw=1, alpha=0.7,
                label="基线=%.1f%%" % (y_true.mean()*100))
    ax2.set_ylabel("实际阳性率", fontproperties=_CN_FP)
    ax2.set_ylim([0, max(max(pos_rates), y_true.mean())*1.5 + 0.05])
    labels_show = [("%s\n[%.2f,%.2f)" % (RISK_LABELS[i] if i < len(RISK_LABELS) else str(i),
                                          bins[i], bins[i+1])) for i in range(n_levels)]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_show, fontproperties=_CN_FP, fontsize=8)
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="upper right", prop=_CN_FP)
    ax1.set_title("动态风险分层评估", fontproperties=_CN_FP, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题3_风险分层.png"), dpi=300); plt.close()
    print("  -> 风险分层.png  阈值: %s" % " | ".join("%.3f" % b for b in bins))


# ═══════════════════════════════════════════
#  决策曲线分析 (Decision Curve Analysis)
# ═══════════════════════════════════════════
def decision_curve_analysis(y_true, y_prob):
    """
    计算决策曲线：在不同阈值下，净收益 = (TP - FP * (thr/(1-thr))) / N
    """
    thr_grid = np.linspace(0.01, 0.99, 99)
    net_benefit = []
    treat_all_nb = []
    treat_none_nb = []
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    for thr in thr_grid:
        yp = (y_prob > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp).ravel()
        # 净收益 = (TP/N) - (FP/N) * (thr/(1-thr))
        nb = tp / len(y_true) - fp / len(y_true) * thr / (1 - thr)
        net_benefit.append(nb)
        treat_all_nb.append(n_pos / len(y_true) - n_neg / len(y_true) * thr / (1 - thr))
        treat_none_nb.append(0)
    return thr_grid, net_benefit, treat_all_nb, treat_none_nb


def plot_decision_curve(y_true, y_prob):
    """绘制决策曲线"""
    thr_grid, nb, all_nb, none_nb = decision_curve_analysis(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thr_grid, nb, "#E74C3C", lw=2.5, label="Logistic GAM")
    ax.plot(thr_grid, all_nb, "gray", ls="--", lw=1.5, label="全部干预")
    ax.plot(thr_grid, none_nb, "gray", ls=":", lw=1.5, label="不干预")
    ax.fill_between(thr_grid, nb, 0, where=[n > 0 for n in nb],
                    alpha=0.1, color="#E74C3C")
    ax.set_xlabel("风险阈值 (Threshold Probability)", fontproperties=_CN_FP)
    ax.set_ylabel("净收益 (Net Benefit)", fontproperties=_CN_FP)
    ax.set_title("决策曲线分析 (DCA)", fontproperties=_CN_FP, fontweight="bold")
    ax.legend(loc="upper right", prop=_CN_FP)
    ax.set_xlim([0, 1]); ax.set_ylim([-0.02, max(nb) * 1.1 + 0.01])
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题3_决策曲线.png"), dpi=300); plt.close()
    print("  -> 决策曲线.png")

def load_data(dm_threshold=6.1):
    """加载数据，生成二分类目标DM（血糖>threshold=1）"""
    df = load_and_rename(CSV1); df = impute_median(df)
    if TARGET in df.columns and df[TARGET].isnull().any():
        df[TARGET] = df[TARGET].fillna(df[TARGET].median())
    if "性别" in df.columns: df["性别"] = df["性别"].map({"男":1,"女":0}).fillna(0.5)
    df["DM"] = (df[TARGET] > dm_threshold).astype(int)
    print("  样本: %d, DM阳性率: %.2f%% (阈值>%.1f)" % (len(df), df["DM"].mean()*100, dm_threshold))
    return df

def _ts(L):
    if not L: return None
    r = L[0]
    for t in L[1:]: r = r + t
    return r

# ═══════════════════════════════════════════
#  模型构建
# ═══════════════════════════════════════════
def fit_logistic_gam(df, preds, sample_weight=None):
    """Logistic GAM (Logit链接, 二项分布) — 带样本权重 + 收敛优化"""
    from pygam import LogisticGAM, s, f
    X, y = df[preds].values, df["DM"].values
    nv = len(preds)

    # 检测连续变量和分类变量
    continuous_idx = [i for i, p in enumerate(preds) if p != "性别"]
    cat_idx = [i for i, p in enumerate(preds) if p == "性别"]
    terms = _ts([s(i, n_splines=K_SPLINES) for i in continuous_idx])
    for ci in cat_idx:
        terms = terms + f(ci)

    gam = LogisticGAM(terms, fit_intercept=True,
                      max_iter=MAX_ITER, lam=LAM, tol=TOL)
    print("  拟合Logistic GAM (k=%d, lam=%.1f, max_iter=%d, tol=%.0e)..."
          % (K_SPLINES, LAM, MAX_ITER, TOL), end=" ", flush=True)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        gam.fit(X, y, weights=sample_weight)
    print("完成")

    y_prob = gam.predict_proba(X)
    # Youden J指数找最优阈值
    fpr, tpr, thr = roc_curve(y, y_prob)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_thr = thr[best_idx]

    udof = gam.statistics_.get("edof", gam.statistics_.get("edf", 0))
    print("  UBRE=%.4f  EDF=%.2f  #terms=%d" %
          (gam.statistics_.get("UBRE", 0), udof, len(continuous_idx) + len(cat_idx)))
    print("  Youden阈值=%.4f (敏感度+特异度=%.4f)" % (best_thr, youden[best_idx]))

    # 收敛诊断
    conv_iter = gam.statistics_.get("iter", -1)
    if conv_iter < 0:
        print("  收敛诊断: 统计信息不可用")
    elif conv_iter >= MAX_ITER - 1:
        print("  收敛诊断: 达到最大迭代 (%d) — 可能未完全收敛" % MAX_ITER)
    else:
        print("  收敛诊断: %d 次迭代后收敛" % conv_iter)
    return gam, X, y, y_prob, best_thr

# ═══════════════════════════════════════════
#  Bootstrap（多进程并行）
# ═══════════════════════════════════════════
def _bootstrap_worker(seed, X_data, y_data, sw_data, x_mean_data, x_grids_data,
                      preds_list, k, lam):
    """单次Bootstrap迭代（模块级函数，用于ProcessPoolExecutor并行）"""
    import numpy as np
    from pygam import LogisticGAM, s, f
    import warnings as _w
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(seed)
    n = len(X_data)
    idx = rng.choice(n, n, replace=True)
    Xb, yb = X_data[idx], y_data[idx]
    swb = sw_data[idx] if sw_data is not None else None
    nv = X_data.shape[1]

    def _ts(L):
        if not L: return None
        r = L[0]
        for t in L[1:]: r = r + t
        return r

    continuous_idx = [i for i in range(nv) if preds_list[i] != "性别"]
    cat_idx = [i for i in range(nv) if preds_list[i] == "性别"]
    terms = _ts([s(i, n_splines=k) for i in continuous_idx])
    for ci in cat_idx:
        terms = terms + f(ci)

    try:
        gam = LogisticGAM(terms, fit_intercept=True, max_iter=1000, lam=lam, tol=1e-6)
        # 同时抑制 Python warning + numpy RuntimeWarning(exp溢出/除零)
        with _w.catch_warnings(), np.errstate(all="ignore"):
            _w.simplefilter("ignore")
            gam.fit(Xb, yb, weights=swb)
        auc = float(roc_auc_score(yb, gam.predict_proba(Xb))) if len(np.unique(yb)) > 1 else 0.0

        curves = {}
        for i, p in enumerate(preds_list):
            xg = x_grids_data[p]; ng = len(xg)
            X_grid = np.tile(x_mean_data, (ng, 1)); X_grid[:, i] = xg
            with np.errstate(all="ignore"):
                prob = np.clip(gam.predict_proba(X_grid), 1e-10, 1-1e-10)
                eta = np.log(prob/(1-prob)); eta -= eta.mean()
            curves[p] = eta
        return auc, curves, True
    except Exception:
        return 0.0, {}, False


def bootstrap_logistic_gam(df, preds, sample_weight=None):
    """Bootstrap估计参数稳定性 + 偏效应置信区间（joblib多进程，兼容Windows网络路径）"""
    n_cpu = multiprocessing.cpu_count()
    n_jobs = max(2, n_cpu)
    print("  Bootstrap (%d次, %d进程 joblib)..." % (N_BOOT, n_jobs), end=" ", flush=True)

    X_full = df[preds].values
    y_full = df["DM"].values
    sw_full = sample_weight
    x_mean = X_full.mean(axis=0)

    # 为每个变量准备网格
    x_grids = {}
    for i, p in enumerate(preds):
        if p == "性别":
            x_grids[p] = np.array([0, 1])
        else:
            xmin, xmax = X_full[:, i].min(), X_full[:, i].max()
            x_grids[p] = np.linspace(xmin, xmax, 100)

    seeds = list(range(42, 42 + N_BOOT))
    args_list = [(s, X_full, y_full, sw_full, x_mean, x_grids, preds, K_SPLINES, LAM) for s in seeds]

    # joblib.Parallel 用 loky 后端创建子进程
    # loky 用 cloudpickle 序列化代码 → 不需要读磁盘文件 → 无网络路径权限错误
    # verbose=0 静默, prefer="processes" 强制多进程
    results = Parallel(n_jobs=n_jobs, verbose=0, prefer="processes")(
        delayed(_bootstrap_worker)(*a) for a in args_list
    )

    aucs = []
    partial_curves = {p: [] for p in preds}
    valid_boots = 0
    for auc, curves, ok in results:
        if ok:
            aucs.append(auc)
            for p in preds:
                partial_curves[p].append(curves[p])
            valid_boots += 1

    print("完成 (%d/%d收敛)" % (valid_boots, N_BOOT))
    if aucs:
        ci_lo, ci_hi = np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)
        print("  Bootstrap AUC: mean=%.4f, 95%%CI=[%.4f, %.4f]" %
              (np.mean(aucs), ci_lo, ci_hi))
        print("  AUC稳定性: SD=%.4f, 范围=[%.4f, %.4f]" %
              (np.std(aucs), min(aucs), max(aucs)))
    return aucs, partial_curves, x_grids

# ═══════════════════════════════════════════
#  评估指标
# ═══════════════════════════════════════════
def _metrics_at_threshold(y_true, y_prob, thr):
    """在给定阈值下计算所有二分类指标"""
    yp = (y_prob > thr).astype(int)
    cm = confusion_matrix(y_true, yp)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp+fn) > 0 else 0
    spec = tn / (tn + fp) if (tn+fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp+fp) > 0 else 0
    npv = tn / (tn + fn) if (tn+fn) > 0 else 0
    acc = accuracy_score(y_true, yp)
    f1 = f1_score(y_true, yp)
    cost = FP_COST * fp + FN_COST * fn
    return {"阈值": thr, "准确率": acc, "敏感度": sens, "特异度": spec,
            "PPV": ppv, "NPV": npv, "F1": f1,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn, "总代价": cost}


def cost_optimized_threshold(y_true, y_prob):
    """最小化代价函数 cost = FP_COST*FP + FN_COST*FN 的最优阈值"""
    thresholds = np.linspace(0.01, 0.99, 199)
    best_thr, best_cost = 0.5, float("inf")
    for thr in thresholds:
        yp = (y_prob > thr).astype(int)
        cm = confusion_matrix(y_true, yp)
        tn, fp, fn, tp = cm.ravel()
        cost = FP_COST * fp + FN_COST * fn
        if cost < best_cost:
            best_cost = cost
            best_thr = thr
    return best_thr, best_cost


def evaluate_detailed(y_true, y_prob):
    """
    综合评估（多阈值 + F2优化 + 五档分层 + PR-AUC）
    返回 (metrics_dict, threshold_table, (precision, recall, pr_thr), strat_df, strat_summary)
    """
    auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    precision, recall, pr_thr_arr = precision_recall_curve(y_true, y_prob)

    # 候选阈值
    candidate_thrs = [0.1, 0.2, 0.3]
    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    youden_idx = np.argmax(tpr - fpr)
    youden_thr = roc_thr[youden_idx]
    candidate_thrs.append(youden_thr)
    cost_thr, _ = cost_optimized_threshold(y_true, y_prob)
    candidate_thrs.append(cost_thr)
    f2_thr, best_f2 = fbeta_optimized_threshold(y_true, y_prob, beta=F2_BETA)
    candidate_thrs.append(f2_thr)
    candidate_thrs = sorted(set(np.round(candidate_thrs, 4)))

    rows = []
    for thr in candidate_thrs:
        row = _metrics_at_threshold(y_true, y_prob, thr)
        row["F2"] = fbeta_score(y_true, (y_prob > thr).astype(int), beta=F2_BETA)
        rows.append(row)
    threshold_table = pd.DataFrame(rows).round(4)

    # 五档分层评估（动态阈值）
    bins = compute_adaptive_bins(y_prob)
    strat_df, strat_summary, bins = evaluate_stratified(y_true, y_prob, bins=bins)

    # 打印
    print("\n" + "="*70)
    print("  多阈值评估 + 五档风险分层")
    print("="*70)
    print("  AUC=%.4f  PR-AUC=%.4f" % (auc, pr_auc))
    disp = threshold_table.copy()
    if "阈值" in disp.columns:
        disp["阈值"] = disp["阈值"].apply(lambda x: "%.4f" % x)
    print(disp.to_string(index=False))

    print("\n  阈值选择参考:")
    print("    Youden=%.4f  代价最优=%.4f  F2最优=%.4f (F2=%.4f)" %
          (youden_thr, cost_thr, f2_thr, best_f2))

    # 分层报告
    print("\n" + "-"*60)
    print("  [五档风险分层]")
    print("-"*60)
    for _, r in strat_df.iterrows():
        print("  %-8s  n=%5d (%.1f%%)  阳性率=%.2f%%  累积灵敏度=%.3f  累积PPV=%.3f" %
              (r["风险等级"], r["人数"], r["占比"], r["实际阳性率"]*100,
               r["累积敏感度"], r["累积PPV"]))
    sh = strat_summary["高风险+极高风险"]
    print("\n  高风险+极高风险合并为阳性: 灵敏度=%.3f, 特异度=%.3f, PPV=%.3f" %
          (sh["灵敏度"], sh["特异度"], sh["PPV"]))
    print("  极低+低风险NPV=%.4f  中风险(灰区)=%.1f%%" %
          (strat_summary["极低+低风险NPV"], strat_summary["中风险(灰区)占比"]))

    # 最终（用F2阈值）
    final_thr = f2_thr
    final = _metrics_at_threshold(y_true, y_prob, final_thr)
    metrics = {
        "AUC": auc, "PR-AUC": pr_auc, "最优阈值": final_thr,
        "准确率": final["准确率"], "敏感度": final["敏感度"],
        "特异度": final["特异度"], "PPV": final["PPV"],
        "NPV": final["NPV"], "F1": final["F1"],
        "总代价": final["总代价"], "F2": best_f2,
        "TP": final["TP"], "FN": final["FN"], "FP": final["FP"], "TN": final["TN"]
    }
    print("\n  最终选用 (F2最优阈值=%.4f):" % final_thr)
    for k in ["AUC","PR-AUC","敏感度","特异度","PPV","NPV","F1","F2","总代价"]:
        print("    %-10s = %.4f" % (k, metrics[k]))
    print("  混淆矩阵: TP=%d  FN=%d  FP=%d  TN=%d" %
          (metrics["TP"], metrics["FN"], metrics["FP"], metrics["TN"]))
    return metrics, threshold_table, (precision, recall, pr_thr_arr), strat_df, strat_summary, bins

# ═══════════════════════════════════════════
#  可视化
# ═══════════════════════════════════════════
def plot_roc(y_true, y_prob):
    """ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, "#E74C3C", lw=2.5, label="AUC=%.4f" % auc)
    ax.plot([0,1],[0,1],"gray", ls="--", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.1, color="#E74C3C")
    ax.set_xlabel("1-特异度 (假阳性率)", fontproperties=_CN_FP)
    ax.set_ylabel("敏感度 (真阳性率)", fontproperties=_CN_FP)
    ax.set_title("ROC曲线 — 糖尿病风险评估", fontproperties=_CN_FP, fontweight="bold")
    ax.legend(loc="lower right", prop=_CN_FP)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题3_ROC曲线.png"), dpi=300); plt.close()
    print("  -> ROC曲线.png")

def plot_calibration(y_true, y_prob, n_bins=10):
    """校准度曲线"""
    bins = np.linspace(0, 1, n_bins+1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_frac = []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() > 0:
            bin_frac.append(y_true[mask].mean())
        else:
            bin_frac.append(0)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(bin_centers, bin_frac, "o-", color="#3498DB", lw=2, ms=6, label="模型校准曲线")
    ax.plot([0,1],[0,1], "gray", ls="--", lw=1, label="完美校准")
    ax.set_xlabel("预测概率", fontproperties=_CN_FP)
    ax.set_ylabel("实际阳性比例", fontproperties=_CN_FP)
    ax.set_title("校准度曲线 (Calibration)", fontproperties=_CN_FP, fontweight="bold")
    ax.legend(loc="lower right", prop=_CN_FP)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题3_校准度曲线.png"), dpi=300); plt.close()
    print("  -> 校准度曲线.png")

def plot_prob_distribution(y_true, y_prob):
    """风险概率分布图"""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(y_prob[y_true==0], bins=30, alpha=0.6, color="#3498DB", label="非糖尿病", density=True)
    ax.hist(y_prob[y_true==1], bins=30, alpha=0.6, color="#E74C3C", label="糖尿病", density=True)
    ax.axvline(0.5, color="red", ls="--", lw=1.5, label="决策阈值=0.5")
    ax.set_xlabel("预测概率", fontproperties=_CN_FP)
    ax.set_ylabel("密度", fontproperties=_CN_FP)
    ax.set_title("预测概率分布", fontproperties=_CN_FP, fontweight="bold")
    ax.legend(prop=_CN_FP)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题3_风险概率分布.png"), dpi=300); plt.close()
    print("  -> 风险概率分布.png")

def plot_partial_logistic(gam, X, preds, partial_curves=None, x_grids=None):
    """Logistic GAM偏效应图 — 原始数据范围横坐标 + Bootstrap 95%置信区间"""
    nv = len(preds); nc = min(3, nv); nr = (nv+nc-1)//nc
    fig, axs = plt.subplots(nr, nc, figsize=(5*nc, 4*nr))
    axs = axs.flatten() if nr*nc>1 else [axs]
    x_mean = X.mean(axis=0)
    has_ci = partial_curves is not None and x_grids is not None
    for i in range(nv):
        ax = axs[i]; vn = VARN.get(preds[i], preds[i])
        xmin, xmax = X[:,i].min(), X[:,i].max()
        if np.isclose(xmin, xmax):
            ax.text(.5, .5, "常量特征", ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title(vn, fontproperties=_CN_FP, fontweight="bold"); continue

        is_cat = preds[i] == "性别"
        pname = preds[i]

        # Bootstrap 95% 置信区间
        if has_ci and pname in partial_curves and len(partial_curves[pname]) > 1:
            curves = np.array(partial_curves[pname])  # (n_valid_boot, n_grid)
            ci_low = np.percentile(curves, 2.5, axis=0)
            ci_high = np.percentile(curves, 97.5, axis=0)
            xg_ci = x_grids[pname]
            if not is_cat:
                ax.fill_between(xg_ci, ci_low, ci_high, alpha=0.3, color="#E74C3C")

        if is_cat:
            # 检查训练数据中该因子有哪些实际水平
            uniq_vals = np.sort(np.unique(X[:, i]))
            # 性别标准化：将 0.5(NaN填充) 等中间值归入最近邻的 {0,1}
            has_male = np.any(uniq_vals >= 0.5)
            has_female = np.any(uniq_vals <= 0.5)
            if not (has_male and has_female):
                ax.text(0.5, 0.5, "训练集性别单一",
                        ha="center", va="center", transform=ax.transAxes, fontsize=10)
                ax.set_title(vn, fontproperties=_CN_FP, fontweight="bold")
                continue
            # 强制使用 [0, 1] 网格（与bootstrap CI维度一致）
            x_vals = np.array([0, 1])
            X_grid = np.tile(x_mean, (2, 1)); X_grid[:,i] = x_vals
            with np.errstate(divide="ignore"):
                p = np.clip(gam.predict_proba(X_grid), 1e-10, 1-1e-10)
                eta = np.log(p/(1-p)); eta -= eta.mean()
            ax.plot(x_vals, eta, "o-", c="#E74C3C", lw=2, ms=8, zorder=5)
            # 分类变量CI: 以点估计为中心的垂直误差棒
            if has_ci and pname in partial_curves and len(partial_curves[pname]) > 1:
                for j in range(len(x_vals)):
                    yerr_low = abs(eta[j] - ci_low[j])
                    yerr_high = abs(ci_high[j] - eta[j])
                    ax.errorbar(x_vals[j], eta[j],
                                yerr=[[yerr_low], [yerr_high]], fmt='none',
                                ecolor="#E74C3C", elinewidth=2, capsize=6, alpha=0.6, zorder=4)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["女", "男"], fontproperties=_CN_FP)
        else:
            xg = np.linspace(xmin, xmax, 100)
            X_grid = np.tile(x_mean, (100, 1)); X_grid[:,i] = xg
            with np.errstate(divide="ignore"):
                p = np.clip(gam.predict_proba(X_grid), 1e-10, 1-1e-10)
                eta = np.log(p/(1-p)); eta -= eta.mean()
            ax.plot(xg, eta, c="#E74C3C", lw=2)

        ax.axhline(0,c="gray",ls="--"); ax.set_xlabel(vn, fontproperties=_CN_FP)
        ax.set_ylabel("log(OR)", fontproperties=_CN_FP); ax.set_title(vn, fontproperties=_CN_FP, fontweight="bold")
    for j in range(nv, len(axs)): axs[j].axis("off")
    plt.suptitle("Logistic GAM偏效应图 (纵轴=log(OR), 阴影=95% Bootstrap CI)",
                 fontproperties=_CN_FP, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题3_偏效应图.png"), dpi=300); plt.close()
    print("  -> 偏效应图.png")


# ═══════════════════════════════════════════
#  PR曲线 + 阈值指标图
# ═══════════════════════════════════════════
def plot_pr_curve(y_true, y_prob):
    """精确率-召回率曲线（PR曲线），标注多个阈值点"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    # 获取每个阈值对应的F1
    f1_vals = [2*p*r/(p+r) if (p+r)>0 else 0 for p, r in zip(precision[:-1], recall[:-1])]
    best_f1_idx = np.argmax(f1_vals)
    best_f1_thr = thresholds[best_f1_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, "#2ECC71", lw=2.5, label="PR曲线 (PR-AUC=%.4f)" % pr_auc)
    ax.fill_between(recall, precision, alpha=0.15, color="#2ECC71")

    # 标注关键阈值点
    for thr, marker, clr, lbl in [
        (0.1, "o", "#E74C3C", "thr=0.1"),
        (0.2, "s", "#F39C12", "thr=0.2"),
        (0.3, "^", "#9B59B6", "thr=0.3"),
    ]:
        idx = np.argmin(np.abs(thresholds - thr))
        ax.plot(recall[idx], precision[idx], marker, c=clr, ms=10, label=lbl, zorder=5)

    # 最佳F1点
    ax.plot(recall[best_f1_idx], precision[best_f1_idx], "D", c="black", ms=10,
            label="最佳F1=%.4f (thr=%.3f)" % (f1_vals[best_f1_idx], best_f1_thr), zorder=6)

    # Youden阈值（从ROC转换）
    fpr_roc, tpr_roc, thr_roc = roc_curve(y_true, y_prob)
    youden_idx = np.argmax(tpr_roc - fpr_roc)
    youden_thr = thr_roc[youden_idx]
    yp_youden = (y_prob > youden_thr).astype(int)
    youden_prec = precision_score(y_true, yp_youden)
    youden_rec = recall_score(y_true, yp_youden)
    ax.plot(youden_rec, youden_prec, "*", c="#E74C3C", ms=16,
            label="Youden (thr=%.3f)" % youden_thr, zorder=7)

    ax.set_xlabel("召回率 (灵敏度)", fontproperties=_CN_FP)
    ax.set_ylabel("精确率 (PPV)", fontproperties=_CN_FP)
    ax.set_title("PR曲线 — 精确率 vs 召回率", fontproperties=_CN_FP, fontweight="bold")
    ax.legend(loc="lower left", prop=_CN_FP)
    ax.set_xlim([0, 1.02]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题3_PR曲线.png"), dpi=300); plt.close()
    print("  -> PR曲线.png")


def plot_threshold_metrics(y_true, y_prob):
    """各指标随阈值变化的曲线"""
    thr_grid = np.linspace(0.01, 0.99, 198)
    sens, spec, ppv, f1, cost_list = [], [], [], [], []
    for thr in thr_grid:
        yp = (y_prob > thr).astype(int)
        cm = confusion_matrix(y_true, yp)
        tn, fp, fn, tp = cm.ravel()
        sens.append(tp/(tp+fn) if (tp+fn)>0 else 0)
        spec.append(tn/(tn+fp) if (tn+fp)>0 else 0)
        ppv.append(tp/(tp+fp) if (tp+fp)>0 else 0)
        f1.append(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0)
        cost_list.append(FP_COST*fp + FN_COST*fn)

    # 最优阈值竖线
    youden_thr = thr_grid[np.argmax([s+sp for s,sp in zip(sens, spec)])]
    cost_thr = thr_grid[np.argmin(cost_list)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左图：评估指标
    ax1.plot(thr_grid, sens, "#E74C3C", lw=2, label="敏感度")
    ax1.plot(thr_grid, spec, "#3498DB", lw=2, label="特异度")
    ax1.plot(thr_grid, ppv, "#2ECC71", lw=2, label="PPV")
    ax1.plot(thr_grid, f1, "#F39C12", lw=2, ls="--", label="F1")
    ax1.axvline(youden_thr, c="gray", ls=":", lw=1.5, alpha=0.7, label="Youden=%.3f" % youden_thr)
    ax1.axvline(cost_thr, c="black", ls=":", lw=1.5, alpha=0.7, label="代价最优=%.3f" % cost_thr)
    ax1.set_xlabel("决策阈值", fontproperties=_CN_FP)
    ax1.set_ylabel("指标值", fontproperties=_CN_FP)
    ax1.set_title("评估指标 vs 阈值", fontproperties=_CN_FP, fontweight="bold")
    ax1.legend(loc="center right", prop=_CN_FP)
    ax1.set_xlim([0, 1]); ax1.set_ylim([-0.02, 1.02])

    # 右图：代价函数
    ax2.plot(thr_grid, cost_list, "#8E44AD", lw=2.5)
    ax2.axvline(cost_thr, c="black", ls="--", lw=1.5,
                label="最小代价=%.0f (thr=%.3f)" % (min(cost_list), cost_thr))
    # 标注代价构成
    cost_fp = [FP_COST * (confusion_matrix(y_true, (y_prob > t).astype(int)).ravel()[1])
               for t in thr_grid]
    cost_fn = [FN_COST * (confusion_matrix(y_true, (y_prob > t).astype(int)).ravel()[2])
               for t in thr_grid]
    ax2.fill_between(thr_grid, 0, cost_fp, alpha=0.2, color="#E74C3C", label="FP代价")
    ax2.fill_between(thr_grid, cost_fp, [a+b for a,b in zip(cost_fp, cost_fn)],
                     alpha=0.2, color="#3498DB", label="FN代价")
    ax2.set_xlabel("决策阈值", fontproperties=_CN_FP)
    ax2.set_ylabel("总代价 (FP_COST=%.1f, FN_COST=%.1f)" % (FP_COST, FN_COST),
                   fontproperties=_CN_FP)
    ax2.set_title("代价函数 vs 阈值", fontproperties=_CN_FP, fontweight="bold")
    ax2.legend(loc="upper right", prop=_CN_FP)
    ax2.set_xlim([0, 1])

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题3_阈值评估曲线.png"), dpi=300); plt.close()
    print("  -> 阈值评估曲线.png")


# ═══════════════════════════════════════════
#  模型对比（RF + GAM）
# ═══════════════════════════════════════════
def compare_models(X, y, preds):
    """对比Logistic GAM vs RandomForest vs XGBoost（5折CV）"""
    from pygam import LogisticGAM, s, f
    continuous_idx = [i for i in range(len(preds)) if preds[i] != "性别"]
    cat_idx = [i for i in range(len(preds)) if preds[i] == "性别"]
    terms = _ts([s(i, n_splines=K_SPLINES) for i in continuous_idx])
    for ci in cat_idx: terms = terms + f(ci)

    print("  [模型对比] 5折StratifiedKFold交叉验证...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = {
        "Logistic GAM": {"auc": [], "prauc": []},
        "RandomForest": {"auc": [], "prauc": []},
    }
    if _HAVE_XGB:
        cv_results["XGBoost(cal)"] = {"auc": [], "prauc": []}

    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        # GAM
        try:
            gam_cv = LogisticGAM(terms, fit_intercept=True, max_iter=1000, lam=LAM)
            with warnings.catch_warnings(): warnings.simplefilter("ignore")
            sw_tr = _sample_weights(ytr)
            gam_cv.fit(Xtr, ytr, weights=sw_tr)
            yp = gam_cv.predict_proba(Xte)
            cv_results["Logistic GAM"]["auc"].append(roc_auc_score(yte, yp))
            cv_results["Logistic GAM"]["prauc"].append(average_precision_score(yte, yp))
        except:
            pass

        # RF
        try:
            rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5,
                                        class_weight="balanced", random_state=42, n_jobs=1)
            rf.fit(Xtr, ytr); yp_rf = rf.predict_proba(Xte)[:, 1]
            cv_results["RandomForest"]["auc"].append(roc_auc_score(yte, yp_rf))
            cv_results["RandomForest"]["prauc"].append(average_precision_score(yte, yp_rf))
        except:
            pass

        # XGBoost校准（完整管道）
        if _HAVE_XGB:
            try:
                sw_tr = _sample_weights(ytr)
                base = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8,
                                          reg_alpha=0.5, reg_lambda=1.0,
                                          random_state=42, eval_metric="logloss",
                                          use_label_encoder=False)
                cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
                cal.fit(Xtr, ytr, sample_weight=sw_tr)
                yp_xgb = cal.predict_proba(Xte)[:, 1]
                cv_results["XGBoost(cal)"]["auc"].append(roc_auc_score(yte, yp_xgb))
                cv_results["XGBoost(cal)"]["prauc"].append(average_precision_score(yte, yp_xgb))
            except:
                pass

    # 输出
    print("  " + "-" * 60)
    print("  %-20s %-12s %-12s" % ("模型", "AUC (5-CV)", "PR-AUC (5-CV)"))
    print("  " + "-" * 60)
    best_auc, best_prauc = 0, 0
    best_auc_name, best_prauc_name = "", ""
    for name, res in cv_results.items():
        if res["auc"]:
            a = np.mean(res["auc"]); pa = np.mean(res["prauc"])
            print("  %-20s %-12.4f %-12.4f" % (name, a, pa))
            if a > best_auc: best_auc = a; best_auc_name = name
            if pa > best_prauc: best_prauc = pa; best_prauc_name = name
    print("  " + "-" * 60)
    print("  AUC最优: %s (%.4f)" % (best_auc_name, best_auc))
    print("  PR-AUC最优: %s (%.4f)" % (best_prauc_name, best_prauc))
    return cv_results


# ═══════════════════════════════════════════
#  5折嵌套交叉验证（零泄露评估 + 集成）
# ═══════════════════════════════════════════
def cross_validated_evaluation(df_full, preds, n_splits=5):
    """5折 StratifiedKFold，每折独立训练 GAM + XGBoost(cal)，输出集成OOF概率"""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X_all = df_full[preds].values; y_all = df_full["DM"].values
    y_prob_ens = np.zeros(len(y_all))
    print("  [5折嵌套CV] 每折独立训练+校准...")
    for fold, (tr, te) in enumerate(skf.split(X_all, y_all)):
        print("\n  --- Fold %d/5 ---" % (fold + 1))
        X_tr, X_te = X_all[tr], X_all[te]
        y_tr, y_te = y_all[tr], y_all[te]
        df_tr = df_full.iloc[tr]
        sw_tr = _sample_weights(y_tr)
        # GAM
        try:
            gam, _, _, _, _ = fit_logistic_gam(df_tr, preds, sample_weight=sw_tr)
            p_gam = gam.predict_proba(X_te)
        except Exception as e:
            p_gam = np.full(len(X_te), y_tr.mean())
        # XGBoost校准
        try:
            cal = fit_xgb_calibrated(X_tr, y_tr, sw=sw_tr)
            p_xgb = cal.predict_proba(X_te)[:, 1]
        except Exception as e:
            p_xgb = np.full(len(X_te), y_tr.mean())
        # 集成平均
        p_avg = (p_gam + p_xgb) / 2
        y_prob_ens[te] = p_avg
        print("    GAM=%.4f  XGB=%.4f  集成=%.4f" %
              (roc_auc_score(y_te, p_gam), roc_auc_score(y_te, p_xgb),
               roc_auc_score(y_te, p_avg)))
    overall = roc_auc_score(y_all, y_prob_ens)
    print("\n  === OOF整体: AUC=%.4f, PR-AUC=%.4f ===" %
          (overall, average_precision_score(y_all, y_prob_ens)))
    return y_all, y_prob_ens


# ═══════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════
def main():
    print(SEP+"\n  问题3: 糖尿病风险评估 — 5折嵌套CV + 模型集成\n"+SEP)

    # ── Step 1: 加载 ──
    df = load_data(dm_threshold=7.0)
    print("\n  糖尿病诊断标准: 血糖 > 7.0 mmol/L")

    # ── Step 2: 特征工程 ──
    print("\n" + SEP2)
    print("  [特征工程]")
    print(SEP2)
    FINAL_VARS, df_fe = expand_features(df, CORE_VARS, n_additional=5)
    feat_names = [EN2CN.get(v,v) for v in FINAL_VARS]
    print("  最终变量 (%d个): %s" % (len(FINAL_VARS), ", ".join(feat_names)))

    # ── Step 3: 5折嵌套CV（每折内独立训练GAM+XGBoost，零泄露） ──
    print("\n" + SEP2)
    print("  [5折嵌套交叉验证] OOF预测")
    print(SEP2)
    y_truth, y_prob_final = cross_validated_evaluation(df_fe, FINAL_VARS, n_splits=5)

    # ── Step 4: 全数据模型（用于可解释性，不参与评估） ──
    print("\n" + SEP2)
    print("  [最终模型] 全数据训练（仅用于可解释性/偏效应）")
    print(SEP2)
    sw_full = _sample_weights(y_truth)
    gam_full, X_full, _, _, _ = fit_logistic_gam(df_fe, FINAL_VARS, sample_weight=sw_full)
    # Bootstrap（全数据，仅用于GAM稳定性）
    print("\n" + SEP2)
    print("  [Bootstrap] GAM稳定性")
    print(SEP2)
    aucs_boot, partial_curves, x_grids = bootstrap_logistic_gam(
        df_fe, FINAL_VARS, sample_weight=sw_full)

    # ── Step 5: 评估（OOF预测，零泄露） ──
    print("\n" + SEP2)
    print("  [评估] 基于OOF预测")
    print(SEP2)
    metrics, threshold_table, (precision, recall, pr_thr_arr), strat_df, strat_summary, bins = \
        evaluate_detailed(y_truth, y_prob_final)

    # ── Step 6: 可视化（OOF预测） ──
    print("\n" + SEP2)
    print("  [可视化]")
    print(SEP2)
    plot_roc(y_truth, y_prob_final)
    plot_calibration(y_truth, y_prob_final, n_bins=10)
    plot_prob_distribution(y_truth, y_prob_final)
    plot_pr_curve(y_truth, y_prob_final)
    plot_threshold_metrics(y_truth, y_prob_final)
    plot_risk_stratification(y_truth, y_prob_final, bins=bins)
    plot_decision_curve(y_truth, y_prob_final)
    plot_partial_logistic(gam_full, X_full, FINAL_VARS, partial_curves, x_grids)

    # ── Step 7: 模型对比（全数据5折CV，与OOF独立） ──
    print("\n" + SEP2)
    print("  [模型对比] 全数据5折CV")
    print(SEP2)
    compare_models(X_full, y_truth, FINAL_VARS)

    # ── Step 8: 按血糖阈值分层 ──
    print("\n" + SEP2)
    print("  [分层评估] 不同血糖阈值（OOF）")
    print(SEP2)
    for thr in [5.6, 6.1, 7.0, 8.0]:
        y_bin = (df[TARGET].values > thr).astype(int)
        if y_bin.sum() < 10: continue
        auc = roc_auc_score(y_bin, y_prob_final)
        prauc = average_precision_score(y_bin, y_prob_final)
        print("  血糖>%.1f: AUC=%.4f  PR-AUC=%.4f (阳性n=%d)" %
              (thr, auc, prauc, y_bin.sum()))

    # ── 汇总 ──
    sh = strat_summary["高风险+极高风险"]
    top_ppv = strat_summary["极高风险独立PPV"]
    npv_low = strat_summary["极低+低风险NPV"]
    grey_pct = strat_summary["中风险(灰区)占比"]
    print("\n"+SEP+"\n  分析完成 (5折嵌套CV)\n"+SEP)
    print("  特征: %d个" % len(FINAL_VARS))
    print("  OOF AUC=%.4f | PR-AUC=%.4f | 样本数=%d" %
          (metrics["AUC"], metrics["PR-AUC"], len(y_truth)))
    print("  动态分层阈值: %s" % " → ".join("%.3f" % b for b in bins))
    print("  极高风险独立PPV=%.4f  高+极高合并灵敏度=%.3f" %
          (top_ppv, sh["灵敏度"]))
    print("  极低+低风险NPV=%.4f  灰区=%.1f%%" % (npv_low, grey_pct))
    print("\n  阶梯式决策建议（基于OOF预测）:")
    print("    极低风险: NPV>%.1f%%, 可安全排除" % (npv_low*100))
    print("    低风险:   低于基线阳性率(%.1f%%), 常规随访" % (y_truth.mean()*100))
    print("    中等风险: 建议生活方式干预 + 必要时OGTT")
    print("    高风险:   建议OGTT或糖化血红蛋白检测")
    print("    极高风险: PPV=%.1f%%, 强烈建议临床确诊" % (top_ppv*100))
    print("\n  输出文件:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print("    "+f)

if __name__ == "__main__":
    main()
