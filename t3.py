# -*- coding: utf-8 -*-
"""
t3.py — 问题3：糖尿病风险评估（Logistic GAM）
===============================================
构建Logistic广义可加模型，将血糖值转化为二分类风险预测。
标准：血糖 > 6.1 mmol/L 定义为高风险（空腹血糖受损/糖尿病前期）
      血糖 > 7.0 mmol/L 定义为糖尿病
"""

import warnings, os, numpy as np, pandas as pd, concurrent.futures, multiprocessing
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy import stats
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score,
                             brier_score_loss, mean_squared_error, r2_score, mean_absolute_error)
from sklearn.model_selection import KFold
from functools import reduce

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

from 数据预处理 import load_and_rename, impute_median, EN2CN, TARGET, CSV1

FINAL_VARS = ["年龄","TG","RBC","MCHC","HGB","ALT","性别"]
VARN = {v: EN2CN.get(v,v) for v in FINAL_VARS}
SEP = "=" * 70

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
def fit_logistic_gam(df, preds, k=6):
    """Logistic GAM (Logit链接, 二项分布)"""
    from pygam import LogisticGAM, s, f
    X, y = df[preds].values, df["DM"].values
    nv = len(preds)
    # 前6个连续变量用s()，最后1个(性别)用f()因子项
    terms = _ts([s(i, n_splines=k) for i in range(nv - 1)]) + f(nv - 1)
    gam = LogisticGAM(terms, fit_intercept=True, max_iter=1000, lam=0.6)
    print("  拟合Logistic GAM...", end=" ")
    import warnings as _w; _w.filterwarnings("ignore")
    gam.fit(X, y); _w.filterwarnings("default")
    print("完成")
    y_prob = gam.predict_proba(X)
    # 用Youden J指数找最优阈值
    fpr, tpr, thr = roc_curve(y, y_prob)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_thr = thr[best_idx]
    y_pred_opt = (y_prob > best_thr).astype(int)
    print("  UBRE=%.4f  EDF=%.2f" % (gam.statistics_.get("UBRE",0), gam.statistics_.get("edof",0)))
    print("  最优阈值(Youden)=%.4f (敏感度+特异度=%.4f)" % (best_thr, youden[best_idx]))
    return gam, X, y, y_prob, y_pred_opt, best_thr

# ═══════════════════════════════════════════
#  Bootstrap（多进程并行）
# ═══════════════════════════════════════════
def _bootstrap_worker(seed, X_data, y_data, x_mean_data, x_grids_data,
                      preds_list, k, lam):
    """单次Bootstrap迭代（模块级函数，用于ProcessPoolExecutor并行）"""
    import numpy as np
    from pygam import LogisticGAM, s, f
    import warnings as _w
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(seed)
    n = len(X_data)
    idx = rng.choice(n, n, replace=True)  # 完整有放回抽样
    Xb, yb = X_data[idx], y_data[idx]
    nv = X_data.shape[1]

    def _ts(L):
        if not L: return None
        r = L[0]
        for t in L[1:]: r = r + t
        return r

    terms = _ts([s(i, n_splines=k) for i in range(nv-1)]) + f(nv-1)
    try:
        gam = LogisticGAM(terms, fit_intercept=True, max_iter=1000, lam=lam)
        with _w.catch_warnings(): _w.simplefilter("ignore")
        gam.fit(Xb, yb)
        auc = float(roc_auc_score(yb, gam.predict_proba(Xb)))

        curves = {}
        for i, p in enumerate(preds_list):
            xg = x_grids_data[p]; ng = len(xg)
            X_grid = np.tile(x_mean_data, (ng, 1)); X_grid[:, i] = xg
            prob = np.clip(gam.predict_proba(X_grid), 1e-10, 1-1e-10)
            eta = np.log(prob/(1-prob)); eta -= eta.mean()
            curves[p] = eta
        return auc, curves, True
    except Exception:
        return 0.0, {}, False


def bootstrap_logistic_gam(df, preds, n_boot=200, k=6, lam=0.6):
    """Bootstrap估计参数稳定性 + 偏效应置信区间（多进程并行）"""
    n_cpu = multiprocessing.cpu_count()
    print("  Bootstrap (%d次, %d核并行)..." % (n_boot, n_cpu), end=" ", flush=True)

    X_full = df[preds].values
    y_full = df["DM"].values
    x_mean = X_full.mean(axis=0)

    # 为每个变量准备网格
    x_grids = {}
    for i, p in enumerate(preds):
        if p == "性别":
            x_grids[p] = np.array([0, 1])
        else:
            xmin, xmax = X_full[:, i].min(), X_full[:, i].max()
            x_grids[p] = np.linspace(xmin, xmax, 100)

    # 并行提交
    seeds = list(range(42, 42 + n_boot))
    args_list = [(s, X_full, y_full, x_mean, x_grids, preds, k, lam) for s in seeds]

    aucs = []
    partial_curves = {p: [] for p in preds}
    valid_boots = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpu) as ex:
        futures = [ex.submit(_bootstrap_worker, *a) for a in args_list]
        for f in concurrent.futures.as_completed(futures):
            try:
                auc, curves, ok = f.result()
                if ok:
                    aucs.append(auc)
                    for p in preds:
                        partial_curves[p].append(curves[p])
                    valid_boots += 1
            except Exception:
                pass

    print("完成 (%d/%d收敛)" % (valid_boots, n_boot))
    if aucs:
        print("  Bootstrap AUC: mean=%.4f, 95%%CI=[%.4f, %.4f]" %
              (np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)))
    return aucs, partial_curves, x_grids

# ═══════════════════════════════════════════
#  评估指标
# ═══════════════════════════════════════════
def evaluate(y_true, y_prob, best_thr):
    """综合评估指标（使用Youden最优阈值）"""
    auc = roc_auc_score(y_true, y_prob)
    y_pred = (y_prob > best_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp+fn)>0 else 0
    spec = tn / (tn + fp) if (tn+fp)>0 else 0
    ppv = tp / (tp + fp) if (tp+fp)>0 else 0
    npv = tn / (tn + fn) if (tn+fn)>0 else 0
    metrics = {
        "AUC": auc, "阈值": best_thr, "准确率": accuracy_score(y_true, y_pred),
        "敏感度": sens, "特异度": spec, "阳性预测值": ppv, "阴性预测值": npv,
        "F1": f1_score(y_true, y_pred), "Brier": brier_score_loss(y_true, y_prob)
    }
    # 同时显示默认阈值0.5的结果作对比
    y_pred50 = (y_prob > 0.5).astype(int)
    cm50 = confusion_matrix(y_true, y_pred50)
    tn5, fp5, fn5, tp5 = cm50.ravel()
    print("\n  评价指标 (Youden阈值=%.4f):" % best_thr)
    for k, v in metrics.items():
        print("    %-10s = %.4f" % (k, v))
    print("  混淆矩阵: TP=%d  FN=%d  FP=%d  TN=%d" % (tp, fn, fp, tn))
    print("\n  [对比] 默认阈值0.5: 敏感度=%.4f, 特异度=%.4f, TP=%d" %
          (tp5/(tp5+fn5) if (tp5+fn5)>0 else 0, tn5/(tn5+fp5) if (tn5+fp5)>0 else 0, tp5))
    return metrics, cm

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
            x_vals = np.array([0, 1])
            X_grid = np.tile(x_mean, (2, 1)); X_grid[:,i] = x_vals
            with np.errstate(divide="ignore"):
                p = np.clip(gam.predict_proba(X_grid), 1e-10, 1-1e-10)
                eta = np.log(p/(1-p)); eta -= eta.mean()
            ax.plot(x_vals, eta, "o-", c="#E74C3C", lw=2, ms=8, zorder=5)
            # 分类变量CI: 以点估计为中心的垂直误差棒
            if has_ci and pname in partial_curves and len(partial_curves[pname]) > 1:
                for j in range(len(x_vals)):
                    yerr = np.array([[eta[j]-ci_low[j]], [ci_high[j]-eta[j]]])
                    ax.errorbar(x_vals[j], eta[j], yerr=yerr, fmt='none',
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
#  主流程
# ═══════════════════════════════════════════
def main():
    print(SEP+"\n  问题3: 糖尿病风险评估 — Logistic GAM\n"+SEP)
    
    # 加载（阈值7.0=糖尿病诊断标准，也可尝试6.1=空腹血糖受损）
    df = load_data(dm_threshold=7.0)
    print("\n  糖尿病诊断标准: 血糖 > 7.0 mmol/L")
    
    # Logistic GAM
    gam, X, y, y_prob, y_pred_opt, best_thr = fit_logistic_gam(df, FINAL_VARS, k=6)
    
    # Bootstrap
    aucs, partial_curves, x_grids = bootstrap_logistic_gam(df, FINAL_VARS, n_boot=200)
    
    # 评估
    metrics, cm = evaluate(y, y_prob, best_thr)
    
    # 可视化
    plot_roc(y, y_prob)
    plot_calibration(y, y_prob, n_bins=10)
    plot_prob_distribution(y, y_prob)
    plot_partial_logistic(gam, X, FINAL_VARS, partial_curves, x_grids)
    
    # 按血糖分层展示
    print("\n"+"="*50)
    print("  按血糖水平分层评估")
    print("="*50)
    for thr in [5.6, 6.1, 7.0, 8.0]:
        y_bin = (df[TARGET].values > thr).astype(int)
        if y_bin.sum() < 10: continue
        yp = gam.predict_proba(X)
        auc = roc_auc_score(y_bin, yp)
        print("  血糖>%.1f: AUC=%.4f (阳性n=%d)" % (thr, auc, y_bin.sum()))
    
    print("\n"+SEP+"\n  完成!\n"+SEP)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print("  "+f)

if __name__ == "__main__":
    main()
