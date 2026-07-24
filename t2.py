# -*- coding: utf-8 -*-
"""t2.py — 问题2: GAM血糖预测模型（自动比较交互项）"""
import warnings, os, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy import stats; from functools import reduce
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

_CN_FP = None
for _fp in ["C:/Windows/Fonts/msyh.ttc","C:/Windows/Fonts/simhei.ttf"]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp); _CN_FP = fm.FontProperties(fname=_fp); _CN_FP.set_size(9); break
if not _CN_FP:
    for _f in fm.fontManager.ttflist:
        if any(k in _f.name for k in ["YaHei","SimHei","PingFang"]):
            _CN_FP = fm.FontProperties(family=_f.name); _CN_FP.set_size(9); break
plt.rcParams["font.family"] = "sans-serif"; plt.rcParams["axes.unicode_minus"] = False
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "问题2")
os.makedirs(OUTPUT_DIR, exist_ok=True)
from 数据预处理 import load_and_rename, impute_median, EN2CN, TARGET, CSV1
FINAL_VARS = ["年龄","TG","RBC","MCHC","HGB","ALT","性别"]
VARN = {v: EN2CN.get(v,v) for v in FINAL_VARS}

def load_data():
    df = load_and_rename(CSV1); df = impute_median(df)
    if TARGET in df.columns and df[TARGET].isnull().any():
        df[TARGET] = df[TARGET].fillna(df[TARGET].median())
    if "性别" in df.columns: df["性别"] = df["性别"].map({"男":1,"女":0}).fillna(0.5)
    print("  样本: %d, %s: mean=%.4f, std=%.4f" % (len(df), TARGET, df[TARGET].mean(), df[TARGET].std()))
    return df

def _ts(terms):
    if not terms: return None
    r = terms[0]
    for t in terms[1:]: r = r + t
    return r

def fit_gam(df, preds, k=10, interaction=False):
    from pygam import LinearGAM, s
    if interaction: from pygam import te
    X, y = df[preds].values, df[TARGET].values
    nv = len(preds)
    if interaction:
        terms = s(0) + s(1) + te(0,1) + _ts([s(i, n_splines=k) for i in range(2, nv)])
    else:
        terms = _ts([s(i, n_splines=k) for i in range(nv)])
    gam = LinearGAM(terms, fit_intercept=True)
    gam.fit(X, y)
    return gam, X, y, preds

def diagnostics(gam, X, y, preds, name="GAM"):
    yp = gam.predict(X); res = y - yp
    fig, axs = plt.subplots(2,2,figsize=(12,10))
    (osm, osr), (slope, intercept, r) = stats.probplot(res, dist="norm")
    axs[0,0].scatter(osm, osr, s=4, c="#3498DB", alpha=0.5, edgecolors="none")
    axs[0,0].plot(osm, slope * osm + intercept, color="#E74C3C", linewidth=1.5)
    axs[0,0].set_title("Q-Q图", fontproperties=_CN_FP)
    axs[0,1].scatter(yp, res, alpha=0.3, s=8); axs[0,1].axhline(0,c="r",ls="--")
    # x轴范围限定在1%~99%分位数，剔除极端离群值，聚焦主要数据
    x_lo, x_hi = np.percentile(yp, [1, 99])
    axs[0,1].set_xlim(x_lo, x_hi)
    axs[0,1].set_xlabel("拟合值", fontproperties=_CN_FP)
    axs[0,1].set_ylabel("残差", fontproperties=_CN_FP)
    axs[0,1].set_title("残差vs拟合值（已剔除两端1%极端值）", fontproperties=_CN_FP)
    axs[1,0].hist(res, bins=50, density=1, alpha=0.7, color="steelblue")
    rn = np.linspace(res.min(), res.max(), 100)
    axs[1,0].plot(rn, stats.norm.pdf(rn, res.mean(), res.std()), "r-")
    axs[1,0].set_title("残差分布", fontproperties=_CN_FP)
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(res, lags=30, ax=axs[1,1], zero=False)
    axs[1,1].set_title("残差自相关", fontproperties=_CN_FP)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,"问题2_残差诊断_"+name.replace(" ","_")+".png"), dpi=300); plt.close()
    sw = stats.shapiro(res[:5000])
    c, p = stats.pearsonr(np.abs(res), yp)
    het = p<0.05 and c>0.1 if not np.isnan(p) else False
    rmse = np.sqrt(mean_squared_error(y, yp)); r2 = r2_score(y, yp)
    n,pv = len(y), X.shape[1]; ar2 = 1-(1-r2)*(n-1)/(n-pv-1)
    print("  Shapiro p=%.4e 异方差r=%.4f RMSE=%.4f R²=%.6f adjR²=%.6f" % (sw[1], c, rmse, r2, ar2))
    return res, yp, het, {"RMSE":rmse,"R2":r2,"adj_R2":ar2}

def cv_gam(X, y, preds, cv=10, k=10):
    from pygam import LinearGAM, s
    kf = KFold(cv, shuffle=True, random_state=42); rmses, r2s = [], []
    for fold, (tr, te) in enumerate(kf.split(X), 1):
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        terms = _ts([s(i, n_splines=k) for i in range(len(preds))])
        gam = LinearGAM(terms, fit_intercept=True).fit(X_tr, y_tr)
        yp = gam.predict(X_te)
        rmses.append(np.sqrt(mean_squared_error(y_te, yp))); r2s.append(r2_score(y_te, yp))
        print("  %2d/%d: RMSE=%.4f R²=%.4f" % (fold, cv, rmses[-1], r2s[-1]))
    print("  >> CV-RMSE=%.4f±%.4f  CV-R²=%.4f±%.4f" % (np.mean(rmses),np.std(rmses),np.mean(r2s),np.std(r2s)))
    return {"cv_rmse":np.mean(rmses)}

def plot_partial(gam, X, y, preds, name="GAM"):
    nv = len(preds); nc = min(3, nv); nr = (nv+nc-1)//nc
    fig, axs = plt.subplots(nr, nc, figsize=(5*nc, 4*nr))
    axs = axs.flatten() if nr*nc>1 else [axs]
    # 交互模型有te()额外项，term索引需要偏移
    n_terms = len(gam.terms)
    is_interact = n_terms > nv
    for i in range(nv):
        ax = axs[i]; vn = VARN.get(preds[i], preds[i])
        # 计算正确的term索引
        term_i = i if not is_interact or i < 2 else i + 1
        try:
            XX = gam.generate_X_grid(term=term_i)
            pd, ci = gam.partial_dependence(term=term_i, X=XX, width=0.95)
            ax.plot(XX[:,i], pd, c="#E74C3C", lw=2)
            ax.fill_between(XX[:,i], ci[:,0], ci[:,1], alpha=0.2, color="#E74C3C")
        except: ax.text(.5,.5,"N/A",ha="center",va="center",transform=ax.transAxes)
        ax.axhline(0,c="gray",ls="--"); ax.set_xlabel(vn, fontproperties=_CN_FP)
        ax.set_ylabel("f(%s)"%vn, fontproperties=_CN_FP); ax.set_title(vn, fontproperties=_CN_FP, fontweight="bold")
    for j in range(nv, len(axs)): axs[j].axis("off")
    plt.suptitle("偏效应图 — %s"%name, fontproperties=_CN_FP, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,"问题2_偏效应图_"+name.replace(" ","_")+".png"), dpi=300); plt.close()
    print("  -> 偏效应图已保存")


def compare(g1, X, y, preds):
    y1 = g1.predict(X)
    bins = [0,5.6,6.1,7.0,20]; lbl = ["<5.6正常","5.6-6.1临界","6.1-7.0偏高",">7.0高值"]
    yb = pd.cut(y, bins=bins, labels=lbl)
    print("\n  %-18s %10s"%("血糖区间","RMSE")); print("  "+"-"*30)
    for l in lbl:
        m = yb==l
        if m.sum()<3: continue
        print("  %-18s %10.4f"%(l,np.sqrt(mean_squared_error(y[m],y1[m]))))
    print("  "+"="*30)
    def m(y,yp): r=np.sqrt(mean_squared_error(y,yp)); r2=r2_score(y,yp); return r,r2,1-(1-r2)*(len(y)-1)/(len(y)-X.shape[1]-1)
    print("  %-18s %10.4f"%("RMSE",m(y,y1)[0]))
    print("  %-18s %10.6f"%("R²",m(y,y1)[1]))
    print("  %-18s %10.6f"%("调整R²",m(y,y1)[2]))
    print("  %-18s %10.2f"%("AIC",g1.statistics_.get("AIC",0)))
    print("  "+"="*30)

def print_model_params(gam, preds, name="模型"):
    """输出完整的模型参数表。"""
    s = gam.statistics_
    print("\n  " + "=" * 50)
    print("  [%s] 完整参数表" % name)
    print("  " + "=" * 50)
    print("  拟合优度指标:")
    print("    AIC           = %.2f" % s.get("AIC", 0))
    print("    AICc          = %.2f" % s.get("AICc", 0))
    print("    GCV           = %.6f" % s.get("GCV", 0))
    print("    总有效自由度EDF = %.2f" % s.get("edof", 0))
    print("    尺度参数(scale)= %.6f" % s.get("scale", 0))
    pr2 = s.get("pseudo_r2", {})
    if isinstance(pr2, dict):
        for k, v in pr2.items():
            print("    伪R²(%s)     = %.6f" % (k, v))
    else:
        print("    伪R²          = %.6f" % pr2)
    print("    对数似然      = %.2f" % s.get("llf", 0))
    print("\n  各变量平滑项:")
    n_terms = len(gam.terms)
    # 打印每项的lambda和有效自由度
    for i in range(n_terms):
        lam = gam.lambda_[i] if hasattr(gam, 'lambda_') and i < len(gam.lambda_) else 0
        print("    项%d: lambda=%.6f" % (i, lam))
    print("  截距项: β₀ = %.6f" % gam.coef_[0])
    print("  " + "=" * 50)

def main():
    print("="*72+"\n  问题2: GAM血糖预测模型（自动比较交互项）\n"+"="*72)
    df = load_data(); X, y = df[FINAL_VARS].values, df[TARGET].values

    print("="*60+"\n  模型A: 加性GAM（无交互）\n"+"="*60)
    gam_a, X, y, _ = fit_gam(df, FINAL_VARS, k=10, interaction=False)
    aic_a = gam_a.statistics_.get("AIC", 0)
    diagnostics(gam_a, X, y, FINAL_VARS, "加性模型")
    cv_gam(X, y, FINAL_VARS, cv=10)

    print("="*60+"\n  模型B: 交互GAM（年龄×TG张量积）\n"+"="*60)
    gam_b, X, y, _ = fit_gam(df, FINAL_VARS, k=10, interaction=True)
    aic_b = gam_b.statistics_.get("AIC", 0)
    daic = aic_a - aic_b
    print("  AIC(加性)=%.2f  AIC(交互)=%.2f  ΔAIC=%.2f" % (aic_a, aic_b, daic))

    if daic > 2:
        print("  >> ΔAIC > 2: 交互项显著改善模型 → 采用模型B")
        final_model, final_name = gam_b, "交互模型"
        plot_interaction = True
    elif daic < -2:
        print("  >> ΔAIC < -2: 交互项使模型变差 → 采用模型A")
        final_model, final_name = gam_a, "加性模型"
        plot_interaction = False
    else:
        print("  >> |ΔAIC| ≤ 2: 两者无显著差异 → 选加性模型（简约原则）")
        final_model, final_name = gam_a, "加性模型"
        plot_interaction = False

    print("\n"+"="*60+"\n  最终模型输出\n"+"="*60)
    print("  最终选择: %s" % final_name)
    diagnostics(final_model, X, y, FINAL_VARS, final_name)
    print_model_params(final_model, FINAL_VARS, "最终交互GAM")
    print_model_params(gam_a, FINAL_VARS, "加性GAM对比")
    plot_partial(final_model, X, y, FINAL_VARS, final_name)

    if plot_interaction:
        print("="*60+"\n  交互效应等高线图\n"+"="*60)
        x1g = np.linspace(X[:,0].min(), X[:,0].max(), 50)
        x2g = np.linspace(X[:,1].min(), X[:,1].max(), 50)
        xx1, xx2 = np.meshgrid(x1g, x2g)
        Xg = np.zeros((2500,7))
        for i in range(7): Xg[:,i] = X[:,i].mean()
        Xg[:,0] = xx1.flatten(); Xg[:,1] = xx2.flatten()
        zg = gam_b.predict(Xg).reshape(50,50)
        fig, ax = plt.subplots(figsize=(8,6))
        ct = ax.contourf(xx1, xx2, zg, levels=20, cmap="RdYlBu_r")
        cbar = plt.colorbar(ct, ax=ax)
        cbar.set_label("预测血糖", fontproperties=_CN_FP)
        ax.scatter(X[:,0], X[:,1], c=y, cmap="RdYlBu_r", edgecolors="gray", s=8, alpha=0.3)
        ax.set_xlabel(VARN.get("年龄"), fontproperties=_CN_FP)
        ax.set_ylabel(VARN.get("TG"), fontproperties=_CN_FP)
        ax.set_title("年龄 × TG 交互效应热力图", fontproperties=_CN_FP, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR,"问题2_交互效应等高线.png"), dpi=300); plt.close()
        print("  -> 交互效应等高线图已保存")
    else:
        print("  交互项未入选，跳过等高线图")

    

    compare(final_model, X, y, FINAL_VARS)

    print("\n"+"="*72+"\n  完成!\n"+"="*72)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.startswith("问题2"): print("  "+f)

if __name__ == "__main__":
    main()
