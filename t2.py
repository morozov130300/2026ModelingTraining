# -*- coding: utf-8 -*-
"""
t2.py — 问题2：血糖值预测模型（广义可加模型 GAM）
7阶段完整建模：高斯GAM → 交互项 → 诊断 → CV → Gamma GAM → 模型对比
"""
import warnings, os, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from functools import reduce

# ── 字体 ──
_CN_FP = None
for _fp in ["C:/Windows/Fonts/msyh.ttc","C:/Windows/Fonts/simhei.ttf"]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp); _CN_FP = fm.FontProperties(fname=_fp); _CN_FP.set_size(9); break
if not _CN_FP:
    for _f in fm.fontManager.ttflist:
        if any(k in _f.name for k in ["YaHei","SimHei","PingFang"]):
            _CN_FP = fm.FontProperties(family=_f.name); _CN_FP.set_size(9); break
plt.rcParams["font.family"] = "sans-serif"; plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from 数据预处理 import load_and_rename, impute_median, EN2CN, TARGET, CSV1
FINAL_VARS = ["年龄", "TG", "RBC", "MCHC", "HGB", "ALT", "性别"]
VARN = {v: EN2CN.get(v,v) for v in FINAL_VARS}

def load_data():
    df = load_and_rename(CSV1); df = impute_median(df)
    if TARGET in df.columns and df[TARGET].isnull().any():
        df[TARGET] = df[TARGET].fillna(df[TARGET].median())
    if "性别" in df.columns: df["性别"] = df["性别"].map({"男":1,"女":0}).fillna(0.5)
    print("  样本: %d, %s: mean=%.4f, std=%.4f" % (len(df), TARGET, df[TARGET].mean(), df[TARGET].std()))
    return df

def _term_sum(terms):
    """安全拼接pyGAM Term列表"""
    if not terms: return None
    r = terms[0]
    for t in terms[1:]: r = r + t
    return r

# ════════════════════════════════════════════
#  第一阶段：高斯GAM
# ════════════════════════════════════════════
def fit_gaussian_gam(df, preds, k=10):
    print("="*60+"\n  第一阶段：高斯GAM\n"+"="*60)
    try:
        from pygam import LinearGAM, s
    except ImportError:
        print("  !! pip install pygam"); return None,None,None,None
    X, y = df[preds].values, df[TARGET].values
    terms = [s(i, n_splines=k) for i in range(len(preds))]
    gam = LinearGAM(_term_sum(terms), fit_intercept=True)
    print("  拟合 (REML/P-IRLS)...", end=" "); gam.fit(X, y); print("完成")
    print("  EDF=%.2f  AIC=%.2f  AICc=%.2f  GCV=%.6f" %
          (gam.statistics_.get("edof", gam.statistics_.get("edf", 0)),
           gam.statistics_.get("AIC", 0),
           gam.statistics_.get("AICc", 0), gam.statistics_.get("GCV", 0)))
    print("  各变量平滑项总EDF=%.2f" % gam.statistics_.get("edof", 0))
    return gam, X, y, preds

# ════════════════════════════════════════════
#  第四阶段：模型诊断
# ════════════════════════════════════════════
def model_diagnostics(gam, X, y, preds, name="Gaussian GAM"):
    print("="*60+"\n  第四阶段：模型诊断 — %s\n" % name+"="*60)
    yp = gam.predict(X); res = y - yp
    fig, axs = plt.subplots(2,2,figsize=(12,10))
    stats.probplot(res, dist="norm", plot=axs[0,0])
    axs[0,0].set_title("Q-Q图", fontproperties=_CN_FP)
    axs[0,1].scatter(yp, res, alpha=0.3, s=8); axs[0,1].axhline(0,c="r",ls="--")
    axs[0,1].set_title("残差vs拟合值", fontproperties=_CN_FP)
    axs[1,0].hist(res, bins=50, density=1, alpha=0.7, color="steelblue")
    rn = np.linspace(res.min(), res.max(), 100)
    axs[1,0].plot(rn, stats.norm.pdf(rn, res.mean(), res.std()), "r-")
    axs[1,0].set_title("残差分布", fontproperties=_CN_FP)
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(res, lags=30, ax=axs[1,1], zero=False)
    axs[1,1].set_title("残差自相关", fontproperties=_CN_FP)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题2_残差诊断_"+name.replace(" ","_")+".png"), dpi=300); plt.close()
    # 正态性
    sw = stats.shapiro(res[:5000])
    print("  Shapiro-Wilk: W=%.4f, p=%.4e %s" % (sw[0], sw[1], "非正态" if sw[1]<0.05 else "正态"))
    # 异方差
    c, p = stats.pearsonr(np.abs(res), yp)
    het = p<0.05 and c>0.1
    print("  异方差: |e|vsŷ r=%.4f p=%.4e %s" % (c, p, "检测到异方差!" if het else "无异方差"))
    # 精度
    rmse = np.sqrt(mean_squared_error(y, yp)); r2 = r2_score(y, yp)
    n,p_var = len(y), X.shape[1]; ar2 = 1-(1-r2)*(n-1)/(n-p_var-1)
    print("  RMSE=%.4f  R²=%.6f  adjR²=%.6f" % (rmse, r2, ar2))
    return res, yp, het, {"RMSE":rmse,"R2":r2,"adj_R2":ar2}

# ════════════════════════════════════════════
#  10折交叉验证
# ════════════════════════════════════════════
def cv_gam(X, y, preds, cv=10, k=10):
    print("="*60+"\n  10折交叉验证\n"+"="*60)
    try:
        from pygam import LinearGAM, s
    except: return None
    kf = KFold(cv, shuffle=True, random_state=42); rmses, r2s = [], []
    for fold, (tr, te) in enumerate(kf.split(X), 1):
        X_tr, X_te = X[tr], X[te]; y_tr, y_te = y[tr], y[te]
        terms = [s(i, n_splines=k) for i in range(len(preds))]
        gam = LinearGAM(_term_sum(terms), fit_intercept=True).fit(X_tr, y_tr)
        yp = gam.predict(X_te)
        rmses.append(np.sqrt(mean_squared_error(y_te, yp))); r2s.append(r2_score(y_te, yp))
        print("  %2d/%d: RMSE=%.4f R²=%.4f" % (fold, cv, rmses[-1], r2s[-1]))
    print("  >> CV-RMSE=%.4f±%.4f  CV-R²=%.4f±%.4f" % (np.mean(rmses),np.std(rmses),np.mean(r2s),np.std(r2s)))
    return {"cv_rmse":np.mean(rmses),"cv_rmse_std":np.std(rmses),"cv_r2":np.mean(r2s)}

# ════════════════════════════════════════════
#  第五阶段：偏效应图
# ════════════════════════════════════════════
def plot_partial(gam, X, y, preds, name="Gaussian GAM"):
    print("="*60+"\n  第五阶段：偏效应图\n"+"="*60)
    nv = len(preds); nc = min(3, nv); nr = (nv+nc-1)//nc
    fig, axs = plt.subplots(nr, nc, figsize=(5*nc, 4*nr))
    axs = axs.flatten() if nr*nc>1 else [axs]
    for i in range(nv):
        ax = axs[i]; vn = VARN.get(preds[i], preds[i])
        try:
            XX = gam.generate_X_grid(term=i)
            pd, ci = gam.partial_dependence(term=i, X=XX, width=0.95)
            ax.plot(XX[:,i], pd, c="#E74C3C", lw=2)
            ax.fill_between(XX[:,i], ci[:,0], ci[:,1], alpha=0.2, color="#E74C3C")
        except: ax.text(.5,.5,"N/A", ha="center", va="center", transform=ax.transAxes)
        ax.axhline(0,c="gray",ls="--"); ax.set_xlabel(vn, fontproperties=_CN_FP)
        ax.set_ylabel("f(%s)"%vn, fontproperties=_CN_FP); ax.set_title(vn, fontproperties=_CN_FP, fontweight="bold")
    for j in range(nv, len(axs)): axs[j].axis("off")
    plt.suptitle("偏效应图 — %s"%name, fontproperties=_CN_FP, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "问题2_偏效应图_"+name.replace(" ","_")+".png"), dpi=300); plt.close()
    print("  -> 偏效应图已保存")

# ════════════════════════════════════════════
#  第六阶段：Gamma GAM
# ════════════════════════════════════════════
def fit_gamma_gam(df, preds, k=10):
    print("="*60+"\n  第六阶段：Gamma GAM（对数链接）\n"+"="*60)
    try:
        from pygam import GammaGAM, s
    except: return None,None,None
    X, y = df[preds].values, df[TARGET].values
    if (y<=0).any(): y = y + 0.001
    terms = [s(i, n_splines=k) for i in range(len(preds))]
    gam = GammaGAM(_term_sum(terms), fit_intercept=True)
    print("  拟合...", end=" "); gam.fit(X, y); print("完成")
    yp = gam.predict(X); rmse = np.sqrt(mean_squared_error(y, yp))
    r2 = r2_score(y, yp); n,p = len(y), X.shape[1]; ar2 = 1-(1-r2)*(n-1)/(n-p-1)
    print("  EDF=%.2f AIC=%.2f RMSE=%.4f R²=%.6f adjR²=%.6f" %
          (gam.statistics_.get("edof", gam.statistics_.get("edf", 0)),
           gam.statistics_.get("AIC", 0), rmse, r2, ar2))
    return gam, X, yp

# ════════════════════════════════════════════
#  第七阶段：模型对比
# ════════════════════════════════════════════
def compare_models(g1, g2, X, y, preds):
    print("="*60+"\n  第七阶段：模型对比\n"+"="*60)
    y1 = g1.predict(X)
    y2 = g2.predict(X) if g2 else y1
    bins = [0,5.6,6.1,7.0,20]; lbl = ["<5.6正常","5.6-6.1临界","6.1-7.0偏高",">7.0高值"]
    yb = pd.cut(y, bins=bins, labels=lbl)
    print("\n  %-18s %10s %10s"%("血糖区间","高斯RMSE","GammaRMSE"))
    print("  "+"-"*42)
    for l in lbl:
        m = yb==l
        if m.sum()<3: continue
        r1 = np.sqrt(mean_squared_error(y[m], y1[m]))
        r2 = np.sqrt(mean_squared_error(y[m], y2[m]))
        print("  %-18s %10.4f %10.4f"%(l,r1,r2))
    print("  "+"="*42)
    print("  %-18s %10s %10s"%("指标","高斯GAM","Gamma GAM"))
    print("  "+"-"*42)
    def metrics(y, yp):
        rmse=np.sqrt(mean_squared_error(y,yp)); r2=r2_score(y,yp)
        n,p=len(y),X.shape[1]; return rmse,r2,1-(1-r2)*(n-1)/(n-p-1)
    m1 = metrics(y,y1); m2 = metrics(y,y2)
    print("  %-18s %10.4f %10.4f"%("RMSE",m1[0],m2[0]))
    print("  %-18s %10.6f %10.6f"%("R²",m1[1],m2[1]))
    print("  %-18s %10.6f %10.6f"%("调整R²",m1[2],m2[2]))
    print("  %-18s %10.2f %10.2f"%("AIC",g1.statistics_.get("AIC",0),
          g2.statistics_.get("AIC",0) if g2 else 0))
    print("  "+"="*42)

# ════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════
def main():
    print("="*72+"\n  问题2: 血糖值预测模型 — GAM\n"+"="*72)
    df = load_data()
    X, y = df[FINAL_VARS].values, df[TARGET].values
    gam, X, y, preds = fit_gaussian_gam(df, FINAL_VARS, k=10)
    if gam is None: return
    res, yp, het, met = model_diagnostics(gam, X, y, FINAL_VARS)
    cv_results = cv_gam(X, y, FINAL_VARS, cv=10, k=10)
    plot_partial(gam, X, y, FINAL_VARS)
    r = input("\n  加入年龄×TG交互项? (y/n): ").strip().lower()
    if r == 'y':
        print("="*60+"\n  第二阶段：张量积交互项\n"+"="*60)
        try:
            from pygam import LinearGAM, s, te
            # 交互模型: s(年龄) + s(TG) + te(年龄,TG) + 其他主效应
            gam_int = LinearGAM(s(0) + s(1) + te(0, 1) +
                                _term_sum([s(i, n_splines=10) for i in range(2, 7)]),
                                fit_intercept=True)
            gam_int.fit(X, y)
            print("  交互模型 AIC=%.2f (ΔAIC=%.2f)" % (gam_int.statistics_.get("AIC", 0),
                  gam.statistics_.get("AIC", 0)-gam_int.statistics_.get("AIC", 0)))
            # 等高线图
            x1g = np.linspace(X[:,0].min(), X[:,0].max(), 50)
            x2g = np.linspace(X[:,1].min(), X[:,1].max(), 50)
            xx1, xx2 = np.meshgrid(x1g, x2g)
            Xg = np.zeros((2500,7))
            for i in range(7):
                Xg[:,i] = X[:,i].mean()
            Xg[:,0] = xx1.flatten(); Xg[:,1] = xx2.flatten()
            zg = gam_int.predict(Xg).reshape(50,50)
            fig, ax = plt.subplots(figsize=(8,6))
            ct = ax.contourf(xx1, xx2, zg, levels=20, cmap="RdYlBu_r")
            plt.colorbar(ct, ax=ax, label="预测血糖")
            ax.scatter(X[:,0], X[:,1], c=y, cmap="RdYlBu_r", edgecolors="gray", s=8, alpha=0.3)
            ax.set_xlabel(VARN.get(FINAL_VARS[0]), fontproperties=_CN_FP)
            ax.set_ylabel(VARN.get(FINAL_VARS[1]), fontproperties=_CN_FP)
            ax.set_title("%s × %s 交互效应" % (VARN.get(FINAL_VARS[0]), VARN.get(FINAL_VARS[1])),
                        fontproperties=_CN_FP, fontweight="bold")
            plt.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, "问题2_交互效应等高线.png"), dpi=300); plt.close()
            print("  -> 交互等高线图已保存")
        except Exception as e:
            print("  !! 交互项失败:", e)
    g2 = None
    if het:
        print("\n  >> 异方差显著 → 升级Gamma GAM")
        g2, _, _ = fit_gamma_gam(df, FINAL_VARS)
        if g2: model_diagnostics(g2, X, y, FINAL_VARS, "Gamma_GAM")
    compare_models(gam, g2, X, y, FINAL_VARS)
    print("\n"+"="*72+"\n  完成!\n"+"="*72)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.startswith("问题2"): print("  "+f)

if __name__ == "__main__":
    main()
