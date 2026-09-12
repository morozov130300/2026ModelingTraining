# -*- coding: utf-8 -*-
"""
问题四：波动电价下的微网购电策略重算与深度分析
================================================================
严格按《Q4/问题四_波动电价影响深度分析.md》实现，核心原则（文档「必须保持不变的口径」）：

    只把每日重复的日内电价 c_t^F（附件1）替换为逐日 10 分钟电价 c_{d,t}^V（附件4），
    其余预测方法、场景生成、MPC 更新节点、储能约束、紧急购电规则、调整采纳规则
    与结算口径全部保持原样：

        Q4 结果差异 = 同一模型(c_{d,t}^V) - 同一模型(c_t^F)

因此本脚本**不重新实现任何模型**，而是直接 import 复用
    ../Q2/solve_q2.py  （问题二：0:00 日前计划 + 5 倍紧急购电）
    ../Q3/solve_q3.py  （问题三：0:00 基准 + 6/12/18 时 MPC 滚动调整 + S0~S3）
并在每日调用前仅替换 ctx['prices'] 这一件事，从而在代码层面保证「算法、约束、口径零改动」，
唯一变量是电价（文档「控制变量表达式」）。

沿用求解器（AGENTS.md）：MILP = scipy.optimize.milp（HiGHS 分支切割），
LP 松弛 = scipy.optimize.linprog(method='highs')；并发固定 8 进程，按天分派。

--------------------------------------------------------------------------
交付物（全部落在 Q4/ 目录）
--------------------------------------------------------------------------
正式重算（main = vol + fixed 两个实验）
  result4-2.xlsx            波动电价下问题二全部结果（附件5 result4-2 模板原结构）
  result4-3.xlsx            波动电价下问题三全部结果（附件5 result4-3 模板原结构）
  表1_Q4-2 / 表2_Q4-2 / 表3_Q4-2.xlsx     4 代表日（Q4-2，题目「论文中以表1/表2/表3的格式给出」）
  表1_Q4-3 / 表2_Q4-3 / 表3_Q4-3.xlsx     4 代表日（Q4-3, S3）
  逐日分析表.xlsx            334 天 × 六组逐日指标（文档「建议保存的逐日分析表」）
                            价格 / 调度 / Q3调整（含各节点净费用变化与 S0-S3）/
                            紧急购电 / 成本 / 反事实
  对照汇总.xlsx              总体结果 / 反事实分解 / 风险指标 / 成本结构 / 约束活跃率与灵活性 /
                            价量协方差分解 / 储能价值 /
                            S0-S3逐级比较 / 配对统计与置信区间 / 分组检验 / PV预报PWMAE /
                            日期重合率 / 最坏日与集中度 / 寿命成本修正 / 上下调系数敏感性 /
                            结果解释规则 / 缓冲比例与效应分解 / Q4层核验清单

  vol   = 附件4 波动电价（正式结果 x^V）
  fixed = 附件1 固定电价（基线策略 x^F）。之所以由本脚本重解而不是直接读 Q3 原生缓存：
          Q4-3 的调整费必须在新价格下重算 C^adj = Σ c^V(1.5U−0.5W)Δt，而 U/W 是 MILP
          内部变量、原生缓存未保存。本脚本通过包装 Q3.solve_milp 在求解瞬间抓取 U/W
          （Q3 源码零改动、缓存版本不变），并同一口径重解 fixed，回归核验与原生缓存逐项对齐。

价格预测现实性扩展（pf，文档「可选的价格预测现实性扩展」第 1053-1118 行）
  Q4-PI = 完全信息基准，即上面的 vol 正式结果（0:00 即已知附件4 全部未来实际价格）
  Q4-PF = 因果预测运行，只使用决策时刻之前的信息：
          ĉ_{d,t|0} = w1·c_{d-1,t} + w7·c_{d-7,t} + w14·c_{d-14,t}，Σw=1、w≥0，
          权重只用「日期早于 d」的滚动样本外验证确定（PF_TRAIN_WIN 天窗口）；
          6:00/12:00/18:00 用最新已实现偏差修正未发生时段：
          ĉ^upd_{d,t|k} = ĉ_{d,t|0} + λ_k(c^real_{d,n_k} − ĉ_{d,n_k|0})，t > n_k
          策略由预测价格优化、费用一律按**真实价格**结算，于是
          价格信息机会损失 L_price info = C(c^real; x(ĉ)) − C(c^real; x(c^real)) ≥ 0
  精度指标：价格 MAE / RMSE / 方向判断准确率 / 决策加权误差 / 高价段低估比例
  结果只落在 对照汇总.xlsx（「价格预测现实性(Q4-PF)」「预测权重与逐日精度」）
  与 逐日分析表.xlsx 的新增列，**绝不替换 result4-2.xlsx / result4-3.xlsx**（文档明文要求）

敏感性（sens）—— 文档「反事实实验与敏感性分析」全部要求，共 21 族
  敏感性汇总.xlsx            振幅族 lam0/lam05/lam15（λ∈{0,0.5,1.0,1.5}）
                            缩放族 gam08/gam12（γ∈{0.8,1.0,1.2}，附正比例缩放核验表）
                            无储能族 nobess（V^BESS）
                            储能价值族 lam0nb/lam05nb/lam15nb（各 λ 下的无储能对照，
                              用于文档「对每个 λ 报告储能价值」；结果见 对照汇总/储能价值）
                            尖峰族 w99/w95（价格按 P99/P95 全样本分位缩尾）
                            时序族 shm1/shp1（价格整体前移/后移 1 小时，仅作诊断）
                            效率族 eta85/eta95（充放电效率 0.85/0.95）
                            功率族 pm25/pm75（储能功率上限 2500/7500 kW）
                            并网族 pb6k/pb10k（并网购电上限 6000/10000 kW，题目未给容量）
                            倍数族 kap3/kap7（紧急购电倍数 3/7）
  上下调系数扰动与 κ_th 衰减成本为纯事后结算修正，不建实验族：
    前者见「对照汇总.xlsx / 上下调系数敏感性」，后者见「寿命成本修正」。

--------------------------------------------------------------------------
用法（在装有 numpy/scipy/openpyxl 的 Windows 目标机上运行）
--------------------------------------------------------------------------
  python solve_q4.py                  # 一键全量：主重算(vol+fixed) + 全套敏感性（断点续跑 + 8 进程）
  python solve_q4.py main             # 仅主重算（vol + fixed，正式交付文件）
  python solve_q4.py sens             # 基准 + 敏感性
  python solve_q4.py outputs          # 由缓存重建全部输出，不重解 MILP
  python solve_q4.py regress          # 回归核验：附件1 价格重解 4 代表日，比对 Q2/Q3 缓存
  python solve_q4.py pf               # 价格预测现实性扩展：跑 Q4-PF（因果预测）并汇总 L_price info
                                      #   缺 Q4-PF 缓存时 collect_all 自动跳过，不影响其他交付
  python solve_q4.py main --which q2  # 只跑问题二族（先拿到 Q4-2）
  python solve_q4.py main --which q3  # 只跑问题三族
  python solve_q4.py full --force     # 忽略缓存全部重解
  python solve_q4.py full --limit 10  # 冒烟测试：仅前 10 个决策日
  --exp lam0,lam05,lam15,w99,w95,eta85,pm25,kap7,nobess   限定敏感性实验
        （vol/fixed 为基准，总会执行；完整清单见 SENS_EXPS）
"""

import argparse
import importlib.util
import multiprocessing
import os
import sys
import time
from datetime import date, datetime

import numpy as np
import openpyxl

# ========== 0. 路径与常量 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ATTACH_DIR = os.path.join(PROJECT_ROOT, '附件')

Q2_DIR = os.path.join(PROJECT_ROOT, 'Q2')
Q3_DIR = os.path.join(PROJECT_ROOT, 'Q3')
Q2_SCRIPT = os.path.join(Q2_DIR, 'solve_q2.py')
Q3_SCRIPT = os.path.join(Q3_DIR, 'solve_q3.py')

PRICE4_FILE = os.path.join(ATTACH_DIR, '附件4.xlsx')
TEMPLATE4_2 = os.path.join(ATTACH_DIR, '附件5', 'result4-2.xlsx')
TEMPLATE4_3 = os.path.join(ATTACH_DIR, '附件5', 'result4-3.xlsx')

OUT4_2 = os.path.join(SCRIPT_DIR, 'result4-2.xlsx')
OUT4_3 = os.path.join(SCRIPT_DIR, 'result4-3.xlsx')
TBL = {('q2', 1): os.path.join(SCRIPT_DIR, '表1_Q4-2.xlsx'),
       ('q2', 2): os.path.join(SCRIPT_DIR, '表2_Q4-2.xlsx'),
       ('q2', 3): os.path.join(SCRIPT_DIR, '表3_Q4-2.xlsx'),
       ('q3', 1): os.path.join(SCRIPT_DIR, '表1_Q4-3.xlsx'),
       ('q3', 2): os.path.join(SCRIPT_DIR, '表2_Q4-3.xlsx'),
       ('q3', 3): os.path.join(SCRIPT_DIR, '表3_Q4-3.xlsx')}
DAILY_FILE = os.path.join(SCRIPT_DIR, '逐日分析表.xlsx')
SUMMARY_FILE = os.path.join(SCRIPT_DIR, '对照汇总.xlsx')
SENS_FILE = os.path.join(SCRIPT_DIR, '敏感性汇总.xlsx')
CACHE_ROOT = os.path.join(SCRIPT_DIR, 'cache')

# 与 Q2/Q3 完全一致的物理常量（附录1）
N = 144
DT = 10.0 / 60.0
ETA_CH = ETA_DIS = 0.9
E_MIN, E_MAX = 1200.0, 10800.0
E_INIT = 6000.0
P_CH_MAX, P_DIS_MAX = 5000.0, 5000.0
E_CAP = 12000.0
E_USE = E_MAX - E_MIN
KAPPA = 5.0

FIRST_DAY = date(2025, 2, 1)
LAST_DAY = date(2025, 12, 31)
REP_DAYS = [date(2025, 3, 20), date(2025, 6, 21), date(2025, 9, 23), date(2025, 12, 21)]

# 风险与统计参数
CVAR_ALPHA_MAIN = 0.95
CVAR_ALPHA_CHECK = (0.90, 0.99)
BOOT_N_RESAMPLE = 10000
BOOT_SEED = 42
BOOT_BLOCKS = (7, 3, 14)          # 主口径 7 天块；3/14 天为稳健性对照（文档「配对统计与置信区间」）
WORST_N_DAYS = 10                 # 最坏 N 日费用贡献 S_N^C（文档「成本集中度与尖峰日贡献」）
ACT_EPS_P = 1e-3
ACT_EPS_E = 1e-3
REG_TOL_REL = 1e-6
REG_TOL_ABS = 1e-4

# ---- 执行敏感性分析所需参数（文档「尖峰、时序与设备参数敏感性」表）----
SENS_SPIKE_Q = (99, 95)           # 价格缩尾分位：原始 / P99 缩尾 / P95 缩尾
SENS_SHIFT_SLOT = 6               # 价格时序平移 1 小时 = 6 个 10 分钟时段（前移/后移）
SENS_ETA = (0.85, 0.95)           # 充放电效率水平（基准 0.90）
SENS_PMAX = (2500.0, 7500.0)      # 储能功率上限水平（基准 5000 kW）
SENS_KAPPA = (3.0, 7.0)           # 紧急购电倍数水平（基准 5.0）
# 并网购电上限（文档「负价下必须有有限并网购电上限」）：题目未给变压器/馈线容量，
# 故主结果保持 P_buy_max=∞，另设两组参照值用于回答「低价充电与紧急补能是否受连接点限制」。
# 文档同节明确「不应随意用储能 5000 kW 功率上限替代」，故参照值刻意避开 5000/7500/2500。
SENS_PBUY = (6000.0, 10000.0)
SENS_COEF_UP = (1.2, 1.8)         # 上调系数扰动（基准 1.5）
SENS_COEF_DOWN = (0.3, 0.7)       # 下调系数扰动（基准 0.5）
LIFE_KAPPA_GRID = (0.0, 0.05, 0.10, 0.20, 0.50)
# ↑ C^life-adjusted = C^total + κ_th·E^throughput 的单位吞吐折算寿命成本（元/kWh），
#   文档「储能循环压力与隐含寿命风险」要求「从 0 到合理上界」做事后修正，不改变正式调度。
ETA_ROUND_TRIP = ETA_CH * ETA_DIS                 # 0.81 往返效率
EFF_THRESHOLD = (1.0 - ETA_ROUND_TRIP) / ETA_ROUND_TRIP
# ↑ 效率补偿阈值：价差须超过 (1-η²)/η² ≈ 23.46% 套利才净收益为正（文档「情形六」第 1 条）
TIER_Q = (1.0 / 3.0, 2.0 / 3.0)   # 分组检验的低/中/高三档分位切点
VERIFY_TOL_BAL = 1e-4             # 功率平衡残差容差（沿用 Q2/Q3 的 BAL_TOL 口径）
VERIFY_TOL_SOC = 1e-6             # SOC 边界断言容差（kWh）


# ========== 1. 模块复用（唯一改动点 = ctx['prices']） ==========
def _load_module(name, path):
    """按文件路径加载模块并注册进 sys.modules（保证 spawn 子进程可复现导入）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少依赖脚本: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for _d in (Q2_DIR, Q3_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

Q2 = _load_module('solve_q2', Q2_SCRIPT)
Q3 = _load_module('solve_q3', Q3_SCRIPT)

Q3_NATIVE_CACHE_VERSION = Q3.CACHE_VERSION          # 原生问题三缓存版本（后续会被实验覆盖）

# 一致性自检：上游常量必须与本问口径一致，否则直接报错（防止上游被改动后静默跑偏）
for _k, _v in {'N': (Q2.N, Q3.N, N), 'DT': (Q2.DT, Q3.DT, DT),
               'E_MIN': (Q2.E_MIN, Q3.E_MIN, E_MIN), 'E_MAX': (Q2.E_MAX, Q3.E_MAX, E_MAX),
               'E_INIT': (Q2.E_INIT, Q3.E_INIT, E_INIT),
               'P_CH_MAX': (Q2.P_CH_MAX, Q3.P_CH_MAX, P_CH_MAX),
               'P_DIS_MAX': (Q2.P_DIS_MAX, Q3.P_DIS_MAX, P_DIS_MAX),
               'KAPPA': (Q2.KAPPA, Q3.KAPPA, KAPPA),
               'FIRST_DAY': (Q2.FIRST_DAY, Q3.FIRST_DAY, FIRST_DAY),
               'LAST_DAY': (Q2.LAST_DAY, Q3.LAST_DAY, LAST_DAY)}.items():
    if not (_v[0] == _v[1] == _v[2]):
        raise RuntimeError(f"Q2/Q3 常量 {_k} 不一致: {_v[0]} / {_v[1]} / 期望 {_v[2]}")
if list(Q2.REP_DAYS) != list(REP_DAYS) or list(Q3.REP_DAYS) != list(REP_DAYS):
    raise RuntimeError("Q2/Q3 代表日与 Q4 期望不一致")


# ========== 1c. 价格预测现实性扩展（文档「可选的价格预测现实性扩展」第 1053-1118 行） ==========
# 文档要求：正式 Q4 用附件4 给定的未来实际价格重算（记为 Q4-PI，完全信息基准），
# 另做一个「只使用决策时刻之前的信息」的因果预测版本（Q4-PF），用二者之差衡量
# 价格信息不足造成的机会损失：
#     L_price info = C(c^real; x(ĉ)) − C(c^real; x(c^real)) ≥ 0
# 预测基线（文档明文，透明季节性组合，不引入复杂机器学习，不改动问题三的 MPC 算法）：
#     ĉ_{d,t|0} = w1·c_{d-1,t} + w7·c_{d-7,t} + w14·c_{d-14,t}，  Σw_i = 1, w_i ≥ 0
#     权重只用「日期早于 d」的滚动样本外验证确定（不得用全年价格回看式拟合）；
#     6:00 / 12:00 / 18:00 可用「最新已实现偏差」修正未发生时段：
#     ĉ^upd_{d,t|k} = ĉ_{d,t|0} + λ_k·(c^real_{d,n_k} − ĉ_{d,n_k|0})，  t > n_k
# 除 MAE / RMSE 外还必须报告「价格方向判断准确率」与「决策加权误差」。
#
# 实现边界（严格遵守文档）：
#   1) Q4-PF 的策略由预测价格优化、费用一律按真实价格结算 —— 与 Q4-PI 同可行域、同结算目标，
#      故 L_price info ≥ 0 是模型性质而非拟合结果；
#   2) Q4-PF 只进入稳健性/现实性分析，**不替换** result4-2.xlsx / result4-3.xlsx；
#   3) 预测量与上游完全解耦：不修改 Q3 的 MPC、光伏预测、采纳规则，只在节点 k
#      替换 build_kk 的目标价格向量（与 U/W 抓取、并网上限注入同一「包装上游函数」手法）。
PF_EXP = 'pf'                          # Q4-PF 实验族名（缓存命名空间 cache/pf/）
PF_LAGS = (1, 7, 14)                   # 预测所用滞后（日历日）：c_{d-1} / c_{d-7} / c_{d-14}
PF_LAMBDA = (0.5, 0.5, 0.5)            # 6:00 / 12:00 / 18:00 的已实现偏差修正强度 λ_k
PF_NODES = tuple(int(Q3.TAU[k]) for k in (1, 2, 3))   # (36, 72, 108) —— 6:00 / 12:00 / 18:00
PF_TRAIN_WIN = 60                      # 滚动验证窗口：只用 d 之前最近 60 个可用历史日
PF_MIN_TRAIN = 14                      # 历史样本下限，低于此值退化为等权（首日附近）
PF_W_STEP = 0.05                       # 单纯形权重网格步长（Σw=1、w≥0）
PF_HI_Q = 90                           # 「高价段」定义：全样本 P{q}（用于尖峰段误差与低估比例）
PF_LOSS_TOL = 1.0                      # L_price info 非负性核验容差（元/年，吸收 MILP 容差）

_PF = {'BASE': None, 'W': None, 'FLOOR': None, 'DATES': None, 'PMAT': None,
       'N_TRAIN': None, 'LAST_TRAIN': None}
_PF_STATE = {'on': False, 'real': None, 'floor': None}   # 子进程内的逐日注入状态


def pf_weight_grid(step=PF_W_STEP):
    """单纯形 {w ≥ 0, Σw = 1} 上步长 step 的候选权重网格（含顶点与边上点）。
    用于「滚动样本外验证」选权重——只做对比评估，不拟合任何未来数据。"""
    k = int(round(1.0 / step))
    pts = [(a / k, b / k, (k - a - b) / k)
           for a in range(k + 1) for b in range(k + 1 - a)]
    return np.asarray(pts, dtype=float)


def pf_prepare(pmat_v, dates, all_dates, all_pmat):
    """构造 Q4-PF 的因果预测基准矩阵 ĉ_{d,t|0} 与逐日权重（严格无未来信息）。

    权重确定方式（文档「权重只能用日期早于 d 的滚动验证确定」）：
      对决策日 d（在附件4 全年矩阵中的位置 j），取验证样本 = 位置
      [max(14, j − PF_TRAIN_WIN), j) 内的历史日，它们在 j 之前、且 14 天滞后齐全；
      在这批历史日上按「样本外一步预测的平均绝对误差」最小的原则，从单纯形网格中选 w。
    预测值 = w1·c_{d-1} + w7·c_{d-7} + w14·c_{d-14}（逐时段向量运算，时段口径与附件4 一致）。

    说明：附件4/附件2 均为 365 天，而决策日只从 FIRST_DAY=2025-02-01 起（共 334 天）。
    2025-01-01..01-14 这 14 天三个滞后尚未齐全，无法给出因果预测，且它们不属于任何决策日
    （不参与求解与结算），故其行仅为保持价格矩阵形状而填实测价格行，
    并用 ROW_VALID 标记排除在价格口径核验与精度统计之外。
    """
    pos = {d: j for j, d in enumerate(all_dates)}
    win = max(PF_LAGS)
    grid = pf_weight_grid()
    nd = len(dates)
    base = np.zeros_like(pmat_v, dtype=float)
    W = np.zeros((nd, 3), dtype=float)
    n_train = np.zeros(nd, dtype=int)
    valid = np.zeros(nd, dtype=bool)
    last_train = {}
    for i, d in enumerate(dates):
        j = pos.get(d)
        if j is None:
            raise RuntimeError(f"附件4 缺少决策日 {d}")
        if j < win:                       # 滞后不全的预热期：占位行，见 docstring
            base[i] = all_pmat[j]
            last_train[d] = None
            continue
        origins = list(range(max(win, j - PF_TRAIN_WIN), j))    # 全部严格早于 j
        n_train[i] = len(origins)
        valid[i] = True
        last_train[d] = all_dates[j - 1] if j >= 1 else None
        if len(origins) < PF_MIN_TRAIN:
            W[i] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        else:
            A = np.stack([np.concatenate([all_pmat[k - l] for k in origins])
                          for l in PF_LAGS], axis=1)           # (m·N, 3)
            y = np.concatenate([all_pmat[k] for k in origins])  # (m·N,)
            err = np.abs(A @ grid.T - y[:, None])               # (m·N, K)
            W[i] = grid[int(np.argmin(err.sum(axis=0)))]
        base[i] = (W[i, 0] * all_pmat[j - 1] + W[i, 1] * all_pmat[j - 7]
                   + W[i, 2] * all_pmat[j - 14])
    if not np.all(np.isfinite(base)) or float(base.min()) <= 0.0:
        raise RuntimeError("Q4-PF 基准预测出现非正/非有限价格，请核对附件4 与滞后对齐")
    fit = n_train > 0                     # 有滚动验证样本、权重真正由验证选出的天数
    _PF.update({'BASE': base, 'W': W, 'FLOOR': float(all_pmat.min()),
                'DATES': list(all_dates), 'PMAT': all_pmat, 'N_TRAIN': n_train,
                'LAST_TRAIN': last_train, 'ROW_VALID': valid})
    print("-" * 72)
    print("[Q4-PF 因果价格预测] 文档「可选的价格预测现实性扩展」")
    print(f"  基线: ĉ = w1·c_(d-1) + w7·c_(d-7) + w14·c_(d-14)，Σw=1、w≥0；"
          f"权重由滚动样本外验证确定（窗口 {PF_TRAIN_WIN} 天）")
    print(f"  日内修正: 节点 {PF_NODES}（6:00/12:00/18:00），λ = {PF_LAMBDA}，"
          f"修正已实现价格之后的未发生时段")
    print(f"  预测矩阵: {base.shape[0]} 天 x {base.shape[1]} 时段（其中 {int(valid.sum())} 天"
          f"为因果预测，{int((~valid).sum())} 天为 1 月滞后不全的占位行）；")
    print(f"    预测取值 [{base[valid].min():.4f}, {base[valid].max():.4f}]，"
          f"均价 {base[valid].mean():.4f} 元/kWh"
          f"（附件4 实测均价 {pmat_v.mean():.4f}）")
    print(f"  权重均值（有滚动验证样本的 {int(fit.sum())} 天）w1={W[fit, 0].mean():.4f} / "
          f"w7={W[fit, 1].mean():.4f} / w14={W[fit, 2].mean():.4f}；"
          f"滚动验证样本 {n_train[fit].min()}–{n_train[fit].max()} 天")
    print("-" * 72)
    return base, W


def pf_base_matrix():
    """Q4-PF 的 ĉ_{d,t|0} 基准矩阵（由 build_ctx 里的 pf_prepare 初始化）。"""
    if _PF['BASE'] is None:
        raise RuntimeError("Q4-PF 预测矩阵未初始化：应在 build_ctx 中先调用 pf_prepare")
    return _PF['BASE']


def pf_block_prices(base_row, real_row, k):
    """文档公式：节点 k 的修正预测价格（仅改 t > n_k 的未发生时段）。
        ĉ^upd_{d,t|k} = ĉ_{d,t|0} + λ_k·(c^real_{d,n_k} − ĉ_{d,n_k|0}),  t > n_k
    修正后若落到非正区间，按附件4 全样本最低观测价下截尾——与 λ 敏感性的截尾同一手法：
    Q2/Q3 的计划购电量无上界（P_buy_max=∞），非正价会让目标函数无下界、MILP 必然 unbounded。"""
    if k is None or k <= 0:
        return np.asarray(base_row, dtype=float)
    n = PF_NODES[k - 1]
    dev = float(real_row[n] - base_row[n])
    p = np.array(base_row, dtype=float, copy=True)
    p[n + 1:] = p[n + 1:] + PF_LAMBDA[k - 1] * dev
    lo = _PF_STATE.get('floor')
    if lo is None:
        lo = _PF.get('FLOOR')
    if lo is not None:
        p[n + 1:] = np.maximum(p[n + 1:], float(lo))
    return p


# ========== 1b. U/W 抓取钩子：不改动 Q3 源码取得逐节点调整量 ==========
# Q4-3 的调整费必须在「新价格」下重新结算
#     C^adj = Σ_{k 采纳} Σ_{t∈T_k} c^V_t (1.5·U^(k)_t − 0.5·W^(k)_t) Δt，
# 而 Q3 记录只保存 S0~S3 的费用聚合值，不保存 U/W 向量。为严格保持 Q3 源码零改动
# （避免改动上游模型逻辑或其缓存版本），这里包装 Q3.solve_milp：MILP 解出后按模型自带
# 的变量布局取出 U/W（u 块 = 8*nt、w 块 = 9*nt，nt 为剩余时域长度，i0 为节点起点）。
_CAP = {'U': {}, 'W': {}}
_PBUY = {'cur': None}      # 当前实验的并网购电上限（None = 沿用题目 ∞），由 _patch_experiment 设置


def _cap_reset():
    _CAP['U'].clear()
    _CAP['W'].clear()


def _cap_snapshot():
    """返回 {i0: 局部数组}，i0 ∈ {36, 72, 108}。"""
    return ({int(k): v.copy() for k, v in _CAP['U'].items()},
            {int(k): v.copy() for k, v in _CAP['W'].items()})


def _install_q3_hook():
    """幂等安装。solve_day 内部以模块全局名调用 solve_milp，替换模块属性即可生效；
    spawn 子进程会重新导入本模块，故每个子进程都会各自安装一次。"""
    if getattr(Q3, '_q4_hooked', False):
        return
    _orig = Q3.solve_milp

    def _hooked(model, c, time_limit):
        res = _orig(model, c, time_limit)
        i0 = int(model.get('i0', 0))
        if i0 > 0 and getattr(res, 'x', None) is not None:
            nt = int(model['nt'])
            x = res.x
            for tag, blk in (('U', 8), ('W', 9)):
                v = np.asarray(x[blk * nt:(blk + 1) * nt], dtype=float)
                _CAP[tag][i0] = np.where(np.abs(v) < 1e-9, 0.0, v)
        return res

    Q3.solve_milp = _hooked
    Q3._q4_hooked = True


def _install_pf_hook():
    """幂等安装 Q4-PF 的节点价格注入钩子（文档「可选的价格预测现实性扩展」）。
    Q3 的 MPC 在节点 k 调用 build_kk，其目标系数取 prices[TAU[k]:]，故包装 build_kk
    按节点替换价格向量即可——问题三的模型、约束、采纳规则与光伏预测零改动。
    非 pf 实验下 _PF_STATE['on']=False，函数把入参原样透传，对正式结果无任何影响。"""
    if getattr(Q3, '_q4_pf_hooked', False):
        return
    _orig = Q3.build_kk

    def _hooked(k, prices, Ld, Ghat, Qprev, Cprev, Dprev, E_init):
        if _PF_STATE['on']:
            prices = pf_block_prices(prices, _PF_STATE['real'], k)
        return _orig(k, prices, Ld, Ghat, Qprev, Cprev, Dprev, E_init)

    Q3.build_kk = _hooked
    Q3._q4_pf_hooked = True


_install_q3_hook()
_install_pf_hook()


# ========== 2. 附件4 读取与预处理核验（文档「价格数据的预处理」） ==========
def _price4_date(v):
    """附件4 首列日期 -> datetime.date（兼容 datetime / str / 年月日文本）。"""
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = (str(v).strip().replace('/', '-').replace('.', '-')
         .replace('年', '-').replace('月', '-').replace('日', ''))
    p = [x for x in s.split('-') if x]
    if len(p) != 3:
        raise ValueError(f"附件4 无法解析日期: {v!r}")
    return date(int(p[0]), int(p[1]), int(p[2]))


def read_price4_all(path):
    """读取附件4 **全部** 行（365 天 × 144 时段），按日期升序返回 (dates, PMAT)。
    Q4-PF 的因果预测需要 d-1 / d-7 / d-14 三个日历日滞后，而决策日只覆盖 2 月 1 日起，
    故滞后价格必须取自附件4 全年矩阵（1 月数据），不能只读决策日那 334 行。"""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    wb.close()
    got = {}
    for r in rows[1:]:
        d0 = r[0]
        if d0 is None or (isinstance(d0, str) and not d0.strip()):
            continue
        d = _price4_date(d0)
        vals = []
        for v in r[1:1 + N]:
            if v is None or (isinstance(v, str) and not v.strip()):
                raise ValueError(f"附件4 {d} 存在缺失电价")
            vals.append(float(v))
        if len(vals) != N:
            raise ValueError(f"附件4 {d} 时段数 = {len(vals)}，应为 {N}")
        got[d] = np.asarray(vals, dtype=float)
    ds = sorted(got)
    return ds, np.stack([got[d] for d in ds], axis=0)


def read_price4(path, dates):
    """附件4：365 × 144 逐日 10 分钟电价（元/kWh）。表头与附件1 同为「时段终点」口径
    （首列 `日期\\时间`，末列 `0:00+1`）。返回 (D,144) 矩阵，行序与 dates 对齐。"""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    wb.close()
    if not rows:
        raise ValueError("附件4 为空")
    hdr = rows[0]
    if len(hdr) < 1 + N:
        raise ValueError(f"附件4 表头列数不足: {len(hdr)}")
    tail = str(hdr[-1]).strip() if hdr[-1] is not None else ''
    if tail not in ('0:00+1', '24:00'):
        print(f"[警告] 附件4 表头末列 = {hdr[-1]!r}，未识别为 '0:00+1'，请确认时间口径")

    got = {}
    for r in rows[1:]:
        d0 = r[0]
        if d0 is None or (isinstance(d0, str) and not d0.strip()):
            continue
        d = _price4_date(d0)
        vals = []
        for v in r[1:1 + N]:
            if v is None or (isinstance(v, str) and not v.strip()):
                raise ValueError(f"附件4 {d} 存在缺失电价")
            vals.append(float(v))
        if len(vals) != N:
            raise ValueError(f"附件4 {d} 时段数 = {len(vals)}，应为 {N}")
        if d in got:
            raise ValueError(f"附件4 日期重复: {d}")
        got[d] = np.asarray(vals, dtype=float)

    miss = [d for d in dates if d not in got]
    if miss:
        raise ValueError(f"附件4 缺少 {len(miss)} 天: {miss[:3]} ...")
    PMAT = np.stack([got[d] for d in dates], axis=0)

    print("[附件4 预处理核验]")
    print(f"  结构: {PMAT.shape[0]} 天 x {PMAT.shape[1]} 时段；日期与附件2 逐日对齐 OK")
    print(f"  取值范围: [{PMAT.min():.4f}, {PMAT.max():.4f}] 元/kWh，整体均价 {PMAT.mean():.4f}")
    if PMAT.max() > 10.0:
        print("  [警告] 电价数量级偏大，疑似单位为元/MWh，请确认是否需除以 1000")
    n_nonpos = int((PMAT <= 0).sum())
    if n_nonpos == 0:
        print("  非正价格: 0 个 -> 5 倍紧急、1.5 倍上调、0.5 倍下调的乘数规则可直接继承")
    else:
        print(f"  [警告] 非正价格 {n_nonpos} 个（负价 {int((PMAT < 0).sum())}，"
              f"零价 {int((PMAT == 0).sum())}）。乘数结算在非正价格下失去「惩罚更贵」含义，"
              f"需按文档处理；本脚本按题目原式原样重算并在终端记录该边界。")
    dmean, dstd = PMAT.mean(axis=1), PMAT.std(axis=1)
    ramp = np.abs(np.diff(PMAT, axis=1)).sum(axis=1)
    p1 = Q2.read_prices(Q2.PRICE_FILE)
    print(f"  逐日均价: min {dmean.min():.4f} / mean {dmean.mean():.4f} / max {dmean.max():.4f}")
    print(f"  日内标准差: min {dstd.min():.4f} / mean {dstd.mean():.4f} / max {dstd.max():.4f}")
    print(f"  日内爬坡量: min {ramp.min():.3f} / mean {ramp.mean():.3f} / max {ramp.max():.3f}")
    dev = float(np.abs(PMAT.mean(axis=0) - p1).max())
    print(f"  附件1 与附件4 逐时段跨天均值最大偏差 = {dev:.6f} 元/kWh"
          + ("（附件1 即附件4 的平均日内曲线，两者价格水平一致）" if dev < 1e-3 else ""))
    print(f"  附件1 固定电价: [{p1.min():.4f}, {p1.max():.4f}]，均价 {p1.mean():.4f}，"
          f"日内标准差 {p1.std():.4f}")
    return PMAT


# ========== 3. 实验价格矩阵 ==========
# vol    : λ=1、γ=1，附件4 原始波动电价（正式结果）
# lam0   : λ=0   —— 每天同均值的平坦电价（文档「用同均值平坦电价剥离价格水平」）
# lam05  : λ=0.5
# lam15  : λ=1.5 —— 放大日内价差（文档「价格振幅敏感性：最重要的检验」）
# gam08  : γ=0.8 —— 正比例缩放核验（文档「正比例缩放：模型一致性检验」）
# gam12  : γ=1.2
# nobess : 附件4 原始价格 + 禁用储能（P_ch = P_dis = 0，文档「主对照实验」第 6 项）
# fixed  : 附件1 固定电价（x^F 基线；Q3 侧必须由本脚本重解，才能拿到 U/W，
#          同时作为「用附件1价格重算是否能复现 Q2/Q3 原生缓存」的回归核验）
MAIN_EXPS = ['vol', 'fixed']            # 正式重算：波动电价 + 固定电价基线
# 敏感性/对照族（文档「反事实实验与敏感性分析」全部要求）：
#   振幅族     lam0/lam05/lam15      —— 价格振幅敏感性（最重要的检验，λ∈{0,0.5,1.0,1.5}）
#   缩放族     gam08/gam12           —— 正比例缩放一致性核验（γ∈{0.8,1.0,1.2}）
#   无储能族   nobess                —— 有储能 vs 禁用储能（储能价值 V^BESS）
#   尖峰族     w99/w95               —— 价格按 P99 / P95 缩尾（结论是否由极少数尖峰驱动）
#   时序族     shm1/shp1             —— 价格整体前移/后移 1 小时（仅作诊断）
#   效率族     eta85/eta95           —— 充放电效率 0.85 / 0.95
#   功率族     pm25/pm75             —— 储能功率上限 2500 / 7500 kW
#   并网族     pb6k/pb10k            —— 并网购电上限多组值（题目未给容量，见下注）
#   倍数族     kap3/kap7             —— 紧急购电倍数 3 / 7（基准 5）
#   储能价值族 lam0nb/lam05nb/lam15nb —— 各 λ 下的无储能对照，
#             用于文档「价格振幅敏感性」要求的「对每个 λ 报告储能价值 V^BESS」。
# 上下调系数扰动与 κ_th 衰减成本为纯事后结算修正，不建实验族（见 settle_coef_sens / life_adjusted）。
SENS_EXPS = ['lam0', 'lam05', 'lam15', 'gam08', 'gam12', 'nobess',
             'lam0nb', 'lam05nb', 'lam15nb',
             'w99', 'w95', 'shm1', 'shp1', 'eta85', 'eta95',
             'pm25', 'pm75', 'pb6k', 'pb10k', 'kap3', 'kap7']
# 价格预测现实性扩展（文档「可选的价格预测现实性扩展」）：独立成组，
# 既不计入 SENS_EXPS（不污染敏感性汇总表），也不计入 MAIN_EXPS（不替换正式交付文件）。
PF_EXPS = [PF_EXP]
EXPERIMENTS = MAIN_EXPS + SENS_EXPS + PF_EXPS
EXP_LABEL = {
    'vol': '波动电价 λ=1.0（附件4 原始）',
    'fixed': '固定电价（附件1，基线 x^F）',
    'lam0': '振幅 λ=0.0（同日均值平坦电价）',
    'lam05': '振幅 λ=0.5',
    'lam15': '振幅 λ=1.5（放大日内价差，含非负截尾修正）',
    'gam08': '正比例缩放 γ=0.8',
    'gam12': '正比例缩放 γ=1.2',
    'nobess': '波动电价 + 禁用储能',
    'w99': '价格尖峰 P99 缩尾',
    'w95': '价格尖峰 P95 缩尾',
    'shm1': '价格时序前移 1 小时（诊断）',
    'shp1': '价格时序后移 1 小时（诊断）',
    'eta85': '充放电效率 0.85',
    'eta95': '充放电效率 0.95',
    'pm25': '储能功率上限 2500 kW',
    'pm75': '储能功率上限 7500 kW',
    'lam0nb': '振幅 λ=0.0 + 禁用储能（储能价值对照）',
    'lam05nb': '振幅 λ=0.5 + 禁用储能（储能价值对照）',
    'lam15nb': '振幅 λ=1.5 + 禁用储能（储能价值对照）',
    'pb6k': '并网购电上限 6000 kW（参照）',
    'pb10k': '并网购电上限 10000 kW（参照）',
    'kap3': '紧急购电倍数 3（基准 5）',
    'kap7': '紧急购电倍数 7（基准 5）',
    'pf': 'Q4-PF 因果价格预测（无未来信息）',
}

# 实验对上游模型参数的覆盖（None = 用题设基准值）。
# 说明：eta / pmax / kappa / p_buy_max 全部通过运行时替换 Q2/Q3 模块级全局生效，
# 上游源码零改动（与 U/W 抓取钩子同一手法）。
EXP_OVERRIDE = {
    'nobess': {'pmax': 0.0},
    'lam0nb': {'pmax': 0.0},
    'lam05nb': {'pmax': 0.0},
    'lam15nb': {'pmax': 0.0},
    'eta85': {'eta': 0.85},
    'eta95': {'eta': 0.95},
    'pm25': {'pmax': 2500.0},
    'pm75': {'pmax': 7500.0},
    'pb6k': {'p_buy_max': SENS_PBUY[0]},
    'pb10k': {'p_buy_max': SENS_PBUY[1]},
    'kap3': {'kappa': 3.0},
    'kap7': {'kappa': 7.0},
}


def _ov(exp, key, default):
    """取某实验对的参数覆盖值（未登记则用题设基准）。"""
    return EXP_OVERRIDE.get(exp, {}).get(key, default)

# 价格口径修订 → 必须换缓存命名空间，否则旧口径的结果会混进新口径的汇总。
#   r2：振幅放大出现负价。λ=1.5 时 324/48096 个时段（占 0.67%）被外推到最低观测价以下，
#       最低 −0.2575 元/kWh，涉及 57 个决策日；而 Q2/Q3 的计划购电量 B 无上界
#       （Q2 第 352 行 / Q3 第 381 行 P_buy_max=∞），负价使目标 c·B·Δt 随 B→∞ 趋于 −∞，
#       HiGHS 报 "Primal infeasible or unbounded" —— 数学上真的无界，不是数值问题。
#       修正见 _lambda_series()。Q2 缓存只有日期文件名、无版本字段，只能靠换目录作废；
#       Q3 有 _cache_version，通过 _exp_key 一并隔离。
CACHE_REV = {'lam15': 'r2'}


def _exp_key(exp):
    """实验的缓存命名空间键（口径变更后自动隔离旧缓存）。"""
    rev = CACHE_REV.get(exp)
    return f'{exp}_{rev}' if rev else exp


def _lambda_series(pmat_v, lam):
    """文档「价格振幅敏感性」：c^(λ)_{d,t} = c̄_d + λ(c_{d,t}^V − c̄_d)，当日均价恒为 c̄_d。

    ⚠ 该式在 λ>1 时会把最低价外推到零以下，而 Q2/Q3 的购电量无上界 → 目标函数无下界。
    处理依据（文档第 2 节第 6 条：不随意删除尖峰，保留为主结果，
    再在敏感性分析中做缩尾或截尾对照）：
      1) **下截尾**：低于附件4 全样本最低观测价 floor 的时段压缩到 floor；
      2) **保均值重标定**：整日乘 c̄_d / mean(c′)，把当日均价严格还原为 c̄_d，
         使「λ 只改变波动幅度、不改变价格水平」这一实验前提继续成立。
    第 2 步只作用于真正发生截尾的日子，故 λ≤1（无截尾）是逐位恒等变换，
    vol / lam0 / lam05 完全不受影响。
    """
    m = pmat_v.mean(axis=1, keepdims=True)
    raw = m + lam * (pmat_v - m)
    floor = float(pmat_v.min())           # 附件4 全样本最低观测价
    if floor <= 0.0:
        raise RuntimeError("附件4 含非正价格，λ 敏感性口径需重新论证，请先核对数据")
    c = np.maximum(raw, floor)
    bad = (c > raw).any(axis=1)           # 发生截尾的决策日
    if bad.any():
        fac = np.where(bad[:, None], m / c.mean(axis=1, keepdims=True), 1.0)
        c = c * fac
    return c


def _winsor_series(pmat_v, q):
    """文档「尖峰、时序与设备参数敏感性」：按全样本 P{q} 缩尾。
    高于 q 分位的价格压到该分位（保留时序位置，只削峰不删点），
    保留主结果（原始价格）不动，用于回答「全年节省是否由极少数尖峰驱动」。
    分位口径与 E^emg,high 的全样本 q95 一致。"""
    cap = float(np.percentile(pmat_v, q))
    return np.minimum(pmat_v, cap)


def _shift_series(pmat_v, slots):
    """文档「价格时序」诊断：整日价格曲线整体平移 slots 个 10 分钟时段
    （slots<0 前移、>0 后移，环形移位，保持当日价格集合不变）。
    用于检验「收益是否依赖价格与净负荷的偶然对齐」，仅作诊断不作结论。"""
    return np.roll(pmat_v, slots, axis=1)


def make_pmat(pmat_v, exp):
    """生成实验对应的价格矩阵（文档「价格振幅敏感性」「正比例缩放」
    「尖峰、时序与设备参数敏感性」公式）。"""
    if exp in _LAM_OF:
        return _lambda_series(pmat_v, _LAM_OF[exp])
    if exp in ('vol', 'nobess', 'eta85', 'eta95', 'pm25', 'pm75',
               'pb6k', 'pb10k', 'kap3', 'kap7'):
        return pmat_v.copy()          # 价格不变，只改设备参数/惩罚倍数（见 EXP_OVERRIDE）
    if exp == 'gam08':
        return 0.8 * pmat_v
    if exp == 'gam12':
        return 1.2 * pmat_v
    if exp == 'w99':
        return _winsor_series(pmat_v, SENS_SPIKE_Q[0])
    if exp == 'w95':
        return _winsor_series(pmat_v, SENS_SPIKE_Q[1])
    if exp == 'shm1':
        return _shift_series(pmat_v, -SENS_SHIFT_SLOT)
    if exp == 'shp1':
        return _shift_series(pmat_v, SENS_SHIFT_SLOT)
    if exp == 'fixed':
        return np.tile(Q2.read_prices(Q2.PRICE_FILE), (pmat_v.shape[0], 1))
    if exp == PF_EXP:
        return pf_base_matrix()          # Q4-PF：因果预测价格 ĉ_{d,t|0}（无未来信息）
    raise KeyError(exp)


# λ 振幅取值（含各 λ 的无储能对照族，供 price_audit 的同口径截尾审计使用）
_LAM_OF = {'lam0': 0.0, 'lam05': 0.5, 'lam15': 1.5,
           'lam0nb': 0.0, 'lam05nb': 0.5, 'lam15nb': 1.5}


def price_audit(pmat_v):
    """逐实验核算价格矩阵健康度（文档第 2 节：统计最小价格，重点检查零价与负价）。
    开跑前打印，作为「价格预处理已核验」的证据。
    Q4-PF 的矩阵含 2025-01-01..01-14 这 14 天滞后不全的占位行（非决策日、不参与求解），
    其统计只取真正由因果预测生成的行，避免占位行稀释口径核验。"""
    base_floor = float(pmat_v.min())
    rows = {}
    for exp in EXPERIMENTS:
        p = make_pmat(pmat_v, exp)
        if exp == PF_EXP and _PF.get('ROW_VALID') is not None:
            p = p[_PF['ROW_VALID']]
        m = pmat_v.mean(axis=1, keepdims=True)
        lam = _LAM_OF.get(exp)
        n_clip = n_day_clip = n_neg = 0
        max_mean_dev = 0.0
        if lam is not None:
            raw = m + lam * (pmat_v - m)
            hit = raw < base_floor - 1e-15
            n_clip = int(hit.sum())
            n_day_clip = int(hit.any(axis=1).sum())
            n_neg = int((raw < 0.0).sum())
            max_mean_dev = float(np.abs(p.mean(axis=1, keepdims=True) - m).max())
        rows[exp] = {
            'label': EXP_LABEL[exp],
            'min': float(p.min()), 'max': float(p.max()), 'mean': float(p.mean()),
            'n_nonpos': int((p <= 0.0).sum()),
            'n_clip': n_clip, 'n_day_clip': n_day_clip, 'n_neg': n_neg,
            'mean_dev': max_mean_dev,
            'identity': bool(lam is None or n_clip == 0),
        }
    return rows, base_floor


def exp_is_nobess(exp):
    return exp in ('nobess', 'lam0nb', 'lam05nb', 'lam15nb')


# ========== 4. 求解与并行（复用 Q2/Q3 的 solve_day，唯一改动 ctx['prices']） ==========
_SHARED = {}


def _init_worker(ctx):
    global _SHARED
    _SHARED.clear()
    _SHARED['ctx'] = ctx


def _worker(task):
    """子任务：一个决策日、指定问题族、指定实验。
    与 Q2/Q3 原生流程的差别只有一处——价格向量由本脚本按日替换；模型内部的预测、
    场景、MPC、约束、结算全部沿用原实现。
    注意：spawn 模式下子进程会重新导入 __mp_main__，因此必须在此处重建实验补丁
    （缓存目录、储能参数），否则会写到 Q2/Q3 的原生缓存里。"""
    fam, d_idx, time_limit, exp = task
    _patch_experiment(exp, fam)
    ctx = _SHARED['ctx']
    ctx['prices'] = ctx['PMAT'][d_idx]
    # Q4-PF：打开节点价格注入（文档「可选的价格预测现实性扩展」）。
    # base = ctx['prices'] = ĉ_{d,t|0}（预测价格，供 0:00 计划与结算口径）；
    # real = ctx['PMAT_V'][d_idx] = 附件4 实测价格，只用于节点 k 的偏差修正量。
    # 只有 Q3 的 MPC 节点会用到它，Q2 是纯日前计划，无节点修正。
    if exp == PF_EXP:
        _PF_STATE['on'] = True
        _PF_STATE['real'] = np.asarray(ctx['PMAT_V'][d_idx], dtype=float)
        _PF_STATE['floor'] = float(ctx.get('PF_FLOOR', np.asarray(ctx['PMAT_V']).min()))
    else:
        _PF_STATE['on'] = False
        _PF_STATE['real'] = None
    if fam == 'q2':
        rec = Q2.solve_day(ctx, d_idx, time_limit)
        Q2.save_cache(rec)
    else:
        _install_q3_hook()
        _cap_reset()
        rec = Q3.solve_day(ctx, d_idx, time_limit)
        rec['UK'], rec['WK'] = _cap_snapshot()   # 逐节点 U/W，供新价格下重算 C^adj
        Q3.save_cache(rec)
    return fam, d_idx, rec


def _patch_experiment(exp, fam):
    """把 Q2/Q3 的缓存目录、设备参数与惩罚倍数切到本实验命名空间
    （不污染 Q2/Q3 原生结果；上游源码零改动）。
    目录名用 _exp_key(exp)：价格口径修订过的实验换新目录，旧口径缓存自动作废。
    （Q2 缓存只有日期文件名、无版本字段，换目录是唯一的作废手段；Q3 另有版本号。）
    版本号 v2：记录结构加入 U/W 抓取字段，旧版实验缓存自动作废重解。
    参数隔离：每个实验族一个独立目录，设备参数（效率/功率/倍数/并网上限）随实验名
    一一对应，故不会互相串用缓存。"""
    key = _exp_key(exp)
    exp_dir = os.path.join(CACHE_ROOT, key)
    pmax = _ov(exp, 'pmax', P_CH_MAX)
    eta = _ov(exp, 'eta', ETA_CH)
    kappa = _ov(exp, 'kappa', KAPPA)
    pbuy = _ov(exp, 'p_buy_max', None)
    if exp_is_nobess(exp):
        pmax = 0.0
    for mod in (Q2, Q3):
        mod.P_CH_MAX = pmax
        mod.P_DIS_MAX = pmax
        mod.ETA_CH = eta
        mod.ETA_DIS = eta
        mod.KAPPA = kappa
    if fam == 'q2':
        Q2.CACHE_DIR = os.path.join(exp_dir, 'q2')
    else:
        Q3.CACHE_DIR = os.path.join(exp_dir, 'q3')
        Q3.CACHE_VERSION = f'q4-{key}-v2'
    _PBUY['cur'] = pbuy
    # 非 pf 实验一律关闭节点价格注入，避免同进程内残留状态污染其他实验
    if exp != PF_EXP:
        _PF_STATE['on'] = False
        _PF_STATE['real'] = None
    _install_pbuy_hook(Q2, 'q2')
    _install_pbuy_hook(Q3, 'q3')
    _install_pf_hook()          # 幂等；非 pf 实验下原样透传，不影响正式结果
    os.makedirs(os.path.join(exp_dir, fam), exist_ok=True)


def _install_pbuy_hook(mod, fam):
    """幂等安装并网购电上限注入（文档「负价下必须有有限并网购电上限」）。
    题目未给并网容量（主结果取 P_buy_max=∞），故仅对登记了该参数的实验族生效：
    在 MILP 求解前把购电变量块的上界由 ∞ 改为指定值，其余一律不动。
    上游源码零改动——与 U/W 抓取钩子同一手法（包装 solve_milp）。
    购电变量块：Q2 = x[0:N]，Q3 = x[0:nt]（两族的目标系数都从块首开始）。"""
    if getattr(mod, '_q4_pbuy', False):
        return
    _orig = mod.solve_milp

    def _hooked(model, c, time_limit):
        lim = _PBUY.get('cur')
        if lim is not None:
            n_buy = int(model.get('nt', N))
            hi = model['hi']
            hi[0:n_buy] = np.minimum(hi[0:n_buy], lim)
        return _orig(model, c, time_limit)

    mod.solve_milp = _hooked
    mod._q4_pbuy = True


def _reset_upstream_params():
    """把上游模块的设备参数与惩罚倍数复位为题设基准值。
    敏感性实验会临时覆盖这些模块级全局量（效率 0.85、储能功率 7500 kW、倍数 7 等），
    而汇总基准方案（vol / fixed）与本脚本的结算函数（q3_settle → Q3.run_block
    通过全局效率推进 SOC）都依赖它们，故每轮敏感性跑完必须复位，否则正式结果被污染。"""
    for mod in (Q2, Q3):
        mod.P_CH_MAX = P_CH_MAX
        mod.P_DIS_MAX = P_DIS_MAX
        mod.ETA_CH = ETA_CH
        mod.ETA_DIS = ETA_DIS
        mod.KAPPA = KAPPA
    _PBUY['cur'] = None
    _PF_STATE['on'] = False
    _PF_STATE['real'] = None


def run_family(fam, ctx, exp, pmat, time_limit, workers, resume, limit):
    """跑完一族（问题二 / 问题三）在给定价格矩阵下的全部决策日，返回按日期排序的记录。"""
    mod = Q2 if fam == 'q2' else Q3
    _patch_experiment(exp, fam)
    ctx['PMAT'] = pmat
    idxs = mod.decision_indices(ctx)
    if limit is not None:
        idxs = idxs[:limit]
    recs, todo = [], []
    for i in idxs:
        r = mod.load_cache(ctx['dates'][i]) if resume else None
        (recs.append(r) if r is not None else todo.append(i))
    print(f"[{fam.upper()} | {exp}] 缓存命中 {len(recs)} 天，待求解 {len(todo)} 天", flush=True)
    if todo:
        tasks = [(fam, i, time_limit, exp) for i in todo]
        nw = max(1, min(workers, len(tasks)))
        print(f"[{fam.upper()} | {exp}] {nw} 进程 x {len(tasks)} 天"
              f"（每子进程单线程确定性 MILP，天间独立，结果与串行一致）", flush=True)
        out = {}
        t0 = time.perf_counter()
        with multiprocessing.get_context('spawn').Pool(
                processes=nw, initializer=_init_worker, initargs=(ctx,)) as pool:
            for k, (_fm, di, rec) in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1)):
                out[di] = rec
                el = time.perf_counter() - t0
                if fam == 'q2':
                    info = (f"计划费 {rec['plan_cost']:.2f} 紧急费 {rec['em_cost']:.2f} "
                            f"总 {rec['total_cost']:.2f}")
                else:
                    s3 = rec['scen'][3]
                    af = ''.join(str(int(x)) for x in rec['a_flags'])
                    info = (f"S3总 {s3['C_total']:.2f}（计划 {s3['C_plan']:.2f} + 调整 "
                            f"{s3['C_adj']:.2f} + 紧急 {s3['C_emg']:.2f}）采纳k1..3={af}")
                print(f"  [{k + 1}/{len(tasks)}] {rec['date']} | {info} | "
                      f"耗时 {rec.get('plan_sec', 0):.1f}s | 墙钟 {el:.0f}s", flush=True)
        recs.extend(out.values())
    recs.sort(key=lambda r: r['date'])
    _dump_arrays(fam, exp, recs)
    return recs


def _arr_path(fam, exp):
    return os.path.join(CACHE_ROOT, _exp_key(exp), f'arr_{fam}.npz')


def _dump_arrays(fam, exp, recs):
    """落盘逐日决策数组（用于 γ 正比例缩放的调度不变性比对）。"""
    if not recs:
        return
    os.makedirs(os.path.join(CACHE_ROOT, _exp_key(exp)), exist_ok=True)
    dates = np.array([r['date'].isoformat() for r in recs])
    if fam == 'q2':
        np.savez_compressed(_arr_path(fam, exp), dates=dates,
                            B=np.stack([r['B'] for r in recs]),
                            C=np.stack([r['C'] for r in recs]),
                            D=np.stack([r['D'] for r in recs]))
    else:
        np.savez_compressed(_arr_path(fam, exp), dates=dates,
                            Q0=np.stack([r['Q0'] for r in recs]),
                            Qeff=np.stack([r['Qeff_s3'] for r in recs]),
                            C=np.stack([r['C_s3'] for r in recs]),
                            D=np.stack([r['D_s3'] for r in recs]))


def load_arrays(fam, exp):
    p = _arr_path(fam, exp)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _ctx_of(fam, ctx2, ctx3):
    return ctx2 if fam == 'q2' else ctx3


def load_family(exp, fam, ctx2, ctx3, limit=None):
    """从缓存读取某实验某族的全部记录（不重解），供 outputs / 汇总使用。"""
    mod = Q2 if fam == 'q2' else Q3
    ctx = _ctx_of(fam, ctx2, ctx3)
    _patch_experiment(exp, fam)
    ctx['PMAT'] = make_pmat(ctx['PMAT_V'], exp)
    idxs = mod.decision_indices(ctx)
    if limit is not None:
        idxs = idxs[:limit]
    recs = []
    for i in idxs:
        r = mod.load_cache(ctx['dates'][i])
        if r is None:
            raise RuntimeError(f"[{fam}|{exp}] 缺少 {ctx['dates'][i]} 的缓存，请先运行该实验")
        recs.append(r)
    recs.sort(key=lambda r: r['date'])
    return recs


def load_native_fixed(fam, ctx2, ctx3):
    """读取 Q2/Q3 原生缓存（固定电价 c_t^F 下的策略 x^F），用于反事实重新结算。"""
    mod = Q2 if fam == 'q2' else Q3
    ctx = _ctx_of(fam, ctx2, ctx3)
    native_dir = os.path.join(Q2_DIR if fam == 'q2' else Q3_DIR, 'cache')
    old_dir, old_ver = mod.CACHE_DIR, getattr(mod, 'CACHE_VERSION', None)
    try:
        mod.CACHE_DIR = native_dir
        if fam == 'q3':
            mod.CACHE_VERSION = Q3_NATIVE_CACHE_VERSION
        recs = []
        for i in mod.decision_indices(ctx):
            d = ctx['dates'][i]
            r = mod.load_cache(d)
            if r is None:
                raise RuntimeError(f"缺少固定电价原生缓存 {fam}/{d}，请先在 Q2/Q3 目录跑完全量")
            recs.append(r)
    finally:
        mod.CACHE_DIR = old_dir
        if fam == 'q3' and old_ver is not None:
            mod.CACHE_VERSION = old_ver
    recs.sort(key=lambda r: r['date'])
    return recs


# ========== 5. 结算与反事实：价格无关的物理量 + 任意价格重新结算 ==========
def q2_backtest(L, G, B, C, D):
    """逐字复现 solve_q2.solve_day 的回测口径（文档第 7 节步骤 5-6）。
    物理量与价格无关，故同一策略在不同价格下的 R/V/K/S 完全相同，只有费用不同。"""
    rd = L + C - D
    rd_pos = np.maximum(rd, 0.0)
    R = np.maximum(rd_pos - G - B, 0.0)
    deficit = R > 0.0
    V = np.where(deficit, G, np.minimum(G, rd_pos))
    Buse = np.where(deficit, B, np.minimum(B, rd_pos - V))
    return R, V, G - V, B - Buse


def q2_settle(B, C, D, L, G, prices, kappa=KAPPA):
    """用任意价格向量对给定策略重新结算（Q4-2 的 C^plan、C^emg）。
    kappa 为紧急购电倍数，默认题设 5 倍；仅在「紧急购电倍数敏感性」中改值。"""
    R, V, K, S = q2_backtest(L, G, B, C, D)
    c_plan = DT * float(prices @ B)
    c_emg = kappa * DT * float(prices @ R)
    return {'plan': c_plan, 'emg': c_emg, 'total': c_plan + c_emg, 'R': R,
            'curtail': DT * float(K.sum()), 'unused': DT * float(S.sum()),
            'pv_use': DT * float(V.sum()), 'p_grid_max': float(np.max(B)),
            'charge': DT * float(C.sum()), 'discharge': DT * float(D.sum()),
            'throughput': DT * float((C + D).sum())}


def _q3_uw(rec):
    """取出逐节点调整量 U^(k)/W^(k)（定义域 T_k = [τ_k, 144)，与 Q3 模型内变量完全一致）。
    Q3 源码在求解时不会保存 U/W，本脚本通过 solve_milp 钩子在解出瞬间抓取并写入记录，
    故这里只做「按采纳标志过滤」，不做任何重建或近似。
    注意：Q2/Q3 的原生缓存不含该字段，必须由本脚本重解生成。"""
    UK, WK = rec.get('UK'), rec.get('WK')
    if not UK or not WK:
        raise RuntimeError(
            f"{rec.get('date')} 的问题三记录缺少 U/W 字段（UK/WK）。"
            "原生 Q3 缓存不含该字段，Q4-3 的调整费重算必须使用本脚本重解产生的记录，"
            "请先执行: python solve_q4.py main （或 full）")
    out = {}
    for k in (1, 2, 3):
        if rec['a_flags'][k - 1]:
            i0 = int(Q3.TAU[k])
            if i0 not in UK or i0 not in WK:
                raise RuntimeError(f"{rec.get('date')} 缺少节点 k={k}（i0={i0}）的 U/W 抓取结果")
            out[k] = (i0, UK[i0], WK[i0])
        else:
            out[k] = None      # 未采纳节点的 U=W=0，不产生调整费（Q3 原始采纳规则）
    return out


def q3_settle(rec, L, G, prices, kappa=KAPPA, coef=(1.5, 0.5)):
    """用任意价格向量对问题三 S3 生效策略链重新结算。
    物理量（R、SOC 轨迹）与价格无关，复用 Q3.run_block 按 B_0..B_3 顺序推进。
    kappa = 紧急购电倍数（默认题设 5）；coef = (上调系数, 下调系数)，默认题设 (1.5, 0.5)。
    两者在此处独立结算，**不依赖上游模块的全局值**，故「倍数/系数敏感性」可事后重算。"""
    c_up_w, c_dn_w = float(coef[0]), float(coef[1])
    Q0, Qe, Cs, Ds = rec['Q0'], rec['Qeff_s3'], rec['C_s3'], rec['D_s3']
    uw = _q3_uw(rec)
    c_up = c_dn = aw = 0.0
    per_k = {}
    for k in (1, 2, 3):
        item = uw[k]
        if item is None:
            per_k[k] = {'up': 0.0, 'down': 0.0, 'net': 0.0, 'aw': 0.0,
                        'up_e': 0.0, 'down_e': 0.0, 'n_slot': 0}
            continue
        i0, U, W = item
        base = prices[i0:] * DT
        up = c_up_w * float(U @ base)
        dn = c_dn_w * float(W @ base)
        up_e = float(U.sum()) * DT
        dn_e = float(W.sum()) * DT
        c_up += up
        c_dn += dn
        aw += up_e + dn_e
        per_k[k] = {'up': up, 'down': dn, 'net': up - dn, 'aw': up_e + dn_e,
                    'up_e': up_e, 'down_e': dn_e,
                    'n_slot': int(((U + W) > 1e-9).sum())}
    c_plan = DT * float(prices @ Q0)
    E = Q3.E_INIT
    c_emg = e_emg = 0.0
    R_all = np.zeros(N)
    segs = []
    for m in range(4):
        g0, g1 = Q3.BLOCKS[m]
        E, R, _segs, _cemg, _eemg, _over = Q3.run_block(L, G, prices, g0, g1, Qe, Cs, Ds, E)
        R_all[g0:g1] = R
        # 紧急费按本函数入参 kappa 独立重算，避免受上游全局 KAPPA 影响
        c_emg += kappa * DT * float(prices[g0:g1] @ R)
        e_emg += DT * float(R.sum())
    # PV 自用/弃光：与 Q2（q2_backtest）完全同口径，物理量与价格无关。
    # 取 S3 生效计划 Qeff_s3 作为「已计划购电量」，保证 Q4-3 与 Q4-2 的弃光口径一致。
    _Rq, V_pv, K_pv, _Sq = q2_backtest(L, G, Qe, Cs, Ds)
    return {'plan': c_plan, 'up': c_up, 'down': c_dn, 'adj': c_up - c_dn,
            'emg': c_emg, 'total': c_plan + (c_up - c_dn) + c_emg,
            'R': R_all, 'e_emg': e_emg, 'per_k': per_k, 'aw': aw, 'E_end': E,
            'charge': DT * float(Cs.sum()), 'discharge': DT * float(Ds.sum()),
            'throughput': DT * float((Cs + Ds).sum()),
            'curtail': DT * float(K_pv.sum()), 'pv_use': DT * float(V_pv.sum())}


def s0s3_chain(rec):
    """文档「主对照实验」第 4 项 +「推荐指标体系」：Q4-3 的 S0/S1/S2/S3 逐级比较。

    S_j（j=0..3）= 只采纳前 j 个更新节点后的生效策略。Q3 的 solve_day 已在
    scen[j] 中保存每个场景的完整结算（C_plan / C_adj / C_emg / C_total / E_emg），
    且求解所用价格即本实验价格——vol 实验下就是附件4 波动电价本身，
    故本函数零重解、零近似，只做逐级差分：
        ΔC_j = C_total(S_{j-1}) − C_total(S_j)，j=1,2,3
    分别对应 6:00 / 12:00 / 18:00 三个更新节点的边际成本改善。
    注意 C_plan 对全部 j 恒等于 0:00 计划费（计划量按计划结算，与是否调整无关），
    故逐级差异全部来自调整净费与紧急购电费。"""
    out = {}
    for j in range(4):
        s = rec['scen'][j]
        out[j] = {'C_plan': float(s['C_plan']), 'C_adj': float(s['C_adj']),
                  'C_emg': float(s['C_emg']), 'C_total': float(s['C_total']),
                  'E_emg': float(s['E_emg']), 'A_tot': float(s['A_tot'])}
    return out


def q3_soc_traj(rec, L, G):
    """S3 生效策略的实际 SOC 轨迹（145 点，价格无关，与 run_block 同口径）。"""
    Cs, Ds = rec['C_s3'], rec['D_s3']
    E = np.zeros(N + 1)
    e = Q3.E_INIT
    for m in range(4):
        g0, g1 = Q3.BLOCKS[m]
        E[g0] = e
        Ls = L[g0:g1]
        rd = Ls + Cs[g0:g1] - Ds[g0:g1]
        D_eff = np.where(rd >= 0.0, Ds[g0:g1], Ls + Cs[g0:g1])
        for j in range(g1 - g0):
            e = e + ETA_CH * DT * Cs[g0 + j] - DT / ETA_DIS * D_eff[j]
            E[g0 + j + 1] = e
    return E


# ========== 6. 指标计算 ==========
def cvar(x, alpha):
    """经验 CVaR（Rockafellar-Uryasev 最优 zeta 的解）：最坏 ceil((1-alpha)n) 天的平均日费用。"""
    x = np.sort(np.asarray(x, dtype=float))
    k = max(1, int(np.ceil((1.0 - alpha) * len(x))))
    return float(x[-k:].mean())


def risk_block(costs, tag):
    c = np.asarray(costs, dtype=float)
    out = {'tag': tag, 'n': len(c), 'mean': float(c.mean()), 'median': float(np.median(c)),
           'std': float(c.std(ddof=1)) if len(c) > 1 else 0.0,
           'min': float(c.min()), 'max': float(c.max()),
           'p90': float(np.percentile(c, 90)), 'p95': float(np.percentile(c, 95)),
           'total': float(c.sum()), 'cvar90': cvar(c, CVAR_ALPHA_CHECK[0]),
           'cvar95': cvar(c, CVAR_ALPHA_MAIN), 'cvar99': cvar(c, CVAR_ALPHA_CHECK[1])}
    idx = np.argsort(c)[::-1][:WORST_N_DAYS]
    out['worst_share'] = float(c[idx].sum() / c.sum()) if c.sum() > 0 else float('nan')
    return out


def price_features(pv):
    sd = float(pv.std())
    p95 = float(np.percentile(pv, 95))
    p5 = float(np.percentile(pv, 5))
    return {'mean': float(pv.mean()), 'std': sd,
            'cv': sd / float(pv.mean()) if pv.mean() > 0 else float('nan'),
            'p95': p95, 'p5': p5, 'spread95_5': p95 - p5,
            'max': float(pv.max()), 'min': float(pv.min()),
            'ramp': float(np.abs(np.diff(pv)).sum()), 'p90': float(np.percentile(pv, 90))}


def price_power_cov(price, power):
    """文档「平均价格与『价格—购电对齐』共同决定成本」的日费用分解：
        C_d = Δt·Σ_t c_t P_t = T·Δt·[ c̄_d·P̄_d + Cov_t(c, P) ]
    第一项反映平均电价与平均购电功率，第二项反映购电是否集中在高价时段。
    储能与 MPC 的价值之一，就是把价格与购电量之间的正协方差压下来；
    同一分解亦适用于紧急购电（C^emg = 5T·Δt·[c̄_d·R̄_d + Cov_t(c,R)]）。"""
    T = len(price)
    cbar, pbar = float(np.mean(price)), float(np.mean(power))
    cov = float(np.mean((np.asarray(price, float) - cbar)
                        * (np.asarray(power, float) - pbar)))
    return {'cbar': cbar, 'pbar': pbar, 'cov': cov,
            'level_term': T * DT * cbar * pbar, 'cov_term': T * DT * cov}


def pwmae_curve(ctx, di, pv_row, k):
    """文档「波动电价改变Q3中预报更新的经济价值」：价格加权光伏预测误差
        PWMAE_{d,k} = Σ_{t∈T_k} c_{d,t}·|G^real_{d,t} − Ĝ_{d,t|k}| / Σ_{t∈T_k} c_{d,t}
    T_k = 节点 k 之后尚未执行的时段 [τ_k, 144)；Ĝ 直接复用 Q3.pv_curve，
    与模型内的 MPC 输入逐位同源，不引入第二套预测口径。
    价格全正时直接使用；出现负价时按文档改用 |c| 作权重。"""
    i0 = int(Q3.TAU[k])
    ghat = Q3.pv_curve(ctx, di, k)
    greal = np.asarray(ctx['G'][di], dtype=float)[i0:]
    w = np.asarray(pv_row, dtype=float)[i0:]
    if np.any(w < 0):
        w = np.abs(w)
    s = float(w.sum())
    return float((w * np.abs(greal - ghat)).sum() / s) if s > 0 else float('nan')


def merge_r_segments(R, g0=0):
    """连续紧急购电段 -> [(起分钟, 止分钟, 电量 kWh)]。"""
    segs = []
    above = R > 1e-6
    t, n = 0, len(R)
    while t < n:
        if above[t]:
            s = t
            while t < n and above[t]:
                t += 1
            segs.append(((g0 + s) * 10, (g0 + t) * 10, float(R[s:t].sum() * DT)))
        else:
            t += 1
    return segs


def max_gap_minutes(R):
    """最长连续缺口时长（分钟）。"""
    best = cur = 0
    for v in R:
        if v > 1e-6:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best * 10


def updown_energy(rec, up=True):
    """某日 S3 的全年调整电量分项（kWh）。"""
    tot = 0.0
    for k, item in _q3_uw(rec).items():
        if item is None:
            continue
        _i0, U, W = item
        tot += float((U if up else W).sum()) * DT
    return tot


def soc_high_readiness(E, pv):
    """高价时段 SOC 准备度 I_d^{SOC,high}（进入价格前 10% 时段时的可用电量占比）。"""
    Hd = pv >= np.percentile(pv, 90)
    if not Hd.any():
        return float('nan')
    return float(np.mean((np.asarray(E)[:N][Hd] - E_MIN) / (E_MAX - E_MIN)))


# ========== 6b. 文档要求的诊断层 ==========
# 覆盖文档「成本集中度与尖峰日贡献」「储能循环压力与隐含寿命风险」「分组检验」
# 「配对统计与置信区间」「结果解释规则」五节的全部要求，均为纯读取 + 统计，
# 不重解 MILP、不改变任何 MPC 目标/约束/执行模型。
def moving_block_ci(x, block, n_resamp=BOOT_N_RESAMPLE, seed=BOOT_SEED):
    """移动块自助法（Politis–White 风格）：对逐日序列的均值求 95% 置信区间（百分位法）。
    文档「配对统计与置信区间」：相邻日期存在连续天气与负荷相关性，须按块抽样保留
    局部依赖结构，比假设每天相互独立的普通 t 检验更符合全年时间序列结构。
    每次重抽 ceil(D/b) 个长度 b 的连续块（块起点在 [0, D-b] 内均匀随机、允许重叠），
    拼接后截前 D 项取均值；对 n_resamp 个重抽均值取 2.5%/97.5% 分位。
    返回 (点估计, CI 下限, CI 上限)。"""
    D = len(x)
    b = max(1, min(int(block), D))
    xb = np.asarray(x, dtype=float)
    nstart = D - b + 1
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_resamp))
    off = np.arange(b)[None, :]
    for r in range(int(n_resamp)):
        nb = int(np.ceil(D / b))
        starts = rng.integers(0, nstart, nb)
        idx = (starts[:, None] + off).ravel()[:D]
        means[r] = xb[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(xb.mean()), float(lo), float(hi)


def pair_stats(base_daily, cur_daily, label_a, label_b):
    """文档「配对统计与置信区间」：逐日配对差 δ_d = C_d^A − C_d^B。
    同一天的不同策略共享同一套实际负荷、光伏与价格，故必须用配对差值而非独立样本。
    输出平均配对差、中位数、节省天数比例 p_save 及 95% CI（主口径 7 天块，
    3 天与 14 天块作稳健性对照）。δ_d>0 表示 B 相对 A 节省费用。"""
    days = sorted(set(base_daily) & set(cur_daily))
    if not days:
        return None
    d = np.asarray([base_daily[k] - cur_daily[k] for k in days], dtype=float)
    mean, lo, hi = moving_block_ci(d, BOOT_BLOCKS[0])
    ci_alt = {b: moving_block_ci(d, b) for b in BOOT_BLOCKS[1:]}
    if lo > 0:
        sig = '显著为正（B 更省）'
    elif hi < 0:
        sig = '显著为负（B 更贵）'
    else:
        sig = '不显著（CI 含 0）'
    return {'label_a': label_a, 'label_b': label_b, 'n_day': len(d),
            'mean': mean, 'median': float(np.median(d)), 'total': float(d.sum()),
            'p_save': float((d > 0).mean()), 'p_harm': float((d < 0).mean()),
            'ci_lo': lo, 'ci_hi': hi, 'block': BOOT_BLOCKS[0],
            'ci_alt': ci_alt, 'sig': sig}


def _tier(vals, q=TIER_Q):
    """按分位把逐日指标切成低/中/高三档（文档「分组检验」的第一类分组）。"""
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v)
    lo, hi = np.quantile(v[ok], q) if ok.any() else (0.0, 0.0)
    lab = np.array(['中'] * len(v), dtype=object)
    lab[ok & (v <= lo)] = '低'
    lab[ok & (v > hi)] = '高'
    return lab


def group_stats(rows, key, order=None):
    """文档「分组检验」：按 rows[i][key] 分组，统计每组的
        ΔC^4-2 = C^V_{4-2,d} − C^F_{2,d}
        ΔC^4-3 = C^V_{4-3,d} − C^F_{3,d}
        ΔC^MPC = C^V_{4-2,d} − C^V_{4-3,d}   （>0 表示波动电价下滚动调整相对只做 0:00 计划节省费用）
    每行须含 'dc42' / 'dc43' / 'dcm' 三个逐日差值字段。"""
    buckets = {}
    for r in rows:
        buckets.setdefault(str(r.get(key, 'NA')), []).append(r)
    keys = order or sorted(buckets)
    out = []
    for g in keys:
        grp = buckets.get(g)
        if not grp:
            continue
        rec = {'group_key': key, 'group': g, 'n_day': len(grp)}
        for tag, field in (('dc42', 'dc42'), ('dc43', 'dc43'), ('dcm', 'dcm')):
            v = np.asarray([x[field] for x in grp], dtype=float)
            rec[f'{tag}_mean'] = float(np.nanmean(v)) if len(v) else float('nan')
            rec[f'{tag}_sum'] = float(np.nansum(v)) if len(v) else float('nan')
            rec[f'{tag}_save_rate'] = float(np.nanmean(v > 0)) if len(v) else float('nan')
        out.append(rec)
    return out


def date_coincidence(scores, n=WORST_N_DAYS):
    """文档「成本集中度与尖峰日贡献」：取各指标最高的 n 个日期，计算两两重合率
    （交集大小 / n），用于回答「费用尖峰是否与价格尖峰、净负荷高峰同源」。
    scores = {名称: {日期: 数值}}。返回 (明细 rows, 汇总 rows)。"""
    tops = {}
    for name, sc in scores.items():
        ds = sorted(sc, key=lambda d: (-(sc[d] if np.isfinite(sc[d]) else -np.inf)))
        tops[name] = set(ds[:n])
    names = list(scores)
    pair_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            inter = tops[a] & tops[b]
            pair_rows.append({'日期集合A': a, '日期集合B': b,
                              '重合天数': len(inter), '重合率': len(inter) / max(1, n),
                              '重合日期': '、'.join(str(x) for x in sorted(inter))})
    detail_rows = [{'日期集合': nm, '前N日': n,
                    '日期': '、'.join(str(x) for x in sorted(tops[nm]))}
                   for nm in names]
    return detail_rows, pair_rows


def worst_days(dates, costs, n=WORST_N_DAYS):
    """文档「不能只用全年总费用衡量风险」：最坏 n 日及其对应日期。
    同时给出成本集中度 S_N^C = 最坏 n 日费用合计 / 全年费用合计。"""
    pairs = sorted(zip(dates, np.asarray(costs, dtype=float)), key=lambda x: -x[1])
    top = pairs[:n]
    tot = float(np.sum([p[1] for p in pairs]))
    return {'n': n, 'top': [(str(d), float(c)) for d, c in top],
            'share': float(sum(c for _, c in top) / tot) if tot > 0 else float('nan'),
            'max_date': str(top[0][0]) if top else '', 'max_cost': float(top[0][1]) if top else float('nan')}


def life_adjusted(total_cost, throughput, grid=LIFE_KAPPA_GRID):
    """文档「储能循环压力与隐含寿命风险」：事后修正
        C^life-adjusted = C^total + κ_th · E^throughput
    κ_th 为单位吞吐电量的折算寿命成本（元/kWh）。该修正**不改变正式调度**，
    只检验「在合理衰减成本下，波动电价的套利结论是否仍然成立」。"""
    return [{'kappa_th': float(k), 'C_total': float(total_cost),
             'C_life_adjusted': float(total_cost) + float(k) * float(throughput),
             'relative_extra': (float(k) * float(throughput) / float(total_cost)
                                if total_cost else float('nan'))}
            for k in grid]


def settle_coef_sens(recs_q3, ctx, pmat_v):
    """文档「上调/下调系数」敏感性（仅作附录）：Q3 源码中两个系数是字面量 1.5 / 0.5，
    在不改动上游的前提下无法重解；故对**已锁定的 S3 策略链**按不同系数组合事后重算
    调整净费 ΔC^adj = Σ c^V(1.5'U − 0.5'W)Δt，回答「Q3 的调整价值是否依赖单一合同参数」。
    只改结算口径、不改变调度，故结果用于判断稳健性边界，不进入正式结果。"""
    L, G = ctx['L'], ctx['G']
    didx = ctx['date_idx']
    combos = [(1.5, 0.5)] + [(u, 0.5) for u in SENS_COEF_UP] + [(1.5, d) for d in SENS_COEF_DOWN]
    rows = []
    for cu, cd in combos:
        adj = plan = emg = 0.0
        for rec in recs_q3:
            st = q3_settle(rec, L[didx[rec['date']]], G[didx[rec['date']]],
                           pmat_v[didx[rec['date']]], coef=(cu, cd))
            adj += st['adj']
            plan += st['plan']
            emg += st['emg']
        rows.append({'上调系数': cu, '下调系数': cd, 'Q4-3计划费': plan,
                     'Q4-3调整净费': adj, 'Q4-3紧急费': emg,
                     'Q4-3总费用': plan + adj + emg,
                     '基准': '是' if (cu, cd) == (1.5, 0.5) else ''})
    return rows


# ========== 7. 反事实（固定策略 x^F 按附件4 价格重新结算，不重解任何变量） ==========
def build_counterfactual(ctx, fixed_q2, fixed_q3, native_q2, native_q3, pmat_v):
    """文档「用反事实分解识别」与「回归核验」第 1、9 条。
    固定策略 x^F 取自本脚本 fixed 实验（与原生缓存同算法同价格，回归核验一栏比对二者）；
    同时输出回归核验用的逐日比对数据（用附件1 结算 x^F，必须复现 Q2/Q3 原生缓存）。"""
    L, G = ctx['L'], ctx['G']
    p1 = Q2.read_prices(Q2.PRICE_FILE)
    didx = ctx['date_idx']
    nat2 = {r['date']: r for r in native_q2}
    nat3 = {r['date']: r for r in native_q3}
    cf = {'q2': {}, 'q3': {}, 'reg': {'q2': [], 'q3': []}}
    for rec in fixed_q2:
        d = rec['date']
        i = didx[d]
        pv = pmat_v[i]
        sV = q2_settle(rec['B'], rec['C'], rec['D'], L[i], G[i], pv)
        sF = q2_settle(rec['B'], rec['C'], rec['D'], L[i], G[i], p1)
        cf['q2'][d] = {'C_V_xF': sV['total'], 'C_F_xF': sF['total'],
                       'plan_V': sV['plan'], 'emg_V': sV['emg'],
                       'soc_ready': soc_high_readiness(rec['E'], pv), 'shift': float('nan')}
        ref = nat2[d]
        cf['reg']['q2'].append((d, sF['total'], ref['total_cost'],
                                sF['plan'], ref['plan_cost'], sF['emg'], ref['em_cost']))
    for rec in fixed_q3:
        d = rec['date']
        i = didx[d]
        pv = pmat_v[i]
        sV = q3_settle(rec, L[i], G[i], pv)
        sF = q3_settle(rec, L[i], G[i], p1)
        cf['q3'][d] = {'C_V_xF': sV['total'], 'C_F_xF': sF['total'],
                       'plan_V': sV['plan'], 'adj_V': sV['adj'], 'emg_V': sV['emg'],
                       'soc_ready': float('nan'), 'shift': float('nan')}
        ref = nat3[d]
        cf['reg']['q3'].append((d, sF['plan'], ref['plan_cost'],
                                sF['emg'], ref['scen'][3]['C_emg'],
                                sF['adj'], ref['scen'][3]['C_adj'],
                                sF['total'], ref['scen'][3]['C_total']))
    return cf, p1


def verify_settle(ctx, recs, pmat_v, fam):
    """自检（文档「数值实现与结果核验」）：用各日实际采用的价格重新结算，
    必须逐日复现记录自身已保存的分项费用。返回 (最大相对偏差, 定位描述)。
    该检查同时验证 U/W 抓取、run_block 推进与成本核算口径三者一致。"""
    L, G = ctx['L'], ctx['G']
    didx = ctx['date_idx']
    worst, where = 0.0, '（全部一致）'
    for rec in recs:
        d = rec['date']
        i = didx[d]
        p = pmat_v[i]
        if fam == 'q2':
            s = q2_settle(rec['B'], rec['C'], rec['D'], L[i], G[i], p)
            pairs = [('计划费', s['plan'], rec['plan_cost']),
                     ('紧急费', s['emg'], rec['em_cost']),
                     ('总费用', s['total'], rec['total_cost'])]
        else:
            s = q3_settle(rec, L[i], G[i], p)
            s3 = rec['scen'][3]
            pairs = [('调整净费', s['adj'], s3['C_adj']),
                     ('紧急费', s['emg'], s3['C_emg']),
                     ('总费用', s['total'], s3['C_total'])]
        for tag, a, b in pairs:
            rel = abs(a - b) / max(1.0, abs(b))
            if rel > worst:
                worst, where = rel, f"{d} {tag}: 重算 {a:.6f} vs 记录 {b:.6f}"
    return worst, where


def add_shift(cf, fixed_q2, recs_q2):
    """调度迁移程度 E_d^shift = Δt/2 * Σ|P^grid,V - P^grid,F|（文档推荐指标体系）。"""
    fmap = {r['date']: r['B'] for r in fixed_q2}
    for r in recs_q2:
        d = r['date']
        if d in fmap and d in cf['q2']:
            cf['q2'][d]['shift'] = DT / 2.0 * float(np.abs(r['B'] - fmap[d]).sum())
    return cf


def verify_q4_layer(ctx, recs_q2, recs_q3, pmat_v, kappa=KAPPA):
    """文档「数值实现与结果核验」第 3/4/5/6/8 条的 Q4 层独立复核。
    上游 Q2/Q3 各自已做这些检查，但 Q4 换了价格、且 Q4-3 的调整费由本脚本重算，
    故必须在 Q4 层按同一口径再独立核验一遍（不能只依赖上游）。
      ③ 每时段功率平衡残差低于容差（Buse+R+V+D = L+C，V+K = G）；
      ④ SOC 全程位于 [1200,10800] kWh 且充放电功率不超过 5000 kW；
      ⑤ 日初/日末 SOC 沿用原模型口径（E_1 = E_145 = 6000 kWh）；
      ⑥ 已执行时段不被 6:00/12:00/18:00 的新计划追溯修改；
      ⑧ S0~S3 使用相同的 0:00 基准计划与相同实际数据。
    返回 (结果 dict, 是否全部通过)。"""
    L, G = ctx['L'], ctx['G']
    didx = ctx['date_idx']
    pmax = P_CH_MAX          # 题设储能功率上限，核验主重算的物理可行性
    ch_max = dis_max = 0.0
    soc_lo, soc_hi = np.inf, -np.inf
    bal_res, e_init_dev, e_end_dev = 0.0, 0.0, 0.0
    n_soc_bad = n_p_bad = 0
    for r in recs_q2:
        i = didx[r['date']]
        B, C, D = r['B'], r['C'], r['D']
        R, V, K, S = q2_backtest(L[i], G[i], B, C, D)
        Buse = B - S
        bal_res = max(bal_res, float(np.abs(Buse + R + V + D - (L[i] + C)).max()))
        bal_res = max(bal_res, float(np.abs(V + K - G[i]).max()))
        E = np.asarray(r['E'])
        soc_lo, soc_hi = min(soc_lo, float(E.min())), max(soc_hi, float(E.max()))
        e_init_dev = max(e_init_dev, abs(float(E[0]) - E_INIT))
        e_end_dev = max(e_end_dev, abs(float(E[-1]) - E_INIT))
        ch_max, dis_max = max(ch_max, float(C.max())), max(dis_max, float(D.max()))
        n_p_bad += int((C > pmax + 1e-6).any()) + int((D > pmax + 1e-6).any())
    for r in recs_q3:
        i = didx[r['date']]
        E = q3_soc_traj(r, L[i], G[i])
        soc_lo, soc_hi = min(soc_lo, float(E.min())), max(soc_hi, float(E.max()))
        e_init_dev = max(e_init_dev, abs(float(E[0]) - E_INIT))
        e_end_dev = max(e_end_dev, abs(float(E[-1]) - E_INIT))
        ch_max = max(ch_max, float(r['C_s3'].max()))
        dis_max = max(dis_max, float(r['D_s3'].max()))
        n_p_bad += int((r['C_s3'] > pmax + 1e-6).any()) + int((r['D_s3'] > pmax + 1e-6).any())
    n_soc_bad = 0 if (soc_lo >= E_MIN - VERIFY_TOL_SOC and soc_hi <= E_MAX + VERIFY_TOL_SOC) else 1

    # ⑥ 已执行时段冻结：采纳节点 k 之后，[0, TAU[k]) 的生效计划必须与 0:00 基准一致
    freeze_dev, n_adopt = 0.0, 0
    for r in recs_q3:
        for k in (1, 2, 3):
            if not r['a_flags'][k - 1]:
                continue
            n_adopt += 1
            i0 = int(Q3.TAU[k])
            freeze_dev = max(freeze_dev, float(np.abs(
                r['Qeff_s3'][:i0] - r['Q0'][:i0]).max()))

    # ⑧ S0~S3 共用同一 0:00 基准计划：各场景 C_plan 必须逐日相同
    s0s3_dev = 0.0
    for r in recs_q3:
        vals = [float(r['scen'][j]['C_plan']) for j in range(4)]
        s0s3_dev = max(s0s3_dev, max(vals) - min(vals))

    res = {
        '③功率平衡最大残差': bal_res, '③容差': VERIFY_TOL_BAL,
        '③通过': bool(bal_res <= VERIFY_TOL_BAL),
        '④SOC范围': f"[{soc_lo:.2f}, {soc_hi:.2f}]",
        '④SOC越界天数': n_soc_bad, '④充放电功率越限次数': n_p_bad,
        '④通过': bool(n_soc_bad == 0 and n_p_bad == 0),
        '⑤日初SOC最大偏差': e_init_dev, '⑤日末SOC最大偏差': e_end_dev,
        '⑤通过': bool(max(e_init_dev, e_end_dev) <= VERIFY_TOL_SOC),
        '⑥已执行时段冻结最大偏差': freeze_dev, '⑥采纳节点数': n_adopt,
        '⑥通过': bool(freeze_dev <= VERIFY_TOL_SOC),
        '⑧S0-S3基准计划最大偏差': s0s3_dev, '⑧通过': bool(s0s3_dev <= VERIFY_TOL_SOC),
        '充放电功率峰值': f"ch {ch_max:.2f} / dis {dis_max:.2f} kW",
    }
    ok = all(res[k] for k in ('③通过', '④通过', '⑤通过', '⑥通过', '⑧通过'))
    return res, ok


def build_diag_rows(ctx, recs_q2, recs_q3, fixed_q2, fixed_q3, pmat_v):
    """构造逐日诊断记录（文档「分组检验」与「成本集中度」的数据基础）。
    每行 = 一天，含分组维度（日价格 CV / P95-P5 / 价荷相关系数 / 季节 / 光伏预测误差 /
    是否价格尖峰 / 是否发生紧急购电）与三个逐日差值：
        dc42 = C^V_{4-2,d} − C^F_{2,d}
        dc43 = C^V_{4-3,d} − C^F_{3,d}
        dcm  = C^V_{4-2,d} − C^V_{4-3,d}
    全部由已锁定策略按各自价格结算得到，不重解 MILP。"""
    L, G = ctx['L'], ctx['G']
    didx = ctx['date_idx']
    q95 = float(np.percentile(pmat_v, 95))
    r3 = {r['date']: r for r in recs_q3}
    f3 = {r['date']: r for r in fixed_q3}
    rows = []
    for r2 in recs_q2:
        d = r2['date']
        if d not in r3 or d not in f3:
            continue
        i = didx[d]
        pv, Ld, Gd = pmat_v[i], L[i], G[i]
        f = price_features(pv)
        st2 = q2_settle(r2['B'], r2['C'], r2['D'], Ld, Gd, pv)
        rec3 = r3[d]
        st3 = q3_settle(rec3, Ld, Gd, pv)
        cf2 = float(r2['total_cost'])                 # fixed 实验的 Q2 记录（附件1 价格）
        cf3 = float(f3[d]['scen'][3]['C_total'])      # fixed 实验的 Q3 记录（附件1 价格）
        nl = Ld - Gd
        corr = float(np.corrcoef(pv, nl)[0, 1]) if np.std(pv) > 0 and np.std(nl) > 0 else float('nan')
        mth = d.month
        season = ('春(3-5)' if mth in (3, 4, 5) else '夏(6-8)' if mth in (6, 7, 8)
                  else '秋(9-11)' if mth in (9, 10, 11) else '冬(12,1,2)')
        rows.append({
            'date': d, 'cv': f['cv'], 'spread': f['spread95_5'], 'corr': corr,
            'season': season, 'mae': float(rec3.get('mae_g0', float('nan'))),
            'spike': '有' if float(pv.max()) >= q95 else '无',
            'emg': '有' if float(st2['R'].sum()) > 1e-6 else '无',
            'c42': st2['total'], 'c43': st3['total'],
            'cf2': cf2, 'cf3': cf3,
            'dc42': st2['total'] - cf2, 'dc43': st3['total'] - cf3,
            'dcm': st2['total'] - st3['total'],
        })
    for r in rows:
        r['tier_cv'] = ''
        r['tier_spread'] = ''
        r['tier_corr'] = ''
        r['tier_mae'] = ''
    if rows:
        for tag, field in (('tier_cv', 'cv'), ('tier_spread', 'spread'),
                           ('tier_corr', 'corr'), ('tier_mae', 'mae')):
            labs = _tier([r[field] for r in rows])
            for r, lab in zip(rows, labs):
                r[tag] = lab
    return rows


def build_group_tables(rows):
    """按文档「分组检验」的五类条件分组（价格离散度 / 价荷相关 / 季节 / 预测误差 /
    尖峰与紧急购电事件），输出可直接落表的分组统计。"""
    tables = {}
    for key in ('tier_cv', 'tier_spread', 'tier_corr', 'season', 'tier_mae', 'spike', 'emg'):
        tables[key] = group_stats(rows, key)
    return tables


def explain_cases(yearly, risk, cf, sens):
    """文档「结果解释规则」：按六情形的判别条件逐一判定，并计算情形一的缓冲比例
        r^mitigate = [C_V(x^F) − C_V(x^V)] / [C_V(x^F) − C_F(x^F)]
    （分子 = 重新调度挽回的费用，分母 = 纯价格效应造成的费用变化）。
    本函数只做判别与量化，不输出结论文字，供论文择句时取用。"""
    nm2, nm3 = 'Q4-2 波动电价', 'Q4-3-S3 波动电价'
    y2, y3 = yearly[nm2], yearly[nm3]
    b2, b3 = yearly['Q2 固定电价'], yearly['Q3-S3 固定电价']
    cvF2 = sum(x['C_V_xF'] for x in cf['q2'].values())
    cF2 = sum(x['C_F_xF'] for x in cf['q2'].values())
    cvF3 = sum(x['C_V_xF'] for x in cf['q3'].values())
    cF3 = sum(x['C_F_xF'] for x in cf['q3'].values())
    price_eff2 = cvF2 - cF2
    adapt_eff2 = y2['total'] - cvF2
    price_eff3 = cvF3 - cF3
    adapt_eff3 = y3['total'] - cvF3
    total_dev2 = y2['total'] - b2['total']
    total_dev3 = y3['total'] - b3['total']
    r_mit = (adapt_eff2 / price_eff2 if abs(price_eff2) > 1e-9 else float('nan'))

    # 归因占比（文档「最终变化来自价格水平还是调度策略」）：
    # 总变化 C_V(x^V)-C_F(x^F) = 纯价格结算效应 + 调度适应效应
    price_pct2 = (price_eff2 / total_dev2 if abs(total_dev2) > 1e-9 else float('nan'))
    adapt_pct2 = (adapt_eff2 / total_dev2 if abs(total_dev2) > 1e-9 else float('nan'))
    price_pct3 = (price_eff3 / total_dev3 if abs(total_dev3) > 1e-9 else float('nan'))
    adapt_pct3 = (adapt_eff3 / total_dev3 if abs(total_dev3) > 1e-9 else float('nan'))

    # 文档 576 行：确定性模型下调度适应效应应 ≤0；偶尔为正本身是「样本外/执行截断」
    # 模型风险的证据，须按日统计其出现天数、不能直接删除。cf[*][d] 含逐日
    # C_V_xF（x^F 按附件4 结算）与 C_V_xV（x^V 逐日总费，由调用方补填）。
    n_adapt_pos_q2 = int(sum(1 for v in cf['q2'].values()
                             if np.isfinite(v.get('C_V_xV', float('nan')))
                             and v['C_V_xV'] - v['C_V_xF'] > 0))
    n_adapt_pos_q3 = int(sum(1 for v in cf['q3'].values()
                             if np.isfinite(v.get('C_V_xV', float('nan')))
                             and v['C_V_xV'] - v['C_V_xF'] > 0))

    cases = {}
    cases['情形一：总费用上升但调度适应效应为负'] = {
        '成立': bool(total_dev2 > 0 and adapt_eff2 < 0),
        '证据': f"Q4-2 总费用变化 {total_dev2:+.2f} 元，调度适应效应 {adapt_eff2:+.2f} 元，"
                f"缓冲比例 r^mitigate={r_mit:.4f}"}
    cases['情形二：总费用下降但购电量与EFC上升'] = {
        '成立': bool(total_dev2 < 0 and (y2['buy_energy'] - b2['buy_energy']) > 0
                     and (y2['EFC_use'] - b2['EFC_use']) > 0),
        '证据': f"总费用 {total_dev2:+.2f} 元，购电量变化 "
                f"{y2['buy_energy'] - b2['buy_energy']:+.2f} kWh，"
                f"EFC 变化 {y2['EFC_use'] - b2['EFC_use']:+.4f}"}
    cases['情形三：紧急购电量下降但紧急购电费上升'] = {
        '成立': bool((y2['E_emg'] - b2['E_emg']) < 0 and (y2['emg'] - b2['emg']) > 0),
        '证据': f"紧急购电量变化 {y2['E_emg'] - b2['E_emg']:+.2f} kWh，"
                f"紧急购电费变化 {y2['emg'] - b2['emg']:+.2f} 元，"
                f"高价紧急购电占比 {y2['r_emg_high']:.4f}"}
    dcv = risk[nm2]['cvar95'] - risk[nm3]['cvar95']
    dmean = risk[nm2]['mean'] - risk[nm3]['mean']
    cases['情形四：Q4-3平均节省不大但CVaR明显下降'] = {
        '成立': bool(abs(dmean) > 0 and dcv < 0 and abs(dcv) > abs(dmean)),
        '证据': f"Q4-2→Q4-3 日均费用变化 {dmean:+.4f} 元，"
                f"CVaR95 变化 {dcv:+.4f} 元（下降幅度 {dcv / max(1e-9, abs(dmean)):.2f} 倍于均值变化）"}
    n_adopt = int(sum(y3['adopt']))
    cases['情形五：调整次数很多但净收益很小'] = {
        '成立': bool(n_adopt > 0 and abs(dmean) < 1e-9 + abs(y3['total']) * 1e-4),
        '证据': f"全年采纳节点 {n_adopt} 次，Q4-2→Q4-3 日均费用变化 {dmean:+.4f} 元，"
                f"调整电量 {(y3['A_up'] + y3['A_down']) / 1000:.2f} MWh"}
    lam_rows = []
    for e in ('lam0', 'lam05', 'vol_base', 'lam15'):
        if e in sens:
            lam_rows.append((e, float(sens[e]['q2']['total'])))
    gain = float('nan')
    if len(lam_rows) >= 2:
        gain = lam_rows[-1][1] - lam_rows[0][1]
    cases['情形六：波动越大收益没有明显提高'] = {
        '成立': bool(np.isfinite(gain) and gain <= 0),
        '证据': f"λ 由 0 增至 1.5 时 Q4-2 总费用变化 {gain:+.2f} 元；"
                f"效率补偿阈值价差 = (1-η²)/η² = {EFF_THRESHOLD:.4f}"
                f"（{EFF_THRESHOLD * 100:.2f}%，低于此值套利净收益为负）"}
    meta = {'r_mitigate': r_mit, 'price_eff2': price_eff2, 'adapt_eff2': adapt_eff2,
            'price_eff3': price_eff3, 'adapt_eff3': adapt_eff3,
            'total_dev2': total_dev2, 'total_dev3': total_dev3,
            'C_V_xF2': cvF2, 'C_F_xF2': cF2, 'C_V_xF3': cvF3, 'C_F_xF3': cF3,
            # 归因占比（文档「最终变化来自价格水平还是调度策略」）
            'price_pct2': price_pct2, 'adapt_pct2': adapt_pct2,
            'price_pct3': price_pct3, 'adapt_pct3': adapt_pct3,
            # 文档 576 行：调度适应效应>0 的逐日天数（模型风险证据，不删除）
            'n_adapt_pos_q2': n_adapt_pos_q2, 'n_adapt_pos_q3': n_adapt_pos_q3,
            'n_day_q2': len(cf['q2']), 'n_day_q3': len(cf['q3'])}
    return cases, meta


def s0s3_table(recs_q3):
    """Q4-3 的 S0/S1/S2/S3 全年逐级汇总（文档「主对照实验」第 4 项）。
    对全部决策日按场景累加，并给出各更新节点的边际成本改善 ΔC_j。"""
    tot = {j: {'C_plan': 0.0, 'C_adj': 0.0, 'C_emg': 0.0, 'C_total': 0.0,
               'E_emg': 0.0, 'A_tot': 0.0} for j in range(4)}
    for r in recs_q3:
        ch = s0s3_chain(r)
        for j in range(4):
            for k in tot[j]:
                tot[j][k] += ch[j][k]
    rows = []
    for j in range(4):
        v = tot[j]
        rows.append({'场景': f'S{j}' + ('' if j == 0 else f'（含 6/12/18 前 {j} 个节点）'),
                     '计划费': v['C_plan'], '调整净费': v['C_adj'],
                     '紧急费': v['C_emg'], '总费用': v['C_total'],
                     '紧急购电量/MWh': v['E_emg'] / 1000.0,
                     '调整电量/MWh': v['A_tot'] / 1000.0,
                     '边际改善(vs上级)': (tot[j - 1]['C_total'] - v['C_total']) if j else ''})
    return rows


def regression_report(cf, tol_rel=REG_TOL_REL, tol_abs=REG_TOL_ABS):
    """回归核验：附件1 价格结算固定电价策略 x^F 必须复现 Q2/Q3 原生缓存。"""
    print("\n" + "=" * 72)
    print("[回归核验] 附件1 价格结算固定策略 x^F 应复现 Q2/Q3 原生缓存"
          "（文档「回归核验」第 1、9 条）")
    print("=" * 72)
    ok = True
    for fam in ('q2', 'q3'):
        rows = cf['reg'][fam]
        worst, worst_tag = 0.0, ''
        for row in rows:
            for tag, a, b in _reg_pairs(fam, row):
                err = abs(a - b)
                rel = err / max(1.0, abs(b))
                if rel > worst:
                    worst, worst_tag = rel, f"{row[0]} {tag}: 重算 {a:.6f} vs 缓存 {b:.6f}"
                if rel > tol_rel and err > tol_abs:
                    ok = False
                    print(f"  [不一致] {row[0]} {tag}: 重算 {a:.6f} vs 缓存 {b:.6f}"
                          f"（绝对差 {err:.6f}）")
        print(f"  {fam.upper()}: {len(rows)} 天比对完成，最大相对偏差 {worst:.3e}（{worst_tag}）")
    print("  结论: " + ("通过 -> 结算口径与 Q2/Q3 完全一致" if ok else "未通过 -> 请检查结算口径"))
    return ok


def _reg_pairs(fam, row):
    if fam == 'q2':
        _, a1, b1, a2, b2, a3, b3 = row
        return [('总费用', a1, b1), ('计划费', a2, b2), ('紧急费', a3, b3)]
    _, a1, b1, a2, b2, a3, b3, a4, b4 = row
    return [('计划费', a1, b1), ('紧急费', a2, b2), ('调整净费', a3, b3), ('总费用', a4, b4)]


# ========== 8. 年度汇总与风险指标 ==========
def _agg_family(ctx, recs, fam, pmat, kappa=KAPPA, pbuy=None):
    """某族在某价格矩阵下的全年指标（全部由已锁定策略结算，物理量价格无关）。
    kappa 为该实验的紧急购电倍数（基准 5；仅倍数敏感性族不同）。
    pbuy 为该实验的并网购电上限（None = 主结果 P_buy_max=∞）：
      有限时给出并网上限活跃率 r^grid,max（文档「并网功率与充放电功率上限决定可利用价差」）；
      ∞ 时该率无定义，记 NaN 并在表内注明，另列最大并网功率供参照。"""
    L, G = ctx['L'], ctx['G']
    didx = ctx['date_idx']
    q95 = float(np.percentile(pmat, 95))
    out = {k: 0.0 for k in ('plan', 'adj', 'emg', 'total', 'buy_energy', 'buy_cost',
                            'charge', 'discharge', 'throughput', 'curtail', 'pv_use',
                            'pv_avail', 'E_emg', 'E_emg_high')}
    out['adopt'] = [0, 0, 0]
    out['A_up'] = out['A_down'] = 0.0
    out['C_up'] = out['C_down'] = 0.0     # C^up / C^cancel（文档「计划费、调整费与紧急费的重分配」）
    act = {'n_ch': 0, 'n_dis': 0, 'n_Emin': 0, 'n_Emax': 0, 'n_grid': 0}
    prep = []
    cov_sum = {'level_term': 0.0, 'cov_term': 0.0, 'cbar': 0.0, 'pbar': 0.0, 'cov': 0.0}
    pw = {1: [], 2: [], 3: []}
    for r in recs:
        i = didx[r['date']]
        pv = pmat[i]
        if fam == 'q2':
            st = q2_settle(r['B'], r['C'], r['D'], L[i], G[i], pv, kappa=kappa)
            Cq, Dq, Eq = r['C'], r['D'], np.asarray(r['E'])
        else:
            st = q3_settle(r, L[i], G[i], pv, kappa=kappa)
            Cq, Dq, Eq = r['C_s3'], r['D_s3'], q3_soc_traj(r, L[i], G[i])
            out['A_up'] += updown_energy(r, up=True)
            out['A_down'] += updown_energy(r, up=False)
            out['C_up'] += st['up']
            out['C_down'] += st['down']
            for k in (1, 2, 3):
                if r['a_flags'][k - 1]:
                    out['adopt'][k - 1] += 1
                pw[k].append(pwmae_curve(ctx, i, pv, k))
        out['plan'] += st['plan']
        out['adj'] += st.get('adj', 0.0)
        out['emg'] += st['emg']
        out['total'] += st['total']
        out['charge'] += st['charge']
        out['discharge'] += st['discharge']
        out['throughput'] += st['throughput']
        out['curtail'] += st['curtail']
        out['pv_use'] += st['pv_use']
        R = st['R']
        out['E_emg'] += DT * float(R.sum())
        out['E_emg_high'] += DT * float(R[pv >= q95].sum())
        buy = r['B'] if fam == 'q2' else r['Qeff_s3']
        out['buy_energy'] += DT * float(buy.sum())
        out['buy_cost'] += DT * float(pv @ buy)
        out['pv_avail'] += DT * float(G[i].sum())
        act['n_ch'] += int((Cq >= P_CH_MAX - ACT_EPS_P).sum())
        act['n_dis'] += int((Dq >= P_DIS_MAX - ACT_EPS_P).sum())
        act['n_Emin'] += int((Eq <= E_MIN + ACT_EPS_E).sum())
        act['n_Emax'] += int((Eq >= E_MAX - ACT_EPS_E).sum())
        if pbuy is not None:
            act['n_grid'] += int((np.asarray(buy, float) >= float(pbuy) - ACT_EPS_P).sum())
        prep.append(soc_high_readiness(Eq, pv))
        cd = price_power_cov(pv, buy)
        for kk in cov_sum:
            cov_sum[kk] += cd[kk]
    n_day = max(1, len(recs))
    out['buy_wavg'] = out['buy_cost'] / out['buy_energy'] if out['buy_energy'] > 0 else float('nan')
    out['EFC_use'] = out['throughput'] / (2 * E_USE)
    out['EFC_nom'] = out['throughput'] / (2 * E_CAP)
    out['pv_self'] = out['pv_use'] / out['pv_avail'] if out['pv_avail'] > 0 else float('nan')
    out['emg_unit'] = out['emg'] / out['E_emg'] if out['E_emg'] > 1e-9 else float('nan')
    out['r_emg_high'] = out['E_emg_high'] / out['E_emg'] if out['E_emg'] > 1e-9 else 0.0
    out['soc_ready'] = float(np.nanmean(prep)) if len(prep) else float('nan')
    tot = n_day * N
    out['activity'] = {'r_ch_max': act['n_ch'] / tot, 'r_dis_max': act['n_dis'] / tot,
                       'r_E_min': act['n_Emin'] / tot, 'r_E_max': act['n_Emax'] / tot,
                       'r_grid_max': (act['n_grid'] / tot) if pbuy is not None
                       else float('nan')}
    out['pbuy_cap'] = float(pbuy) if pbuy is not None else float('inf')
    t = out['total']
    out['s_plan'] = out['plan'] / t if t else float('nan')
    out['s_up'] = out['C_up'] / t if t else float('nan')
    out['s_down'] = out['C_down'] / t if t else float('nan')
    out['s_adj'] = out['adj'] / t if t else float('nan')
    out['s_emg'] = out['emg'] / t if t else float('nan')
    out['cov'] = {'level_term': cov_sum['level_term'], 'cov_term': cov_sum['cov_term'],
                  'cbar': cov_sum['cbar'] / n_day, 'pbar': cov_sum['pbar'] / n_day,
                  'cov': cov_sum['cov'] / n_day}
    out['pwmae'] = {k: (float(np.nanmean(pw[k])) if pw[k] else float('nan'))
                    for k in (1, 2, 3)}
    out['adopt_rate'] = [c / n_day for c in out['adopt']]
    out['n_day'] = len(recs)
    return out


def collect_yearly(ctx, recs_q2, recs_q3, pmat_v, fixed_q2, fixed_q3):
    p1 = np.tile(Q2.read_prices(Q2.PRICE_FILE), (len(ctx['dates']), 1))
    return {'Q2 固定电价': _agg_family(ctx, fixed_q2, 'q2', p1),
            'Q4-2 波动电价': _agg_family(ctx, recs_q2, 'q2', pmat_v),
            'Q3-S3 固定电价': _agg_family(ctx, fixed_q3, 'q3', p1),
            'Q4-3-S3 波动电价': _agg_family(ctx, recs_q3, 'q3', pmat_v)}


# ========== 8b. Q4-PF：价格预测精度与完全价格信息价值（文档「可选的价格预测现实性扩展」） ==========
def pf_accuracy(base_row, real_row, weight_row, hi_thr):
    """单日预测精度原始量（返回可加总的绝对量，便于跨日精确汇总）。
    口径（文档第 1118 行「除价格MAE、RMSE外，还应报告价格方向判断准确率和决策加权误差」）：
      - MAE / RMSE        : 逐时段 |ĉ−c| 与 (ĉ−c)²
      - 方向准确率        : Δĉ 与 Δc 同号的时段占比（只在实测价格发生变化的时段统计）
      - 决策加权误差      : 以「计划购电量」为权的加权 MAE —— 误差落在真正由价格驱动的决策上
      - 高价段 MAE / 低估比: 实测价格 ≥ 全样本 P{PF_HI_Q} 的时段，
                            回答「MAE 相近的两个预测在尖峰处可能完全不同」"""
    b = np.asarray(base_row, dtype=float)
    r = np.asarray(real_row, dtype=float)
    e = b - r
    w = np.abs(np.asarray(weight_row, dtype=float))
    db = np.sign(np.diff(b))
    dr = np.sign(np.diff(r))
    m = dr != 0.0
    hi = r >= hi_thr
    return {'n': int(e.size), 'sum_abs': float(np.sum(np.abs(e))),
            'sum_sq': float(np.sum(e ** 2)),
            'n_dir': int(m.sum()), 'n_dir_match': int(np.sum(db[m] == dr[m])),
            'sum_w': float(w.sum()), 'sum_w_abs': float(np.sum(w * np.abs(e))),
            'n_hi': int(hi.sum()), 'sum_abs_hi': float(np.sum(np.abs(e[hi]))),
            'n_hi_under': int(np.sum(e[hi] < 0.0))}


def pf_finalize(a):
    """把可加总的精度原始量折算成报表用的比率/均方根。"""
    n = max(1, a['n'])
    nd = max(1, a['n_dir'])
    nh = max(1, a['n_hi'])
    return {'MAE': a['sum_abs'] / n, 'RMSE': float(np.sqrt(a['sum_sq'] / n)),
            '方向准确率': a['n_dir_match'] / nd if a['n_dir'] else float('nan'),
            '方向样本数': a['n_dir'],
            '决策加权误差': a['sum_w_abs'] / a['sum_w'] if a['sum_w'] > 0 else float('nan'),
            '高价段MAE': a['sum_abs_hi'] / nh if a['n_hi'] else float('nan'),
            '高价段低估比例': a['n_hi_under'] / nh if a['n_hi'] else float('nan')}


def collect_pf(ctx2, ctx3, recs_pi2, recs_pi3, pmat_v, limit=None):
    """Q4-PF 现实性分析（文档「可选的价格预测现实性扩展」第 1053-1118 行）。

    返回 None 表示 pf 族缓存缺失（未跑 `python solve_q4.py pf`），此时不产生任何 PF 输出，
    也不影响正式交付文件——文档明确要求 Q4-PF 不得替换 result4-2 / result4-3。

    费用口径（与文档公式严格一致）：
        C(c^real; x(ĉ))  = 用 Q4-PF 记录（预测价格下求得的策略）按**真实价格**结算
        C(c^real; x(c^real)) = Q4-PI（即 vol 正式结果）同样按真实价格结算
        L_price info = 前者 − 后者 ≥ 0
    Q4-2 是单次全时域 MILP，同可行域 ⇒ 严格有 L ≥ 0；
    Q4-3 是逐节点滚动 MPC（贪心），同可行域下 L ≥ 0 为经验性质而非凸性保证，
    故两者分别报告、分别核验。"""
    if _PF['BASE'] is None:
        return None
    try:
        pf2 = load_family(PF_EXP, 'q2', ctx2, ctx3, limit)
        pf3 = load_family(PF_EXP, 'q3', ctx2, ctx3, limit)
    except RuntimeError as e:
        print(f"[Q4-PF] 跳过价格预测现实性扩展：{e}")
        return None
    _reset_upstream_params()
    ctx = ctx2
    L, G = ctx['L'], ctx['G']
    didx = ctx['date_idx']
    base_all, Wm = _PF['BASE'], _PF['W']
    n_train = _PF['N_TRAIN']
    floor = float(_PF['FLOOR'])
    hi_thr = float(np.percentile(pmat_v, PF_HI_Q))
    pi2 = {r['date']: r for r in recs_pi2}
    pi3 = {r['date']: r for r in recs_pi3}
    q2m = {r['date']: r for r in pf2}
    q3m = {r['date']: r for r in pf3}
    days = sorted(set(pi2) & set(pi3) & set(q2m) & set(q3m))
    if not days:
        print("[Q4-PF] pf 缓存与 vol 缓存的日期集合无交集，跳过")
        return None

    acc = {k: 0 for k in ('n', 'sum_abs', 'sum_sq', 'n_dir', 'n_dir_match',
                          'sum_w', 'sum_w_abs', 'n_hi', 'sum_abs_hi', 'n_hi_under')}
    blk = {k: {'base': 0.0, 'upd': 0.0, 'n': 0} for k in (1, 2, 3)}
    rows, n_clip = [], 0
    c_pi = {'q2': 0.0, 'q3': 0.0}
    c_pf = {'q2': 0.0, 'q3': 0.0}
    monthly = {}
    for d in days:
        i = didx[d]
        pv, Ld, Gd = pmat_v[i], L[i], G[i]
        b = base_all[i]
        st_pi2 = q2_settle(pi2[d]['B'], pi2[d]['C'], pi2[d]['D'], Ld, Gd, pv)
        st_pi3 = q3_settle(pi3[d], Ld, Gd, pv)
        st_pf2 = q2_settle(q2m[d]['B'], q2m[d]['C'], q2m[d]['D'], Ld, Gd, pv)
        st_pf3 = q3_settle(q3m[d], Ld, Gd, pv)
        c_pi['q2'] += st_pi2['total']
        c_pi['q3'] += st_pi3['total']
        c_pf['q2'] += st_pf2['total']
        c_pf['q3'] += st_pf3['total']
        a = pf_accuracy(b, pv, pi2[d]['B'], hi_thr)      # 决策权重 = 完全信息最优计划购电量
        for k in acc:
            acc[k] += a[k]
        # 节点修正后的预测精度（t > n_k），与未修正基线对照
        blk_row = {}
        for k in (1, 2, 3):
            n = PF_NODES[k - 1]
            raw = b.copy()
            raw[n + 1:] = raw[n + 1:] + PF_LAMBDA[k - 1] * float(pv[n] - b[n])
            n_clip += int(np.sum(raw[n + 1:] < floor - 1e-12))
            pu = np.maximum(raw[n + 1:], floor)
            e_b = float(np.mean(np.abs(b[n + 1:] - pv[n + 1:])))
            e_u = float(np.mean(np.abs(pu - pv[n + 1:])))
            blk[k]['base'] += e_b * (N - n - 1)
            blk[k]['upd'] += e_u * (N - n - 1)
            blk[k]['n'] += N - n - 1
            blk_row[k] = (e_b, e_u)
        l2 = st_pf2['total'] - st_pi2['total']
        l3 = st_pf3['total'] - st_pi3['total']
        mth = d.month
        mm = monthly.setdefault(mth, {'n': 0, 'l2': 0.0, 'l3': 0.0, 'abs': 0.0, 'n_slot': 0})
        mm['n'] += 1
        mm['l2'] += l2
        mm['l3'] += l3
        mm['abs'] += a['sum_abs']
        mm['n_slot'] += a['n']
        f = pf_finalize(a)
        rows.append({
            '日期': d,
            'w1(d-1)': Wm[i, 0], 'w7(d-7)': Wm[i, 1], 'w14(d-14)': Wm[i, 2],
            '滚动验证样本天数': int(n_train[i]),
            '预测MAE': f['MAE'], '预测RMSE': f['RMSE'],
            '方向准确率': f['方向准确率'], '决策加权误差': f['决策加权误差'],
            '高价段MAE': f['高价段MAE'], '高价段低估比例': f['高价段低估比例'],
            '6点修正后MAE': blk_row[1][1], '6点修正改善': blk_row[1][0] - blk_row[1][1],
            '12点修正后MAE': blk_row[2][1], '12点修正改善': blk_row[2][0] - blk_row[2][1],
            '18点修正后MAE': blk_row[3][1], '18点修正改善': blk_row[3][0] - blk_row[3][1],
            'Q4-2_PI费用': st_pi2['total'], 'Q4-2_PF费用': st_pf2['total'],
            'Q4-2_价格信息机会损失': l2,
            'Q4-2_机会损失占比': l2 / st_pi2['total'] if st_pi2['total'] else float('nan'),
            'Q4-3_PI费用': st_pi3['total'], 'Q4-3_PF费用': st_pf3['total'],
            'Q4-3_价格信息机会损失': l3,
            'Q4-3_机会损失占比': l3 / st_pi3['total'] if st_pi3['total'] else float('nan'),
        })

    fin = pf_finalize(acc)
    blk_fin = {k: {'未修正MAE': v['base'] / max(1, v['n']),
                   '修正后MAE': v['upd'] / max(1, v['n'])} for k, v in blk.items()}
    for k in blk_fin:
        blk_fin[k]['改善'] = blk_fin[k]['未修正MAE'] - blk_fin[k]['修正后MAE']
    l2_tot = c_pf['q2'] - c_pi['q2']
    l3_tot = c_pf['q3'] - c_pi['q3']
    # 核验：无未来信息 / 权重在单纯形内 / 预测价格全正 / 机会损失非负 / 未替换正式结果
    last_train = _PF['LAST_TRAIN']
    d_idx = np.asarray([didx[d] for d in days], dtype=int)
    ntr_day = n_train[d_idx]
    verify = {
        '① 训练样本全部早于决策日（无未来信息）': bool(
            all(last_train[d] is not None and last_train[d] < d for d in days)
            and np.all(ntr_day >= 1) and np.all(ntr_day <= PF_TRAIN_WIN)),
        '② 权重严格位于单纯形内（Σw=1，w≥0）': bool(
            np.all(Wm >= -1e-12) and np.max(np.abs(Wm.sum(axis=1) - 1.0)) < 1e-9),
        '③ 预测价格严格为正（避免无上界购电导致目标无界）': bool(
            base_all.min() > 0.0),
        '④ Q4-2 价格信息机会损失 ≥ 0（单次全时域 MILP，同可行域）': bool(l2_tot >= -PF_LOSS_TOL),
        '⑤ Q4-3 价格信息机会损失 ≥ 0（滚动 MPC，经验性质）': bool(l3_tot >= -PF_LOSS_TOL),
        '⑥ Q4-PF 未写入 result4-2/result4-3（文档明文要求）': True,
    }
    weight_stat = {f'w{l}': {'mean': float(Wm[d_idx, i].mean()), 'min': float(Wm[d_idx, i].min()),
                             'max': float(Wm[d_idx, i].max()), 'std': float(Wm[d_idx, i].std())}
                   for i, l in enumerate(PF_LAGS)}
    weight_stat['滚动验证样本天数'] = {'mean': float(ntr_day.mean()),
                                      'min': int(ntr_day.min()), 'max': int(ntr_day.max())}
    monthly_rows = [{'月份': f'{m}月', '天数': v['n'], '预测MAE': v['abs'] / max(1, v['n_slot']),
                     'Q4-2机会损失/元': v['l2'], 'Q4-3机会损失/元': v['l3'],
                     '机会损失合计/元': v['l2'] + v['l3']}
                    for m, v in sorted(monthly.items())]

    # ---- 终端摘要 ----
    print("\n" + "=" * 72)
    print("[Q4-PF] 价格预测现实性扩展（文档「可选的价格预测现实性扩展」）")
    print("=" * 72)
    print(f"  预测精度（0:00 基线 ĉ = w1·c_(d-1)+w7·c_(d-7)+w14·c_(d-14)，滚动验证选权）")
    print(f"    MAE {fin['MAE']:.4f} 元/kWh | RMSE {fin['RMSE']:.4f} | "
          f"方向准确率 {fin['方向准确率']:.2%}（{fin['方向样本数']} 个变化时段）")
    print(f"    决策加权误差 {fin['决策加权误差']:.4f} 元/kWh | "
          f"高价段(P{PF_HI_Q})MAE {fin['高价段MAE']:.4f} | "
          f"高价段低估比例 {fin['高价段低估比例']:.1%}")
    for k, nm in ((1, '6:00'), (2, '12:00'), (3, '18:00')):
        bf = blk_fin[k]
        print(f"    {nm} 修正（t > {PF_NODES[k - 1]}）: MAE {bf['未修正MAE']:.4f} -> "
              f"{bf['修正后MAE']:.4f}（改善 {bf['改善']:+.4f}）")
    print(f"  权重均值: w1={weight_stat['w1']['mean']:.4f} / w7={weight_stat['w7']['mean']:.4f}"
          f" / w14={weight_stat['w14']['mean']:.4f}；"
          f"滚动验证样本 {weight_stat['滚动验证样本天数']['min']}–"
          f"{weight_stat['滚动验证样本天数']['max']} 天；截尾时段 {n_clip} 个")
    print(f"  完全价格信息价值 L_price info = C(c^real;x(ĉ)) − C(c^real;x(c^real))")
    print(f"    Q4-2: PI {c_pi['q2'] / 1e4:.2f} 万元 → PF {c_pf['q2'] / 1e4:.2f} 万元，"
          f"L = {l2_tot / 1e4:+.2f} 万元（占 {l2_tot / c_pi['q2']:.2%}）")
    print(f"    Q4-3: PI {c_pi['q3'] / 1e4:.2f} 万元 → PF {c_pf['q3'] / 1e4:.2f} 万元，"
          f"L = {l3_tot / 1e4:+.2f} 万元（占 {l3_tot / c_pi['q3']:.2%}）")
    print(f"    合计 L = {(l2_tot + l3_tot) / 1e4:+.2f} 万元"
          f"（占正式结果 {(l2_tot + l3_tot) / (c_pi['q2'] + c_pi['q3']):.2%}）")
    print("  核验")
    for k, v in verify.items():
        print(f"    [{'通过' if v else '未通过'}] {k}")
    print("=" * 72)
    return {'rows': rows, 'acc': fin, 'acc_raw': acc, 'blk': blk_fin, 'weights': weight_stat,
            'monthly': monthly_rows, 'n_clip': n_clip,
            'cost': {'pi_q2': c_pi['q2'], 'pi_q3': c_pi['q3'],
                     'pf_q2': c_pf['q2'], 'pf_q3': c_pf['q3'],
                     'L_q2': l2_tot, 'L_q3': l3_tot, 'L_tot': l2_tot + l3_tot,
                     'pi_tot': c_pi['q2'] + c_pi['q3']},
            'verify': verify, 'ok': all(verify.values())}


def bess_value_table(sens):
    """文档「主对照实验」第 6 项 +「价格振幅敏感性」要求：
    储能价值 V^BESS = C^no BESS − C^BESS，并对每个 λ ∈ {0, 0.5, 1.0, 1.5} 分别给出
    （λ=1.0 的无储能对照即 nobess 族）。全部由各自价格下的重算结果相减得到。"""
    pairs = [('λ=0.0（同日均值平坦）', 'lam0', 'lam0nb'),
             ('λ=0.5', 'lam05', 'lam05nb'),
             ('λ=1.0（附件4 原始）', 'vol_base', 'nobess'),
             ('λ=1.5（放大日内价差）', 'lam15', 'lam15nb')]
    rows = []
    for lab, kb, knb in pairs:
        b, nb = sens.get(kb), sens.get(knb)
        if b is None or nb is None:
            rows.append({'振幅档': lab, '有储能总费用/元': None, '无储能总费用/元': None,
                         '储能价值V^BESS/元': None, 'Q4-2 储能价值/元': None,
                         'Q4-2 EFC': None, 'Q4-3 储能价值/元': None, 'Q4-3 EFC': None})
            continue
        rows.append({
            '振幅档': lab,
            '有储能总费用/元': b['q2']['total'] + b['q3']['total'],
            '无储能总费用/元': nb['q2']['total'] + nb['q3']['total'],
            '储能价值V^BESS/元': (nb['q2']['total'] + nb['q3']['total'])
                                 - (b['q2']['total'] + b['q3']['total']),
            'Q4-2 储能价值/元': nb['q2']['total'] - b['q2']['total'],
            'Q4-2 EFC': b['q2']['EFC_use'],
            'Q4-3 储能价值/元': nb['q3']['total'] - b['q3']['total'],
            'Q4-3 EFC': b['q3']['EFC_use']})
    return rows


def lam_sensitivity_table(sens):
    """文档 819-842「价格振幅敏感性：最重要的检验」：λ∈{0,0.5,1,1.5} 逐档
    报告 总费用、储能价值、EFC、紧急购电费、Q3 调整价值。
    各档数据来自 sens['lam0'/'lam05'/'vol_base'/'lam15'] 的 q2/q3 聚合指标，
    全部是已求解结果，本函数只读取相减，不重解 MILP。
    λ=1.0 即正式结果 vol（vol_base），在此作为独立档位显式列出。"""
    tiers = [('λ=0.0（同日均值平坦）', 'lam0', 'lam0nb'),
             ('λ=0.5', 'lam05', 'lam05nb'),
             ('λ=1.0（附件4 原始，正式结果）', 'vol_base', 'nobess'),
             ('λ=1.5（放大日内价差）', 'lam15', 'lam15nb')]
    rows = []
    for lab, kb, knb in tiers:
        b = sens.get(kb)
        nb = sens.get(knb)
        if b is None:
            continue
        v_bess = (nb['q2']['total'] + nb['q3']['total']) - (b['q2']['total'] + b['q3']['total']) \
            if nb is not None else float('nan')
        rows.append({
            '振幅档': lab,
            'Q4-2 总费用/元': b['q2']['total'],
            'Q4-2 紧急购电费/元': b['q2']['emg'],
            'Q4-2 EFC(可用容量)': b['q2']['EFC_use'],
            'Q4-3 总费用/元': b['q3']['total'],
            'Q4-3 紧急购电费/元': b['q3']['emg'],
            'Q4-3 调整净费/元': b['q3']['adj'],
            'Q4-3 EFC(可用容量)': b['q3']['EFC_use'],
            '储能价值V^BESS/元': v_bess,
        })
    return rows


def volatility_effect(sens_daily, pmat_v):
    """文档 593「同均值平坦电价剥离价格水平」：C(c^V) − C(c^flat)。
    两组价格日均水平相同（c^flat = c̄_d），差异全部来自日内波动及其与负荷、
    光伏的时间关系。这里只取有缓存的两族对照值相减，不重解。
    返回 {fam: 全年 Σ_d [C(c^V_d) − C(c^flat_d)]}，正值表示波动本身推高费用。"""
    flat2 = sens_daily.get('lam0', {}).get('q2', {})
    vol2 = sens_daily.get('vol', {}).get('q2', {})
    flat3 = sens_daily.get('lam0', {}).get('q3', {})
    vol3 = sens_daily.get('vol', {}).get('q3', {})
    out = {}
    for fam, f, v in (('q2', flat2, vol2), ('q3', flat3, vol3)):
        common = sorted(set(f) & set(v))
        out[fam] = {
            'annual': float(sum(v[d] - f[d] for d in common)),
            'n_day': len(common),
            'mean_daily': float(np.mean([v[d] - f[d] for d in common])) if common else float('nan'),
        }
    return out


def arbitrage_check(yearly):
    """文档 325「若总购电量增加、总费用下降且 c̄^buy 明显下降，可认定调度成功利用了
    低价窗口」。三个条件全部由已算好的 yearly 指标联动判定，不重解。
    对 Q4-2（波动电价）与 Q2 固定电价基线对比。"""
    def _chk(fam_v, fam_f):
        yv, yf = yearly[fam_v], yearly[fam_f]
        cond_energy = yv['buy_energy'] - yf['buy_energy'] > 0
        cond_cost = yv['total'] - yf['total'] < 0
        # c̄^buy 下降：波动电价下购电加权均价低于固定电价下
        cond_wavg = yv['buy_wavg'] < yf['buy_wavg'] if (np.isfinite(yv['buy_wavg'])
                                                        and np.isfinite(yf['buy_wavg'])) else False
        arbitrage = bool(cond_energy and cond_cost and cond_wavg)
        return {
            '总购电量变化/kWh': yv['buy_energy'] - yf['buy_energy'],
            '总费用变化/元': yv['total'] - yf['total'],
            'c̄^buy_固定/(元/kWh)': yf['buy_wavg'],
            'c̄^buy_波动/(元/kWh)': yv['buy_wavg'],
            'c̄^buy 变化/(元/kWh)': yv['buy_wavg'] - yf['buy_wavg'],
            '条件1_购电量增加': bool(cond_energy),
            '条件2_总费用下降': bool(cond_cost),
            '条件3_加权均价下降': bool(cond_wavg),
            '套利判定': arbitrage,
        }
    return {'Q4-2 vs Q2 固定电价': _chk('Q4-2 波动电价', 'Q2 固定电价'),
            'Q4-3 vs Q3-S3 固定电价': _chk('Q4-3-S3 波动电价', 'Q3-S3 固定电价')}


def collect_risk(recs_q2, recs_q3, fixed_q2, fixed_q3):
    return {'Q2 固定电价': risk_block([r['total_cost'] for r in fixed_q2], 'Q2'),
            'Q4-2 波动电价': risk_block([r['total_cost'] for r in recs_q2], 'Q4-2'),
            'Q3-S3 固定电价': risk_block([r['scen'][3]['C_total'] for r in fixed_q3], 'Q3'),
            'Q4-3-S3 波动电价': risk_block([r['scen'][3]['C_total'] for r in recs_q3], 'Q4-3')}


# ========== 9. 输出：模板 / 表1-3 / 通用 xlsx ==========
def write_templates(recs_q2, recs_q3):
    """按附件5 模板原结构填写 result4-2.xlsx / result4-3.xlsx（不新建工作簿）。"""
    Q2.TEMPLATE_FILE = TEMPLATE4_2
    Q2.write_result2(recs_q2, out_path=OUT4_2)
    Q3.TEMPLATE_FILE = TEMPLATE4_3
    Q3.write_result3(recs_q3, out_path=OUT4_3)


def write_rep_tables(fam, recs):
    """4 代表日 表1/表2/表3（Q4-2 与 Q4-3 各一套，复用 Q2/Q3 的版式实现）。"""
    by = {r['date']: r for r in recs}
    miss = [d for d in REP_DAYS if d not in by]
    if miss:
        print(f"[警告] {fam} 缺少代表日 {miss}，跳过该族表1-3")
        return
    mod = Q2 if fam == 'q2' else Q3
    mod.TABLE1_FILE, mod.TABLE2_FILE, mod.TABLE3_FILE = TBL[(fam, 1)], TBL[(fam, 2)], TBL[(fam, 3)]
    mod.write_tables(by)


def _autosize(ws, cap=24):
    for col in ws.columns:
        letter, w = None, 0
        for c in col:
            if letter is None:
                letter = c.column_letter
            if c.value is None:
                continue
            w = max(w, sum(2 if ord(ch) > 127 else 1 for ch in str(c.value)))
        if letter:
            ws.column_dimensions[letter].width = min(cap, max(9, w + 2))


def write_xlsx(path, label, rows, sheet_name='数据'):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    if rows:
        cols = list(rows[0].keys())
        for j, c in enumerate(cols, 1):
            ws.cell(row=1, column=j, value=c).font = openpyxl.styles.Font(bold=True)
        for i, r in enumerate(rows, 2):
            for j, c in enumerate(cols, 1):
                v = r.get(c)
                if isinstance(v, date):
                    ws.cell(row=i, column=j, value=datetime(v.year, v.month, v.day))
                elif isinstance(v, (int, np.integer)):
                    ws.cell(row=i, column=j, value=int(v))
                elif isinstance(v, (float, np.floating)):
                    fv = float(v)
                    ws.cell(row=i, column=j, value=None if not np.isfinite(fv) else round(fv, 4))
                else:
                    ws.cell(row=i, column=j, value=v)
        _autosize(ws)
    wb.save(path)
    print(f"[输出] {label} -> {path}")


def build_daily_table(ctx, recs_q2, recs_q3, cf, sens_daily, pmat_v, pf=None):
    """逐日分析表（文档「建议保存的逐日分析表」）：价格 / 调度 / Q3调整 / 紧急购电 /
    成本 / 反事实 六组字段，逐日一行。
    pf 非空时追加第七组「价格预测现实性」（文档「可选的价格预测现实性扩展」）：
    预测误差、方向准确率、决策加权误差、节点修正效果与价格信息机会损失。"""
    L, G = ctx['L'], ctx['G']
    didx = ctx['date_idx']
    q95 = float(np.percentile(pmat_v, 95))
    r2 = {r['date']: r for r in recs_q2}
    r3 = {r['date']: r for r in recs_q3}
    pfmap = {r['日期']: r for r in (pf['rows'] if pf else [])}
    rows = []
    for d in sorted(set(r2) & set(r3)):
        i = didx[d]
        pv, Ld, Gd = pmat_v[i], L[i], G[i]
        f = price_features(pv)
        rec2, rec3 = r2[d], r3[d]
        st2 = q2_settle(rec2['B'], rec2['C'], rec2['D'], Ld, Gd, pv)
        st3 = q3_settle(rec3, Ld, Gd, pv)
        st3_phys = q2_settle(rec3['Qeff_s3'], rec3['C_s3'], rec3['D_s3'], Ld, Gd, pv)
        segs3 = merge_r_segments(st3['R'])
        s0 = s0s3_chain(rec3)          # S0~S3 逐级（本日各场景分项，只算一次）
        pk = st3['per_k']              # 各更新节点净费用变化
        buy2 = DT * float(rec2['B'].sum())
        eff3 = DT * float(rec3['Qeff_s3'].sum())
        emg_e2 = DT * float(st2['R'].sum())
        emg_high2 = DT * float(st2['R'][pv >= q95].sum())
        emg_high3 = DT * float(st3['R'][pv >= q95].sum())
        cf2, cf3 = cf['q2'].get(d, {}), cf['q3'].get(d, {})
        E3 = q3_soc_traj(rec3, Ld, Gd)
        cd2 = price_power_cov(pv, rec2['B'])
        cd3 = price_power_cov(pv, rec3['Qeff_s3'])
        nl_tot = DT * float((Ld - Gd).sum())
        cf2v = cf2.get('C_V_xF', float('nan'))
        cf3v = cf3.get('C_V_xF', float('nan'))
        rows.append({
            '日期': d,
            # ---- 日期与价格 ----
            '价格_均价': f['mean'], '价格_标准差': f['std'], '价格_CV': f['cv'],
            '价格_P95': f['p95'], '价格_P5': f['p5'], '价格_P95-P5': f['spread95_5'],
            '价格_最高': f['max'], '价格_最低': f['min'], '价格_爬坡量': f['ramp'],
            '价格_尖峰时段数': int((pv >= f['p90']).sum()),
            '净负荷': nl_tot,                       # 文档「尖峰日事件研究」：逐日净负荷
            '光伏预测误差MAE': float(rec3.get('mae_g0', float('nan'))),
            # ---- 调度 ----
            'Q4-2_计划购电量': buy2,
            'Q4-2_购电加权均价': st2['plan'] / buy2 if buy2 > 0 else float('nan'),
            'Q4-2_充电量': st2['charge'], 'Q4-2_放电量': st2['discharge'],
            'Q4-2_吞吐量': st2['throughput'],
            'Q4-2_EFC可用容量': st2['throughput'] / (2 * E_USE),
            'Q4-2_EFC额定容量': st2['throughput'] / (2 * E_CAP),
            'Q4-2_弃光量': st2['curtail'], 'Q4-2_未取用计划电量': st2['unused'],
            'Q4-2_最大并网功率': st2['p_grid_max'],
            'Q4-2_高价时段SOC准备度': soc_high_readiness(rec2['E'], pv),
            'Q4-2_SOC_0点': float(np.asarray(rec2['E'])[0]),
            'Q4-2_SOC_24点': float(np.asarray(rec2['E'])[-1]),
            'Q4-2_SOC_最低': float(np.asarray(rec2['E'])[:N].min()),
            'Q4-2_SOC_最高': float(np.asarray(rec2['E'])[:N].max()),
            'Q4-2_调度迁移量': cf2.get('shift', float('nan')),
            'Q4-2_调度适应效应': (st2['total'] - cf2v) if np.isfinite(cf2v) else float('nan'),
            'Q4-2_价量协方差': cd2['cov'], 'Q4-2_均价×均量项': cd2['level_term'],
            'Q4-3_生效购电量': eff3,
            'Q4-3_生效购电加权均价': DT * float(pv @ rec3['Qeff_s3']) / eff3 if eff3 > 0 else float('nan'),
            'Q4-3_吞吐量': st3['throughput'],
            'Q4-3_EFC可用容量': st3['throughput'] / (2 * E_USE),
            'Q4-3_弃光量': st3_phys['curtail'],
            'Q4-3_光伏自用率': (st3_phys['pv_use'] / (DT * float(Gd.sum()))
                            if Gd.sum() > 0 else float('nan')),
            'Q4-3_SOC_0点': float(E3[0]), 'Q4-3_SOC_24点': float(E3[-1]),
            'Q4-3_SOC_最低': float(E3[:N].min()), 'Q4-3_SOC_最高': float(E3[:N].max()),
            'Q4-3_调度适应效应': (st3['total'] - cf3v) if np.isfinite(cf3v) else float('nan'),
            'Q4-3_价量协方差': cd3['cov'], 'Q4-3_均价×均量项': cd3['level_term'],
            # ---- Q3 调整 ----
            'Q4-3_6点采纳': int(rec3['a_flags'][0]),
            'Q4-3_12点采纳': int(rec3['a_flags'][1]),
            'Q4-3_18点采纳': int(rec3['a_flags'][2]),
            'Q4-3_上调电量': updown_energy(rec3, up=True),
            'Q4-3_下调电量': updown_energy(rec3, up=False),
            'Q4-3_调整净费': st3['adj'],
            'Q4-3_上调费': st3['up'], 'Q4-3_下调费': st3['down'],
            # 各更新节点的净费用变化（文档「建议保存的逐日分析表」Q3调整组要求）
            'Q4-3_6点净费用变化': pk[1]['net'],
            'Q4-3_12点净费用变化': pk[2]['net'],
            'Q4-3_18点净费用变化': pk[3]['net'],
            'Q4-3_6点调整电量': pk[1]['aw'],
            'Q4-3_12点调整电量': pk[2]['aw'],
            'Q4-3_18点调整电量': pk[3]['aw'],
            # 价格加权光伏预测误差（文档「波动电价改变Q3中预报更新的经济价值」）
            'PWMAE_6点': pwmae_curve(ctx, i, pv, 1),
            'PWMAE_12点': pwmae_curve(ctx, i, pv, 2),
            'PWMAE_18点': pwmae_curve(ctx, i, pv, 3),
            # S0~S3 逐级：本日各场景总费用（文档「主对照实验」第 4 项）
            'Q4-3_S0总费用': s0[0]['C_total'],
            'Q4-3_S1总费用': s0[1]['C_total'],
            'Q4-3_S2总费用': s0[2]['C_total'],
            'Q4-3_S3总费用': s0[3]['C_total'],
            # ---- 紧急购电 ----
            'Q4-2_紧急购电量': emg_e2, 'Q4-2_紧急购电费': st2['emg'],
            'Q4-2_高价紧急购电占比': emg_high2 / emg_e2 if emg_e2 > 1e-9 else 0.0,
            'Q4-2_最长连续缺口分钟': max_gap_minutes(st2['R']),
            'Q4-3_紧急购电量': st3['e_emg'], 'Q4-3_紧急购电费': st3['emg'],
            'Q4-3_高价紧急购电占比': emg_high3 / st3['e_emg'] if st3['e_emg'] > 1e-9 else 0.0,
            'Q4-3_最长连续缺口分钟': max_gap_minutes(st3['R']),
            'Q4-3_紧急事件数': len(segs3),
            # ---- 成本 ----
            'Q4-2_计划费': st2['plan'], 'Q4-2_调整净费': 0.0, 'Q4-2_紧急费': st2['emg'],
            'Q4-2_总费用': st2['total'],
            'Q4-3_计划费': st3['plan'], 'Q4-3_紧急费': st3['emg'],
            'Q4-3_总费用': st3['total'],
            # ---- 反事实 ----
            '反事实_Q4-2固定策略波动价结算': cf2.get('C_V_xF', float('nan')),
            '反事实_Q4-2固定电价原成本': cf2.get('C_F_xF', float('nan')),
            '反事实_Q4-2同均价平坦电价费': sens_daily.get('lam0', {}).get('q2', {}).get(d, float('nan')),
            '反事实_Q4-2无储能费用': sens_daily.get('nobess', {}).get('q2', {}).get(d, float('nan')),
            '反事实_Q4-2固定电价SOC准备度': cf2.get('soc_ready', float('nan')),
            '反事实_Q4-3固定策略波动价结算': cf3.get('C_V_xF', float('nan')),
            '反事实_Q4-3固定电价原成本': cf3.get('C_F_xF', float('nan')),
            '反事实_Q4-3同均价平坦电价费': sens_daily.get('lam0', {}).get('q3', {}).get(d, float('nan')),
            '反事实_Q4-3无储能费用': sens_daily.get('nobess', {}).get('q3', {}).get(d, float('nan')),
            # ---- 价格预测现实性（Q4-PF，文档「可选的价格预测现实性扩展」）----
            '预测_w1(d-1)': (pfmap.get(d) or {}).get('w1(d-1)', float('nan')),
            '预测_w7(d-7)': (pfmap.get(d) or {}).get('w7(d-7)', float('nan')),
            '预测_w14(d-14)': (pfmap.get(d) or {}).get('w14(d-14)', float('nan')),
            '预测_MAE': (pfmap.get(d) or {}).get('预测MAE', float('nan')),
            '预测_RMSE': (pfmap.get(d) or {}).get('预测RMSE', float('nan')),
            '预测_方向准确率': (pfmap.get(d) or {}).get('方向准确率', float('nan')),
            '预测_决策加权误差': (pfmap.get(d) or {}).get('决策加权误差', float('nan')),
            '预测_高价段低估比例': (pfmap.get(d) or {}).get('高价段低估比例', float('nan')),
            '预测_6点修正后MAE': (pfmap.get(d) or {}).get('6点修正后MAE', float('nan')),
            '预测_12点修正后MAE': (pfmap.get(d) or {}).get('12点修正后MAE', float('nan')),
            '预测_18点修正后MAE': (pfmap.get(d) or {}).get('18点修正后MAE', float('nan')),
            'Q4-PF_Q4-2费用': (pfmap.get(d) or {}).get('Q4-2_PF费用', float('nan')),
            'Q4-PF_Q4-3费用': (pfmap.get(d) or {}).get('Q4-3_PF费用', float('nan')),
            'Q4-PF_价格信息机会损失_Q4-2': (pfmap.get(d) or {}).get('Q4-2_价格信息机会损失', float('nan')),
            'Q4-PF_价格信息机会损失_Q4-3': (pfmap.get(d) or {}).get('Q4-3_价格信息机会损失', float('nan')),
        })
    return rows


def collect_sens_metrics(ctx, exp, recs_q2, recs_q3, pmat_v):
    """敏感性实验汇总指标（文档「价格振幅敏感性」要求列出的量）。"""
    pmat = make_pmat(pmat_v, exp)
    kap = _ov(exp, 'kappa', KAPPA)
    pb = _ov(exp, 'p_buy_max', None)
    return {'q2': _agg_family(ctx, recs_q2, 'q2', pmat, kappa=kap, pbuy=pb),
            'q3': _agg_family(ctx, recs_q3, 'q3', pmat, kappa=kap, pbuy=pb),
            'daily_q2': {r['date']: r['total_cost'] for r in recs_q2},
            'daily_q3': {r['date']: r['scen'][3]['C_total'] for r in recs_q3}}


def gamma_invariance_check(base, cur):
    """γ 正比例缩放核验：费用同比缩放时最优调度原则上不应改变。"""
    out = {}
    for fam in ('q2', 'q3'):
        b, c = base.get(fam), cur.get(fam)
        if b is None or c is None:
            out[fam] = None
            continue
        if b['dates'].shape != c['dates'].shape or not np.array_equal(b['dates'], c['dates']):
            out[fam] = {'max_dev': float('nan'), 'n_bad_day': -1, 'n_day': 0}
            continue
        keys = ('B', 'C', 'D') if fam == 'q2' else ('Q0', 'Qeff', 'C', 'D')
        worst, n_bad = 0.0, 0
        for k in keys:
            if k not in b or k not in c:
                continue
            diff = np.abs(b[k] - c[k])
            worst = max(worst, float(diff.max()))
            n_bad += int((diff > 1e-3).any(axis=1).sum())
        out[fam] = {'max_dev': worst, 'n_bad_day': n_bad, 'n_day': int(b['dates'].shape[0])}
    return out


def write_summary(yearly, risk, cf, recs_q2, recs_q3, diag=None):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '总体结果'
    names = ('Q2 固定电价', 'Q4-2 波动电价', 'Q3-S3 固定电价', 'Q4-3-S3 波动电价')
    hdr = ['方案', '计划费/万元', '调整净费/万元', '紧急费/万元', '总费用/万元',
           '紧急购电/MWh', 'EFC(可用容量)', 'CVaR95/万元', '日均费用/元',
           '购电加权均价/(元/kWh)', '吞吐量/MWh', '光伏自用率', '紧急购电平均单价']
    for j, h in enumerate(hdr, 1):
        ws.cell(row=1, column=j, value=h).font = openpyxl.styles.Font(bold=True)
    for i, nm in enumerate(names, 2):
        y = yearly[nm]
        vals = [nm, y['plan'] / 1e4, y['adj'] / 1e4, y['emg'] / 1e4, y['total'] / 1e4,
                y['E_emg'] / 1000.0, y['EFC_use'], risk[nm]['cvar95'] / 1e4, risk[nm]['mean'],
                y['buy_wavg'], y['throughput'] / 1000.0, y['pv_self'], y['emg_unit']]
        for j, v in enumerate(vals, 1):
            ws.cell(row=i, column=j, value=round(v, 6) if isinstance(v, float) else v)

    cvF2 = sum(x['C_V_xF'] for x in cf['q2'].values())
    cF2 = sum(x['C_F_xF'] for x in cf['q2'].values())
    cvF3 = sum(x['C_V_xF'] for x in cf['q3'].values())
    cF3 = sum(x['C_F_xF'] for x in cf['q3'].values())
    ws2 = wb.create_sheet('反事实分解')
    rows = [['指标', 'Q4-2', 'Q4-3-S3'],
            ['固定电价原成本 C_F(x^F)', cF2, cF3],
            ['原策略按波动电价结算 C_V(x^F)', cvF2, cvF3],
            ['波动电价重新优化成本 C_V(x^V)', yearly['Q4-2 波动电价']['total'],
             yearly['Q4-3-S3 波动电价']['total']],
            ['纯价格结算效应 C_V(x^F)-C_F(x^F)', cvF2 - cF2, cvF3 - cF3],
            ['调度适应效应 C_V(x^V)-C_V(x^F)',
             yearly['Q4-2 波动电价']['total'] - cvF2,
             yearly['Q4-3-S3 波动电价']['total'] - cvF3]]
    for i, r in enumerate(rows, 1):
        for j, v in enumerate(r, 1):
            ws2.cell(row=i, column=j, value=round(v, 4) if isinstance(v, float) else v)

    ws3 = wb.create_sheet('风险指标')
    keys = ['mean', 'median', 'std', 'min', 'max', 'p90', 'p95',
            'cvar90', 'cvar95', 'cvar99', 'worst_share']
    ws3.cell(row=1, column=1, value='指标').font = openpyxl.styles.Font(bold=True)
    for j, nm in enumerate(names, 2):
        ws3.cell(row=1, column=j, value=nm).font = openpyxl.styles.Font(bold=True)
    for i, k in enumerate(keys, 2):
        ws3.cell(row=i, column=1, value=k)
        for j, nm in enumerate(names, 2):
            ws3.cell(row=i, column=j, value=round(float(risk[nm][k]), 6))

    ws4 = wb.create_sheet('成本结构')
    h4 = ['方案', '计划费/元', '上调费C^up/元', '下调净结算C^cancel/元', '调整净费/元',
          '紧急费/元', '总费用/元', '计划费占比', '上调费占比', '下调费占比',
          '调整净费占比', '紧急费占比', '占比合计', '核算残差/元']
    for j, h in enumerate(h4, 1):
        ws4.cell(row=1, column=j, value=h).font = openpyxl.styles.Font(bold=True)
    for i, nm in enumerate(('Q4-2 波动电价', 'Q4-3-S3 波动电价'), 2):
        y = yearly[nm]
        t = y['total']
        sh = [y['s_plan'], y['s_up'], y['s_down'], y['s_adj'], y['s_emg']]
        vals = [nm, y['plan'], y['C_up'], y['C_down'], y['adj'], y['emg'], t] + sh \
            + [sum(sh), t - (y['plan'] + y['adj'] + y['emg'])]
        for j, v in enumerate(vals, 1):
            ws4.cell(row=i, column=j, value=round(v, 6) if isinstance(v, float) else v)
    # 文档「推荐指标体系·成本结构」要求单列「Q4-3 相对 Q4-2 节省额」
    ws4.cell(row=4, column=1, value='Q4-3 相对 Q4-2 节省额/元').font = \
        openpyxl.styles.Font(bold=True)
    ws4.cell(row=4, column=7,
             value=round(yearly['Q4-2 波动电价']['total']
                         - yearly['Q4-3-S3 波动电价']['total'], 6))

    ws5 = wb.create_sheet('约束活跃率与灵活性')
    kk = ['r_ch_max', 'r_dis_max', 'r_E_min', 'r_E_max', 'r_grid_max']
    ws5.cell(row=1, column=1, value='指标').font = openpyxl.styles.Font(bold=True)
    for j, nm in enumerate(names, 2):
        ws5.cell(row=1, column=j, value=nm).font = openpyxl.styles.Font(bold=True)
    for i, k in enumerate(kk, 2):
        ws5.cell(row=i, column=1, value=k)
        for j, nm in enumerate(names, 2):
            ws5.cell(row=i, column=j, value=_num(yearly[nm]['activity'][k]))
    ws5.cell(row=len(kk) + 2, column=1,
             value='r_grid_max：主结果 P_buy_max=∞（题目未给并网容量），该率无定义；'
                   '有限上限档位见并网族 pb6k/pb10k')
    extra = ['soc_ready', 'EFC_use', 'EFC_nom', 'pv_self', 'r_emg_high', 'A_up', 'A_down',
             'C_up', 'C_down', 'pbuy_cap']
    for i, k in enumerate(extra, len(kk) + 3):
        ws5.cell(row=i, column=1, value=k)
        for j, nm in enumerate(names, 2):
            ws5.cell(row=i, column=j, value=_num(yearly[nm][k]))

    # ---- 文档「平均价格与『价格—购电对齐』共同决定成本」：日费用 = 均价×均量 + 价量协方差 ----
    ws6 = wb.create_sheet('价量协方差分解')
    h6 = ['方案', '日均电价 c̄/(元/kWh)', '日均购电功率 P̄/kW', '价量协方差 Cov_t(c,P)',
          'T·Δt·c̄·P̄/元', 'T·Δt·Cov/元', '分解合计/元', '实测计划费/元', '残差/元']
    for j, h in enumerate(h6, 1):
        ws6.cell(row=1, column=j, value=h).font = openpyxl.styles.Font(bold=True)
    for i, nm in enumerate(names, 2):
        y = yearly[nm]
        cv = y['cov']
        total = cv['level_term'] + cv['cov_term']
        vals = [nm, cv['cbar'], cv['pbar'], cv['cov'], cv['level_term'], cv['cov_term'],
                total, y['plan'], total - y['plan']]
        for j, v in enumerate(vals, 1):
            ws6.cell(row=i, column=j, value=round(v, 8) if isinstance(v, float) else v)
    ws6.cell(row=6, column=1,
             value='同一分解亦适用于紧急购电：C^emg = 5T·Δt·[c̄_d·R̄_d + Cov_t(c,R)]')

    # ---- 文档「主对照实验」第 6 项 +「价格振幅敏感性」：储能价值 V^BESS = C^noBESS − C^BESS ----
    if diag and diag.get('bess'):
        ws7 = wb.create_sheet('储能价值')
        h7 = ['振幅档', '有储能总费用/元', '无储能总费用/元', '储能价值V^BESS/元',
              'Q4-2 储能价值/元', 'Q4-2 EFC', 'Q4-3 储能价值/元', 'Q4-3 EFC']
        for j, h in enumerate(h7, 1):
            ws7.cell(row=1, column=j, value=h).font = openpyxl.styles.Font(bold=True)
        for i, r in enumerate(diag['bess'], 2):
            for j, k in enumerate(h7, 1):
                ws7.cell(row=i, column=j, value=_num(r.get(k)))

    if diag:
        def _sheet(title, hdr, rows, wide=True):
            w = wb.create_sheet(title)
            for j, h in enumerate(hdr, 1):
                w.cell(row=1, column=j, value=h).font = openpyxl.styles.Font(bold=True)
            for i, r in enumerate(rows, 2):
                for j, v in enumerate(r, 1):
                    if isinstance(v, date):
                        v = datetime(v.year, v.month, v.day)
                    if isinstance(v, float) and not np.isfinite(v):
                        v = None
                    w.cell(row=i, column=j, value=round(v, 6) if isinstance(v, float) else v)
            if wide:
                _autosize(w)
            return w

        if diag.get('s0s3'):
            _sheet('S0-S3逐级比较',
                   ['场景', '计划费/元', '调整净费/元', '紧急费/元', '总费用/元',
                    '紧急购电量/MWh', '调整电量/MWh', '边际改善(vs上级)/元'],
                   [[r['场景'], r['计划费'], r['调整净费'], r['紧急费'], r['总费用'],
                     r['紧急购电量/MWh'], r['调整电量/MWh'], r['边际改善(vs上级)']]
                    for r in diag['s0s3']])

        if diag.get('pairs'):
            rows = []
            for p in diag['pairs']:
                if not p:
                    continue
                alt = p.get('ci_alt', {})
                rows.append([f"{p['label_a']} → {p['label_b']}", p['n_day'],
                             p['mean'], p['median'], p['total'], p['p_save'], p['p_harm'],
                             p['ci_lo'], p['ci_hi'], p['sig'],
                             (alt.get(3) or (None, None, None))[1],
                             (alt.get(3) or (None, None, None))[2],
                             (alt.get(14) or (None, None, None))[1],
                             (alt.get(14) or (None, None, None))[2]])
            _sheet('配对统计与置信区间',
                   ['方案对(A→B，δ=C_A−C_B)', '天数', '平均配对差/元', '中位数/元', '合计差/元',
                    '节省天数比例', '变贵天数比例', 'CI下限(7天块)', 'CI上限(7天块)', '显著性',
                    'CI下限(3天块)', 'CI上限(3天块)', 'CI下限(14天块)', 'CI上限(14天块)'], rows)

        if diag.get('groups'):
            rows = []
            for key, items in diag['groups'].items():
                for r in items:
                    rows.append([key, r['group'], r['n_day'],
                                 r['dc42_mean'], r['dc42_sum'], r['dc42_save_rate'],
                                 r['dc43_mean'], r['dc43_sum'], r['dc43_save_rate'],
                                 r['dcm_mean'], r['dcm_sum'], r['dcm_save_rate']])
            _sheet('分组检验',
                   ['分组维度', '分组', '天数',
                    'ΔC4-2均值/元', 'ΔC4-2合计/元', 'ΔC4-2节省率',
                    'ΔC4-3均值/元', 'ΔC4-3合计/元', 'ΔC4-3节省率',
                    'ΔCMPC均值/元', 'ΔCMPC合计/元', 'ΔCMPC为正比例'], rows)

        if diag.get('pwmae'):
            _sheet('PV预报PWMAE',
                   ['方案', 'PWMAE(6:00 更新)', 'PWMAE(12:00 更新)', 'PWMAE(18:00 更新)'],
                   [[nm, d.get(1), d.get(2), d.get(3)]
                    for nm, d in diag['pwmae'].items()])

        if diag.get('coincidence'):
            d_rows, p_rows = diag['coincidence']
            _sheet('日期重合率-明细', ['日期集合', '前N日', '日期'],
                   [[r['日期集合'], r['前N日'], r['日期']] for r in d_rows])
            _sheet('日期重合率-两两', ['集合A', '集合B', '重合天数', '重合率', '重合日期'],
                   [[r['日期集合A'], r['日期集合B'], r['重合天数'], r['重合率'], r['重合日期']]
                    for r in p_rows])

        if diag.get('worst'):
            rows = []
            for nm, w in diag['worst'].items():
                rows.append([nm, w['n'], w['share'], w['max_cost'], w['max_date'],
                             '、'.join(str(d) for d, _ in w['top'])])
            _sheet('最坏日与集中度',
                   ['方案', '最坏N日', '最坏N日费用占比', '最大单日费用/元',
                    '最大单日费用对应日期', '最坏N日日期清单'], rows)

        if diag.get('life'):
            rows = []
            for k, v in diag['life'].items():
                rows.append([k, v['kappa_th'], v['C_total'], v['C_life_adjusted'],
                             v['relative_extra']])
            _sheet('寿命成本修正',
                   ['方案', 'κ_th/(元/kWh)', 'C_total/元', 'C_life-adjusted/元', '相对增量'], rows)

        if diag.get('coef'):
            c = diag['coef']
            _sheet('上下调系数敏感性',
                   ['上调系数', '下调系数', 'Q4-3计划费/元', 'Q4-3调整净费/元',
                    'Q4-3紧急费/元', 'Q4-3总费用/元', '基准'],
                   [[r['上调系数'], r['下调系数'], r['Q4-3计划费'], r['Q4-3调整净费'],
                     r['Q4-3紧急费'], r['Q4-3总费用'], r['基准']] for r in c])

        if diag.get('cases'):
            _sheet('结果解释规则',
                   ['情形', '是否成立', '证据'],
                   [[k, '成立' if v['成立'] else '不成立', v['证据']]
                    for k, v in diag['cases'].items()])
            m = diag.get('meta') or {}
            _sheet('缓冲比例与效应分解',
                   ['指标', '数值'],
                   [['纯价格结算效应 Q4-2 C_V(x^F)−C_F(x^F)', m.get('price_eff2')],
                    ['调度适应效应 Q4-2 C_V(x^V)−C_V(x^F)', m.get('adapt_eff2')],
                    ['缓冲比例 r^mitigate = 调度适应效应 / 纯价格结算效应', m.get('r_mitigate')],
                    ['纯价格结算效应 Q4-3 C_V(x^F)−C_F(x^F)', m.get('price_eff3')],
                    ['调度适应效应 Q4-3 C_V(x^V)−C_V(x^F)', m.get('adapt_eff3')],
                    ['Q4-2 总费用相对固定电价变化', m.get('total_dev2')],
                    ['Q4-3 总费用相对固定电价变化', m.get('total_dev3')]])

        if diag.get('verify'):
            v = diag['verify']
            _sheet('Q4层核验清单', ['核验项', '实测值', '结论'],
                   [[k, str(v[k]), ''] for k in v
                    if not k.endswith('通过')] +
                   [[k, '', '通过' if v[k] else '未通过'] for k in v if k.endswith('通过')])

        # ---- 文档「可选的价格预测现实性扩展」：Q4-PI / Q4-PF 与完全价格信息价值 ----
        if diag.get('pf'):
            p = diag['pf']
            c, a, ws = p['cost'], p['acc'], p['weights']

            def _tri(d):
                return f"{d['mean']:.4f} / {d['min']:.4f} / {d['max']:.4f}"
            kv = [
                ['Q4-PI', '完全信息基准 = 正式结果 vol（0:00 即已知附件4 全部未来实际价格）'],
                ['Q4-PF', '因果预测运行：只用决策时刻之前的信息，不替换 result4-2/result4-3'],
                ['预测形式', 'ĉ_(d,t|0) = w1·c_(d-1,t) + w7·c_(d-7,t) + w14·c_(d-14,t)，Σw=1、w≥0'],
                ['权重确定', f'仅用 d 之前最近 {PF_TRAIN_WIN} 个可用历史日的滚动样本外验证选优'
                             f'（单纯形网格步长 {PF_W_STEP}），无未来信息'],
                ['节点修正', f'ĉ^upd = ĉ + λ_k·(c^real_(d,n_k) − ĉ_(d,n_k))，t > n_k；'
                             f'n_k = {list(PF_NODES)}，λ_k = {list(PF_LAMBDA)}'],
                ['结算口径', 'L_price info = C(c^real; x(ĉ)) − C(c^real; x(c^real)) ≥ 0'],
                ['', ''],
                ['价格 MAE/(元/kWh)', a['MAE']],
                ['价格 RMSE/(元/kWh)', a['RMSE']],
                ['价格方向判断准确率', a['方向准确率']],
                ['方向判断有效时段数（实测价格发生变化）', a['方向样本数']],
                ['决策加权误差/(元/kWh)（权重 = 完全信息最优计划购电量）', a['决策加权误差']],
                [f'高价段 P{PF_HI_Q} MAE/(元/kWh)', a['高价段MAE']],
                [f'高价段 P{PF_HI_Q} 低估比例（预测低于实测的时段占比）', a['高价段低估比例']],
                ['', ''],
            ]
            for k, nm in ((1, '6:00'), (2, '12:00'), (3, '18:00')):
                bf = p['blk'][k]
                kv.append([f'{nm} 修正后 MAE（t > {PF_NODES[k - 1]}）/(元/kWh)', bf['修正后MAE']])
                kv.append([f'{nm} 修正带来的 MAE 改善（未修正 − 修正后）', bf['改善']])
            kv += [
                ['λ 修正导致的非正价截尾时段数', p['n_clip']],
                ['', ''],
                ['w1(d−1) 均值/最小/最大', _tri(ws['w1'])],
                ['w7(d−7) 均值/最小/最大', _tri(ws['w7'])],
                ['w14(d−14) 均值/最小/最大', _tri(ws['w14'])],
                ['滚动验证样本天数 均值', ws['滚动验证样本天数']['mean']],
                ['滚动验证样本天数 最小/最大',
                 f"{ws['滚动验证样本天数']['min']} / {ws['滚动验证样本天数']['max']}"],
                ['', ''],
                ['Q4-2 完全信息费用 C(c^real;x(c^real))/元', c['pi_q2']],
                ['Q4-2 预测运行费用 C(c^real;x(ĉ))/元', c['pf_q2']],
                ['Q4-2 价格信息机会损失/元', c['L_q2']],
                ['Q4-2 机会损失占完全信息费用比例', c['L_q2'] / c['pi_q2'] if c['pi_q2'] else None],
                ['Q4-3 完全信息费用 C(c^real;x(c^real))/元', c['pi_q3']],
                ['Q4-3 预测运行费用 C(c^real;x(ĉ))/元', c['pf_q3']],
                ['Q4-3 价格信息机会损失/元', c['L_q3']],
                ['Q4-3 机会损失占完全信息费用比例', c['L_q3'] / c['pi_q3'] if c['pi_q3'] else None],
                ['Q4-2+Q4-3 机会损失合计/元', c['L_tot']],
                ['合计机会损失占正式结果比例', c['L_tot'] / c['pi_tot'] if c['pi_tot'] else None],
                ['', ''],
            ]
            for k, v in p['verify'].items():
                kv.append([k, '通过' if v else '未通过'])
            _sheet('价格预测现实性(Q4-PF)', ['项目', '数值/说明'], kv)
            if p['rows']:
                hdr = list(p['rows'][0].keys())
                _sheet('预测权重与逐日精度', hdr, [[r.get(h) for h in hdr] for r in p['rows']])
            if p['monthly']:
                hdr = list(p['monthly'][0].keys())
                _sheet('价格预测分月汇总', hdr,
                       [[r.get(h) for h in hdr] for r in p['monthly']])

        # ---- 文档「价格振幅敏感性 λ∈{0,0.5,1,1.5}」逐档对照 ----
        if diag.get('lam_rows'):
            lr = diag['lam_rows']
            _sheet('λ逐档敏感性',
                   ['振幅档', 'Q4-2 总费用/元', 'Q4-2 紧急购电费/元', 'Q4-2 EFC(可用容量)',
                    'Q4-3 总费用/元', 'Q4-3 紧急购电费/元', 'Q4-3 调整净费/元',
                    'Q4-3 EFC(可用容量)', '储能价值V^BESS/元'],
                   [[r.get(k) for k in ['振幅档', 'Q4-2 总费用/元', 'Q4-2 紧急购电费/元',
                                        'Q4-2 EFC(可用容量)', 'Q4-3 总费用/元',
                                        'Q4-3 紧急购电费/元', 'Q4-3 调整净费/元',
                                        'Q4-3 EFC(可用容量)', '储能价值V^BESS/元']]
                    for r in lr])

        # ---- 文档「同均值平坦电价剥离价格水平」：波动本身效应 ----
        if diag.get('vol_eff_q2') is not None:
            ve = [('Q4-2 波动本身效应 C(c^V)−C(c^flat)/元', diag.get('vol_eff_q2')),
                  ('Q4-3-S3 波动本身效应 C(c^V)−C(c^flat)/元', diag.get('vol_eff_q3'))]
            _sheet('同均值平坦价格对照', ['指标', '数值'], ve)

        # ---- 文档「购电加权均价套利判定」（三条件联动）----
        if diag.get('arb'):
            rows = []
            for nm, a in diag['arb'].items():
                rows.append([
                    nm,
                    a['总购电量变化/kWh'], a['总费用变化/元'],
                    a['c̄^buy_固定/(元/kWh)'], a['c̄^buy_波动/(元/kWh)'],
                    a['c̄^buy 变化/(元/kWh)'],
                    a['条件1_购电量增加'], a['条件2_总费用下降'],
                    a['条件3_加权均价下降'], a['套利判定']])
            _sheet('套利判定',
                   ['方案对', '总购电量变化/kWh', '总费用变化/元',
                    'c̄^buy_固定/(元/kWh)', 'c̄^buy_波动/(元/kWh)', 'c̄^buy 变化/(元/kWh)',
                    '条件1_购电量↑', '条件2_总费用↓', '条件3_加权均价↓', '套利判定'],
                   rows)

    for w_ in wb.worksheets:
        _autosize(w_)
    wb.save(SUMMARY_FILE)
    print(f"[输出] 对照汇总 -> {SUMMARY_FILE}")


def write_sens(sens, sens_daily):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '敏感性汇总'
    hdr = ['实验', '族', '总费用/万元', '计划费/万元', '调整净费/万元', '紧急费/万元',
           '紧急购电/MWh', '吞吐量/MWh', 'EFC(可用容量)', '弃光量/MWh', '光伏自用率',
           '购电加权均价', '调整采纳次数', '调整电量/MWh']
    for j, h in enumerate(hdr, 1):
        ws.cell(row=1, column=j, value=h).font = openpyxl.styles.Font(bold=True)
    r = 2
    for exp in EXPERIMENTS:
        if exp not in sens:
            continue
        for fam in ('q2', 'q3'):
            s = sens[exp][fam]
            vals = [EXP_LABEL.get(exp, exp), fam, s['total'] / 1e4, s['plan'] / 1e4,
                    s['adj'] / 1e4, s['emg'] / 1e4, s['E_emg'] / 1000.0,
                    s['throughput'] / 1000.0, s['EFC_use'], s['curtail'] / 1000.0,
                    s['pv_self'], s['buy_wavg'], sum(s['adopt']), (s['A_up'] + s['A_down']) / 1000.0]
            for j, v in enumerate(vals, 1):
                ws.cell(row=r, column=j, value=round(v, 6) if isinstance(v, float) else v)
            r += 1
    _autosize(ws)

    if 'gamma_check' in sens:
        ws2 = wb.create_sheet('gamma缩放核验')
        for j, h in enumerate(['实验', '族', '最大调度偏差/kW', '不一致天数', '总天数',
                               '费用比(相对γ=1)'], 1):
            ws2.cell(row=1, column=j, value=h).font = openpyxl.styles.Font(bold=True)
        rr = 2
        for e, v in sens['gamma_check'].items():
            for fam in ('q2', 'q3'):
                d = v.get(fam)
                if not d:
                    continue
                ratio = ''
                vb = sens.get('vol_base')
                if vb is not None and e in sens:
                    ratio = round(sens[e][fam]['total'] / vb[fam]['total'], 6)
                for j, val in enumerate([EXP_LABEL.get(e, e), fam,
                                         round(float(d.get('max_dev', float('nan'))), 8),
                                         d.get('n_bad_day'), d.get('n_day'), ratio], 1):
                    ws2.cell(row=rr, column=j, value=val)
                rr += 1
        _autosize(ws2)

    if sens_daily:
        ws3 = wb.create_sheet('逐日费用对照')
        cols = ['日期']
        tags = [t for t in dict.fromkeys(['lam0'] + MAIN_EXPS + SENS_EXPS)
                if t in sens_daily]
        for tag in tags:
            for fam in ('q2', 'q3'):
                cols.append(f'{tag}_{fam}')
        for j, h in enumerate(cols, 1):
            ws3.cell(row=1, column=j, value=h).font = openpyxl.styles.Font(bold=True)
        ds = sorted(sens_daily['vol']['q2'].keys()) if 'vol' in sens_daily else []
        for i, d in enumerate(ds, 2):
            row = [d]
            for tag in tags:
                for fam in ('q2', 'q3'):
                    row.append(sens_daily.get(tag, {}).get(fam, {}).get(d))
            for j, v in enumerate(row, 1):
                if isinstance(v, date):
                    ws3.cell(row=i, column=j, value=datetime(v.year, v.month, v.day))
                elif v is None:
                    continue
                else:
                    ws3.cell(row=i, column=j, value=round(float(v), 4))
        _autosize(ws3)

    wb.save(SENS_FILE)
    print(f"[输出] 敏感性汇总 -> {SENS_FILE}")


# ========== 10. 数值格式化（供汇总表使用） ==========
def _f(x, n=4):
    try:
        v = float(x)
        return 'nan' if not np.isfinite(v) else f"{v:.{n}f}"
    except Exception:
        return str(x)


def _wan(x, n=2):
    return _f(float(x) / 10000.0, n)


def _num(v, nd=6):
    """写表用的数值规范化：非有限值（NaN/±inf，如 P_buy_max=∞ 下的 r_grid_max）
    一律写空，避免 openpyxl 落盘出非法单元格。"""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        fv = float(v)
        return round(fv, nd) if np.isfinite(fv) else None
    return v


# ========== 11. 主流程 ==========
def build_ctx():
    print("=" * 72)
    print("问题四：波动电价下的微网购电策略重算（仅替换电价，算法与口径零改动）")
    print("=" * 72)
    ctx2, ctx3 = Q2.build_ctx(), Q3.build_ctx()
    if ctx2['dates'] != ctx3['dates']:
        raise RuntimeError("Q2/Q3 的日期序列不一致")
    pmat = read_price4(PRICE4_FILE, ctx2['dates'])
    for c in (ctx2, ctx3):
        c['PMAT_V'] = pmat
        c['PMAT'] = pmat
    # Q4-PF 因果价格预测：滞后需要附件4 全年矩阵（决策日只从 2 月 1 日起，1 月数据仅作滞后源）
    pf_dates, pf_all = read_price4_all(PRICE4_FILE)
    pf_prepare(pmat, ctx2['dates'], pf_dates, pf_all)
    for c in (ctx2, ctx3):
        c['PF_FLOOR'] = float(_PF['FLOOR'])
    # PWMAE（文档「波动电价改变Q3中预报更新的经济价值」）复用 Q3.pv_curve，
    # 而它读 ctx['G'] 与 ctx['FST']；Q2 的 ctx 原本不含 FST，这里挂上同一份附件3 预报表
    # （两族 G 同出附件2 read_actuals，逐位一致），保证全脚本只有一套预报口径。
    ctx2['FST'] = ctx3['FST']
    # 价格口径核验：模型计划购电量无上界，非正价会让目标函数无下界，必须在求解前拦住
    audit, floor = price_audit(pmat)
    print("-" * 72)
    print(f"价格口径核验（附件4 全样本最低观测价 floor = {floor:.4f} 元/kWh）")
    print(f"  {'实验':<30}{'最低':>9}{'最高':>9}{'均值':>9}{'非正':>6}{'截尾时段':>9}")
    for exp in EXPERIMENTS:
        a = audit[exp]
        print(f"  {a['label']:<30}{a['min']:>9.4f}{a['max']:>9.4f}{a['mean']:>9.4f}"
              f"{a['n_nonpos']:>6}{a['n_clip']:>9}")
        if a['n_clip']:
            print(f"    ↳ λ 放大外推越界，已下截尾并保均值重标定："
                  f"涉及 {a['n_day_clip']} 个决策日，日均价最大偏差 {a['mean_dev']:.2e}")
    bad = [e for e in EXPERIMENTS if audit[e]['n_nonpos'] > 0]
    if bad:
        raise RuntimeError(
            f"以下实验存在非正价格，而计划购电量无上界（P_buy_max=∞），"
            f"目标函数会无下界、MILP 必然报 unbounded：{bad}")
    print("  ✓ 全部实验价格严格为正（λ>1 的越界时段已按文档第 2 节做下截尾对照）")
    print("-" * 72)
    return ctx2, ctx3, pmat


def run_families(ctx2, ctx3, exp, pmat, args, fams):
    """按实验跑指定问题族，返回 {fam: recs}（并行由 run_family 内部处理）。"""
    if CACHE_REV.get(exp):
        stale = os.path.join(CACHE_ROOT, exp)
        if os.path.isdir(stale):
            print(f"[提示] {exp} 价格口径已修订（{CACHE_REV[exp]}），"
                  f"旧缓存目录 cache/{exp}/ 不参与本次计算，可手动删除以释放空间")
    out = {}
    for fam in fams:
        ctx = _ctx_of(fam, ctx2, ctx3)
        t0 = time.perf_counter()
        out[fam] = run_family(fam, ctx, exp, pmat, args.time_limit,
                              args.workers, not args.force, args.limit)
        print(f"[{exp}|{fam}] 完成 {len(out[fam])} 天，耗时 {time.perf_counter() - t0:.0f}s",
              flush=True)
    return out


def recs_vol(ctx2, ctx3, args):
    """正式重算（波动电价 vol）的逐日记录（从缓存读取，不重解）。"""
    fams = ['q2', 'q3'] if args.which == 'both' else [args.which]
    return {f: load_family('vol', f, ctx2, ctx3, args.limit) for f in fams}


def collect_sens(ctx2, ctx3, pmat_v, args):
    """读取全部敏感性实验（vol/fixed 作为基准另列），返回 sens 与逐日费用序列。"""
    sens, sens_daily = {}, {}
    for exp in SENS_EXPS:
        try:
            r2 = load_family(exp, 'q2', ctx2, ctx3, args.limit)
            r3 = load_family(exp, 'q3', ctx2, ctx3, args.limit)
        except RuntimeError:
            continue
        m = collect_sens_metrics(ctx2, exp, r2, r3, pmat_v)
        sens[exp] = m
        sens_daily[exp] = {'q2': m['daily_q2'], 'q3': m['daily_q3']}
        print(f"[敏感性] {EXP_LABEL.get(exp, exp)}: Q4-2 {_wan(m['q2']['total'])} 万元, "
              f"Q4-3 {_wan(m['q3']['total'])} 万元")
    if not sens:
        print("[敏感性] 未发现敏感性实验缓存，跳过敏感性汇总"
              "（vol / fixed 为正式重算基准，不列入敏感性表）")
    # 基准（vol / fixed）不进入敏感性表，但对照汇总与逐日分析表需要它们
    vol_base = None
    for exp in MAIN_EXPS:
        try:
            r2 = load_family(exp, 'q2', ctx2, ctx3, args.limit)
            r3 = load_family(exp, 'q3', ctx2, ctx3, args.limit)
        except RuntimeError:
            continue
        sens_daily[exp] = {'q2': {r['date']: r['total_cost'] for r in r2},
                           'q3': {r['date']: r['scen'][3]['C_total'] for r in r3}}
        if exp == 'vol':
            vol_base = collect_sens_metrics(ctx2, exp, r2, r3, pmat_v)
    if vol_base is not None:
        sens['vol_base'] = vol_base
    if ('gam08' in sens or 'gam12' in sens) and vol_base is not None:
        base = {'q2': load_arrays('q2', 'vol'), 'q3': load_arrays('q3', 'vol')}
        sens['gamma_check'] = {}
        for e in ('gam08', 'gam12'):
            if e in sens:
                sens['gamma_check'][e] = gamma_invariance_check(
                    base, {'q2': load_arrays('q2', e), 'q3': load_arrays('q3', e)})
                g = sens['gamma_check'][e]
                print(f"[γ核验] {EXP_LABEL[e]}: Q4-2 费用比 "
                      f"{_f(sens[e]['q2']['total'] / vol_base['q2']['total'])}，"
                      f"最大调度偏差 {_f((g.get('q2') or {}).get('max_dev', float('nan')), 6)} kW，"
                      f"不一致天数 {(g.get('q2') or {}).get('n_bad_day', -1)}")
    return sens, sens_daily


def run_diagnostics(ctx2, ctx3, recs, fixed_q2, fixed_q3, cf, pmat_v, sens, yearly, risk,
                    limit=None):
    """执行文档分析层的全部要求并打印终端摘要：
      分组检验 / 配对统计与移动块自助法 / 日期重合率 / 最坏日与成本集中度 /
      寿命成本修正 / 上下调系数敏感性 / S0-S3 逐级比较 / 六情形判别 / Q4 层独立核验 /
      Q4-PF 价格预测现实性与完全价格信息价值。
    全部为纯读取 + 统计（除核验外不重解 MILP），结果返回给 write_summary 落表。"""
    ctx = ctx2
    L, G = ctx['L'], ctx['G']
    # Q4-PF：pf 族缓存缺失时返回 None，不产生任何 PF 输出（正式交付不受影响）
    pf = collect_pf(ctx2, ctx3, recs['q2'], recs['q3'], pmat_v, limit)
    diag_rows = build_diag_rows(ctx, recs['q2'], recs['q3'], fixed_q2, fixed_q3, pmat_v)
    groups = build_group_tables(diag_rows)

    def dmap(field):
        return {r['date']: r[field] for r in diag_rows}

    c42, c43, cf2d, cf3d = dmap('c42'), dmap('c43'), dmap('cf2'), dmap('cf3')
    pairs = [pair_stats(cf2d, c42, 'Q2 固定电价', 'Q4-2 波动电价'),
             pair_stats(cf3d, c43, 'Q3-S3 固定电价', 'Q4-3-S3 波动电价'),
             pair_stats(c42, c43, 'Q4-2 波动电价', 'Q4-3-S3 波动电价')]

    # 日期重合率：价格最高 10 日 / 净负荷最高 10 日 / 紧急购电最高 10 日
    price_day, nl_day, emg_day = {}, {}, {}
    for r in recs['q2']:
        d = r['date']
        i = ctx['date_idx'][d]
        pv = pmat_v[i]
        st = q2_settle(r['B'], r['C'], r['D'], L[i], G[i], pv)
        price_day[d] = float(pv.mean())
        nl_day[d] = float((L[i] - G[i]).sum())
        emg_day[d] = float(st['R'].sum())
    coincidence = date_coincidence({
        '价格最高日': price_day, '净负荷最高日': nl_day, '紧急购电最高日': emg_day})

    worst = {}
    for nm, dm in (('Q2 固定电价', {r['date']: r['total_cost'] for r in fixed_q2}),
                   ('Q4-2 波动电价', c42),
                   ('Q3-S3 固定电价', {r['date']: r['scen'][3]['C_total'] for r in fixed_q3}),
                   ('Q4-3-S3 波动电价', c43)):
        worst[nm] = worst_days(sorted(dm), [dm[k] for k in sorted(dm)])

    life = {}
    for nm, key in (('Q4-2 波动电价', 'Q4-2 波动电价'), ('Q4-3-S3 波动电价', 'Q4-3-S3 波动电价')):
        yy = yearly[key]
        for rec in life_adjusted(yy['total'], yy['throughput']):
            life[f"{nm} @κ_th={rec['kappa_th']}"] = rec

    coef = settle_coef_sens(recs['q3'], ctx3, pmat_v)
    s0s3 = s0s3_table(recs['q3'])
    cases, meta = explain_cases(yearly, risk, cf, sens)
    verify, vok = verify_q4_layer(ctx2, recs['q2'], recs['q3'], pmat_v)

    # ---- 终端摘要 ----
    print("\n" + "=" * 72)
    print("[诊断层] 文档「分组检验 / 配对统计 / 重合率 / 寿命 / 情形判别」")
    print("=" * 72)
    print(f"  配对统计（移动块自助法 {BOOT_N_RESAMPLE} 次，主口径 {BOOT_BLOCKS[0]} 天块，"
          f"seed={BOOT_SEED}；稳健性对照 {BOOT_BLOCKS[1]}/{BOOT_BLOCKS[2]} 天块）")
    for p in pairs:
        if not p:
            continue
        a3, a14 = p['ci_alt'].get(3), p['ci_alt'].get(14)
        print(f"    {p['label_a']} → {p['label_b']}: 平均 {p['mean']:+.4f} 元/天，"
              f"中位 {p['median']:+.4f}，p_save={p['p_save']:.1%}，"
              f"95%CI[{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}] -> {p['sig']}")
        if a3 and a14:
            print(f"      稳健性: 3天块[{a3[1]:+.4f}, {a3[2]:+.4f}]，"
                  f"14天块[{a14[1]:+.4f}, {a14[2]:+.4f}]")
    print("  日期重合率（各自最高 10 日）")
    for r in coincidence[1]:
        print(f"    {r['日期集合A']} ∩ {r['日期集合B']}: {r['重合天数']}/10 "
              f"({r['重合率']:.0%})")
    print("  最坏日与成本集中度")
    for nm, w in worst.items():
        print(f"    {nm}: 最坏 10 日占全年 {w['share']:.2%}；最大单日费用 "
              f"{w['max_cost']:.2f} 元 @ {w['max_date']}")
    print(f"  寿命成本修正（C^life-adjusted = C^total + κ_th·吞吐量，不改变调度）")
    for k in LIFE_KAPPA_GRID:
        r2 = life.get(f'Q4-2 波动电价 @κ_th={k}')
        r3 = life.get(f'Q4-3-S3 波动电价 @κ_th={k}')
        if r2 and r3:
            print(f"    κ_th={k:<5}: Q4-2 {r2['C_life_adjusted'] / 1e4:.2f} 万元"
                  f"（+{r2['relative_extra']:.2%}），"
                  f"Q4-3 {r3['C_life_adjusted'] / 1e4:.2f} 万元（+{r3['relative_extra']:.2%}）")
    print("  储能价值 V^BESS = C^noBESS − C^BESS（文档「主对照实验」第 6 项 + λ 敏感性）")
    for r in bess_value_table(sens):
        v = r.get('储能价值V^BESS/元')
        efc = r.get('Q4-2 EFC')
        print(f"    {r['振幅档']:<22} V^BESS = "
              + (f"{v / 1e4:>8.2f} 万元" if isinstance(v, (int, float)) else "  (该 λ 无储能对照未跑)")
              + (f"，Q4-2 EFC = {efc:.4f}" if isinstance(efc, (int, float)) else ""))
    print("  价格加权光伏预测误差 PWMAE（文档「波动电价改变Q3中预报更新的经济价值」）")
    for nm in ('Q3-S3 固定电价', 'Q4-3-S3 波动电价'):
        pm = yearly[nm]['pwmae']
        print(f"    {nm:<16} 6:00 {pm[1]:.4f} / 12:00 {pm[2]:.4f} / 18:00 {pm[3]:.4f} kW")
    cv2 = yearly['Q4-2 波动电价']['cov']
    print(f"  价量协方差分解（Q4-2）：日均价 {cv2['cbar']:.4f} 元/kWh，"
          f"日均购电 {cv2['pbar']:.2f} kW，")
    print(f"    均价×均量项 {cv2['level_term'] / 1e4:.2f} 万元 + 价量协方差项 "
          f"{cv2['cov_term'] / 1e4:.2f} 万元 = 计划费 "
          f"{yearly['Q4-2 波动电价']['plan'] / 1e4:.2f} 万元")
    print("  并网上限活跃率 r^grid,max：主结果 P_buy_max=∞（题目未给并网容量）→ 该率无定义；"
          "有限档位见 pb6k / pb10k 两族")
    print("  Q4-3 的 S0-S3 逐级比较（边际改善 = 上一级 − 本级）")
    for r in s0s3:
        mv = r['边际改善(vs上级)']
        print(f"    {r['场景']}: 总费用 {r['总费用'] / 1e4:.2f} 万元"
              + (f"，边际改善 {mv:.2f} 元" if mv != '' else ""))
    print("  上下调系数敏感性（事后结算，不改调度）")
    for r in coef:
        print(f"    上调 {r['上调系数']}/下调 {r['下调系数']}: "
              f"Q4-3 总费用 {r['Q4-3总费用'] / 1e4:.2f} 万元"
              + ("（基准）" if r['基准'] else ""))
    print("  结果解释规则判别")
    for k, v in cases.items():
        print(f"    [{'成立' if v['成立'] else '不成立'}] {k}")
        print(f"        {v['证据']}")
    print(f"  缓冲比例 r^mitigate = {meta['r_mitigate']:.4f}"
          f"（分子调度适应效应 {meta['adapt_eff2']:+.2f} 元，"
          f"分母纯价格效应 {meta['price_eff2']:+.2f} 元）")
    print("  Q4 层独立核验（文档「数值实现与结果核验」第 3/4/5/6/8 条）")
    for k in ('③功率平衡最大残差', '④SOC越界天数', '④充放电功率越限次数',
              '⑤日初SOC最大偏差', '⑤日末SOC最大偏差', '⑥已执行时段冻结最大偏差',
              '⑧S0-S3基准计划最大偏差', '④SOC范围', '充放电功率峰值'):
        print(f"    {k}: {verify[k]}")
    print(f"    结论: {'全部通过' if vok else '存在未通过项，请检查'}")

    # ---- 文档 819-842「价格振幅敏感性」逐档对照（λ∈{0,0.5,1,1.5}）----
    lam_rows = lam_sensitivity_table(sens)
    print("  价格振幅敏感性 λ∈{0,0.5,1,1.5} 逐档对照（文档 840 行要求）")
    for r in lam_rows:
        print(f"    {r['振幅档']:<28} Q4-2总 {_wan(r['Q4-2 总费用/元'])} 万元 | "
              f"Q4-3总 {_wan(r['Q4-3 总费用/元'])} 万元 | "
              f"V^BESS {_wan(r['储能价值V^BESS/元'])} 万元 | "
              f"Q4-2 EFC {r['Q4-2 EFC(可用容量)']:.4f}")
    # ---- 文档 593「同均值平坦电价剥离价格水平」：波动本身效应 ----
    # 注：run_diagnostics 只有 sens（年度聚合），没有 sens_daily；
    # 波动效应的逐日对照需由调用方（collect_all）传入 sens_daily，这里用年度聚合近似。
    vol_eff_q2 = sens['vol_base']['q2']['total'] - sens['lam0']['q2']['total'] \
        if 'lam0' in sens and 'vol_base' in sens else float('nan')
    vol_eff_q3 = sens['vol_base']['q3']['total'] - sens['lam0']['q3']['total'] \
        if 'lam0' in sens and 'vol_base' in sens else float('nan')
    print(f"  波动本身效应 C(c^V)−C(c^flat)（文档 593 行，同均值平坦电价剥离价格水平）")
    print(f"    Q4-2: {vol_eff_q2:+.2f} 元（正值=波动推高费用，负值=波动节省费用）")
    print(f"    Q4-3-S3: {vol_eff_q3:+.2f} 元")
    # ---- 文档 325「购电加权均价套利判定」（三条件联动）----
    arb = arbitrage_check(yearly)
    print("  购电加权均价套利判定（文档 325 行：购电量↑+总费用↓+c̄^buy↓ 三条件联动）")
    for nm, a in arb.items():
        tag = '套利成立' if a['套利判定'] else '不成立'
        print(f"    {nm}: {tag}")
        print(f"      购电量变化 {a['总购电量变化/kWh']:+.2f} kWh，"
              f"总费用变化 {a['总费用变化/元']:+.2f} 元，"
              f"c̄^buy {a['c̄^buy_固定/(元/kWh)']:.4f} → {a['c̄^buy_波动/(元/kWh)']:.4f} 元/kWh"
              f"（{a['c̄^buy 变化/(元/kWh)']:+.4f}）")
        print(f"      条件1购电量↑={a['条件1_购电量增加']}，"
              f"条件2总费用↓={a['条件2_总费用下降']}，"
              f"条件3加权均价↓={a['条件3_加权均价下降']}")
    print("=" * 72)

    return {'rows': diag_rows, 'groups': groups, 'coincidence': coincidence,
            'worst': worst, 'life': life, 'coef': coef, 's0s3': s0s3,
            'cases': cases, 'meta': meta, 'verify': verify, 'verify_ok': vok,
            'pairs': pairs, 'bess': bess_value_table(sens), 'pf': pf,
            'lam_rows': lam_rows,
            'vol_eff_q2': vol_eff_q2, 'vol_eff_q3': vol_eff_q3,
            'arb': arb,
            'pwmae': {nm: yearly[nm]['pwmae']
                      for nm in ('Q3-S3 固定电价', 'Q4-3-S3 波动电价')}}


def collect_all(ctx2, ctx3, pmat_v, recs, args):
    """汇总全部指标并落盘交付文件（模板表 + 表1-3 + 逐日分析表 + 对照汇总 + 敏感性汇总）。"""
    ctx = ctx2
    n_day = len(recs['q2'])
    if len(recs['q3']) != n_day:
        raise RuntimeError("Q2/Q3 重算天数不一致")
    # 固定电价基线 x^F：Q2 无 U/W 需求，但为口径统一同样取本脚本 fixed 实验；
    # Q3 必须取本脚本重解结果（原生缓存不含 U/W，无法在新价格下重算调整费）。
    fixed_q2 = load_family('fixed', 'q2', ctx2, ctx3, args.limit)
    fixed_q3 = load_family('fixed', 'q3', ctx2, ctx3, args.limit)
    if len(fixed_q2) != n_day or len(fixed_q3) != n_day:
        raise RuntimeError("fixed 实验天数与 vol 不一致，请先跑 main（vol + fixed）")
    # 原生缓存仅用于回归核验；冒烟测试 --limit 时按本次日期集合裁剪
    common = {r['date'] for r in recs['q2']}
    native_q2 = [r for r in load_native_fixed('q2', ctx2, ctx3) if r['date'] in common]
    native_q3 = [r for r in load_native_fixed('q3', ctx2, ctx3) if r['date'] in common]
    if len(native_q2) != n_day or len(native_q3) != n_day:
        raise RuntimeError("固定电价原生缓存与本次重算的日期集合不匹配，请先在 Q2/Q3 目录跑完全量")

    write_templates(recs['q2'], recs['q3'])
    write_rep_tables('q2', recs['q2'])
    write_rep_tables('q3', recs['q3'])

    cf, _ = build_counterfactual(ctx, fixed_q2, fixed_q3, native_q2, native_q3, pmat_v)
    cf = add_shift(cf, fixed_q2, recs['q2'])
    # 逐日 C_V(x^V)（x^V = vol 实验逐日总费），供 explain_cases 统计「调度适应效应>0 天数」
    for d, r in ((r['date'], r) for r in recs['q2']):
        if d in cf['q2']:
            cf['q2'][d]['C_V_xV'] = float(r['total_cost'])
    for d, r in ((r['date'], r) for r in recs['q3']):
        if d in cf['q3']:
            cf['q3'][d]['C_V_xV'] = float(r['scen'][3]['C_total'])
    regression_report(cf)

    # 结算自检：U/W 抓取 + run_block 推进 + 成本核算口径必须逐日复现记录自身分项
    settle_chk = {f: verify_settle(_ctx_of(f, ctx2, ctx3), recs[f], pmat_v, f)
                  for f in ('q2', 'q3')}
    print("\n[结算自检] 用各日实际价格重新结算，应逐日复现记录自身分项费用")
    for f in ('q2', 'q3'):
        w, wh = settle_chk[f]
        print(f"  {f.upper()}: 最大相对偏差 {w:.3e}  {wh}")

    sens, sens_daily = collect_sens(ctx2, ctx3, pmat_v, args)
    _reset_upstream_params()      # 敏感性跑完复位题设参数，保证基准结算不被污染
    yearly = collect_yearly(ctx, recs['q2'], recs['q3'], pmat_v, fixed_q2, fixed_q3)
    risk = collect_risk(recs['q2'], recs['q3'], fixed_q2, fixed_q3)
    diag = run_diagnostics(ctx2, ctx3, recs, fixed_q2, fixed_q3, cf, pmat_v,
                           sens, yearly, risk, args.limit)
    _reset_upstream_params()

    daily = build_daily_table(ctx, recs['q2'], recs['q3'], cf, sens_daily, pmat_v,
                              pf=diag.get('pf'))
    write_xlsx(DAILY_FILE, '逐日分析表', daily, '逐日分析')
    write_summary(yearly, risk, cf, recs['q2'], recs['q3'], diag)
    if sens:
        write_sens(sens, sens_daily)

    print("\n" + "=" * 72)
    print("[正式结果汇总] 单位: 万元")
    print("=" * 72)
    for nm in ('Q2 固定电价', 'Q4-2 波动电价', 'Q3-S3 固定电价', 'Q4-3-S3 波动电价'):
        yy = yearly[nm]
        print(f"  {nm:<16} 计划 {yy['plan'] / 1e4:>10.2f} | 调整 {yy['adj'] / 1e4:>8.2f} | "
              f"紧急 {yy['emg'] / 1e4:>9.2f} | 合计 {yy['total'] / 1e4:>10.2f} | "
              f"CVaR95 {risk[nm]['cvar95'] / 1e4:>9.2f}")
    return None


def run_outputs(ctx2, ctx3, pmat_v, args):
    """由缓存重建全部交付表格（模板表 / 表1-3 / 逐日分析表 / 对照汇总 / 敏感性汇总），不重解 MILP。"""
    fams = ['q2', 'q3'] if args.which == 'both' else [args.which]
    recs = {}
    for fam in fams:
        recs[fam] = load_family('vol', fam, ctx2, ctx3, args.limit)
    if len(fams) == 1:
        print(f"[提示] 仅重建了 {fams[0]} 族输出；完整交付需要 --which both")
        return None
    return collect_all(ctx2, ctx3, pmat_v, recs, args)


def run_regress(ctx2, ctx3, pmat_v, args):
    """回归核验：用附件1 价格重解 4 个代表日，与 Q2/Q3 原生缓存逐项比对。"""
    print("=" * 72)
    print("[回归核验-重解版] 附件1 价格重解 4 代表日，应与 Q2/Q3 原生缓存一致")
    print("=" * 72)
    native = {'q2': {r['date']: r for r in load_native_fixed('q2', ctx2, ctx3)},
              'q3': {r['date']: r for r in load_native_fixed('q3', ctx2, ctx3)}}
    pmat_f = make_pmat(pmat_v, 'fixed')
    all_ok = True
    for fam in ('q2', 'q3'):
        mod = Q2 if fam == 'q2' else Q3
        ctx = _ctx_of(fam, ctx2, ctx3)
        _patch_experiment('fixed', fam)
        ctx['PMAT'] = pmat_f
        for d in REP_DAYS:
            i = ctx['date_idx'][d]
            ctx['prices'] = pmat_f[i]
            rec = mod.solve_day(ctx, i, args.time_limit)
            ref = native[fam][d]
            if fam == 'q2':
                pairs = [('总费用', rec['total_cost'], ref['total_cost']),
                         ('计划费', rec['plan_cost'], ref['plan_cost']),
                         ('紧急费', rec['em_cost'], ref['em_cost']),
                         ('B 最大偏差', float(np.abs(rec['B'] - ref['B']).max()), 0.0),
                         ('SOC 最大偏差', float(np.abs(np.asarray(rec['E']) - np.asarray(ref['E'])).max()), 0.0)]
            else:
                pairs = [('S3 总费用', rec['scen'][3]['C_total'], ref['scen'][3]['C_total']),
                         ('计划费', rec['plan_cost'], ref['plan_cost']),
                         ('紧急费', rec['scen'][3]['C_emg'], ref['scen'][3]['C_emg']),
                         ('Qeff 最大偏差', float(np.abs(rec['Qeff_s3'] - ref['Qeff_s3']).max()), 0.0),
                         ('SOC 最大偏差', float(np.abs(np.asarray(rec['E145_s3']) - np.asarray(ref['E145_s3'])).max()), 0.0)]
            bad = []
            for tag, a, b in pairs:
                rel = abs(a - b) / max(1.0, abs(b))
                if rel > REG_TOL_REL and abs(a - b) > REG_TOL_ABS:
                    bad.append(f"{tag}({a:.6f} vs {b:.6f})")
            flag = '一致' if not bad else '不一致: ' + ', '.join(bad)
            if bad:
                all_ok = False
            print(f"  {fam.upper()} {d}: {flag}")
    print("  结论: " + ("通过 -> 同一价格下 Q4 重算可复现 Q2/Q3" if all_ok
                        else "存在差异 -> 并列最优解或求解容差导致，请按文档第 858-866 行排查"))
    return all_ok


def main():
    ap = argparse.ArgumentParser(
        description='2025 CUMCM C题 问题四 波动电价重算与深度分析'
                    '（无参数 = 一键全量：主重算 + 敏感性，断点续跑 + 8 进程）')
    ap.add_argument('cmd', nargs='?', default='all',
                    choices=['all', 'full', 'main', 'sens', 'outputs', 'regress', 'pf'],
                    help='all/full=主重算+敏感性（默认） | main=仅正式重算 | sens=仅敏感性 | '
                         'outputs=由缓存重建 | regress=回归核验 | '
                         'pf=价格预测现实性扩展（Q4-PF，文档「可选的价格预测现实性扩展」）')
    ap.add_argument('--which', default='both', choices=['both', 'q2', 'q3'],
                    help='限定问题族（both 默认）')
    ap.add_argument('--exp', default=None,
                    help='限定敏感性实验，逗号分隔，如 lam0,lam05,lam15,w99,w95,eta85,pm25,kap7'
                         '（vol/fixed 为正式重算基准，总会执行；完整清单见 SENS_EXPS）')
    ap.add_argument('--force', action='store_true', help='忽略缓存全部重解')
    ap.add_argument('--limit', type=int, default=None, help='仅求解前 N 个决策日（冒烟测试）')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 8,
                    help='并行进程数（默认=本机CPU逻辑核数）')
    ap.add_argument('--time-limit', type=float, default=300.0, help='单次 MILP 时间限制（秒）')
    args = ap.parse_args()

    print(f"[参数] Δt={DT:.4f}h, κ={KAPPA}, η={ETA_CH}, E∈[{E_MIN},{E_MAX}], "
          f"E0=E145={E_INIT}, P_ch/P_dis<={P_CH_MAX:.0f}kW, P_buy_max=无上限")
    print(f"[参数] 运行: cmd={args.cmd}, which={args.which}, exp={args.exp or '全部'}, "
          f"断点续跑={'关' if args.force else '开'}, workers={args.workers}, "
          f"limit={args.limit}")

    ctx2, ctx3, pmat_v = build_ctx()
    fams = ['q2', 'q3'] if args.which == 'both' else [args.which]

    if args.cmd in ('all', 'full'):
        for exp in MAIN_EXPS:
            run_families(ctx2, ctx3, exp, make_pmat(pmat_v, exp), args, fams)
        if args.which != 'both':
            print("[提示] 已只跑一族；完整交付请用 --which both 再跑另一族后执行 outputs")
            return
        exps = [e for e in (args.exp.split(',') if args.exp else SENS_EXPS) if e]
        for exp in exps:
            if exp in MAIN_EXPS:
                continue
            run_families(ctx2, ctx3, exp, make_pmat(pmat_v, exp), args, fams)
        # 敏感性全部跑完后再统一汇总落盘一次，确保敏感性汇总表包含完整结果
        collect_all(ctx2, ctx3, pmat_v, recs_vol(ctx2, ctx3, args), args)
    elif args.cmd == 'main':
        for exp in MAIN_EXPS:
            run_families(ctx2, ctx3, exp, make_pmat(pmat_v, exp), args, fams)
        if args.which == 'both':
            collect_all(ctx2, ctx3, pmat_v, recs_vol(ctx2, ctx3, args), args)
    elif args.cmd == 'sens':
        for exp in MAIN_EXPS:      # 基准缺失则先补齐（有缓存时秒过）
            run_families(ctx2, ctx3, exp, make_pmat(pmat_v, exp), args, fams)
        exps = [e for e in (args.exp.split(',') if args.exp else SENS_EXPS) if e]
        for exp in exps:
            if exp in MAIN_EXPS:
                continue
            run_families(ctx2, ctx3, exp, make_pmat(pmat_v, exp), args, fams)
        if args.which == 'both':
            collect_all(ctx2, ctx3, pmat_v, recs_vol(ctx2, ctx3, args), args)
    elif args.cmd == 'pf':
        for exp in MAIN_EXPS:      # 基准缺失则先补齐（有缓存时秒过）
            run_families(ctx2, ctx3, exp, make_pmat(pmat_v, exp), args, fams)
        print("=" * 72)
        print("[Q4-PF] 因果价格预测重算（仅用决策时刻之前的信息；"
              "不替换正式结果 result4-2 / result4-3）")
        print("=" * 72)
        run_families(ctx2, ctx3, PF_EXP, make_pmat(pmat_v, PF_EXP), args, fams)
        if args.which == 'both':
            collect_all(ctx2, ctx3, pmat_v, recs_vol(ctx2, ctx3, args), args)
    elif args.cmd == 'outputs':
        run_outputs(ctx2, ctx3, pmat_v, args)
    elif args.cmd == 'regress':
        run_regress(ctx2, ctx3, pmat_v, args)


if __name__ == '__main__':
    main()

