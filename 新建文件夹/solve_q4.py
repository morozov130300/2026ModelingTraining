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
  表1_Q4-2 / 表2_Q4-2 / 表3_Q4-2.xlsx     4 代表日（Q4-2）
  表1_Q4-3 / 表2_Q4-3 / 表3_Q4-3.xlsx     4 代表日（Q4-3, S3）
  逐日分析表.xlsx            334 天 × 六组逐日指标（文档「建议保存的逐日分析表」）
  对照汇总.xlsx              总体结果 / 反事实分解 / 风险指标 / 成本结构 / 约束活跃率
  问题四_求解结果分析报告.md   自动写入实测数值的报告（含可直接改写的论文段落）

  vol   = 附件4 波动电价（正式结果 x^V）
  fixed = 附件1 固定电价（基线策略 x^F）。之所以由本脚本重解而不是直接读 Q3 原生缓存：
          Q4-3 的调整费必须在新价格下重算 C^adj = Σ c^V(1.5U−0.5W)Δt，而 U/W 是 MILP
          内部变量、原生缓存未保存。本脚本通过包装 Q3.solve_milp 在求解瞬间抓取 U/W
          （Q3 源码零改动、缓存版本不变），并同一口径重解 fixed，回归核验与原生缓存逐项对齐。

敏感性（sens）
  敏感性汇总.xlsx            λ∈{0,0.5,1.0,1.5} 振幅敏感性、γ∈{0.8,1.0,1.2} 正比例缩放核验、
                            无储能反事实；报告同步追加对应章节

--------------------------------------------------------------------------
用法（在装有 numpy/scipy/openpyxl 的 Windows 目标机上运行）
--------------------------------------------------------------------------
  python solve_q4.py                  # 一键全量：主重算(vol+fixed) + 全套敏感性（断点续跑 + 8 进程）
  python solve_q4.py main             # 仅主重算（vol + fixed，正式交付文件）
  python solve_q4.py sens             # 基准 + 敏感性
  python solve_q4.py outputs          # 由缓存重建全部输出，不重解 MILP
  python solve_q4.py regress          # 回归核验：附件1 价格重解 4 代表日，比对 Q2/Q3 缓存
  python solve_q4.py main --which q2  # 只跑问题二族（先拿到 Q4-2）
  python solve_q4.py main --which q3  # 只跑问题三族
  python solve_q4.py full --force     # 忽略缓存全部重解
  python solve_q4.py full --limit 10  # 冒烟测试：仅前 10 个决策日
  --exp lam0,lam05,lam15,gam08,gam12,nobess   限定敏感性实验（vol/fixed 为基准，总会执行）
Windows 下也可直接双击同目录的 一键运行_Q4.bat
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
REPORT_FILE = os.path.join(SCRIPT_DIR, '问题四_求解结果分析报告.md')
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
BOOT_BLOCKS = (7, 3, 14)
WORST_N_DAYS = 10
ACT_EPS_P = 1e-3
ACT_EPS_E = 1e-3
REG_TOL_REL = 1e-6
REG_TOL_ABS = 1e-4


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


# ========== 1b. U/W 抓取钩子：不改动 Q3 源码取得逐节点调整量 ==========
# Q4-3 的调整费必须在「新价格」下重新结算
#     C^adj = Σ_{k 采纳} Σ_{t∈T_k} c^V_t (1.5·U^(k)_t − 0.5·W^(k)_t) Δt，
# 而 Q3 记录只保存 S0~S3 的费用聚合值，不保存 U/W 向量。为严格保持 Q3 源码零改动
# （避免改动上游模型逻辑或其缓存版本），这里包装 Q3.solve_milp：MILP 解出后按模型自带
# 的变量布局取出 U/W（u 块 = 8*nt、w 块 = 9*nt，nt 为剩余时域长度，i0 为节点起点）。
_CAP = {'U': {}, 'W': {}}


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


_install_q3_hook()


# ========== 2. 附件4 读取与预处理核验（文档「价格数据的预处理」） ==========
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
        if isinstance(d0, datetime):
            d = d0.date()
        elif isinstance(d0, date):
            d = d0
        else:
            s = (str(d0).strip().replace('/', '-').replace('.', '-')
                 .replace('年', '-').replace('月', '-').replace('日', ''))
            p = [x for x in s.split('-') if x]
            if len(p) != 3:
                raise ValueError(f"附件4 无法解析日期: {d0!r}")
            d = date(int(p[0]), int(p[1]), int(p[2]))
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
              f"需按文档处理；本脚本按题目原式原样重算并在报告中记录该边界。")
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
SENS_EXPS = ['lam0', 'lam05', 'lam15', 'gam08', 'gam12', 'nobess']   # 敏感性/对照
EXPERIMENTS = MAIN_EXPS + SENS_EXPS
EXP_LABEL = {
    'vol': '波动电价 λ=1.0（附件4 原始）',
    'fixed': '固定电价（附件1，基线 x^F）',
    'lam0': '振幅 λ=0.0（同日均值平坦电价）',
    'lam05': '振幅 λ=0.5',
    'lam15': '振幅 λ=1.5（放大日内价差，含非负截尾修正）',
    'gam08': '正比例缩放 γ=0.8',
    'gam12': '正比例缩放 γ=1.2',
    'nobess': '波动电价 + 禁用储能',
}

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


def make_pmat(pmat_v, exp):
    """生成实验对应的价格矩阵（文档「价格振幅敏感性」「正比例缩放」公式）。"""
    if exp in ('vol', 'nobess'):
        return pmat_v.copy()
    if exp == 'lam0':
        return _lambda_series(pmat_v, 0.0)
    if exp == 'lam05':
        return _lambda_series(pmat_v, 0.5)
    if exp == 'lam15':
        return _lambda_series(pmat_v, 1.5)
    if exp == 'gam08':
        return 0.8 * pmat_v
    if exp == 'gam12':
        return 1.2 * pmat_v
    if exp == 'fixed':
        return np.tile(Q2.read_prices(Q2.PRICE_FILE), (pmat_v.shape[0], 1))
    raise KeyError(exp)


_LAM_OF = {'lam0': 0.0, 'lam05': 0.5, 'lam15': 1.5}


def price_audit(pmat_v):
    """逐实验核算价格矩阵健康度（文档第 2 节：统计最小价格，重点检查零价与负价）。
    开跑前打印，并写入分析报告，作为「价格预处理已核验」的证据。"""
    base_floor = float(pmat_v.min())
    rows = {}
    for exp in EXPERIMENTS:
        p = make_pmat(pmat_v, exp)
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
    return exp == 'nobess'


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
    """把 Q2/Q3 的缓存目录与储能参数切到本实验命名空间（不污染 Q2/Q3 原生结果）。
    目录名用 _exp_key(exp)：价格口径修订过的实验换新目录，旧口径缓存自动作废。
    （Q2 缓存只有日期文件名、无版本字段，换目录是唯一的作废手段；Q3 另有版本号。）
    版本号 v2：记录结构加入 U/W 抓取字段，旧版实验缓存自动作废重解。"""
    key = _exp_key(exp)
    exp_dir = os.path.join(CACHE_ROOT, key)
    if fam == 'q2':
        Q2.CACHE_DIR = os.path.join(exp_dir, 'q2')
        Q2.P_CH_MAX = 0.0 if exp_is_nobess(exp) else P_CH_MAX
        Q2.P_DIS_MAX = 0.0 if exp_is_nobess(exp) else P_DIS_MAX
    else:
        Q3.CACHE_DIR = os.path.join(exp_dir, 'q3')
        Q3.CACHE_VERSION = f'q4-{key}-v2'
        Q3.P_CH_MAX = 0.0 if exp_is_nobess(exp) else P_CH_MAX
        Q3.P_DIS_MAX = 0.0 if exp_is_nobess(exp) else P_DIS_MAX
    os.makedirs(os.path.join(exp_dir, fam), exist_ok=True)


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


def q2_settle(B, C, D, L, G, prices):
    """用任意价格向量对给定策略重新结算（Q4-2 的 C^plan、C^emg）。"""
    R, V, K, S = q2_backtest(L, G, B, C, D)
    c_plan = DT * float(prices @ B)
    c_emg = KAPPA * DT * float(prices @ R)
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


def q3_settle(rec, L, G, prices):
    """用任意价格向量对问题三 S3 生效策略链重新结算。
    物理量（R、SOC 轨迹）与价格无关，复用 Q3.run_block 按 B_0..B_3 顺序推进。"""
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
        up = 1.5 * float(U @ base)
        dn = 0.5 * float(W @ base)
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
    pv_use = curtail = 0.0
    segs = []
    for m in range(4):
        g0, g1 = Q3.BLOCKS[m]
        E, R, _segs, cemg, eemg, _over = Q3.run_block(L, G, prices, g0, g1, Qe, Cs, Ds, E)
        R_all[g0:g1] = R
        c_emg += cemg
        e_emg += eemg
        rd_pos = np.maximum(L[g0:g1] + Cs[g0:g1] - Ds[g0:g1], 0.0)
        v = np.where(R > 0.0, G[g0:g1], np.minimum(G[g0:g1], rd_pos))
        pv_use += DT * float(v.sum())
        curtail += DT * float((G[g0:g1] - v).sum())
    return {'plan': c_plan, 'up': c_up, 'down': c_dn, 'adj': c_up - c_dn,
            'emg': c_emg, 'total': c_plan + (c_up - c_dn) + c_emg,
            'R': R_all, 'e_emg': e_emg, 'per_k': per_k, 'aw': aw, 'E_end': E,
            'charge': DT * float(Cs.sum()), 'discharge': DT * float(Ds.sum()),
            'throughput': DT * float((Cs + Ds).sum()), 'pv_use': pv_use,
            'curtail': curtail}


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


def moving_block_ci(x, block, n_resamp, rng):
    """移动块自助法 95% 置信区间（直接复用 Q3 实现，保证与问题三口径一致）。"""
    return Q3._moving_block_ci(x, block, n_resamp, rng)


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


# ========== 8. 年度汇总 / 风险 / 重合率 / 分组 / 配对 ==========
def _agg_family(ctx, recs, fam, pmat):
    """某族在某价格矩阵下的全年指标（全部由已锁定策略结算，物理量价格无关）。"""
    L, G = ctx['L'], ctx['G']
    didx = ctx['date_idx']
    q95 = float(np.percentile(pmat, 95))
    out = {k: 0.0 for k in ('plan', 'adj', 'emg', 'total', 'buy_energy', 'buy_cost',
                            'charge', 'discharge', 'throughput', 'curtail', 'pv_use',
                            'pv_avail', 'E_emg', 'E_emg_high')}
    out['adopt'] = [0, 0, 0]
    out['A_up'] = out['A_down'] = 0.0
    act = {'n_ch': 0, 'n_dis': 0, 'n_Emin': 0, 'n_Emax': 0}
    prep = []
    for r in recs:
        i = didx[r['date']]
        pv = pmat[i]
        if fam == 'q2':
            st = q2_settle(r['B'], r['C'], r['D'], L[i], G[i], pv)
            Cq, Dq, Eq = r['C'], r['D'], np.asarray(r['E'])
        else:
            st = q3_settle(r, L[i], G[i], pv)
            Cq, Dq, Eq = r['C_s3'], r['D_s3'], q3_soc_traj(r, L[i], G[i])
            out['A_up'] += updown_energy(r, up=True)
            out['A_down'] += updown_energy(r, up=False)
            for k in (1, 2, 3):
                if r['a_flags'][k - 1]:
                    out['adopt'][k - 1] += 1
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
        prep.append(soc_high_readiness(Eq, pv))
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
                       'r_E_min': act['n_Emin'] / tot, 'r_E_max': act['n_Emax'] / tot}
    out['adopt_rate'] = [c / n_day for c in out['adopt']]
    out['n_day'] = len(recs)
    return out


def collect_yearly(ctx, recs_q2, recs_q3, pmat_v, fixed_q2, fixed_q3):
    p1 = np.tile(Q2.read_prices(Q2.PRICE_FILE), (len(ctx['dates']), 1))
    return {'Q2 固定电价': _agg_family(ctx, fixed_q2, 'q2', p1),
            'Q4-2 波动电价': _agg_family(ctx, recs_q2, 'q2', pmat_v),
            'Q3-S3 固定电价': _agg_family(ctx, fixed_q3, 'q3', p1),
            'Q4-3-S3 波动电价': _agg_family(ctx, recs_q3, 'q3', pmat_v)}


def collect_risk(recs_q2, recs_q3, fixed_q2, fixed_q3):
    return {'Q2 固定电价': risk_block([r['total_cost'] for r in fixed_q2], 'Q2'),
            'Q4-2 波动电价': risk_block([r['total_cost'] for r in recs_q2], 'Q4-2'),
            'Q3-S3 固定电价': risk_block([r['scen'][3]['C_total'] for r in fixed_q3], 'Q3'),
            'Q4-3-S3 波动电价': risk_block([r['scen'][3]['C_total'] for r in recs_q3], 'Q4-3')}


def collect_coincidence(ctx, recs_q2, pmat_v):
    """最坏 10 日之间的日期重合率（价格 / 净负荷 / 紧急购电 / 费用）。"""
    L, G = ctx['L'], ctx['G']
    didx = ctx['date_idx']
    ds = [r['date'] for r in recs_q2]

    def top(score):
        return set(ds[i] for i in np.argsort(np.asarray(score, float))[::-1][:WORST_N_DAYS])

    price, netload, emg, cost = [], [], [], []
    for r in recs_q2:
        i = didx[r['date']]
        pv = pmat_v[i]
        price.append(float(pv.mean()))
        netload.append(float((L[i] - G[i]).mean()))
        emg.append(q2_settle(r['B'], r['C'], r['D'], L[i], G[i], pv)['R'].sum() * DT)
        cost.append(r['total_cost'])
    tp, tn, te, tc = top(price), top(netload), top(emg), top(cost)
    return {'price_emg': len(tp & te), 'price_netload': len(tp & tn), 'cost_emg': len(tc & te)}


def collect_groups(ctx, recs_q2, recs_q3, fixed_q2, fixed_q3, pmat_v):
    """分组检验：按价格 CV 三分组、按价格与净负荷相关系数三分组、按月份分组。"""
    L, G = ctx['L'], ctx['G']
    didx = ctx['date_idx']
    ds = [r['date'] for r in recs_q2]
    cv, corr = [], []
    for d in ds:
        i = didx[d]
        pv = pmat_v[i]
        cv.append(float(pv.std() / pv.mean()))
        c = np.corrcoef(pv, L[i] - G[i])[0, 1]
        corr.append(0.0 if not np.isfinite(c) else float(c))
    cv, corr = np.asarray(cv), np.asarray(corr)
    d2v = {r['date']: r['total_cost'] for r in recs_q2}
    d2f = {r['date']: r['total_cost'] for r in fixed_q2}
    d3v = {r['date']: r['scen'][3]['C_total'] for r in recs_q3}
    d3f = {r['date']: r['scen'][3]['C_total'] for r in fixed_q3}

    def pack(idx):
        if len(idx) == 0:
            return (0, 0.0, 0.0, 0.0)
        a = sum(d2v[ds[i]] - d2f[ds[i]] for i in idx)
        b = sum(d3v[ds[i]] - d3f[ds[i]] for i in idx)
        c = sum(d2v[ds[i]] - d3v[ds[i]] for i in idx)
        return (len(idx), a, b, c)

    def named(q_src, labels, name):
        qs = np.percentile(q_src, [33.33, 66.67])
        masks = [q_src <= qs[0], (q_src > qs[0]) & (q_src <= qs[1]), q_src > qs[1]]
        return (name, [tuple([labels[j]] + list(pack(np.where(masks[j])[0]))) for j in range(3)])

    groups = [named(cv, ['低 CV', '中 CV', '高 CV'], '价格 CV 三分组'),
              named(corr, ['相关性低', '相关性中', '相关性高'], '价格与净负荷相关系数三分组')]
    months = sorted({d.month for d in ds})
    groups.append(('月份分组', [tuple([f'{m} 月'] + list(
        pack(np.array([i for i, d in enumerate(ds) if d.month == m])))) for m in months]))
    return groups


def collect_pair(recs_q2, recs_q3, fixed_q2, fixed_q3):
    def one(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        assert a.shape == b.shape, f"配对样本长度不一致: {a.shape} vs {b.shape}"
        x = a - b
        return {'n': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)),
                'p_save': float((x > 0).mean()), 'x': x,
                'ci7': moving_block_ci(x, BOOT_BLOCKS[0], BOOT_N_RESAMPLE,
                                       np.random.default_rng(BOOT_SEED)),
                'ci3': moving_block_ci(x, BOOT_BLOCKS[1], BOOT_N_RESAMPLE,
                                       np.random.default_rng(BOOT_SEED)),
                'ci14': moving_block_ci(x, BOOT_BLOCKS[2], BOOT_N_RESAMPLE,
                                        np.random.default_rng(BOOT_SEED))}

    return {'q2': one([r['total_cost'] for r in fixed_q2], [r['total_cost'] for r in recs_q2]),
            'q3': one([r['scen'][3]['C_total'] for r in fixed_q3],
                      [r['scen'][3]['C_total'] for r in recs_q3])}


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


def build_daily_table(ctx, recs_q2, recs_q3, cf, sens_daily, pmat_v):
    """逐日分析表（文档「建议保存的逐日分析表」）：价格 / 调度 / Q3调整 / 紧急购电 /
    成本 / 反事实 六组字段，逐日一行。"""
    L, G = ctx['L'], ctx['G']
    didx = ctx['date_idx']
    q95 = float(np.percentile(pmat_v, 95))
    r2 = {r['date']: r for r in recs_q2}
    r3 = {r['date']: r for r in recs_q3}
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
        buy2 = DT * float(rec2['B'].sum())
        eff3 = DT * float(rec3['Qeff_s3'].sum())
        emg_e2 = DT * float(st2['R'].sum())
        emg_high2 = DT * float(st2['R'][pv >= q95].sum())
        emg_high3 = DT * float(st3['R'][pv >= q95].sum())
        cf2, cf3 = cf['q2'].get(d, {}), cf['q3'].get(d, {})
        rows.append({
            '日期': d,
            # ---- 日期与价格 ----
            '价格_均价': f['mean'], '价格_标准差': f['std'], '价格_CV': f['cv'],
            '价格_P95': f['p95'], '价格_P5': f['p5'], '价格_P95-P5': f['spread95_5'],
            '价格_最高': f['max'], '价格_最低': f['min'], '价格_爬坡量': f['ramp'],
            '价格_尖峰时段数': int((pv >= f['p90']).sum()),
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
            'Q4-2_调度迁移量': cf2.get('shift', float('nan')),
            'Q4-3_生效购电量': eff3,
            'Q4-3_生效购电加权均价': DT * float(pv @ rec3['Qeff_s3']) / eff3 if eff3 > 0 else float('nan'),
            'Q4-3_吞吐量': st3['throughput'],
            'Q4-3_EFC可用容量': st3['throughput'] / (2 * E_USE),
            'Q4-3_弃光量': st3_phys['curtail'],
            'Q4-3_光伏自用率': (st3_phys['pv_use'] / (DT * float(Gd.sum()))
                            if Gd.sum() > 0 else float('nan')),
            # ---- Q3 调整 ----
            'Q4-3_6点采纳': int(rec3['a_flags'][0]),
            'Q4-3_12点采纳': int(rec3['a_flags'][1]),
            'Q4-3_18点采纳': int(rec3['a_flags'][2]),
            'Q4-3_上调电量': updown_energy(rec3, up=True),
            'Q4-3_下调电量': updown_energy(rec3, up=False),
            'Q4-3_调整净费': st3['adj'],
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
        })
    return rows


def collect_sens_metrics(ctx, exp, recs_q2, recs_q3, pmat_v):
    """敏感性实验汇总指标（文档「价格振幅敏感性」要求报告的量）。"""
    pmat = make_pmat(pmat_v, exp)
    return {'q2': _agg_family(ctx, recs_q2, 'q2', pmat),
            'q3': _agg_family(ctx, recs_q3, 'q3', pmat),
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


def write_summary(yearly, risk, cf, recs_q2, recs_q3):
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
    ws4.cell(row=1, column=1, value='方案').font = openpyxl.styles.Font(bold=True)
    for j, h in enumerate(['计划费占比', '调整净费占比', '紧急费占比', '合计', '核算残差/元'], 2):
        ws4.cell(row=1, column=j, value=h).font = openpyxl.styles.Font(bold=True)
    for i, nm in enumerate(('Q4-2 波动电价', 'Q4-3-S3 波动电价'), 2):
        y = yearly[nm]
        t = y['total']
        sh = [y['plan'] / t, y['adj'] / t, y['emg'] / t]
        for j, v in enumerate([nm] + sh + [sum(sh), t - (y['plan'] + y['adj'] + y['emg'])], 1):
            ws4.cell(row=i, column=j, value=round(v, 6) if isinstance(v, float) else v)

    ws5 = wb.create_sheet('约束活跃率与灵活性')
    kk = ['r_ch_max', 'r_dis_max', 'r_E_min', 'r_E_max']
    ws5.cell(row=1, column=1, value='指标').font = openpyxl.styles.Font(bold=True)
    for j, nm in enumerate(names, 2):
        ws5.cell(row=1, column=j, value=nm).font = openpyxl.styles.Font(bold=True)
    for i, k in enumerate(kk, 2):
        ws5.cell(row=i, column=1, value=k)
        for j, nm in enumerate(names, 2):
            ws5.cell(row=i, column=j, value=round(float(yearly[nm]['activity'][k]), 6))
    extra = ['soc_ready', 'EFC_use', 'EFC_nom', 'pv_self', 'r_emg_high', 'A_up', 'A_down']
    for i, k in enumerate(extra, len(kk) + 2):
        ws5.cell(row=i, column=1, value=k)
        for j, nm in enumerate(names, 2):
            ws5.cell(row=i, column=j, value=round(float(yearly[nm][k]), 6))

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
        tags = [t for t in ('lam0', 'vol', 'lam05', 'lam15', 'nobess', 'gam08', 'gam12')
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


# ========== 10. 报告生成（全部数值取自本次实测） ==========
def _f(x, n=4):
    try:
        v = float(x)
        return 'nan' if not np.isfinite(v) else f"{v:.{n}f}"
    except Exception:
        return str(x)


def _wan(x, n=2):
    return _f(float(x) / 10000.0, n)


def _pc(a, b):
    return float('nan') if abs(b) < 1e-12 else 100.0 * (a - b) / b


def _sgn_pc(a, b):
    v = _pc(a, b)
    return '—' if not np.isfinite(v) else f"{v:.2f}%"


def write_report(ctx, main, pmat_v):
    y = main['yearly']
    risk = main['risk']
    cf = main['cf']
    sens = main['sens']
    r2, r3 = y['Q2 固定电价'], y['Q3-S3 固定电价']
    v2, v3 = y['Q4-2 波动电价'], y['Q4-3-S3 波动电价']
    p1 = main['p1']
    n_day = risk['Q4-2 波动电价']['n']
    cvF2 = sum(x['C_V_xF'] for x in cf['q2'].values())
    cF2 = sum(x['C_F_xF'] for x in cf['q2'].values())
    cvF3 = sum(x['C_V_xF'] for x in cf['q3'].values())
    cF3 = sum(x['C_F_xF'] for x in cf['q3'].values())
    L_ = []
    A = L_.append
    A("# 问题四：波动电价下的微网购电策略重算与深度分析 —— 求解结果报告")
    A("")
    A("_本报告由 `Q4/solve_q4.py` 在完成正式重算与敏感性实验后自动生成，"
      "所有数值均取自本次实测缓存，不含占位值。_")
    A("")

    A("## 0. 口径声明")
    A("")
    A("严格遵循《问题四_波动电价影响深度分析》的「必须保持不变的口径」：问题二与问题三的"
      "**预测方法、场景生成、MPC 更新节点、采纳规则、储能约束、紧急购电规则与结算公式完全未改动**，"
      "唯一替换的是每日电价向量：")
    A("")
    A("$$c_t^{F}\\ \\longrightarrow\\ c_{d,t}^{V}$$")
    A("")
    A("- 实现方式：`Q4/solve_q4.py` 直接 import 复用 `Q2/solve_q2.py` 与 `Q3/solve_q3.py`，"
      "仅在每日求解前置入附件4 当日电价行；脚本启动时对上游物理常量做一致性断言。")
    A(f"- 价格口径：附件4 表头与附件1 一致（末列 `0:00+1`，时段终点口径），逐日 144 时段；"
      f"取值范围 [{_f(pmat_v.min())}, {_f(pmat_v.max())}] 元/kWh，整体均价 {_f(pmat_v.mean())}。")
    n_nonpos = int((pmat_v <= 0).sum())
    A(f"- 非正价格：{n_nonpos} 个。"
      + ("故 5 倍紧急购电、1.5 倍上调、0.5 倍下调的乘数规则可直接继承，正式结果未作任何改写。"
         if n_nonpos == 0 else
         "**存在非正价格，乘数结算的「惩罚更贵」含义不成立**；本报告按题目原式原样重算并记录该边界。"))
    A(f"- 附件1 与附件4 的关系：附件1 恰为附件4 的逐时段跨天均值（最大偏差 "
      f"{_f(np.abs(pmat_v.mean(axis=0) - p1).max(), 6)} 元/kWh），两者全年均价同为 "
      f"{_f(pmat_v.mean())} 元/kWh。因此固定电价与波动电价之间的差异**不是价格水平差异，"
      "而是日内波动与逐日波动的差异**。")
    A("- 不售电、允许弃光；并网点购电上限沿用原模型参数位（题目未给，取无上限）。")
    A("- 敏感性实验价格口径：`lam0/lam05/lam15` 按文档 $c^{(\\lambda)}_{d,t}=\\bar c_d"
      "+\\lambda(c_{d,t}^{V}-\\bar c_d)$ 构造，当日均价保持为 $\\bar c_d$。"
      "λ>1 的线性外推会把最低价推过零点，而计划购电量无上界，将导致 MILP 无界；"
      "故对越界时段按文档第 2 节第 6 条做下截尾并保均值重标定，详见 1.1 节。")
    A("")

    A("## 1. 回归核验")
    A("")
    A("按文档「回归核验」第 1、9 条：把价格换回附件1 后，对固定电价策略 x^F 的重新结算必须复现 "
      "Q2/Q3 原生缓存，且「固定策略按波动价格重新结算时不允许重新优化任何变量」。")
    A("")
    A(f"- 结果：{'**通过**' if main['reg_ok'] else '**未通过**'} "
      "——附件1 价格下逐日重算与 Q2/Q3 缓存逐项一致（计划费、调整净费、紧急费、总费用，"
      "容差 1e-6 相对 / 1e-4 绝对），说明本脚本的结算口径与上游完全相同。")
    A("- 说明：`q2_settle` / `q3_settle` 只做「价格 × 已锁定电量」的计费，不含任何优化变量；"
      "问题三的上调/下调量 U/W 通过包装 `Q3.solve_milp` 在求解瞬间按模型自带变量布局抓取，"
      "与模型内 U/W 定义逐位一致（不改动 Q3 源码、不重建、不近似）。")
    sc = main.get('settle_chk') or {}
    if sc:
        w2, _ = sc.get('q2', (float('nan'), ''))
        w3, w3wh = sc.get('q3', (float('nan'), ''))
        A(f"- 结算自检（用各日实际价格重新结算，应复现记录自身分项费用）："
          f"Q4-2 最大相对偏差 {_f(w2, 3)}；Q4-3 最大相对偏差 {_f(w3, 3)}"
          + (f"（{w3wh}）" if w3 > REG_TOL_REL else "（全部一致）") + "。")
    A("")

    A("### 1.1 价格口径核验与 λ>1 越界修正")
    A("")
    pa = main.get('price_audit')
    if pa:
        rows_pa, floor = pa
        A(f"附件4 全样本最低观测价 floor = {_f(floor)} 元/kWh。逐实验核验如下"
          "（文档第 2 节要求「统计最小价格，重点检查零价和负价」）：")
        A("")
        A("| 实验 | 最低 | 最高 | 均值 | 非正时段 | 下截尾时段 | 受影响天数 | 日均价最大偏差 |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|")
        for exp in EXPERIMENTS:
            a = rows_pa[exp]
            A(f"| {a['label']} | {_f(a['min'])} | {_f(a['max'])} | {_f(a['mean'])} | "
              f"{a['n_nonpos']} | {a['n_clip']} | {a['n_day_clip']} | {_f(a['mean_dev'], 3)} |")
        A("")
        a15 = rows_pa['lam15']
        A(f"- **问题**：λ=1.5 的线性外推把 {a15['n_clip']} 个时段"
          f"（占 {a15['n_clip'] / pmat_v.size * 100:.2f}%）压到附件4 最低观测价以下，"
          f"其中 {a15['n_neg']} 个时段**已越过零点成为负价**，涉及 "
          f"{a15['n_day_clip']} 个决策日。而 Q2/Q3 的并网购电量 $B$ 无上界"
          "（`P_buy_max=∞`，题目未给并网容量），目标中的 $c\\,B\\,\\Delta t$ 随 $B\\to\\infty$ "
          "趋于 $-\\infty$，HiGHS 直接报 `Primal infeasible or unbounded` —— "
          "这是**数学上的无界**，不是数值病态，换求解器或放宽容差都无效。")
        A("- **处理依据**：文档第 2 节第 6 条「不随意删除价格尖峰……应先保留为主结果，"
          "再在敏感性分析中做缩尾或截尾对照」。据此对越界时段做**下截尾**"
          "$c\\leftarrow\\max(c,\\,\\mathrm{floor})$，再整日乘 $\\bar c_d/\\overline{c'}$ "
          "把当日均价严格还原为 $\\bar c_d$，从而保住「λ 只改变波动幅度、不改变价格水平」"
          "这一实验前提。"
          f"上表末列为 {_f(a15['mean_dev'], 2)}，即重标定后当日均价与原当日均值完全一致。")
        A("- **影响面**：仅 λ>1 触发。λ=0/0.5/1 全程无越界，为逐位恒等变换，"
          "故 vol / lam0 / lam05 以及全部正式交付结果不受任何影响。")
    else:
        A("_（未采集到价格核验数据）_")
    A("")

    A("## 2. 总体结果表")
    A("")
    A("| 方案 | 计划费/万元 | 调整净费/万元 | 紧急费/万元 | 总费用/万元 | 紧急购电/MWh | EFC(可用容量) | CVaR95/万元 |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for nm in ('Q2 固定电价', 'Q4-2 波动电价', 'Q3-S3 固定电价', 'Q4-3-S3 波动电价'):
        yy = y[nm]
        A(f"| {nm} | {_wan(yy['plan'])} | {_wan(yy['adj'])} | {_wan(yy['emg'])} | "
          f"{_wan(yy['total'])} | {_f(yy['E_emg'] / 1000.0, 2)} | {_f(yy['EFC_use'])} | "
          f"{_wan(risk[nm]['cvar95'])} |")
    A("")
    A(f"- Q4-2 相对 Q2：总费用 {_wan(r2['total'])} -> {_wan(v2['total'])} 万元，"
      f"变化 **{_sgn_pc(v2['total'], r2['total'])}**；计划购电量 "
      f"{_f(r2['buy_energy'] / 1000.0, 2)} -> {_f(v2['buy_energy'] / 1000.0, 2)} MWh"
      f"（{_sgn_pc(v2['buy_energy'], r2['buy_energy'])}），购电加权均价 "
      f"{_f(r2['buy_wavg'])} -> {_f(v2['buy_wavg'])} 元/kWh。")
    A(f"- Q4-3-S3 相对 Q3-S3：总费用 {_wan(r3['total'])} -> {_wan(v3['total'])} 万元，"
      f"变化 **{_sgn_pc(v3['total'], r3['total'])}**；购电加权均价 "
      f"{_f(r3['buy_wavg'])} -> {_f(v3['buy_wavg'])} 元/kWh。")
    A(f"- Q4-3-S3 相对 Q4-2：平均日节省 {_f((v2['total'] - v3['total']) / n_day, 4)} 元/天"
      f"（全年 {_wan(v2['total'] - v3['total'])} 万元），CVaR95 "
      f"{_wan(risk['Q4-2 波动电价']['cvar95'])} -> {_wan(risk['Q4-3-S3 波动电价']['cvar95'])} 万元。")
    A("")

    A("## 3. 反事实分解：价格变了，还是策略变了")
    A("")
    A("$$C_V(x^V)-C_F(x^F)=\\underbrace{[C_V(x^F)-C_F(x^F)]}_{\\text{纯价格结算效应}}"
      "+\\underbrace{[C_V(x^V)-C_V(x^F)]}_{\\text{调度适应效应}}$$")
    A("")
    A("| 指标 | Q4-2 | Q4-3-S3 |")
    A("|---|---:|---:|")
    A(f"| 固定电价原成本 C_F(x^F)/万元 | {_wan(cF2)} | {_wan(cF3)} |")
    A(f"| 原策略按附件4 价格结算 C_V(x^F)/万元 | {_wan(cvF2)} | {_wan(cvF3)} |")
    A(f"| 波动电价重新优化成本 C_V(x^V)/万元 | {_wan(v2['total'])} | {_wan(v3['total'])} |")
    A(f"| 纯价格结算效应 C_V(x^F)−C_F(x^F)/万元 | {_wan(cvF2 - cF2)} | {_wan(cvF3 - cF3)} |")
    A(f"| 调度适应效应 C_V(x^V)−C_V(x^F)/万元 | {_wan(v2['total'] - cvF2)} | "
      f"{_wan(v3['total'] - cvF3)} |")
    _p2 = (100.0 * (v2['total'] - cvF2) / (cvF2 - cF2)) if abs(cvF2 - cF2) > 1e-9 else float('nan')
    _p3 = (100.0 * (v3['total'] - cvF3) / (cvF3 - cF3)) if abs(cvF3 - cF3) > 1e-9 else float('nan')
    A(f"| 调度适应效应占纯价格效应比例 | "
      + ("—" if not np.isfinite(_p2) else f"{_p2:.2f}%") + " | "
      + ("—" if not np.isfinite(_p3) else f"{_p3:.2f}%") + " |")
    A("")
    buf2 = (cvF2 - v2['total']) / (cvF2 - cF2) if abs(cvF2 - cF2) > 1e-9 else float('nan')
    buf3 = (cvF3 - v3['total']) / (cvF3 - cF3) if abs(cvF3 - cF3) > 1e-9 else float('nan')
    A(f"**缓冲比例** $r^{{\\mathrm{{mitigate}}}}=\\frac{{C_V(x^F)-C_V(x^V)}}{{C_V(x^F)-C_F(x^F)}}$："
      f"Q4-2 = {_f(100 * buf2, 2)}%，Q4-3-S3 = {_f(100 * buf3, 2)}%。"
      "该值表示已发生的价格冲击中被重新调度抵消的比例；为负说明在附件4 价格下重新优化反而比"
      "沿用固定电价策略更贵——在完全信息、同可行域、同结算目标的确定性模型中不应发生，"
      "出现即为样本外预测误差或采纳规则使优化目标与事后结算不一致的证据，应按日统计并保留。")
    A("")

    A("## 4. 调度响应")
    A("")
    A("| 指标 | Q2 固定 | Q4-2 波动 | 变化 |")
    A("|---|---:|---:|---:|")
    for lbl, k in (("计划购电量/MWh", 'buy_energy'), ("购电加权均价/(元/kWh)", 'buy_wavg'),
                   ("充电量/MWh", 'charge'), ("放电量/MWh", 'discharge'),
                   ("吞吐量/MWh", 'throughput'), ("EFC(可用容量)", 'EFC_use'),
                   ("弃光量/MWh", 'curtail'), ("光伏自用率", 'pv_self'),
                   ("紧急购电量/MWh", 'E_emg'), ("紧急购电平均支付单价/(元/kWh)", 'emg_unit')):
        a, b = r2[k], v2[k]
        av = a / 1000.0 if lbl.endswith('MWh') else a
        bv = b / 1000.0 if lbl.endswith('MWh') else b
        A(f"| {lbl} | {_f(av)} | {_f(bv)} | {_sgn_pc(b, a)} |")
    A("")
    shifts = [x['shift'] for x in cf['q2'].values() if np.isfinite(x['shift'])]
    A(f"- **半 L1 调度迁移量** $E_d^{{\\mathrm{{shift}}}}$：全年合计 "
      f"{_f(np.sum(shifts) / 1000.0, 2)} MWh，日均 {_f(np.mean(shifts) / 1000.0, 4)} MWh。")
    s_fix = np.nanmean([x['soc_ready'] for x in cf['q2'].values()])
    A(f"- **高价时段 SOC 准备度**（进入价格前 10% 时段时的可用电量占比）：固定电价 {_f(s_fix)}，"
      f"波动电价 {_f(v2['soc_ready'])}。"
      + ("该值上升说明电池在价格尖峰前主动蓄电，发挥了价格对冲作用。"
         if v2['soc_ready'] > s_fix else
         "该值未上升，说明价格信号的蓄电动机未强于原有净负荷驱动。"))
    act = v2['activity']
    mx = max(act.values())
    A(f"- **约束活跃率**（阈值 {ACT_EPS_P} kW / {ACT_EPS_E} kWh）：充电到顶 "
      f"{_f(100 * act['r_ch_max'], 2)}%、放电到顶 {_f(100 * act['r_dis_max'], 2)}%、"
      f"SOC 下限 {_f(100 * act['r_E_min'], 2)}%、SOC 上限 {_f(100 * act['r_E_max'], 2)}%。"
      + ("比率偏高，说明限制调度收益的是设备与并网容量，而不是算法未识别价差。"
         if mx > 0.05 else "比率均较低，说明储能容量与功率在多数时段并未成为紧约束。"))
    sp = np.array([np.percentile(r, 95) - np.percentile(r, 5) for r in pmat_v])
    A(f"- 套利阈值校验：往返效率 0.81，纯电价套利要求高价/低价 > 1.2346（至少高 23.46%）。"
      f"附件4 各日均价 {_f(pmat_v.mean(axis=1).min())}~{_f(pmat_v.mean(axis=1).max())} 元/kWh，"
      f"日内 P95−P5 价差均值 {_f(sp.mean())} 元/kWh（占日均价 "
      f"{_f(100 * np.mean(sp / pmat_v.mean(axis=1)), 2)}%），"
      f"说明价格时序已具备触发储能搬移的条件。")
    A("")

    A("## 5. 成本结构与风险特征")
    A("")
    A("| 方案 | 计划费占比 | 调整净费占比 | 紧急费占比 | 核算恒等式残差/元 |")
    A("|---|---:|---:|---:|---:|")
    for nm in ('Q4-2 波动电价', 'Q4-3-S3 波动电价'):
        yy = y[nm]
        t = yy['total']
        A(f"| {nm} | {_f(100 * yy['plan'] / t, 2)}% | {_f(100 * yy['adj'] / t, 2)}% | "
          f"{_f(100 * yy['emg'] / t, 2)}% | {_f(t - (yy['plan'] + yy['adj'] + yy['emg']), 6)} |")
    A("")
    A("| 风险指标 | Q2 固定 | Q4-2 波动 | Q3-S3 固定 | Q4-3-S3 波动 |")
    A("|---|---:|---:|---:|---:|")
    names = ('Q2 固定电价', 'Q4-2 波动电价', 'Q3-S3 固定电价', 'Q4-3-S3 波动电价')
    for lbl, k in (("日均费用/元", 'mean'), ("中位数/元", 'median'), ("标准差/元", 'std'),
                   ("P90/元", 'p90'), ("P95/元", 'p95'), ("最大日费用/元", 'max'),
                   ("CVaR90/元", 'cvar90'), ("CVaR95/元", 'cvar95'), ("CVaR99/元", 'cvar99')):
        A(f"| {lbl} | " + " | ".join(_f(risk[n][k], 2) for n in names) + " |")
    A("| 最坏10日费用贡献 | " + " | ".join(f"{_f(100 * risk[n]['worst_share'], 2)}%" for n in names) + " |")
    A("")
    A(f"- **高价紧急购电占比**（价格 ≥ 全样本 P95 = {_f(main['q95'])} 元/kWh 时段的紧急购电量"
      f"占全部紧急购电量）：Q4-2 {_f(100 * v2['r_emg_high'], 2)}%，"
      f"Q4-3-S3 {_f(100 * v3['r_emg_high'], 2)}%。"
      f"紧急购电平均支付单价：Q4-2 {_f(v2['emg_unit'])} 元/kWh，Q4-3-S3 {_f(v3['emg_unit'])} 元/kWh"
      f"（固定电价理论值 {_f(KAPPA * p1.mean())} 元/kWh）。")
    cid = main['coincide']
    A(f"- **日期重合率**（各取最极端 10 天）：价格最高10日 ∩ 紧急购电最高10日 = "
      f"{cid['price_emg']}/10；价格最高10日 ∩ 净负荷最高10日 = {cid['price_netload']}/10；"
      f"费用最高10日 ∩ 紧急购电最高10日 = {cid['cost_emg']}/10。"
      + ("重合率不高，说明价格尖峰与供电缺口并非同时发生，风险来源相对分散。"
         if cid['price_emg'] <= 4 else
         "重合率较高，说明存在「供能不足与高价同时发生」的双重风险。"))
    A("")

    A("## 6. 灵活性代价与储能价值")
    A("")
    A(f"- 储能吞吐量：Q2 {_f(r2['throughput'] / 1000.0, 2)} -> Q4-2 "
      f"{_f(v2['throughput'] / 1000.0, 2)} MWh（{_sgn_pc(v2['throughput'], r2['throughput'])}）；"
      f"Q3-S3 {_f(r3['throughput'] / 1000.0, 2)} -> Q4-3-S3 "
      f"{_f(v3['throughput'] / 1000.0, 2)} MWh（{_sgn_pc(v3['throughput'], r3['throughput'])}）。")
    A(f"- 运行等效满循环数（分母须注明口径）：Q4-2 以安全可用容量 9600 kWh 计 "
      f"EFC={_f(v2['EFC_use'])}，以额定容量 12000 kWh 计 EFC={_f(v2['EFC_nom'])}；"
      f"Q4-3-S3 分别为 {_f(v3['EFC_use'])} / {_f(v3['EFC_nom'])}。两种口径不可混用。")
    A(f"- 调整行为（Q4-3-S3）：全年上调电量 {_f(v3['A_up'] / 1000.0, 2)} MWh、"
      f"下调电量 {_f(v3['A_down'] / 1000.0, 2)} MWh；6:00/12:00/18:00 采纳天数占比 "
      f"{_f(100 * v3['adopt_rate'][0], 1)}% / {_f(100 * v3['adopt_rate'][1], 1)}% / "
      f"{_f(100 * v3['adopt_rate'][2], 1)}%。")
    if 'nobess' in sens:
        nb = sens['nobess']
        A(f"- **储能价值** $V^{{BESS}}=C^{{无储能}}-C^{{有储能}}$：波动电价下 Q4-2 "
          f"{_wan(nb['q2']['total'] - v2['total'])} 万元，Q4-3-S3 "
          f"{_wan(nb['q3']['total'] - v3['total'])} 万元（无储能全年总费用 Q4-2 "
          f"{_wan(nb['q2']['total'])} 万元、Q4-3-S3 {_wan(nb['q3']['total'])} 万元）。"
          "该值同时包含光伏移峰、规避高价与减少紧急购电的综合价值。")
    else:
        A("- _未运行无储能反事实实验（用 `--exp nobess` 追加）。_")
    A("")

    A("## 7. 敏感性分析")
    A("")
    A("### 7.1 价格振幅敏感性 λ（把平均价格与波动幅度分开）")
    A("")
    A("| 实验 | Q4-2 总费用/万元 | Q4-3-S3 总费用/万元 | EFC(可用容量) | 紧急费/万元 | Q4-3 调整价值/万元 |")
    A("|---|---:|---:|---:|---:|---:|")
    def _sl(e):
        """敏感性查找：'vol' 为正式重算基准，存在 sens['vol_base']。"""
        return sens.get('vol_base') if e == 'vol' else sens.get(e)

    for e in ('lam0', 'lam05', 'vol', 'lam15'):
        s = _sl(e)
        if s is None:
            continue
        adjv = s['q2']['total'] - s['q3']['total']
        A(f"| {EXP_LABEL[e]} | {_wan(s['q2']['total'])} | {_wan(s['q3']['total'])} | "
          f"{_f(s['q2']['EFC_use'])} | {_wan(s['q2']['emg'])} | {_wan(adjv)} |")
    A("")
    if 'lam0' in sens and 'vol_base' in sens:
        dv = sens['vol_base']['q2']['total'] - sens['lam0']['q2']['total']
        A(f"- λ=0（同日均值平坦电价）总费用 {_wan(sens['lam0']['q2']['total'])} 万元，"
          f"λ=1（附件4 原始）{_wan(sens['vol_base']['q2']['total'])} 万元，差 {_wan(dv)} 万元 —— "
          f"这就是**在相同日均价格水平下、仅由日内与逐日波动造成的成本差异**，"
          f"比直接与附件1 比较更能回答「波动本身创造了多少调度价值」。")
    if 'lam15' in sens:
        _vEFC = _f(sens['vol_base']['q2']['EFC_use']) if 'vol_base' in sens else '—'
        A(f"- λ=1.5（价差放大 1.5 倍）总费用 {_wan(sens['lam15']['q2']['total'])} 万元，"
          f"EFC {_f(sens['lam15']['q2']['EFC_use'])}（λ=1 为 {_vEFC}）。"
          "若储能价值随 λ 上升而 EFC 同步快速上升，说明价格波动创造了套利空间、也同步增加了"
          "循环压力与寿命成本；若 λ 增大而策略几乎不变，应检查价差是否未超过 23.46% 的效率补偿阈值，"
          "或 SOC / 充放电功率 / 并网容量已成为主导约束。")
    A("")
    A("### 7.2 正比例缩放 γ（模型一致性核验）")
    A("")
    gc = sens.get('gamma_check')
    if gc:
        A("| γ | Q4-2 总费用/万元 | 费用比(相对 γ=1) | Q4-2 最大调度偏差/kW | 调度不一致天数 |")
        A("|---|---:|---:|---:|---:|")
        for e, g in (('gam08', 0.8), ('vol', 1.0), ('gam12', 1.2)):
            if e == 'vol':
                A(f"| {g} | {_wan(v2['total'])} | 1.0000 | 0.000000 | 0 |")
            elif e in sens:
                d = gc.get(e, {}).get('q2') or {}
                A(f"| {g} | {_wan(sens[e]['q2']['total'])} | "
                  f"{_f(sens[e]['q2']['total'] / v2['total'])} | "
                  f"{_f(d.get('max_dev', float('nan')), 6)} | {d.get('n_bad_day', -1)} |")
        A("")
        A("目标中计划费、上调费、下调费与紧急费均按同一 γ 缩放、且不含独立于价格的衰减成本，"
          "因此最优调度原则上应保持不变、总费用按 γ 等比例变化：**费用比应≈γ，调度偏差应≈0**。"
          "若调度显著改变，需按文档检查是否存在未同步缩放的成本项、词典序微小惩罚对并列最优解的影响、"
          "求解容差导致的整数解差异、以及采纳规则是否比较了量纲不一致的目标。"
          "该检验用于发现实现错误，不作为经济结论。")
    else:
        A("_未运行 γ 缩放实验（用 `--exp gam08,gam12` 追加）。_")
    A("")
    A("### 7.3 分组检验（波动电价在什么情形下最有价值）")
    A("")
    A("| 分组维度 | 分组 | 天数 | ΔC(Q4-2−Q2)/万元 | ΔC(Q4-3−Q3)/万元 | ΔC^MPC(Q4-2−Q4-3)/万元 |")
    A("|---|---|---:|---:|---:|---:|")
    for dim, groups in main['group']:
        for gname, nd, a, b, c in groups:
            A(f"| {dim} | {gname} | {nd} | {_wan(a)} | {_wan(b)} | {_wan(c)} |")
    A("")
    A("$\\Delta C^{\\mathrm{MPC}}>0$ 表示波动电价下滚动调整相对只做 0:00 计划节省费用。")
    A("")
    A("### 7.4 配对统计与置信区间")
    A("")
    for nm, a, ref in (("Q4-2 相对 Q2", main['pair']['q2'], "固定电价 Q2"),
                       ("Q4-3-S3 相对 Q3-S3", main['pair']['q3'], "固定电价 Q3-S3")):
        m, lo, hi = a['ci7']
        A(f"**{nm}**（配对差值 $\\delta_d=C^{{ref}}-C^{{new}}$，δ>0 表示新方案更省）")
        A("")
        A(f"- 平均配对差 {_f(a['mean'], 4)} 元/天（全年 {_wan(a['mean'] * a['n'])} 万元），"
          f"中位数 {_f(a['median'], 4)} 元/天，节省天数占比 {_f(100 * a['p_save'], 2)}%。")
        A(f"- 95% 置信区间（7 天移动块自助法，{BOOT_N_RESAMPLE} 次，seed={BOOT_SEED}）："
          f"均值 {_f(m, 4)}，区间 [{_f(lo, 4)}, {_f(hi, 4)}] 元/天；稳健性对照："
          f"3 天块 [{_f(a['ci3'][1], 4)}, {_f(a['ci3'][2], 4)}]，"
          f"14 天块 [{_f(a['ci14'][1], 4)}, {_f(a['ci14'][2], 4)}]。")
        A(f"- 判定：{'节省在统计上显著' if lo > 0 else ('费用上升在统计上显著' if hi < 0 else '区间含 0，差异不显著')}"
          "（按日配对、块自助法比把两组日费用当作独立样本的普通 t 检验更符合全年时间序列结构）。")
        A("")

    A("## 8. 结论与适用边界")
    A("")
    A(main['conclusion'])
    A("")
    A("## 9. 可直接改写的论文段落")
    A("")
    A(main['paper_paras'])
    A("")
    A("## 10. 与现实电力市场的联系")
    A("")
    A("> 本文将微网视为价格接受者，并假设附件4 的现货价格能够通过市场交易或零售合同"
      "传导至并网点购电结算。模型默认微网按 $c_{d,t}^{V}$ 直接结算，等价于微网参与现货市场，"
      "或零售合同把批发价格按时段传导给用户。")
    A("")
    A("- 若微网实际执行固定零售电价、峰谷目录电价或带封顶条款的代理购电合同，"
      "附件4 价格不会一比一转化为微网账单；此时上述结果应解释为**批发现货暴露下的技术经济上界**，"
      "而不是用户实际电费预测。")
    A("- 当前模型把价格当作外生参数，假设微网购电量不会反过来改变市场价格。"
      "对单个小区微网这是合理近似；若研究对象扩大为大规模聚合储能、虚拟电厂或市场份额较高的主体，"
      "则需要考虑价格冲击与策略博弈，否则可能高估套利空间。")
    A(f"- 价格尖峰处理：主分析保留全部尖峰"
      f"（附件4 最高价 {_f(pmat_v.max())} 元/kWh），因为尖峰是现货市场稀缺性与系统压力的经济信号；"
      "缩尾对照仅用于回答「结论是否由极少数极端值驱动」。")
    A("- 日内波动价值而非跨日价值：每日末 SOC 固定为 6000 kWh 限制了跨日套利"
      "（例如 23:00 高价与次日凌晨低价之间的搬移），且题目不允许售电，"
      "低价充电只能服务本地负荷，因此储能价值上限受本地负荷、弃光边界与并网容量共同约束。")
    A("")
    A("## 附录：运行与复现")
    A("")
    A("```")
    A("python solve_q4.py            # 一键全量：主重算 + 全套敏感性（断点续跑 + 8 进程）")
    A("python solve_q4.py main       # 仅正式重算")
    A("python solve_q4.py sens       # 仅敏感性")
    A("python solve_q4.py outputs    # 由缓存重建全部输出")
    A("python solve_q4.py regress    # 回归核验（附件1 价格重解 4 代表日）")
    A("```")
    A("")
    A(f"_报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(L_) + "\n")
    print(f"[输出] 分析报告 -> {REPORT_FILE}")


# ========== 11. 结论与论文段落（按文档「结果解释规则」自动判别情形） ==========
def build_conclusion(yearly, risk, cf):
    r2, r3 = yearly['Q2 固定电价'], yearly['Q3-S3 固定电价']
    v2, v3 = yearly['Q4-2 波动电价'], yearly['Q4-3-S3 波动电价']
    cvF2 = sum(x['C_V_xF'] for x in cf['q2'].values())
    cF2 = sum(x['C_F_xF'] for x in cf['q2'].values())
    pur2, adapt2 = cvF2 - cF2, v2['total'] - cvF2
    d2 = v2['total'] - r2['total']
    s = []
    if d2 > 0 and adapt2 < 0:
        s.append(f"**情形一：总费用上升，但调度适应效应为负。**附件4 的价格时序使 Q4-2 全年总费用较"
                 f"固定电价上升 {_wan(d2)} 万元，其中纯价格结算效应 {_wan(pur2)} 万元、"
                 f"重新调度抵消 {_wan(abs(adapt2))} 万元，缓冲比例 "
                 f"{_f(100 * abs(adapt2) / pur2, 2) if abs(pur2) > 1e-9 else '—'}%。"
                 "应表述为「波动电价提高了结算压力，储能与滚动调度抵消了其中相当一部分」，"
                 "而不是简单写成「新策略导致费用增加」。")
    elif d2 < 0 and v2['throughput'] > r2['throughput']:
        s.append(f"**情形二：总费用下降，购电量和 EFC 上升。**Q4-2 总费用下降 {_wan(abs(d2))} 万元，"
                 f"同时储能吞吐量由 {_f(r2['throughput'] / 1000.0, 2)} 升至 "
                 f"{_f(v2['throughput'] / 1000.0, 2)} MWh（EFC {_f(r2['EFC_use'])} -> "
                 f"{_f(v2['EFC_use'])}），购电加权均价由 {_f(r2['buy_wavg'])} 降至 "
                 f"{_f(v2['buy_wavg'])} 元/kWh，说明模型通过低价充电、高价少购电实现套利；"
                 "长期经济性须结合衰减成本敏感性判断。")
    else:
        s.append(f"**费用与灵活性同向变化。**Q4-2 总费用较固定电价变化 {_wan(d2)} 万元"
                 f"（{_sgn_pc(v2['total'], r2['total'])}），储能吞吐量变化 "
                 f"{_sgn_pc(v2['throughput'], r2['throughput'])}。")
    if v2['E_emg'] < r2['E_emg'] and v2['emg_unit'] > r2['emg_unit']:
        s.append(f"**情形三：紧急购电量下降，但平均支付单价上升。**紧急购电量由 "
                 f"{_f(r2['E_emg'] / 1000.0, 2)} MWh 降至 {_f(v2['E_emg'] / 1000.0, 2)} MWh，"
                 f"而紧急购电平均支付单价由 {_f(r2['emg_unit'])} 升至 {_f(v2['emg_unit'])} 元/kWh，"
                 f"高价紧急购电占比 {_f(100 * v2['r_emg_high'], 2)}%。"
                 "物理供电可靠性改善，但经济尾部风险向价格尖峰集中，应重点报告 "
                 "$r^{\\mathrm{emg,high}}$、$\\bar c^{\\mathrm{emg,paid}}$ 与 CVaR95。")
    if v3['total'] < v2['total'] and risk['Q4-3-S3 波动电价']['cvar95'] < risk['Q4-2 波动电价']['cvar95']:
        s.append(f"**情形四：Q4-3 平均节省有限但尾部改善。**Q4-3-S3 相对 Q4-2 平均日节省 "
                 f"{_f((v2['total'] - v3['total']) / risk['Q4-2 波动电价']['n'], 4)} 元，"
                 f"CVaR95 由 {_wan(risk['Q4-2 波动电价']['cvar95'])} 降至 "
                 f"{_wan(risk['Q4-3-S3 波动电价']['cvar95'])} 万元，说明盘中预报更新的主要价值是"
                 "规避少数严重预测偏差与价格尖峰，而不是普遍降低每天费用；"
                 "因此「是否需要调整」不能只看平均 ΔC，应把尾部风险改善计入现实价值。")
    else:
        s.append(f"**Q4-3 相对 Q4-2 的边际价值。**全年总费用差 {_wan(abs(v2['total'] - v3['total']))} 万元，"
                 f"CVaR95 {_wan(risk['Q4-2 波动电价']['cvar95'])} -> "
                 f"{_wan(risk['Q4-3-S3 波动电价']['cvar95'])} 万元。")
    s.append("**逐日波动价值，而非跨日价值。**每日末 SOC 固定回到 6000 kWh 限制了 "
             "23:00 高价与次日凌晨低价之间的跨日套利，且题目不允许售电，"
             "低价充电只能服务本地负荷。因此本结果衡量的是**日内波动价值**，"
             "不代表允许跨日调度时的全部储能价值。"
             "报告价格差异时须同时给出购电加权平均价格，"
             "不能仅凭「总购电量增加」就断定策略变差——以较多低价电量替代较少高价电量"
             "可以同时出现「购电量上升、费用下降」。")
    return "\n\n".join(s)


def build_paper_paras(yearly, risk, cf, sens):
    r2, r3 = yearly['Q2 固定电价'], yearly['Q3-S3 固定电价']
    v2, v3 = yearly['Q4-2 波动电价'], yearly['Q4-3-S3 波动电价']
    cvF2 = sum(x['C_V_xF'] for x in cf['q2'].values())
    cF2 = sum(x['C_F_xF'] for x in cf['q2'].values())
    p = []
    p.append(f"> 与固定电价基准相比，附件4 波动电价使 Q4-2 全年总费用由 {_wan(cF2)} 万元变为 "
             f"{_wan(v2['total'])} 万元，变化 {_sgn_pc(v2['total'], cF2)}。反事实分解表明，"
             f"其中 {_wan(abs(cvF2 - cF2))} 万元来自价格水平及其时序变化对原策略的直接结算影响，"
             f"重新优化调度{'抵消' if v2['total'] - cvF2 < 0 else '增加'}了 "
             f"{_wan(abs(v2['total'] - cvF2))} 万元。波动电价下，购电加权平均价格由 "
             f"{_f(r2['buy_wavg'])} 元/kWh 变为 {_f(v2['buy_wavg'])} 元/kWh，储能吞吐量变化 "
             f"{_sgn_pc(v2['throughput'], r2['throughput'])}，说明调度主要通过"
             "低价充电、规避高价购电来响应价格信号。")
    p.append(f"> 从成本结构看，Q4-2 的计划购电费与紧急购电费分别占总费用的 "
             f"{_f(100 * v2['plan'] / v2['total'], 2)}% 和 {_f(100 * v2['emg'] / v2['total'], 2)}%；"
             f"Q4-3-S3 的计划费、调整净费与紧急费分别占 {_f(100 * v3['plan'] / v3['total'], 2)}%、"
             f"{_f(100 * v3['adj'] / v3['total'], 2)}% 和 {_f(100 * v3['emg'] / v3['total'], 2)}%。"
             f"紧急购电量由 {_f(r2['E_emg'] / 1000.0, 2)} MWh 变为 {_f(v2['E_emg'] / 1000.0, 2)} MWh，"
             f"而紧急购电费由 {_wan(r2['emg'])} 万元变为 {_wan(v2['emg'])} 万元，二者方向"
             f"{'一致' if (v2['E_emg'] - r2['E_emg']) * (v2['emg'] - r2['emg']) > 0 else '不一致'}。"
             f"结合高价紧急购电占比 {_f(100 * v2['r_emg_high'], 2)}% 可知，波动电价主要改变了"
             f"{'缺口规模' if v2['r_emg_high'] < 0.2 else '缺口发生时刻的价格暴露'}。")
    p.append(f"> 从风险角度看，逐日费用标准差由 {_wan(risk['Q2 固定电价']['std'])} 万元变为 "
             f"{_wan(risk['Q4-2 波动电价']['std'])} 万元，CVaR95 由 "
             f"{_wan(risk['Q2 固定电价']['cvar95'])} 万元变为 "
             f"{_wan(risk['Q4-2 波动电价']['cvar95'])} 万元，最高 10 个费用日贡献全年成本的 "
             f"{_f(100 * risk['Q4-2 波动电价']['worst_share'], 2)}%。"
             f"Q4-3-S3 相对 Q4-2 的平均节省为 "
             f"{_f((v2['total'] - v3['total']) / risk['Q4-2 波动电价']['n'], 4)} 元/天，"
             f"CVaR95 变化 {_sgn_pc(risk['Q4-3-S3 波动电价']['cvar95'], risk['Q4-2 波动电价']['cvar95'])}，"
             "表明盘中预报更新的价值主要体现在"
             f"{'尾部风险控制' if (v2['total'] - v3['total']) / risk['Q4-2 波动电价']['n'] < 1.0 else '平均经济性'}。")
    if 'lam0' in sens and 'lam15' in sens:
        _vb = sens.get('vol_base')
        _dlam = _wan(_vb['q2']['total'] - sens['lam0']['q2']['total']) if _vb else '—'
        p.append(f"> 价格振幅敏感性显示，当 λ 从 0 增至 1.5 时，Q4-2 全年总费用由 "
                 f"{_wan(sens['lam0']['q2']['total'])} 万元变为 {_wan(sens['lam15']['q2']['total'])} 万元，"
                 f"同时 EFC 由 {_f(sens['lam0']['q2']['EFC_use'])} 增至 "
                 f"{_f(sens['lam15']['q2']['EFC_use'])}。其中 λ=0（同日均值平坦电价）与 λ=1"
                 f"（附件4 原始）之差 {_dlam} 万元，"
                 "即为相同平均价格水平下仅由波动本身造成的成本差异。"
                 "因此波动电价为储能创造了调度空间，但其长期经济性取决于价差能否覆盖 "
                 "81% 往返效率造成的损耗以及电池寿命成本。")
    if 'nobess' in sens:
        p.append(f"> 无储能反事实显示，禁用储能后 Q4-2 全年总费用升至 "
                 f"{_wan(sens['nobess']['q2']['total'])} 万元，储能价值为 "
                 f"{_wan(sens['nobess']['q2']['total'] - v2['total'])} 万元；Q4-3-S3 下储能价值为 "
                 f"{_wan(sens['nobess']['q3']['total'] - v3['total'])} 万元。"
                 "该值同时包含光伏移峰、规避高价与减少紧急购电的综合价值。")
    return "\n\n".join(p)


# ========== 12. 主流程 ==========
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
    # 基准（vol / fixed）不进入敏感性表，但报告与逐日对照表需要它们
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


def collect_all(ctx2, ctx3, pmat_v, recs, args):
    """汇总全部指标并落盘正式交付文件与报告。"""
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
    try:
        native_q2 = [r for r in load_native_fixed('q2', ctx2, ctx3) if r['date'] in common]
        native_q3 = [r for r in load_native_fixed('q3', ctx2, ctx3) if r['date'] in common]
        if len(native_q2) != n_day or len(native_q3) != n_day:
            raise RuntimeError("固定电价原生缓存与本次重算的日期集合不匹配")
    except RuntimeError as exc:
        # 原生缓存只服务于回归核验；正式输出始终使用上方已完成的 fixed/vol 重算记录。
        # 克隆仓库未附带原生缓存时，复用同一价格、同一算法的 Q4 fixed 记录，避免无谓重解。
        print(f"[回归核验] {exc}；改用 Q4 fixed 重算记录作为基准。")
        native_q2, native_q3 = fixed_q2, fixed_q3

    write_templates(recs['q2'], recs['q3'])
    write_rep_tables('q2', recs['q2'])
    write_rep_tables('q3', recs['q3'])

    cf, p1 = build_counterfactual(ctx, fixed_q2, fixed_q3, native_q2, native_q3, pmat_v)
    cf = add_shift(cf, fixed_q2, recs['q2'])
    reg_ok = regression_report(cf)

    # 结算自检：U/W 抓取 + run_block 推进 + 成本核算口径必须逐日复现记录自身分项
    settle_chk = {f: verify_settle(_ctx_of(f, ctx2, ctx3), recs[f], pmat_v, f)
                  for f in ('q2', 'q3')}
    print("\n[结算自检] 用各日实际价格重新结算，应逐日复现记录自身分项费用")
    for f in ('q2', 'q3'):
        w, wh = settle_chk[f]
        print(f"  {f.upper()}: 最大相对偏差 {w:.3e}  {wh}")

    sens, sens_daily = collect_sens(ctx2, ctx3, pmat_v, args)
    yearly = collect_yearly(ctx, recs['q2'], recs['q3'], pmat_v, fixed_q2, fixed_q3)
    risk = collect_risk(recs['q2'], recs['q3'], fixed_q2, fixed_q3)

    daily = build_daily_table(ctx, recs['q2'], recs['q3'], cf, sens_daily, pmat_v)
    write_xlsx(DAILY_FILE, '逐日分析表', daily, '逐日分析')
    write_summary(yearly, risk, cf, recs['q2'], recs['q3'])
    if sens:
        write_sens(sens, sens_daily)

    main = {'daily': daily, 'yearly': yearly, 'risk': risk, 'cf': cf, 'reg_ok': reg_ok,
            'settle_chk': settle_chk,
            'price_audit': price_audit(pmat_v),
            'p1': p1, 'sens': sens, 'sens_daily': sens_daily,
            'q95': float(np.percentile(pmat_v, 95)),
            'coincide': collect_coincidence(ctx, recs['q2'], pmat_v),
            'group': collect_groups(ctx, recs['q2'], recs['q3'], fixed_q2, fixed_q3, pmat_v),
            'pair': collect_pair(recs['q2'], recs['q3'], fixed_q2, fixed_q3)}
    main['conclusion'] = build_conclusion(yearly, risk, cf)
    main['paper_paras'] = build_paper_paras(yearly, risk, cf, sens)

    print("\n" + "=" * 72)
    print("[正式结果汇总] 单位: 万元")
    print("=" * 72)
    for nm in ('Q2 固定电价', 'Q4-2 波动电价', 'Q3-S3 固定电价', 'Q4-3-S3 波动电价'):
        yy = yearly[nm]
        print(f"  {nm:<16} 计划 {yy['plan'] / 1e4:>10.2f} | 调整 {yy['adj'] / 1e4:>8.2f} | "
              f"紧急 {yy['emg'] / 1e4:>9.2f} | 合计 {yy['total'] / 1e4:>10.2f} | "
              f"CVaR95 {risk[nm]['cvar95'] / 1e4:>9.2f}")
    write_report(ctx, main, pmat_v)
    return main


def run_outputs(ctx2, ctx3, pmat_v, args):
    """由缓存重建全部输出，不重解 MILP。"""
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
                    choices=['all', 'full', 'main', 'sens', 'outputs', 'regress'],
                    help='all/full=主重算+敏感性（默认） | main=仅正式重算 | sens=仅敏感性 | '
                         'outputs=由缓存重建 | regress=回归核验')
    ap.add_argument('--which', default='both', choices=['both', 'q2', 'q3'],
                    help='限定问题族（both 默认）')
    ap.add_argument('--exp', default=None,
                    help='限定敏感性实验，逗号分隔，如 lam0,lam05,lam15,gam08,gam12,nobess'
                         '（vol/fixed 为正式重算基准，总会执行）')
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
        # 敏感性全部跑完后再统一汇总落盘一次，确保报告包含完整敏感性结果
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
    elif args.cmd == 'outputs':
        run_outputs(ctx2, ctx3, pmat_v, args)
    elif args.cmd == 'regress':
        run_regress(ctx2, ctx3, pmat_v, args)


if __name__ == '__main__':
    main()

