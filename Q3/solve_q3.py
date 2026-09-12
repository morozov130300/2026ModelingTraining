# -*- coding: utf-8 -*-
"""
问题三：基于 MPC 的微网日内滚动购电策略 —— 确定性经济型 MPC 逐日滚动求解
================================================================
严格按《Q3/# 问题三：基于 MPC 的微网日内滚动购电策略模型.txt》实现，
求解器遵循 AGENTS.md（scipy.optimize.milp / linprog，HiGHS），AGENTS.md 规定 8 进程并行。

已与用户确认的决策（2025 竞赛 C 题问题三）：
  1. 负载口径：制定计划与滚动优化时直接使用附件2当天实际负载 L_{d,t}
     （附件3只给光伏预报，题目未给负载预报；各对照方案对称使用同一负载，公平可比）；
  2. 光伏预报口径：附件3的"预报h小时"解释为未来 24 小时整点瞬时值（元组说明：
     附件3 时间列与附件1一致为"时段终点"口径，即预报h小时↔h:00 整点值），
     10 分钟时段用相邻整点线性插值（文档第10节第6条公式）；
     B_k 起点锚点取"时刻之前的实现值"：0:00 锚点 = 昨日 24:00 实际光伏（附件2 第144列），
     6:00/12:00/18:00 锚点 = 当天 5:00/11:00/17:00 实际光伏（附件2 第36/72/108列）；
     夜间预报小负值按 max(0,·) 截断后再插值；
     k 次滚动（6/12/18:00）使用附件3该时刻行发布的预报 F[di, k]，0:00 基准用 F[di, 0]；
  3. 结算口径：增量结算（文档第6节建议口径）——仅用于报告/费用汇总层：
     总费用 C^total = C^plan + Σ_{k:采纳}(1.5cU − 0.5cW)Δt + 5cR^real·Δt。
     上调付 1.5 倍；下调退回原购电费、只付 0.5 倍违约费（净 −0.5cW）。
     紧急购电费用只按实际执行阶段发生的 R^real 计算（模型内 5cR̂ 项仅用于预测成本比较，不重复计费）。
     注意：k≥1 滚动模型的**优化目标**采用文档第6节第一式 1.5cU+0.5cW+5cR̂（违约口径，
     W 项为正系数以保证问题有界），增量口径只出现在 adj/scen 的费用汇总与交付表中。
  4. 求解方法：k=0 基准计划与 k≥1 滚动子问题均为 MILP（保留充放电互斥二元 u，文档第12节
     主模型口径）；k=0 追加 LP 松弛检验互斥性与费用一致性（文档第12节第2-4条），
     并追加词典序二阶段（费用 C*+ε 下最小化储能吞吐量 Σ(C+D)Δt，同 Q2 决策4，
     消除 C/D 无成本导致的退化循环）；滚动子问题为单阶段（文档未要求，
     其目标含 5cR 项已主动塑形 C/D，互斥性由 u 二元变量保证）。
  5. "是否需要调整"判定（文档第10节第18条）：对第 k 次滚动，比较
     J_adapt（重优化模型目标，文档第6节原式 1.5cU+0.5cW+5cR̂，正系数保证有界）
     与 J_frozen（不调购电、沿用上一版计划的预测紧急购电费 5c·R_frozen，
     R_frozen = max(0, L+C_prev−D_prev−Q_prev−Ĝ_k)）；
     J_adapt < J_frozen 才采纳调整，否则冻结沿用上一版（购电与储能计划均不变）。
     注意：模型目标用第6节第一式（违约口径，W 项 +0.5cW）；报告/结算用增量口径
     （决策3，W 项净 −0.5cW）——两者不可互换（−0.5cW 作目标会导致 W 无界）。
  6. result3.xlsx（按附件5模板原结构填写，不新建工作簿）：
     - "计划购电量"表：0:00 预报制定的 Q^(0) 各时段购电量(kWh)、全天购电量、全天购电费
       （购电费 = Σ c_t·Q^(0)·Δt，纯计划费用口径，与 result2 一致）；
     - "调整购电量"表：S3 方案最终生效购电策略 Q^eff 各时段、全天购电量、
       全天购电费 = Σ c_t·Q^eff·Δt（生效购电纯费用口径，用户确认"生效购电纯费用"；
       增量结算/违约/紧急费用只出现在对照汇总与论文分析中，不在模板表内重复计费）；
     - "充放电量"表：S3 各时段实际生效版本的充放电计划（4小时段分块），
       0:00 储电量 = 6000（模型初始值），24:00 储电量 = 实际执行轨迹的日末储电量 E^real_145；
     - "紧急购电量"表：S3 实际执行中连续紧急购电段（R^real>0 的连续十分钟段合并），
       每段一行、超段追加行，无紧急购电的日期保留日期行、时段与购电量留空（同 Q2 决策10）。
  7. 对照实验（文档第11节）：四组方案 S0~S3 全部各跑 334 天（S_j = 使用
     {0:00, 6:00, ..., 6h·j} 共 j+1 个时刻的预报）：
     指标 = 全年 C^total、C^emg、紧急购电量 E^emg、调整电量 A^total；
     边际收益 ΔC_6 = C(S0)−C(S1)、ΔC_12 = C(S1)−C(S2)、ΔC_18 = C(S2)−C(S3)。
     4 组方案的 0:00 基准计划相同（同一 0:00 预报），差异仅在执行版本链，
     故每天只需解 1 个 k=0 MILP + 3 个 k≥1 滚动子问题（j=3 主路径共享），
     各 S_j 由同一套 k 解按版本链切片模拟得到（确定性、与串行逐日一致）。
  8. 执行模拟（回测）：B_k 区间内使用"该区间最终生效版本"的计划 Q/C/D，
     按附件2实际 L、G 逐时段结算（信息边界：k≥1 模型只用 k 时刻前信息——
     实际 L/G 仅在模型锁定后用于执行模拟，k 次滚动的 E 初值 E^real 来自此前区间的执行轨迹）；
     过放电截断沿用 Q2 规则（计划放电超过可吸收量 L+C 时 D_eff = L+C，未执行部分留在电池内，
     日末储电量以实际轨迹为准，剩余顺延次日，计划仍以 6000 起步，同 Q2 先例）；
     若 E^real(τ_k+1) 因过放电截断越出 [1200,10800]，滚动子问题初值裁剪到边界并记录告警。
  9. 储能参数（文档第3节/附录1）：E∈[1200,10800] kWh，P_ch=P_dis≤5000 kW，
     η_ch=η_dis=0.9，每日 E_1=E_145=6000（模型层面硬约束）；紧急购电价 κ=5 倍；
     P_buy,max = ∞（题目未给并网容量，模型保留参数位，同 Q2 决策8，不擅自设 5000）。
 10. 代表日 = 2025.3.20 / 6.21 / 9.23 / 12.21（题目表1-3指定日期，同 Q2），
     表1/表2/表3 按 S3 方案、4 代表日纵向分块交付。
 11. 运行流程：先 4 个代表日验证，再全量 334 天；每日结果落盘缓存可断点续跑；
     全量按天分派 8 进程并行（天与天之间完全独立，并行结果与串行逐日一致）。

模型与文档对应关系：
  - 第2节   时间离散与执行区间 B_k/T_k、Q^eff 定义：常量与版本链切片
  - 第5节   0:00 基准计划模型（含充放电互斥）：build_k0 / 变量布局
  - 第6/7节 滚动调整模型（U/W、增量结算目标、功率平衡、储能动态）：build_kk
  - 第8节   实际执行与紧急购电：run_block（过放电截断 + R^real 结算）
  - 第9节   总费用（增量结算口径）：solve_day 场景汇总
  - 第10节  MPC 算法步骤：solve_day 主循环
  - 第11节  对照实验 S0~S3 与边际收益：场景切片 + 年度汇总
  - 第12节  求解算法（MILP 主模型 + LP 松弛检验）：solve_milp / lp_relaxation_check

用法（在装有 numpy/scipy/openpyxl 的目标机上运行）：
  python solve_q3.py                       # 无参数=一键全量334天（默认断点续跑+8进程并行）
  python solve_q3.py repdays              # 4 个代表日求解与全面核验
  python solve_q3.py full --force         # 忽略缓存全部重解
  python solve_q3.py full --limit 10      # 冒烟测试：仅前 10 个决策日
  python solve_q3.py full --workers 8     # 指定并行进程数（AGENTS 固定 8）
  python solve_q3.py outputs              # 由缓存重建全部输出文件
  Windows 下也可直接双击同目录的 一键运行.bat
"""

import argparse
import multiprocessing
import os
import pickle
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix

from datetime import date, datetime, time as dtime, timedelta

import openpyxl

# ========== 路径 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ATTACH_DIR = os.path.join(PROJECT_ROOT, '附件')
PRICE_FILE = os.path.join(ATTACH_DIR, '附件1.xlsx')          # 附件1：电价 c_t（单日曲线）
ACT_FILE = os.path.join(ATTACH_DIR, '附件2.xlsx')            # 附件2：实际负载与光伏（365×144）
FCST_FILE = os.path.join(ATTACH_DIR, '附件3.xlsx')           # 附件3：4 时刻光伏预报（365天×4×24）
TEMPLATE_FILE = os.path.join(ATTACH_DIR, '附件5', 'result3.xlsx')  # 附件5模板（只读）
OUT_FILE = os.path.join(SCRIPT_DIR, 'result3.xlsx')
COMP_FILE = os.path.join(SCRIPT_DIR, '对照汇总.xlsx')
TABLE1_FILE = os.path.join(SCRIPT_DIR, '表1.xlsx')
TABLE2_FILE = os.path.join(SCRIPT_DIR, '表2.xlsx')
TABLE3_FILE = os.path.join(SCRIPT_DIR, '表3.xlsx')
CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')

# ========== 常量（文档第3节 + 附录1） ==========
N = 144                       # 每天十分钟时段数
DT = 10.0 / 60.0             # Δt = 1/6 h
ETA_CH = 0.9
ETA_DIS = 0.9
E_MIN, E_MAX = 1200.0, 10800.0
E_INIT = 6000.0              # E_{d,1}（文档第3节：每日首末 6000）
P_CH_MAX = 5000.0
P_DIS_MAX = 5000.0
KAPPA = 5.0                  # 紧急购电加价系数

# 滚动时刻（1-based τ_k，0-based 区间起点 i0）：τ = 0,36,72,108 → i0 = 0,36,72,108
TAU = [0, 36, 72, 108]
BLOCKS = [(0, 36), (36, 72), (72, 108), (108, 144)]          # 执行区间 B_0..B_3

# 求解与校验
MIP_GAP = 1e-9
TIME_LIMIT_DEFAULT = 300.0
TAU_EXCL = 1e-6            # 充放电互斥校验阈值 (kW)
EPS_REL = 1e-6             # 词典序二阶段费用松弛系数（同 Q2）

# 决策期（与题目一致：2025.2.1 ~ 12.31）
FIRST_DAY = date(2025, 2, 1)
LAST_DAY = date(2025, 12, 31)
REP_DAYS = [date(2025, 3, 20), date(2025, 6, 21), date(2025, 9, 23), date(2025, 12, 21)]

# 验证分析参数（全量缓存读取，纯统计；不改变任何 MPC 目标/约束/求解流程）
VERIFY_EPS = 1e-4              # δ_{d,k} 的 ε 阈值（元）：排除求解器数值误差（mip_rel_gap=1e-9 量级）
BOOT_N_RESAMPLE = 10000        # 移动块自助法重抽次数（7 天主口径，3/14 天同数稳健性对照）
BOOT_SEED = 42                # 固定随机种子，保证 CI 可复现
BOOT_BLOCKS = (7, 3, 14)      # 块长（天）：7 为主口径，3/14 为稳健性对照


# ========== 1. 数据读取 ==========
def read_prices(path):
    """附件1：144 个时段电价 c_t（元/kWh），时段终点口径（同 Q2）。
    兼容共享目录可能带来的空字符串单元格（'') 与短行。"""
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.active
    prices = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        v = row[1] if len(row) > 1 else None
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        prices.append(float(v))
    wb.close()
    if len(prices) != N:
        raise ValueError(f"附件1电价期望{N}个时段，实际{len(prices)}个")
    return np.asarray(prices, dtype=float)


def read_actuals(path):
    """附件2：两张 365x144 宽表（小区负载 / 光伏实际功率），单位 kW（同 Q2 读取规则）。"""
    wb = openpyxl.load_workbook(path, read_only=True)
    sheets = wb.sheetnames
    load_sheet = next(s for s in sheets if '负载' in s)
    pv_sheet = next(s for s in sheets if '光伏' in s)

    def read_sheet(name):
        ws = wb[name]
        dates, rows = [], []
        for row in ws.iter_rows(min_row=2, values_only=True):
            d0 = row[0]
            if d0 is None or (isinstance(d0, str) and not d0.strip()):
                continue
            vals = list(row[1:1 + N])
            vals += [None] * (N - len(vals))
            if any(v is None for v in vals):
                raise ValueError(f"{name} 行 {d0} 数据不足或含缺失")
            # 统一为 date（共享目录可能把日期读成 '2025-1-1' 字符串）
            if isinstance(d0, datetime):
                d_norm = d0.date()
            elif isinstance(d0, date):
                d_norm = d0
            else:
                s = str(d0).strip()
                parts = s.replace('/', '-').replace('.', '-').replace('年', '-').replace('月', '-').replace('日', '').split('-')
                parts = [p for p in parts if p]
                if len(parts) != 3:
                    raise ValueError(f"{name} 行无法解析日期: {d0!r}")
                d_norm = date(int(parts[0]), int(parts[1]), int(parts[2]))
            dates.append(d_norm)
            rows.append([float(v) for v in vals])
        return dates, np.asarray(rows, dtype=float)

    dates_l, L = read_sheet(load_sheet)
    dates_g, G = read_sheet(pv_sheet)
    wb.close()
    if dates_l != dates_g:
        raise ValueError("附件2两张工作表日期不一致")
    expected = []
    d = dates_l[0]
    while d <= dates_l[-1]:
        expected.append(d)
        d += timedelta(days=1)
    if dates_l != expected:
        raise ValueError("附件2日期不连续")
    print(f"[数据] 附件2: {L.shape[0]}天x{N}时段, 负载[{L.min():.4f},{L.max():.4f}] kW, "
          f"光伏[{G.min():.4f},{G.max():.4f}] kW（缺失0已校验）")
    return dates_l, L, G


def read_forecasts(path):
    """附件3：365 天 × 4 时刻(0:00/6:00/12:00/18:00) × 24 个"预报h小时"值（kW）。
    日期列仅每天首行有值，其余 3 行为空（共享目录可能读成空字符串 ''，需兼容）；
    时间列取值 '0:00','6:00','12:00','18:00'（也可能为 datetime.time）。
    返回 {date: (4,24)}，按天对齐校验。"""
    tidx = {'0:00': 0, '6:00': 1, '12:00': 2, '18:00': 3}

    def parse_time(t0):
        if t0 is None:
            return None
        s = str(t0).strip()
        # 规范化：'0:00' / '00:00' / '0:00:00' / datetime.time
        if isinstance(t0, dtime):
            s = f"{t0.hour}:{t0.minute:02d}"
        s2 = s.split('.')[0]
        m = s2.split(':')
        if len(m) >= 2:
            s2 = f"{int(m[0])}:{m[1]}"
        return tidx.get(s2)

    def parse_date(d0):
        if isinstance(d0, datetime):
            return d0.date()
        s = str(d0).strip()
        if not s:
            return None
        # 兼容 '2025-1-1'、'2025/1/1'、'2025年1月1日'、'01-01' 等
        parts = s.replace('/', '-').replace('.', '-').replace('年', '-').replace('月', '-').replace('日', '').split('-')
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) == 3:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        raise ValueError(f"附件3无法解析日期单元格: {d0!r}")

    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.active
    data = {}
    seen = {}
    cur = None
    for row in ws.iter_rows(min_row=2, values_only=True):
        d0 = row[0] if len(row) > 0 else None
        t0 = row[1] if len(row) > 1 else None
        if d0 is not None and str(d0).strip() != '':
            cur = parse_date(d0)
        if cur is None:
            raise ValueError("附件3首行缺少日期")
        ti = parse_time(t0)
        if ti is None:
            # 时刻列读成空值（共享目录可能）：按 0:00/6:00/12:00/18:00 顺序回退
            got = len(seen.get(cur, ()))
            if got >= 4:
                raise ValueError(f"附件3 {cur} 时刻标签异常({t0!r})且当天已有4行，无法定位")
            ti = got
        vals = list(row[2:26])
        vals += [None] * (24 - len(vals))
        if any(v is None for v in vals):
            raise ValueError(f"附件3 {cur} 第{ti}行含缺失")
        arr = np.asarray(vals, dtype=float)
        data.setdefault(cur, np.zeros((4, 24)))[ti] = arr
        seen.setdefault(cur, set()).add(ti)
    wb.close()
    if len(data) != 365:
        raise ValueError(f"附件3期望365天，实际{len(data)}天")
    for d, a in data.items():
        if a.shape != (4, 24) or not np.all(np.isfinite(a)):
            raise ValueError(f"附件3 {d} 结构异常")
        if len(seen.get(d, ())) != 4:
            raise ValueError(f"附件3 {d} 预报时刻数={len(seen.get(d, ()))}，应为 4")
    return data


# ========== 2. 光伏预报 → 10 分钟级（用户决策2：整点瞬时 + 线性插值） ==========
def pv_curve(ctx, di, k):
    """第 di 天第 k 次预测对 T_k 时段的光伏预测曲线（kW）。
    k=0: 锚点 P_0 = 昨日 24:00 实际光伏（附件2 第144列），P_1..P_24 = 0:00 预报的 24 个值；
    k≥1: 锚点 P_{h0} = 当天 i0−1 槽（发布时刻前一槽）实际光伏，h0 = i0//6（6/12/18 点），
         P_{h0+1}..P_24 = 该时刻"预报1小时..预报(24−h0)小时"（18/12/6 个值）；
    时段 j（0-based）取值时刻 m = 10j+5 分钟，h = m//60，f = (m−60h)/60：
         Ĝ_j = (1−f)·P_h + f·P_{h+1}（文档第10节第6条公式）。
    夜间预报小负值先按 max(0,·) 截断。"""
    G, F = ctx['G'], ctx['FST']
    gday = G[di]
    i0 = TAU[k]
    P = np.zeros(25)
    if k == 0:
        P[0] = max(0.0, float(G[di - 1, 143]))            # 昨日 24:00 实际值
        P[1:25] = np.maximum(F[di, 0], 0.0)
    else:
        h0 = i0 // 6                                        # 锚点整点：6/12/18 点
        P[h0] = max(0.0, float(gday[i0 - 1]))              # 发布时刻前一槽实际值（5:00/11:00/17:00）
        P[h0 + 1:25] = np.maximum(F[di, k], 0.0)[:24 - h0]  # 该时刻 k 发布的预报（6/12/18:00）
    m = 10 * np.arange(i0, N) + 5.0
    h = (m // 60).astype(int)
    f = (m - 60.0 * h) / 60.0
    val = (1.0 - f) * P[h] + f * P[h + 1]
    return np.maximum(val, 0.0)


# ========== 3. 模型构建与求解 ==========
def _assemble(eq_r, eq_c, eq_v, beq, ub_r, ub_c, ub_v, bub, lo, hi, integrality, nvars):
    """按 COO（data, (row, col)）组装稀疏矩阵。
    eq_r/eq_c/eq_v: 等式约束的行号/列号/系数列表；beq: 等式右端。
    ub_r/ub_c/ub_v: 不等式（≤）的行号/列号/系数；bub: 不等式右端。
    nvars: 变量总数（用于确定稀疏矩阵宽度）。"""
    A_eq = csr_matrix((np.asarray(eq_v, dtype=float),
                       (np.asarray(eq_r), np.asarray(eq_c))), shape=(len(beq), nvars))
    A_ub = csr_matrix((np.asarray(ub_v, dtype=float),
                       (np.asarray(ub_r), np.asarray(ub_c))), shape=(len(bub), nvars))
    return {'A_eq': A_eq, 'b_eq': np.asarray(beq, dtype=float),
            'A_ub': A_ub, 'b_ub': np.asarray(bub, dtype=float),
            'lo': lo, 'hi': hi, 'integrality': integrality}


def build_k0(prices, Ld, Ghat, E_init=E_INIT, E_end=E_INIT):
    """文档第5节 0:00 基准计划模型（MILP，保留充放电互斥 u）。
    变量布局（N=144）：
      Q: 0..N-1          计划购电功率
      Puse: N..2N-1      计划购电使用
      S: 2N..3N-1        未使用计划购电
      R: 3N..4N-1        预测紧急购电
      Pv: 4N..5N-1      光伏利用
      K: 5N..6N-1       弃光
      C: 6N..7N-1       充电
      D: 7N..8N-1       放电
      E: 8N..8N+144     145 个 SOC 点（首尾固定）
      u: 9N..9N+143     充电状态二元
    目标（第5节）：min Σ(c_t·Q + 5c_t·R)Δt。"""
    n = N
    iQ, iP, iS, iR, iPv, iK, iC, iD = [s * n for s in range(8)]
    iE = 8 * n
    iU = iE + n + 1
    tot = iU + n

    eq_r, eq_c, eq_v, beq = [], [], [], []

    def add_eq(entries, b):
        r = len(beq)
        for ci, v in entries:
            eq_r.append(r); eq_c.append(ci); eq_v.append(v)
        beq.append(float(b))

    for t in range(n):                                   # SOC 转移 + 三组功率平衡
        add_eq([(iE + t + 1, 1.0), (iE + t, -1.0),
                (iC + t, -ETA_CH * DT), (iD + t, DT / ETA_DIS)], 0.0)
        add_eq([(iP + t, 1.0), (iR + t, 1.0), (iPv + t, 1.0),
                (iD + t, 1.0), (iC + t, -1.0)], Ld[t])
        add_eq([(iP + t, 1.0), (iS + t, 1.0), (iQ + t, -1.0)], 0.0)
        add_eq([(iPv + t, 1.0), (iK + t, 1.0)], Ghat[t])
    add_eq([(iE, 1.0)], E_init)
    add_eq([(iE + n, 1.0)], E_end)

    ub_r, ub_c, ub_v, bub = [], [], [], []

    def add_ub(entries, b):
        r = len(bub)
        for ci, v in entries:
            ub_r.append(r); ub_c.append(ci); ub_v.append(v)
        bub.append(float(b))

    for t in range(n):                                   # C ≤ 5000u; D ≤ 5000(1−u)
        add_ub([(iC + t, 1.0), (iU + t, -P_CH_MAX)], 0.0)
        add_ub([(iD + t, 1.0), (iU + t, P_DIS_MAX)], P_DIS_MAX)

    lo = np.zeros(tot)
    hi = np.full(tot, np.inf)                              # Q/Puse/S/R/Pv/K 无上界（P_buy_max=∞，决策9）
    hi[iC:iC + n] = P_CH_MAX
    hi[iD:iD + n] = P_DIS_MAX
    lo[iE:iE + n + 1] = E_MIN
    hi[iE:iE + n + 1] = E_MAX
    lo[iE] = hi[iE] = E_init
    lo[iE + n] = hi[iE + n] = E_end
    hi[iU:iU + n] = 1.0
    integrality = np.zeros(tot)
    integrality[iU:iU + n] = 1

    c_cost = np.zeros(tot)
    c_cost[iQ:iQ + n] = prices * DT                        # 计划购电费
    c_cost[iR:iR + n] = KAPPA * prices * DT                 # 预测紧急购电费（5 倍）
    c_thru = np.zeros(tot)
    c_thru[iC:iC + n] = DT                                   # 词典序二阶段目标：Σ(C+D)Δt
    c_thru[iD:iD + n] = DT

    model = _assemble(eq_r, eq_c, eq_v, beq, ub_r, ub_c, ub_v, bub,
                      lo, hi, integrality, tot)
    model.update({'n_tot': tot, 'c_cost': c_cost, 'c_thru': c_thru})
    return model


def build_kk(k, prices, Ld, Ghat, Qprev, Cprev, Dprev, E_init):
    """文档第6/7节 第 k 次（k≥1）滚动调整模型（MILP，T_k=[i0..144) 局部索引 j=0..nt−1）。
    目标（第6节第一式，违约口径）：min Σ[1.5c·U + 0.5c·W + 5c·R]Δt。
    （W 项必须为正系数：W 无上界，负系数会使问题无界；增量结算 1.5cU−0.5cW
     仅用于报告/费用汇总层，见决策3/5。）
    变量布局（nt = 144−i0）：
      Q/Puse/S/R/Pv/K/C/D/U/W/u 各 nt 个（基址 j*nt），E: 11nt..12nt（nt+1 个 SOC 点，首尾固定）"""
    i0 = TAU[k]
    nt = N - i0
    E_init_c = float(np.clip(E_init, E_MIN, E_MAX))      # 决策8：越界裁剪并记录
    iQ, iP, iS, iR, iPv, iK, iC, iD, iUv, iW = [s * nt for s in range(10)]
    iE = 11 * nt
    iUb = 10 * nt
    tot = iE + nt + 1
    Ls = Ld[i0:]
    Qp, Cp, Dp = Qprev[i0:], Cprev[i0:], Dprev[i0:]

    eq_r, eq_c, eq_v, beq = [], [], [], []

    def add_eq(entries, b):
        r = len(beq)
        for ci, v in entries:
            eq_r.append(r); eq_c.append(ci); eq_v.append(v)
        beq.append(float(b))

    for j in range(nt):
        g = i0 + j
        add_eq([(iE + j + 1, 1.0), (iE + j, -1.0),
                (iC + j, -ETA_CH * DT), (iD + j, DT / ETA_DIS)], 0.0)
        add_eq([(iP + j, 1.0), (iR + j, 1.0), (iPv + j, 1.0),
                (iD + j, 1.0), (iC + j, -1.0)], Ls[j])
        add_eq([(iP + j, 1.0), (iS + j, 1.0), (iQ + j, -1.0)], 0.0)
        add_eq([(iPv + j, 1.0), (iK + j, 1.0)], Ghat[j])
    add_eq([(iE, 1.0)], E_init_c)
    add_eq([(iE + nt, 1.0)], E_INIT)                        # 当天结束仍回到 6000（文档第7节）

    ub_r, ub_c, ub_v, bub = [], [], [], []

    def add_ub(entries, b):
        r = len(bub)
        for ci, v in entries:
            ub_r.append(r); ub_c.append(ci); ub_v.append(v)
        bub.append(float(b))

    for j in range(nt):
        add_ub([(iC + j, 1.0), (iUb + j, -P_CH_MAX)], 0.0)
        add_ub([(iD + j, 1.0), (iUb + j, P_DIS_MAX)], P_DIS_MAX)
        add_ub([(iUv + j, -1.0), (iQ + j, 1.0)], Qp[j])        # U ≥ Q − Qprev
        add_ub([(iW + j, -1.0), (iQ + j, -1.0)], -Qp[j])       # W ≥ Qprev − Q

    lo = np.zeros(tot)
    hi = np.full(tot, np.inf)
    hi[iC:iC + nt] = P_CH_MAX
    hi[iD:iD + nt] = P_DIS_MAX
    lo[iE:iE + nt + 1] = E_MIN
    hi[iE:iE + nt + 1] = E_MAX
    lo[iE] = hi[iE] = E_init_c
    lo[iE + nt] = hi[iE + nt] = E_INIT
    hi[iUb:iUb + nt] = 1.0
    integrality = np.zeros(tot)
    integrality[iUb:iUb + nt] = 1

    c_cost = np.zeros(tot)
    c_cost[iUv:iUv + nt] = 1.5 * prices[i0:] * DT
    c_cost[iW:iW + nt] = 0.5 * prices[i0:] * DT   # 文档第6节目标原式（正系数，保证 W 有界；
    c_cost[iR:iR + nt] = KAPPA * prices[i0:] * DT  # 增量结算 1.5cU−0.5cW 仅作报告口径，见决策3）

    model = _assemble(eq_r, eq_c, eq_v, beq, ub_r, ub_c, ub_v, bub,
                      lo, hi, integrality, tot)
    model.update({'nt': nt, 'i0': i0, 'n_tot': tot, 'E_init_used': E_init_c,
                  'c_cost': c_cost})
    return model


def solve_milp(model, c, time_limit):
    res = milp(c=c,
               constraints=[LinearConstraint(model['A_eq'], model['b_eq'], model['b_eq']),
                            LinearConstraint(model['A_ub'], -np.inf, model['b_ub'])],
               integrality=model['integrality'],
               bounds=Bounds(model['lo'], model['hi']),
               options={'mip_rel_gap': MIP_GAP, 'time_limit': time_limit, 'presolve': True})
    if not res.success or res.x is None:
        raise RuntimeError(f"MILP 求解失败: status={res.status}, msg={res.message}")
    return res


def lp_relaxation_check(model, c):
    """文档第12节第2-4条：去掉二元互斥求 LP 松弛，检验充放电互斥性与费用一致性。"""
    res = linprog(c, A_ub=model['A_ub'], b_ub=model['b_ub'],
                  A_eq=model['A_eq'], b_eq=model['b_eq'],
                  bounds=list(zip(model['lo'], model['hi'])), method='highs')
    if not res.success:
        raise RuntimeError(f"LP 松弛求解失败: {res.message}")
    return res


# ========== 4. 单日求解：k=0 基准 + k=1..3 滚动 + 四场景模拟 ==========
def _merge_segments_local(R, g0):
    """R（局部数组）>阈值的连续段 → [(起分钟, 止分钟, 电量kWh)]，起始加全局偏移 g0。"""
    segs = []
    above = R > 1e-6
    t = 0
    while t < len(R):
        if above[t]:
            s = t
            while t < len(R) and above[t]:
                t += 1
            segs.append(((g0 + s) * 10, (g0 + t) * 10, float(np.sum(R[s:t]) * DT)))
        else:
            t += 1
    return segs


def merge_blocks(segs_per_block):
    """把 4 个 B 区间各自的段列表拼接，并合并跨块边界连续的段
    （前块止分钟 == 后块起分钟 时合并电量）。"""
    out = []
    for segs in segs_per_block:
        for (s, e, en) in segs:
            if out and out[-1][1] == s:
                out[-1] = (out[-1][0], e, out[-1][2] + en)
            else:
                out.append([s, e, en])
    return out


def run_block(Ld, Gd, prices, i0, i1, Qv, Cv, Dv, E_start):
    """文档第8节 实际执行结算（同 Q2 过放电截断规则，决策8）。
    B 区间 [i0,i1)：用版本计划的 Q/C/D + 实际 L/G 逐时段算 R^real 与 SOC 轨迹。"""
    Ls = Ld[i0:i1]
    Gs = Gd[i0:i1]
    Qs, Cs, Ds = Qv[i0:i1], Cv[i0:i1], Dv[i0:i1]
    rd = Ls + Cs - Ds
    over_dis = int(np.sum(rd < -1e-6))
    rdp = np.maximum(rd, 0.0)
    xdef = rdp - Gs - Qs
    R = np.maximum(xdef, 0.0)
    D_eff = np.where(rd >= 0.0, Ds, Ls + Cs)                # 过放电截断（未执行放电留电池内）
    E_end = E_start + ETA_CH * DT * float(Cs.sum()) - DT / ETA_DIS * float(D_eff.sum())
    segs = _merge_segments_local(R, i0)
    c_emg = KAPPA * DT * float(prices[i0:i1] @ R)
    e_emg = DT * float(R.sum())
    return E_end, R, segs, c_emg, e_emg, over_dis


def solve_day(ctx, di, time_limit=TIME_LIMIT_DEFAULT):
    prices = ctx['prices']
    Ld = ctx['L'][di]
    Gd = ctx['G'][di]
    d = ctx['dates'][di]

    # ---- k=0：0:00 基准计划（文档第5节 + 第12节 MILP 主模型 + LP 松弛检验） ----
    # 词典序二阶段（同 Q2）：费用最优下最小化储能吞吐量，消除 C/D 无成本导致的退化循环。
    t0 = time.perf_counter()
    Ghat0 = pv_curve(ctx, di, 0)
    m0 = build_k0(prices, Ld, Ghat0)
    r0 = solve_milp(m0, m0['c_cost'], time_limit)
    J0 = float(r0.fun)
    c_star = J0
    # 第二阶段：在费用 ≤ C*+ε 下最小化吞吐量 Σ(C+D)Δt
    eps = EPS_REL * max(1.0, abs(c_star))
    A_ub2 = np.vstack([m0['A_ub'].toarray(), m0['c_cost'].reshape(1, -1)])
    b_ub2 = np.concatenate([m0['b_ub'], [c_star + eps]])
    m0_lex = {**m0, 'A_ub': csr_matrix(A_ub2), 'b_ub': b_ub2}
    r0lex = solve_milp(m0_lex, m0['c_thru'], time_limit)
    x0 = r0lex.x
    n = N
    iQ, iP, iS, iR, iPv, iK, iC, iD = [s * n for s in range(8)]
    iE = 8 * n
    iU = iE + n + 1
    Q0, C0, D0 = x0[iQ:iQ + n], x0[iC:iC + n], x0[iD:iD + n]
    E0 = x0[iE:iE + n + 1]
    Q0 = np.where(np.abs(Q0) < 1e-9, 0.0, Q0)
    C0 = np.where(np.abs(C0) < 1e-9, 0.0, C0)
    D0 = np.where(np.abs(D0) < 1e-9, 0.0, D0)
    J0 = float(r0.fun)
    lp0 = lp_relaxation_check(m0, m0['c_cost'])
    C0lp = lp0.x[iC:iC + n]
    D0lp = lp0.x[iD:iD + n]
    excl_milp = float(np.min(np.vstack([C0, D0]), axis=0).max())
    excl_lp = float(np.min(np.vstack([C0lp, D0lp]), axis=0).max())
    eq_res0 = float(np.abs(m0['A_eq'] @ x0 - m0['b_eq']).max())
    ub_res0 = max(0.0, float(np.max(m0['A_ub'] @ x0 - m0['b_ub']))) if m0['A_ub'].shape[0] else 0.0
    u_ok = bool(np.all(np.abs(x0[iU:iU + n] - np.round(x0[iU:iU + n])) <= 1e-9))
    lex_cost = float(m0['c_cost'] @ x0)                     # 词典序终解的原始费用（应 ≤ C*+ε）
    checks0 = {'J0': J0, 'eq_res': eq_res0, 'ub_res': ub_res0, 'excl_milp': excl_milp,
               'u_binary_ok': u_ok, 'lp_cost': float(lp0.fun),
               'lp_rel_diff': abs(J0 - float(lp0.fun)) / max(1.0, abs(J0)),
               'lp_excl_max': excl_lp, 'lp_excl_ok': bool(excl_lp <= TAU_EXCL),
               'soc_min': float(E0.min()), 'soc_max': float(E0.max()),
               'e0': float(E0[0]), 'e145': float(E0[-1]),
               'lex_cost': lex_cost, 'lex_within': bool(lex_cost <= c_star + eps + 1e-6)}
    if not checks0['lex_within']:
        print(f"[警告] {d} k=0 词典序终解费用 {lex_cost:.6f} 超出 C*+eps={c_star + eps:.6f}")
    c_plan = float(prices @ Q0) * DT                        # C^plan_d（文档第9节）
    plan_e = float(Q0.sum()) * DT
    mae_g0 = float(np.mean(np.abs(Gd - Ghat0)))

    # ---- 版本链（j=3 主路径）：k=1..3 依次 求解→判定→执行下一区间 ----
    curQ, curC, curD = Q0.copy(), C0.copy(), D0.copy()      # 最新采纳版本的 144 槽全局数组
    E = E_INIT
    a = [False, False, False]
    Jopt = [None, None, None]
    Jfrz = [None, None, None]
    adj = {}                                                 # k: (up_k, down_k, net_k, aw_k)
    verK = {}                                                # k: (Qk_local, Ck_local, Dk_local)
    E_init_used = {}
    Emark = {1: 0.0, 2: 0.0, 3: 0.0}
    Eclamped = {}
    checksK = {}

    # B_0 执行（版本 0）
    E, *_ = run_block(Ld, Gd, prices, 0, 36, curQ, curC, curD, E)
    Emark[1] = E

    for k in (1, 2, 3):
        i0 = TAU[k]
        Ghatk = pv_curve(ctx, di, k)
        Rf = np.maximum(Ld[i0:] + curC[i0:] - curD[i0:] - curQ[i0:] - Ghatk, 0.0)
        Jfrz[k - 1] = KAPPA * DT * float(prices[i0:] @ Rf)
        mk = build_kk(k, prices, Ld, Ghatk, curQ, curC, curD, Emark[k])
        rk = solve_milp(mk, mk['c_cost'], time_limit)
        xk = rk.x
        nt = mk['nt']
        Qk = xk[:nt]
        Ck = xk[6 * nt:7 * nt]
        Dk = xk[7 * nt:8 * nt]
        Uk = xk[8 * nt:9 * nt]
        Wk = xk[9 * nt:10 * nt]
        Ek = xk[11 * nt:12 * nt + 1]
        Qk = np.where(np.abs(Qk) < 1e-9, 0.0, Qk)
        Ck = np.where(np.abs(Ck) < 1e-9, 0.0, Ck)
        Dk = np.where(np.abs(Dk) < 1e-9, 0.0, Dk)
        Uk = np.where(np.abs(Uk) < 1e-9, 0.0, Uk)
        Wk = np.where(np.abs(Wk) < 1e-9, 0.0, Wk)
        Jopt[k - 1] = float(rk.fun)
        clamped = abs(mk['E_init_used'] - Emark[k]) > 1e-9
        E_init_used[k] = mk['E_init_used']
        Eclamped[k] = clamped
        base = prices[i0:] * DT
        sumU = float(Uk @ base)                     # Σ c·U·Δt
        sumW = float(Wk @ base)                     # Σ c·W·Δt
        c_up_k = 1.5 * sumU                          # C^up（1.5 倍电价，第9节）
        c_down_k = 0.5 * sumW                        # C^down（0.5 倍违约费，第9节）
        c_cancel_k = sumW                            # C^cancel（退回原购电费，第9节）
        net_k = c_up_k + c_down_k - c_cancel_k       # 增量净成本 = 1.5ΣcU − 0.5ΣcW（第6节）
        aw_k = float((Uk + Wk).sum()) * DT           # A^total 调整电量（第11节，纯电量）
        adj[k] = {'up': c_up_k, 'down': c_down_k, 'cancel': c_cancel_k,
                  'net': net_k, 'aw': aw_k}
        checksK[k] = {'eq_res': float(np.abs(mk['A_eq'] @ xk - mk['b_eq']).max()),
                      'ub_res': max(0.0, float(np.max(mk['A_ub'] @ xk - mk['b_ub']))),
                      'excl': float(np.min(np.vstack([Ck, Dk]), axis=0).max()),
                      'soc_min': float(Ek.min()), 'soc_max': float(Ek.max()),
                      'e_init_used': mk['E_init_used'], 'clamped': clamped,
                      'Jopt': Jopt[k - 1], 'Jfrz': Jfrz[k - 1]}
        # 文档第10节第18条：调整方案预测成本更低才采纳，否则冻结（U=W=0 无调整费）
        a[k - 1] = bool(Jopt[k - 1] < Jfrz[k - 1])
        if a[k - 1]:
            verK[k] = (Qk, Ck, Dk)
            curQ[i0:] = Qk
            curC[i0:] = Ck
            curD[i0:] = Dk
        # 执行 B_k（j=3 语义：采纳则版本 k，否则沿用 cur）→ 取 Emark 供下一次滚动初值
        g0, g1 = BLOCKS[k]
        E, *_ = run_block(Ld, Gd, prices, g0, g1, curQ, curC, curD, E)
        if k < 3:
            Emark[k + 1] = E

    # ---- 场景 S_j（j=0..3）：版本链切片 + 全日重新模拟（纯数组运算，确定性） ----
    scen = {}
    for j in range(4):
        Qj, Cj, Dj = Q0.copy(), C0.copy(), D0.copy()
        for k in range(1, j + 1):
            if a[k - 1]:
                i0 = TAU[k]
                Qk_, Ck_, Dk_ = verK[k]
                Qj[i0:] = Qk_
                Cj[i0:] = Ck_
                Dj[i0:] = Dk_
        E = E_INIT
        segs_per_block = []
        c_emg = 0.0
        e_emg = 0.0
        for m_ in range(4):
            g0, g1 = BLOCKS[m_]
            E, R_, segs_, cemg_, eemg_, _ = run_block(Ld, Gd, prices, g0, g1, Qj, Cj, Dj, E)
            segs_per_block.append(segs_)
            c_emg += cemg_
            e_emg += eemg_
        segs_all = merge_blocks(segs_per_block)
        c_adj = 0.0
        a_tot = 0.0
        up_tot = 0.0
        down_tot = 0.0
        cancel_tot = 0.0
        for k in range(1, j + 1):
            if a[k - 1]:
                up_tot += adj[k]['up']
                down_tot += adj[k]['down']
                cancel_tot += adj[k]['cancel']
                c_adj += adj[k]['net']
                a_tot += adj[k]['aw']
        total = c_plan + c_adj + c_emg
        scen[j] = {'C_total': total, 'C_plan': c_plan, 'C_adj': c_adj,
                   'C_up': up_tot, 'C_down': down_tot, 'C_cancel': cancel_tot,
                   'C_emg': c_emg, 'E_emg': e_emg, 'A_tot': a_tot, 'segs': segs_all,
                   'E_end': E, 'Q': Qj}

    # S3 交付量
    Cj3, Dj3 = _s3_cd(Q0, C0, D0, a, verK)
    rec = {
        'date': d,
        'Q0': Q0, 'C0': C0, 'D0': D0,
        'Qeff_s3': scen[3]['Q'], 'C_s3': Cj3, 'D_s3': Dj3,
        'E145_s3': scen[3]['E_end'],
        'a_flags': a, 'Emark': Emark, 'E_init_used': E_init_used,
        'scen': scen,
        'plan_energy': plan_e, 'plan_cost': c_plan,
        'eff_energy': float(scen[3]['Q'].sum()) * DT,
        'eff_cost': float(prices @ scen[3]['Q']) * DT,
        'mae_g0': mae_g0,
        'checks0': checks0, 'checksK': checksK, 'Eclamped': Eclamped,
        'plan_sec': time.perf_counter() - t0,
    }
    return rec


def _s3_cd(Q0, C0, D0, a, verK):
    """S3 场景各槽生效版本的 C/D 计划值（144 槽，未截断；交付'充放电量'表用）。"""
    Cj, Dj = C0.copy(), D0.copy()
    for k in range(1, 4):
        if a[k - 1]:
            i0 = TAU[k]
            Qk_, Ck_, Dk_ = verK[k]
            Cj[i0:] = Ck_
            Dj[i0:] = Dk_
    return Cj, Dj


# ========== 5. 缓存 ==========
CACHE_VERSION = 'q3-mpc-v2'   # 记录结构/模型变更时升级此值，旧缓存自动作废重解


def cache_path(d):
    return os.path.join(CACHE_DIR, f"day_{d.strftime('%Y%m%d')}.pkl")


def save_cache(rec):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path(rec['date']), 'wb') as f:
        pickle.dump({'_cache_version': CACHE_VERSION, 'rec': rec}, f)


def load_cache(d):
    p = cache_path(d)
    if os.path.exists(p):
        with open(p, 'rb') as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and obj.get('_cache_version') == CACHE_VERSION:
            return obj['rec']
        if isinstance(obj, dict) and 'rec' not in obj:
            return obj          # 兼容首版无前缀的旧缓存
    return None


# ========== 6. 输出（决策6：result3.xlsx 按模板原结构填写） ==========
def fmt_min(m):
    if m >= 24 * 60:
        return '24:00'
    return f"{m // 60}:{m % 60:02d}"


def write_result3(records, out_path=OUT_FILE):
    """按附件5模板原结构填写 result3.xlsx（不修改模板文件本身）。
    位置对应映射：第 j 个时间列 ↔ 当天第 j 个十分钟时段（同 Q2 决策9 填法）。"""
    wb = openpyxl.load_workbook(TEMPLATE_FILE)
    n_expected = (LAST_DAY - FIRST_DAY).days + 1

    def fill_plan_sheet(name, qty_key, energy, cost):
        ws = wb[name]
        hdr = [ws.cell(row=1, column=j).value for j in range(1, ws.max_column + 1)]
        j_energy = hdr.index('全天购电量') + 1
        j_cost = hdr.index('全天购电费') + 1
        date_row = {}
        for r in range(2, ws.max_row + 1):
            v = ws.cell(row=r, column=1).value
            v = v.date() if isinstance(v, datetime) else v
            if v is not None:
                if v in date_row:
                    raise ValueError(f"模板'{name}'日期重复: {v}")
                date_row[v] = r
        if len(date_row) != n_expected or min(date_row) != FIRST_DAY or max(date_row) != LAST_DAY:
            raise ValueError(f"模板'{name}'日期范围异常: {min(date_row)} ~ {max(date_row)} 共 {len(date_row)} 行")
        rec_by_date = {rec['date']: rec for rec in records}
        missing = [d for d in sorted(date_row) if d not in rec_by_date]
        if missing:
            print(f"[警告] '{name}' 部分输出：{len(missing)} 个日期尚无结果（冒烟测试模式），"
                  f"首日 {missing[0]}，末日 {missing[-1]}")
        for d, r in date_row.items():
            rec = rec_by_date.get(d)
            if rec is None:
                continue
            q = rec[qty_key]
            for t in range(N):
                ws.cell(row=r, column=2 + t).value = round(float(q[t]) * DT, 4)
            ws.cell(row=r, column=j_energy).value = round(energy(rec), 4)
            ws.cell(row=r, column=j_cost).value = round(cost(rec), 4)

    # ---- 计划购电量：0:00 基准 Q^(0)，纯计划费用口径（决策6） ----
    fill_plan_sheet('计划购电量', 'Q0',
                    lambda rec: rec['plan_energy'], lambda rec: rec['plan_cost'])
    # ---- 调整购电量：S3 生效策略 Q^eff，生效购电纯费用口径（决策6） ----
    fill_plan_sheet('调整购电量', 'Qeff_s3',
                    lambda rec: rec['eff_energy'], lambda rec: rec['eff_cost'])

    # ---- 充放电量（334 天 × 6 段，写入前清理模板占位行） ----
    ws2 = wb['充放电量']
    for r in range(2, ws2.max_row + 1):
        for c in range(1, ws2.max_column + 1):
            ws2.cell(row=r, column=c).value = None
    chunk_labels = ['0:00-4:00', '4:00-8:00', '8:00-12:00',
                    '12:00-16:00', '16:00-20:00', '20:00-24:00']
    chunks = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
    r = 2
    for rec in records:
        for k, (h1, h2) in enumerate(chunks):
            ws2.cell(row=r, column=2).value = chunk_labels[k]
            ws2.cell(row=r, column=3).value = round(float(np.sum(rec['C_s3'][h1 * 6:h2 * 6]) * DT), 4)
            ws2.cell(row=r, column=4).value = round(float(np.sum(rec['D_s3'][h1 * 6:h2 * 6]) * DT), 4)
            if k == 0:
                ws2.cell(row=r, column=1).value = datetime(rec['date'].year, rec['date'].month, rec['date'].day)
                ws2.cell(row=r, column=5).value = dtime(0, 0)
                ws2.cell(row=r, column=6).value = round(E_INIT, 4)          # 0:00 储电量（模型初值）
            elif k == 1:
                ws2.cell(row=r, column=5).value = '24:00'
                ws2.cell(row=r, column=6).value = round(rec['E145_s3'], 4)  # 24:00 实际执行轨迹（决策6）
            r += 1

    # ---- 紧急购电量：S3 执行段（每段一行，无段保留日期行，同 Q2 决策10） ----
    ws3 = wb['紧急购电量']
    for r0 in range(2, ws3.max_row + 1):
        for c in range(1, ws3.max_column + 1):
            ws3.cell(row=r0, column=c).value = None
    r = 2
    for rec in records:
        segs = rec['scen'][3]['segs']
        if not segs:
            ws3.cell(row=r, column=1).value = datetime(rec['date'].year, rec['date'].month, rec['date'].day)
            r += 1
        else:
            for k, (s_min, e_min, energy) in enumerate(segs):
                if k == 0:
                    ws3.cell(row=r, column=1).value = datetime(rec['date'].year, rec['date'].month, rec['date'].day)
                ws3.cell(row=r, column=2).value = f"{fmt_min(s_min)}-{fmt_min(e_min)}"
                ws3.cell(row=r, column=3).value = round(energy, 4)
                r += 1
    wb.save(out_path)
    print(f"[输出] result3.xlsx -> {out_path}（模板原结构，已填 {len(records)}/{n_expected} 天，"
          f"计划表=Q^(0)，调整表=S3生效Q^eff，充放电/紧急=S3执行）")


def write_comparison(records, out_path=COMP_FILE, write=True):
    """对照实验汇总（文档第11节）：S0~S3 全年指标 + 边际收益 ΔC6/ΔC12/ΔC18。
    write=False 时只计算并打印，不落盘（代表日模式避免用小样本覆盖全量结果）。"""
    keys = ['C_total', 'C_plan', 'C_adj', 'C_up', 'C_down', 'C_cancel', 'C_emg', 'E_emg', 'A_tot']
    yearly = {}
    for j in range(4):
        agg = {k: 0.0 for k in keys}
        agg['emg_days'] = 0
        for rec in records:
            s = rec['scen'][j]
            for k in keys:
                agg[k] += s[k]
            if s['E_emg'] > 1e-6:
                agg['emg_days'] += 1
        yearly[j] = agg
    n = len(records)
    # 各时刻采纳次数（决策日数中触发调整的日数）
    adopt = {k: sum(1 for rec in records if rec['a_flags'][k - 1]) for k in (1, 2, 3)}
    # 全年调整电量（按 S3 口径累计，即所有被采纳时刻的 U+W；文档第11节 A^total）
    A_all = sum(rec['scen'][3]['A_tot'] for rec in records)
    d6 = yearly[0]['C_total'] - yearly[1]['C_total']
    d12 = yearly[1]['C_total'] - yearly[2]['C_total']
    d18 = yearly[2]['C_total'] - yearly[3]['C_total']

    if write:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = '对照汇总'
        ws.cell(row=1, column=1, value='方案').font = openpyxl.styles.Font(bold=True)
        labels = ['使用预报时刻', '全年总费用C', '计划费C^plan', '调整净费C^adj',
                  '上调费C^up', '违约费C^down', '退费C^cancel', '紧急费C^emg',
                  '紧急电量E^emg(kWh)', '调整电量A(kWh)', '紧急购电天数']
        for i, v in enumerate(labels[1:], 2):
            ws.cell(row=1, column=i, value=v).font = openpyxl.styles.Font(bold=True)
        s_labels = ['S0: 仅0:00', 'S1: 0:00+6:00', 'S2: 0:00+6:00+12:00', 'S3: 全4时刻']
        colmap = ['C_total', 'C_plan', 'C_adj', 'C_up', 'C_down', 'C_cancel',
                  'C_emg', 'E_emg', 'A_tot', 'emg_days']
        for j in range(4):
            r = 2 + j
            ws.cell(row=r, column=1, value=s_labels[j])
            for i, k in enumerate(colmap, 2):
                ws.cell(row=r, column=i).value = round(yearly[j][k], 4) if k != 'emg_days' else yearly[j][k]
        r = 7
        for label, val in (('ΔC_6 = C(S0)−C(S1)', d6), ('ΔC_12 = C(S1)−C(S2)', d12),
                            ('ΔC_18 = C(S2)−C(S3)', d18)):
            ws.cell(row=r, column=1, value=label)
            ws.cell(row=r, column=2).value = round(val, 4)
            r += 1
        r += 1
        ws.cell(row=r, column=1, value='各时刻采纳调整天数(共%d天)' % n).font = openpyxl.styles.Font(bold=True)
        r += 1
        for k in (1, 2, 3):
            ws.cell(row=r, column=1, value=f"{TAU[k]//6}:00 采纳={adopt[k]} 天 ({adopt[k]/n:.1%})")
            r += 1
        wb.save(out_path)
        print(f"[输出] 对照汇总.xlsx -> {out_path}")

    # 终端摘要（4 位小数）
    print("\n[对照实验] 指标（S0~S3，文档第11节；样本 %d 天）" % n)
    print("=" * 72)
    print(f"{'方案':<20}{'C^total':>13}{'C^plan':>13}{'C^adj':>11}{'C^emg':>11}{'E^emg kWh':>11}")
    s_labels = ['S0: 仅0:00', 'S1: 0:00+6:00', 'S2: +12:00', 'S3: +18:00']
    for j in range(4):
        a = yearly[j]
        print(f"{s_labels[j]:<20}{a['C_total']:>13.4f}{a['C_plan']:>13.4f}"
              f"{a['C_adj']:>11.4f}{a['C_emg']:>11.4f}{a['E_emg']:>11.4f}")
    print("-" * 72)
    print(f"边际收益: ΔC_6={d6:.4f} 元, ΔC_12={d12:.4f} 元, ΔC_18={d18:.4f} 元"
          f"（>0 表示引入该时刻预报有净收益）")
    for k in (1, 2, 3):
        print(f"  {TAU[k] // 6}:00 时刻采纳调整 {adopt[k]}/{n} 天 ({adopt[k]/n:.1%})")
    print(f"  全年调整电量 A^total(S3口径) = {A_all:.4f} kWh")
    return yearly, adopt


def _block_title(ws, r, text):
    ws.cell(row=r, column=1, value=text).font = openpyxl.styles.Font(bold=True)


def write_tables(records_by_date):
    """表1/表2/表3（S3 方案，4 代表日纵向分块，同 Q2 版式，决策10）。
    表1 购电量/费用 = S3 生效策略 Q^eff 与生效购电纯费用；
    表2 充放电 = S3 生效版本计划 C/D（未截断），0:00/24:00 储电量 = 6000 / 实际日末；
    表3 紧急购电 = S3 执行段。"""
    slots = [('10:00-10:10', 60), ('12:00-12:10', 72), ('14:00-14:10', 84),
             ('16:00-16:10', 96), ('18:00-18:10', 108), ('20:00-20:10', 120)]
    chunk_pairs = [((0, 4), (4, 8)), ((8, 12), (12, 16)), ((16, 20), (20, 24))]

    def qty(rec, arr, h1, h2):
        return float(np.sum(arr[h1 * 6:h2 * 6]) * DT)

    # ---- 表1 ----
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '表1'
    r = 1
    for dd in REP_DAYS:
        rec = records_by_date[dd]
        q = rec['Qeff_s3']
        _block_title(ws, r, f"{dd.year}.{dd.month}.{dd.day}（S3方案）")
        r += 1
        for c, v in enumerate(['时间段', '购电量', '时间段', '购电量', '时间段', '购电量'], 1):
            ws.cell(row=r, column=c, value=v).font = openpyxl.styles.Font(bold=True)
        r += 1
        for grp in (slots[:3], slots[3:]):
            for i, (lbl, idx) in enumerate(grp):
                ws.cell(row=r, column=2 * i + 1, value=lbl)
                ws.cell(row=r, column=2 * i + 2, value=round(float(q[idx]) * DT, 4))
            r += 1
        ws.cell(row=r, column=1, value='全 天购电量')
        ws.cell(row=r, column=2, value=round(rec['eff_energy'], 4))
        ws.cell(row=r, column=3, value='全 天购电费')
        ws.cell(row=r, column=4, value=round(rec['eff_cost'], 4))
        r += 2
    wb.save(TABLE1_FILE)
    print(f"[输出] 表1 -> {TABLE1_FILE}")

    # ---- 表2 ----
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '表2'
    r = 1
    for dd in REP_DAYS:
        rec = records_by_date[dd]
        _block_title(ws, r, f"{dd.year}.{dd.month}.{dd.day}（S3方案）")
        r += 1
        for c, v in enumerate(['时间段', '充电量', '放电量', '时间段', '充电量', '放电量'], 1):
            ws.cell(row=r, column=c, value=v).font = openpyxl.styles.Font(bold=True)
        r += 1
        for (a, b_) in chunk_pairs:
            ws.cell(row=r, column=1, value=f"{a[0]}:00-{a[1]}:00")
            ws.cell(row=r, column=2, value=round(qty(rec, rec['C_s3'], *a), 4))
            ws.cell(row=r, column=3, value=round(qty(rec, rec['D_s3'], *a), 4))
            ws.cell(row=r, column=4, value=f"{b_[0]}:00-{b_[1]}:00")
            ws.cell(row=r, column=5, value=round(qty(rec, rec['C_s3'], *b_), 4))
            ws.cell(row=r, column=6, value=round(qty(rec, rec['D_s3'], *b_), 4))
            r += 1
        ws.cell(row=r, column=1, value='0:00 储电量')
        ws.cell(row=r, column=2, value=round(E_INIT, 4))
        ws.cell(row=r, column=4, value='24:00 储电量')
        ws.cell(row=r, column=5, value=round(rec['E145_s3'], 4))
        r += 2
    wb.save(TABLE2_FILE)
    print(f"[输出] 表2 -> {TABLE2_FILE}")

    # ---- 表3 ----
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '表3'
    for i, dd in enumerate(REP_DAYS):
        ws.cell(row=1, column=2 * i + 1, value=f"{dd.year}.{dd.month}.{dd.day}（S3方案）").font = openpyxl.styles.Font(bold=True)
        ws.cell(row=2, column=2 * i + 1, value='时间段').font = openpyxl.styles.Font(bold=True)
        ws.cell(row=2, column=2 * i + 2, value='购电量').font = openpyxl.styles.Font(bold=True)
    max_seg = max(len(records_by_date[dd]['scen'][3]['segs']) for dd in REP_DAYS)
    for i, dd in enumerate(REP_DAYS):
        for k, (s_min, e_min, energy) in enumerate(records_by_date[dd]['scen'][3]['segs']):
            ws.cell(row=3 + k, column=2 * i + 1, value=f"{fmt_min(s_min)}-{fmt_min(e_min)}")
            ws.cell(row=3 + k, column=2 * i + 2, value=round(energy, 4))
    if max_seg == 0:
        for i in range(4):
            ws.cell(row=3, column=2 * i + 2, value='无紧急购电')
    wb.save(TABLE3_FILE)
    print(f"[输出] 表3 -> {TABLE3_FILE}")


def print_day_detail(rec):
    d = rec['date']
    ck = rec['checks0']
    weekday_cn = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'][d.weekday()]
    print(f"\n—— {d}（{weekday_cn}）——")
    print(f"  k=0: J0={ck['J0']:.4f} 元, 词典序终解费用={ck['lex_cost']:.4f} 元"
          f"{' (≤C*+ε ✓)' if ck['lex_within'] else ' (超出C*+ε!)'}, LP松弛={ck['lp_cost']:.4f} 元 "
          f"(相对差 {ck['lp_rel_diff']:.2e}), LP互斥 {'通过' if ck['lp_excl_ok'] else '不通过'} "
          f"(max min(C,D)={ck['lp_excl_max']:.2e}), MILP互斥 max={ck['excl_milp']:.2e}")
    print(f"  等式残差={ck['eq_res']:.2e}, SOC∈[{ck['soc_min']:.2f},{ck['soc_max']:.2f}], "
          f"E0={ck['e0']:.4f}, E145={ck['e145']:.4f}, 二元可行={'是' if ck['u_binary_ok'] else '否'}")
    print(f"  计划购电 {rec['plan_energy']:.4f} kWh / {rec['plan_cost']:.4f} 元; "
          f"S3生效购电 {rec['eff_energy']:.4f} kWh / 纯费用 {rec['eff_cost']:.4f} 元; "
          f"S3实际日末储电 {rec['E145_s3']:.4f} kWh; 0:00光伏预测MAE={rec['mae_g0']:.4f} kW")
    for k in (1, 2, 3):
        ck2 = rec['checksK'][k]
        flag = f"采纳" if rec['a_flags'][k - 1] else "冻结"
        cl = " [E初值越界已裁剪]" if rec['Eclamped'].get(k) else ""
        print(f"  k={k}({TAU[k] // 6}:00): Jopt={ck2['Jopt']:.4f} vs Jfrozen={ck2['Jfrz']:.4f} → {flag}{cl}, "
              f"互斥max={ck2['excl']:.2e}, SOC∈[{ck2['soc_min']:.2f},{ck2['soc_max']:.2f}], "
              f"调整净费={rec['scen'][min(k,3)]['C_adj'] - (rec['scen'][min(k - 1, 3)]['C_adj'] if k > 1 else 0):.4f} 元")
    for j in range(4):
        s = rec['scen'][j]
        seg = ';'.join(f"{fmt_min(a)}-{fmt_min(b)}({e:.2f})" for a, b, e in s['segs'][:3])
        print(f"  S{j}: C_total={s['C_total']:.4f} 元 (计划 {s['C_plan']:.4f} + 调整净 {s['C_adj']:.4f} "
              f"+ 紧急 {s['C_emg']:.4f}), 紧急电量 {s['E_emg']:.4f} kWh, "
              f"调整电量 {s['A_tot']:.4f} kWh, 紧急段 {len(s['segs'])}"
              f"{'：' + seg if seg else ''}")
    if any(rec['Eclamped'].values()):
        print(f"  [告警] 滚动初值越界裁剪: {rec['Eclamped']}")


def _moving_block_ci(x, block, n_resamp, rng):
    """移动块自助法（Politis–White 风格）：对平均每日边际收益 Δ̄δ 求 95% CI（百分位法）。
    x 按 334 天日期序排列；每次重抽取 nb=ceil(D/b) 个长度 b 的连续块，
    块起点在 [0, D-b] 内随机（滑动、允许重叠，保留 ≤b 天的局部依赖结构），
    拼接后截前 D 天取均值；对 n_resamp 个重抽均值取 2.5%/97.5% 分位。
    循环实现（该步为后置统计分析，非热路径，可读性与可审计性优先）。"""
    D = len(x)
    b = min(block, D)          # 小样本冒烟测试时退化
    xb = np.asarray(x, dtype=float)
    nstart = D - b + 1         # 合法块起点数（保证整块落在 [0, D-1]）
    means = np.empty(n_resamp)
    for r in range(n_resamp):
        nb = int(np.ceil(D / b))
        starts = rng.integers(0, nstart, nb)
        idx = (starts[:, None] + np.arange(b)[None, :]).ravel()[:D]
        means[r] = xb[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(xb.mean()), float(lo), float(hi)


def _clamp_flags(rec):
    """k≥1 滚动初值越界裁剪标志（与 build_kk 的 E_init_c 裁剪规则一致）。
    主路径已直接记录 k=3；k=1/2 用缓存已存的 Emark/E_init_used 按同规则反推。"""
    out = {}
    eiu = rec.get('E_init_used', {})
    for k in (1, 2):
        out[k] = bool(abs(eiu.get(k, 0.0) - rec['Emark'][k]) > 1e-9)
    out[3] = bool(rec.get('Eclamped', {}).get(3, False))
    return out


def verify_analysis(records):
    """在现有缓存结果上追加的四组验证分析（纯读取 + 统计，不重解 MILP、
    不改变任何 MPC 目标/约束/执行模型，结果仅终端打印）。

    三、 ①每日边际成本 δ_{d,k}  ②边际收益来源分解
         ③紧急购电改善 ΔE/ΔC^emg/N^{emg-day}/N^{event}
         ④实际费用风险 均值/中位数/95%分位/最大日费/经验CVaR_0.95
    四、 ①调整采纳率 r^adopt   ②实际有效率 r^useful / 有害率 r^harmful
         ③新增调整电量 ΔA 与单位调整净收益 η^adj
    五、 公平性诊断 e^E_{d,j}=E_end(S_j)−6000、相邻策略末储差、N^{clamp}
    六、 稳定性：Δ̄δ_k、Median、ρ⁺/ρ⁻（ε=VERIFY_EPS 元）、
         7 天移动块自助法 95% CI（3/14 天块长稳健性对照，BOOT_N_RESAMPLE 次，seed=BOOT_SEED）"""
    n = len(records)
    D = n
    S = [rec['scen'] for rec in records]
    C_tot = {j: np.array([S[d][j]['C_total'] for d in range(D)]) for j in range(4)}
    C_emg = {j: np.array([S[d][j]['C_emg'] for d in range(D)]) for j in range(4)}
    C_adj = {j: np.array([S[d][j]['C_adj'] for d in range(D)]) for j in range(4)}
    E_emg = {j: np.array([S[d][j]['E_emg'] for d in range(D)]) for j in range(4)}
    A_tot = {j: np.array([S[d][j]['A_tot'] for d in range(D)]) for j in range(4)}
    ev = {j: np.array([len(S[d][j]['segs']) for d in range(D)]) for j in range(4)}
    E_end = {j: np.array([S[d][j]['E_end'] for d in range(D)]) for j in range(4)}
    flags = {k: np.array([1 if rec['a_flags'][k - 1] else 0 for rec in records], dtype=int) for k in (1, 2, 3)}
    clamp = {k: int(sum(1 for rec in records if _clamp_flags(rec)[k])) for k in (1, 2, 3)}

    t_label = {1: '6:00', 2: '12:00', 3: '18:00'}
    print("\n[验证分析] 缓存读取统计（样本 %d 天，3~6 节；仅统计、不重解 MILP）" % n)
    print("=" * 72)

    print("三① 每日边际成本 δ_{d,k}（δ>0 实际节省；δ=0 无调整或调整未改结果；δ<0 实际反贵）")
    for k in (1, 2, 3):
        delta = C_tot[k - 1] - C_tot[k]
        eps = VERIFY_EPS
        print(f"  {t_label[k]}: 节省 {int((delta > eps).sum())} 天 / 不变 {int((np.abs(delta) <= eps).sum())} 天 / "
              f"反贵 {int((delta < -eps).sum())} 天；日最大反贵 {float(delta.min()):.4f} 元，"
              f"日最大节省 {float(delta.max()):.4f} 元")

    print("三② 边际收益来源分解（净收益 = 避免的紧急购电费 − 新增调整净费；按预报时刻）")
    for k in (1, 2, 3):
        d_emg = float(C_emg[k - 1].sum() - C_emg[k].sum())
        d_adj = float(C_adj[k].sum() - C_adj[k - 1].sum())
        d_tot = float(C_tot[k - 1].sum() - C_tot[k].sum())
        print(f"  {t_label[k]}: 净收益 {d_tot:.4f} = 避免紧急费 {d_emg:.4f} − 新增调整费 {d_adj:.4f}"
              f"（分项差 {d_emg - d_adj:.4f}，勾稽差 {abs(d_tot - (d_emg - d_adj)):.4e}）")

    print("三③ 紧急购电改善（ΔE/ΔC^emg = 相邻两方案之差；N^{emg-day} 按各方案单独统计）")
    for j in range(4):
        print(f"  S{j}: 紧急购电天数 N={int((E_emg[j] > 1e-6).sum())}/{D}，紧急事件数={int(ev[j].sum())}"
              f"（连续十分钟段已合并为事件）")
    for k in (1, 2, 3):
        dE = float(E_emg[k - 1].sum() - E_emg[k].sum())
        dC = float(C_emg[k - 1].sum() - C_emg[k].sum())
        print(f"  引入 {t_label[k]}: ΔE^emg={dE:.4f} kWh, ΔC^emg={dC:.4f} 元 "
              f"(E 由 {E_emg[k - 1].sum():.4f} → {E_emg[k].sum():.4f} kWh)")

    m_cv = max(1, int(np.ceil(0.05 * n)))
    print(f"三④ 实际日费用风险 C_d(S_j)（经验CVaR_0.95 = 费用最高 {m_cv} 天（=ceil(5%×{n})）的均值；仅统计，不改 MPC 目标）")
    cv = {}
    for j in range(4):
        top = np.sort(C_tot[j])[-m_cv:]
        cv[j] = float(top.mean())
        print(f"  S{j}: 均值 {C_tot[j].mean():.4f}, 中位 {np.median(C_tot[j]):.4f}, "
              f"P95 {np.percentile(C_tot[j], 95):.4f}, 最大 {C_tot[j].max():.4f}, CVaR_0.95 {cv[j]:.4f} 元")
    for k in (1, 2, 3):
        print(f"  引入 {t_label[k]}: ΔCVaR_0.95 = {cv[k - 1] - cv[k]:+.4f} 元"
              f"（{cv[k - 1]:.4f} → {cv[k]:.4f}；若显著下降则该时刻属风险控制型预报）")

    print("四 调整行为验证（r^useful/r^harmful 只在被采纳日中统计 δ_{d,k} 实际盈亏）")
    for k in (1, 2, 3):
        n_ad = int(flags[k].sum())
        delta = C_tot[k - 1] - C_tot[k]
        ad = flags[k] == 1
        useful = int(((delta > VERIFY_EPS) & ad).sum())
        harmful = int(((delta < -VERIFY_EPS) & ad).sum())
        dA = float(A_tot[k].sum() - A_tot[k - 1].sum())
        if n_ad:
            eta = float(delta.sum()) / (dA + 1e-12)
            print(f"  {t_label[k]}: 采纳率 r^adopt={n_ad}/{D} ({n_ad / D:.1%})；"
                  f"实际有效率 r^useful={useful}/{n_ad} ({useful / n_ad:.1%})，"
                  f"有害率 r^harmful={harmful}/{n_ad} ({harmful / n_ad:.1%})；"
                  f"ΔA={dA:.4f} kWh，单位调整净收益 η^adj={eta:.4f} 元/kWh")
        else:
            print(f"  {t_label[k]}: 无采纳日（n_ad=0），r^useful/r^harmful 不适用；"
                  f"ΔA={dA:.4f} kWh")

    print("五 公平性诊断（实际日末储电对 6000 的偏差；费用优势若伴随 SOC 明显偏低，ΔC 不能全归因于新预报）")
    for j in range(4):
        e = E_end[j] - E_INIT
        print(f"  S{j}: e^E 均值 {float(e.mean()):+.4f}, 最小 {float(e.min()):+.4f}, "
              f"最大 {float(e.max()):+.4f} kWh（日末储电 {float(E_end[j].mean()):.4f} kWh）")
    for k in (1, 2, 3):
        de = E_end[k] - E_end[k - 1]
        print(f"  引入 {t_label[k]}: ΔE^end 均值 {float(de.mean()):+.4f}, "
              f"最小 {float(de.min()):+.4f}, 最大 {float(de.max()):+.4f} kWh")
    for k in (1, 2, 3):
        print(f"  N^clamp({t_label[k]})={clamp[k]}/{D}（滚动初值越出 [1200,10800] 被裁剪的天数；"
              f"频繁则预报时刻价值须注明是在当前执行与裁剪规则下得到）")

    print(f"六 稳定性检验（ε=VERIFY_EPS 元；95%CI 由 {BOOT_N_RESAMPLE} 次移动块自助法，"
          f"seed={BOOT_SEED}，主口径块长 7 天，3/14 天块为稳健性对照）")
    for k in (1, 2, 3):
        delta = C_tot[k - 1] - C_tot[k]
        block7 = _moving_block_ci(delta, 7, BOOT_N_RESAMPLE, np.random.default_rng(BOOT_SEED))
        block3 = _moving_block_ci(delta, 3, BOOT_N_RESAMPLE, np.random.default_rng(BOOT_SEED))
        block14 = _moving_block_ci(delta, 14, BOOT_N_RESAMPLE, np.random.default_rng(BOOT_SEED))
        print(f"  {t_label[k]}: Δ̄δ_k={float(delta.mean()):.4f} 元/天（合计 {float(delta.sum()):.4f} 元），"
              f"Median={float(np.median(delta)):.4f} 元")
        print(f"    ρ+={(delta > VERIFY_EPS).mean():.1%}, ρ-={(delta < -VERIFY_EPS).mean():.1%}，零占比={(np.abs(delta) <= VERIFY_EPS).mean():.1%}")
        if block7[1] > 0:
            cnote = 'CI 不含 0 且为正：边际收益统计显著为正'
        elif block7[2] < 0:
            cnote = 'CI 不含 0 且为负：该时刻边际收益统计显著为负（实际反贵）'
        else:
            cnote = 'CI 含 0：边际收益不显著'
        print(f"    95%CI(7天块) Δ̄δ_k∈[{block7[1]:.4f}, {block7[2]:.4f}]"
              f"；稳健性对照: 3天块[{block3[1]:.4f}, {block3[2]:.4f}], "
              f"14天块[{block14[1]:.4f}, {block14[2]:.4f}]"
              f"（{cnote}）")
    print("=" * 72)


def mae_common_analysis(ctx, records):
    """共同窗口 MAE 精度改善率与边际收益联合分析（纯统计，不重解 MILP、
    不改变任何 MPC 目标/约束/执行模型，结果仅终端打印）。

    核心思想：不同时刻的预报预测窗口长度不同（0:00 预报覆盖 0:00-24:00 共 24h，
    6:00 预报只覆盖 6:00-24:00 共 18h，12:00 只 12h，18:00 只 6h），
    不能直接比较各自全窗口 MAE。应在共同剩余窗口上比较：
      评价 6:00 更新：0:00 与 6:00 预报在 [6:00, 24:00] 的 MAE
      评价 12:00 更新：0:00、6:00、12:00 预报在 [12:00, 24:00] 的 MAE
      评价 18:00 更新：四次预报在 [18:00, 24:00] 的 MAE

    精度改善率（相邻时刻）：
      I_k^MAE = (MAE_{(k-1)→window} - MAE_{k→window}) / MAE_{(k-1)→window}

    与 ΔC_k = C(S_{k-1}) - C(S_k) 联合分析，回答：
      ① 预测精度提高是否转化为经济收益（逐天 corr(ΔMAE,δ) Pearson/Spearman）；
      ② 精度提高多少以后才值得支付调整费用（盈亏平衡临界改善 ΔMAE* 与临界改善率 I*）；
      ③ 哪个更新时刻的信息利用率最高（单位 MAE 改善年收益 + 收益/调整费覆盖倍数）。"""
    D = len(records)
    G = ctx['G']
    date_idx = ctx['date_idx']
    t_label = {1: '6:00', 2: '12:00', 3: '18:00'}

    # 各天共同窗口 MAE 矩阵
    # mae_all[k] shape (D, k+1)：第 k 次更新的共同窗口上，j=0..k 时刻预报的 MAE
    mae_all = {k: np.empty((D, k + 1)) for k in (1, 2, 3)}

    for idx, rec in enumerate(records):
        di = date_idx[rec['date']]
        Gd = G[di]
        # 预计算 4 次 pv_curve（纯插值，极轻量）
        preds = [pv_curve(ctx, di, j) for j in range(4)]
        # preds[0]: 144 值（全局 [0,144)）, preds[1]: 108 值（全局 [36,144)）
        # preds[2]: 72 值（全局 [72,144)）, preds[3]: 36 值（全局 [108,144)）

        for k in (1, 2, 3):
            i0 = TAU[k]  # 36, 72, 108
            window_actual = Gd[i0:]  # 共同窗口上的实际值
            for j in range(k + 1):
                offset = i0 - TAU[j]  # j 时刻预报在全局 [TAU[j],144) 中的局部起点
                pred_w = preds[j][offset:]
                mae_all[k][idx, j] = np.mean(np.abs(window_actual - pred_w))

    # 精度改善率（相邻时刻 k-1 → k）
    I_mae = {}
    for k in (1, 2, 3):
        base = mae_all[k][:, k - 1]
        curr = mae_all[k][:, k]
        I_mae[k] = np.where(base > 1e-12, (base - curr) / base, 0.0)

    # 边际收益
    S = [rec['scen'] for rec in records]
    C_tot = {j: np.array([S[d][j]['C_total'] for d in range(D)]) for j in range(4)}
    C_adj = {j: np.array([S[d][j]['C_adj'] for d in range(D)]) for j in range(4)}

    print("\n[共同窗口MAE] 精度改善率与边际收益联合分析（%d 天样本）" % D)
    print("=" * 72)

    for k in (1, 2, 3):
        i0 = TAU[k]
        n_win = N - i0
        print(f"\n  {t_label[k]} 更新 — 共同窗口 [{i0//6}:00, 24:00]（{n_win} 个 10 分钟时段）：")

        # 各时刻在窗口上的 MAE（年均值 + 中位数）
        for j in range(k + 1):
            j_label = f"{TAU[j]//6}:00" if j > 0 else "0:00"
            print(f"    MAE({j_label}→窗口) = {mae_all[k][:, j].mean():.4f} kW（均值）, "
                  f"{np.median(mae_all[k][:, j]):.4f} kW（中位）")

        # 精度改善率
        base_mean = mae_all[k][:, k - 1].mean()
        curr_mean = mae_all[k][:, k].mean()
        abs_improve = base_mean - curr_mean
        I_mean = I_mae[k].mean()
        I_med = np.median(I_mae[k])
        I_ratio = abs_improve / base_mean if base_mean > 1e-12 else 0.0
        print(f"    精度改善率 I_{k}^MAE = ({base_mean:.4f} - {curr_mean:.4f}) / {base_mean:.4f}"
              f" = {I_ratio:.4f}（逐天均值 {I_mean:.4f}，中位数 {I_med:.4f}）")

        # 边际收益
        dC = C_tot[k - 1].sum() - C_tot[k].sum()
        dC_adj = C_adj[k].sum() - C_adj[k - 1].sum()
        print(f"    边际收益 ΔC_{i0//6} = C(S{k-1}) - C(S{k}) = {dC:.4f} 元")
        print(f"    新增调整费 ΔC^adj = {dC_adj:.4f} 元")

        # 信息利用率指标
        if abs_improve > 1e-12:
            benefit_per_kW_day = dC / (abs_improve * D)
            print(f"    年收益 / 单位 MAE 改善 = {benefit_per_kW_day:.4f} 元/(kW·天)")
        else:
            print(f"    年收益 / 单位 MAE 改善：不适用（MAE 改善 ≈ 0）")
        if dC_adj > 1e-12:
            coverage = dC / dC_adj
            print(f"    收益/调整费覆盖倍数 = {coverage:.4f}"
                  f"{'（净收益为正）' if coverage > 1 else '（调整费超过节省）'}")
        else:
            print(f"    调整费增量 ≈ 0，收益/调整费覆盖不适用")

    # 汇总对照表
    print("\n  [汇总对照表]")
    print(f"  {'时刻':<6}{'MAE改善(kW)':>12}{'I^MAE(均值)':>12}{'ΔC(元)':>12}"
          f"{'ΔC^adj(元)':>12}{'收益/调整费':>10}")
    print("  " + "-" * 64)
    for k in (1, 2, 3):
        base_mean = mae_all[k][:, k - 1].mean()
        curr_mean = mae_all[k][:, k].mean()
        abs_improve = base_mean - curr_mean
        I_mean = I_mae[k].mean()
        dC = C_tot[k - 1].sum() - C_tot[k].sum()
        dC_adj = C_adj[k].sum() - C_adj[k - 1].sum()
        coverage = dC / dC_adj if dC_adj > 1e-12 else float('inf')
        print(f"  {t_label[k]:<6}{abs_improve:>12.4f}{I_mean:>12.4f}"
              f"{dC:>12.4f}{dC_adj:>12.4f}{coverage:>10.4f}")

    print("\n  [结论指标]")
    for k in (1, 2, 3):
        base_mean = mae_all[k][:, k - 1].mean()
        curr_mean = mae_all[k][:, k].mean()
        abs_improve = base_mean - curr_mean
        I_mean = I_mae[k].mean()
        dC = C_tot[k - 1].sum() - C_tot[k].sum()
        dC_adj = C_adj[k].sum() - C_adj[k - 1].sum()
        utilization = dC / dC_adj if dC_adj > 1e-12 else float('inf')
        worth = "值得" if dC > 0 else "不值得"
        print(f"  {t_label[k]}: MAE 改善 {abs_improve:.4f} kW（改善率 {I_mean:.4f}）→ "
              f"净收益 {dC:.4f} 元，{worth}支付调整费 {dC_adj:.4f} 元，"
              f"信息利用率 {utilization:.4f}")
    print("=" * 72)


# ========== 7. 主流程与并行（AGENTS：8 进程，按天分派，天间独立与串行一致） ==========
def build_ctx():
    prices = read_prices(PRICE_FILE)
    dates, L, G = read_actuals(ACT_FILE)
    F = read_forecasts(FCST_FILE)
    miss = [dd for dd in dates if dd not in F]
    extra = [dd for dd in F if dd not in {x for x in dates}]
    if miss or extra:
        raise ValueError(f"附件2/附件3日期不对齐: 缺 {len(miss)} 天 {miss[:3]}..., 多余 {len(extra)} 天 {extra[:3]}")
    FST = np.stack([F[dd] for dd in dates], axis=0)      # (365, 4, 24)
    date_idx = {dd: i for i, dd in enumerate(dates)}
    print(f"[数据] 附件3: {FST.shape[0]}天x4时刻x24值, 缺失0(已校验), "
          f"范围[{FST.min():.4f}, {FST.max():.4f}] kW; 与附件2 365 天日期对齐 ✓")
    ctx = {'prices': prices, 'dates': dates, 'L': L, 'G': G, 'FST': FST, 'date_idx': date_idx}
    return ctx


_CTX = {}


def _init_worker(ctx):
    global _CTX
    _CTX.clear()
    _CTX.update(ctx)


def _worker_solve(args):
    """并行子任务：求解一个决策日（4 场景共享 4 个子问题）并写缓存。"""
    di, time_limit = args
    ctx = _CTX
    rec = solve_day(ctx, di, time_limit)
    save_cache(rec)
    return di, rec


def decision_indices(ctx):
    i0 = ctx['date_idx'][FIRST_DAY]
    i1 = ctx['date_idx'][LAST_DAY]
    return list(range(i0, i1 + 1))


def run_repdays(ctx, time_limit):
    print("=" * 64)
    print("代表日模式：4 个指定日期（S3 + S0~S3 对照）逐日求解与全面核验")
    print("=" * 64)
    records = {}
    for dd in REP_DAYS:
        di = ctx['date_idx'][dd]
        rec = solve_day(ctx, di, time_limit)
        save_cache(rec)
        records[dd] = rec
        print_day_detail(rec)
    write_tables(records)
    print_aggregates(list(records.values()), '4个代表日')
    print("\n[提示] 代表日结果未写 result3.xlsx（需全量 334 天后生成）。"
          "确认无误后运行: python solve_q3.py（一键全量，默认断点续跑+8进程并行）")


def _run_parallel(idxs, ctx, time_limit, workers):
    tasks = [(i, time_limit) for i in idxs]
    n_workers = max(1, min(workers, len(tasks)))
    print(f"[并行] {n_workers} 进程 × {len(tasks)} 天（每子进程单线程确定性 MILP，"
          f"天间独立，结果与串行逐日一致；AGENTS 固定 8 核）", flush=True)
    out = {}
    t_start = time.perf_counter()
    with multiprocessing.get_context('spawn').Pool(
            processes=n_workers, initializer=_init_worker, initargs=(ctx,)) as pool:
        for k, (di, rec) in enumerate(pool.imap_unordered(_worker_solve, tasks, chunksize=1)):
            out[di] = rec
            done = len(out)
            el = time.perf_counter() - t_start
            sc = [rec['scen'][j]['C_total'] for j in range(4)]
            af = ''.join(str(int(x)) for x in rec['a_flags'])
            print(f"[{done}/{len(tasks)}] {rec['date']} 求解 | S0={sc[0]:.2f} S1={sc[1]:.2f} "
                  f"S2={sc[2]:.2f} S3={sc[3]:.2f} | 采纳k1..3={af} | "
                  f"耗时 {rec['plan_sec']:.1f}s | 墙钟 {el:.0f}s", flush=True)
    return [out[i] for i in sorted(out)]


def run_full(ctx, time_limit, resume=True, limit=None, workers=8):
    print("=" * 64)
    print(f"全量模式：{FIRST_DAY} ~ {LAST_DAY} 共 334 天 × S0~S3 四场景"
          f"（resume={resume}, limit={limit}, workers={workers}）")
    print("=" * 64)
    idxs = decision_indices(ctx)
    if limit is not None:
        idxs = idxs[:limit]
    records = []
    todo = []
    for di in idxs:
        dd = ctx['dates'][di]
        rec = load_cache(dd) if resume else None
        if rec is not None:
            records.append(rec)
        else:
            todo.append(di)
    print(f"[缓存] 命中 {len(records)} 天，待求解 {len(todo)} 天", flush=True)
    if todo:
        recs = _run_parallel(todo, ctx, time_limit, workers)
        records.extend(recs)
    records.sort(key=lambda r: r['date'])
    write_result3(records)
    write_comparison(records)
    by_date = {rec['date']: rec for rec in records}
    missing = [dd for dd in REP_DAYS if dd not in by_date]
    if missing:
        print(f"[警告] 代表日 {missing} 不在本批记录中，表1/2/3 跳过")
    else:
        write_tables(by_date)
        for dd in REP_DAYS:
            print_day_detail(by_date[dd])
    print_aggregates(records, f"全量{len(records)}天", write_comp=False)
    if len(records) > 4:
        verify_analysis(records)
        mae_common_analysis(ctx, records)


def run_outputs(ctx):
    print("=" * 64)
    print("输出重建模式：从缓存生成 result3.xlsx / 对照汇总 / 表1-3")
    print("=" * 64)
    records = []
    for di in decision_indices(ctx):
        dd = ctx['dates'][di]
        rec = load_cache(dd)
        if rec is None:
            raise RuntimeError(f"缺少 {dd} 的缓存，请先运行 full")
        records.append(rec)
    write_result3(records)
    write_comparison(records)
    by_date = {rec['date']: rec for rec in records}
    write_tables(by_date)
    for dd in REP_DAYS:
        print_day_detail(by_date[dd])
    print_aggregates(records, f"全量{len(records)}天", write_comp=False)
    verify_analysis(records)
    mae_common_analysis(ctx, records)


def print_aggregates(records, title, write_comp=True):
    n = len(records)
    # 代表日模式（n≤4）只打印、不落盘，避免小样本覆盖全量对照结果
    yearly, adopt = write_comparison(records, COMP_FILE, write=write_comp and n > 4)
    if n <= 4:
        return
    print(f"\n[全量汇总:{title}] 共 {n} 天（S3 方案交付口径）")
    print("=" * 72)
    s3 = yearly[3]
    print(f"  S3 全年总费用 : {s3['C_total']:.4f} 元（{s3['C_total']/10000:.2f} 万元）")
    print(f"    ├─ 计划购电费 C^plan : {s3['C_plan']:.4f} 元（{s3['C_plan']/10000:.2f} 万）")
    print(f"    ├─ 调整净费   C^adj  : {s3['C_adj']:.4f} 元（上调 {s3['C_up']:.4f} − 违约退费 "
          f"{s3['C_cancel']:.4f} + 违约 {s3['C_down']:.4f}）")
    print(f"    └─ 紧急购电费 C^emg  : {s3['C_emg']:.4f} 元（{s3['C_emg']/10000:.2f} 万）")
    print(f"  S3 紧急购电量 : {s3['E_emg']:.4f} kWh, 调整电量 : {s3['A_tot']:.4f} kWh, "
          f"紧急购电天数 {s3['emg_days']}/{n}")
    print(f"  对照结论: 引入 6:00 省 {yearly[0]['C_total'] - yearly[1]['C_total']:.4f} 元, "
          f"12:00 省 {yearly[1]['C_total'] - yearly[2]['C_total']:.4f} 元, "
          f"18:00 省 {yearly[2]['C_total'] - yearly[3]['C_total']:.4f} 元")


def main():
    ap = argparse.ArgumentParser(
        description='2025 CUMCM C题 问题三 MPC日内滚动求解'
                    '（无参数 = 一键全量334天×4场景：默认断点续跑 + 8进程并行）')
    ap.add_argument('cmd', nargs='?', default='full',
                   choices=['full', 'repdays', 'outputs'],
                   help='full=全量334天（默认） | repdays=4代表日核验 | outputs=由缓存重建输出')
    ap.add_argument('--force', action='store_true', help='忽略缓存全部重解')
    ap.add_argument('--limit', type=int, default=None, help='仅求解前 N 个决策日（冒烟测试）')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 8,
                    help='并行进程数（默认=本机CPU逻辑核数）')
    ap.add_argument('--time-limit', type=float, default=TIME_LIMIT_DEFAULT,
                    help='单次 MILP 时间限制（秒）')
    args = ap.parse_args()

    print(f"[参数] Δt={DT:.4f}h, κ={KAPPA}, η={ETA_CH}, E∈[{E_MIN},{E_MAX}], E0=E145={E_INIT}, "
          f"P_ch/P_dis<={P_CH_MAX:.0f}kW, P_buy_max=∞, mip_rel_gap={MIP_GAP}")
    print(f"[参数] 预报: 整点瞬时+线性插值, 锚点=发布时刻前一槽实际光伏, 负值截断0; "
          f"负载=附件2实际值; 结算=增量口径(1.5cU−0.5cW); 对照=S0~S3全跑")
    ctx = build_ctx()
    if args.cmd == 'repdays':
        run_repdays(ctx, TIME_LIMIT_DEFAULT)
    elif args.cmd == 'full':
        run_full(ctx, args.time_limit, resume=not args.force,
                 limit=args.limit, workers=args.workers)
    elif args.cmd == 'outputs':
        run_outputs(ctx)


if __name__ == '__main__':
    main()