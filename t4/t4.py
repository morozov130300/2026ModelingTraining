#!/usr/bin/env python3
"""问题四：多区域“算--储--电”联合优化。

实现 t4_plan.md 的两层分解框架：
1. 外层以整数小时开工、分钟级重叠和 SLA/容量约束进行滚动贪心调度；
2. 内层复用问题三的逐区域储能 LP，并把能源平衡约束的影子价格反馈给外层；
3. 输出 A0--A4 消融、Pareto 扫描和三类压力情景结果。

运行：python t4/t4.py --data-dir 题目 --output-dir t4/output
依赖：python>=3.9, numpy, pandas, openpyxl, scipy, matplotlib
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

CPU_COUNT = max(1, os.cpu_count() or 1)
LP_THREADS = max(1, math.ceil(CPU_COUNT / 6))
# 固定使用本机全部逻辑 CPU：最多 6 个区域 LP 同时运行，
# 每个 HiGHS/BLAS 子问题分配 CPU_COUNT/6 个线程。
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = str(LP_THREADS)

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

TASK_TYPES = ["RealTimeInference", "BatchInference", "AITraining"]
REGIONS = [f"Region{x}" for x in "ABCDEF"]
HORIZON = 2407
LAST_EXEC_HOUR = 2405
MAIN_HOURS = 2400
EPS = 1e-8
LP_VARIABLES = ("RenewableCharge_MW", "GridCharge_MW", "DischargePower_MW",
                "GridPurchase_MW", "GridSell_MW", "DirectRenewable_MW",
                "Curtailment_MW", "SOC_MWh")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="求解数学建模训练题问题四")
    p.add_argument("--data-dir", type=Path, default=here.parent / "题目")
    p.add_argument("--output-dir", type=Path, default=here / "output")
    p.add_argument("--carbon-price", type=float, default=200.0)
    p.add_argument("--window-hours", type=int, default=48, help="外层滚动候选窗口")
    p.add_argument("--batch-size", type=int, default=2000, help="滚动调度的任务批规模")
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--workers", type=int, default=min(8, max(1, os.cpu_count() or 1)), help="并行工作进程数；8核机器默认使用8个 worker")
    p.add_argument("--convergence-tol", type=float, default=1e-3)
    p.add_argument("--renewable-alpha", type=float, default=1.0)
    p.add_argument("--renewable-scale", type=float, default=1.0)
    p.add_argument("--capacity-factors", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    p.add_argument("--alpha-values", type=float, nargs="*", default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    p.add_argument("--lambda-values", type=float, nargs="*", default=[0, 50, 100, 200, 300, 500])
    p.add_argument("--skip-sensitivity", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


def require_columns(name: str, frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} 缺少字段: {missing}")


def load_data(data_dir: Path, renewable_scale: float = 1.0):
    workload = pd.read_excel(data_dir / "workload_trace.xlsx", sheet_name="Sheet1")
    gpu = pd.read_excel(data_dir / "GPU_information.xlsx", sheet_name="GPU中心基础情况")
    latency = pd.read_excel(data_dir / "network_latency.xlsx", sheet_name="network_latency")
    power = pd.read_excel(data_dir / "power_mapping.xlsx", sheet_name="任务功率映射")
    time_data = pd.read_excel(data_dir / "region_time_data.xlsx", sheet_name="region_time_data")
    storage = pd.read_excel(data_dir / "storage_information.xlsx", sheet_name="storage_information")
    require_columns("workload", workload, ["TaskID", "TaskType", "ArrivalHour", "GPU_Demand", "EstimatedDuration_min", "SourceRegion", "MaxLatency_ms", "LatestFinishHour"])
    require_columns("gpu", gpu, ["Region", "Available_GPU", "Max_IT_Power_MW", "PUE", "Max_Facility_Power_MW"])
    require_columns("latency", latency, ["FromRegion", "ToRegion", "NetworkLatency_ms"])
    require_columns("power", power, ["TaskType", "GPU_Power_MW_per_EquivalentGPU"])
    require_columns("time", time_data, ["Hour", "Region", "NonAI_IT_Load_MW", "AvailableRenewable_MW", "ElectricityPrice_CNY_per_MWh", "SellPrice_CNY_per_MWh", "CarbonIntensity_tCO2_per_MWh"])
    require_columns("storage", storage, ["Region", "StorageCapacity_MWh", "MinSOC_MWh", "InitialSOC_MWh", "MaxChargePower_MW", "MaxDischargePower_MW", "ChargeEfficiency", "DischargeEfficiency", "MaxGridImport_MW", "MaxGridExport_MW"])
    workload = workload.copy()
    workload["SourceRegion"] = workload["SourceRegion"].astype(str)
    workload["Duration_h"] = workload["EstimatedDuration_min"].astype(float) / 60.0
    workload["GPU_Demand"] = workload["GPU_Demand"].astype(float)
    workload["GPU_h"] = workload["GPU_Demand"] * workload["Duration_h"]
    power_map = power.set_index("TaskType")["GPU_Power_MW_per_EquivalentGPU"].astype(float).to_dict()
    if set(workload["TaskType"]) - set(power_map):
        raise ValueError("power_mapping 缺少任务类型")
    workload["PowerRate_MW_per_GPU"] = workload["TaskType"].map(power_map)
    workload["IT_MWh"] = workload["GPU_h"] * workload["PowerRate_MW_per_GPU"]
    gpu = gpu.copy(); gpu["Region"] = gpu["Region"].astype(str)
    storage = storage.copy(); storage["Region"] = storage["Region"].astype(str)
    if set(gpu.Region) != set(storage.Region):
        raise ValueError("GPU_information 与 storage_information 区域不一致")
    if not np.allclose(gpu["Max_IT_Power_MW"] * gpu["PUE"], gpu["Max_Facility_Power_MW"], atol=1e-6, rtol=0):
        raise ValueError("设施功率容量不等于 IT 容量×PUE")
    pue = gpu.set_index("Region")["PUE"].astype(float)
    storage = storage.set_index("Region").loc[REGIONS].copy()
    storage["PUE"] = pue.loc[REGIONS]
    time_data = time_data.copy(); time_data["Region"] = time_data["Region"].astype(str)
    expected = {(h, r) for h in range(HORIZON) for r in REGIONS}
    actual = set(zip(time_data["Hour"].astype(int), time_data["Region"]))
    if expected != actual:
        raise ValueError("region_time_data 必须完整覆盖 0--2406 的六区域数据")
    time_data["AvailableRenewable_MW"] = time_data["AvailableRenewable_MW"].astype(float) * renewable_scale
    latency_map = {(str(x.FromRegion), str(x.ToRegion)): float(x.NetworkLatency_ms) for x in latency.itertuples()}
    return workload, gpu.set_index("Region").loc[REGIONS], storage, time_data, power_map, latency_map


def region_series(time_data: pd.DataFrame, region: str, field: str) -> np.ndarray:
    part = time_data.loc[time_data["Region"].eq(region), ["Hour", field]].set_index("Hour").reindex(range(HORIZON))
    if part[field].isna().any():
        raise ValueError(f"{region}/{field} 缺失小时")
    return part[field].to_numpy(float)


def overlap_arrays(start: int, duration: float):
    finish = start + duration
    hours = np.arange(start, min(HORIZON, int(math.ceil(finish - EPS))))
    overlaps = np.maximum(0.0, np.minimum(hours + 1.0, finish) - np.maximum(hours.astype(float), float(start)))
    mask = overlaps > EPS
    return hours[mask], overlaps[mask]


def candidate_regions(row, regions, latency_map):
    source = str(row.SourceRegion)
    return [r for r in regions if latency_map.get((source, r), float("inf")) <= float(row.MaxLatency_ms) + EPS]


def prepare_energy_arrays(time_data, storage, regions, alpha=1.0):
    arrays = {}
    for r in regions:
        arrays[r] = {
            "nonai": region_series(time_data, r, "NonAI_IT_Load_MW"),
            "renewable": region_series(time_data, r, "AvailableRenewable_MW"),
            "price": region_series(time_data, r, "ElectricityPrice_CNY_per_MWh"),
            "sell_price": region_series(time_data, r, "SellPrice_CNY_per_MWh"),
            "carbon": region_series(time_data, r, "CarbonIntensity_tCO2_per_MWh"),
            "alpha": alpha,
            "storage": storage.loc[r].to_dict(),
        }
    return arrays


def initial_signal(energy, carbon_price):
    """按问题二口径计算无储能基准下的逐时边际供电成本信号。

    新能源富余且超过外送上限时，增加任务只会减少弃电，边际成本为 0；
    富余不超过外送上限时，增加任务会减少外送，边际成本为售电机会成本；
    新能源不足时，边际成本为购电成本加碳价成本。
    """
    signal = {}
    for r, a in energy.items():
        total = a["nonai"] * float(a["storage"]["PUE"])
        renewable = a["renewable"] * float(a.get("alpha", 1.0))
        surplus = np.maximum(0.0, renewable - total)
        shortage = np.maximum(0.0, total - renewable)
        export_limit = float(a["storage"]["MaxGridExport_MW"])
        opportunity = np.where(
            (surplus > EPS) & (surplus <= export_limit + EPS),
            a["sell_price"],
            0.0,
        )
        signal[r] = np.where(
            shortage > EPS,
            a["price"] + carbon_price * a["carbon"],
            opportunity,
        )
    return signal


def schedule_tasks(workload, gpu, energy, latency_map, signals, window_hours, batch_size, mode="joint", progress_label=None):
    """滚动窗口贪心外层；实时任务严格本地、到达即执行。"""
    regions = list(gpu.index)
    ri = {r: i for i, r in enumerate(regions)}
    horizon = HORIZON
    used_gpu = np.zeros((len(regions), horizon))
    used_it = np.zeros((len(regions), horizon))
    records, failed = [], []
    tasks = workload.copy()
    tasks["LatestStart"] = np.floor(tasks["LatestFinishHour"] - tasks["Duration_h"] + EPS).astype(int)
    tasks["Urgency"] = tasks["LatestStart"] - tasks["ArrivalHour"]
    tasks["Priority"] = np.where(tasks["TaskType"].eq("RealTimeInference"), 0, np.where(tasks["TaskType"].eq("AITraining"), 1, 2))
    tasks["TaskGroup"] = np.where(tasks["TaskType"].eq("RealTimeInference"), 0, 1)
    # 先完成全部实时任务，再按弹性任务截止紧迫度处理；这样弹性任务不会
    # 先占用本地容量，导致后续实时任务无法按题目要求到达即执行。
    tasks = tasks.sort_values(
        ["TaskGroup", "ArrivalHour", "LatestStart", "Priority", "GPU_Demand"],
        ascending=[True, True, True, True, False],
    )
    cap_gpu = gpu["Available_GPU"].to_dict()
    max_it = gpu["Max_IT_Power_MW"].to_dict()
    pue = gpu["PUE"].to_dict()

    for number, row in enumerate(tasks.itertuples(index=False), 1):
        source = str(row.SourceRegion)
        candidates = candidate_regions(row, regions, latency_map)
        if mode == "local":
            candidates = [source] if source in candidates else []
        elif row.TaskType == "RealTimeInference":
            candidates = [source] if source in candidates else []
        duration = float(row.Duration_h); demand = float(row.GPU_Demand); rate = float(row.PowerRate_MW_per_GPU)
        latest = min(int(row.LatestStart), int(math.floor(2406 - duration + EPS)))
        if latest < int(row.ArrivalHour) or not candidates:
            failed.append({"TaskID": row.TaskID, "Reason": "no SLA/time candidate"}); continue
        if row.TaskType == "RealTimeInference":
            starts = [int(row.ArrivalHour)]
        elif mode == "arrival":
            # A1 优先到达即执行；若到达时刻容量不足，再允许弹性任务在截止窗口内后移，
            # 否则迁移基准会因单个拥塞小时直接整体失败。
            starts = [int(row.ArrivalHour)]
        else:
            end = min(latest, int(row.ArrivalHour) + max(1, window_hours) - 1)
            starts = list(range(int(row.ArrivalHour), end + 1))
            if latest > end: starts.append(latest)
        def feasible_starts(start_values):
            starts_array = np.asarray(start_values, dtype=np.int64)
            if starts_array.size == 0:
                return None
            span = max(1, int(math.ceil(duration - EPS)))
            offsets = np.arange(span, dtype=np.int64)
            starts_array = starts_array[
                starts_array + span - 1 <= LAST_EXEC_HOUR
            ]
            if starts_array.size == 0:
                return None
            hours_matrix = starts_array[:, None] + offsets[None, :]
            overlap_template = np.maximum(
                0.0,
                np.minimum(offsets.astype(float) + 1.0, duration) - offsets.astype(float),
            )
            gpu_add = demand * overlap_template
            it_add = demand * rate * overlap_template
            best_choice = None
            for r in candidates:
                rj = ri[r]
                current_gpu = used_gpu[rj, hours_matrix]
                current_it = used_it[rj, hours_matrix]
                feasible = np.all(
                    current_gpu + gpu_add[None, :] <= float(cap_gpu[r]) + EPS,
                    axis=1,
                )
                feasible &= np.all(
                    current_it + it_add[None, :]
                    <= (float(max_it[r]) - energy[r]["nonai"][hours_matrix]) + EPS,
                    axis=1,
                )
                if not np.any(feasible):
                    continue
                scores = np.sum(
                    signals[r][hours_matrix]
                    * demand * rate * pue[r]
                    * overlap_template[None, :],
                    axis=1,
                )
                wait = np.maximum(0, starts_array - int(row.ArrivalHour))
                wait_penalty = 0.0001 * wait * demand * duration
                if mode == "local":
                    scores[:] = 0.0 if r == source else 1e15
                if mode == "arrival":
                    scores += wait_penalty * 10000
                scores += (
                    0.02 * float(latency_map.get((source, r), 0.0)) * demand * duration
                    + wait_penalty
                )
                scores = np.where(feasible, scores, np.inf)
                index = int(np.argmin(scores))
                start = int(starts_array[index])
                hours, overlap = overlap_arrays(start, duration)
                key = (float(scores[index]), start, r != source, r)
                if best_choice is None or key < best_choice[0]:
                    best_choice = (key, r, start, hours, overlap)
            return best_choice

        best = feasible_starts(starts)
        if best is None and mode == "arrival" and row.TaskType != "RealTimeInference":
            # 不再受滚动窗口限制：A1 的回退只为消除到达小时拥塞，
            # 必须检查该弹性任务完整的到达--截止可行区间。
            fallback_starts = list(range(int(row.ArrivalHour), latest + 1))
            best = feasible_starts(fallback_starts)
        if best is None:
            failed.append({"TaskID": row.TaskID, "Reason": "capacity infeasible"}); continue
        _, r, start, hours, overlap = best; rj = ri[r]
        used_gpu[rj, hours] += demand * overlap
        used_it[rj, hours] += demand * rate * overlap
        records.append({"TaskID": row.TaskID, "TaskType": row.TaskType, "SourceRegion": source, "ExecRegion": r,
                        "ArrivalHour": int(row.ArrivalHour), "StartHour": int(start), "Duration_h": duration,
                        "FinishHour": float(start + duration), "GPU_Demand": demand, "PowerRate_MW_per_GPU": rate,
                        "GPU_h": demand * duration, "IT_MWh": demand * duration * rate,
                        "MaxLatency_ms": float(row.MaxLatency_ms), "ActualLatency_ms": latency_map[(source, r)],
                        "LatestFinishHour": float(row.LatestFinishHour)})
        if progress_label and number % batch_size == 0:
            print(
                f"{progress_label} 外层调度进度: {number}/{len(tasks)}",
                flush=True,
            )
    if failed:
        raise RuntimeError(f"外层调度有 {len(failed)} 个任务失败；首个任务 {failed[0]}")
    schedule = pd.DataFrame(records).sort_values("TaskID").reset_index(drop=True)
    return schedule, used_gpu, used_it


def _schedule_case_job(args):
    label, workload, gpu, energy, latency_map, signals, window_hours, batch_size, mode = args
    return label, schedule_tasks(workload, gpu, energy, latency_map, signals,
                                 window_hours, batch_size, mode=mode,
                                 progress_label=label)


def _running_from_unc_path():
    """Windows 网络共享路径无法保证 spawn 子进程可重新读取主脚本。"""
    return str(Path(__file__).resolve()).startswith(("\\\\", "//"))


def _relaunch_from_local_copy():
    """从 UNC 启动时复制主脚本到本地临时目录，启用 Windows 进程池。"""
    if not _running_from_unc_path() or os.environ.get("T4_LOCAL_COPY") == "1":
        return False
    source = Path(__file__).resolve()
    temp_dir = Path(tempfile.mkdtemp(prefix="t4_local_"))
    local_script = temp_dir / "t4.py"
    shutil.copy2(source, local_script)
    arguments = list(sys.argv[1:])
    if "--data-dir" not in arguments:
        arguments.extend(["--data-dir", str(source.parent.parent / "题目")])
    if "--output-dir" not in arguments:
        arguments.extend(["--output-dir", str(source.parent / "output")])
    env = os.environ.copy()
    env["T4_LOCAL_COPY"] = "1"
    subprocess.run([sys.executable, str(local_script), *arguments], check=True, env=env)
    return True


def run_independent_schedules(workload, gpu, energy, latency_map, args):
    base_signal = _neutral_signal(energy)
    joint_signal = initial_signal(energy, args.carbon_price)
    jobs = [
        ("A0", workload, gpu, energy, latency_map, base_signal,
         args.window_hours, args.batch_size, "local"),
        # A1 与 A2 使用同一问题二口径的逐时边际能源信号；
        # 两者只在是否允许弹性任务时间平移上区分。
        ("A1", workload, gpu, energy, latency_map, joint_signal,
         args.window_hours, args.batch_size, "arrival"),
        ("A2", workload, gpu, energy, latency_map, joint_signal,
         args.window_hours, args.batch_size, "joint"),
    ]
    max_workers = min(8, max(1, int(args.workers)), len(jobs))
    # UNC 启动时已由本地副本重新执行，因此这里可以安全使用进程池。
    executor_type = ProcessPoolExecutor
    if max_workers == 1:
        solved = [_schedule_case_job(job) for job in jobs]
    else:
        with executor_type(max_workers=max_workers) as executor:
            futures = [executor.submit(_schedule_case_job, job) for job in jobs]
            solved = [future.result() for future in as_completed(futures)]
    return dict(solved)


def variable_slice(name):
    i = LP_VARIABLES.index(name)
    return slice(i * HORIZON, (i + 1) * HORIZON)


def build_loads(schedule, energy, gpu, regions):
    loads = {r: energy[r]["nonai"].copy() for r in regions}
    for row in schedule.itertuples(index=False):
        hours, overlap = overlap_arrays(int(row.StartHour), float(row.Duration_h))
        loads[row.ExecRegion][hours] += float(row.GPU_Demand) * float(row.PowerRate_MW_per_GPU) * overlap
    return loads


def solve_region_lp(region, loads, ctx, carbon_price):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import lil_matrix
    except ImportError as e:
        raise RuntimeError("问题四需要 scipy.optimize.linprog") from e
    n = HORIZON; size = len(LP_VARIABLES) * n
    p = ctx["storage"]; pue = float(p["PUE"])
    total = loads * pue
    c = np.zeros(size)
    c[variable_slice("GridPurchase_MW")] = ctx["price"] + carbon_price * ctx["carbon"]
    c[variable_slice("GridSell_MW")] = -ctx["sell_price"]
    bounds = []
    for name in LP_VARIABLES:
        if name in ("RenewableCharge_MW", "GridCharge_MW", "DischargePower_MW"):
            bounds.extend((0.0, 0.0 if t == HORIZON - 1 else (float(p["MaxChargePower_MW"]) if name != "DischargePower_MW" else float(p["MaxDischargePower_MW"]))) for t in range(n))
        elif name == "GridPurchase_MW": bounds.extend((0.0, float(p["MaxGridImport_MW"])) for _ in range(n))
        elif name == "GridSell_MW": bounds.extend((0.0, float(p["MaxGridExport_MW"])) for _ in range(n))
        elif name == "DirectRenewable_MW": bounds.extend((0.0, float(x)) for x in total)
        elif name == "Curtailment_MW": bounds.extend((0.0, None) for _ in range(n))
        else: bounds.extend((float(p["MinSOC_MWh"]), float(p["StorageCapacity_MWh"])) for _ in range(n))
    aeq = lil_matrix((3*n, size)); beq = np.zeros(3*n)
    cr,cg,d,q,y,u,w,s = [variable_slice(x) for x in LP_VARIABLES]
    for t in range(n):
        # q + R + d = TL + cR + cG + y + w；直接消纳 u 由下一条新能源守恒隐含。
        aeq[t,q.start+t]=1; aeq[t,d.start+t]=1; aeq[t,cr.start+t]=-1; aeq[t,cg.start+t]=-1; aeq[t,y.start+t]=-1; aeq[t,w.start+t]=-1; beq[t]=total[t]-ctx["renewable"][t]
        aeq[n+t,u.start+t]=1; aeq[n+t,cr.start+t]=1; aeq[n+t,y.start+t]=1; aeq[n+t,w.start+t]=1; beq[n+t]=ctx["renewable"][t]
        aeq[2*n+t,s.start+t]=1; aeq[2*n+t,cr.start+t]=-float(p["ChargeEfficiency"]); aeq[2*n+t,cg.start+t]=-float(p["ChargeEfficiency"]); aeq[2*n+t,d.start+t]=1/float(p["DischargeEfficiency"])
        if t:
            aeq[2*n+t,s.start+t-1]=-1
        else:
            beq[2*n+t]=float(p["InitialSOC_MWh"])
    aub = lil_matrix((6*n+1, size)); bub = np.zeros(6*n+1); k=0
    for t in range(n):
        for coeff, rhs in [
            ([(cr.start+t,1),(cg.start+t,1)], float(p["MaxChargePower_MW"])),
            ([(d.start+t,1)], float(p["MaxDischargePower_MW"])),
            ([(cr.start+t,1),(cg.start+t,1),(d.start+t,1)], max(float(p["MaxChargePower_MW"]), float(p["MaxDischargePower_MW"]))),
            ([(cg.start+t,1),(q.start+t,-1)], 0.0),
            ([(u.start+t,1),(cr.start+t,1),(y.start+t,1)], ctx["alpha"] * ctx["renewable"][t]),
            ([(q.start+t,1),(y.start+t,-1)], float(p["MaxGridImport_MW"]))]:
            for col,val in coeff: aub[k,col]=val
            bub[k]=rhs; k+=1
    aub[k,s.stop-1]=-1; bub[k]=-float(p["InitialSOC_MWh"])
    result = linprog(c, A_ub=aub.tocsr(), b_ub=bub, A_eq=aeq.tocsr(), b_eq=beq, bounds=bounds, method="highs")
    if not result.success: return {"success":False,"region":region,"message":result.message}
    values = {name: result.x[variable_slice(name)] for name in LP_VARIABLES}
    values.update({"success":True,"region":region,"objective":float(result.fun),"total_load":total,
                   "shadow_price":np.maximum(0.0, np.asarray(result.eqlin.marginals[:n],float))})
    return values


def _solve_region_job(args):
    region, loads, ctx, carbon_price = args
    return region, solve_region_lp(region, loads, ctx, carbon_price)


def _build_energy_frame(region, loads, ctx, result):
    n = HORIZON
    f = pd.DataFrame({"Hour": np.arange(n), "Region": region, "IT_Load_MW": loads,
                      "Total_Load_MW": result["total_load"], "AvailableRenewable_MW": ctx["renewable"],
                      "ElectricityPrice_CNY_per_MWh": ctx["price"], "SellPrice_CNY_per_MWh": ctx["sell_price"],
                      "CarbonIntensity_tCO2_per_MWh": ctx["carbon"]})
    for name in LP_VARIABLES:
        f[name] = result[name]
    f["ChargePower_MW"] = f["RenewableCharge_MW"] + f["GridCharge_MW"]
    f["NetGridImport_MW"] = f["GridPurchase_MW"] - f["GridSell_MW"]
    f["Cost_CNY"] = f["GridPurchase_MW"] * f["ElectricityPrice_CNY_per_MWh"] - f["GridSell_MW"] * f["SellPrice_CNY_per_MWh"]
    f["CarbonEmission_tCO2"] = f["GridPurchase_MW"] * f["CarbonIntensity_tCO2_per_MWh"]
    return f


def solve_energy(schedule, energy, gpu, storage, carbon_price, alpha, workers=1):
    loads = build_loads(schedule, energy, gpu, list(gpu.index))
    jobs = []
    for r in gpu.index:
        ctx = dict(energy[r]); ctx["alpha"] = alpha
        jobs.append((r, loads[r], ctx, carbon_price))
    results = {}; frames = []
    max_workers = max(1, min(int(workers), len(jobs)))
    if max_workers == 1:
        solved = [_solve_region_job(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_solve_region_job, job) for job in jobs]
            solved = [future.result() for future in as_completed(futures)]
    for r, result in solved:
        if not result["success"]:
            raise RuntimeError(f"内层 LP {r} 不可行：{result['message']}")
        results[r] = result
        frames.append(_build_energy_frame(r, loads[r], energy[r], result))
    frames.sort(key=lambda frame: list(gpu.index).index(frame["Region"].iloc[0]))
    return results, pd.concat(frames, ignore_index=True)


def _capacity_infeasible_error(error):
    return isinstance(error, RuntimeError) and "capacity infeasible" in str(error)


def _neutral_signal(energy):
    return {region: np.zeros(HORIZON) for region in energy}


def _schedule_with_fallback(
    workload, gpu, energy, latency_map, signal, args, progress_label,
    previous_schedule=None,
):
    try:
        return schedule_tasks(
            workload, gpu, energy, latency_map, signal,
            args.window_hours, args.batch_size, mode="joint",
            progress_label=progress_label,
        )
    except RuntimeError as error:
        if not _capacity_infeasible_error(error):
            raise
        if previous_schedule is not None:
            if progress_label:
                print(
                    f"{progress_label} | 新影子价格导致构造式调度不可行，"
                    "回退到上一轮完整可行任务方案",
                    flush=True,
                )
            return previous_schedule.copy(), None, None
        fallback_attempts = [
            ("初始边际成本信号", initial_signal(energy, args.carbon_price), "joint"),
            ("中性成本信号", _neutral_signal(energy), "joint"),
            ("到达优先可行性调度", _neutral_signal(energy), "arrival"),
        ]
        for fallback_index, (fallback_name, fallback_signal, fallback_mode) in enumerate(
            fallback_attempts, 1
        ):
            if progress_label:
                print(
                    f"{progress_label} | 容量候选不可行，执行第{fallback_index}次回退："
                    f"{fallback_name}",
                    flush=True,
                )
            try:
                return schedule_tasks(
                    workload, gpu, energy, latency_map, fallback_signal,
                    args.window_hours, args.batch_size, mode=fallback_mode,
                    progress_label=progress_label,
                )
            except RuntimeError as fallback_error:
                if not _capacity_infeasible_error(fallback_error):
                    raise
        raise error


def feedback_schedule(workload, gpu, energy, latency_map, args, initial=None, progress_label="A4"):
    signal = initial or initial_signal(energy, args.carbon_price)
    previous_obj = None
    history = []
    schedule = None
    energy_frame = None
    energy_results = None
    best_schedule = None
    best_energy_frame = None
    best_energy_results = None
    best_objective = float("inf")
    best_iteration = None
    total_iterations = int(args.max_iterations)
    for iteration in range(total_iterations):
        iteration_label = (
            f"{progress_label} | 反馈迭代 {iteration + 1}/{total_iterations}"
            if progress_label else None
        )
        if iteration_label:
            print(f"{iteration_label} | 开始任务调度", flush=True)
        schedule, used_gpu, used_it = _schedule_with_fallback(
            workload, gpu, energy, latency_map, signal, args,
            iteration_label, previous_schedule=schedule,
        )
        if iteration_label:
            print(f"{iteration_label} | 任务调度完成，开始区域能源 LP", flush=True)
        energy_results, energy_frame = solve_energy(schedule, energy, gpu, None, args.carbon_price, args.renewable_alpha, args.workers)
        objective=float(sum(x["objective"] for x in energy_results.values()))
        if objective < best_objective - EPS:
            best_schedule = schedule.copy()
            best_energy_frame = energy_frame.copy()
            best_energy_results = energy_results
            best_objective = objective
            best_iteration = iteration + 1
        history.append({
            "Iteration": iteration + 1,
            "Objective_CNY": objective,
            "ScheduledTasks": len(schedule),
            "SignalMean": float(np.mean([v.mean() for v in signal.values()])),
            "BestObjective_CNY": best_objective,
            "BestIteration": best_iteration,
        })
        relative_change = None if previous_obj is None else abs(objective-previous_obj)/max(abs(objective),1.0)
        if iteration_label:
            if relative_change is None:
                print(f"{iteration_label} | LP 完成，目标值={objective:.6f}，继续反馈", flush=True)
            else:
                print(f"{iteration_label} | LP 完成，目标值={objective:.6f}，相对变化={relative_change:.6g}", flush=True)
        if relative_change is not None and relative_change < args.convergence_tol:
            if iteration_label:
                print(f"{iteration_label} | 已收敛，停止反馈迭代", flush=True)
            break
        previous_obj=objective
        signal={r:energy_results[r]["shadow_price"] for r in gpu.index}
        if iteration_label and iteration + 1 < total_iterations:
            print(f"{iteration_label} | 未收敛，更新影子价格进入下一轮", flush=True)
    if progress_label and best_iteration != len(history):
        print(
            f"{progress_label} | 输出历史最优第{best_iteration}轮方案，"
            f"目标值={best_objective:.6f}",
            flush=True,
        )
    return best_schedule, best_energy_frame, best_energy_results, pd.DataFrame(history)


def _scenario_label(kind, parameter):
    if kind == "renewable_minus_20_percent":
        return "新能源-20%"
    return f"{kind}={parameter:g}"


def _run_sensitivity_job(spec):
    kind, parameter, workload, gpu, storage, time_data, energy, latency_map, args = spec
    local = argparse.Namespace(**vars(args))
    local.workers = 1
    if kind == "lambda":
        local.carbon_price = float(parameter)
        scenario_energy = energy
        metric_price = float(parameter)
    elif kind == "alpha":
        local.renewable_alpha = float(parameter)
        scenario_energy = prepare_energy_arrays(time_data, storage, REGIONS, local.renewable_alpha)
        metric_price = float(args.carbon_price)
    elif kind == "capacity":
        altered = storage.copy()
        altered["StorageCapacity_MWh"] *= float(parameter)
        altered["MinSOC_MWh"] *= float(parameter)
        altered["InitialSOC_MWh"] *= float(parameter)
        scenario_energy = prepare_energy_arrays(time_data, altered, REGIONS, args.renewable_alpha)
        metric_price = float(args.carbon_price)
    else:
        scenario_energy = prepare_energy_arrays(time_data, storage, REGIONS, args.renewable_alpha)
        for value in scenario_energy.values():
            value["renewable"] = value["renewable"] * 0.8
        metric_price = float(args.carbon_price)
    label = _scenario_label(kind, parameter)
    schedule, frame, _, history = feedback_schedule(
        workload, gpu, scenario_energy, latency_map, local,
        progress_label=None,
    )
    result = {"Scenario": kind, "Parameter": parameter, **metrics(kind, schedule, frame, metric_price)}
    return result


def run_sensitivity(workload, gpu, storage, time_data, energy, latency_map, args):
    specs = []
    for value in sorted(set(args.lambda_values)):
        specs.append(("lambda", value, workload, gpu, storage, time_data, energy, latency_map, args))
    for value in sorted(set(args.alpha_values)):
        specs.append(("alpha", value, workload, gpu, storage, time_data, energy, latency_map, args))
    for value in sorted(set(args.capacity_factors)):
        specs.append(("capacity", value, workload, gpu, storage, time_data, energy, latency_map, args))
    specs.append(("renewable_minus_20_percent", 0.8, workload, gpu, storage, time_data, energy, latency_map, args))
    total = len(specs)
    max_workers = min(8, max(1, int(args.workers)), total)
    print(f"敏感性分析: 0/{total} 个场景完成", flush=True)
    if max_workers == 1:
        rows = []
        for index, spec in enumerate(specs, 1):
            result = _run_sensitivity_job(spec)
            rows.append(result)
            print(
                f"敏感性分析: {index}/{total} 完成 "
                f"({_scenario_label(spec[0], spec[1])})",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_spec = {
                executor.submit(_run_sensitivity_job, spec): spec
                for spec in specs
            }
            rows = []
            for index, future in enumerate(as_completed(future_to_spec), 1):
                spec = future_to_spec[future]
                rows.append(future.result())
                print(
                    f"敏感性分析: {index}/{total} 完成 "
                    f"({_scenario_label(spec[0], spec[1])})",
                    flush=True,
                )
    return pd.DataFrame(rows)


def no_storage_frame(schedule, energy, gpu, regions, alpha=1.0):
    loads=build_loads(schedule,energy,gpu,regions); frames=[]
    for r in regions:
        a=energy[r]; total=loads[r]*float(gpu.loc[r,"PUE"]); usable=a["renewable"]*alpha; direct=np.minimum(total,usable); purchase=np.maximum(0,total-direct); surplus=usable-direct; sell=np.minimum(float(a["storage"]["MaxGridExport_MW"]),surplus); curtail=a["renewable"]-direct-sell
        frames.append(pd.DataFrame({"Hour":np.arange(HORIZON),"Region":r,"Total_Load_MW":total,"AvailableRenewable_MW":a["renewable"],"DirectRenewable_MW":direct,"RenewableCharge_MW":0.0,"GridCharge_MW":0.0,"DischargePower_MW":0.0,"GridPurchase_MW":purchase,"GridSell_MW":sell,"Curtailment_MW":curtail,"SOC_MWh":float(a["storage"]["InitialSOC_MWh"]),"NetGridImport_MW":purchase-sell,"Cost_CNY":purchase*a["price"]-sell*a["sell_price"],"CarbonEmission_tCO2":purchase*a["carbon"]}))
    return pd.concat(frames,ignore_index=True)


def metrics(name, schedule, frame, carbon_price):
    gpu_h=float(schedule["GPU_h"].sum()); latency=float((schedule["ActualLatency_ms"]*schedule["GPU_h"]).sum()/max(gpu_h,EPS));
    return {"Scenario":name,"TaskCount":int(len(schedule)),"MigratedTasks":int((schedule.SourceRegion!=schedule.ExecRegion).sum()),"WeightedAverageLatency_ms":latency,"MaxLatency_ms":float(schedule.ActualLatency_ms.max()),"OperatingCost_CNY":float(frame.Cost_CNY.sum()),"CarbonEmission_tCO2":float(frame.CarbonEmission_tCO2.sum()),"Objective_CNY":float(frame.Cost_CNY.sum()+carbon_price*frame.CarbonEmission_tCO2.sum()),"GridPurchase_MWh":float(frame.GridPurchase_MW.sum()),"GridSell_MWh":float(frame.GridSell_MW.sum()),"Curtailment_MWh":float(frame.Curtailment_MW.sum()),"PeakNetGridImport_MW":float(frame.groupby("Region").NetGridImport_MW.max().max()),"RenewableUtilization":float((frame.DirectRenewable_MW+frame.RenewableCharge_MW+frame.GridSell_MW).sum()/max(frame.AvailableRenewable_MW.sum(),EPS))}


def verify(schedule, frame, workload, gpu, storage, latency_map):
    m=workload.merge(schedule,on="TaskID",suffixes=("_input","_schedule")); checks=[]
    def add(n,v,d): checks.append({"Constraint":n,"ViolationCount":int(v>EPS),"MaxViolation":float(max(0,v)),"Status":"PASS" if v<=EPS else "FAIL","Detail":d})
    add("C1_TaskAssignment", len(schedule)!=len(workload), "每任务恰执行一次")
    add("C2_RealTime", float(np.max(np.abs(m.loc[m.TaskType_input.eq("RealTimeInference"),"StartHour"]-m.loc[m.TaskType_input.eq("RealTimeInference"),"ArrivalHour_input"]))) if m.TaskType_input.eq("RealTimeInference").any() else 0, "实时任务到达即开工")
    add("C3_Latency", float(np.maximum(m.ActualLatency_ms-m.MaxLatency_ms_input,0).max()), "网络时延 SLA")
    add("C4_Deadline", float(np.maximum(m.FinishHour-m.LatestFinishHour_input,0).max()), "截止时间")
    add("C5_Horizon", float(np.maximum(m.FinishHour-2406,0).max()), "不占 2406 小时")
    for r in gpu.index:
        part=frame[frame.Region.eq(r)]
        add(f"C6_GPU_{r}", 0, "外层已逐时检查 GPU 容量")
        add(f"C7_IT_{r}", float(np.maximum(part.IT_Load_MW - gpu.loc[r,"Max_IT_Power_MW"], 0).max()), "IT 功率容量")
        add(f"C8_GridImport_{r}", float(np.maximum(part.GridPurchase_MW - storage.loc[r,"MaxGridImport_MW"], 0).max()), "毛购电上限")
        add(f"C9_SOC_{r}", float(max(storage.loc[r,"MinSOC_MWh"]-part.SOC_MWh.min(),part.SOC_MWh.max()-storage.loc[r,"StorageCapacity_MWh"],0)), "SOC 上下限")
        add(f"C10_TerminalSOC_{r}", float(max(storage.loc[r,"InitialSOC_MWh"]-part.loc[part.Hour.eq(2406),"SOC_MWh"].iloc[0],0)), "终端 SOC")
        add(f"C11_Hour2406Idle_{r}", float(np.abs(part.loc[part.Hour.eq(2406),["RenewableCharge_MW","GridCharge_MW","DischargePower_MW"]].to_numpy()).max()), "2406 不充放电")
    return pd.DataFrame(checks)


def make_plots(out, schedule, energy_frame, convergence, ablation, scenario_frame):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError: return
    d=out/"plots"; d.mkdir(parents=True,exist_ok=True)
    convergence.plot(x="Iteration",y="Objective_CNY",marker="o",title="Benders-like feedback convergence",grid=True).get_figure().savefig(d/"反馈收敛图.png",dpi=150); plt.close("all")
    ablation.set_index("Scenario")[["OperatingCost_CNY","CarbonEmission_tCO2","PeakNetGridImport_MW"]].plot(kind="bar",subplots=True,figsize=(12,10),title="A0-A4 ablation"); plt.tight_layout(); plt.savefig(d/"消融分析图.png",dpi=150); plt.close("all")
    for r,g in energy_frame.groupby("Region"):
        g=g[g.Hour.between(0,95)]; plt.plot(g.Hour,g.Total_Load_MW,label=f"{r} load"); plt.plot(g.Hour,g.SOC_MWh,label=f"{r} SOC")
    plt.legend(ncol=3,fontsize=7); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(d/"区域负荷储能状态图_96小时.png",dpi=150); plt.close("all")
    scenario_frame.pivot(index="Parameter",columns="Scenario",values="Objective_CNY").plot(marker="o",figsize=(9,5)); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(d/"敏感性场景对比图.png",dpi=150); plt.close("all")


def main():
    args=parse_args()
    if args.carbon_price < 0 or args.window_hours < 1 or args.batch_size < 1 or args.max_iterations < 1 or args.workers < 1:
        raise ValueError("参数必须为正，碳价不得为负")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    print("读取原题数据并构造任务/能源参数……",flush=True)
    workload,gpu,storage,time_data,power_map,latency_map=load_data(args.data_dir,args.renewable_scale)
    energy=prepare_energy_arrays(time_data,storage,REGIONS,args.renewable_alpha)
    out=args.output_dir; (out/"schedule").mkdir(exist_ok=True); (out/"energy").mkdir(exist_ok=True); (out/"reports").mkdir(exist_ok=True)
    # A0：纯本地、到达即执行；A1：空间迁移；A2：空间+时间；A3：固定 A2 后储能；A4：完整反馈。
    # A0/A1/A2 的容量状态互相独立，使用多进程同时执行，结果与原串行算法完全一致。
    print(
        f"并行配置：CPU={CPU_COUNT}，worker={args.workers}，"
        "敏感性场景最多8进程；每场景内部区域LP串行",
        flush=True,
    )
    print("阶段 1/5：并行计算 A0/A1/A2 调度方案……", flush=True)
    independent = run_independent_schedules(workload, gpu, energy, latency_map, args)
    print("阶段 1/5 完成：A0/A1/A2 调度完成", flush=True)
    a0, _ug, _ui = independent["A0"]
    a1, _ug, _ui = independent["A1"]
    a2, _ug, _ui = independent["A2"]
    print("阶段 2/5：计算 A3 固定调度的区域能源 LP……", flush=True)
    a3_results,a3_energy=solve_energy(a2,energy,gpu,storage,args.carbon_price,args.renewable_alpha,args.workers)
    print("阶段 2/5 完成：A3 能源 LP 完成", flush=True)
    print("阶段 3/5：运行 A4 反馈迭代……", flush=True)
    a4,a4_energy,a4_results,convergence=feedback_schedule(workload,gpu,energy,latency_map,args)
    print("阶段 3/5 完成：A4 反馈迭代完成", flush=True)
    frames={"A0":no_storage_frame(a0,energy,gpu,REGIONS,args.renewable_alpha),"A1":no_storage_frame(a1,energy,gpu,REGIONS,args.renewable_alpha),"A2":no_storage_frame(a2,energy,gpu,REGIONS,args.renewable_alpha),"A3":a3_energy,"A4":a4_energy}
    schedules={"A0":a0,"A1":a1,"A2":a2,"A3":a2,"A4":a4}
    ablation=pd.DataFrame([metrics(k,schedules[k],frames[k],args.carbon_price) for k in ["A0","A1","A2","A3","A4"]])
    print("阶段 4/5：执行约束验证并写入主结果……", flush=True)
    verification=verify(a4,a4_energy,workload,gpu,storage,latency_map)
    if (verification.ViolationCount>0).any(): raise AssertionError("问题四约束验证失败")
    a4.to_csv(out/"schedule"/"schedule_A4.csv",index=False,encoding="utf-8-sig"); a3_energy.to_csv(out/"energy"/"energy_A3.csv",index=False,encoding="utf-8-sig"); a4_energy.to_csv(out/"energy"/"energy_A4.csv",index=False,encoding="utf-8-sig")
    a0.to_csv(out/"schedule"/"schedule_A0.csv",index=False,encoding="utf-8-sig"); a1.to_csv(out/"schedule"/"schedule_A1.csv",index=False,encoding="utf-8-sig"); a2.to_csv(out/"schedule"/"schedule_A2.csv",index=False,encoding="utf-8-sig")
    verification.to_csv(out/"reports"/"constraint_verification.csv",index=False,encoding="utf-8-sig"); ablation.to_csv(out/"reports"/"ablation_A0_A4.csv",index=False,encoding="utf-8-sig"); convergence.to_csv(out/"reports"/"convergence.csv",index=False,encoding="utf-8-sig")
    sens=[]
    if not args.skip_sensitivity:
        print("阶段 5/5：开始敏感性分析……", flush=True)
        scenario_frame = run_sensitivity(workload, gpu, storage, time_data, energy, latency_map, args)
        scenario_frame.to_csv(out/"reports"/"scenario_sensitivity.csv",index=False,encoding="utf-8-sig")
        print("阶段 5/5 完成：敏感性分析结果已写入", flush=True)
    else:
        scenario_frame = pd.DataFrame()
    if not args.skip_plots: make_plots(out,a4,a4_energy,convergence,ablation,scenario_frame if not scenario_frame.empty else pd.DataFrame({"Scenario":[],"Parameter":[],"Objective_CNY":[]}))
    summary={"model":"two-layer Benders-like task scheduling plus regional storage LP","carbon_price_CNY_per_tCO2":args.carbon_price,"window_hours":args.window_hours,"batch_size":args.batch_size,"iterations":int(len(convergence)),"task_count":int(len(workload)),"scheduled_task_count":int(len(a4)),"all_constraints_passed":True,"pressure_scenario":"AvailableRenewable_MW × 0.8","scenarios":ablation.to_dict(orient="records")}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"完成。结果目录：{out.resolve()}",flush=True)


if __name__ == "__main__":
    if not _relaunch_from_local_copy():
        main()
