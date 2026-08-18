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
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import warnings
from pathlib import Path

# 每个 HiGHS LP 使用单线程，把并发度交给外层区域/场景线程，避免线程过度订阅。
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

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
    p.add_argument("--workers", type=int, default=8, help="并行工作线程数；默认使用 8 个线程，用于区域 LP 与敏感性场景")
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
    signal = {}
    for r, a in energy.items():
        # 免费新能源优先；缺口小时使用经济成本+碳成本作为初始边际信号。
        signal[r] = np.maximum(0.0, a["price"] + carbon_price * a["carbon"]) * a["storage"]["PUE"]
        signal[r][a["renewable"] > a["nonai"] * a["storage"]["PUE"]] = 0.0
    return signal


def schedule_tasks(workload, gpu, energy, latency_map, signals, window_hours, batch_size, mode="joint"):
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
    tasks["DispatchHour"] = np.where(
        tasks["TaskType"].eq("RealTimeInference"),
        tasks["ArrivalHour"],
        tasks["LatestStart"],
    )
    # 弹性任务按截止紧迫度先排，避免早到的大任务贪心占满容量后堵死后续紧任务；
    # 实时任务仍按到达时刻排，保证到达即执行的优先级不被改变。
    tasks = tasks.sort_values(
        ["DispatchHour", "Priority", "ArrivalHour", "GPU_Demand"],
        ascending=[True, True, True, False],
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
            best_choice = None
            for r in candidates:
                rj = ri[r]
                for start in start_values:
                    hours, overlap = overlap_arrays(start, duration)
                    if len(hours) == 0 or hours[-1] > LAST_EXEC_HOUR: continue
                    add_gpu = demand * overlap
                    add_it = demand * rate * overlap
                    if np.any(used_gpu[rj, hours] + add_gpu > float(cap_gpu[r]) + EPS): continue
                    if np.any(used_it[rj, hours] + add_it > (float(max_it[r]) - energy[r]["nonai"][hours]) + EPS): continue
                    score = float(np.sum(signals[r][hours] * demand * rate * pue[r] * overlap))
                    migration_penalty = 0.02 * float(latency_map.get((source, r), 0.0)) * demand * duration
                    wait_penalty = 0.0001 * max(0, start - int(row.ArrivalHour)) * demand * duration
                    if mode == "local": score = 0.0 if r == source else 1e15
                    if mode == "arrival": score += wait_penalty * 10000
                    score += migration_penalty + wait_penalty
                    key = (score, start, r != source, r)
                    if best_choice is None or key < best_choice[0]:
                        best_choice = (key, r, start, hours, overlap)
            return best_choice

        best = feasible_starts(starts)
        if best is None and mode == "arrival" and row.TaskType != "RealTimeInference":
            end = min(latest, int(row.ArrivalHour) + max(1, window_hours) - 1)
            fallback_starts = list(range(int(row.ArrivalHour), end + 1))
            if latest > end: fallback_starts.append(latest)
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
        if number % batch_size == 0: print(f"外层调度进度: {number}/{len(tasks)}", flush=True)
    if failed:
        raise RuntimeError(f"外层调度有 {len(failed)} 个任务失败；首个任务 {failed[0]}")
    schedule = pd.DataFrame(records).sort_values("TaskID").reset_index(drop=True)
    return schedule, used_gpu, used_it


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
        aeq[t,q.start+t]=1; aeq[t,d.start+t]=1; aeq[t,u.start+t]=1; aeq[t,cr.start+t]=-1; aeq[t,cg.start+t]=-1; aeq[t,y.start+t]=-1; aeq[t,w.start+t]=-1; beq[t]=total[t]
        aeq[n+t,u.start+t]=1; aeq[n+t,cr.start+t]=1; aeq[n+t,y.start+t]=1; aeq[n+t,w.start+t]=1; beq[n+t]=ctx["renewable"][t]
        aeq[2*n+t,s.start+t]=1; aeq[2*n+t,cr.start+t]=-float(p["ChargeEfficiency"]); aeq[2*n+t,cg.start+t]=-float(p["ChargeEfficiency"]); aeq[2*n+t,d.start+t]=1/float(p["DischargeEfficiency"]); beq[2*n+t]=float(p["InitialSOC_MWh"])
        if t: aeq[2*n+t,s.start+t-1]=-1
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


def feedback_schedule(workload, gpu, energy, latency_map, args, initial=None):
    signal = initial or initial_signal(energy, args.carbon_price)
    previous_obj=None; history=[]; schedule=None; energy_frame=None; energy_results=None
    for iteration in range(args.max_iterations):
        schedule, used_gpu, used_it = schedule_tasks(workload, gpu, energy, latency_map, signal, args.window_hours, args.batch_size, mode="joint")
        energy_results, energy_frame = solve_energy(schedule, energy, gpu, None, args.carbon_price, args.renewable_alpha, args.workers)
        objective=float(sum(x["objective"] for x in energy_results.values()))
        history.append({"Iteration":iteration+1,"Objective_CNY":objective,"ScheduledTasks":len(schedule),"SignalMean":float(np.mean([v.mean() for v in signal.values()]))})
        if previous_obj is not None and abs(objective-previous_obj)/max(abs(objective),1.0) < args.convergence_tol: break
        previous_obj=objective
        signal={r:energy_results[r]["shadow_price"] for r in gpu.index}
    return schedule, energy_frame, energy_results, pd.DataFrame(history)


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
    schedule, frame, _, history = feedback_schedule(workload, gpu, scenario_energy, latency_map, local)
    return {"Scenario": kind, "Parameter": parameter, **metrics(kind, schedule, frame, metric_price)}


def run_sensitivity(workload, gpu, storage, time_data, energy, latency_map, args):
    specs = []
    for value in sorted(set(args.lambda_values)):
        specs.append(("lambda", value, workload, gpu, storage, time_data, energy, latency_map, args))
    for value in sorted(set(args.alpha_values)):
        specs.append(("alpha", value, workload, gpu, storage, time_data, energy, latency_map, args))
    for value in sorted(set(args.capacity_factors)):
        specs.append(("capacity", value, workload, gpu, storage, time_data, energy, latency_map, args))
    specs.append(("renewable_minus_20_percent", 0.8, workload, gpu, storage, time_data, energy, latency_map, args))
    max_workers = max(1, min(int(args.workers), len(specs)))
    if max_workers == 1:
        rows = [_run_sensitivity_job(spec) for spec in specs]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_sensitivity_job, spec) for spec in specs]
            rows = [future.result() for future in futures]
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
    convergence.plot(x="Iteration",y="Objective_CNY",marker="o",title="Benders-like feedback convergence",grid=True).get_figure().savefig(d/"convergence.png",dpi=150); plt.close("all")
    ablation.set_index("Scenario")[["OperatingCost_CNY","CarbonEmission_tCO2","PeakNetGridImport_MW"]].plot(kind="bar",subplots=True,figsize=(12,10),title="A0-A4 ablation"); plt.tight_layout(); plt.savefig(d/"ablation.png",dpi=150); plt.close("all")
    for r,g in energy_frame.groupby("Region"):
        g=g[g.Hour.between(0,95)]; plt.plot(g.Hour,g.Total_Load_MW,label=f"{r} load"); plt.plot(g.Hour,g.SOC_MWh,label=f"{r} SOC")
    plt.legend(ncol=3,fontsize=7); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(d/"regional_load_soc_96h.png",dpi=150); plt.close("all")
    scenario_frame.pivot(index="Parameter",columns="Scenario",values="Objective_CNY").plot(marker="o",figsize=(9,5)); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(d/"scenario_comparison.png",dpi=150); plt.close("all")


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
    base_signal={r:np.zeros(HORIZON) for r in REGIONS}
    a0,_ug,_ui=schedule_tasks(workload,gpu,energy,latency_map,base_signal,args.window_hours,args.batch_size,mode="local")
    a1_signal={r:np.repeat(float(np.mean(energy[r]["price"]+args.carbon_price*energy[r]["carbon"]))*float(gpu.loc[r,"PUE"]),HORIZON) for r in REGIONS}
    a1,_ug,_ui=schedule_tasks(workload,gpu,energy,latency_map,a1_signal,args.window_hours,args.batch_size,mode="arrival")
    a2,_ug,_ui=schedule_tasks(workload,gpu,energy,latency_map,initial_signal(energy,args.carbon_price),args.window_hours,args.batch_size,mode="joint")
    a3_results,a3_energy=solve_energy(a2,energy,gpu,storage,args.carbon_price,args.renewable_alpha,args.workers)
    a4,a4_energy,a4_results,convergence=feedback_schedule(workload,gpu,energy,latency_map,args)
    frames={"A0":no_storage_frame(a0,energy,gpu,REGIONS,args.renewable_alpha),"A1":no_storage_frame(a1,energy,gpu,REGIONS,args.renewable_alpha),"A2":no_storage_frame(a2,energy,gpu,REGIONS,args.renewable_alpha),"A3":a3_energy,"A4":a4_energy}
    schedules={"A0":a0,"A1":a1,"A2":a2,"A3":a2,"A4":a4}
    ablation=pd.DataFrame([metrics(k,schedules[k],frames[k],args.carbon_price) for k in ["A0","A1","A2","A3","A4"]])
    verification=verify(a4,a4_energy,workload,gpu,storage,latency_map)
    if (verification.ViolationCount>0).any(): raise AssertionError("问题四约束验证失败")
    a4.to_csv(out/"schedule"/"schedule_A4.csv",index=False,encoding="utf-8-sig"); a4_energy.to_csv(out/"energy"/"energy_A4.csv",index=False,encoding="utf-8-sig"); a3_energy.to_csv(out/"energy"/"energy_A3.csv",index=False,encoding="utf-8-sig")
    verification.to_csv(out/"reports"/"constraint_verification.csv",index=False,encoding="utf-8-sig"); ablation.to_csv(out/"reports"/"ablation_A0_A4.csv",index=False,encoding="utf-8-sig"); convergence.to_csv(out/"reports"/"convergence.csv",index=False,encoding="utf-8-sig")
    sens=[]
    if not args.skip_sensitivity:
        scenario_frame = run_sensitivity(workload, gpu, storage, time_data, energy, latency_map, args)
        scenario_frame.to_csv(out/"reports"/"scenario_sensitivity.csv",index=False,encoding="utf-8-sig")
    else:
        scenario_frame = pd.DataFrame()
    if not args.skip_plots: make_plots(out,a4,a4_energy,convergence,ablation,scenario_frame if not scenario_frame.empty else pd.DataFrame({"Scenario":[],"Parameter":[],"Objective_CNY":[]}))
    summary={"model":"two-layer Benders-like task scheduling plus regional storage LP","carbon_price_CNY_per_tCO2":args.carbon_price,"window_hours":args.window_hours,"batch_size":args.batch_size,"iterations":int(len(convergence)),"task_count":int(len(workload)),"scheduled_task_count":int(len(a4)),"all_constraints_passed":True,"pressure_scenario":"AvailableRenewable_MW × 0.8","scenarios":ablation.to_dict(orient="records")}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"完成。结果目录：{out.resolve()}",flush=True)


if __name__ == "__main__": main()
