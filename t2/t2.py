#!/usr/bin/env python3
"""问题二：可再生优先的碳感知 AI 工作负载调度。

依据 t2_plan.md 实现：实时任务固定本地；弹性任务先作区域粗指派，
再在可达区域和整数开工时刻中以 C + lambda * E 的边际增量选优。
所有容量和能量计算均采用分钟级小时重叠量，2406 仅用于结算而不执行任务。

运行：
    python t2/t2.py --data-dir 题目 --output-dir t2/output --carbon-price 200
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

TASK_TYPES = ["RealTimeInference", "BatchInference", "AITraining"]
HORIZON = 2407
LAST_EXEC_HOUR = 2405
EPS = 1e-9
EXPORT_LIMITS = {"RegionA": 0.0, "RegionB": 0.0, "RegionC": 0.0,
                 "RegionD": 180.0, "RegionE": 220.0, "RegionF": 220.0}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="求解数学建模训练题问题二（碳感知调度）")
    parser.add_argument("--data-dir", type=Path, default=here.parent / "题目")
    parser.add_argument("--output-dir", type=Path, default=here / "output")
    parser.add_argument("--carbon-price", type=float, default=200.0,
                        help="碳价 lambda，单位元/tCO2，默认 200")
    parser.add_argument("--lambda-values", type=float, nargs="*", default=[0, 50, 100, 150, 200, 300, 500],
                        help="灵敏度分析碳价值列表；主方案的 carbon-price 会自动加入")
    parser.add_argument("--q1-schedule", type=Path, default=here.parent / "t1" / "output" / "schedule" / "schedule.csv",
                        help="问题一 schedule.csv；默认读取 t1/output/schedule/schedule.csv")
    parser.add_argument("--skip-sensitivity", action="store_true", help="仅运行主碳价，跳过额外 lambda 调度")
    parser.add_argument("--workers", type=int, default=8,
                        help="并行运行独立方案/灵敏度任务的进程数；1 为串行，建议不超过物理核心数")
    parser.add_argument("--skip-plots", action="store_true", help="跳过 PNG 制图")
    return parser.parse_args()


def require_columns(name: str, frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} 缺少字段: {missing}")


def load_data(data_dir: Path):
    workload = pd.read_excel(data_dir / "workload_trace.xlsx", sheet_name="Sheet1")
    gpu = pd.read_excel(data_dir / "GPU_information.xlsx", sheet_name="GPU中心基础情况")
    latency = pd.read_excel(data_dir / "network_latency.xlsx", sheet_name="network_latency")
    power = pd.read_excel(data_dir / "power_mapping.xlsx", sheet_name="任务功率映射")
    time_data = pd.read_excel(data_dir / "region_time_data.xlsx", sheet_name="region_time_data")
    require_columns("workload", workload, ["TaskID", "TaskType", "ArrivalHour", "GPU_Demand",
                                             "EstimatedDuration_min", "SourceRegion", "MaxLatency_ms", "LatestFinishHour"])
    require_columns("gpu", gpu, ["Region", "Available_GPU", "Max_IT_Power_MW", "PUE", "Max_Facility_Power_MW"])
    require_columns("latency", latency, ["FromRegion", "ToRegion", "NetworkLatency_ms"])
    require_columns("power", power, ["TaskType", "GPU_Power_MW_per_EquivalentGPU"])
    require_columns("region_time_data", time_data, ["Hour", "Region", "ElectricityPrice_CNY_per_MWh",
        "SellPrice_CNY_per_MWh", "CarbonIntensity_tCO2_per_MWh", "AvailableRenewable_MW", "NonAI_IT_Load_MW"])

    workload = workload.copy()
    workload["Duration_h"] = workload["EstimatedDuration_min"].astype(float) / 60.0
    workload["GPU_h"] = workload["GPU_Demand"].astype(float) * workload["Duration_h"]
    power_map = power.set_index("TaskType")["GPU_Power_MW_per_EquivalentGPU"].astype(float).to_dict()
    unknown = set(workload["TaskType"]) - set(power_map)
    if unknown:
        raise ValueError(f"power_mapping 缺少任务类型: {sorted(unknown)}")
    workload["IT_MWh"] = workload["GPU_h"] * workload["TaskType"].map(power_map)

    regions = gpu["Region"].astype(str).tolist()
    cap = gpu.set_index("Region").loc[regions]
    if not np.allclose(cap["Max_IT_Power_MW"].to_numpy(float) * cap["PUE"].to_numpy(float),
                       cap["Max_Facility_Power_MW"].to_numpy(float), atol=1e-6, rtol=0):
        raise ValueError("Max_Facility_Power_MW 必须等于 Max_IT_Power_MW × PUE")
    missing_limits = set(regions) - set(EXPORT_LIMITS)
    if missing_limits:
        raise ValueError(f"未配置区域外送上限: {sorted(missing_limits)}")

    latency_map = {(str(x.FromRegion), str(x.ToRegion)): float(x.NetworkLatency_ms)
                   for x in latency.itertuples(index=False)}
    energy = time_data.pivot_table(index="Hour", columns="Region", aggfunc="first")
    energy = energy.reindex(index=range(HORIZON))
    if not isinstance(energy.columns, pd.MultiIndex):
        raise ValueError("region_time_data 透视后未形成字段×区域多级列，请检查 Hour、Region 与能源字段")
    for field in ["NonAI_IT_Load_MW", "ElectricityPrice_CNY_per_MWh", "SellPrice_CNY_per_MWh",
                  "CarbonIntensity_tCO2_per_MWh", "AvailableRenewable_MW"]:
        if field not in energy.columns.get_level_values(0):
            raise ValueError(f"region_time_data 透视结果缺少字段: {field}")
        if energy[field].reindex(columns=regions).isna().any().any():
            raise ValueError(f"region_time_data 的 {field} 必须覆盖所有区域的 0–2406 小时")
    return workload, gpu, energy, power_map, latency_map, regions


def energy_array(energy: pd.DataFrame, field: str, regions: list[str]) -> np.ndarray:
    table = energy[field].reindex(index=range(HORIZON), columns=regions)
    if table.isna().any().any():
        raise ValueError(f"region_time_data 的 {field} 存在区域或小时缺失")
    return table.to_numpy(float).T


def overlap_arrays(start: int, duration: float) -> tuple[np.ndarray, np.ndarray]:
    finish = start + duration
    hours = np.arange(start, min(HORIZON, int(math.ceil(finish - EPS))))
    overlap = np.maximum(0.0, np.minimum(hours + 1.0, finish) - np.maximum(hours.astype(float), start))
    mask = overlap > EPS
    return hours[mask], overlap[mask]


def build_context(gpu: pd.DataFrame, energy: pd.DataFrame, regions: list[str]) -> dict:
    cap = gpu.set_index("Region").loc[regions]
    nonai = energy_array(energy, "NonAI_IT_Load_MW", regions)
    max_it = cap["Max_IT_Power_MW"].to_numpy(float)
    it_margin = max_it[:, None] - nonai
    if np.any(it_margin < -EPS):
        raise ValueError("存在非 AI IT 负荷超过区域 IT 功率上限的时段")
    return {
        "regions": regions,
        "index": {region: index for index, region in enumerate(regions)},
        "available_gpu": cap["Available_GPU"].to_numpy(float),
        "pue": cap["PUE"].to_numpy(float),
        "it_margin": it_margin,
        "nonai": nonai,
        "price": energy_array(energy, "ElectricityPrice_CNY_per_MWh", regions),
        "carbon": energy_array(energy, "CarbonIntensity_tCO2_per_MWh", regions),
        "renewable": energy_array(energy, "AvailableRenewable_MW", regions),
        "sell_price": energy_array(energy, "SellPrice_CNY_per_MWh", regions),
        "export_limit": np.asarray([EXPORT_LIMITS[r] for r in regions], dtype=float),
    }


def facility_objective(load: np.ndarray, renewable: np.ndarray, price: np.ndarray,
                       carbon: np.ndarray, sell_price: np.ndarray, export_limit: float,
                       carbon_price: float) -> np.ndarray:
    """单小时设施负荷的 C + lambda E，含可再生优先、外送与弃电解析最优。"""
    purchase = np.maximum(0.0, load - renewable)
    surplus = np.maximum(0.0, renewable - load)
    sell = np.minimum(export_limit, surplus)
    return purchase * (price + carbon_price * carbon) - sell * sell_price


def energy_balance(ctx: dict, ai_it: np.ndarray) -> tuple[pd.DataFrame, dict]:
    total_it = ctx["nonai"] + ai_it
    total_load = total_it * ctx["pue"][:, None]
    direct = np.minimum(ctx["renewable"], total_load)
    purchase = np.maximum(0.0, total_load - ctx["renewable"])
    surplus = np.maximum(0.0, ctx["renewable"] - total_load)
    sell = np.minimum(ctx["export_limit"][:, None], surplus)
    curtail = ctx["renewable"] - direct - sell
    cost = purchase * ctx["price"] - sell * ctx["sell_price"]
    emission = purchase * ctx["carbon"]
    rows = []
    for rj, region in enumerate(ctx["regions"]):
        for hour in range(HORIZON):
            rows.append({"Hour": hour, "Region": region, "AI_IT_Load_MW": ai_it[rj, hour],
                "NonAI_IT_Load_MW": ctx["nonai"][rj, hour], "IT_Load_MW": total_it[rj, hour],
                "Total_Load_MW": total_load[rj, hour], "AvailableRenewable_MW": ctx["renewable"][rj, hour],
                "DirectRenewable_MW": direct[rj, hour], "GridPurchase_MW": purchase[rj, hour],
                "GridSell_MW": sell[rj, hour], "Curtailment_MW": curtail[rj, hour],
                "ElectricityPrice_CNY_per_MWh": ctx["price"][rj, hour],
                "CarbonIntensity_tCO2_per_MWh": ctx["carbon"][rj, hour],
                "SellPrice_CNY_per_MWh": ctx["sell_price"][rj, hour], "Cost_CNY": cost[rj, hour],
                "CarbonEmission_tCO2": emission[rj, hour]})
    metrics = {"OperatingCost_CNY": float(cost.sum()), "CarbonEmission_tCO2": float(emission.sum()),
        "AvailableRenewable_MWh": float(ctx["renewable"].sum()), "DirectRenewable_MWh": float(direct.sum()),
        "GridSell_MWh": float(sell.sum()), "Curtailment_MWh": float(curtail.sum()),
        "GridPurchase_MWh": float(purchase.sum())}
    metrics["RenewableUtilization"] = ((metrics["DirectRenewable_MWh"] + metrics["GridSell_MWh"]) /
                                        metrics["AvailableRenewable_MWh"] if metrics["AvailableRenewable_MWh"] else 0.0)
    return pd.DataFrame(rows), metrics


def allowed_regions(row, ctx: dict, latency_map: dict, local_only: bool = False) -> list[str]:
    source = str(row.SourceRegion)
    if row.TaskType == "RealTimeInference" or local_only:
        return [source]
    return [r for r in ctx["regions"] if latency_map.get((source, r), float("inf")) <= float(row.MaxLatency_ms) + EPS]


def stage_one_assign(workload: pd.DataFrame, ctx: dict, power_map: dict, latency_map: dict,
                     carbon_price: float) -> dict:
    """按平均能源价格作粗粒度可达区域指派，并使用 GPU-hour 松弛容量避免区域过度集中。"""
    region_mean = ctx["price"].mean(axis=1) + carbon_price * ctx["carbon"].mean(axis=1)
    assignment, assigned_gpuh = {}, np.zeros(len(ctx["regions"]))
    relaxed_cap = ctx["available_gpu"] * HORIZON
    tasks = workload[workload["TaskType"] != "RealTimeInference"].sort_values(
        ["GPU_h", "LatestStart"], ascending=[False, True])
    for row in tasks.itertuples(index=False):
        candidates = allowed_regions(row, ctx, latency_map)
        rate = float(power_map[row.TaskType])
        candidates.sort(key=lambda r: (region_mean[ctx["index"][r]] * rate * ctx["pue"][ctx["index"][r]], r))
        feasible = [r for r in candidates if assigned_gpuh[ctx["index"][r]] + float(row.GPU_h) <= relaxed_cap[ctx["index"][r]] + EPS]
        selected = (feasible or candidates)[0]
        assignment[row.TaskID] = selected
        assigned_gpuh[ctx["index"][selected]] += float(row.GPU_h)
    return assignment


def choose_placement(row, candidates: list[str], ctx: dict, used_gpu: np.ndarray, used_it: np.ndarray,
                     power_rate: float, carbon_price: float, preferred: str | None) -> tuple | None:
    duration, demand = float(row.Duration_h), float(row.GPU_Demand)
    latest = min(int(row.LatestStart), int(math.floor(2406.0 - duration + EPS)))
    starts = [int(row.ArrivalHour)] if row.TaskType == "RealTimeInference" else list(range(int(row.ArrivalHour), latest + 1))
    best = None
    for region in candidates:
        rj = ctx["index"][region]
        # 阶段一仅影响同等可行解的稳定排序；阶段二仍允许所有 SLA 可达区域重指派。
        region_bias = 0.0 if region == preferred else 1e-7
        for start in starts:
            hours, overlap = overlap_arrays(start, duration)
            if not len(hours) or hours[-1] > LAST_EXEC_HOUR:
                continue
            gpu_add = demand * overlap
            it_add = demand * power_rate * overlap
            if (np.any(used_gpu[rj, hours] + gpu_add > ctx["available_gpu"][rj] + EPS) or
                    np.any(used_it[rj, hours] + it_add > ctx["it_margin"][rj, hours] + EPS)):
                continue
            before = (ctx["nonai"][rj, hours] + used_it[rj, hours]) * ctx["pue"][rj]
            after = before + it_add * ctx["pue"][rj]
            delta = facility_objective(after, ctx["renewable"][rj, hours], ctx["price"][rj, hours],
                ctx["carbon"][rj, hours], ctx["sell_price"][rj, hours], ctx["export_limit"][rj], carbon_price) - \
                facility_objective(before, ctx["renewable"][rj, hours], ctx["price"][rj, hours],
                ctx["carbon"][rj, hours], ctx["sell_price"][rj, hours], ctx["export_limit"][rj], carbon_price)
            candidate = (float(delta.sum()) + region_bias, region, rj, start, hours, overlap)
            if best is None or candidate[0] < best[0] - EPS or (abs(candidate[0] - best[0]) <= EPS and (start, region) < (best[3], best[1])):
                best = candidate
    return best


def schedule_tasks(workload: pd.DataFrame, ctx: dict, power_map: dict, latency_map: dict,
                   carbon_price: float, local_only: bool = False, label: str = "scheme_b"):
    tasks = workload.copy()
    tasks["LatestStart"] = np.floor(tasks["LatestFinishHour"] - tasks["Duration_h"] + EPS).astype(int)
    tasks["TypePriority"] = np.where(tasks["TaskType"].eq("RealTimeInference"), 0, 1)
    tasks = tasks.sort_values(["TypePriority", "LatestStart", "GPU_Demand", "ArrivalHour"],
                              ascending=[True, True, False, True])
    stage_one = {} if local_only else stage_one_assign(tasks, ctx, power_map, latency_map, carbon_price)
    used_gpu = np.zeros((len(ctx["regions"]), HORIZON), dtype=float)
    used_it = np.zeros_like(used_gpu)
    records, failures = [], []
    for number, row in enumerate(tasks.itertuples(index=False), 1):
        candidates = allowed_regions(row, ctx, latency_map, local_only)
        preferred = str(row.SourceRegion) if row.TaskType == "RealTimeInference" or local_only else stage_one.get(row.TaskID)
        placement = choose_placement(row, candidates, ctx, used_gpu, used_it, float(power_map[row.TaskType]), carbon_price, preferred)
        if placement is None:
            failures.append({"TaskID": row.TaskID, "TaskType": row.TaskType, "Reason": "no feasible region/start"})
            continue
        _, region, rj, start, hours, overlap = placement
        demand, rate = float(row.GPU_Demand), float(power_map[row.TaskType])
        used_gpu[rj, hours] += demand * overlap
        used_it[rj, hours] += demand * rate * overlap
        records.append({"TaskID": row.TaskID, "TaskType": row.TaskType, "SourceRegion": str(row.SourceRegion),
            "ExecRegion": region, "ArrivalHour": int(row.ArrivalHour), "StartHour": start,
            "Duration_h": float(row.Duration_h), "FinishHour": start + float(row.Duration_h),
            "GPU_Demand": demand, "GPU_h": float(row.GPU_h), "IT_MWh": float(row.IT_MWh),
            "MaxLatency_ms": float(row.MaxLatency_ms), "ActualLatency_ms": latency_map[(str(row.SourceRegion), region)],
            "LatestFinishHour": float(row.LatestFinishHour), "StageOneRegion": preferred})
        if number % 5000 == 0 or number == len(tasks):
            print(f"{label} 调度进度: {number}/{len(tasks)}")
    if failures:
        raise RuntimeError(f"{label} 有 {len(failures)} 个任务无法满足 C1–C7；首项：{failures[0]}")
    return pd.DataFrame(records).sort_values("TaskID"), used_gpu, used_it


def run_schedule_case(case: str, workload: pd.DataFrame, ctx: dict, power_map: dict,
                      latency_map: dict, carbon_price: float, local_only: bool = False):
    """进程池工作单元：每个方案拥有独立容量数组，因此不共享可变调度状态。"""
    schedule, used_gpu, used_it = schedule_tasks(
        workload, ctx, power_map, latency_map, carbon_price, local_only=local_only, label=case)
    return case, carbon_price, schedule, used_gpu, used_it


def usage_table(ctx: dict, used_gpu: np.ndarray, used_it: np.ndarray) -> pd.DataFrame:
    rows = []
    for rj, region in enumerate(ctx["regions"]):
        for hour in range(HORIZON):
            rows.append({"Hour": hour, "Region": region, "Scheduled_GPU_h": used_gpu[rj, hour],
                "GPU_Utilization_Percent": 100 * used_gpu[rj, hour] / ctx["available_gpu"][rj],
                "AI_IT_Load_MW": used_it[rj, hour], "IT_Power_Margin_MW": ctx["it_margin"][rj, hour] - used_it[rj, hour]})
    return pd.DataFrame(rows)


def verify_constraints(schedule: pd.DataFrame, workload: pd.DataFrame, usage: pd.DataFrame, ctx: dict,
                       latency_map: dict, fixed_realtime_local: bool = True) -> pd.DataFrame:
    merged = workload.merge(schedule, on="TaskID", suffixes=("_input", "_schedule"), how="left")
    checks = []
    def add(name, mask, detail):
        count = int(np.asarray(mask).sum())
        checks.append({"Constraint": name, "ViolationCount": count, "Status": "PASS" if count == 0 else "FAIL", "Detail": detail})
    add("C1_UniqueAssignment", merged["ExecRegion"].isna() | merged["TaskID"].duplicated(), "每个任务恰有一个执行区域")
    rt = merged["TaskType_input"].eq("RealTimeInference")
    add("C2_RealTimeFixedLocal", rt & ((merged["StartHour"] != merged["ArrivalHour_input"]) | (merged["ExecRegion"] != merged["SourceRegion_input"])), "实时任务本地、到达即开工")
    add("C3_Latency", merged["ActualLatency_ms"] > merged["MaxLatency_ms_input"] + EPS, "实际网络时延不超过 SLA")
    add("C4_Deadline", (merged["StartHour"] < merged["ArrivalHour_input"] - EPS) | (merged["FinishHour"] > merged["LatestFinishHour_input"] + EPS), "到达与截止时限")
    caps = dict(zip(ctx["regions"], ctx["available_gpu"]))
    add("C5_GPUCapacity", usage["Scheduled_GPU_h"] > usage["Region"].map(caps) + EPS, "逐时 GPU 容量")
    add("C6_ITPower", usage["IT_Power_Margin_MW"] < -EPS, "逐时 IT 功率容量")
    add("C7_NoHour2406", merged["FinishHour"] > 2406 + EPS, "任务不占用第 2406 小时")
    result = pd.DataFrame(checks)
    if (result["ViolationCount"] > 0).any():
        raise AssertionError("调度约束验证失败")
    return result


def schedule_metrics(name: str, schedule: pd.DataFrame, energy_metrics: dict, carbon_price: float) -> dict:
    work = schedule["GPU_h"].to_numpy(float)
    latency = schedule["ActualLatency_ms"].to_numpy(float)
    return {"Scenario": name, "OperatingCost_CNY": energy_metrics["OperatingCost_CNY"],
        "CarbonEmission_tCO2": energy_metrics["CarbonEmission_tCO2"],
        "Objective_CNY": energy_metrics["OperatingCost_CNY"] + carbon_price * energy_metrics["CarbonEmission_tCO2"],
        "RenewableUtilization": energy_metrics["RenewableUtilization"], "GridPurchase_MWh": energy_metrics["GridPurchase_MWh"],
        "GridSell_MWh": energy_metrics["GridSell_MWh"], "Curtailment_MWh": energy_metrics["Curtailment_MWh"],
        "MigratedTasks": int((schedule["SourceRegion"] != schedule["ExecRegion"]).sum()),
        "MigrationRate": float((schedule["SourceRegion"] != schedule["ExecRegion"]).mean()),
        "WeightedAverageLatency_ms": float(np.average(latency, weights=work)), "MaxLatency_ms": float(latency.max()),
        "AverageFlexibleWait_h": float((schedule.loc[~schedule["TaskType"].eq("RealTimeInference"), "StartHour"] -
                                         schedule.loc[~schedule["TaskType"].eq("RealTimeInference"), "ArrivalHour"]).mean())}


def reconstruct_ai_it(schedule: pd.DataFrame, ctx: dict, power_map: dict) -> np.ndarray:
    """由已有调度表按分钟级重叠重建逐时 AI IT 负荷，用于 Q1 方案统一口径重算。"""
    require_columns("已有调度", schedule, ["TaskID", "TaskType", "ExecRegion", "StartHour", "Duration_h", "GPU_Demand"])
    ai_it = np.zeros((len(ctx["regions"]), HORIZON), dtype=float)
    for row in schedule.itertuples(index=False):
        if str(row.ExecRegion) not in ctx["index"]:
            raise ValueError(f"已有调度包含未知执行区域: {row.ExecRegion}")
        hours, overlap = overlap_arrays(int(row.StartHour), float(row.Duration_h))
        if len(hours) and hours[-1] > LAST_EXEC_HOUR:
            raise ValueError(f"已有调度任务 {row.TaskID} 占用了第 2406 小时")
        ai_it[ctx["index"][str(row.ExecRegion)], hours] += float(row.GPU_Demand) * float(power_map[row.TaskType]) * overlap
    return ai_it


def baseline_from_time_data(data_dir: Path) -> dict:
    source = pd.read_excel(data_dir / "region_time_data.xlsx", sheet_name="region_time_data")
    required = ["NetGridImport_MW", "CarbonEmission_tCO2", "ElectricityPrice_CNY_per_MWh", "GridSell_MW", "SellPrice_CNY_per_MWh", "AvailableRenewable_MW", "Curtailment_MW"]
    if not set(required).issubset(source.columns):
        return {"OperatingCost_CNY": np.nan, "CarbonEmission_tCO2": np.nan, "RenewableUtilization": np.nan,
                "GridPurchase_MWh": np.nan, "GridSell_MWh": np.nan, "Curtailment_MWh": np.nan}
    cost = (source["NetGridImport_MW"] * source["ElectricityPrice_CNY_per_MWh"] -
            source["GridSell_MW"] * source["SellPrice_CNY_per_MWh"]).sum()
    renewable = float(source["AvailableRenewable_MW"].sum())
    curtail = float(source["Curtailment_MW"].sum())
    return {"OperatingCost_CNY": float(cost), "CarbonEmission_tCO2": float(source["CarbonEmission_tCO2"].sum()),
        "RenewableUtilization": 1 - curtail / renewable if renewable else 0.0,
        "GridPurchase_MWh": float(source["NetGridImport_MW"].clip(lower=0).sum()),
        "GridSell_MWh": float(source["GridSell_MW"].sum()), "Curtailment_MWh": curtail}


def make_plots(out_dir: Path, metrics: pd.DataFrame, scheme_b: pd.DataFrame, usage: pd.DataFrame, ctx: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = out_dir / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False
    plotted = metrics.dropna(subset=["OperatingCost_CNY"])
    ax = plotted.set_index("Scenario")[["OperatingCost_CNY", "CarbonEmission_tCO2"]].plot(kind="bar", secondary_y="CarbonEmission_tCO2", figsize=(10, 5))
    ax.set_ylabel("Operating cost (CNY)"); ax.right_ax.set_ylabel("Carbon emissions (tCO2)"); plt.tight_layout(); plt.savefig(plot_dir / "scenario_comparison.png", dpi=180); plt.close()
    migration = scheme_b.groupby(["SourceRegion", "ExecRegion"], observed=False)["GPU_h"].sum().unstack(fill_value=0)
    ax = migration.plot(kind="bar", stacked=True, figsize=(11, 5)); ax.set_ylabel("Migrated / executed GPU-hour"); plt.tight_layout(); plt.savefig(plot_dir / "migration_flow_by_source.png", dpi=180); plt.close()
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for axis, region in zip(axes.flat, ctx["regions"]):
        part = usage[(usage["Region"] == region) & usage["Hour"].between(2376, 2405)]
        axis.plot(part["Hour"], part["GPU_Utilization_Percent"]); axis.set_title(region); axis.set_ylabel("GPU utilization (%)"); axis.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(plot_dir / "gpu_utilization_2376_2405.png", dpi=180); plt.close(fig)


def run_schedule_cases(cases: list[tuple[str, float, bool]], workload: pd.DataFrame, ctx: dict,
                       power_map: dict, latency_map: dict, workers: int) -> dict:
    """执行相互独立的方案；单方案内部保持串行以保证容量累计的确定性。"""
    if workers <= 1 or len(cases) == 1:
        return {case: run_schedule_case(case, workload, ctx, power_map, latency_map, price, local_only)
                for case, price, local_only in cases}
    results = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(cases))) as executor:
        futures = {
            executor.submit(run_schedule_case, case, workload, ctx, power_map, latency_map, price, local_only): case
            for case, price, local_only in cases
        }
        for future in as_completed(futures):
            case = futures[future]
            results[case] = future.result()
            print(f"{case} 已完成")
    return results


def main() -> None:
    args = parse_args()
    if args.carbon_price < 0:
        raise ValueError("carbon-price 不得为负")
    if args.workers < 1:
        raise ValueError("workers 至少为 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("读取问题二输入数据……")
    workload, gpu, energy, power_map, latency_map, regions = load_data(args.data_dir)
    ctx = build_context(gpu, energy, regions)
    lambda_values = sorted(set(args.lambda_values + [args.carbon_price])) if not args.skip_sensitivity else [args.carbon_price]
    cases = [("方案A", args.carbon_price, True), ("方案B", args.carbon_price, False)]
    cases.extend((f"lambda={value:g}", value, False) for value in lambda_values if abs(value - args.carbon_price) > EPS)
    if args.workers > 1:
        print(f"并行运行 {len(cases)} 个独立调度任务，最多使用 {min(args.workers, len(cases))} 个进程……")
    else:
        print("串行运行调度任务；可通过 --workers 8 并行方案与 lambda 灵敏度计算。")
    results = run_schedule_cases(cases, workload, ctx, power_map, latency_map, args.workers)
    _, _, scheme_a, a_gpu, a_it = results["方案A"]
    a_usage = usage_table(ctx, a_gpu, a_it)
    a_verify = verify_constraints(scheme_a, workload, a_usage, ctx, latency_map)
    a_energy, a_energy_metrics = energy_balance(ctx, a_it)
    _, _, scheme_b, b_gpu, b_it = results["方案B"]
    b_usage = usage_table(ctx, b_gpu, b_it)
    b_verify = verify_constraints(scheme_b, workload, b_usage, ctx, latency_map)
    b_energy, b_energy_metrics = energy_balance(ctx, b_it)

    scheme_dir = args.output_dir / "schedule"; energy_dir = args.output_dir / "energy"; report_dir = args.output_dir / "reports"
    for directory in [scheme_dir, energy_dir, report_dir]: directory.mkdir(parents=True, exist_ok=True)
    scheme_a.to_csv(scheme_dir / "scheme_a_no_migration_schedule.csv", index=False, encoding="utf-8-sig")
    scheme_b.to_csv(scheme_dir / "scheme_b_carbon_aware_schedule.csv", index=False, encoding="utf-8-sig")
    a_usage.to_csv(scheme_dir / "scheme_a_hourly_usage.csv", index=False, encoding="utf-8-sig")
    b_usage.to_csv(scheme_dir / "scheme_b_hourly_usage.csv", index=False, encoding="utf-8-sig")
    pd.concat([a_verify.assign(Scenario="SchemeA"), b_verify.assign(Scenario="SchemeB")]).to_csv(scheme_dir / "constraint_verification.csv", index=False, encoding="utf-8-sig")
    a_energy.to_csv(energy_dir / "scheme_a_energy_balance.csv", index=False, encoding="utf-8-sig")
    b_energy.to_csv(energy_dir / "scheme_b_energy_balance.csv", index=False, encoding="utf-8-sig")

    baseline = baseline_from_time_data(args.data_dir)
    rows = [{"Scenario": "AttachmentBaseline", **baseline}]
    if args.q1_schedule is not None:
        q1_schedule = pd.read_csv(args.q1_schedule)
        q1_ai_it = reconstruct_ai_it(q1_schedule, ctx, power_map)
        _, q1_energy_metrics = energy_balance(ctx, q1_ai_it)
        rows.append(schedule_metrics("Q1Schedule_Recalculated", q1_schedule, q1_energy_metrics, args.carbon_price))
    rows.extend([
        schedule_metrics("SchemeA_RenewableFirst_NoMigration", scheme_a, a_energy_metrics, args.carbon_price),
        schedule_metrics("SchemeB_CarbonAware", scheme_b, b_energy_metrics, args.carbon_price),
    ])
    metrics = pd.DataFrame(rows)
    metrics.to_csv(report_dir / "scenario_comparison.csv", index=False, encoding="utf-8-sig")
    scheme_a_row = metrics.loc[metrics["Scenario"] == "SchemeA_RenewableFirst_NoMigration"].iloc[0]
    scheme_b_row = metrics.loc[metrics["Scenario"] == "SchemeB_CarbonAware"].iloc[0]
    marginal = scheme_a_row.copy()
    for key in ["OperatingCost_CNY", "CarbonEmission_tCO2", "Objective_CNY", "GridPurchase_MWh", "GridSell_MWh", "Curtailment_MWh", "MigratedTasks"]:
        marginal[key] = scheme_a_row[key] - scheme_b_row[key]
    marginal["Scenario"] = "SchemeB_marginal_benefit_vs_SchemeA"
    pd.DataFrame([marginal]).to_csv(report_dir / "migration_marginal_contribution.csv", index=False, encoding="utf-8-sig")

    if not args.skip_sensitivity:
        sensitivity = []
        for value in lambda_values:
            if abs(value - args.carbon_price) <= EPS:
                candidate, candidate_energy = scheme_b, b_energy_metrics
            else:
                _, _, candidate, _, candidate_it = results[f"lambda={value:g}"]
                candidate_energy = energy_balance(ctx, candidate_it)[1]
            sensitivity.append(schedule_metrics(f"lambda={value:g}", candidate, candidate_energy, value))
        pd.DataFrame(sensitivity).to_csv(report_dir / "lambda_sensitivity.csv", index=False, encoding="utf-8-sig")
    if not args.skip_plots:
        make_plots(args.output_dir, metrics, scheme_b, b_usage, ctx)
    summary = {"carbon_price_CNY_per_tCO2": args.carbon_price, "task_count": int(len(workload)),
        "scheme_a_scheduled_tasks": int(len(scheme_a)), "scheme_b_scheduled_tasks": int(len(scheme_b)),
        "all_constraints_passed": True, "scheme_a": a_energy_metrics, "scheme_b": b_energy_metrics,
        "migration_marginal_cost_reduction_CNY": float(a_energy_metrics["OperatingCost_CNY"] - b_energy_metrics["OperatingCost_CNY"]),
        "migration_marginal_carbon_reduction_tCO2": float(a_energy_metrics["CarbonEmission_tCO2"] - b_energy_metrics["CarbonEmission_tCO2"])}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成。结果目录：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
