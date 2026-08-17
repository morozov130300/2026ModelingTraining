#!/usr/bin/env python3
"""问题二 GPU 加速版。

使用 CuPy 将候选开工时刻的容量筛选和边际能源目标计算放到 CUDA GPU；
调度顺序仍保持 EDF，容量状态仍按任务顺序累计，因此结果口径与 t2.py 一致。
若未安装 CuPy，会明确提示并回退到 CPU 版 t2.py。

安装示例：
    python -m pip install cupy-cuda12x
运行示例：
    python t2/t2_GPU.py --workers 1 --skip-plots
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cupy as cp
except ImportError:
    cp = None

import t2 as base


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="问题二 CuPy GPU 加速调度")
    parser.add_argument("--data-dir", type=Path, default=here.parent / "题目")
    parser.add_argument("--output-dir", type=Path, default=here / "output_gpu")
    parser.add_argument("--carbon-price", type=float, default=200.0)
    parser.add_argument("--lambda-values", type=float, nargs="*", default=[0, 50, 100, 150, 200, 300, 500])
    parser.add_argument("--q1-schedule", type=Path, default=here.parent / "t1" / "output" / "schedule" / "schedule.csv")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--gpu-device", type=int, default=0, help="CUDA 设备编号")
    parser.add_argument("--gpu-memory-pool-limit", type=int, default=0,
                        help="GPU 内存池上限（字节），0 表示不限制")
    return parser.parse_args()


def gpu_objective(load, renewable, price, carbon, sell_price, export_limit, carbon_price):
    purchase = cp.maximum(0.0, load - renewable)
    surplus = cp.maximum(0.0, renewable - load)
    sell = cp.minimum(export_limit, surplus)
    return purchase * (price + carbon_price * carbon) - sell * sell_price


def gpu_choose_placement(row, candidates, ctx_gpu, used_gpu, used_it, power_rate,
                         carbon_price, preferred):
    """批量评估一个任务在一个区域所有候选整数开工时刻。"""
    duration = float(row.Duration_h)
    demand = float(row.GPU_Demand)
    latest = min(int(row.LatestStart), int(math.floor(2406.0 - duration + base.EPS)))
    if row.TaskType == "RealTimeInference":
        starts = np.asarray([int(row.ArrivalHour)], dtype=np.int32)
    else:
        if latest < int(row.ArrivalHour):
            return None
        starts = np.arange(int(row.ArrivalHour), latest + 1, dtype=np.int32)
    if starts.size == 0:
        return None

    max_span = max(1, int(math.ceil(duration - base.EPS)))
    start_gpu = cp.asarray(starts)
    offsets = cp.arange(max_span, dtype=cp.int32)
    hours = start_gpu[:, None] + offsets[None, :]
    finish = start_gpu.astype(cp.float64) + duration
    overlap = cp.maximum(0.0, cp.minimum(hours.astype(cp.float64) + 1.0, finish[:, None]) - hours)
    valid_hour = (hours >= 0) & (hours <= base.LAST_EXEC_HOUR)
    overlap = cp.where(valid_hour, overlap, 0.0)
    gpu_add = demand * overlap
    it_add = demand * power_rate * overlap

    best = None
    for region in candidates:
        rj = ctx_gpu["index"][region]
        safe_hours = cp.clip(hours, 0, base.LAST_EXEC_HOUR)
        current_gpu = used_gpu[rj][safe_hours]
        current_it = used_it[rj][safe_hours]
        feasible = cp.all((current_gpu + gpu_add <= ctx_gpu["available_gpu"][rj] + base.EPS) &
                          (current_it + it_add <= ctx_gpu["it_margin"][rj][safe_hours] + base.EPS), axis=1)
        feasible &= cp.any(overlap > base.EPS, axis=1)
        if not bool(cp.any(feasible).item()):
            continue

        before = (ctx_gpu["nonai"][rj][safe_hours] + current_it) * ctx_gpu["pue"][rj]
        after = before + it_add * ctx_gpu["pue"][rj]
        renewable = ctx_gpu["renewable"][rj][safe_hours]
        price = ctx_gpu["price"][rj][safe_hours]
        carbon = ctx_gpu["carbon"][rj][safe_hours]
        sell_price = ctx_gpu["sell_price"][rj][safe_hours]
        delta = gpu_objective(after, renewable, price, carbon, sell_price,
                              ctx_gpu["export_limit"][rj], carbon_price) - gpu_objective(
                                  before, renewable, price, carbon, sell_price,
                                  ctx_gpu["export_limit"][rj], carbon_price)
        score = cp.sum(cp.where(overlap > base.EPS, delta, 0.0), axis=1)
        score = cp.where(feasible, score, cp.inf)
        index = int(cp.argmin(score).item())
        value = float(score[index].item())
        start = int(starts[index])
        region_bias = 0.0 if region == preferred else 1e-7
        candidate = (value + region_bias, region, rj, start,
                     cp.asnumpy(hours[index][overlap[index] > base.EPS]).astype(int),
                     cp.asnumpy(overlap[index][overlap[index] > base.EPS]))
        if best is None or candidate[0] < best[0] - base.EPS or \
                (abs(candidate[0] - best[0]) <= base.EPS and (start, region) < (best[3], best[1])):
            best = candidate
    return best


def gpu_schedule_tasks(workload, ctx, power_map, latency_map, carbon_price,
                       local_only=False, label="GPU"):
    tasks = workload.copy()
    tasks["LatestStart"] = np.floor(tasks["LatestFinishHour"] - tasks["Duration_h"] + base.EPS).astype(int)
    tasks["TypePriority"] = np.where(tasks["TaskType"].eq("RealTimeInference"), 0, 1)
    tasks = tasks.sort_values(["TypePriority", "LatestStart", "GPU_Demand", "ArrivalHour"],
                              ascending=[True, True, False, True])
    stage_one = {} if local_only else base.stage_one_assign(tasks, ctx, power_map, latency_map, carbon_price)
    n_regions = len(ctx["regions"])
    used_gpu = cp.zeros((n_regions, base.HORIZON), dtype=cp.float64)
    used_it = cp.zeros_like(used_gpu)
    ctx_gpu = {"index": ctx["index"], "regions": ctx["regions"]}
    for key in ["available_gpu", "pue", "it_margin", "nonai", "price", "carbon", "renewable", "sell_price", "export_limit"]:
        ctx_gpu[key] = cp.asarray(ctx[key])
    records = []
    for number, row in enumerate(tasks.itertuples(index=False), 1):
        candidates = base.allowed_regions(row, ctx, latency_map, local_only)
        preferred = str(row.SourceRegion) if row.TaskType == "RealTimeInference" or local_only else stage_one.get(row.TaskID)
        placement = gpu_choose_placement(row, candidates, ctx_gpu, used_gpu, used_it,
                                         float(power_map[row.TaskType]), carbon_price, preferred)
        if placement is None:
            raise RuntimeError(f"{label} 任务 {row.TaskID} 无可行区域/时刻")
        _, region, rj, start, hours, overlap = placement
        demand = float(row.GPU_Demand)
        rate = float(power_map[row.TaskType])
        hours_gpu = cp.asarray(hours)
        used_gpu[rj, hours_gpu] += demand * cp.asarray(overlap)
        used_it[rj, hours_gpu] += demand * rate * cp.asarray(overlap)
        records.append({"TaskID": row.TaskID, "TaskType": row.TaskType, "SourceRegion": str(row.SourceRegion),
            "ExecRegion": region, "ArrivalHour": int(row.ArrivalHour), "StartHour": start,
            "Duration_h": float(row.Duration_h), "FinishHour": start + float(row.Duration_h),
            "GPU_Demand": demand, "GPU_h": float(row.GPU_h), "IT_MWh": float(row.IT_MWh),
            "MaxLatency_ms": float(row.MaxLatency_ms), "ActualLatency_ms": latency_map[(str(row.SourceRegion), region)],
            "LatestFinishHour": float(row.LatestFinishHour), "StageOneRegion": preferred})
        if number % 5000 == 0 or number == len(tasks):
            print(f"{label} 调度进度: {number}/{len(tasks)}", flush=True)
    return pd.DataFrame(records).sort_values("TaskID"), cp.asnumpy(used_gpu), cp.asnumpy(used_it)


def main():
    args = parse_args()
    if cp is None:
        raise RuntimeError(
            "未安装 CuPy，无法运行 GPU 版本。请按 CUDA 版本安装对应 CuPy，"
            "例如：python -m pip install cupy-cuda12x"
        )
    if args.carbon_price < 0:
        raise ValueError("carbon-price 不得为负")
    with cp.cuda.Device(args.gpu_device):
        if args.gpu_memory_pool_limit > 0:
            cp.get_default_memory_pool().set_limit(size=args.gpu_memory_pool_limit)
        props = cp.cuda.runtime.getDeviceProperties(args.gpu_device)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        print(f"使用 GPU: {name}", flush=True)
        workload, gpu, energy, power_map, latency_map, regions = base.load_data(args.data_dir)
        ctx = base.build_context(gpu, energy, regions)
        lambda_values = sorted(set(args.lambda_values + [args.carbon_price])) if not args.skip_sensitivity else [args.carbon_price]
        cases = [("方案A", args.carbon_price, True), ("方案B", args.carbon_price, False)]
        cases += [(f"lambda={value:g}", value, False) for value in lambda_values if abs(value - args.carbon_price) > base.EPS]
        results = {}
        for case, value, local_only in cases:
            results[case] = (case, value, *gpu_schedule_tasks(workload, ctx, power_map, latency_map, value, local_only, case))
        _, _, scheme_a, a_gpu, a_it = results["方案A"]
        _, _, scheme_b, b_gpu, b_it = results["方案B"]
        out = args.output_dir
        schedule_dir, energy_dir, report_dir = out / "schedule", out / "energy", out / "reports"
        for directory in (schedule_dir, energy_dir, report_dir): directory.mkdir(parents=True, exist_ok=True)
        a_usage = base.usage_table(ctx, a_gpu, a_it); b_usage = base.usage_table(ctx, b_gpu, b_it)
        a_verify = base.verify_constraints(scheme_a, workload, a_usage, ctx, latency_map)
        b_verify = base.verify_constraints(scheme_b, workload, b_usage, ctx, latency_map)
        a_energy, a_metrics = base.energy_balance(ctx, a_it); b_energy, b_metrics = base.energy_balance(ctx, b_it)
        scheme_a.to_csv(schedule_dir / "scheme_a_no_migration_schedule.csv", index=False, encoding="utf-8-sig")
        scheme_b.to_csv(schedule_dir / "scheme_b_carbon_aware_schedule.csv", index=False, encoding="utf-8-sig")
        a_usage.to_csv(schedule_dir / "scheme_a_hourly_usage.csv", index=False, encoding="utf-8-sig")
        b_usage.to_csv(schedule_dir / "scheme_b_hourly_usage.csv", index=False, encoding="utf-8-sig")
        pd.concat([a_verify.assign(Scenario="SchemeA"), b_verify.assign(Scenario="SchemeB")]).to_csv(schedule_dir / "constraint_verification.csv", index=False, encoding="utf-8-sig")
        a_energy.to_csv(energy_dir / "scheme_a_energy_balance.csv", index=False, encoding="utf-8-sig")
        b_energy.to_csv(energy_dir / "scheme_b_energy_balance.csv", index=False, encoding="utf-8-sig")
        rows = [base.schedule_metrics("SchemeA_RenewableFirst_NoMigration", scheme_a, a_metrics, args.carbon_price),
                base.schedule_metrics("SchemeB_CarbonAware", scheme_b, b_metrics, args.carbon_price)]
        if not args.skip_sensitivity:
            for value in lambda_values:
                candidate = scheme_b if abs(value - args.carbon_price) <= base.EPS else results[f"lambda={value:g}"][2]
                candidate_it = b_it if abs(value - args.carbon_price) <= base.EPS else results[f"lambda={value:g}"][4]
                rows.append(base.schedule_metrics(f"lambda={value:g}", candidate, base.energy_balance(ctx, candidate_it)[1], value))
            pd.DataFrame(rows[2:]).to_csv(report_dir / "lambda_sensitivity.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(rows).to_csv(report_dir / "scenario_comparison.csv", index=False, encoding="utf-8-sig")
        if not args.skip_plots:
            base.make_plots(out, pd.DataFrame(rows), scheme_b, b_usage, ctx)
        (out / "summary.json").write_text(pd.Series({"backend": "cupy", "gpu": name, "task_count": len(workload), "scheme_a_scheduled_tasks": len(scheme_a), "scheme_b_scheduled_tasks": len(scheme_b), "all_constraints_passed": True}).to_json(force_ascii=False, indent=2), encoding="utf-8")
        cp.cuda.Stream.null.synchronize()
        print(f"GPU 调度完成。结果目录：{out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
