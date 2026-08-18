#!/usr/bin/env python3
"""问题一：统计分析、短期预测与基础算力调度。

依赖：python>=3.9, numpy, pandas, openpyxl, scikit-learn, matplotlib
运行：python t1.py [--data-dir ../题目] [--output-dir output]
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

TASK_TYPES = ["RealTimeInference", "BatchInference", "AITraining"]
HORIZON = 2407
LAST_EXEC_HOUR = 2405
EPS = 1e-9


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="求解数学建模训练题问题一")
    parser.add_argument("--data-dir", type=Path, default=here.parent / "题目")
    parser.add_argument("--output-dir", type=Path, default=here / "output")
    parser.add_argument("--skip-forecast", action="store_true", help="跳过短期预测")
    parser.add_argument("--skip-plots", action="store_true", help="跳过绘图")
    return parser.parse_args()


def load_data(data_dir: Path):
    workload = pd.read_excel(data_dir / "workload_trace.xlsx", sheet_name="Sheet1")
    gpu = pd.read_excel(data_dir / "GPU_information.xlsx", sheet_name="GPU中心基础情况")
    latency = pd.read_excel(data_dir / "network_latency.xlsx", sheet_name="network_latency")
    power = pd.read_excel(data_dir / "power_mapping.xlsx", sheet_name="任务功率映射")
    time_data = pd.read_excel(data_dir / "region_time_data.xlsx", sheet_name="region_time_data")

    required = {
        "workload": (workload, ["TaskID", "TaskType", "ArrivalHour", "GPU_Demand", "EstimatedDuration_min", "SourceRegion", "MaxLatency_ms", "LatestFinishHour"]),
        "gpu": (gpu, ["Region", "Available_GPU", "Max_IT_Power_MW", "PUE", "Max_Facility_Power_MW"]),
        "latency": (latency, ["FromRegion", "ToRegion", "NetworkLatency_ms"]),
        "power": (power, ["TaskType", "GPU_Power_MW_per_EquivalentGPU"]),
        "time": (time_data, ["Hour", "Region", "NonAI_IT_Load_MW"]),
    }
    for name, (frame, columns) in required.items():
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} 缺少字段: {sorted(missing)}")

    workload = workload.copy()
    workload["Duration_h"] = workload["EstimatedDuration_min"].astype(float) / 60.0
    workload["GPU_h"] = workload["GPU_Demand"] * workload["Duration_h"]
    power_map = power.set_index("TaskType")["GPU_Power_MW_per_EquivalentGPU"].astype(float).to_dict()
    unknown = set(workload["TaskType"]) - set(power_map)
    if unknown:
        raise ValueError(f"power_mapping 缺少任务类型: {sorted(unknown)}")
    workload["IT_MWh"] = workload["GPU_h"] * workload["TaskType"].map(power_map)

    regions = gpu["Region"].astype(str).tolist()
    implied_facility_cap = gpu["Max_IT_Power_MW"].astype(float) * gpu["PUE"].astype(float)
    if not np.allclose(implied_facility_cap, gpu["Max_Facility_Power_MW"].astype(float), rtol=0.0, atol=1e-6):
        raise ValueError("Max_Facility_Power_MW 不等于 Max_IT_Power_MW × PUE，不能省略设施功率约束")
    latency_map = {(str(x.FromRegion), str(x.ToRegion)): float(x.NetworkLatency_ms) for x in latency.itertuples()}
    return workload, gpu, time_data, power_map, latency_map, regions


def statistical_analysis(workload, gpu, time_data, power_map, regions, out_dir):
    stats_dir = out_dir / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    count_pivot = pd.pivot_table(workload, index="SourceRegion", columns="TaskType", values="TaskID", aggfunc="count", fill_value=0, margins=True)
    gpuh_pivot = pd.pivot_table(workload, index="SourceRegion", columns="TaskType", values="GPU_h", aggfunc="sum", fill_value=0, margins=True)
    energy_pivot = pd.pivot_table(workload, index="SourceRegion", columns="TaskType", values="IT_MWh", aggfunc="sum", fill_value=0, margins=True)
    count_pivot.to_csv(stats_dir / "region_type_task_count.csv", encoding="utf-8-sig")
    gpuh_pivot.to_csv(stats_dir / "region_type_gpuh.csv", encoding="utf-8-sig")
    energy_pivot.to_csv(stats_dir / "region_type_it_energy.csv", encoding="utf-8-sig")

    type_summary = workload.groupby("TaskType", observed=False).agg(TaskCount=("TaskID", "count"), GPU_h=("GPU_h", "sum"), IT_MWh=("IT_MWh", "sum")).reindex(TASK_TYPES)
    for col in ["TaskCount", "GPU_h", "IT_MWh"]:
        type_summary[col + "_Share"] = type_summary[col] / type_summary[col].sum()
    type_summary.to_csv(stats_dir / "task_type_summary.csv", encoding="utf-8-sig")

    hourly = workload.groupby(["ArrivalHour", "SourceRegion", "TaskType"], observed=False)["GPU_h"].sum().unstack(["SourceRegion", "TaskType"], fill_value=0)
    full_index = pd.Index(range(2400), name="ArrivalHour")
    full_columns = pd.MultiIndex.from_product([regions, TASK_TYPES], names=["SourceRegion", "TaskType"])
    hourly = hourly.reindex(index=full_index, columns=full_columns, fill_value=0.0)
    hourly.to_csv(stats_dir / "hourly_region_type_gpuh.csv", encoding="utf-8-sig")

    daily_profile = hourly.groupby(hourly.index % 24).mean()
    daily_profile.index.name = "HourOfDay"
    daily_profile.to_csv(stats_dir / "daily_profile_gpuh.csv", encoding="utf-8-sig")

    cap = gpu.set_index("Region")
    margins = time_data[["Hour", "Region", "NonAI_IT_Load_MW"]].copy()
    margins["IT_Power_Margin_MW"] = margins["Region"].map(cap["Max_IT_Power_MW"]) - margins["NonAI_IT_Load_MW"]
    margins.to_csv(stats_dir / "it_power_margin_timeseries.csv", index=False, encoding="utf-8-sig")
    margin_summary = margins.groupby("Region")["IT_Power_Margin_MW"].agg(["min", "mean", "max"])
    margin_summary.to_csv(stats_dir / "it_power_margin_summary.csv", encoding="utf-8-sig")

    structure = workload.pivot_table(index="SourceRegion", columns="TaskType", values="TaskID", aggfunc="count", fill_value=0)
    structure = structure.div(structure.sum(axis=1), axis=0)
    structure.to_csv(stats_dir / "region_task_type_share.csv", encoding="utf-8-sig")
    return hourly, type_summary, margins


def feature_vector(history: list[float], t: int) -> list[float]:
    def lag(k):
        return history[t - k] if t >= k else 0.0

    last12 = history[max(0, t - 12):t]
    last24 = history[max(0, t - 24):t]
    recent = last24 if last24 else [0.0]
    angle = 2.0 * math.pi * (t % 24) / 24.0
    return [
        lag(1), lag(2), lag(3), lag(6),
        float(np.mean(last12)) if last12 else 0.0,
        float(np.mean(last24)) if last24 else 0.0,
        float(np.max(recent)), float(np.std(recent)),
        lag(24), lag(48), lag(72), lag(168),
        math.sin(angle), math.cos(angle),
    ]


def supervised_xy(y: np.ndarray, end_exclusive: int, min_t: int = 168):
    history = y.tolist()
    times = range(min_t, end_exclusive)
    x = np.asarray([feature_vector(history, t) for t in times], dtype=float)
    target = y[min_t:end_exclusive]
    return x, target


def recursive_predict(model, known: np.ndarray, start: int, end: int) -> np.ndarray:
    history = known[:start].astype(float).tolist()
    predictions = []
    for t in range(start, end):
        pred = max(0.0, float(model.predict(np.asarray([feature_vector(history, t)]))[0]))
        history.append(pred)
        predictions.append(pred)
    return np.asarray(predictions)


def metric_row(actual, pred, train):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    denom = np.abs(actual) + np.abs(pred)
    smape_terms = np.zeros_like(denom, dtype=float)
    np.divide(2 * np.abs(actual - pred), denom, out=smape_terms, where=denom > EPS)
    smape = float(np.mean(smape_terms) * 100)
    seasonal_scale = float(np.mean(np.abs(train[24:] - train[:-24]))) if len(train) > 24 else float("nan")
    mase = float(np.mean(np.abs(actual - pred)) / seasonal_scale) if seasonal_scale > EPS else float("nan")
    return rmse, smape, mase


def forecast(hourly, regions, out_dir):
    from sklearn.ensemble import HistGradientBoostingRegressor

    forecast_dir = out_dir / "forecast"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    param_grid = [
        {"learning_rate": 0.05, "max_iter": 150, "max_leaf_nodes": 15, "l2_regularization": 1.0},
        {"learning_rate": 0.05, "max_iter": 220, "max_leaf_nodes": 31, "l2_regularization": 1.0},
        {"learning_rate": 0.10, "max_iter": 150, "max_leaf_nodes": 15, "l2_regularization": 2.0},
    ]
    prediction_rows, metric_rows, selected_rows = [], [], []

    for region in regions:
        for task_type in TASK_TYPES:
            y = hourly[(region, task_type)].to_numpy(dtype=float)
            x_train, y_train = supervised_xy(y, 2352)
            best_params, best_rmse = None, float("inf")
            for params in param_grid:
                model = HistGradientBoostingRegressor(random_state=2026, **params).fit(x_train, y_train)
                val_pred = recursive_predict(model, y, 2352, 2376)
                score = metric_row(y[2352:2376], val_pred, y[:2352])[0]
                if score < best_rmse:
                    best_rmse, best_params = score, params

            x_refit, y_refit = supervised_xy(y, 2376)
            model = HistGradientBoostingRegressor(random_state=2026, **best_params).fit(x_refit, y_refit)
            gbdt_pred = recursive_predict(model, y, 2376, 2400)
            last_value = np.repeat(y[2375], 24)
            seasonal_mean = np.asarray([np.mean([y[t - 24], y[t - 48], y[t - 72]]) for t in range(2376, 2400)])
            actual = y[2376:2400]

            selected_rows.append({"Region": region, "TaskType": task_type, "Validation_RMSE": best_rmse, **best_params})
            for name, pred in [("GBDT", gbdt_pred), ("LastValue", last_value), ("SeasonalMean", seasonal_mean)]:
                rmse, smape, mase = metric_row(actual, pred, y[:2376])
                metric_rows.append({"Level": "Series", "Region": region, "TaskType": task_type, "Model": name, "RMSE": rmse, "sMAPE_percent": smape, "MASE": mase})
                for hour, a, p in zip(range(2376, 2400), actual, pred):
                    prediction_rows.append({"Hour": hour, "Region": region, "TaskType": task_type, "Model": name, "Actual_GPU_h": a, "Predicted_GPU_h": p})

    predictions = pd.DataFrame(prediction_rows)
    for model_name, group in predictions.groupby("Model"):
        for level, keys in [("Region", ["Hour", "Region"]), ("Total", ["Hour"])]:
            agg = group.groupby(keys)[["Actual_GPU_h", "Predicted_GPU_h"]].sum().reset_index()
            if level == "Region":
                for region, rg in agg.groupby("Region"):
                    rmse, smape, mase = metric_row(rg["Actual_GPU_h"], rg["Predicted_GPU_h"], np.array([]))
                    metric_rows.append({"Level": level, "Region": region, "TaskType": "ALL", "Model": model_name, "RMSE": rmse, "sMAPE_percent": smape, "MASE": np.nan})
            else:
                rmse, smape, mase = metric_row(agg["Actual_GPU_h"], agg["Predicted_GPU_h"], np.array([]))
                metric_rows.append({"Level": level, "Region": "ALL", "TaskType": "ALL", "Model": model_name, "RMSE": rmse, "sMAPE_percent": smape, "MASE": np.nan})

    predictions.to_csv(forecast_dir / "predictions_2376_2399.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(forecast_dir / "forecast_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(forecast_dir / "selected_parameters.csv", index=False, encoding="utf-8-sig")
    return predictions


def overlap_arrays(start: int, duration: float):
    finish = start + duration
    hours = np.arange(start, min(HORIZON, int(math.ceil(finish - EPS))))
    overlaps = np.maximum(0.0, np.minimum(hours + 1.0, finish) - np.maximum(hours.astype(float), float(start)))
    mask = overlaps > EPS
    return hours[mask], overlaps[mask]


def schedule_tasks(workload, gpu, time_data, power_map, latency_map, regions, out_dir):
    schedule_dir = out_dir / "schedule"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    r_index = {r: j for j, r in enumerate(regions)}
    cap = gpu.set_index("Region")
    available_gpu = cap.loc[regions, "Available_GPU"].to_numpy(float)
    max_it = cap.loc[regions, "Max_IT_Power_MW"].to_numpy(float)

    nonai_table = time_data.pivot_table(index="Hour", columns="Region", values="NonAI_IT_Load_MW", aggfunc="first").reindex(index=range(HORIZON), columns=regions)
    if nonai_table.isna().any().any():
        raise ValueError("region_time_data 缺少 0-2406 某些区域的 NonAI_IT_Load_MW")
    it_margin = max_it[:, None] - nonai_table.to_numpy(float).T
    if np.any(it_margin < -EPS):
        raise ValueError("存在 NonAI_IT_Load_MW 已超过 Max_IT_Power_MW 的时段")

    used_gpu = np.zeros((len(regions), HORIZON), dtype=float)
    used_it = np.zeros((len(regions), HORIZON), dtype=float)
    records, unscheduled = [], []

    tasks = workload.copy()
    tasks["LatestStart"] = np.floor(tasks["LatestFinishHour"] - tasks["Duration_h"] + EPS).astype(int)
    tasks["TypePriority"] = np.where(tasks["TaskType"].eq("RealTimeInference"), 0, 1)
    tasks = tasks.sort_values(["TypePriority", "LatestStart", "GPU_Demand", "ArrivalHour"], ascending=[True, True, False, True])

    def feasible(rj, start, duration, gpu_demand, power_rate):
        hours, overlaps = overlap_arrays(start, duration)
        if len(hours) == 0 or hours[-1] > LAST_EXEC_HOUR:
            return False, hours, overlaps
        gpu_add = gpu_demand * overlaps
        it_add = gpu_demand * power_rate * overlaps
        ok = np.all(used_gpu[rj, hours] + gpu_add <= available_gpu[rj] + EPS) and np.all(used_it[rj, hours] + it_add <= it_margin[rj, hours] + EPS)
        return bool(ok), hours, overlaps

    total = len(tasks)
    for number, row in enumerate(tasks.itertuples(index=False), 1):
        source = str(row.SourceRegion)
        candidates = [r for r in regions if latency_map.get((source, r), float("inf")) <= float(row.MaxLatency_ms) + EPS]
        candidates.sort(key=lambda r: (r != source, used_gpu[r_index[r]].sum() / max(available_gpu[r_index[r]], EPS)))
        duration, demand = float(row.Duration_h), float(row.GPU_Demand)
        rate = float(power_map[row.TaskType])
        latest_start = min(int(row.LatestStart), int(math.floor(2406.0 - duration + EPS)))
        starts = [int(row.ArrivalHour)] if row.TaskType == "RealTimeInference" else range(int(row.ArrivalHour), latest_start + 1)
        chosen = None
        for r in candidates:
            rj = r_index[r]
            for start in starts:
                ok, hours, overlaps = feasible(rj, start, duration, demand, rate)
                if ok:
                    chosen = (r, rj, start, hours, overlaps)
                    break
            if chosen is not None:
                break
        if chosen is None:
            unscheduled.append({"TaskID": row.TaskID, "TaskType": row.TaskType, "ArrivalHour": row.ArrivalHour, "Reason": "no feasible region/start"})
            continue

        r, rj, start, hours, overlaps = chosen
        used_gpu[rj, hours] += demand * overlaps
        used_it[rj, hours] += demand * rate * overlaps
        records.append({
            "TaskID": row.TaskID, "TaskType": row.TaskType, "SourceRegion": source, "ExecRegion": r,
            "ArrivalHour": int(row.ArrivalHour), "StartHour": start, "Duration_h": duration,
            "FinishHour": start + duration, "GPU_Demand": demand, "GPU_h": demand * duration,
            "IT_MWh": demand * duration * rate, "MaxLatency_ms": float(row.MaxLatency_ms),
            "ActualLatency_ms": latency_map[(source, r)], "LatestFinishHour": float(row.LatestFinishHour),
        })
        if number % 5000 == 0 or number == total:
            print(f"调度进度: {number}/{total}")

    schedule = pd.DataFrame(records)
    if unscheduled:
        pd.DataFrame(unscheduled).to_csv(schedule_dir / "unscheduled.csv", index=False, encoding="utf-8-sig")
        raise RuntimeError(f"有 {len(unscheduled)} 个任务无法调度，详见 schedule/unscheduled.csv")
    schedule = schedule.sort_values("TaskID")
    schedule.to_csv(schedule_dir / "schedule.csv", index=False, encoding="utf-8-sig")

    usage_rows = []
    for r, rj in r_index.items():
        for hour in range(HORIZON):
            usage_rows.append({"Hour": hour, "Region": r, "Scheduled_GPU_h": used_gpu[rj, hour], "GPU_Utilization_Percent": 100 * used_gpu[rj, hour] / available_gpu[rj], "AI_IT_Load_MW": used_it[rj, hour], "IT_Power_Margin_MW": it_margin[rj, hour] - used_it[rj, hour]})
    usage = pd.DataFrame(usage_rows)
    usage.to_csv(schedule_dir / "scheduled_hourly_usage.csv", index=False, encoding="utf-8-sig")

    migration = schedule.assign(Migrated=schedule["SourceRegion"] != schedule["ExecRegion"]).groupby(["TaskType", "SourceRegion"], observed=False).agg(Tasks=("TaskID", "count"), MigratedTasks=("Migrated", "sum"), AverageWait_h=("StartHour", lambda s: float(np.mean(s.to_numpy() - schedule.loc[s.index, "ArrivalHour"].to_numpy()))), GPU_h=("GPU_h", "sum")).reset_index()
    migration["MigrationRate"] = migration["MigratedTasks"] / migration["Tasks"]
    migration.to_csv(schedule_dir / "migration_statistics.csv", index=False, encoding="utf-8-sig")
    return schedule, usage, used_gpu, used_it, it_margin, available_gpu


def verify_constraints(schedule, workload, usage, gpu, latency_map, power_map, time_data, out_dir):
    merged = workload.merge(schedule, on="TaskID", suffixes=("_input", "_schedule"), how="left")
    checks = []

    def add(name, mask, detail):
        count = int(np.asarray(mask).sum())
        checks.append({"Constraint": name, "ViolationCount": count, "Status": "PASS" if count == 0 else "FAIL", "Detail": detail})

    add("C1_UniqueAssignment", merged["ExecRegion"].isna(), "每个任务恰好有一个执行区域")
    rt = merged["TaskType_input"].eq("RealTimeInference")
    add("C2_RealTimeImmediateStart", rt & (np.abs(merged["StartHour"] - merged["ArrivalHour_input"]) > EPS), "实时任务到达即开工")
    add("C3_Latency", merged["ActualLatency_ms"] > merged["MaxLatency_ms_input"] + EPS, "实际网络时延不超过 SLA")
    add("C4_Deadline", merged["FinishHour"] > merged["LatestFinishHour_input"] + EPS, "任务在最晚完成时点前完成")
    cap = gpu.set_index("Region")
    usage2 = usage.copy()
    usage2["GPUCap"] = usage2["Region"].map(cap["Available_GPU"])
    add("C5_GPUCapacity", usage2["Scheduled_GPU_h"] > usage2["GPUCap"] + EPS, "逐时折算 GPU-hour 不超过可用 GPU")
    add("C6_ITPower", usage2["IT_Power_Margin_MW"] < -EPS, "NonAI 与 AI IT 功率之和不超过 IT 上限")
    add("C7_NoHour2406", merged["FinishHour"] > 2406 + EPS, "所有任务在时点 2406 前完成")
    add("ArrivalBoundary", merged["StartHour"] < merged["ArrivalHour_input"] - EPS, "任务不得早于到达时刻开始")

    result = pd.DataFrame(checks)
    result.to_csv(out_dir / "schedule" / "constraint_verification.csv", index=False, encoding="utf-8-sig")
    if (result["ViolationCount"] > 0).any():
        raise AssertionError("调度约束验证失败，请检查 constraint_verification.csv")
    return result


def make_plots(type_summary, margins, predictions, schedule, usage, time_data, regions, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    shares = type_summary[["TaskCount_Share", "GPU_h_Share", "IT_MWh_Share"]].T * 100
    ax = shares.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Share (%)"); ax.set_xlabel(""); ax.legend(title="Task type"); ax.grid(axis="y", alpha=.25)
    plt.tight_layout(); plt.savefig(plot_dir / "任务类型占比图.png", dpi=180); plt.close()

    if predictions is not None:
        total = predictions[predictions["Model"].eq("GBDT")].groupby("Hour")[["Actual_GPU_h", "Predicted_GPU_h"]].sum()
        ax = total.plot(figsize=(10, 5), marker="o")
        ax.set_ylabel("GPU-hour"); ax.set_title("Forecast vs Actual (All regions and task types)", fontsize=15); ax.grid(alpha=.25)
        plt.tight_layout(); plt.savefig(plot_dir / "预测与实际对比图_2376至2399.png", dpi=180); plt.close()

    colors = {"RealTimeInference": "#4C78A8", "BatchInference": "#F58518", "AITraining": "#54A24B"}
    last = schedule[(schedule["ArrivalHour"] >= 2376) & (schedule["ArrivalHour"] <= 2399)].copy()
    fig, axes = plt.subplots(len(regions), 1, figsize=(14, 12), sharex=True)
    for ax, region in zip(axes, regions):
        part = last[last["ExecRegion"].eq(region)].sort_values(["StartHour", "TaskID"]).reset_index(drop=True)
        for j, row in part.iterrows():
            ax.barh(j, row["Duration_h"], left=row["StartHour"], height=.8, color=colors[row["TaskType"]])
        ax.set_ylabel(region); ax.set_yticks([]); ax.axvline(2400, color="black", ls="--", lw=.8); ax.grid(axis="x", alpha=.2)
    axes[-1].set_xlabel("Hour"); axes[-1].set_xlim(2376, 2406)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[t], label=t) for t in TASK_TYPES]
    fig.legend(handles=handles, loc="upper center", ncol=3); fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(plot_dir / "调度甘特图_2376至2406.png", dpi=180); plt.close(fig)

    baseline = time_data[["Hour", "Region", "GPU_Utilization_Percent"]]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for ax, region in zip(axes.flat, regions):
        ours = usage[(usage["Region"].eq(region)) & usage["Hour"].between(2376, 2405)]
        base = baseline[(baseline["Region"].eq(region)) & baseline["Hour"].between(2376, 2405)]
        ax.plot(ours["Hour"], ours["GPU_Utilization_Percent"], label="Scheduled")
        ax.plot(base["Hour"], base["GPU_Utilization_Percent"], label="Baseline", alpha=.75)
        ax.set_title(region, fontsize=14); ax.set_ylabel("GPU utilization (%)"); ax.grid(alpha=.2)
    axes.flat[0].legend(); fig.tight_layout(); fig.savefig(plot_dir / "图形处理器利用率图_2376至2405.png", dpi=180); plt.close(fig)


def write_summary(out_dir, workload, schedule, type_summary, verification):
    summary = {
        "task_count": int(len(workload)),
        "scheduled_task_count": int(len(schedule)),
        "total_GPU_h": float(workload["GPU_h"].sum()),
        "total_IT_MWh": float(workload["IT_MWh"].sum()),
        "migrated_task_count": int((schedule["SourceRegion"] != schedule["ExecRegion"]).sum()),
        "average_flexible_wait_h": float((schedule.loc[~schedule["TaskType"].eq("RealTimeInference"), "StartHour"] - schedule.loc[~schedule["TaskType"].eq("RealTimeInference"), "ArrivalHour"]).mean()),
        "all_constraints_passed": bool((verification["ViolationCount"] == 0).all()),
        "type_summary": type_summary.reset_index().to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("读取数据……")
    workload, gpu, time_data, power_map, latency_map, regions = load_data(args.data_dir)
    print("统计分析……")
    hourly, type_summary, margins = statistical_analysis(workload, gpu, time_data, power_map, regions, args.output_dir)
    predictions = None
    if not args.skip_forecast:
        print("训练并评价短期预测模型……")
        predictions = forecast(hourly, regions, args.output_dir)
    print("构造基础调度方案……")
    schedule, usage, *_ = schedule_tasks(workload, gpu, time_data, power_map, latency_map, regions, args.output_dir)
    print("验证 C1-C7……")
    verification = verify_constraints(schedule, workload, usage, gpu, latency_map, power_map, time_data, args.output_dir)
    if not args.skip_plots:
        print("绘制结果图……")
        make_plots(type_summary, margins, predictions, schedule, usage, time_data, regions, args.output_dir)
    write_summary(args.output_dir, workload, schedule, type_summary, verification)
    print(f"完成。结果目录：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
