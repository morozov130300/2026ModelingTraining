#!/usr/bin/env python3
"""问题三：给定负荷下的储能协同线性规划。

严格依据 t3_plan.md：不调整任务迁移或开工时段，仅使用附件给定的
Baseline_AI_IT_Load_MW 与 NonAI_IT_Load_MW，逐区域求解储能充放电、
购售电和新能源分配的 LP。求解器为 SciPy HiGHS（scipy.optimize.linprog）。

运行示例：
    python t3/t3.py --data-dir 题目 --output-dir t3/output --carbon-price 200
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

# 进程池按“场景×区域”并行，每个 HiGHS 进程固定单线程，避免线程过度订阅。
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"

import numpy as np
import pandas as pd

HORIZON = 2407
LAST_OPERATION_HOUR = 2405
EPS = 1e-7
VARIABLES = ("RenewableCharge_MW", "GridCharge_MW", "DischargePower_MW",
             "GridPurchase_MW", "GridSell_MW", "DirectRenewable_MW",
             "Curtailment_MW", "SOC_MWh")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="求解数学建模训练题问题三（储能协同 LP）")
    parser.add_argument("--data-dir", type=Path, default=here.parent / "题目")
    parser.add_argument("--output-dir", type=Path, default=here / "output")
    parser.add_argument("--carbon-price", type=float, default=200.0,
                        help="主方案碳价 lambda，单位元/tCO2")
    parser.add_argument("--lambda-values", type=float, nargs="*",
                        default=[0, 50, 100, 150, 200, 300, 500],
                        help="碳价灵敏度取值；主碳价会自动加入")
    parser.add_argument("--renewable-alphas", type=float, nargs="*",
                        default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help="可再生消纳上限 alpha 灵敏度取值")
    parser.add_argument("--capacity-factors", type=float, nargs="*",
                        default=[0.5, 1.0, 2.0], help="储能容量灵敏度系数")
    parser.add_argument("--peak-targets", type=float, nargs="*",
                        default=[0, 25, 50, 100, 150, 200, 250],
                        help="方案3逐区域净购电峰值上限扫描值（MW）")
    parser.add_argument("--utilization-targets", type=float, nargs="*",
                        default=[0.33, 0.35, 0.40, 0.45, 0.50],
                        help="方案4新能源利用率下限扫描值（0--1）")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="并行 LP 工作进程数；默认使用全部逻辑 CPU 核心")
    parser.add_argument("--skip-sensitivity", action="store_true", help="仅运行方案0--4，不运行敏感性场景")
    parser.add_argument("--skip-plots", action="store_true", help="跳过 PNG 制图")
    return parser.parse_args()


def require_columns(name: str, frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} 缺少字段: {missing}")


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    time_data = pd.read_excel(data_dir / "region_time_data.xlsx", sheet_name="region_time_data")
    storage = pd.read_excel(data_dir / "storage_information.xlsx", sheet_name="storage_information")
    gpu = pd.read_excel(data_dir / "GPU_information.xlsx", sheet_name="GPU中心基础情况")
    require_columns("region_time_data", time_data, [
        "Hour", "Region", "Baseline_AI_IT_Load_MW", "NonAI_IT_Load_MW",
        "AvailableRenewable_MW", "ElectricityPrice_CNY_per_MWh",
        "SellPrice_CNY_per_MWh", "CarbonIntensity_tCO2_per_MWh",
        "GridPurchase_MW", "GridSell_MW", "Curtailment_MW", "SOC_MWh",
        "RenewableCharge_MW", "GridCharge_MW", "DischargePower_MW",
    ])
    require_columns("storage_information", storage, [
        "Region", "StorageCapacity_MWh", "MinSOC_MWh", "InitialSOC_MWh",
        "MaxChargePower_MW", "MaxDischargePower_MW", "ChargeEfficiency",
        "DischargeEfficiency", "MaxGridImport_MW", "MaxGridExport_MW",
    ])
    require_columns("GPU_information", gpu, ["Region", "PUE"])
    regions = storage["Region"].astype(str).tolist()
    if len(regions) != 6 or len(set(regions)) != len(regions):
        raise ValueError("storage_information 必须恰有六个唯一区域")
    if set(regions) != set(gpu["Region"].astype(str)):
        raise ValueError("storage_information 与 GPU_information 的区域集合不一致")
    expected = pd.MultiIndex.from_product([range(HORIZON), regions], names=["Hour", "Region"])
    actual = pd.MultiIndex.from_frame(time_data[["Hour", "Region"]].assign(Region=lambda x: x["Region"].astype(str)))
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("region_time_data 必须完整覆盖六区域的 0--2406 小时且无重复")
    pue = gpu.assign(Region=gpu["Region"].astype(str)).set_index("Region").loc[regions, "PUE"].astype(float)
    if (pue <= 0).any():
        raise ValueError("PUE 必须为正")
    return time_data.copy(), storage.set_index("Region").loc[regions].copy(), regions


def region_array(time_data: pd.DataFrame, region: str, field: str) -> np.ndarray:
    values = (time_data.loc[time_data["Region"].astype(str).eq(region), ["Hour", field]]
              .set_index("Hour").reindex(range(HORIZON))[field])
    if values.isna().any():
        raise ValueError(f"{region} 的 {field} 缺失小时数据")
    return values.to_numpy(dtype=float)


def build_region_context(time_data: pd.DataFrame, storage: pd.DataFrame, region: str,
                         capacity_factor: float = 1.0,
                         renewable_alpha: float = 1.0) -> dict:
    if capacity_factor < 0 or renewable_alpha <= 0 or renewable_alpha > 1:
        raise ValueError("capacity_factor 必须非负，renewable_alpha 必须在 (0, 1] 内")
    row = storage.loc[region]
    capacity = float(row.StorageCapacity_MWh) * capacity_factor
    min_soc = float(row.MinSOC_MWh) * capacity_factor
    initial_soc = float(row.InitialSOC_MWh) * capacity_factor
    if not (0 <= min_soc <= initial_soc <= capacity + EPS):
        raise ValueError(f"{region} 的容量、最小 SOC、初始 SOC 参数不合法")
    it_load = (region_array(time_data, region, "Baseline_AI_IT_Load_MW") +
               region_array(time_data, region, "NonAI_IT_Load_MW"))
    pue = float(row.get("PUE", np.nan))
    # PUE 存在 GPU 表中；由 main 在 storage 表内合并后传入。
    if not np.isfinite(pue):
        raise ValueError(f"{region} 缺少 PUE")
    total_load = it_load * pue
    return {
        "region": region, "it_load": it_load, "total_load": total_load,
        "renewable": region_array(time_data, region, "AvailableRenewable_MW"),
        "price": region_array(time_data, region, "ElectricityPrice_CNY_per_MWh"),
        "sell_price": region_array(time_data, region, "SellPrice_CNY_per_MWh"),
        "carbon": region_array(time_data, region, "CarbonIntensity_tCO2_per_MWh"),
        "capacity": capacity, "min_soc": min_soc, "initial_soc": initial_soc,
        "max_charge": float(row.MaxChargePower_MW),
        "max_discharge": float(row.MaxDischargePower_MW),
        "eta_c": float(row.ChargeEfficiency), "eta_d": float(row.DischargeEfficiency),
        "max_import": float(row.MaxGridImport_MW), "max_export": float(row.MaxGridExport_MW),
        "renewable_alpha": renewable_alpha,
    }


def variable_slice(name: str) -> slice:
    index = VARIABLES.index(name)
    return slice(index * HORIZON, (index + 1) * HORIZON)


def solve_region_lp(ctx: dict, carbon_price: float, peak_target: float | None = None,
                    utilization_target: float | None = None) -> dict:
    """求解单区域 LP；GridPurchase 为含 GridCharge 的毛购电。"""
    try:
        from scipy.optimize import linprog
        from scipy.sparse import lil_matrix
    except ImportError as error:
        raise RuntimeError("问题三需要 scipy（scipy.optimize.linprog 的 HiGHS 求解器）") from error
    if carbon_price < 0:
        raise ValueError("carbon_price 不得为负")
    if peak_target is not None and peak_target < 0:
        raise ValueError("peak_target 不得为负")
    if utilization_target is not None and not 0 <= utilization_target <= 1:
        raise ValueError("utilization_target 必须在 [0, 1]")

    n = HORIZON
    size = len(VARIABLES) * n
    c = np.zeros(size)
    c[variable_slice("GridPurchase_MW")] = ctx["price"] + carbon_price * ctx["carbon"]
    c[variable_slice("GridSell_MW")] = -ctx["sell_price"]

    # 变量顺序 cR, cG, d, q, y, u, w, soc；功率单位 MW，步长为 1h。
    bounds: list[tuple[float, float | None]] = []
    for name in VARIABLES:
        if name == "RenewableCharge_MW":
            bounds.extend((0.0, 0.0 if t == HORIZON - 1 else None) for t in range(n))
        elif name == "GridCharge_MW":
            bounds.extend((0.0, 0.0 if t == HORIZON - 1 else None) for t in range(n))
        elif name == "DischargePower_MW":
            bounds.extend((0.0, 0.0 if t == HORIZON - 1 else None) for t in range(n))
        elif name == "GridPurchase_MW":
            bounds.extend((0.0, ctx["max_import"]) for _ in range(n))
        elif name == "GridSell_MW":
            bounds.extend((0.0, ctx["max_export"]) for _ in range(n))
        elif name == "DirectRenewable_MW":
            bounds.extend((0.0, float(value)) for value in ctx["total_load"])
        elif name == "Curtailment_MW":
            bounds.extend((0.0, None) for _ in range(n))
        else:
            bounds.extend((ctx["min_soc"], ctx["capacity"]) for _ in range(n))

    # 等式：能量平衡、可再生守恒、SOC 递推。使用稀疏矩阵，避免 6×2407 小时 LP 的稠密内存开销。
    a_eq = lil_matrix((3 * n, size), dtype=float)
    b_eq = np.zeros(3 * n)
    c_r, c_g, discharge = variable_slice("RenewableCharge_MW"), variable_slice("GridCharge_MW"), variable_slice("DischargePower_MW")
    purchase, sell, direct, curtail, soc = (variable_slice("GridPurchase_MW"), variable_slice("GridSell_MW"),
                                             variable_slice("DirectRenewable_MW"), variable_slice("Curtailment_MW"),
                                             variable_slice("SOC_MWh"))
    for t in range(n):
        # q + R + d = TL + cR + cG + y + w
        a_eq[t, purchase.start + t] = 1
        a_eq[t, discharge.start + t] = 1
        a_eq[t, c_r.start + t] = -1
        a_eq[t, c_g.start + t] = -1
        a_eq[t, sell.start + t] = -1
        a_eq[t, curtail.start + t] = -1
        b_eq[t] = ctx["total_load"][t] - ctx["renewable"][t]
        # R = u + cR + y + w
        row = n + t
        a_eq[row, direct.start + t] = 1
        a_eq[row, c_r.start + t] = 1
        a_eq[row, sell.start + t] = 1
        a_eq[row, curtail.start + t] = 1
        b_eq[row] = ctx["renewable"][t]
        # s(t) = s(t-1) + eta_c(cR+cG) - d/eta_d
        row = 2 * n + t
        a_eq[row, soc.start + t] = 1
        a_eq[row, c_r.start + t] = -ctx["eta_c"]
        a_eq[row, c_g.start + t] = -ctx["eta_c"]
        a_eq[row, discharge.start + t] = 1 / ctx["eta_d"]
        if t == 0:
            b_eq[row] = ctx["initial_soc"]
        else:
            a_eq[row, soc.start + t - 1] = -1

    rows_per_hour = 5 + int(peak_target is not None)
    extra_rows = 1 + int(utilization_target is not None)
    a_ub = lil_matrix((rows_per_hour * n + extra_rows, size), dtype=float)
    b_ub = np.empty(rows_per_hour * n + extra_rows, dtype=float)
    row_index = 0
    for t in range(n):
        # cR + cG <= P_ch；d <= P_dis。
        a_ub[row_index, c_r.start + t] = 1; a_ub[row_index, c_g.start + t] = 1
        b_ub[row_index] = ctx["max_charge"]; row_index += 1
        a_ub[row_index, discharge.start + t] = 1
        b_ub[row_index] = ctx["max_discharge"]; row_index += 1
        # 计划允许的线性充放互斥松弛：同一小时充电与放电总和不超过较大功率上限。
        a_ub[row_index, c_r.start + t] = 1; a_ub[row_index, c_g.start + t] = 1; a_ub[row_index, discharge.start + t] = 1
        b_ub[row_index] = max(ctx["max_charge"], ctx["max_discharge"]); row_index += 1
        # GridPurchase 为毛购电，故电网充电必须包含于毛购电。
        a_ub[row_index, c_g.start + t] = 1; a_ub[row_index, purchase.start + t] = -1
        b_ub[row_index] = 0.0; row_index += 1
        # 直接消纳 + 新能源充电 + 外送不超过 alpha×可用新能源。
        a_ub[row_index, direct.start + t] = 1; a_ub[row_index, c_r.start + t] = 1; a_ub[row_index, sell.start + t] = 1
        b_ub[row_index] = ctx["renewable_alpha"] * ctx["renewable"][t]; row_index += 1
        if peak_target is not None:
            a_ub[row_index, purchase.start + t] = 1; a_ub[row_index, sell.start + t] = -1
            b_ub[row_index] = peak_target; row_index += 1
    # 终端库存不低于初始库存，2406 的充放电边界已由 bounds 固化。
    a_ub[row_index, soc.stop - 1] = -1
    b_ub[row_index] = -ctx["initial_soc"]; row_index += 1
    if utilization_target is not None:
        a_ub[row_index, direct] = -1; a_ub[row_index, c_r] = -1; a_ub[row_index, sell] = -1
        b_ub[row_index] = -utilization_target * float(ctx["renewable"].sum())

    result = linprog(c, A_ub=a_ub.tocsr(), b_ub=b_ub,
                     A_eq=a_eq.tocsr(), b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        return {"success": False, "message": result.message, "region": ctx["region"]}
    values = {name: result.x[variable_slice(name)] for name in VARIABLES}
    return {"success": True, "message": result.message, "region": ctx["region"],
            "objective": float(result.fun), "ctx": ctx, **values}


def solve_case(time_data: pd.DataFrame, storage: pd.DataFrame, regions: list[str],
               carbon_price: float, capacity_factor: float = 1.0,
               renewable_alpha: float = 1.0, peak_target: float | None = None,
               utilization_target: float | None = None) -> tuple[dict, pd.DataFrame]:
    """串行兼容接口；批量全量运行由 solve_case_batch 调度。"""
    results, frames = {}, []
    for region in regions:
        ctx = build_region_context(time_data, storage, region, capacity_factor, renewable_alpha)
        result = solve_region_lp(ctx, carbon_price, peak_target, utilization_target)
        if not result["success"]:
            return {"success": False, "region": region, "message": result["message"]}, pd.DataFrame()
        results[region] = result
        frames.append(result_frame(result))
    return {"success": True, "results": results}, pd.concat(frames, ignore_index=True)


def solve_region_job(label: str, region: str, ctx: dict, carbon_price: float,
                     peak_target: float | None, utilization_target: float | None) -> tuple[str, str, dict]:
    """进程池工作单元：每项为一个完全独立的区域 LP。"""
    return label, region, solve_region_lp(ctx, carbon_price, peak_target, utilization_target)


def solve_case_batch(case_specs: list[dict], storage: pd.DataFrame, regions: list[str],
                     workers: int) -> dict[str, tuple[dict, pd.DataFrame]]:
    """一次提交全部场景×区域 LP，使全量扫描能够占满 CPU 而不改变任何模型精度。"""
    jobs = []
    for spec in case_specs:
        for region in regions:
            ctx = build_region_context(spec["time_data"], storage, region,
                                       spec.get("capacity_factor", 1.0),
                                       spec.get("renewable_alpha", 1.0))
            jobs.append((spec["name"], region, ctx, spec["carbon_price"],
                         spec.get("peak_target"), spec.get("utilization_target")))
    if not jobs:
        return {}
    results: dict[str, dict[str, dict]] = {spec["name"]: {} for spec in case_specs}
    max_workers = min(workers, len(jobs))
    print(f"并行提交 {len(jobs)} 个独立区域 LP，使用 {max_workers} 个工作进程……", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(solve_region_job, *job): job[:2] for job in jobs}
        completed = 0
        for future in as_completed(futures):
            label, region = futures[future]
            try:
                _, _, result = future.result()
            except Exception as error:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f"{label}/{region} 的 LP 工作进程异常") from error
            results[label][region] = result
            completed += 1
            print(f"LP 进度: {completed}/{len(jobs)}", flush=True)
    assembled: dict[str, tuple[dict, pd.DataFrame]] = {}
    for spec in case_specs:
        name = spec["name"]
        regional = results[name]
        failed = next((result for result in regional.values() if not result["success"]), None)
        if failed is not None:
            assembled[name] = ({"success": False, "region": failed["region"], "message": failed["message"]}, pd.DataFrame())
        else:
            frames = [result_frame(regional[region]) for region in regions]
            assembled[name] = ({"success": True, "results": regional}, pd.concat(frames, ignore_index=True))
    return assembled


def result_frame(result: dict) -> pd.DataFrame:
    ctx = result["ctx"]
    frame = pd.DataFrame({"Hour": np.arange(HORIZON), "Region": ctx["region"],
                          "IT_Load_MW": ctx["it_load"], "Total_Load_MW": ctx["total_load"],
                          "AvailableRenewable_MW": ctx["renewable"],
                          "ElectricityPrice_CNY_per_MWh": ctx["price"],
                          "SellPrice_CNY_per_MWh": ctx["sell_price"],
                          "CarbonIntensity_tCO2_per_MWh": ctx["carbon"]})
    for name in VARIABLES:
        frame[name] = result[name]
    frame["ChargePower_MW"] = frame["RenewableCharge_MW"] + frame["GridCharge_MW"]
    frame["NetGridImport_MW"] = frame["GridPurchase_MW"] - frame["GridSell_MW"]
    frame["Cost_CNY"] = (frame["GridPurchase_MW"] * frame["ElectricityPrice_CNY_per_MWh"] -
                           frame["GridSell_MW"] * frame["SellPrice_CNY_per_MWh"])
    frame["CarbonEmission_tCO2"] = frame["GridPurchase_MW"] * frame["CarbonIntensity_tCO2_per_MWh"]
    return frame


def no_storage_renewable_first(time_data: pd.DataFrame, storage: pd.DataFrame,
                                regions: list[str], renewable_alpha: float = 1.0) -> pd.DataFrame:
    rows = []
    for region in regions:
        ctx = build_region_context(time_data, storage, region, capacity_factor=1.0,
                                   renewable_alpha=renewable_alpha)
        usable = ctx["renewable"] * renewable_alpha
        direct = np.minimum(ctx["total_load"], usable)
        purchase = np.maximum(0.0, ctx["total_load"] - direct)
        surplus = usable - direct
        sell = np.minimum(ctx["max_export"], surplus)
        curtail = ctx["renewable"] - direct - sell
        rows.append(pd.DataFrame({"Hour": np.arange(HORIZON), "Region": region,
            "IT_Load_MW": ctx["it_load"], "Total_Load_MW": ctx["total_load"],
            "AvailableRenewable_MW": ctx["renewable"],
            "ElectricityPrice_CNY_per_MWh": ctx["price"], "SellPrice_CNY_per_MWh": ctx["sell_price"],
            "CarbonIntensity_tCO2_per_MWh": ctx["carbon"], "DirectRenewable_MW": direct,
            "RenewableCharge_MW": 0.0, "GridCharge_MW": 0.0, "ChargePower_MW": 0.0,
            "DischargePower_MW": 0.0, "GridPurchase_MW": purchase, "GridSell_MW": sell,
            "Curtailment_MW": curtail, "SOC_MWh": ctx["initial_soc"],
            "NetGridImport_MW": purchase - sell,
            "Cost_CNY": purchase * ctx["price"] - sell * ctx["sell_price"],
            "CarbonEmission_tCO2": purchase * ctx["carbon"]}))
    return pd.concat(rows, ignore_index=True)


def attachment_baseline(time_data: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    fields = ["Hour", "Region", "Baseline_AI_IT_Load_MW", "NonAI_IT_Load_MW", "AvailableRenewable_MW",
              "ElectricityPrice_CNY_per_MWh", "SellPrice_CNY_per_MWh", "CarbonIntensity_tCO2_per_MWh",
              "GridPurchase_MW", "GridSell_MW", "Curtailment_MW", "RenewableCharge_MW", "GridCharge_MW",
              "ChargePower_MW", "DischargePower_MW", "SOC_MWh", "NetGridImport_MW", "CarbonEmission_tCO2"]
    frame = time_data[fields].copy()
    frame["IT_Load_MW"] = frame["Baseline_AI_IT_Load_MW"] + frame["NonAI_IT_Load_MW"]
    frame["DirectRenewable_MW"] = (frame["AvailableRenewable_MW"] - frame["RenewableCharge_MW"] -
                                    frame["GridSell_MW"] - frame["Curtailment_MW"])
    frame["Cost_CNY"] = (frame["GridPurchase_MW"] * frame["ElectricityPrice_CNY_per_MWh"] -
                           frame["GridSell_MW"] * frame["SellPrice_CNY_per_MWh"])
    return frame


def metrics(name: str, frame: pd.DataFrame, carbon_price: float) -> dict:
    renewable = float(frame["AvailableRenewable_MW"].sum())
    utilized = float((frame["DirectRenewable_MW"] + frame["RenewableCharge_MW"] + frame["GridSell_MW"]).sum())
    region_peak = frame.groupby("Region", observed=False)["NetGridImport_MW"].max()
    region_std = frame.groupby("Region", observed=False)["NetGridImport_MW"].std(ddof=0)
    return {
        "Scenario": name, "OperatingCost_CNY": float(frame["Cost_CNY"].sum()),
        "CarbonEmission_tCO2": float(frame["CarbonEmission_tCO2"].sum()),
        "Objective_CNY": float(frame["Cost_CNY"].sum() + carbon_price * frame["CarbonEmission_tCO2"].sum()),
        "GridPurchase_MWh": float(frame["GridPurchase_MW"].sum()),
        "GridSell_MWh": float(frame["GridSell_MW"].sum()),
        "Curtailment_MWh": float(frame["Curtailment_MW"].sum()),
        "DirectRenewable_MWh": float(frame["DirectRenewable_MW"].sum()),
        "RenewableCharge_MWh": float(frame["RenewableCharge_MW"].sum()),
        "GridCharge_MWh": float(frame["GridCharge_MW"].sum()),
        "Discharge_MWh": float(frame["DischargePower_MW"].sum()),
        "AvailableRenewable_MWh": renewable,
        "RenewableUtilization": utilized / renewable if renewable > EPS else 0.0,
        "PeakNetGridImport_MW": float(region_peak.max()),
        "MeanRegionalNetGridStd_MW": float(region_std.mean()),
        "MaxRegionalNetGridStd_MW": float(region_std.max()),
    }


def validate_solution(name: str, frame: pd.DataFrame, storage: pd.DataFrame,
                      regions: list[str], renewable_alpha: float = 1.0) -> pd.DataFrame:
    checks = []
    def add(constraint: str, value: float, detail: str) -> None:
        checks.append({"Scenario": name, "Constraint": constraint,
                       "ViolationCount": int(value > EPS), "MaxViolation": max(0.0, float(value)),
                       "Status": "PASS" if value <= EPS else "FAIL", "Detail": detail})
    balance = (frame["GridPurchase_MW"] + frame["AvailableRenewable_MW"] + frame["DischargePower_MW"] -
               frame["Total_Load_MW"] - frame["ChargePower_MW"] - frame["GridSell_MW"] - frame["Curtailment_MW"])
    renewable = (frame["AvailableRenewable_MW"] - frame["DirectRenewable_MW"] - frame["RenewableCharge_MW"] -
                 frame["GridSell_MW"] - frame["Curtailment_MW"])
    add("C1_EnergyBalance", float(np.abs(balance).max()), "附件统一能源平衡")
    add("C2_RenewableConservation", float(np.abs(renewable).max()), "可再生守恒")
    nonnegative_columns = ["GridPurchase_MW", "GridSell_MW", "RenewableCharge_MW",
                           "GridCharge_MW", "DischargePower_MW", "Curtailment_MW"]
    nonnegative_violation = float(np.maximum(-frame[nonnegative_columns].to_numpy(float), 0.0).max())
    add("C3_Nonnegative", nonnegative_violation, "功率变量非负")
    for region in regions:
        part = frame.loc[frame["Region"].astype(str).eq(region)].sort_values("Hour")
        param = storage.loc[region]
        charge = part["ChargePower_MW"].to_numpy(float)
        discharge = part["DischargePower_MW"].to_numpy(float)
        soc = part["SOC_MWh"].to_numpy(float)
        eta_c, eta_d = float(param.ChargeEfficiency), float(param.DischargeEfficiency)
        expected = np.empty(HORIZON); expected[0] = float(param.InitialSOC_MWh) + eta_c * charge[0] - discharge[0] / eta_d
        expected[1:] = soc[:-1] + eta_c * charge[1:] - discharge[1:] / eta_d
        add(f"C4_SOCRecurrence_{region}", float(np.abs(soc - expected).max()), "SOC 时段末递推")
        add(f"C5_SOCBounds_{region}", float(max(float(param.MinSOC_MWh) - soc.min(), float(soc.max() - param.StorageCapacity_MWh), 0.0)), "SOC 上下限")
        add(f"C6_ChargeLimit_{region}", float(max(charge.max() - float(param.MaxChargePower_MW), 0.0)), "充电功率上限")
        add(f"C7_DischargeLimit_{region}", float(max(discharge.max() - float(param.MaxDischargePower_MW), 0.0)), "放电功率上限")
        add(f"C8_GridImport_{region}", float(max(part["GridPurchase_MW"].max() - float(param.MaxGridImport_MW), 0.0)), "毛购电上限")
        add(f"C9_GridExport_{region}", float(max(part["GridSell_MW"].max() - float(param.MaxGridExport_MW), 0.0)), "外送上限")
        add(f"C10_TerminalSOC_{region}", float(max(float(param.InitialSOC_MWh) - soc[-1], 0.0)), "SOC(2406) 不低于初始")
        terminal = part.loc[part["Hour"].eq(HORIZON - 1), ["RenewableCharge_MW", "GridCharge_MW", "DischargePower_MW"]].to_numpy(float)
        add(f"C11_Hour2406Idle_{region}", float(np.abs(terminal).max()), "2406 仅结算，不充放电")
        utilized = part["DirectRenewable_MW"] + part["RenewableCharge_MW"] + part["GridSell_MW"]
        add(f"C12_RenewableAlpha_{region}", float(np.maximum(utilized.to_numpy(float) - renewable_alpha * part["AvailableRenewable_MW"].to_numpy(float), 0.0).max()), "可再生消纳上限")
    return pd.DataFrame(checks)


def storage_state_table(frame: pd.DataFrame, storage: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    rows = []
    for region in regions:
        part = frame.loc[frame["Region"].astype(str).eq(region)]
        param = storage.loc[region]
        charged = float(part["ChargePower_MW"].sum())
        discharged = float(part["DischargePower_MW"].sum())
        rows.append({"Region": region, "InitialSOC_MWh": float(param.InitialSOC_MWh),
                     "FinalSOC_MWh": float(part.loc[part["Hour"].eq(HORIZON - 1), "SOC_MWh"].iloc[0]),
                     "MinSOC_MWh": float(part["SOC_MWh"].min()), "MaxSOC_MWh": float(part["SOC_MWh"].max()),
                     "TotalCharge_MWh": charged, "TotalDischarge_MWh": discharged,
                     "EquivalentCycles": discharged / (2 * float(param.StorageCapacity_MWh)) if float(param.StorageCapacity_MWh) > EPS else 0.0})
    return pd.DataFrame(rows)


def run_frontier(time_data: pd.DataFrame, storage: pd.DataFrame, regions: list[str], carbon_price: float,
                 targets: list[float], kind: str) -> tuple[pd.DataFrame, pd.DataFrame | None, float | None]:
    records, selected, selected_target = [], None, None
    ordered = sorted(set(targets), reverse=(kind == "utilization"))
    for target in ordered:
        kwargs = {"peak_target": target} if kind == "peak" else {"utilization_target": target}
        result, frame = solve_case(time_data, storage, regions, carbon_price, **kwargs)
        record = {"Target": target, "Feasible": bool(result["success"]), "Message": result.get("message", "")}
        if result["success"]:
            record.update(metrics(f"{kind}={target:g}", frame, carbon_price))
            if selected is None:
                selected, selected_target = frame, target
        records.append(record)
    return pd.DataFrame(records).sort_values("Target"), selected, selected_target


def run_sensitivity(time_data: pd.DataFrame, storage: pd.DataFrame, regions: list[str], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    scenarios = []
    scenarios.extend(("carbon_price", value, 1.0, 1.0) for value in sorted(set(args.lambda_values + [args.carbon_price])))
    scenarios.extend(("renewable_alpha", args.carbon_price, 1.0, value) for value in sorted(set(args.renewable_alphas)))
    scenarios.extend(("capacity_factor", args.carbon_price, value, 1.0) for value in sorted(set(args.capacity_factors)))
    for dimension, price, capacity, alpha in scenarios:
        result, frame = solve_case(time_data, storage, regions, price, capacity, alpha)
        record = {"Dimension": dimension, "CarbonPrice_CNY_per_tCO2": price,
                  "CapacityFactor": capacity, "RenewableAlpha": alpha,
                  "Feasible": bool(result["success"]), "Message": result.get("message", "")}
        if result["success"]:
            record.update(metrics(f"{dimension}", frame, price))
        rows.append(record)
    # 计划的极端可再生下降 20% 场景：仅改变可再生输入，不改变给定 IT 负荷。
    stressed = time_data.copy()
    stressed["AvailableRenewable_MW"] *= 0.8
    result, frame = solve_case(stressed, storage, regions, args.carbon_price)
    record = {"Dimension": "renewable_minus_20_percent", "CarbonPrice_CNY_per_tCO2": args.carbon_price,
              "CapacityFactor": 1.0, "RenewableAlpha": 1.0, "Feasible": bool(result["success"]),
              "Message": result.get("message", "")}
    if result["success"]:
        record.update(metrics("renewable_minus_20_percent", frame, args.carbon_price))
    rows.append(record)
    return pd.DataFrame(rows)


def make_plots(out_dir: Path, baseline: pd.DataFrame, scheme1: pd.DataFrame, scheme2: pd.DataFrame,
               scheme3: pd.DataFrame | None, scheme4: pd.DataFrame | None, peak_frontier: pd.DataFrame,
               util_frontier: pd.DataFrame, comparison: pd.DataFrame, sensitivity: pd.DataFrame | None,
               regions: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = out_dir / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False
    # 代表 96 小时窗口：使用 F 区净购电最高的时段附近。
    f_peak = scheme2.loc[scheme2["Region"].eq("RegionF"), ["Hour", "NetGridImport_MW"]]
    anchor = int(f_peak.loc[f_peak["NetGridImport_MW"].idxmax(), "Hour"])
    start = max(0, min(HORIZON - 96, anchor - 48))
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for ax, region in zip(axes.flat, regions):
        part = scheme2[(scheme2["Region"] == region) & scheme2["Hour"].between(start, start + 95)]
        base = baseline[(baseline["Region"] == region) & baseline["Hour"].between(start, start + 95)]
        ax.plot(part["Hour"], part["SOC_MWh"], label="SOC")
        ax.plot(part["Hour"], part["NetGridImport_MW"], label="Optimized net import")
        ax.plot(base["Hour"], base["NetGridImport_MW"], label="Baseline net import", alpha=.65)
        ax.set_title(region); ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(plot_dir / "储能时序图_96小时.png", dpi=180); plt.close(fig)
    names = ["AttachmentBaseline", "Scheme1_NoStorage_RenewableFirst", "Scheme2_StorageLP"]
    frames = [baseline, scheme1, scheme2]
    fig, ax = plt.subplots(figsize=(11, 5)); bottom = np.zeros(len(names))
    for column, label in [("DirectRenewable_MW", "Direct renewable"), ("RenewableCharge_MW", "Renewable charge"),
                          ("GridCharge_MW", "Grid charge"), ("GridSell_MW", "Grid sell"),
                          ("Curtailment_MW", "Curtailment"), ("GridPurchase_MW", "Grid purchase")]:
        values = np.asarray([frame[column].sum() for frame in frames])
        ax.bar(names, values, bottom=bottom, label=label); bottom += values
    ax.tick_params(axis="x", rotation=15); ax.legend(ncol=3, fontsize=8); ax.set_ylabel("MWh"); fig.tight_layout(); fig.savefig(plot_dir / "能源分配对比图.png", dpi=180); plt.close(fig)
    for frontier, filename, x, label in [(peak_frontier, "成本削峰前沿图.png", "PeakNetGridImport_MW", "Peak net import (MW)"),
                                         (util_frontier, "成本利用率前沿图.png", "RenewableUtilization", "Renewable utilization")]:
        feasible = frontier[frontier["Feasible"]]
        if not feasible.empty:
            fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(feasible[x], feasible["OperatingCost_CNY"], marker="o")
            ax.set_xlabel(label); ax.set_ylabel("Operating cost (CNY)"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(plot_dir / filename, dpi=180); plt.close(fig)
    peak_data = pd.DataFrame({name: frame.groupby("Region")["NetGridImport_MW"].max() for name, frame in zip(names, frames)})
    peak_data.plot(kind="bar", figsize=(10, 5)); plt.ylabel("Peak net import (MW)"); plt.tight_layout(); plt.savefig(plot_dir / "区域峰值净购电图.png", dpi=180); plt.close()
    if sensitivity is not None:
        feasible = sensitivity[sensitivity["Feasible"]]
        fig, ax = plt.subplots(figsize=(9, 5))
        for dimension, group in feasible.groupby("Dimension"):
            x = group["CarbonPrice_CNY_per_tCO2"] if dimension == "carbon_price" else (group["CapacityFactor"] if dimension == "capacity_factor" else group["RenewableAlpha"])
            ax.plot(x, group["OperatingCost_CNY"], marker="o", label=dimension)
        ax.set_ylabel("Operating cost (CNY)"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(plot_dir / "敏感性成本图.png", dpi=180); plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.carbon_price < 0:
        raise ValueError("carbon-price 不得为负")
    if any(value < 0 for value in args.lambda_values):
        raise ValueError("lambda-values 不得为负")
    if args.workers < 1:
        raise ValueError("workers 至少为 1")
    if any(value <= 0 or value > 1 for value in args.renewable_alphas):
        raise ValueError("renewable-alphas 必须在 (0, 1]")
    if any(value < 0 for value in args.capacity_factors):
        raise ValueError("capacity-factors 不得为负")
    if any(value < 0 for value in args.peak_targets):
        raise ValueError("peak-targets 不得为负")
    if any(value < 0 or value > 1 for value in args.utilization_targets):
        raise ValueError("utilization-targets 必须在 [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("读取问题三原始负荷、储能和 PUE 数据……")
    time_data, storage, regions = load_data(args.data_dir)
    pue = pd.read_excel(args.data_dir / "GPU_information.xlsx", sheet_name="GPU中心基础情况").assign(Region=lambda x: x["Region"].astype(str)).set_index("Region")["PUE"]
    storage["PUE"] = pue.loc[regions].astype(float)
    print("构造方案0（附件基准）和方案1（无储能、可再生优先）……")
    baseline = attachment_baseline(time_data, regions)
    scheme1 = no_storage_renewable_first(time_data, storage, regions)
    print("构造全部精确 LP 场景，并按 场景×区域 并行求解……")
    case_specs = [{"name": "scheme2", "time_data": time_data, "carbon_price": args.carbon_price}]
    case_specs.extend({"name": f"peak={target:g}", "time_data": time_data, "carbon_price": args.carbon_price,
                       "peak_target": target} for target in sorted(set(args.peak_targets)))
    case_specs.extend({"name": f"utilization={target:g}", "time_data": time_data, "carbon_price": args.carbon_price,
                       "utilization_target": target} for target in sorted(set(args.utilization_targets), reverse=True))
    sensitivity_specs = []
    if not args.skip_sensitivity:
        sensitivity_specs.extend({"name": f"lambda={value:g}", "time_data": time_data, "carbon_price": value,
                                  "dimension": "carbon_price", "capacity_factor": 1.0, "renewable_alpha": 1.0}
                                 for value in sorted(set(args.lambda_values + [args.carbon_price])))
        sensitivity_specs.extend({"name": f"alpha={value:g}", "time_data": time_data, "carbon_price": args.carbon_price,
                                  "dimension": "renewable_alpha", "capacity_factor": 1.0, "renewable_alpha": value}
                                 for value in sorted(set(args.renewable_alphas)))
        sensitivity_specs.extend({"name": f"capacity={value:g}", "time_data": time_data, "carbon_price": args.carbon_price,
                                  "dimension": "capacity_factor", "capacity_factor": value, "renewable_alpha": 1.0}
                                 for value in sorted(set(args.capacity_factors)))
        stressed = time_data.copy()
        stressed["AvailableRenewable_MW"] *= 0.8
        sensitivity_specs.append({"name": "renewable_minus_20_percent", "time_data": stressed,
                                  "carbon_price": args.carbon_price, "dimension": "renewable_minus_20_percent",
                                  "capacity_factor": 1.0, "renewable_alpha": 1.0})
        case_specs.extend(sensitivity_specs)
    batch_results = solve_case_batch(case_specs, storage, regions, args.workers)
    solved, scheme2 = batch_results["scheme2"]
    if not solved["success"]:
        raise RuntimeError(f"方案2 求解失败：{solved['region']}：{solved['message']}")
    peak_records, scheme3, peak_target = [], None, None
    for target in sorted(set(args.peak_targets)):
        result, frame = batch_results[f"peak={target:g}"]
        record = {"Target": target, "Feasible": bool(result["success"]), "Message": result.get("message", "")}
        if result["success"]:
            record.update(metrics(f"peak={target:g}", frame, args.carbon_price))
            if scheme3 is None:
                scheme3, peak_target = frame, target
        peak_records.append(record)
    peak_frontier = pd.DataFrame(peak_records)
    util_records, scheme4, util_target = [], None, None
    for target in sorted(set(args.utilization_targets), reverse=True):
        result, frame = batch_results[f"utilization={target:g}"]
        record = {"Target": target, "Feasible": bool(result["success"]), "Message": result.get("message", "")}
        if result["success"]:
            record.update(metrics(f"utilization={target:g}", frame, args.carbon_price))
            if scheme4 is None:
                scheme4, util_target = frame, target
        util_records.append(record)
    util_frontier = pd.DataFrame(util_records).sort_values("Target")
    if scheme3 is None:
        scheme3, peak_target = scheme2.copy(), None
    if scheme4 is None:
        scheme4, util_target = scheme2.copy(), None

    energy_dir, report_dir = args.output_dir / "energy", args.output_dir / "reports"
    for directory in (energy_dir, report_dir): directory.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(energy_dir / "scheme0_attachment_baseline_energy_balance.csv", index=False, encoding="utf-8-sig")
    scheme1.to_csv(energy_dir / "scheme1_no_storage_renewable_first_energy_balance.csv", index=False, encoding="utf-8-sig")
    scheme2.to_csv(energy_dir / "scheme2_storage_lp_energy_balance.csv", index=False, encoding="utf-8-sig")
    scheme3.to_csv(energy_dir / "scheme3_peak_oriented_energy_balance.csv", index=False, encoding="utf-8-sig")
    scheme4.to_csv(energy_dir / "scheme4_utilization_oriented_energy_balance.csv", index=False, encoding="utf-8-sig")
    verification = validate_solution("Scheme2_StorageLP", scheme2, storage, regions)
    verification.to_csv(report_dir / "constraint_verification.csv", index=False, encoding="utf-8-sig")
    if (verification["ViolationCount"] > 0).any():
        raise AssertionError("方案2 约束验证失败，请检查 reports/constraint_verification.csv")
    pd.concat([storage_state_table(scheme2, storage, regions).assign(Scenario="Scheme2_StorageLP"),
               storage_state_table(scheme3, storage, regions).assign(Scenario="Scheme3_PeakOriented"),
               storage_state_table(scheme4, storage, regions).assign(Scenario="Scheme4_UtilizationOriented")]).to_csv(report_dir / "storage_state_and_cycles.csv", index=False, encoding="utf-8-sig")
    comparison = pd.DataFrame([
        metrics("AttachmentBaseline", baseline, args.carbon_price),
        metrics("Scheme1_NoStorage_RenewableFirst", scheme1, args.carbon_price),
        metrics("Scheme2_StorageLP", scheme2, args.carbon_price),
        metrics("Scheme3_PeakOriented", scheme3, args.carbon_price),
        metrics("Scheme4_UtilizationOriented", scheme4, args.carbon_price),
    ])
    comparison.to_csv(report_dir / "scenario_comparison.csv", index=False, encoding="utf-8-sig")
    attribution = comparison.iloc[1].copy()
    for column in ["OperatingCost_CNY", "CarbonEmission_tCO2", "Objective_CNY", "GridPurchase_MWh", "Curtailment_MWh"]:
        attribution[column] = comparison.iloc[1][column] - comparison.iloc[2][column]
    attribution["Scenario"] = "StorageMarginalBenefit_Scheme1_to_Scheme2"
    pd.DataFrame([attribution]).to_csv(report_dir / "storage_marginal_contribution.csv", index=False, encoding="utf-8-sig")
    peak_frontier.to_csv(report_dir / "cost_peak_frontier.csv", index=False, encoding="utf-8-sig")
    util_frontier.to_csv(report_dir / "cost_utilization_frontier.csv", index=False, encoding="utf-8-sig")
    sensitivity = None
    if not args.skip_sensitivity:
        sensitivity_rows = []
        for spec in sensitivity_specs:
            result, frame = batch_results[spec["name"]]
            record = {"Dimension": spec["dimension"], "CarbonPrice_CNY_per_tCO2": spec["carbon_price"],
                      "CapacityFactor": spec["capacity_factor"], "RenewableAlpha": spec["renewable_alpha"],
                      "Feasible": bool(result["success"]), "Message": result.get("message", "")}
            if result["success"]:
                record.update(metrics(spec["dimension"], frame, spec["carbon_price"]))
            sensitivity_rows.append(record)
        sensitivity = pd.DataFrame(sensitivity_rows)
        sensitivity.to_csv(report_dir / "sensitivity_analysis.csv", index=False, encoding="utf-8-sig")
    if not args.skip_plots:
        make_plots(args.output_dir, baseline, scheme1, scheme2, scheme3, scheme4, peak_frontier, util_frontier, comparison, sensitivity, regions)
    summary = {"carbon_price_CNY_per_tCO2": args.carbon_price, "horizon_hours": HORIZON,
               "given_load": "Baseline_AI_IT_Load_MW + NonAI_IT_Load_MW", "task_schedule_reoptimized": False,
               "solver": "scipy.optimize.linprog(method=highs)", "all_constraints_passed": True,
               "selected_peak_target_MW": peak_target, "selected_utilization_target": util_target,
               "scheme1_to_scheme2_storage_marginal_cost_reduction_CNY": float(comparison.iloc[1]["OperatingCost_CNY"] - comparison.iloc[2]["OperatingCost_CNY"]),
               "scheme1_to_scheme2_storage_marginal_carbon_reduction_tCO2": float(comparison.iloc[1]["CarbonEmission_tCO2"] - comparison.iloc[2]["CarbonEmission_tCO2"]),
               "scenarios": comparison.to_dict(orient="records")}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成。结果目录：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
