#!/usr/bin/env python3
"""合并 CPU 与 GPU 版问题二输出，生成数据全面且正确的 output。

合并原则：
- 以 CPU 版 t2/output 的完整结构为骨架（含附件基线、问题一调度、方案 A/B、
  边际贡献、约束验证、能耗平衡、调度表、小时利用率）。
- 补齐 GPU 版独有的 Lambda 灵敏度数据（CPU 与 GPU 的 lambda 结果一致，
  以 GPU 版为准，避免重复计算）。
- 调度、能耗、约束验证等 CPU/GPU 完全一致的文件直接复用 CPU 版。
- summary.json 合并 CPU 的指标与 GPU 的后端/设备元信息。

运行：
    python3 t2/merge_output.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="合并 CPU 与 GPU 版问题二输出")
    parser.add_argument("--cpu-dir", type=Path, default=here / "output",
                        help="CPU 版结果目录，默认 t2/output")
    parser.add_argument("--gpu-dir", type=Path, default=here / "output_gpu",
                        help="GPU 版结果目录，默认 t2/output_gpu")
    parser.add_argument("--out-dir", type=Path, default=here / "output_merged",
                        help="合并结果目录，默认 t2/output_merged")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cpu, gpu, out = args.cpu_dir, args.gpu_dir, args.out_dir
    if not cpu.exists() or not gpu.exists():
        raise FileNotFoundError(f"缺少 CPU 或 GPU 输出目录: {cpu} / {gpu}")

    # 1. 复制 CPU 版完整结构（调度、能耗、约束验证、边际贡献、场景对比、图表）
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(cpu, out)

    # 2. 用 GPU 版 Lambda 灵敏度覆盖（CPU 与 GPU 结果一致，以 GPU 为准）
    gpu_lambda = gpu / "reports" / "lambda_sensitivity.csv"
    if gpu_lambda.exists():
        shutil.copy2(gpu_lambda, out / "reports" / "lambda_sensitivity.csv")

    # 3. 合并 summary.json：CPU 指标 + GPU 后端/设备元信息
    cpu_summary = json.loads((cpu / "summary.json").read_text(encoding="utf-8"))
    gpu_summary = json.loads((gpu / "summary.json").read_text(encoding="utf-8"))
    merged_summary = dict(cpu_summary)
    for key in ("backend", "gpu"):
        if key in gpu_summary:
            merged_summary[key] = gpu_summary[key]
    (out / "summary.json").write_text(
        json.dumps(merged_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 4. 校验合并结果
    required = [
        out / "reports" / "scenario_comparison.csv",
        out / "reports" / "lambda_sensitivity.csv",
        out / "reports" / "migration_marginal_contribution.csv",
        out / "schedule" / "constraint_verification.csv",
        out / "schedule" / "scheme_a_no_migration_schedule.csv",
        out / "schedule" / "scheme_b_carbon_aware_schedule.csv",
        out / "schedule" / "scheme_a_hourly_usage.csv",
        out / "schedule" / "scheme_b_hourly_usage.csv",
        out / "energy" / "scheme_a_energy_balance.csv",
        out / "energy" / "scheme_b_energy_balance.csv",
        out / "summary.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"合并结果缺少文件: {missing}")

    scenarios = [row["Scenario"] for row in _read_csv(out / "reports" / "scenario_comparison.csv")]
    expected = ["AttachmentBaseline", "Q1Schedule_Recalculated",
                "SchemeA_RenewableFirst_NoMigration", "SchemeB_CarbonAware"]
    if scenarios != expected:
        raise ValueError(f"scenario_comparison.csv 场景顺序/内容异常: {scenarios}")

    lambda_rows = _read_csv(out / "reports" / "lambda_sensitivity.csv")
    if not all(row["Scenario"].startswith("lambda=") for row in lambda_rows):
        raise ValueError("lambda_sensitivity.csv 应只包含 lambda= 场景")

    verify = _read_csv(out / "schedule" / "constraint_verification.csv")
    if any(int(row["ViolationCount"]) > 0 for row in verify):
        raise ValueError("约束验证存在违规")

    print(f"合并完成。结果目录：{out.resolve()}")
    print(f"场景对比：{scenarios}")
    print(f"Lambda 灵敏度：{[row['Scenario'] for row in lambda_rows]}")


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()
