# AGENTS.md

## Git 推送约定

用户明确要求“推送”时，直接将当前工作区全部变更 `git add -A`、提交并推送到当前分支；不要先询问是否包含生成物，不要进行冗长说明或多余检查。除非用户明确指定范围，否则代码、文档和生成结果全部推送。


- `题目/`：只读的原始题目与 Excel 数据。问题一核心程序读取其中 5 个工作簿；`storage_information.xlsx` 和题目 `.docx` 当前不进入问题一程序。
- `t1/`：问题一的代码、建模说明、分析报告及生成结果。
- `t1/output/`：由问题一脚本生成的 CSV、JSON 和 PNG。除非任务明确要求更新结果，不要手工修改这些文件。
- `t2/`：问题二建模方案；目前只有 `t2_plan.md`，尚无可执行代码或结果目录。

仓库当前没有依赖清单、自动化测试、lint 配置或构建系统。不要在本地工作区创建虚拟环境或安装 Python 库；用户会在其他环境中运行 `.py` 文件。这里只记录运行所需的依赖，供外部运行环境准备：`numpy`、`pandas`、`openpyxl`、`scikit-learn`、`matplotlib`。

## 环境与常用命令

要求 Python 3.9+。执行 Python 脚本前，请在用户指定的外部环境中准备上述依赖；不要在当前项目环境中执行 `pip install`、创建 `.venv` 或修改本地依赖配置。

从仓库根目录运行完整问题一流程：

```bash
python t1/t1.py
```

显式指定输入和输出目录：

```bash
python t1/t1.py --data-dir 题目 --output-dir t1/output
```

较快地验证数据读取、统计、调度及约束检查，同时跳过模型训练和绘图：

```bash
python t1/t1.py --data-dir 题目 --output-dir t1/output --skip-forecast --skip-plots
```

基于 `t1/output/statistics/` 中已有统计 CSV 重新生成报告使用的区域结构图：

```bash
python t1/make_charts.py
```

只做语法检查：

```bash
python -m py_compile t1/t1.py t1/make_charts.py
```

当前没有测试套件，因此没有“单个测试”命令。修改纯函数时，可从仓库根目录用内联 Python 做聚焦检查，例如：

```bash
python - <<'PY'
from t1.t1 import overlap_arrays
hours, overlaps = overlap_arrays(3, 1.5)
assert hours.tolist() == [3, 4]
assert overlaps.tolist() == [1.0, 0.5]
PY
```

## 核心架构与数据流

`t1/t1.py` 是主流水线，调用顺序固定：

1. `load_data` 读取并校验输入表字段，构造 `Duration_h`、`GPU_h`、`IT_MWh`，并校验设施功率关系。
2. `statistical_analysis` 将任务聚合成区域×类型统计、18 条小时级 GPU-hour 序列和逐时 IT 功率余量。
3. `forecast` 对 6 区域×3 类型分别训练 `HistGradientBoostingRegressor`，并与 Last-Value、SeasonalMean 基线比较。时间划分固定为训练 `0–2351`、验证 `2352–2375`、测试 `2376–2399`，不得改为随机划分。
4. `schedule_tasks` 直接调度原始实际任务，不使用预测结果。它以构造式启发式逐任务选择区域和整数开工小时，并用分钟级重叠比例累计每小时 GPU 与 IT 功率占用。
5. `verify_constraints` 对 C1–C7 及到达边界进行程序化复核；任何违规都会使运行失败。
6. `make_plots` 和 `write_summary` 生成图表与汇总文件。

预测与调度是并列、解耦模块：预测输出只进入指标和图表；不要把预测值接入 `schedule_tasks`，除非题目要求发生变化。

`t1/make_charts.py` 是独立的报告制图脚本。它不读取原始 Excel，而是消费主流水线生成的 `region_type_task_count.csv` 和 `region_type_gpuh.csv`。

`t2/t2_plan.md` 定义了问题二的目标架构，但不是可执行规范。后续实现应复用问题一的任务字段、分钟级重叠、C1–C7 容量/SLA 约束和 2406 结清边界，再新增能源平衡、购售电、碳排及新能源消纳；计划明确要求分别报告“可再生优先口径收益”和“迁移/时刻优化的边际收益”，不要将二者混为同一调度收益。

## 输入数据契约

主程序依赖以下文件及工作表：

- `workload_trace.xlsx` / `Sheet1`
- `GPU_information.xlsx` / `GPU中心基础情况`
- `network_latency.xlsx` / `network_latency`
- `power_mapping.xlsx` / `任务功率映射`
- `region_time_data.xlsx` / `region_time_data`

字段契约集中在 `load_data` 的 `required` 映射中。添加或重命名输入字段时，应同步检查读取逻辑、派生量、调度约束和报告口径，不要静默填补缺失字段。

关键固定口径：

- 任务需求量纲为 `GPU_h = GPU_Demand × EstimatedDuration_min / 60`。
- IT 能耗为 `IT_MWh = GPU_h × 任务类型功率映射`。
- 调度时域 `HORIZON = 2407`，任务最后执行小时为 `2405`，所有任务须在时点 `2406` 前完成。
- 实时任务到达即开工；弹性任务可延迟但不得超过截止时间。
- 任务不可拆分、不可抢占；执行区域必须满足单向网络时延 SLA。
- GPU 和 IT 功率均按任务在每个小时内的实际重叠时长计量，不能把分钟级时长粗略取整。

## 结果与验证

完整运行会重建 `t1/output/` 下的统计、预测、调度和图表结果。重点检查：

- `t1/output/summary.json` 中任务数与调度数一致，且 `all_constraints_passed` 为 `true`；
- `t1/output/schedule/constraint_verification.csv` 中所有 `ViolationCount` 为 0；
- 若运行预测，确认 `forecast_metrics.csv`、`predictions_2376_2399.csv` 和 `selected_parameters.csv` 均已生成；
- 若修改统计 CSV 的结构，同时验证 `make_charts.py` 仍能读取 `All` 行列和固定的区域、任务类型顺序。

修改模型、统计口径或调度逻辑后，应同步核对 `t1/t1_plan.md`、`t1/分析报告.md`、`t1/数据预处理报告.md` 与实际实现；这些文档包含建模假设和已有结果，但代码与重新生成的输出是最终可执行依据。