# 交差审计 · m16137 · CASE-DG-DALEMBERT-ELASTIC-1D（TensorDG）

审计类型：任务榜交差核验（非重做）｜审计单 m16195｜裁定人 board-audit｜日期 2026-11-12

对象：mission 16137（status=succeeded，decided_by=eco_admin，auto-approve dev_acceptance），
交付 commit `3240063`（分支 `agent/m16137`），工作树 `/root/tp-agent/work/m16137`。

---

## 0. 裁定

**L2（独立功能实现：真代码 输入→计算→输出）**。
数值实质已到 L4 强度（完整算例 + 精确解逐项误差 + 收敛研究 + 可复现），
但两处爬梯事实缺位，按"级别=功能实质、缺一档不越级"判 L2：

| 档 | 判定 | 依据（可复核指针） |
|---|---|---|
| L0 文档 | ✅ | `docs/CASE-DG-DALEMBERT-ELASTIC-1D.md` 六段齐全（①问题/②方法/③参数/④ref-vs-got/⑤收敛敏感性/⑥失败模式与边界） |
| L1 数学/单元 | ✅ | `tests/test_elastic_rod_1d.py::test_exact_solution_solves_the_system`（解析解自身 PDE 残差 <1e-3 相对）、`::test_invariant_is_constant_along_characteristic`、`::test_bar_wave_speed_matches_material_data` |
| L2 独立实现 | ✅ **本单独立复跑** | 真求解器 `tensordg/elastic_1d.py`（456 行：模态 Legendre 基 + 精确 Riemann upwind 通量 + RK4/SSP-RK3，`solve_elastic_rod:298-400`）；本单 `pytest tests/test_elastic_rod_1d.py -q → 15 passed in 44.84 s`（`artifacts/audit_m16137/pytest_rerun.log`） |
| L3 系统集成 | ❌ | **main 无该文件**：`/root/tp-agent/work/m16195`（main `cc1c2df`）`tensordg/elastic_1d.py` 不存在、`git log -- tensordg/elastic_1d.py` 空；`tensordg/__init__.py` 未导出（对照 main 已导出 20+ 求解器 `:1-40`，含 `ns_dg_2d`/`poisson_2d`）；`tensordg/cases/__init__.py` 只导出 `shock_entropy`；仓内无第三方调用方（`git grep elastic_1d 3240063` 只命中自身 bench/test/case） |
| L4 基准验证 | ❌（机器门口径） | `GET /tensor/bench/l4-verdict/16137 → status=CLAIMED_NOT_PROVEN, rows=0, failed_checks=[execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance]`；`tensorbench_results WHERE mission_id=16137 → 0 行`；`tensor_benchmark_cases WHERE anchor_slug='CASE-DG-DALEMBERT-ELASTIC-1D' → 0 行`；`tensor_case_pool` id=96 仍 `status=dispatched, linked_anchor=NULL`。交付跑在本机（225 s）无集群作业号 → `source_cluster_job` 天然缺 |
| L5 科研工程 | ❌ | 无新方法/新物理，属标准 DG 对标准精确解的验证 |

---

## 1. 取证：三件套是否真存在

| 件 | 计数 | 结论 |
|---|---|---|
| run_artifacts | `runs/elastic_rod_dalembert/results.json`（18.6 KB）+ fig1..fig4.png（4 张，248/129/99/191 KB） | **真存在**，且本单 `--quick` 全链重跑在 83.8 s 内重建 `results.json` + 同名 4 图（`artifacts/audit_m16137/bench_quick/`） |
| benchmark ref vs got 数值 | `results.json["probes"].rows` 15 点探针（σ_exact/σ_got/abs_err/rel 三个误差口径）+ `["gate"]` + `["convergence"]` 16 行 + `["reflection"]` + `["energy"]` | **真存在且可真算**：本单用**自写**参考解与**自写** L2 范数复算，逐位一致：`6.429376841719e-07` vs 交付 `6.429376841693551e-07`（相对差 4e-12） |
| repro_cmd 可执行 | `python -m benchmarks.elastic_rod_dalembert [--quick]`、`python -m pytest tests/test_elastic_rod_1d.py -q` | **可执行**：本单从 `git archive 3240063` 的干净副本执行；**全量模式 213.6 s 完成**，重建 `results.json` 的 gate 值 `6.429376841693551e-07` 与交付**逐位一致**（`artifacts/audit_m16137/bench_rerun/`），另 `--quick` 83.8 s 复跑同判据 PASS |

## 2. 基准裁判（我方独立复算，不调用交付侧验证函数）

审计脚本 `docs/board_audit/audit_m16137.py`：只 import 被测求解器 `solve_elastic_rod` 与题面初值，
解析参考解、模态重构、L2 范数、Fourier 谱解、能量积分全部自行实现。
产物 `artifacts/audit_m16137/independent_recompute.json`。

| 项 | ref（独立口径） | got（交付/复算） | 误差 | 容差 | 判 |
|---|---|---|---|---|---|
| 主判据：σ 相对 L2，P3 N=200 CFL=0.3 | 自写 d'Alembert 闭式 | 6.429e-07 | 6.4e-07 | <2%（case_pool id=96） | ✅ 余量 ×31107 |
| **是否 <3%** | — | — | **是，低 5 个数量级** | 3% | ✅ |
| 收敛阶 P1/P2/P3/P4 | 设计 p+1 | 2.328/3.414/4.051/5.011 | 偏差 ≤0.42 | ±0.55 | ✅ 无隐性降阶 |
| 反射壁应力（fixed） | 2.0（Graff Ch.1） | 1.99999978 | 2.2e-7 | 2e-3 | ✅ |
| 反射壁应力（free） | 0.0 | 2.16e-7 | 2.2e-7 | 2e-3 | ✅ |
| 弹性能漂移（本单配置 P3 N=200，0.4 L） | E0=0.11429043 | E1=0.11429043 | 5.5e-10 | <1e-6 | ✅ |

**参考解判别性检验（关键，防"自证"）**：交付的"精确解"若无判别力，任何数值解都能自洽通过。
本单用**完全不同的路径**——Fourier 谱传播子 `exp(±ickt)` 直接解同一 PDE（全程不出现 d'Alembert 叠加）——
与闭式解在 6 个探针点比对：最大绝对差 / 幅值 = **5.5e-15**（`C3_reference_vs_spectral`）。
结论：参考解是真·解析解，非同一离散化的自我引用。

## 3. 对抗性审读（重要但非阻断）

1. **判据是"工况特定"的**：2% 只在 P3/N=200 成立；交付自己披露 P1/N=25 = 38%（唯一超标工况，每波长 5 单元）。
   建议 `tensor_benchmark_cases` 的 `tolerance` 行显式写入分辨率前提，否则他人拿粗网格复现会得到"FAIL"错觉。
2. **时间积分不污染空间阶的宣称成立**：CFL 0.05→0.9 误差恒 1.025e-5（交付 ⑤-3），本单的收敛表内 dt∝h 且 P4 仍见 5.01 阶 → 一致。
3. **"无物理弥散"表述无过度声明**：细杆理论（无侧向惯性）前提在 ⑥ 显式写出，峰值位置误差以"包络宽"归一，诚实。
4. **生态摩擦记录**（交付 ⑥）：`knowledge_base`/`negative_knowledge_ledger` 对 agent 角色 permission denied，三查只能得"无条目"；
   与本单无关但是可复现的基建缺口。
5. **报告归属错位**：`tensor_reports` 中该算例报告挂在 mission_id 16149/16158（后续单），
   16137 自身 `report_id` 为空、`tensor_reports WHERE mission_id=16137 → 0 行`，而交付宣称 `POST /reports → 200 id=report_c222e0c255b4`。
   不影响物理结论，但**审计溯源链断裂**，建议平台把外部 report id 回填到 `tensor_dev_missions.report_id`。

## 4. 结论与返修建议（reopened 时的最小动作）

物理与数值部分**认定通过**：真求解器、真解析解、可复算、误差比判据低 5 个数量级，无造假迹象。

补满档位需 3 个机器动作（均不含新物理）：
1. 分支 `agent/m16137` 合入 main，并在 `tensordg/__init__.py`/`tensordg/cases/__init__.py` 导出
   `solve_elastic_rod`、`solve_case`、`gate_report`（→ 补 L3）；
2. `tensor_benchmark_cases` 落行（anchor_slug/metric/tolerance）+ `tensor_case_pool` id=96 置 `passed` 并填 `linked_anchor`；
3. 用 `run_cluster` 复跑 benchmark 产生 `cluster_job_id` 并 ingest `tensorbench_results`（ref/got/error 一行）→ 补 L4。
   本单已按此路径递交 `cj1264`（stage_dir=干净检出，pytest+benchmark+独立复算），递交时队列 435 位、4.2 min 仍在排队，
   **未取回结果，不计入证据**（诚实标注 not_run）。

## 5. 本单执行清单

- `python -m pytest tests/test_elastic_rod_1d.py -q` → **15 passed in 44.84 s**（干净 `git archive` 副本）
- `python -m benchmarks.elastic_rod_dalembert`（全量）→ results.json + 4 PNG，**213.6 s**，gate `6.429376841693551e-07` 与交付逐位一致（`artifacts/audit_m16137/bench_rerun/`）
- `python -m benchmarks.elastic_rod_dalembert --quick` → results.json + 4 PNG，83.8 s，gate 6.429e-07 PASS
- `python docs/board_audit/audit_m16137.py --repo <干净副本>` → C1/C3/C4/C5 PASS，C2 阶 2.33/3.41/4.05/5.01（`artifacts/audit_m16137/independent_recompute.json`）
- 集群：`cj1264` 已递交未完成（不计数）