# m16319 交差审计：m16248 `CASE-DG-PREFINEMENT-EXPONENTIAL`

**审计单**：m16319（board-audit，交差核验，非重做）｜**被审任务**：m16248（status=succeeded，返修轮 1，父目标 m16180）
**被审仓库/分支**：TensorDG / `agent/m16248`；实现 commit `4942b0b`，交付整理 commit `cf0526c`（两者 **均不在 main**，`git merge-base --is-ancestor agent/m16248 main` = false；main 检出 `54f0a2d`）
**裁定**：**L2**（独立功能实现）——impl_files / entry / run_artifacts 三件套经独立复核**全部为真**；L3 不达（无非自带调用方、未进 main、平台侧无登记消费）；L4 不达（唯一裁判 `tensor_benchmark_cases` 该锚点 **0 行**、`tensorbench_results` 0 行、平台 `l4-verdict` = `CLAIMED_NOT_PROVEN`）
**落表**：`tensor_maturity_verdicts`（repo=TensorDG, scope=`m16248 CASE-DG-PREFINEMENT-EXPONENTIAL`, mission_id=16319, level=2, verdict_by=`board-audit`），`gate_state=pending` 交 maturity_gate 机器复核
**审计工件**：`runs/audit_m16248/{branch/,repro/,doctored/,rerun_emit_bench_json.json,repro_default_run.log,pytest_branch.log,pytest_doctored.log,projection_crosscheck.json,make_figures.py,fig1_convergence_and_amplification.png,fig2_solution_profiles.png,insert_verdict.py,mission_16248.json,api.py}`

---

## 1. 案例介绍（物理背景/几何/参数表）

一维线性对流 `u_t + a u_x = 0`，`a = 1`，`x ∈ [0,1)` 周期边界；初值 `u(x,0) = sin(2πx)`（整函数，处处解析），终时 `T = 1.0`（恰一整个周期），精确解 `u(x,T) = sin(2π(x − at))`。算例用途：在**固定网格**上只提高多项式阶 `p`（p 型加密），验证 DG 误差对解析解呈**指数收敛** `‖u − u_h‖_{L2} ≤ C·exp(−σp)`（Babuska & Suri, SIAM Review 36(4):578–632, 1994, Thm 3.1 / §3.2；任务单给出的 DOI:10.21236/ada260237）。

| 参数 | 值（交付件 `runs/p_refinement_exponential/p_refinement_results.json:parameters` 回读） |
|---|---|
| 方程 / 域 / 边界 | `u_t + u_x = 0` / `[0,1)` / periodic |
| 初值 / 精确解 | `sin(2πx)` / `sin(2π(x − t))`，`T = 1.0` |
| 网格 | `N = 16` 单元**固定**（隔离 h 加密，p 是唯一加密参数） |
| 阶数 | `p = 1,2,3,4,5,6`（DOF = 16(p+1) = 32…112） |
| 基 / 时间积分 / 限制器 | Legendre 模态 / RK4 / none（光滑解不需要限幅） |
| 时间收敛策略 | CFL 自 0.5 逐次对半，直至相邻两次 L2 相对变化 ≤ 1%（`TEMPORAL_TOL`） |
| 精度 / 环境 | float64；loongarch64 / Python 3.11.6 / torch 2.7.1 / numpy 2.4.6 |

## 2. 怎么做的（方法/流程/关键设置/工具链）

交付链路（本审计只读复跑，不改一行）：

- `tensordg/cases/p_refinement.py`（379 行，库内规范算例模块，`CASE_KEY='CASE-DG-PREFINEMENT-EXPONENTIAL':50-56`）：
  - `solve_level():110-125` → 调**生产求解器** `tensordg.advection_1d.solve_advection`，误差用 `l2_error`（真求积 L2）；
  - `temporal_converged_level():128-171` → CFL 对半直至相对变化 ≤1%，把 `cfl/dt/steps/temporal_refinements/temporal_change_rel/temporal_history` 全部**落盘**（正是 m16229 审计 F4 要求的修法）；
  - `best_approximation_error():60-97` → **不含 DG 代码**的纯 NumPy 参考：逐单元 Legendre 矩投影 + 更细求积算 `min_{v∈P_p}‖u−v‖_{L2}`（同网格上的严格下界）；
  - `run_p_refinement():183-…` → 汇总 levels、`_log10_slope`/`_sigma`（least-squares）、7 条可重导门。
- `benchmarks/p_refinement_exponential.py`（420 行，入口 `python -m benchmarks.p_refinement_exponential`，`--quick/--sensitivity/--emit-bench-json`）。
- `tensorbench/CASE-DG-PREFINEMENT-EXPONENTIAL.spec.json`（62 行，平台侧 spec：`workload.command = python3 -m benchmarks.p_refinement_exponential --emit-bench-json`；`got_pointer=metrics.sigma_got / ref_pointer=metrics.sigma_ref, tolerance_rel=0.15`）。
- `tests/test_p_refinement_exponential.py`（323 行，17 个测试）。
- `tools/gen_p_refinement_doc.py`（278 行）+ `docs/CASE-DG-PREFINEMENT-EXPONENTIAL.md`（199 行，数字脚本注入）。
- 本审计做的四件事：①独立复跑（本审计自己的 main 检出 `54f0a2d` 提供 `tensordg`，只把交付 commit 的驱动/算例/测试抽到 `runs/audit_m16248/repro`）；②**第三方**最佳逼近下界重算（法方程投影 + Simpson 10⁵ 点/单元，算法与交付件的矩展开不同）；③p=1 解析渐近值 `√(32π⁴h⁵/180)` 作外部量级锚；④伪造能力测试（把两个工件里的误差换成合成序列再跑测试）。

## 3. 计算结果（数字表）

### 3.1 独立复跑 vs 交付件（逐位比对）

`python -m benchmarks.p_refinement_exponential --emit-bench-json`（`runs/audit_m16248/rerun_emit_bench_json.json`，61.7 s）与交付 JSON **6/6 误差 + 6/6 下界 + σ + 7 门逐位一致（rel diff = 0.00e+00）**；默认 `repro_cmd` 复跑（`repro_default_run.log`，62.5 s）同结论并重现 9 件工件。

| p | DOF | 交付 L2 误差 | 复跑 L2 误差 | 相对差 | 交付 floor | 审计独立 floor | 相对差 | DG/floor |
|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 7.464011748263426e-03 | 同左 | 0.00e+00 | 4.054914489835e-03 | 4.054914491857e-03 | 4.99e-10 | 1.8407 |
| 2 | 48 | 2.089339259291412e-04 | 同左 | 0.00e+00 | 1.346285043777e-04 | 1.346285045659e-04 | 1.40e-09 | 1.5519 |
| 3 | 64 | 5.041730412373844e-06 | 同左 | 0.00e+00 | 3.331328468491e-06 | 3.331328478477e-06 | 3.00e-09 | 1.5134 |
| 4 | 80 | 9.731847331294197e-08 | 同左 | 0.00e+00 | 6.575379604254e-08 | 6.575379641534e-08 | 5.67e-09 | 1.4800 |
| 5 | 96 | 1.592692387684947e-09 | 同左 | 0.00e+00 | 1.079823969262e-09 | 1.079823979395e-09 | 9.38e-09 | 1.4750 |
| 6 | 112 | 2.228240525381386e-11 | 同左 | 0.00e+00 | 1.518521779880e-11 | 1.518522104681e-11 | 2.14e-07 | 1.4674 |

- `σ_DG = 3.9270961610557173`（交付=复跑）；`σ_floor = 3.889714378300684`（交付）/ `3.889714347055`（审计第三方，rel diff 8.0e-9）；相对偏差 `0.009610`（判据 15%）。
- 半对数斜率 `d log10(e)/dp = −1.705516` —— 任务判据「< −0.5/阶」，余量 **3.4×**。
- 缩减比 35.72 → 41.44 → 51.81 → 61.10 → 71.48（逐对）；`e(p=1)/e(p=6) = 3.3497e8`。
- 时间收敛：6/6 `temporal_converged=True`，halving 次数 1/1/1/1/3/4，末尾相对变化 2.06e-6 / 2.70e-7 / 1.29e-8 / 3.69e-3 / 4.24e-5 / 2.22e-4（全部 ≤1%，且元数据已落盘）。
- p=1 解析渐近锚：`√(32π⁴h⁵/180) = 4.063858e-03` vs 交付 4.054914e-03 → **+0.22%**（O(h) 修正符合预期），量级被外部解析式独立确认。
- 7 门（交付=复跑）：G1 斜率 −1.7055≤−0.5 PASS；G2 单调 35.72≥1 PASS；G3 速率对下界 0.96%≤15% PASS；G4 放大上界 1.8407≤2.0 PASS；G5 误差降 3.3497e8≥1e4 PASS；G6 时间收敛 1.0 PASS；G7 逐对超线性 3.5758 PASS。

### 3.2 测试与伪造能力（对抗性核验）

| 检查 | 命令 | 结果 |
|---|---|---|
| 交付测试 | `pytest tests/test_p_refinement_exponential.py -q`（`pytest_branch.log`） | **17 passed in 28.0 s** |
| 伪造工件反例 | 把 `p_refinement_results.json`（`diagnostics.errors` + `levels[].error_l2`）与 `benchmark_ref_vs_got.json`（`rows[].got_dg`）同步换成合成序列 `[1e-2,2e-4,2e-6,1e-8,2e-11,2e-14]` 后重跑 | **1 failed, 16 passed**（`pytest_doctored.log`；失败= `test_quick_sweep_recomputes_the_archived_numbers`）→ 具备端到端重算防伪造力（对照 m16180：同样手法 12/12 passed） |

### 3.3 三件套与「唯一裁判」核验（基准判据是唯一裁判）

| 项 | 判据来源 | 结果 |
|---|---|---|
| run_artifacts | 交付目录 `runs/p_refinement_exponential/` | **9 件全在**（results/table/csv/ref-vs-got/tensorbench_result/param_sensitivity/manifest/3×png）；manifest 列出的 **8 个 sha256 全部与文件逐位相符**（本审计重算），换机可用 |
| benchmark ref vs got | `tensor_benchmark_cases WHERE anchor_slug='CASE-DG-PREFINEMENT-EXPONENTIAL'`（30 行全表） | **0 行**（TensorDG 仅 `CASE-EULER-VORTEX` id=16 passed、`CASE-SHU-OSHER-DG` id=19 passed）→ **无注册容差可比** |
| 平台机裁 | `tensorbench_results WHERE mission_id ∈ {16180,16248}` | **0 行**；`tensorbench_evaluations` = `tbe-397f10…`（state=`submitted`, cluster_job=1290）；`cluster_jobs#1290` = **`queued`，从未启动**（队列当前 468 queued / 15 running，最老 queued 自 11-09）；`GET /tensor/bench/l4-verdict/16248` = `CLAIMED_NOT_PROVEN, rows=0`，failed_checks = execution_success / source_cluster_job / cluster_job_bound / ref_got_error_present / within_tolerance |
| repro_cmd | manifest `repro_cmd = python -m benchmarks.p_refinement_exponential` | **可执行**（本审计两次实跑 70.3 s / 62.5 s，产物齐全）|
| 算例池登记 | `tensor_case_pool#98` `DG-PREFINEMENT-EXPONENTIAL` | `linked_anchor` 为空、`status=dispatched`（未与裁判表锚点连上） |

## 4. 结果图

- `runs/audit_m16248/fig1_convergence_and_amplification.png`：左=复跑 DG L2 误差 vs p（semi-log，含指数拟合 σ=3.9271）叠加**本审计第三方**最佳逼近下界（σ_floor=3.8897）；右=DG/floor 放大比 1.841→1.467，落在严格下界 1.0 与声明带 2.0 之间。
- `runs/audit_m16248/fig2_solution_profiles.png`：场量比较（1 维无云图，用剖面替代）——`T=1` 时精确波 `sin(2π(x−1))` 与 p=1、p=4 的 DG 数值解；p=1 可见幅度/相位误差，p=4 目视重合。
- 交付自带：`runs/p_refinement_exponential/p_refinement_semilog.png`（72 196 B）、`p_refinement_log10.png`（72 974 B）、`p_refinement_amplification.png`（47 582 B），sha256 与 manifest 一致，非空非占位。

## 5. 注意点（坑/局限/适用范围）

1. **唯一裁判缺席（阻塞 L4）**：`tensor_benchmark_cases` 该锚点 0 行、`tensorbench_results` 0 行、平台 l4-verdict `CLAIMED_NOT_PROVEN`。返修交付自带 `tensorbench_result.json` 与 spec，但那是**自产工件**，不是平台重算行；作者在 result.checks 里如实写了「l4-verdict 仍 rows=0（排队中）」，属诚实申报，不构成造假。
2. **集群队列停摆（生态摩擦，非本单过错）**：`cluster_jobs#1290`（11-12 11:52 递交）至今 `queued`；队列 468 queued / 15 running，running 作业自 11-09/11-10 起未推进。凡依赖「平台重算 → tensorbench_results」的 L4 证明，本波在结构上不可达，建议记 friction 而非记在被审者账上。
3. **外部锚是「定律」不是「数值表」（局限）**：Babuska & Suri 1994 只给指数收敛定律与该问题无关的 σ，不提供本算例 σ 表值；交付件的 `ref` 是**同一网格上严格 L2 最佳逼近下界**（自算，但独立于 DG 代码，且经本审计第三方实现与 p=1 解析渐近值双重确认）。因此「ref vs got」是**代码—数学下界比对**，不是文献表值比对；这也正是不能判 L4 的第二条理由。
4. **L3 边界**：`tensordg/cases/p_refinement.py` 是库内规范算例（生产求解器 + 单一真值场所），但仓内调用方只有自带驱动 `benchmarks/p_refinement_exponential.py:42` 与自带测试（`run_p_refinement` docstring 自述「CLI 与 tests 是两个调用方」）；`.github/workflows/tests.yml` 跑全量 pytest（会跑到这 17 项），除此之外**无生产/CI 消费方**，且分支未进 main → 按「接入主求解器/平台」的 L3 定义不成立。
5. **判据判别力**：G1（σ>0.5）对整函数极易满足（余量 3.4×），G2 单调性近乎必然；真正有信息量的是 G3（速率对独立下界 15%）与 G4（放大带 2.0），二者都是**事前声明**（`criterion` 块）并在两次独立复跑中一致，未事后回填。p=6 时 DG 误差 2.23e-11 已接近 float64 舍入地板（L2 求积 ~1e-16 相对），更高 p 会先撞舍入而非撞收敛率——**适用范围 p≤6、N=16**。
6. **交付工件中的环境痕迹**：`manifest.json` 的 `git_revision` 写的是 `e723dd4…-dirty`（父目标 m16180 的 main 检出），而非本单 commit `4942b0b`；`tensorbench_result.json` 的 `implementation.revision` 同值。工件本身可复现（本审计复跑逐位一致），但**版本指针与交付 commit 不对应**，建议后续把 revision 改成「运行时代码所在 commit」。
7. **适用范围**：仅 1 维线性常系数、整函数初值、周期边界。不外推到激波/非光滑解（那属 h 型/限幅器算例职责）。

## 6. 误差判定（逐项，显式回答「是否 <3%」）

| 关键量 | 参考/基准 | 实测 | 误差 | 是否 <3% |
|---|---|---|---|---|
| 6 个 p 级 L2 误差 | 交付件（`4942b0b` 工件） | 本审计独立复跑 | **0.00e+00**（逐位相等） | **是**（0%） |
| σ_DG 衰减率 | 交付 3.9270961610557173 | 本审计复跑同值 | 0.00e+00 | **是** |
| 6 个最佳逼近下界 | 交付件 | 本审计第三方实现（法方程 + Simpson 1e5 点） | ≤ **2.14e-07**（p=6）/ ≤9.4e-9（p≤5） | **是** |
| σ_floor | 交付 3.889714378300684 | 审计第三方 3.889714347055 | **8.0e-09** | **是** |
| p=1 下界 vs 解析渐近 `√(32π⁴h⁵/180)` | 4.063858e-03 | 4.054914e-03 | **0.22%**（O(h) 修正） | **是** |
| 任务判据（半对数斜率 < −0.5/阶） | −0.5 | −1.705516 | 余量 3.4×，通过 | 是（判据为阈值型） |
| G3 速率对下界偏差 | 预声明 15% | 0.961% | 余量 15.6× | **是** |
| **注册基准 ref vs got** | `tensor_benchmark_cases` **0 行** | — | **不可比** | **无法回答**（如实声明 → 不判 L4） |

**结论**：数值是真的、可逐位复现、可被第三方独立实现与解析式双重确认；测试具备端到端重算的防伪造力（伪造即 1 failed）。三件套（impl_files 379 行 / entry 420 行可执行 / run_artifacts 9 件 sha256 齐）**全部核实为真**，达 **L2 独立功能实现**。L3 因「无自带之外的调用方 + 未进 main」不达；L4 因「唯一裁判 0 行 + 平台 l4-verdict CLAIMED_NOT_PROVEN（评测作业 1290 至今 queued）」不达——两者均如实登记，不虚标。**disposition：approve with notes**（L2 消账；返修建议见 §5 第 2/6 条与文末）。

**返修非阻塞建议**：(a) 注册 `tensor_benchmark_cases` 行（anchor=`CASE-DG-PREFINEMENT-EXPONENTIAL`，metric=`σ deviation vs independent floor`，tolerance=`15%`）并把评测作业跑通后 ingest `tensorbench_results`，L4 即可机裁；(b) `manifest/tensorbench_result` 的 revision 改为运行时代码 commit；(c) 合并入 main 并在 `verification_ladder` C4 或 CI 设一个非自带消费方，L3 线即通。