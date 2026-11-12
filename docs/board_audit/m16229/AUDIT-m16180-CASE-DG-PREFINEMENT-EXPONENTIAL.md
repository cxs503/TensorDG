# m16229 交差审计：m16180 `CASE-DG-PREFINEMENT-EXPONENTIAL`

**审计单**：m16229（board-audit，交差核验，非重做）｜**被审任务**：m16180（succeeded）
**被审仓库/分支**：TensorDG / `agent/m16180` = `a7a279c568213757c5e013b2cab46cae17a00145`（2026-11-12 19:14:27 +0800，已 push；m16180 工作树 `afb6ddd` 四件产出 blob sha256 与之一致）
**裁定**：**L2**（独立功能实现）｜不达 L3（无调用方、未进 main）｜不达 L4（唯一裁判 0 行 + 平台 l4-verdict `CLAIMED_NOT_PROVEN`）
**落表**：`tensor_maturity_verdicts` id=450（repo=TensorDG, scope=`m16180 CASE-DG-PREFINEMENT-EXPONENTIAL`, level=2, verdict_by=board-audit, gate_state=pending）
**审计脚本/工件**：`runs/audit_m16180/{write_verdict.py,run_repro.sh,make_figures.py,repro/,projection_crosscheck.json,temporal_meta_quick.json,synthetic_gate_demo/,pytest_rerun.log,fig1_*.png,fig2_*.png}`

---

## 1. 案例介绍（物理背景/几何/参数表）

一维线性对流 `u_t + u_x = 0`，`x∈[0,1]` 周期边界；初值 `u(x,0)=sin(2πx)`（整函数，处处解析），终时 `T=1.0`（恰一个周期），精确解 `u(x,T)=sin(2πx)`。算例用途=验证 p 型（提高多项式阶）加密对解析解的**指数收敛**：Babuska & Suri (1994, SIAM Review 36(4):578-632, Thm 3.1/§3.2) 给出 `‖u−u_h‖_L2 ≤ C·exp(−σ p)`。

| 参数 | 值 |
|---|---|
| 方程 / 域 | `u_t+u_x=0` / `[0,1]` periodic |
| 初值 / 精确解 | `sin(2πx)` / `sin(2π(x−t))`，T=1.0 |
| 网格 | N=16 单元固定（隔离 h 加密） |
| 阶数 | p = 1,2,3,4,5,6 |
| 基 / 通量 / 时间 | Legendre 模态 / upwind / RK4（CFL 对半直至时间误差<1%） |
| 求积 | 空间算子 `max(2p+3,4)`；L2 误差 `max(2p+4,6)` 点 Gauss |
| 精度 | torch.float64；loong64 / PyTorch 2.7.1 |

## 2. 怎么做的（方法/流程/关键设置/工具链）

- 驱动：`benchmarks/p_refinement_exponential.py`（328 行）→ `tensordg.advection_1d.solve_advection`（生产求解器，未改）→ `l2_error`（含 `0.5h` Jacobian 的真求积 L2）→ `tensordg.convergence.p_decay_diagnostics`（既有，m2091 引入）算 lsq 斜率/缩减比/单调性/超线性。
- 每阶先做时间收敛：`temporal_converged_advection:56-96`，CFL 从 0.08 逐次对半，直至相邻两次 L2 相对变化 <1%（`TEMPORAL_TOLERANCE`），最多 6 次。
- 判据（脚本内硬编码，`check_criterion:140-157`）：`σ = −d(ln e)/dp > DECAY_RATE_MIN=0.5` 且误差逐阶单调下降。
- 产出：`p_refinement_results.json`（schema/问题/结果/checks/verdict/figures）、`p_refinement_table.csv`、两张 PNG（semi-log、log10）、文档 `docs/CASE-DG-PREFINEMENT-EXPONENTIAL.md`（156 行，六段）。
- 本审计流程（只读）：从 audited commit 抽驱动到独立目录，用审计人自己的 main 检出（e723dd4）提供 `tensordg`，全量复跑 71.2 s；另写**独立 numpy L2 投影**（不调 tensordg）作量级交叉；再对 12 项测试做**合成工件反例**以测其防伪造能力。

## 3. 计算结果（数字表）

复跑（独立拷贝，main 库）与交付件 **6/6 误差 + 全部 diagnostics 逐位一致（rel diff 0.00e+00）**：

| p | DOFs | 交付 L2 误差 | 复跑 L2 误差 | 独立 L2 投影误差 | DG/投影 |
|---|---|---|---|---|---|
| 1 | 32 | 7.464013366971308e-03 | 7.464013366971308e-03 | 4.054914e-03 | 1.841 |
| 2 | 48 | 2.089339334445052e-04 | 同左 | 1.346285e-04 | 1.552 |
| 3 | 64 | 5.041732675406567e-06 | 同左 | 3.331328e-06 | 1.513 |
| 4 | 80 | 9.731718626951470e-08 | 同左 | 6.575380e-08 | 1.480 |
| 5 | 96 | 1.592692130145561e-09 | 同左 | 1.079824e-09 | 1.475 |
| 6 | 112 | 2.228252443658065e-11 | 同左 | 1.518522e-11 | 1.467 |

`σ_lsq = 3.9270958355627648`（交付=复跑）；缩减比 35.7×→71.5×；单调下降；超线性=True。
时间收敛补证（交付件未落盘，本单 quick 模式重算）：p=1..3 全 `temporal_converged=True`，`refinements=1`，CFL 0.08→0.04→0.02，`temporal_change ≤ 3.6e-9`（`runs/audit_m16180/temporal_meta_quick.json`）。

## 4. 结果图

- `runs/audit_m16180/fig1_convergence_vs_projection.png`：左=复跑 L2 误差 vs p（semi-log）叠加独立 L2 投影与指数拟合 σ=3.927；右=DG/投影比值，p=3..6 稳定 1.47–1.51。
- `runs/audit_m16180/fig2_solution_profiles.png`：场量比较（1D 无云图，用剖面替代）：p=1 与 p=4 的 DG 数值解 `u_h(x,T=1)` vs 精确波（p=1 可见明显相位/幅度误差，p=4 目视重合）。
- 交付自带：`docs/benchmarks/results/p-refinement-exponential/p_refinement_semilog.png`、`p_refinement_log10.png`（均非空，67 kB / 91 kB）。

## 5. 注意点（坑/局限/适用范围）

1. **唯一裁判缺席**（F1）：`tensor_benchmark_cases WHERE anchor_slug='CASE-DG-PREFINEMENT-EXPONENTIAL'` → **0 行**（全表 29 行，TensorDG 仅 EULER-VORTEX / SHU-OSHER-DG）；`tensorbench_results WHERE mission_id=16180` → 0 行；`tensor_case_pool` 无该锚点。平台 `GET /tensor/bench/l4-verdict/16180` = `CLAIMED_NOT_PROVEN, rows=0`，`failed_checks=[execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance]`。作者 "PASS" 为自证。
2. **测试无防伪造能力**（F2）：12 项 pytest 只读已提交 JSON；把 errors 换成合成序列 `[1e-2,2e-4,2e-6,1e-8,2e-11,2e-14]` 并同步 diagnostics 后仍是 **12 passed**（`synthetic_gate_demo/pytest_doctored.log`）。测试通过 ≠ 计算正确。
3. **交付未进主干**（F3）：`git merge-base --is-ancestor a7a279c main` = false，main（e723dd4）无驱动与文档；`grep` 全仓无第二调用方（unified_verification / tools / CI 均未登记）→ 库用户与 CI 不可见。
4. **时间收敛证据被丢弃**（F4）：`p_refinement_study` 算了 cfl/dt/steps/temporal_converged，但 `build_report:231-262` 只持久化 degrees/errors/diagnostics → 交付工件无法自证"时间已收敛"。
5. **绝对路径**（F5）：交付 JSON 的 `figures` 写死 `/root/tp-agent/work/m16180/...`，换机即失效。
6. **判据判别力弱**（F6）：σ>0.5 对整函数极易满足（余量 7.9×），单调性对光滑解析推进近乎必然；参考是定性定理，不存在 delta 型数值 ref。
7. **适用范围**：仅 1 维线性常系数、整函数初值、周期边界；不外推到激波/非光滑解（那些应是 h/限幅器算例的职责）。

## 6. 误差判定（逐项，显式回答）

| 关键量 | 参考/基准 | 实测 | 误差 | 是否 <3% |
|---|---|---|---|---|
| 6 个 p 级 L2 误差 | 交付件（audited commit a7a279c） | 本单独立复跑 | **0.00%**（逐位相等） | **是** |
| σ_lsq | 交付 3.9270958355627648 | 本单复跑同值 | **0.00%** | **是** |
| DG 误差 vs 独立 L2 投影（下界） | p=3..6 投影误差 | DG 误差 | 比值 1.47–1.51（常数因子，理论预期） | **是**（量级一致，非逐点容差判据） |
| 判据 σ>0.5 | 作者自定阈值 | 3.927 | 余量 7.9× 通过 | 是（但阈值自定、判别力弱，F6） |
| 注册基准 ref vs got | **无注册行（0/29）** | — | **不可比** | **无法回答**（如实声明，故不判 L4） |

**结论**：数值本身真且可逐位复现，未发现造假（F7：驱动无硬编码误差常数；代码为真弱形式；独立投影交叉吻合）。但"基准验证"层因唯一裁判表 0 行 + 平台 l4-verdict `CLAIMED_NOT_PROVEN` 无法机器确认；交付又无调用方且不进主干 → **L2 消账，L3/L4 明确不达**。建议返修项（非阻塞 L2 消账）：补注册 `tensor_benchmark_cases` 行并 ingest `tensorbench_results`、把 `levels`（时间收敛元数据）落盘、加一项端到端重算测试、figure 路径改相对、合并入 main。