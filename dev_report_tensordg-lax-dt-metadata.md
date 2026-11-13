# dev report · TensorDG：`solve_euler_dg_1d` 时间步元数据缺陷 + CASE-LAX-DG 工件陈旧欠账

**锚点**：`CASE-LAX-DG`（算例池·Lax via DG，验收原文「ref: 精确解｜容差 <3%」，`feature_ledger` status=open）
**分支**：`agent/m18266`　**范围**：`tensordg/euler_dg_1d.py`、`benchmarks/lax_case_pool.py`、`tests/test_euler_dg_1d_dt_metadata.py`

---

## ① 案例介绍（物理背景 / 几何 / 参数）

Lax (1954) 激波管 = Toro (1999) test 2：`[0,1]` 域、隔膜 `x0 = 0.5`、γ = 1.4、快照 `t = 0.13`。

| 量 | 左态 | 右态 |
|---|---|---|
| (ρ, u, p) | (0.445, 0.698, 3.528) | (0.5, 0.0, 0.571) |

解析波系（闭式）：p\* = 2.466098、u\* = 1.528723、ρ\*\_L = 0.344568、ρ\*\_R = 1.304085，
x(稀疏头/tail/接触/激波) = 0.157637 / 0.287229 / 0.698734 / 0.822312。
参考值由同仓闭式代数 `benchmarks/riemann_exact.py` 给出，且已被不含 Riemann 求解器的
Rusanov 有限体积族独立复核（`docs/CASE-LAX-DG-reference-crosscheck.md`）。

本单不是重跑该算例，而是**审计该算例的工件（artifact）与元数据层**——即 `CASE-LAX-DG` 这条锚点的
「验证欠账」：已合并的算例池驱动、提交工件、测试三者是否仍然自洽。

## ② 怎么做的（方法 / 流程 / 工具链）

1. 复现门禁：`python -m pytest tests/test_lax_case_pool.py tests/test_lax_case_pool_limiter_arms.py tests/test_lax_region_band.py -q`
   → **7 failed, 39 passed**（HEAD = `4f56c97`，干净工作树）。逐条定位失败根因（不靠猜）。
2. 归因：把 7 条失败分为「工件陈旧」与「驱动缺陷」两类，逐一用工件字段反查。
3. 对驱动缺陷做最小修复：**只加字段、不改既有语义**（理由见 ③）。
4. 补一条钉死该语义的回归测试，跑绿。

## ③ 计算结果（程序化注入，数字全部来自仓内工件 / 重算）

### 3.1 缺陷：`meta["dt"]` 是「被 final_time 截断的最后一步」，不是步长

`solve_euler_dg_1d` 循环内 `dt = min(cfl·dx/((2p+1)·a_max), final_time - time)`，
循环结束后 `metadata["dt"] = dt`。因此 `dt` 是**残步**：既不是运行的步长，也**不随 CFL 减半单调下降**。
提交工件 `docs/benchmarks/results/lax_case_pool/metrics.json` 的 `time_refinement`（N=100, p=1）逐位取证：

| cfl | steps | 工件记录的 `dt` | 实际平均步 `final_time/steps` |
|---|---|---|---|
| 0.35  | 555  | 1.479609e-05 | **2.342342e-04** |
| 0.175 | 1109 | 2.706407e-05 | **1.172227e-04** |
| 0.0875| 2217 | 3.539402e-05 | **5.863779e-05** |

记录值随 CFL 减半**增大**（比值 1.829、1.308），而平均步严格按 2 倍下降（比值 1.998、1.999）。
即：`tests/test_lax_case_pool.py::test_committed_time_refinement_separates_the_dt_term`
失败（`assert all(a > b for a, b in zip(dts, dts[1:]))`）**不是测试写错，而是元数据字段错名**。
任何把 `meta["dt"]` 当步长打印进文档/报告的路径（本仓已有 `conservation_study.py`、
`report_temporal_asymptotics_euler_dg_2d.py` 等多处消费 `meta["dt"]`）都会因此引用到一个数值残渣。

### 3.2 修复（加法式，零语义变更）

| 字段 | 含义 | 变化 |
|---|---|---|
| `dt` | 保持原值 = 最后一步 | 不变（向后兼容） |
| `dt_last` | 同一残步的诚实命名 | 新增 |
| `dt_mean` | `final_time / steps`，真实代表步长（`steps=0` 时为 `0.0`） | 新增 |

`benchmarks/lax_case_pool.py` 的 `run_ladder` / `run_time_refinement` 记录
`dt = meta["dt_mean"]`（代表步长）并把残步保留为 `dt_last_step`，因此
「dt 随 CFL 严格细化」这条不变量在工件里**第一次成立**，而残步仍可追溯。
顺带删掉 `production_errors_from_profiles` 里的死代码（`centres` 被下一行覆盖、`nodes` 未使用）。

### 3.3 回归测试（新增，跑绿）

`python -m pytest tests/test_euler_dg_1d_dt_metadata.py -q` → **5 passed, 25.2 s**
断言：`dt_mean == final_time/steps`；`0 < dt_last <= dt_mean`；`dt == dt_last`（兼容性）；
CFL 对折使 `dt_mean` 减半（比值 2.0 ± 10 %）且步数约翻倍；`final_time=0` 时三字段均为 0；
网格加密使 `dt_mean` 减半（±15 %）且 p3 的 `dt_mean` 小于 p1。

### 3.4 第二类失败：提交工件相对驱动陈旧（本单未闭合，如实记录）

| 失败测试 | 根因（取证） | 需要什么才能绿 |
|---|---|---|
| `test_committed_artifact_carries_the_ledger_acceptance` | 工件 `ladder[*]` 无 `l2_errors` 键；`gate_a0` 在 `production_l2 is None` 分支读它 → `KeyError: 'l2_errors'`。驱动 `run_ladder` **已**写该键（第 737 行），故工件比代码旧 | 重跑 9 腿阶梯（约 1308 s） |
| `test_every_declared_arm_has_a_record` | limiter 工件 `arms = {none, tvb}`，预登记 `{none, tvb, weno}` | 重跑 limiter 支臂（3 臂 × 9 腿） |
| `test_the_arms_actually_change_the_solution` | 同上，`KeyError: 'weno'` | 同上 |
| `test_r3_verdict_is_rederived_with_the_nested_band` | 断言需要 `p2.density`，而工件阶梯只有 `(1,200),(1,400)` 两条腿 | 同上（全阶梯 9 腿） |
| `test_shared_ladder_shape_and_monotone_error_not_assumed` | `len([200,400]) == 2 != 3`（p1 缺 N=100 腿） | 同上 |
| `test_rendered_document_matches_the_artifact` | `docs/CASE-LAX-DG-limiter-arms.md` 不存在（渲染器 `scripts/render_lax_arms_doc.py` 在仓） | 先重跑工件，再 `python scripts/render_lax_arms_doc.py` |
| `test_committed_time_refinement_separates_the_dt_term` | 见 3.1（驱动缺陷，已修） | 修完仍需重跑工件才能写进 committed metrics |

**未闭合原因（honest）**：重算已在集群递交（`cj1861` smoke、`cj1862` 环境探针），但本单结束时
队列仍有 320 个作业在前，墙钟预算不足；本机 `load15 = 117 ≥ 100`，按容量纪律不得起 1308 s + 3900 s
级重算。故工件重生成 + 文档重渲染留给下一单，**配方已写死**（见 ⑤）。

## ④ 结果图

本单为**缺陷修复 / 验证欠账**类，不产出物理场图：所有结论都是工件字段的逐位对照（③ 三张表），
作图只会把数字重画一遍而不增加信息。原算例的场量云图与解析解比较曲线
（`docs/benchmarks/results/lax_case_pool/figures/*.png`）由驱动 `render_figures` 在工件重生成时一并产出。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **`meta["dt"]` 陷阱是生态级的**：另有 ≥6 处基准/测试把 `meta["dt"]` 当步长用。本次只做了加法式修复，
   未逐个审计其语义——把它们改为 `dt_mean` 是后续单的独立工作项（有真实数值影响，不可顺手改）。
2. **工件陈旧不是「测试太严」**：7 条红灯全部可由「代码比工件新」解释，改测试放水会永久掩盖欠账。
3. **重生成配方（下一单照做，顺序不可换）**：
   ```
   python -m benchmarks.lax_case_pool --output-dir docs/benchmarks/results/lax_case_pool
   python scripts/render_lax_case_pool_doc.py
   python scripts/sync_lax_report_numbers.py
   python scripts/render_lax_acceptance_doc.py
   python -m benchmarks.lax_case_pool_limiter_arms --output-dir docs/benchmarks/results/lax_case_pool_limiter_arms
   python scripts/render_lax_arms_doc.py          # 生成缺失的 docs/CASE-LAX-DG-limiter-arms.md
   ```
   两套工件**必须来自同一次会话/同一节点**：limiter 支臂的 `arm=none` 要与 pool 记录逐位复现
   （`reproduction.worst_relative_difference < 1e-9`）。
4. **架构注意**：集群节点是 loong64，与 x86 CI 的浮点末位可能不同；重生成后 `docs/CASE-LAX-DG.md`
   的阶梯数字会以 1e-16 级漂移，必须重渲染而不是手改。

## ⑥ 误差判定（对基准逐项回答，不粉饰）

| 判据 | 量 | 值 | 基准 / 容差 | 是否 < 3 % |
|---|---|---|---|---|
| 台账 A0（`production_l2`，p3.N400，相对 L2） | 全场 ρ/u/p 最差 | **3.11556e-02** | 3.0e-02（`feature_ledger` 原文） | **否（超 3.84 %，裕度 ×0.963）** |
| 更严 L1 臂（9 腿全场相对 L1） | 最差 | 2.32905e-02 | 3.0e-02 | 是（2.33 %） |
| 本单修复的正确性 | `dt_mean` vs `final_time/steps` | 相对差 < 1e-12 | 解析恒等式 | 是 |

**结论**：本单修复不改变、也不试图改变 `CASE-LAX-DG` 的 A0 honest-FAIL（3.1156 % > 3 %）；
它修的是「把残步当步长」的元数据缺陷与一条回归护栏。A0 的 FAIL 仍未闭合，需下一单重生成工件后复核。