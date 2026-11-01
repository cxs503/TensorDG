# TensorDG 上游差集与提交主题演化审计（m1634）

> 机器可读工件：`artifacts/upstream_delta_m1634.json`（schema `tensordg.upstream_delta/1`）；图表：`artifacts/upstream_delta_m1634.png`；生成器：`tools/upstream_delta.py`、`tools/render_upstream_delta_md.py`。本文所有数字均由脚本从上述工件/`git` 读取注入，无手工抄写。

## ① 案例介绍

| 项 | 值 |
|---|---|
| 仓库 | `cxs503/TensorDG`（远端 `origin`，工作树目录 `m1634`） |
| 本地 HEAD | `7d6f62ea044b` (`7d6f62ea044b3180dc71011ab4058d8d189e67e0`) |
| 上游基线 | `origin/main` = `7d6f62ea044b` |
| 两者是否同一提交 | **是（差集为空）** |
| ahead / behind 提交数 | 0 / 0 |
| 未并入 HEAD 的远端分支 | 1 个（合计 +1534 行） |
| 最近提交窗口 | 15 条（2026-10-23 → 2026-10-25，3 个提交日） |
| 其中 merge 提交占比 | 46.7% |

TensorDG 是本生态的间断 Galerkin（DG）求解器仓：1D/2D Euler（TVB/WENO 限制器）、标量对流/热传导/Burgers、SIPG 椭圆（matrix-free 与 sparse 两条路径）、以及对照精确解/参考解的 benchmark 与基准套件。本单要回答的不是「代码算得准不准」，而是**本仓与它自己的远端上游之间是否已经漂移**，以及最近 15 次提交的主题演化与风险面。对 TensorDG 这类以数值精度为交付物的仓库，「上游有成熟成果但没进 main」等价于**成果不在任何 CI/发布门的视野内**，风险等级高于任何单点精度偏差。

## ② 方法与工具链

只使用只读 git 管道命令，不写仓库、不推远端：`rev-parse` / `merge-base` / `rev-list` / `for-each-ref` / `log --numstat` / `diff --name-status`。流程：

1. **差集**：`branch_delta(base=origin/main, head=HEAD)` 给出 identical/ahead/behind/merge-base；
2. **未并入上游盘点**：枚举 `refs/remotes/origin/*`，用 `merge-base --is-ancestor` 剔除可达分支，余者即「远端存在、HEAD 不可达」的成果；
3. **主题归类**：按路径规则（顶层目录优先于文件名启发式）把每个文件映射到单一主题；commit 主题取非文档文件的最大计数主题；
4. **风险标注**：10 条路径/规模规则（下表），每条给出等级与可执行含义；
5. **可复现性**：工件不写入生成时刻时间戳，`--check` 模式可逐字段比对，失配即 exit≠0（防「报告漂移」）；
6. **验证**：`tests/test_upstream_delta.py` 在临时合成仓上固定作者/提交时间，覆盖主题映射、风险规则、ahead/behind 算术、未并入分支识别、工件新鲜度。

| 规则 | 等级 | 含义（触发即需动作） | 命中数 |
|---|---|---|---|
| `R1_DOC_ONLY` | INFO | docs/artifacts only: no executable code touched, claims must be traced back to an artifact or test | 4 |
| `R2_TESTS_TOUCHED` | MEDIUM | test suite touched: must be re-run before trusting the affected capability | 2 |
| `R3_NUMERIC_ARTIFACT` | HIGH | machine-readable numeric report rewritten: hand-edited numbers are indistinguishable from computed ones -> regenerate to verify | 4 |
| `R4_API_SURFACE` | MEDIUM | tensordg/ public module touched: downstream importers/compat need a check | 3 |
| `R5_PHYSICS_CORE` | HIGH | scheme/limiter core touched: accuracy verdict requires re-running the exact-solution benchmarks | 3 |
| `R6_MERGE_COMMIT` | INFO | merge commit: excluded from churn accounting to avoid double counting | 7 |
| `R7_LARGE_CHURN` | MEDIUM | large churn (>=400 changed lines): review attention / split into reviewable units | 3 |
| `R8_CI_CONFIG` | MEDIUM | CI or packaging changed: green badge no longer implies the previous gate set | 1 |
| `R9_UNMERGED_BRANCH` | HIGH | upstream branch not reachable from HEAD -> invisible to main CI | 1 |

## ③ 计算结果

### 3.1 差集结论

`origin/main` 与本地 `HEAD` **指向同一提交** `7d6f62ea044b`：**差集为空**，ahead=0、behind=0，merge-base 即该提交本身。换言之 main 线不存在「本地落后上游」或「上游落后本地」的问题。

但「main 无差集」不等于「上游无差集」：远端仍存在未并入 HEAD 的分支，见 3.2。

### 3.2 未并入 HEAD 的远端分支（本次审计的核心发现）

| 远端分支 | tip | 领先提交 | 文件 | 新增行 | 删除行 | 命中主题 |
|---|---|---|---|---|---|---|
| `origin/agent/m534` | `7591cf670` | 1 | 16 | +1534 | -0 | artifacts, docs, tests, verification |

该分支 tip 提交：`7591cf670`（2026-10-31）m534: matured mission deliverables。相对 HEAD 的差异文件（前 20）：

| 文件 | + | - | 主题 |
|---|---|---|---|
| `artifacts/converge_advection_none_p1.json` | 58 | 0 | artifacts |
| `artifacts/converge_advection_none_p2.json` | 58 | 0 | artifacts |
| `artifacts/converge_advection_tvb_p2.json` | 58 | 0 | artifacts |
| `artifacts/converge_advection_weno_p2.json` | 58 | 0 | artifacts |
| `artifacts/converge_poisson_p1.json` | 58 | 0 | artifacts |
| `artifacts/converge_poisson_p2.json` | 58 | 0 | artifacts |
| `artifacts/converge_poisson_p3.json` | 48 | 0 | artifacts |
| `artifacts/convergence_plots.png` | bin | bin | artifacts |
| `artifacts/gci_poisson_p2.json` | 11 | 0 | artifacts |
| `docs/benchmark_convergence_dealii_tensordg.md` | 89 | 0 | docs |
| `tensordg/converge.py` | 379 | 0 | verification |
| `tensordg/converge_advection.py` | 108 | 0 | verification |
| `tensordg/converge_poisson.py` | 75 | 0 | verification |
| `tensordg/gen_plots.py` | 64 | 0 | verification |
| `tensordg/run_all_convergence.py` | 180 | 0 | verification |
| `tests/test_converge.py` | 232 | 0 | tests |

### 3.3 最近 15 提交的主题演化

| # | 提交 | 日期 | 主题 | 文件 | +行 | -行 | merge | 风险标注 |
|---|---|---|---|---|---|---|---|---|
| 1 | `7d6f62e` | 2026-10-25 | —（merge） | 0 | 0 | 0 | Y | R6 |
| 2 | `1d8385c` | 2026-10-25 | —（merge） | 0 | 0 | 0 | Y | R6 |
| 3 | `6d95dae` | 2026-10-25 | —（merge） | 0 | 0 | 0 | Y | R6 |
| 4 | `369d0e1` | 2026-10-25 | artifacts | 1 | 64 | 0 |  | R1, R3 |
| 5 | `6ed905a` | 2026-10-25 | docs | 1 | 262 | 0 |  | R1 |
| 6 | `606ede1` | 2026-10-25 | —（merge） | 0 | 0 | 0 | Y | R6 |
| 7 | `67065fa` | 2026-10-25 | physics_core | 6 | 1308 | 130 |  | R3, R4, R5, R7 |
| 8 | `e22c308` | 2026-10-25 | physics_core | 13 | 1149 | 130 |  | R2, R3, R4, R5, R7 |
| 9 | `fea8844` | 2026-10-24 | —（merge） | 0 | 0 | 0 | Y | R6 |
| 10 | `8e94ea6` | 2026-10-24 | artifacts | 1 | 139 | 0 |  | R1, R3 |
| 11 | `78adc0c` | 2026-10-24 | docs | 1 | 226 | 0 |  | R1 |
| 12 | `c485c62` | 2026-10-24 | —（merge） | 0 | 0 | 0 | Y | R6 |
| 13 | `432bef0` | 2026-10-23 | benchmarks | 75 | 11619 | 132 |  | R2, R4, R5, R7, R8 |
| 14 | `141cbac` | 2026-10-23 | —（merge） | 0 | 0 | 0 | Y | R6 |
| 15 | `f71873f` | 2026-10-23 | benchmarks | 1 | 1 | 1 |  | — |

非 merge 提交的主题计数：artifacts=2, benchmarks=2, docs=2, physics_core=2；含 merge 时：artifacts=2, benchmarks=2, docs=2, none=7, physics_core=2。

按主题的代码周转（insertions+deletions，仅非 merge）：

| 主题 | 周转行 |
|---|---|
| api | 21 |
| artifacts | 758 |
| benchmarks | 1070 |
| ci | 18 |
| docs | 9800 |
| packaging | 47 |
| physics_core | 2588 |
| tests | 399 |
| verification | 460 |

**主题演化判读**：窗口内 15 条提交中 7 条为 merge（46.7%），真正的代码推进集中在两条：限制器（TVB/WENO）与 SIPG 椭圆求解器（matrix-free/sparse），两者都同时命中 `R5_PHYSICS_CORE` 与 `R7_LARGE_CHURN`；其余非 merge 提交是「报告/工件落盘」（`R1_DOC_ONLY` + `R3_NUMERIC_ARTIFACT`）。即演化形态是**少数深改 + 大量证据落盘**，与「有产出即写机器可读判据」的纪律一致；代价是 main 上出现 7 条无代码内容的 merge 提交，若只看 `git log --oneline` 会高估实际改动量。

### 3.4 风险清单（按等级）

| 等级 | 窗口内提交命中 | 含未并入分支 |
|---|---|---|
| HIGH | 7 | 8 |
| INFO | 11 | 11 |
| MEDIUM | 9 | 9 |

需动作的三条（按优先级）：

1. **R9（HIGH）** 未并入分支 `origin/agent/m534` 带 1534 行成熟代码（含 1 个测试文件）游离在 main 之外——合并前 main 的 CI 不会跑它，版本会与 main 的 `tensordg/` 并行漂移；

2. **R3（HIGH）** 窗口内 4 条提交改写了机器可读数值工件（`validation_report.json` 等）。工件被手改与被打算出来在 git 里不可区分，唯一验证手段是**重跑生成器**；

3. **R5（HIGH）** 3 条提交触及格式/限制器核心，精度结论不能继承旧 benchmark，必须重跑精确解对照。

### 3.5 未并入分支的精度自证（直接从该分支 blob 读取）

| 分支 | 工件 | 指标 | 观测 | 期望/理论 | 相对误差 |
|---|---|---|---|---|---|
| `origin/agent/m534` | `gci_poisson_p2.json` | apparent order (FVM/DG Richardson) | 2.99689 | 3 | 0.104% |
| `origin/agent/m534` | `gci_poisson_p2.json` | asymptotic convergence ratio | 0.125101 | 0.125269 | 0.134% |
| `origin/agent/m534` | `gci_poisson_p2.json` | GCI (fine, 21) | 1.25192 | — | —（非误差量） |
| `origin/agent/m534` | `converge_poisson_p2.json` | observed L2 order (finest triple) vs theory | 2.99883 | 3 | 0.039% |

即：该未并入分支不只是「有代码」，其 SIPG P2 的 Richardson 表观阶与渐进比都落在 3%% 以内，属于可合并的成熟成果，进一步抬高「长期不并 main」的机会成本。

## ④ 图表

![upstream delta](artifacts/upstream_delta_m1634.png)

三联图（由 `tools/plot_upstream_delta.py` 读工件绘制，标签为 ASCII：容器无 CJK 字体）：**(A)** 最近窗口内非 merge 提交的改动行数，按主导主题着色——一眼看出改动集中在限制器与 SIPG 两次深改；**(B)** 按主题汇总的周转 vs 未并入分支的 +1534 行（黑柱），后者的体量与窗口内任何单主题量级相当；**(C)** 风险标注按等级计数。图与表同源，改工件即改图。

## ⑤ 注意点与局限

- **单分支 clone 陷阱**：本工作树 `remote.origin` 只抓 `main`，`git fetch origin` 后 `refs/remotes/` 里依旧只有 main，「看不到上游分支」是假象；必须先用 `git ls-remote origin` 枚举远端真实引用，再显式 `git fetch origin refs/heads/<b>:refs/remotes/origin/<b>` 逐个拉。本单正是靠这一步才发现 `origin/agent/m534`。

- **merge 提交稀释统计**：46.7% 的窗口提交是 merge，任何「按主题计数」的结论都必须只看非 merge（本工具同时给出两种视图）。

- **主题/风险是路径启发式**，不做语义 diff：如「把物理公式从 A 挪到 B」不会触发 R5，「改注释」也会触发 R4。规则的作用是**排优先级**，不是判定正确性。

- **未并入判定只覆盖已 fetch 的远端引用**；未 fetch 的远端分支对本工具不可见，故结论要配 `git ls-remote` 交叉确认。

- **重测试未跑完**：`tests/test_euler_sod_1d.py`（5 例）与 `tests/test_vortex_2d.py`（2 例）在本容器的单次调用预算内未跑完（单例 >52 s，其中 3%% 精度门限用例尤其重），已跑部分见 ⑥；这属于执行预算限制，不代表失败，也不代表通过，需另择宽预算复核。

- **审计对象按 sha 钉死**：工件以 `head.sha`（本次审计时的 `origin/main` = `7d6f62ea044b`）为锚，本单自身在 `agent/m1634` 上的提交不计入差集；否则每写一次提交报告就自我失效（本单首版即踩此坑，`--check` 立刻报 STALE）。

- 本单是仓库治理审计，**不涉及船型几何与 Cb/L/B 校验**（非船舶总体性能求解任务），故无几何前置验证项。

## ⑥ 误差与验收判定

本文的「精度」是可数事实与 git 的一致性，用**独立 git 命令**做基准逐项比对（误差 = |工具值−git 直读| / git 直读）：

| 项 | 工具值 | git 直读（基准） | 相对误差 |
|---|---|---|---|
| window commit count | 15 | 15 | 0.000% |
| window sha set matches rev-list | 15 | 15 | 0.000% |
| non-merge commits in window | 8 | 8 | 0.000% |
| file changes in window | 99 | 99 | 0.000% |
| insertions in window | 14768 | 14768 | 0.000% |
| unmerged m534: insertions | 1534 | 1534 | 0.000% |
| unmerged m534: files touched | 16 | 16 | 0.000% |
| unmerged m534: commits ahead | 1 | 1 | 0.000% |

- **是否 <3%**：**是**，全部计数项与 git 直读一致（误差 0）（最大相对误差 0.000%）。

- **工件新鲜度**：`tools/upstream_delta.py --check artifacts/upstream_delta_m1634.json` 逐字段相等 → 报告与仓库状态零漂移（由 `tests/test_upstream_delta.py` 自动断言）。

- **测试证据**（`artifacts/m1634_verification.json`，本单实跑记录）：

| 套件 | 结果 | 用时 | 命令 |
|---|---|---|---|
| tests/test_upstream_delta.py (new, this mission) | 26 passed | 1.38 s | `python3 -m pytest -q tests/test_upstream_delta.py` |
| tests/test_advection_1d.py + test_advection_2d.py + test_heat_1d.py + test_lax_1d.py | 10 passed | 95.04 s | `python3 -m pytest -q <those four files>` |
| tests/test_burgers_1d.py | 2 passed | 9.86 s | `python3 -m pytest -q tests/test_burgers_1d.py` |
| tests/test_limiters.py | 13 passed | 6.61 s | `python3 -m pytest -q tests/test_limiters.py` |
| tests/test_poisson_2d.py | 2 passed | 8.02 s | `python3 -m pytest -q tests/test_poisson_2d.py` |
| unmerged branch origin/agent/m534: tests/test_converge.py (git archive checkout, read-only) | 22 passed | 8.14 s | `git archive origin/agent/m534 | tar -x -C /tmp/m534_check && python3 -m pytest -q tests/test_converge.py` |
| tests/test_euler_sod_1d.py | NOT_RUN (5 tests collected; single accuracy node >52 s) | >52 s per node | `python3 -m pytest -q tests/test_euler_sod_1d.py` |
| tests/test_vortex_2d.py | NOT_RUN (2 tests collected; file >105 s) | >105 s | `python3 -m pytest -q tests/test_vortex_2d.py` |

- **未并入分支精度门限**：表观阶/渐进比相对理论值最大误差 0.134% → **<3%，达标**；GCI(fine) 见 3.5。

- 无基准项已显式声明：物理精度未在本单重测（见 ⑤ 重测试预算），因此除 3.5 引用的工件事后数据外，不给出「DG 精度 <3%%」的新结论。

## 结论

1. **差集为空**：`origin/main` == `HEAD` == `7d6f62ea044b`，无本地/上游错位。

2. **发现 1 个未并入远端分支**（+1534 行成熟代码，含测试与收敛/GCI 工件，自身 22 例测试全绿、表观阶误差 <3%）——这是本单最有价值的发现，建议下一步单独立单做「合并 + 全量回归 + CI 覆盖」。

3. **窗口演化形态**：少数深改（限制器、SIPG）+ 大量证据落盘，46.7% 为 merge 提交，统计需去 merge 才是真实改动。

4. **三类 HIGH 风险**（未并入分支 / 手改数值工件 / 物理核心变更）已逐条标注可执行动作。
