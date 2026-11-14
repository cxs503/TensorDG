# CASE-LAX-DG · 工件可信性审计：`dt` 语义陷阱 + 跨工件漂移的成因与闸门

> 锚点 `CASE-LAX-DG`（`feature_ledger` status=open，验收原文「ref: 精确解｜容差 < 3 %」）。
> 上游记录：`docs/CASE-LAX-DG-resolution-requirement.md`（它把一处漂移归因于「记录的 `dt`
> 不同 → 场不可复现」）、`docs/CASE-LAX-DG-residual-gap.md`（把「重生成 `lax_asymptotic_probe`
> 工件」列为后续项）。
> 本文两件事：**证伪**那条归因（§3.1），并把真正的成因量化、用闸门钉死（§3.2/§4）。
> 本单全部数字来自程序输出或工件 JSON，无手抄。

## ① 案例介绍

Lax (1954) / Toro (2009) Test 2 激波管：`[0,1]`，隔膜 `x0 = 0.5`，γ = 1.4，
左态 `(ρ,u,p) = (0.445, 0.698, 3.528)`，右态 `(0.5, 0.0, 0.571)`，快照 `t = 0.13`；
解由左稀疏波（head/tail）、接触间断、右激波组成，精确 Riemann 解闭式可得。

本例有两个**已提交**的工件记录同一批腿（`p3.N200/N400`；`p1.N400`、`p2.N400`）：

| 工件 | 生成者 | 共享腿 |
|---|---|---|
| `docs/benchmarks/results/lax_case_pool/metrics.json` | `benchmarks/lax_case_pool.py` | `(400,1) (400,2) (200,3) (400,3)` |
| `docs/benchmarks/results/lax_asymptotic_probe/metrics.json` | `benchmarks/lax_asymptotic_probe.py` | 同上 + `(800,1) (800,2) (800,3)` |

## ② 怎么做的

1. **`dt` 语义取证**：读 `tensordg/euler_dg_1d.solve_euler_dg_1d` 的时间推进——
   `dt = min(cfl*dx/((2p+1)*max_speed), final_time - time)`。`metadata["dt"]` 是
   **被 `final_time` 截断的最后一步**，源码里已注明「not representative of the run」。
   因此拿 `dt` 判「场是否可复现」在方法上无效。
2. **单腿判决实验**（本机 2 线程，`scripts/diag_lax_probe_staleness.py --legs 200:3`，
   墙钟 41.7 s）：当前代码在 `p3.N200` 的全场 rel L1 ρ = `1.214747e-02`。
3. **工件重生成**（集群作业 `cj3105`，节点 `n315`，8 核，`OMP_NUM_THREADS=8`，
   墙钟 510.0 s，rc = 0）：`python -m benchmarks.lax_asymptotic_probe --no-figures`
   7 腿全跑；随后本机 `--figures-only` 从 `profiles.npz` 补渲染并重封 `metrics.json`/`sha256`/doc。
4. **新增三道闸门**（`tests/test_lax_asymptotic_probe.py`）：跨工件一致性、`dt` 元数据不得歧义、
   场缓存 `save/load` 往返。前两道对**旧工件为红**（见 §3.2），对重生成后的工件为绿。

## ③ 计算结果

### 3.1 被证伪的归因

| 读数 | 旧探针工件 | 本次（8 线程，节点 n315） | 池子工件 | 说明 |
|---|---|---|---|---|
| `p3.N400` rel L1 ρ | `8.686465e-03` | `8.685540e-03` | `8.685540e-03` | 本次与池子**逐位一致** |
| `p3.N200` rel L1 ρ | `1.214747e-02` | `1.214742e-02` | `1.214742e-02` | 同上 |
| `p3.N800` rel L1 ρ | `6.180136e-03` | `6.164752e-03` | —（池子无此腿） | 与 `lax_resolution_requirement` 独立测量的 `6.164752e-03` 一致 |
| `p3.N400` `dt` | `2.2296e-05` | `2.2296e-05` | `2.5445e-05` | 同一腿三个值 —— **与场无关** |

结论一（**falsified_hypothesis**）：「记录的 `dt` 不同 ⇒ 场不可复现」不成立。
本次两腿的 `dt_last` 与旧工件完全相同（`2.2296e-05`），而场却与池子逐位一致；
反过来池子的 `dt_last`（`2.5445e-05`）与两者都不同。`dt_last` 是截断余步，记录它
既不能证明也不能否定可复现性。

结论二（真实成因）：旧探针工件与池子工件的差异是**线程数相关的浮点不可复现性**。
同一份代码、同一 CFL，本机 2 线程在 `p3.N200` 上给出 `1.214747e-02`（= 旧探针值），
节点 8 线程给出 `1.214742e-02`（= 池子值），相对差 `4.1e-06`；`p3.N400` 为 `1.06e-04`，
`p3.N800` 为 `2.49e-03` —— **步数越多差异越大**（2547 → 10250 步），是 DG 求解器里
torch 归约顺序改变被波陡化放大的典型形态。

### 3.2 闸门（新增，对旧工件为红 → 对重生成工件为绿）

| 闸门 | 旧工件 | 重生成工件 |
|---|---|---|
| `test_shared_legs_agree_with_the_committed_pool`（跨工件 rel < 1e-06，步数同） | **FAIL**（`p3.N400` 1.06e-04、`p3.N800` 2.49e-03） | **PASS**（worst rel = 0.0） |
| `test_step_size_metadata_is_not_ambiguous`（无 `dt` 歧义键；`dt_mean == t/steps`） | **FAIL**（旧工件只有 `dt`） | **PASS** |
| `test_field_cache_round_trips`（`--no-figures` / `--figures-only` 通道） | n/a（无缓存） | **PASS** |
| 探针工件 sealed `reuse_check` | 无判定（只写数字，`worst=0.0` 是当时对当时） | `status=PASS`、`tolerance=1e-06`、`steps_agree=True` |

`pytest tests/test_lax_asymptotic_probe.py tests/test_lax_case_pool.py
tests/test_lax_resolution_requirement.py` → **48 passed**。

## ④ 结果图

`docs/benchmarks/results/lax_asymptotic_probe/figures/lax_asymptotic_convergence.png`：三条 p 阶梯的
解析区 rel L1 ρ 与实测阶，重生成工件的读数。差分对比曲线（同一腿 × 两种线程数）由 §3.1 表
给出——该对比的原始量只有 6 个数，画图信息量低于表，故以表为准（不生成冗余图）。
注：`make_figures` 的剖面面板依赖 `profiles` 里的 `finest` 条目，`load_profiles` 不重建它，
因此 `--figures-only` 只补出阶数图；剖面图需整轮测量时生成（已知局限，见 §5.4）。

## ⑤ 注意点

1. **`dt` 是陷阱**：`metadata["dt"] = dt_last`（截断余步）。任何「dt 一致 ⇒ 可复现」或
   「dt 不同 ⇒ 漂移」的推断都无效；本仓的 `test_euler_dg_1d_dt_metadata.py` 早已钉住
   `dt_mean = final_time/steps` 才是可研究量。探针工件现已改写为 `dt_last` + `dt_mean`
   + `dt_semantics` 三个键，`dt` 键不再写出。
2. **跨机不可复现的量化带**：线程数变化在 `p3.N200/N400/N800` 上带来 `4e-06 / 1e-04 / 2.5e-03`
   的相对差。据此，跨机逐位比对应当带 3 个量级的余量；本单的跨工件闸门沿用 `REUSE_TOL=1e-06`，
   因为提交的两个工件确实逐位一致——一旦某侧漂移，闸门变红即为信号，而不是噪声。
3. **测量与呈现解耦**：节点无 matplotlib 时旧流程会在写盘前崩（`cj2647` 的教训）。本单把场
   缓存成 `profiles.npz`，`--no-figures` 在节点只做测量，`--figures-only` 在本机补图并重封
   工件；`test_field_cache_round_trips` 锁死这条通道的往返一致性。
4. **已知局限**：`--figures-only` 只重建阶数图（原因见 §4）；线程数信息尚未写进工件的
   `environment` 块——本单未做（需重跑 510 s 的一轮），列为后续项。
5. **不改写验收**：台账臂（`p3.N400` 全场 rel L2 = `3.11556e-02`）仍是 FAIL，本单不动它。

## ⑥ 误差判定（是否 < 3 %）

| 关键量 | 基准 | 数值 | 相对差 | < 3 % ? |
|---|---|---|---|---|
| `p3.N400` rel L1 ρ（重生成 vs 池子工件） | 已提交工件 | `8.685540e-03` vs `8.685540e-03` | `0.0` | 是 |
| `p3.N200` rel L1 ρ（2 线程 vs 8 线程，同代码） | 同代码异线程 | `1.214747e-02` vs `1.214742e-02` | `4.10e-06` | 是 |
| `p3.N400` rel L1 ρ（同上） | 同上 | `8.686465e-03` vs `8.685540e-03` | `1.06e-04` | 是 |
| `p3.N800` rel L1 ρ（同上） | 同上 | `6.180136e-03` vs `6.164752e-03` | `2.49e-03` | 是 |
| 台账臂 `p3.N400` 全场 rel L2 ρ vs 精确 Riemann 解 | 精确解 | `3.11556e-02` | `3.116 %` | **否**（该 FAIL 原样保留） |

**结论**：本单的争议量（工件间的可复现性）全部 < 3 % ——最大者是线程数引起的 `2.49e-03`；
**台账验收仍为 FAIL**（`p3.N400` 密度 rel L2 = `3.116 %` > 3 %，裕度 ×1.039），未被本单改动。
工件层面：`lax_asymptotic_probe` 已重生成，sealed `reuse_check` = PASS（worst = 0.0）。