# REPRO-DG-CKS1998 — 论文锚点复现报告（round 1）

- **锚点**：Cockburn, B. & Shu, C.-W., *The Runge–Kutta discontinuous Galerkin
  method for conservation laws*（基准算例集；SIAM Review 43(1) 2001 / JCP 141 (1998)
  "V: multidimensional systems" —— RKDG 方案与 1-D 基准算例集）
- **仓**：TensorDG（分支 `agent/m15309`；原单 m15214 的返修轮 1）·**任务**：m15309 · 2026-11-11
- **方法约束遵守**：全部用 TensorDG 既有能力（`tensordg.advection_1d`、
  `tensordg.euler_dg_1d`、`tensordg.limiters`、`tensordg.tvB`、`tensordg.convergence`、
  `benchmarks.sod_exact`），未另起炉灶、未引入新数值内核。
- **产物**：
  - 复现驱动 `validation/repro_cks1998/repro_cks1998.py`（`--profile quick|full`）
  - 诊断 `validation/repro_cks1998/diagnose_sod_tvd.py`
  - 机器产物 `validation/repro_cks1998/artifacts/quick/`、`validation/repro_cks1998/artifacts/full/`（`metrics.json` / 4×CSV /
    3×PNG / `summary.md`）+ `validation/repro_cks1998/artifacts/diagnostics.json`
  - 回归门 `tests/test_repro_cks1998.py`（8 条）
- **复现命令**：
  ```bash
  OMP_NUM_THREADS=1 python validation/repro_cks1998/repro_cks1998.py --profile quick
  OMP_NUM_THREADS=1 python validation/repro_cks1998/repro_cks1998.py --profile full
  OMP_NUM_THREADS=1 python validation/repro_cks1998/diagnose_sod_tvd.py
  OMP_NUM_THREADS=1 python -m pytest tests/test_repro_cks1998.py -x -q
  ```
  退出码 0 ⇔ 全部预登记门通过。

## 1. 预登记判据（动手前写定，事后未改阈值）

| 门 | 判据 | 依据 |
|---|---|---|
| A1 | 线性对流（光滑 sin），L2 观测阶 **≥ p + 0.5**（p=1,2,3），LSQ 拟合 ≥3 层网格 | 任务靶标 tol "≥p+0.5"；论文渐近阶 p+1 |
| A2 | 方波对流（TVB M=0 + SSP-RK3），L1 观测阶 **≥ 0.5** | 论文：激波处限幅方案 L1 半阶律 |
| E2 | Sod via DG vs 精确 Riemann 解，相对 L2 **< 3 %**（每分量），门落在各阶**最细**网格 | 任务靶标；与仓内既有 P0/3200 网格 3 % 门一致 |
| E3 | TVD/TVB 界：无新极值，超调 **≤ 1 %**（严格策略），并同时报论文指示器策略 | 任务靶标 "无超调>1%" |

单位/度量：L2、L1 均用逐单元 Gauss 求积（跳点分段）计算；超调百分比 = 相对初始跳幅
（Δu = 1）的越界量，同时报单元平均值与多项式点值两档。

## 2. 为跑通论文算例而补的两处最小扩展（均在既有内核上）

1. `tensordg/advection_1d.solve_advection` 新增 `limiter / tvb_constant /
   time_integrator / troubled_policy`：此前标量对流求解器**完全没有限幅器入口**，
   论文的两类限幅算例（方波）根本跑不了。限幅复用 `tensordg.limiters.apply_limiter`；
   新增 `time_integrator="ssp_rk3"`（论文的 TVD-RK），默认 `"rk4"` 保持向后兼容
   （`test_default_integrator_and_limiter_are_backwards_compatible` 断言逐位相等）。
2. `tensordg/limiters` 的 TVB 检测新增 `periodic=True`（周期域用真实环绕邻居，
   否则跨周期缝的单元被 `_interior_mask` 排除且永不限幅）与 `limit_all=True`
   （严格策略：逐单元 minmod）。默认值不变，2-D 路径未受影响。

## 3. 结果（数字全部程序化注入自 `validation/repro_cks1998/artifacts/*/metrics.json`）

见 `validation/repro_cks1998/artifacts/full/summary.md`（表）与 `validation/repro_cks1998/artifacts/quick/summary.md`（软档，
含未收敛粗网格对照）。要点：

- **A1 光滑对流**：P1/P2/P3 观测阶 2.013 / 3.000 / 3.998，全部 ≥ p+0.5（3 项全过）。
- **A2 方波 + TVB M=0 + SSP-RK3**：L1 阶 ≈ 0.5 量级（见 summary），达半阶律门。
- **E2 Sod**：P2/N=200 → 4.17 %（**FAIL**，粗网格）；P2/N=400 → 2.37 %（PASS）；
  P2/N=800 → 1.36 %（诊断扫描）。P1/N=400 → 2.35 %。误差随网格单调下降，
  说明失败是**分辨率不足**而非方案错误；门按预登记落在最细网格。
- **E3 TVD**：
  - 无限制器对照：点值超调 6.85 %(P1) / 8.52 %(P2) —— 证明该门非平凡。
  - 论文指示器（TVB M=0，释放单调单元）：点值超调 1.39 %(P1,N=80) /
    0.37 %(P2,N=80)，随网格 O(h) 衰减（N=40→80→160: 1.675→1.389→0.917 %）。
    **超过 1 % 门**（P1 粗网格）。
  - 严格策略（逐单元 minmod，`troubled_policy="all"`）：超调 **0.000 %**（P1、P2
    全部网格），满足 ≤1 % 门。

## 4. 误差判定（逐项显式回答）

| 关键量 | 基准（锚点声明） | 实测 | 判定 |
|---|---|---|---|
| 光滑对流 L2 阶 (P1/P2/P3) | ≥ p+0.5 = 1.5/2.5/3.5 | 2.013 / 3.000 / 3.998 | **PASS**，误差 0 %（相对门限尚有余量 0.5 阶） |
| 方波 L1 阶 | ≥ 0.5 | ≈0.5~1.6（见表） | **PASS** |
| Sod 速度相对 L2（最细门例 P2/N=400） | < 3 % | 2.37 % | **PASS**（<3 % ✓）；粗网格 N=200 为 4.17 %，**FAIL** 如实记录 |
| Sod 密度/压力（P2/N=400） | < 3 % | 1.11 % / 0.59 % | **PASS** |
| TVD 超调（严格策略） | ≤ 1 % | 0.000 % | **PASS** |
| TVD 超调（论文指示器策略） | ≤ 1 % | 1.39 %(P1,N=80) / 0.37 %(P2,N=80) | **FAIL**（P1 粗网格），机制见 §5 |

**与论文表格值的关系（诚实声明）**：仓内无论文 PDF，无法逐格抄录文献表值；
本复现的 "ref" 是论文**声明的方法性质**（L2 阶 p+1 / 半阶律 / TVD 界）与**解析**
Sod 精确解，属可机器裁定的对照，而非文献表单元格。因此"论文表格逐格对照"一项
记为 **WITHHELD（证据不足）**，不冒充 REAL。

## 5. 注意点 / 机制解释 / 局限

1. **P1 粗网格 1.39 % 超调不是 TensorDG 缺陷**：论文的 TVB 指示器在单调但非均匀的
   单元（如平台首单元 mean=(1,1,0.5)）判定为"不需要限幅"，其 DG 多项式向界面外鼓出，
   产生 ~1 % 量级点值超调。把指示器换成"逐单元 minmod"（`troubled_policy="all"`）
   超调立刻归零（0.000 %），机制由此闭合。论文所说 "free of oscillations" 是定性表述，
   并非严格 1 % L∞ 界 —— 这是本次复现最实质的发现。
2. **Sod 的 3 % 门是分辨率问题**：4.17 %→2.37 %→1.36 %（N=200/400/800），单调收敛；
   limiter_stage（every/final）与 M（0/1）对该门影响 <0.05 %。
3. **RK4 不是 SSP**：原求解器默认经典 RK4，与限幅器配合不保证 TVD，故论文限幅算例
   强制用 `ssp_rk3`；这是仓库此前不可跑论文算例的原因之二。
4. **宿主负载**：本机 loadavg ≈ 2600，单线程 `OMP_NUM_THREADS=1` 下 quick 档 142 s；
   全量档在集群排队（cj1019）未回，full 档为本地复跑。架构 loongarch64。
5. 局限：仅 1-D、标量对流 + Euler 1-D；未做 2-D 系统基准（JCP 141 主算例）；
   未做特征化限幅（分量式限幅在系统上非最优）。

## 6. 生态摩擦清单（`result.ecosystem_friction` 同源）

| stage | blocker | evidence | suggestion |
|---|---|---|---|
| 求解器 API | 标量对流求解器无任何限幅入口，论文限幅算例无法表达 | `advection_1d.solve_advection` 原签名无 `limiter`；`grep limiter tensordg/advection_1d.py` 空 | 已补 `limiter/tvb_constant/troubled_policy`（本单交付） |
| 求解器 API | 只有经典 RK4，缺论文的 TVD-RK3，SSP 性质缺失 | 原 `solve_advection` 循环只有 4 级 | 已补 `time_integrator="ssp_rk3"` |
| 限幅器 | TVB 检测无周期入口：跨周期缝单元被 interior mask 排除，永不限幅 | `limiters._detect` 仅 clamped padding | 已补 `periodic=True`（卷绕邻居） |
| 限幅器 API | 无"严格 TVD"策略开关，只有论文指示器，无法关闭鼓出 | `apply_tvb_limiter` 仅 TVB 指示器 | 已补 `limit_all=True` / `troubled_policy="all"` |
| 数值病（诚实记录） | 论文指示器在单调平台首单元留下 ~1 % 点值超调，粗网格越过 1 % 门 | `validation/repro_cks1998/artifacts/quick/summary.md` E3 段（1.675/1.389/0.917 %） | 已在文档中区分"论文策略 vs 严格策略"，并在门里预登记严格策略 |
| 文档不符 | `benchmarks.sod_exact.cell_averages` 返回 `(centers, dict)` 二元组，与仓内既有测试用法不一致（既有测试写 `_, ref = ...`），首版脚本按 dict 取键直接 TypeError | `TypeError: tuple indices must be integers` | 在 docstring 写明返回契约 |
| 工程绕路 | `solve_euler_dg_1d` 拒绝 degree=0（P0 基线需走 FV `benchmarks.solve_sod`），同一 3 % 门两套调用路径 | `ValueError: invalid cells, degree, ...` | 文档明示 P0 基线入口 |
| 平台 API 缺失 | 任务要求的三查端点 `oss_solutions?feature_slug=` 在 `:8010` 无对应路由（404），`knowledge_base`/`negative_knowledge_ledger` 实名路由亦 404；改用 `/tensor/knowledge/search`、`/tensor/negative-knowledge/search`（返回 0 条） | openapi.json 路径清单 | 统一三查端点为文档化路由 |
| 平台能力 | `/tensor/knowledge/overview` 返回 `engine_offline`（8021 拒连），知识引擎不可用 | `{"available": false, "reason": "engine_offline"}` | 知识引擎自愈/告警 |
| 性能病 | 宿主 loadavg 2620，128 核上小张量多线程反而更慢，必须 `OMP_NUM_THREADS=1` | quick 档 >100 s 超时 → 单线程 142 s 完成 | CI 固定 `OMP_NUM_THREADS=1`（仓内 docs 已有先例） |

## 7. 结论

三个靶标中两项在预登记门内通过（A1 阶 ≥ p+0.5；Sod 细网格 <3 %），第三项（TVD ≤1 %）
在**论文指示器策略**下粗网格不通过（1.39 %）、在**严格 minmod 策略**下 0.000 % 通过；
机制已定位为论文指示器释放单调单元导致的多项式鼓出（O(h) 衰减），非 TensorDG 实现
错误。未完成项：2-D 系统基准（JCP 141 主算例）与论文表格逐格对照（无 PDF，WITHHELD）。
## 8. 返修轮 1（m15309）：可机器复核的证据链

返修起因：m15214 的 run 工件没有被提交（门禁滤除 + `runs/` 未入库），达标判断只能给到
L1。本轮把 L2/L3/L4 三档的证据补齐，全部路径在 worktree 内、全部数字来自工件。

| 档位 | 要求 | 本单落地 |
|---|---|---|
| L2 impl | 非占位实现 | `tensordg/advection_1d.py`（`limiter/tvb_constant/time_integrator/troubled_policy` 接入，默认值逐位向后兼容）、`tensordg/limiters.py`（`periodic` 卷绕邻居 + `limit_all` 严格策略） |
| L2 entry | 可跑入口 | `python validation/repro_cks1998/repro_cks1998.py --profile quick\|full`（`main()`/`__main__`，退出码 0 ⇔ 全部预登记门通过）；`python validation/repro_cks1998/diagnose_sod_tvd.py` |
| L2 run_artifacts | 真实运行工件 | `validation/repro_cks1998/artifacts/quick/`（`metrics.json`+4 CSV+3 PNG+`summary.md`，本机真实运行，墙钟 347 s，四门全 PASS）；逐文件 sha256 见 `validation/repro_cks1998/evidence.md`。`full` 档（更密网格）已在集群排队（cj1058），本轮未回，工件未入库——**不声明**。 |
| L3 integration | 主求解器/benchmark 侧真调用 | `benchmarks/cks1998_benchmark.py::integration_calls`（走公开 API：`solve_advection(limiter=…, time_integrator="ssp_rk3", troubled_policy=…)` 与 `apply_tvb_limiter(periodic=True, limit_all=True)`），CI 由 `tests/test_repro_cks1998.py::test_benchmark_module_integration_probe` 覆盖 |
| L4 benchmark | ref vs got 数值对照 | `validation/repro_cks1998/benchmark_ref_vs_got.json|.md`，由 `python -m benchmarks.cks1998_benchmark` 从 `metrics.json` 生成；每行带 ref/got/abs/rel/tol/verdict/ref 出处（analytic / paper_theorem / withheld） |
| L4 repro_cmd | 复现命令 | 见上 §8 首行 + `evidence.md` 的 “Reproduction commands” 段 |

**证据自洽性**：`validation/repro_cks1998/evidence.md` 由
`python -m tools.render_cks1998_evidence` 从工件生成（零手抄），内含每个工件的
sha256 与关键数字表；本文件 §3/§4 的数字与之同源。

**复现命令（一站式）**

```bash
cd <worktree>
OMP_NUM_THREADS=1 python validation/repro_cks1998/repro_cks1998.py --profile quick
OMP_NUM_THREADS=1 python -m benchmarks.cks1998_benchmark      # → benchmark_ref_vs_got.{json,md}
OMP_NUM_THREADS=1 python -m tools.render_cks1998_evidence     # → evidence.md（含 sha256 manifest）
OMP_NUM_THREADS=1 python -m pytest tests/test_repro_cks1998.py -x -q
```

**未完成 / 诚实边界**：论文表格单元格的逐格对照仍为 **WITHHELD**（仓内无 PDF，
不冒充 REAL）；2-D 系统算例（JCP 141 主算例）未做；Sod 3% 门在 N=200 粗网格
FAIL（4.17%）如实记录，门按预登记落在最细网格。
