# 交差审计 · m15309 REPRO-DG-CKS1998（TensorDG）→ L3

- **被审对象**：任务 m15309（分支 `agent/m15309`，worktree `/root/tp-agent/work/m15309`，HEAD `0615576`，
  commit message: `m15309: CKS1998 reproduction evidence chain`）
- **锚点**：`REPRO-DG-CKS1998`（Cockburn & Shu 1998/2001，RKDG 基准集）
- **审计任务**：m15345（board-audit 交差核验，非重做）
- **本文性质**：交差核验（取证 → 判级 → 落表），不改被审代码
- **裁判规则（任务单给定）**：`tensor_benchmark_cases` 是唯一裁判；报告好看不算数

## 0. 结论摘要

| 项 | 结论 |
|---|---|
| 三件套（run_artifacts / ref-vs-got / repro_cmd） | **真存在**，且 sha256 9/9 与交付 manifest 一致，可复现 |
| 判据数值 | A1/A2/E3 三项与**审计侧独立重算逐位一致**；Sod 参考解经**独立 Toro 求解器**复核（max \|Δ\| = 1.9e-5，即单元均值 vs 点值差）|
| **基准裁判** | **0 行**（`anchor_slug='REPRO-DG-CKS1998'`），`tensorbench_results` 0 行，`GET /tensor/bench/l4-verdict/15309` = `CLAIMED_NOT_PROVEN` |
| 判级 | **L3（系统集成）** —— L2/L3 实证达标；L4 因裁判未登记 + 门例有条件而不开证 |
| 缺陷 | 1 中 1 低（文档声明了未交付的 `artifacts/full/` 与 `diagnostics.json`；ref-vs-got 表把一条越界行标 `informative` 而非 `FAIL`） |

## 1. 基准裁判实况（决定级别上限）

以 postgres 身份直查平台库（任务单指定查询）：

```
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='REPRO-DG-CKS1998';
→ (0 rows)

SELECT count(*) FROM tensorbench_results WHERE mission_id IN (15309,15214,15292);
→ 0

GET /api/v1/tensor/bench/l4-verdict/15309
→ {"status":"CLAIMED_NOT_PROVEN","l4_benchmark_pass":false,"rows":0,
   "failed_checks":["execution_success","source_cluster_job","cluster_job_bound",
                    "ref_got_error_present","within_tolerance"],
   "reason":"no tensorbench_results row for this mission: the L4 benchmark claim is self-declared, not measured"}
```

全库 `tensor_benchmark_cases` 仅 15 行，含 `REPRO-FVM-SOD1978`（id 15）而**无任何 TensorDG 行**。
即：m15292 审计（报告 14372）指出的"裁判未登记"缺口，**m15309 未闭合**——它把 ref-vs-got
做进了仓内（`validation/repro_cks1998/benchmark_ref_vs_got.json`），但没有把算例登记进
`tensor_benchmark_cases`，因此机器裁判面仍是 0 判据。

## 2. 三件套取证（真指针）

| 件 | 指针 | 核验结果 |
|---|---|---|
| impl_files | `tensordg/advection_1d.py:96` `solve_advection(..., limiter, tvb_constant, time_integrator, troubled_policy)`；`:126-129` 参数校验；`:150` 逐级限幅闭包；`:153` `ssp_rk3` 三级分支；`:79/:91` `_limit(periodic=True, limit_all=...)`；`tensordg/limiters.py:91-92` `periodic/limit_all` 形参、`:103-121` 卷绕邻居与严格 minmod | **REAL**（有默认值向后兼容分支，被 `test_default_integrator_and_limiter_are_backwards_compatible` 断言逐位相等） |
| entry | `validation/repro_cks1998/repro_cks1998.py:main()`（`--profile quick\|full --outdir`，退出码 0 ⇔ 全门通过）；`benchmarks/cks1998_benchmark.py:integration_calls/main`（`--probe-only`） | **REAL，可执行**（审计实跑，见 §4） |
| run_artifacts | `validation/repro_cks1998/artifacts/quick/{metrics.json, summary.md, A1_smooth.csv, A2_discontinuous.csv, E2_sod.csv, E3_tvd.csv, a1_l2_order.png, e2_sod.png, e3_limiter.png}`（**已入库**，`git ls-files` 14 个文件含 9 件产物） | **存在且完整**：9/9 sha256 与 `validation/repro_cks1998/evidence.md` manifest 逐字节一致（`artifacts/audit_m15345/gate_recheck.json.artifact_integrity`） |
| tests | `tests/test_repro_cks1998.py`（11 条） | **审计实跑 `pytest tests/test_repro_cks1998.py -q` → 11 passed in 166.95s**（含 Sod N=400 慢测）。先前 m15292 记的 not_run 已闭合 |
| integration_callers | `tensordg/__init__.py:3` 导出 `solve_advection`；仓内调用方 `benchmarks/advection_1d_highorder.py:78,110,125,145,151`、`benchmarks/audit_gate_power.py:61` 等 ≥10 处；`tests/test_repro_cks1998.py::test_benchmark_module_integration` 走公开 API 探针 | **REAL**（库级集成，非平台挂载） |
| benchmark | `validation/repro_cks1998/benchmark_ref_vs_got.json`（54 行：PASS 43 / FAIL 0 / informative 11）+ `.md` | 仓内自建对照**存在且有出处标注**，但**未登记进平台裁判表**（§1）→ 不能构成 L4 证书 |
| repro_cmd | `OMP_NUM_THREADS=1 python validation/repro_cks1998/repro_cks1998.py --profile quick`；`python -m benchmarks.cks1998_benchmark [--probe-only]`；`pytest tests/test_repro_cks1998.py -x -q` | **可执行**（§4 实跑） |
| external_anchor | Cockburn & Shu, SIAM Review 43(1) 2001 / JCP 141 (1998)；Sod 精确解（本审计用独立 Toro 求解器复核） | 锚点条款（≥p+0.5 / Sod<3% / 无超调>1%）与仓内预登记 `gates` 一致 |

## 3. 逐项核 got vs 容差（审计侧独立重算，不采信交付的 verdict 块）

重算脚本 `artifacts/audit_m15345/recompute_gates.py`（仅用 numpy，**不 import tensordg**），
输出 `artifacts/audit_m15345/gate_recheck.json`。

| 锚点条款 | 容差 | 交付 got | **审计独立重算** | 判定 |
|---|---|---|---|---|
| A1 光滑对流 L2 阶 P1/P2/P3 | ≥ p+0.5 = 1.5/2.5/3.5 | 2.013152 / 2.999764 / 3.997689 | **2.013152 / 2.999764 / 3.997689**（逐位相同） | PASS（余量 ≈0.5 阶） |
| A2 方波 L1 阶 P1/P2/P3 | ≥ 0.5 | 0.667656 / 0.664468 / 0.664059 | **0.667656 / 0.664468 / 0.664059** | PASS |
| E2 Sod 相对 L2（门例 = 最细网格 P2/N=400） | < 3 % | 2.368 %（ρ1.11/p0.59） | 由交付 CSV 复核：2.368 % | PASS（门例） |
| E2 Sod 粗网格 P2/N=200 | < 3 % | **4.173 %（越界）** | 4.173 %（CSV 复核） | **FAIL（如实登记于 docs §4、报告 ③ 表）** |
| E3 TVD 严格策略（`troubled_policy="all"`） | ≤ 1 % | 0.000 %（P1/P2） | 0.000 %（CSV 复核） | PASS |
| E3 TVD 论文指示器（TVB M=0） | ≤ 1 % | **P1 1.389 %** | 1.389 %（CSV 复核） | **FAIL（非门例配置，已如实登记）** |

**参考解真实性**：本审计用自写 Sod 精确 Riemann 求解器（Toro 2009 ch.4 sampling，域内
x=0.5 膜片，γ=1.4，L=(1,0,1)/R=(0.125,0,0.1)）逐站复算 `metrics.json` 的 36 个 ref 值，
**max |Δ| = 1.89e-5**，且差异只出现在跨波扇单元——正是"交付 ref 为精确解单元均值、
本审计为点值"的量级，说明 ref 不是手填/拟合表。星状态 p\*=0.3031302、u\*=0.9274526 与
Sod 1978 / Toro 公开值一致到 3e-7。

一致性判断：交付 `metrics.json.verdict` 四项布尔**可由 raw CSV 独立复算得到**，非手抄。

## 4. 复现性（审计实跑）

| 命令 | 结果 |
|---|---|
| `python -m benchmarks.cks1998_benchmark --probe-only` | `probe_pass=true`；L2 阶梯 `[0.0042185, 0.0010601, 0.00026534]`、`observed_order=1.9954`、`l1_error=0.0730074` 与交付 `benchmark_ref_vs_got.json.integration_probe` **逐位相同** |
| `pytest tests/test_repro_cks1998.py -q` | **11 passed in 166.95s**（含 N=400 Sod、半阶律；m15292 的 not_run 项闭合） |
| `repro_cks1998.py --profile quick --outdir runs/audit_m15345/quick_local` | **审计重跑完成**：307.5 s，`verdict.all_pass=true`；`metrics.json` 的**全部数值字段与交付工件逐位相同**（仅每步墙钟 `seconds` 不同），Sod 站点表完全一致 → 交付工件与 HEAD `0615576` 同源，非旧版残留（m15292 的 B1 已修复：quick 档现已含 `strict_all` 行）。产物见 `runs/audit_m15345/quick_local/` |

## 5. 缺陷（按严重度；均不推翻主结论）

**M1（中 · 文档声明了未交付的产物）**
`docs/repro_cks1998.md:13-14` 声明交付 `validation/repro_cks1998/artifacts/full/`
与 `validation/repro_cks1998/artifacts/diagnostics.json`，`docs/repro_cks1998.md:50` 又让读者
"见 artifacts/full/summary.md"；但 `git ls-files validation/repro_cks1998` **只有 quick/ 9 件**，
工作树内 `artifacts/full/` 与 `artifacts/diagnostics.json` **均不存在**。
同源问题：`tests/test_repro_cks1998.py:5` docstring 与 `docs` 引用 `runs/cks1998/`，该目录在本
worktree 只有空壳 `runs/cks1998/quick/`（`find runs -type f` 无文件）。
→ 这是 m15292 审计 B2 的**未修复复发**（同一行声明，同一缺失）。属"声明了未交付"，
建议补跑 full 档/诊断脚本入库，或把这些行改成"本地未入库"。

**M2（中低 · 对照表把一条越界行标成 informative）**
`benchmark_ref_vs_got.json` 中 `E2_sod_p2_n200 / rel L2 error, velocity = 0.041734`
**超过该行自述容差 `< 3%`**，`verdict` 却是 `informative`（docs §4 与报告 ③ 同一数值标 **FAIL**）。
且 `benchmarks/cks1998_benchmark.py:180` 有 `assert n_fail == 0`、PASS/FAIL 只对 strict-TVD 行
给出，故这张表的 FAIL 列**结构上不可能非空**，"43 PASS / 0 FAIL" 的抬头把一次真实越界藏进了
informative。→ 建议：容差列与 verdict 解耦（越界一律 FAIL），"门例选择"用 `gated: true/false`
另列；标题改报"门例 PASS 43 / 越界 1（非门例）"。

**L1（低 · 对照表 6 行 ref 为空）**
6 条聚合误差行（rel L2 三项 × 两网格）`ref=None`，表头却写 "ref vs got"。
→ 建议 ref 填容差阈值（0.03）或标 `threshold`，否则这 6 行无法判读"ref"含义。

**L2（低 · 渠道口径）**
报告 14411 与 docs §3 的 A1 阶 2.013/3.000/3.998 是 **quick 档**值，docs 却挂全文引
"数字全部注入自 artifacts/*/metrics.json"（full 档不存在，见 M1）。数值均在门内，属口径混用。

## 6. 级别裁定（L0-L5）

- **L2 独立功能实现 · 达标**：`solve_advection` 有限幅/SSP-RK3 真分支，`limiters` 有
  `periodic/limit_all` 真逻辑；默认路径向后兼容由测试逐位断言（非桩、非硬编码）。
- **L3 系统集成 · 达标**：公开 API 导出（`tensordg/__init__.py:3`）+ 仓内 ≥10 个调用方 +
  公开 API 探针测试；论文的限幅算例可经库 API 直接表达。锚点三项靶标有完整算例、误差/收敛
  分析与逐位可复现工件。
- **L4 基准验证 · 不开证（阻断项）**：①机器裁判 0 判据（§1：`tensor_benchmark_cases` 0 行、
  `tensorbench_results` 0 行、`l4-verdict` = CLAIMED_NOT_PROVEN）；②即便有裁判，"通过"仍依赖
  门例选择——Sod 只在 N≥400 过（N=200 = 4.17 % 越界），TVD 只在严格 minmod 过（论文指示器
  P1 = 1.389 % 越界）。两条都**诚实登记**（docs §4 + 报告 ③），属"条件达标"而非无条件达标。
- **L5 · 不适用**：无新方法/新物理，是既有框架上的基准复现（发现"论文 TVB 指示器不是 1 %
  L∞ 界"是实质结论，但不构成新算法）。

**升 L4 的机械化路径（给后继单，非本单职责）**：在 `tensor_benchmark_cases` 为
`REPRO-DG-CKS1998` 登记 3 条（A1 阶 `>=p+0.5`、Sod `metric` 内写明网格 N=400 `<0.03`、
TVD `metric` 内写明 `strict_all` `<=0.01`），并由平台 tensorbench 产出 ≥1 条 `tensorbench_results`
（ref/got/within_tolerance 三列），即可由机器裁判替代自证。

## 7. 审计未做（honest not_run）

- **full 档（`--profile full`）未审计重跑**：交付的 `artifacts/full/` 根本不存在（M1），
  本审计只重跑 quick 档；docs 中 full 档数字（如 Sod 1.36 %/N=800 诊断扫描、A1 2.004/3.000/4.000）
  在本 worktree **无工件可核**，记为 **WITHHELD**。
- `validation/repro_cks1998/diagnose_sod_tvd.py` 未运行（其产物 `artifacts/diagnostics.json`
  同样不在树内）。
- 未做 2-D 系统基准（JCP 141 主算例）：锚点条款未要求，mission 自陈为局限。
- 论文 PDF 表值逐格对照：**WITHHELD**（仓内无 PDF，交付方也如此声明，未冒充 REAL）。

## 8. 复现本审计

```bash
python artifacts/audit_m15345/recompute_gates.py     # 独立重算 + 工件 sha256 复核
python artifacts/audit_m15345/make_figures.py        # 图 audit_m15345_checks.png
# 被审对象侧：
cd /root/tp-agent/work/m15309
OMP_NUM_THREADS=1 python -m pytest tests/test_repro_cks1998.py -q          # 11 passed
OMP_NUM_THREADS=1 python -m benchmarks.cks1998_benchmark --probe-only      # probe_pass
OMP_NUM_THREADS=1 python validation/repro_cks1998/repro_cks1998.py --profile quick
```