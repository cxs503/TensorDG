# 交差审计：m15214 REPRO-DG-CKS1998（TensorDG）

- 审计任务：m15292（board-audit）· 对象：m15214（status=succeeded，merge commit `958bacf`，"Merge #275 (m15214) via push-merge"）
- 锚点：`tensor_repro_papers` id=8，paper_key=CKS1998，anchor_slug=`REPRO-DG-CKS1998`，target_repo=TensorDG
- 审计方式：读产物 + 用 raw CSV **独立重算**判据（不导入 `tensordg`）+ 重跑 quick 档复现 + 复核参考解真实性
- **结论：L3（系统集成）confirmed 建议；L4 的实质证据存在但平台侧证书缺失（0 行基准裁判），故不申报 L4。**

## 1. 基准裁判的原状（决定级别上限的硬约束）

| 裁判面 | 查询 | 结果 |
|---|---|---|
| 基准算例登记 | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='REPRO-DG-CKS1998'` | **0 行**（全表仅 14 行，无 TensorDG 任何行） |
| 平台复算结果 | `SELECT count(*) FROM tensorbench_results WHERE mission_id=15214` | **0 行** |
| 平台评测 | `tensorbench_evaluations` mission_id=15214 | **0 行** |
| 目标 API 裁定 | `GET /api/v1/tensor/bench/l4-verdict/15214` | `CLAIMED_NOT_PROVEN`，`rows=0`，failed_checks=`execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance`，"the L4 benchmark claim is self-declared, not measured" |

**判定**：任务单要求的"对照 tensor_benchmark_cases 逐项核 got 是否在容差内"在机器面**无判据可核**——不是 got 不达标，而是**裁判未登记**。按"基准裁判是唯一裁判"的纪律，L4（基准验证）本期不能开证。锚点合同本身是有的（`tensor_repro_papers` id=8，三条 target：`≥p+0.5` / Sod `<3%` / TVD `无超调>1%`），故按合同条款做**审计侧独立重算**替代。

## 2. 三件套取证（真存在 / 真指针）

| 件 | 指针 | 状态 |
|---|---|---|
| impl_files | `tensordg/advection_1d.py:96` `solve_advection(... limiter, tvb_constant, time_integrator, troubled_policy)`（新增限幅入口 + `ssp_rk3` 分支 `L150-160`）；`tensordg/limiters.py:_detect(... periodic, limit_all)` | REAL：`git diff 0324686 f7bde21` 为真实逻辑（SSP-RK3 三级 + 逐级限幅；周期环绕邻居；严格 minmod 开关），默认值不变、向后兼容由测试断言 |
| entry | `validation/repro_cks1998/repro_cks1998.py:419 main()`，`--profile quick|full --outdir`；退出码 0 ⇔ 全门通过 | REAL，可执行（本审计实跑） |
| run_artifacts | 工作树 `/root/tp-agent/work/m15214/runs/cks1998/full/{metrics.json,summary.md,A1_smooth.csv,A2_discontinuous.csv,E2_sod.csv,E3_tvd.csv,3×PNG}`（sha256 见 `artifacts/audit_m15292/MISSION_RUN_SHA256SUMS`） | 存在，但**未入库**：merge commit 只含 6 个代码/文档文件（"22 个文件被门禁滤除"），仓内 `git ls-tree` 无 runs/ |
| tests | `tests/test_repro_cks1998.py` 8 条 | 审计侧重跑：`pytest -k "not sod and not half_order"` → **6 passed in 11.02s**；2 条慢测（N=400 Sod、半阶律）与集群作业 cj1045 在本单截止时仍 queued，**未跑，记 not_run** |
| integration | `tensordg/__init__.py:3` 导出 `solve_advection`；仓内 `grep -l solve_advection` ≥10 个 benchmarks 调用方 | REAL（库级集成，非平台级挂载） |

## 3. 逐项核 got vs 容差（审计侧重算，不采信 metrics.json 的 verdict 块）

重算脚本 `artifacts/audit_m15292/recompute_gates.py`，输出 `gate_recheck_full.json`（full 档）+ 本轮 quick 重跑 `runs/audit_m15292/quick/`。

| 锚点条款 | 容差 | got（full 档，审计重算） | 判定 |
|---|---|---|---|
| 线性对流 L2 阶 p=1/2/3 | ≥ p+0.5 = 1.5/2.5/3.5 | 最差逐对阶 **2.0002 / 2.9998 / 3.9998** | **PASS**（余量 ≥0.5 阶，几乎等于理论 p+1） |
| 方波限幅 L1 半阶（附加项） | ≥0.5 | **0.6637~0.6693**（P1/P2/P3） | **PASS** |
| Sod via DG vs 精确解 | <3%（每分量） | P2/N=400 → worst **2.37%**；P1/N=400 → 2.35%；**P2/N=200 → 4.17% 越界** | 细网格 **PASS**；粗网格**FAIL**（报告已在 §4/§6 如实登记） |
| TVD 无超调 | ≤1% | 严格 minmod（`troubled_policy="all"`）：P1/P2/P3 全 **0.000%**；**论文指示器（TVB M=0）：P1 1.4036% 越界**、P2 0.779%、P3 0.445% | 严格策略 **PASS**；**论文指示器策略 FAIL（P1）** |

交叉一致性：`metrics.json.verdict` 与我的重算**逐项一致**（`verdict_consistent=true`），即产物的 verdict 块不是伪造/手填，四项布尔都能从 raw CSV 复算出来。

**参考解真实性**：`benchmarks/sod_exact.star_state()` → p\*=0.303130178、u\*=0.927452620；与 Sod 1978 / Toro 2009 公开星状态 p\*=0.30313、u\*=0.92745 符合到 3e-7。`cell_averages(200,0.2)` 密度值域 [0.125, 1.0]，符合初值，非硬编码剖面 → 参考值 REAL。

**可复现性（本审计实跑）**：`OMP_NUM_THREADS=1 python validation/repro_cks1998/repro_cks1998.py --profile quick --outdir runs/audit_m15292/quick` → 246.2 s，verdict 全 True。与 m15214 交付的 quick 档逐位对照：

| 量 | 交付 quick | 审计重跑 | 一致 |
|---|---|---|---|
| A1 阶 p1/p2/p3 | 2.0132 / 2.9998 / 3.9977 | 2.0132 / 2.9998 / 3.9977 | **逐位相同** |
| A2 L1 阶 p1/p2/p3 | 0.6677 / 0.6645 / 0.6641 | 同 | **逐位相同** |
| Sod P2/N=200 worst | 4.1734% | 4.1734% | **相同** |
| E3 TVB M=0 点值超调 P1/P2 | 1.3885% / 0.3704% | 1.3885% / 0.3704% | **相同** |

## 4. 发现的缺陷（按严重度；均不推翻主结论，但需返修）

**B1（中，provenance）交付的 `runs/cks1998/quick/` 是旧版脚本产物**：其 `E3_tvd.cases` 只有 `{p1,p2}×{unlimited,tvb_M0}`，**没有 `strict_all` 行**，故 quick 档 `verdict.E3_tvd_no_new_extrema=false`、`all_pass=false`，与报告"严格策略 0.000% 通过"不一致；用 HEAD 脚本重跑则必出 `strict_all` 行（本审计重跑已验证）。→ 入库的 quick 档应与 HEAD 代码同源，否则读者拿到的 `all_pass=false` 会与报告口径冲突。

**B2（中，引用完整性）`docs/repro_cks1998.md` §产物 声明 `runs/cks1998/diagnostics.json`，该文件不存在**（`ls` 无、`find runs` 无）；写它的 `validation/repro_cks1998/diagnose_sod_tvd.py:101` 也从未在本工作树成功落盘。→ 要么补跑该诊断脚本，要么从产物清单删除该行（当前属"声明了未交付"）。

**B3（低，口径）文档 §3 的 A1 阶 2.013/3.000/3.998 是 quick 档值**（full 档为 2.004/3.000/4.000），§3 却挂在 full 表旁边引用；平台报告 14353 摘要同样引用 quick 值。→ 数值都在门内，但口径混用，建议报告显式标注 profile。

**B4（低，产物未入库）runs/ 全档未 merge**（闸门滤除 22 文件），仓内只剩代码；复现依赖重跑（full 档 444.6 s、quick 246 s）。→ 建议把 `summary.md + metrics.json`（几十 KB）作为 curated evidence 入 `docs/` 或 `artifacts/`，避免每个审计者都要重跑。

**B5（低，报告）**：tensor_reports id=14353 六段齐全（①-⑥）、md_len=2912，但 figures 只内嵌 2 张（A1/E3），`e2_sod.png` 已存在未嵌入；① 中 Sod 左态写作四元组 `(ρ,u,p)_L=(1,0,0,1.0)`（多一个分量）笔误。

## 5. 级别裁定（L0-L5）

- **L2 独立功能实现**：达标——`solve_advection` 有限幅/SSP-RK3 真分支，`limiters._detect` 有 periodic/limit_all 真逻辑，非桩。
- **L3 系统集成**：达标——公开 API 导出（`tensordg/__init__.py:3`），仓内 ≥10 个 benchmarks 调用方，paper 算例可经库 API 直接表达。
- **L4 基准验证**：**实质证据齐（完整算例+误差/收敛分析+可复现），但证书不齐**：①平台基准裁判 0 行、`tensorbench_results` 0 行、`l4-verdict` = CLAIMED_NOT_PROVEN；②锚点两条款的"通过"依赖门例选择（Sod 只在 N≥400 过、N=200 4.17% 越界；TVD 只在严格 minmod 过、论文指示器 P1 1.404% 越界）——两者都**诚实登记**，但不是无条件达标。故**不申报 L4**。
- **L5**：不适用（无新方法/新物理，属既有框架上的基准复现）。

**给出 L3（不是把好活判死）**：本单在 L4 的实质内容（含理论阶 p+1 到小数点后三位、精确 Riemann 参考解、逐位可复现）上属于本仓最强证据之一；降级只因机器裁判面缺登记。**升 L4 的机械化路径**：在 `tensor_benchmark_cases` 为 `REPRO-DG-CKS1998` 登记 3 条（A1 阶 / Sod 相对 L2 / TVD 超调，tolerance 分别为 `>=p+0.5` / `<0.03` / `<=0.01`），并由平台 tensorbench 跑出 ≥1 条 `tensorbench_results`（ref/got/within_tolerance 三列），同时把 Sod 门例网格（N=400）与 TVD 策略（strict / paper-indicator 分列）写进 `metric` 字段避免口径歧义。

## 6. 审计未做（honest not_run）

- `pytest tests/test_repro_cks1998.py` 全 8 条的完整结果：仅 6 条实跑（11.02 s），2 条慢测（Sod N=400、半阶律）与集群作业 **cj1045（TensorDG@本地 stage，queued，截止时未派发）** 未回 → 慢测 PASS 未由本审计独立确认。
- full 档全量重跑（444.6 s）未做；仅做 quick 档逐位复现 + full 档 CSV 重算。
- 未做 2-D 系统基准（JCP 141 主算例）——锚点合同未要求，mission 自陈为局限。

## 7. 复现命令

```bash
# 审计侧重算（不导入 tensordg）
python artifacts/audit_m15292/recompute_gates.py \
  --run /root/tp-agent/work/m15214/runs/cks1998/full \
  --json artifacts/audit_m15292/gate_recheck_full.json
# 审计侧图
python artifacts/audit_m15292/audit_plot.py \
  --run /root/tp-agent/work/m15214/runs/cks1998/full \
  --out artifacts/audit_m15292/audit_gates.png
# 交付复现（本审计实跑 246.2 s，verdict 全 True）
OMP_NUM_THREADS=1 python validation/repro_cks1998/repro_cks1998.py --profile quick \
  --outdir runs/audit_m15292/quick
# 回归门（6/8 实跑通过，2 条慢测未跑）
OMP_NUM_THREADS=1 pytest tests/test_repro_cks1998.py -q
```