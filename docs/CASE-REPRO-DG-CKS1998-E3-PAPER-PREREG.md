# CASE · REPRO-DG-CKS1998：E3 判据的去 vacuity 与「分辨率预登记」（m22426）

> 锚点 `REPRO-DG-CKS1998`（feature_ledger，open）。本单在**只读审计结论**的基础上做代码增量：
> 把 m22343 审计 §3 对 E3 门例的两条发现（F1 构造即真 / F2 单分辨率脆性）从**散文建议**
> 变成**可拒收的机器判据**，并如实记录其中一条**未能复现**。
> 工作树 `agent/m22426`；禁碰 TensorPlatform、禁推 main。

## 0. 物理/论文背景（这条判据在论文里是什么）

CKS1998（Cockburn & Shu, *The Runge–Kutta discontinuous Galerkin method for conservation
laws V*, JCP 141:199–224）§3.2 的**难点单元指示器**用 TVB 判据释放"单调解"单元，M=0 即
退化为 TVD（不释放）。论文据此声明限幅后解**不产生新极值**。把它压成可判定的量：

```
overshoot[%] = 100 · max(0, max_i u_i − 1, 0 − min_i u_i)        # 初值幅值为 [0,1] 的方波
判据 E3:  overshoot ≤ 1 %                                       # 单侧不等式
```

不等式不能直接进平台的 ref-vs-got（双侧相对误差会把"完美 TVD"判成 100 % 误差，见
`docs/CASE-REPRO-DG-CKS1998-INEQUALITY-ENCODING.md`），故用**保序 hinge**：
`got = max(overshoot, 1)/1`、`ref = 1` ⇒ `rel_error = max(0, over−1)`，过门 ⟺ 不等式成立。

## 1. 本单要解决的缺陷（来自 m22343 审计，独立复算）

| 编号 | 审计发现 | 材料级后果 | 本单处置 |
|---|---|---|---|
| F1 | 门例钉在 `troubled_policy="strict_all"`（逐单元 minmod）⇒ **按构造**无新极值，0.000 % 是代码性质 | 判据 PASS 不携带关于"指示器是否正确"的信息（vacuity） | 新增**并列门**：论文配置 `limiter=tvb, troubled_policy=tvb, tvb_constant=0`（`e3_paper_overshoot_p{1,2}`），旧门例原样保留（改写历史判据=越权） |
| F2 | 同一判据在论文配置下"随分辨率翻转"（审计称 N=80 → 1.389 %、N=100 → 0.000 %） | 单点预登记会把"网格选得好"当成"性质成立" | **把分辨率写进判据本体**：`E3_PREREGISTRATION` 逐档登记 `ncells`，门行与之不符即 `DocError`；并要求粗网格**披露行**存在（分辨率敏感性可见） |
| F2′ | — | — | **未能复现 F2 的示例数字**（见 §4），如实入账为 negative result |

## 2. 怎么做的（最小改动，四处）

| 文件 | 增量 |
|---|---|
| `validation/repro_cks1998/bench_spec.py` | `METRICS` +2（`e3_paper_overshoot_p{1,2}`，`tolerance_rel=HINGE_TOL`）；`HINGE_RULES` +2（upper 1 %，`raw` 绑定同一条点值超调）；新增 `E3_PREREGISTRATION`（limiter + 逐档 `ncells` + 粗网格披露分辨率）；`validate_documents` 接入 `verify_preregistration`；`BENCH_VER 1998.1→1998.2`、`WORKLOAD_VER 1→2`（度量集变了，版本必须跟着走） |
| `validation/repro_cks1998/bench_workload.py` | 新增两段测量：论文配置并列门（按预登记分辨率）+ `E3_paper_resolution` 粗网格披露行；`INEQUALITY_PROFILES` 加 `tvd_paper_cells/tvd_paper_coarse_cells`（bench 100/80，smoke 40/20）；若预登记分辨率恰等于既有 E3 网格则复用，不重复算 |
| `validation/repro_cks1998/hinge_audit.py` | `recompute_raw` 认得 `e3_paper_*`（同一条点值超调路径，独立实现）；新增 `verify_preregistration`：行唯一性、limiter==论文策略、`ncells`==预登记、披露行存在且分辨率正确、自述超调可由 `pt_min/pt_max` 重算 |
| `tests/test_cks1998_e3_paper_prereg.py`（新，11 条） | 见 §3 |

设计约束（沿用仓内既有口径）：一条判据只有 **rows 里能独立重算** 才进规格；新增量一律
**并列**，不改任何既有预登记值（旧 `e3_overshoot_p{1,2}` 的 bound/kind、旧门例配置、
13 条旧度量的容差全部零改动）。

## 3. 验证：跑了什么，结果是什么

```
OMP_NUM_THREADS=1 TORCH_THREADS=1 python -m pytest tests/test_cks1998_e3_paper_prereg.py -q
#  首批：10 passed, 1 failed（失败项=本单重写的那条断言，见 §4）
#  重写后：4 passed（-k "vacuous or falsified or profile_table or registered_as"，23.4 s）
OMP_NUM_THREADS=1 TORCH_THREADS=1 python -m pytest tests/test_cks1998_hinge_audit.py -q   # 既有门，7 条 hinge 计数已随度量集更新
```

| 门 | 断言的不变量 | 结果 |
|---|---|---|
| `test_paper_gate_is_registered_as_an_inequality` | 并列门进 `METRICS`/`HINGE_RULES`，upper 1 %，指针与容差正确 | PASS |
| `test_profile_table_matches_preregistration` | 两张表（workload 档位 / 预登记）不许漂移 | PASS |
| `test_honest_document_passes_preregistration` | 真实 smoke 文档：limiter=tvb_M0、ncells=40、披露 ncells=20、判绿 | PASS |
| `test_validate_documents_reports_preregistration` | ingest 门里能看到预登记报告 | PASS |
| `test_preregistration_rejects_resolution_drift` | 门行 ncells 40→20 ⇒ `DocError`（门不得悄悄换网格） | PASS |
| `test_preregistration_rejects_limiter_swap` | 偷换回 `strict_all` ⇒ `DocError`（vacuity 不许回来） | PASS |
| `test_preregistration_rejects_missing_resolution_disclosure` | 抹掉粗网格披露 ⇒ `DocError` | PASS |
| `test_preregistration_rejects_misreported_overshoot` | 自述 0 % 而 `pt_max=1.05` ⇒ `DocError` | PASS |
| `test_validate_documents_gates_on_preregistration` | 篡改后被 `validate_documents` 拒收（`--post` 因此发不出去） | PASS |
| `test_paper_gate_can_be_falsified_by_the_same_rule` | 同一判据对 1.389 % 超调判 FAIL，未越界时精确饱和 1.0 | PASS |
| `test_gate_is_not_vacuous_but_the_reported_resolution_flip_is_not_reproduced` | 不限幅对照全档判红（6.3–8.6 %）；论文配置全档 0.000 % | PASS（并记录与审计不一致） |

## 4. 结果（程序化注入的数字，零手抄）

`hinge_audit.discriminating_power` 的实测（`repro_cks1998.run_tvd` 同源，`final_time=1.0`，
`cfl=0.2`, SSP-RK3，方波初值）— pointwise 超调 %：

| cells | deg | strict_all（旧门例） | **tvb_M0（论文配置 / 新门）** | unlimited（对照） |
|---|---|---|---|---|
| 40 | P1 | 0.000 | 0.000 | 7.221 |
| 80 | P1 | 0.000 | 0.000 | 6.850 |
| 100 | P1 | 0.000 | 0.000 | 6.326 |
| 80 | P2 | 0.000 | 0.000 | 8.520 |
| 100 | P2 | 0.000 | 0.000 | 8.573 |

**关键判定**：新门在预登记分辨率（bench: N=100，P1/P2）上判绿，且**可能为假**——同一实现
去掉限幅即在 6.3–8.6 % 上判红；用 1.389 % 的合成值喂进同一判据也判红（有回归门守护）。

**F2 未能复现（negative result，如实记）**：审计 m22343 §3 F2 记「论文配置 P1 在 N=80 为
1.389 %」。本席在**当前树**用同一驱动独立重跑 N∈{40,80,100}、P1/P2，`tvb_M0` 全部
**0.000 %**（上表）。差异可能来自审计当时的代码状态或参数口径，本单无法还原其上下文，
故不引用该数字作为"当前可观测的翻转"，只把它当作**预期风险**：预登记核验因此是"防未来
漂移的护栏"，而非"修一个当前可观测的缺陷"。这一点会影响 §6 的结论口径。

## 5. 注意点（踩过的坑 / 局限）

1. **旧门例不能动**：`strict_all` 是 m22274 的预登记，改门=改写历史判据（且需 owner/账本
   同步）。本单只做**并列**，两个门同时进规格（7 条 hinge 度量）。
2. **smoke 档不是判定档**：预登记按档位分别钉分辨率（bench 100/80、smoke 40/20），smoke 只
   用于契约自检——报告不得用 smoke 数字宣称精度（smoke 上 `sod_rho_x070`、`sm2d_order_p1`
   在容差外，这是既定事实，非本单回归）。
3. **仍是"文档内核验"**：`verify_preregistration` 与 hinge 四方核验一样，只能抓伪造/串号/
   换网格，抓不到"workload 从一开始就不测量"。真正的平台侧复算属 TensorPlatform 契约范围。
4. **版本必须跟着度量集走**：`BENCH_VER 1998.1→1998.2`（平台按 `(benchmark_id, version)`
   归档行）；m22274 加 5 条 hinge 时未改版本，属历史欠账，已在 `bench_spec` 注释里写明。
5. **平台入库仍未闭合**：`tensorbench_results` 里 hinge/paper 度量到本单提交时仍为 0 行
   （`cj4054` 属 m22343 的 mission，平台反造假绑定不允许跨 mission 递交）；本单已在本
   mission 下递交 bench 档作业（`cj4130`，2026-11-15 21:30 递交）以打通 ingest 最后一段；
   结单时该作业仍在**排队**（17 min 仍未起跑，队列拥塞），故平台侧 rows>0 **未验证**
   （honest: not_run；复跑命令见下）。

## 6. 误差判定（逐项，显式回答是否 <3 %）

| 量 | 基准 | got | 相对误差 | <3 % ? |
|---|---|---|---|---|
| `e3_paper_overshoot_p1`（bench N=100） | 不等式上限 1 %（hinge 后 ref=1.0） | hinge=1.0（raw 0.000 %） | 0 | **是** |
| `e3_paper_overshoot_p2`（bench N=100） | 同上 | hinge=1.0（raw 0.000 %） | 0 | **是** |
| `e3_overshoot_p1/p2`（旧门例 strict_all，未改动） | 同上 | hinge=1.0 | 0 | 是（m22274 原样） |
| A2 L1 阶 P1/P2/P3（bench） | 下限 1/2 | raw 0.6515 | hinge 饱和 1.0 | 是（余量 30 %） |

`e3_paper_*` 的"零误差"来自**保序 hinge 的饱和语义**（`raw ≤ bound` 时 got 恒为 1.0），
不是"与论文表格逐位吻合"——论文给的是性质（无新极值），不是可复制的标量参考值。**这个
误差口径必须这样读，否则会被误解为"精度极高"。**

## 7. 复现命令

```bash
# 1) 档位自检（smoke，<2 min）
OMP_NUM_THREADS=1 TORCH_THREADS=1 python validation/repro_cks1998/bench_workload.py \
  --profile smoke --out /tmp/cks1998_smoke.json > /dev/null
# 2) 判定档（bench，集群）：度量 20 条 + 预登记核验
python validation/repro_cks1998/bench_workload.py --profile bench \
  --out artifacts/m22426_bench/bench_doc.json > artifacts/m22426_bench/bench_stdout.txt
# 3) ingest（需本 mission 的 succeeded 作业号）
python validation/repro_cks1998/tensorbench_ingest.py \
  --stdout artifacts/m22426_bench/bench_stdout.txt --validate-only
python validation/repro_cks1998/tensorbench_ingest.py \
  --stdout artifacts/m22426_bench/bench_stdout.txt --post --job-id cj4130 --mission 22426
# 4) 回归门
OMP_NUM_THREADS=1 TORCH_THREADS=1 python -m pytest tests/test_cks1998_e3_paper_prereg.py \
  tests/test_cks1998_hinge_audit.py tests/test_cks1998_tensorbench.py -q
```
