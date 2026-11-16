# AUDIT — CASE-DG-EULER-VORTEX-ERR gate power and threshold calibration

> 锚点：`CASE-DG-EULER-VORTEX-ERR`（feature_ledger status=open）
> 分支：`agent/m22854`　证据：`case_euler_vortex_errconserve/gate_power_matrix.json`、
> `gate_threshold_calibration.json`、`metrics.json`、`fig_gate_power_audit.png`

## 1. 这份审计要回答什么

前 19 轮把 C1–C11 判成 PASS。**PASS 只说明门禁接受了真证据，不说明它会拒绝缺陷证据。**
十一个判据里有十个带手写反例测试，C4（熵预算）没有——也就是说 C4 的鉴别力一直是假设。本单用机器做三件事：

1. **杀伤力**：对每个判据注入一个真实缺陷形态，看它是否翻成 `failed`；
2. **惰性对照**：改与判据无关的字段（墙钟、迭代计数），门禁不许有任何反应；
3. **门限标定**：每个具名阈值两侧各打一针（内侧必过 / 外侧必败），证明「判据*声称*的数」就是「代码*用的*数」。

被审计对象是**已提交**的证据包（40 条 result 记录 + 19 张重放证书），不是合成夹具。

## 2. 结果

* 基线裁决复现：**PASS**；
* 杀伤力：**11/11**，`survivors = []`；
* 惰性对照：`control_inert = True`；
* 门限标定：**7/7** 咬合，`miscalibrated = []`。

## 3. 查出的阻断级缺陷（C9 设计阶混入方向轴）

**现象**：已发布 `metrics.json` 里 `dg_d2_t1` 的观测空间阶读 **4.09**，`dg_d2_t10` 读 **4.64**（本单修正后为 3.0369 / 3.6302）。DG-P2 在渐近区不可能超过 p+1=3——这个数本身就是报警。

**根因**：`_dg_order` 用最小二乘拟合 `log e` 对 `log h`，但筛选条件只钉了 CFL （`ORDER_CFL`），没有钉**对流方向**。C10 方向轴落地后，p=2 在 t=1/10 参与拟合的不再是「两个网格」，而是 `cells = [32,32,32,64]`——`diag45`/`axis0`/`obl30` 三条 32 网格腿加一条 64 网格腿。三个方向的误差在 32 网格上相差 6 倍（`diraxis0` 0.0597% vs 设计点 0.0094%），回归斜率量到的主要是**方向差**，不是网格。

**为什么这是缺陷而不是口味问题**：`_dg_order` 自己的 docstring 写明钉 CFL 的理由是「两个 CFL 在同一网格上是两个离散，拟合过这两点量到的是时间步而不是网格」——同一句话对「两个方向在同一网格上」逐字成立。方向由 C10 判，不该从 C9 后门进来。

**修复**（`benchmarks/case_euler_vortex_errconserve.py`，最小改动）：
新增 `record_direction(record)`（缺 `background_velocity` 的历史记录判为基线 `diag45`，与 LADDER 的默认一致），并把 `_dg_order` 的筛选条件加上`record_direction(r) == DIRECTION_BASELINE`，与已有的 CFL 钉子同构。

**修复前后读数**（程序化读取，零手抄）：

| 判据腿 | 修正前 cells | 修正前 order | 修正后 cells | 修正后 order | 目标 0.8(p+1) |
|---|---|---|---|---|---|
| `dg_d1_t1` | [32, 64] | 2.0424 | [32, 64] | 2.0424 | 1.60 |
| `dg_d1_t10` | [32, 64] | 2.4458 | [32, 64] | 2.4458 | 1.60 |
| `dg_d2_t1` | [32, 32, 32, 64] | 4.0928 | [32, 64] | 3.0369 | 2.40 |
| `dg_d2_t10` | [32, 32, 32, 64] | 4.6390 | [32, 64] | 3.6302 | 2.40 |

裁决**不变**（PASS，C9 四条腿在修正后仍全部达标），所以这不是把 FAIL 粉饰成 PASS 的故事，而是「数读错了、门禁也对它没反应」：修正前，把真正的设计网格细化对（`dg_d2_n32` → `dg_d2_n64`）人为压到 1 阶，C9 照样读 `passed`——它存在的那条失效路径它抓不到。修正后 `k09` 标定在 2.4 两侧咬合，`test_c9_detects_a_genuine_mesh_order_collapse` 锁死。

## 4. 门禁杀伤力矩阵与标定表


见下（由 `tools/audit_vortex_gate_power.py --markdown` 生成，本文件末尾附全矩阵）。

## 5. 关于「算例池 6% 容差覆盖到哪些读数」的诚实说明

C1 只在**设计检查点 t=1** 对全部族施加 6% 容差，最坏读数是 `p0_n64_t1_cfl0.4` 的 4.9260%（容差余量仅 17.9%）。长时检查点由 C2（累积指数）判，原始最坏值逐条留在 C1x（informational，不参与裁决）：P0 一阶交叉验证腿 t=10 读到 14.3%，这是刻意的一阶路径，不是 DG 的精度声明。本单不改这条口径，只把它写清楚：**「<6%」在本算例指的是 t=1 快照容差，不是所有检查点**。

## 6. 注意点与适用边界

* 变异都是「一个缺陷一种形态」，不能证明门禁对*任意*缺陷敏感，只能证明这 14 种形态被照见；
* 标定用 1e-3 的相对余量把探针放在阈值两侧，避免浮点噪声决定结论；
* 附带翻转（如 m02 同时翻 C7/C8/C10）是**判据耦合**，不是缺陷：设计族同时是 C2 的拟合族和 C10 的基线方向，报告照实列出；
* 本审计不重跑任何求解器（纯后处理），因此不涉及数值本身；它审的是「门禁有没有牙」。

## 7. 复现

```
PYTHONPATH=. python3 tools/audit_vortex_gate_power.py --markdown docs/AUDIT-CASE-DG-EULER-VORTEX-ERR-gate-power.md
PYTHONPATH=. python3 -m pytest -q tests/test_case_euler_vortex_errconserve_gate_power.py
```

| 变异 | 目标判据 | 注入的缺陷 | 裁决 | 目标判据读数 | 被杀 | 附带翻转 |
|---|---|---|---|---|---|---|
| `m01` | C1_tolerance | checkpoint above the 6% acceptance | FAIL | failed | 是 | C11_provenance_integrity |
| `m02` | C2_error_conservation | super-linear accumulation, design family | FAIL | failed | 是 | C10_direction_invariance, C11_provenance_integrity, C7_design_space_invariance, C8_degree_invariance |
| `m03` | C3_discrete_conservation | mass leak of 1% | FAIL | failed | 是 | 无 |
| `m04` | C4_entropy_budget | entropy budget overrun | FAIL | failed | 是 | 无 |
| `m05` | C5_crosscheck_order | cross-check path stops converging | FAIL | failed | 是 | C11_provenance_integrity, C1_tolerance |
| `m06` | C6_ladder_completeness | checkpoint evidence missing | FAIL | failed | 是 | C9_design_order |
| `m07` | C7_design_space_invariance | growth off the design point | FAIL | failed | 是 | C11_provenance_integrity, C9_design_order |
| `m08` | C8_degree_invariance | growth at a non-design degree | FAIL | failed | 是 | C7_design_space_invariance |
| `m09` | C9_design_order | refinement stops buying accuracy | FAIL | failed | 是 | C11_provenance_integrity, C7_design_space_invariance |
| `m10` | C10_direction_invariance | growth on a rotated direction | FAIL | failed | 是 | C7_design_space_invariance |
| `m11` | C11_provenance_integrity | certificate no longer reproduces | FAIL | failed | 是 | 无 |
| `m12` | C11_provenance_integrity | legacy record left unverified | INCOMPLETE | not_evaluated | 是 | 无 |
| `c01` | —（对照） | control: timing metadata | PASS | — | — | 无 |
| `c02` | —（对照） | control: iteration counters | PASS | — | — | 无 |

### 全判据 x 全变异状态矩阵

| 判据 | 基线 | m01 | m02 | m03 | m04 | m05 | m06 | m07 | m08 | m09 | m10 | m11 | m12 | c01 | c02 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1_tolerance | passed | failed | passed | passed | passed | failed | passed | passed | passed | passed | passed | passed | passed | passed | passed |
| C2_error_conservation | passed | passed | failed | passed | passed | passed | passed | passed | passed | passed | passed | passed | passed | passed | passed |
| C3_discrete_conservation | passed | passed | passed | failed | passed | passed | passed | passed | passed | passed | passed | passed | passed | passed | passed |
| C4_entropy_budget | passed | passed | passed | passed | failed | passed | passed | passed | passed | passed | passed | passed | passed | passed | passed |
| C5_crosscheck_order | passed | passed | passed | passed | passed | failed | passed | passed | passed | passed | passed | passed | passed | passed | passed |
| C6_ladder_completeness | passed | passed | passed | passed | passed | passed | failed | passed | passed | passed | passed | passed | passed | passed | passed |
| C7_design_space_invariance | passed | passed | failed | passed | passed | passed | passed | failed | failed | failed | failed | passed | passed | passed | passed |
| C8_degree_invariance | passed | passed | failed | passed | passed | passed | passed | passed | failed | passed | passed | passed | passed | passed | passed |
| C9_design_order | passed | passed | passed | passed | passed | passed | not_evaluated | failed | passed | failed | passed | passed | passed | passed | passed |
| C10_direction_invariance | passed | passed | failed | passed | passed | passed | passed | passed | passed | passed | failed | passed | passed | passed | passed |
| C11_provenance_integrity | passed | failed | failed | passed | passed | failed | passed | failed | passed | failed | passed | failed | not_evaluated | passed | passed |

### 门限标定（阈值两侧各一针）

| 标定 | 判据 | 声明阈值 | 来源 | 失败方向 | 内侧读数 | 外侧读数 | 咬合 |
|---|---|---|---|---|---|---|---|
| `k01` | C1_tolerance | 6.000e-02 | `TOLERANCE` | fails above | 5.994e-02 → passed | 6.006e-02 → failed | 是 |
| `k02` | C2_error_conservation | 1.000e+00 | `MAX_GROWTH_EXPONENT` | fails above | 9.990e-01 → passed | 1.001e+00 → failed | 是 |
| `k03` | C3_discrete_conservation | 1.000e-10 | `INVARIANT_TOLERANCE` | fails above | 9.990e-11 → passed | 1.001e-10 → failed | 是 |
| `k04` | C4_entropy_budget | 1.000e-03 | `ENTROPY_TOLERANCE` | fails above | 9.990e-04 → passed | 1.001e-03 → failed | 是 |
| `k05` | C5_crosscheck_order | 8.000e-01 | `0.8 (literal in evaluate_gates)` | fails below | 8.008e-01 → passed | 7.992e-01 → failed | 是 |
| `k09` | C9_design_order | 2.400e+00 | `ORDER_SLACK*(p+1), p=2` | fails below | 2.402e+00 → passed | 2.398e+00 → failed | 是 |
| `k11` | C11_provenance_integrity | 1.000e-09 | `PROVENANCE_REPLAY_TOLERANCE` | fails above | 9.990e-10 → passed | 1.001e-09 → failed | 是 |
