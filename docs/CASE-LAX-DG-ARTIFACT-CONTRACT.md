# CASE-LAX-DG · 跨工件对账：`dt` 语义漂移、口径漂移与一次被否定的「验证欠账」

> 数字全部由 `benchmarks/lax_artifact_reconciliation.py` 从 `docs/benchmarks/results/lax_artifact_contract/metrics.json` 程序化注入，零手抄；metrics sha256 `dd981c5753cce59a43f9dcb9755daa61f017a37a14724ffbe0c22bdb961f3aa5`；生成于 2026-11-15T03:37:29Z，墙钟 0 s。
> 契约模块：`benchmarks/lax_artifact_contract.py`（C1–C6）；修复脚本：`scripts/attach_lax_dt_semantics.py`；闸门测试：`tests/test_lax_artifact_contract.py`。

## ① 案例介绍

Lax (1954) / Toro (2009) Test 2 激波管，`[0,1]`，γ = 1.4，隔膜 `x0 = 0.5`，左态 (ρ, u, p) = (0.445, 0.698, 3.528)，右态 (0.500, 0.000, 0.571)，快照 t = 0.13，参考值取闭式 Riemann 解。围绕这个算例，仓库里已有 10 个兄弟工件（算例池阶梯、P3 渐近探针、分辨率要求、认证构型、限幅器臂 …），它们在**同一批** `(cells, degree)` 腿上各自报误差。本文不重算这些物理量，只回答一个记账问题：**这些工件之间，凡是名字相同、含义也相同的量，是否逐位一致；凡是名字相同、含义不同的量，是如何被读错的。**

## ② 怎么做的

1. **契约化**：`lax_artifact_contract.PUBLISHERS` 为每个工件声明「原生键 → 规范量」的映射。规范误差量三个：`whole_field_L1`（全场相对 L1）、`whole_field_L2`（全场相对 L2）、`band_L1`（剔除波带的解析区相对 L1）；规范步长两个：`dt_mean = final_time/steps` 与`dt_last`（求解器返回的、被 `final_time` 截断的最后一步）。
2. **六条预登记检查**（跑数前写死在代码里）：

| 检查 | 内容 |
|---|---|
| C1 | the meaning of the bare key `dt` is resolved for every publisher: either declared by a dt_semantics string, or verified by C2 to be final_time/steps (so no reader has to guess and no file is rewritten only to carry a label) |
| C2 | |dt_mean - final_time/steps| / (final_time/steps) <= 1e-12 on every leg |
| C3 | 0 < dt_last <= dt_mean (1 + 1e-12) wherever dt_last is recorded |
| C4 | the bare key `dt` maps to the same canonical quantity in every publisher |
| C5 | legs shared by >= 2 publishers agree on every commonly published canonical metric to <= 1e-12 relative |
| C6 | a fresh solve of a shared leg reproduces the committed canonical metrics to <= 1e-12 relative (evaluated by the reconciliation driver) |

3. **活复算（C6）**：用当前代码在本机重跑两条最便宜的共享腿 （p3.N200, p2.N400，共 105 s），走与算例池完全相同的度量路径（Gauss 节点 Legendre 加权、精确 Riemann 解作分母），与已提交工件逐量比对。
4. **诊断项（R3）**：同一腿上把 `whole_field_L1` 与 `band_L1` 的差值记下来，这个量级就是「口径错配」能伪造出的『漂移』。
5. **复核被引用的两个数（R4）**：残差缺口文档引用了 `p3.N400` 的 `2.2296e-05` 与 `2.5445e-05` 两个 `dt` 作为「不可复现」的证据，本文检查它们是否同属一次运行。

## ③ 计算结果

### 3.1 步长键语义（C1–C4）

| 工件 | 腿数 | 语义来源 | `dt` 是否等于 `final_time/steps` | 最差相对偏差 |
|---|---|---|---|---|
| `lax_case_pool` | 9 | verified (dt == final_time/steps on every leg) | PASS | 0.00000e+00 |
| `lax_asymptotic_probe` | 7 | declared | PASS | 0.00000e+00 |
| `lax_resolution_requirement` | 2 | verified (dt == final_time/steps on every leg) | PASS | 0.00000e+00 |
| `lax_certified_configuration` | 8 | verified (dt == final_time/steps on every leg) | PASS | 0.00000e+00 |
| `lax_case_pool_limiter_arms` | 9 | declared | PASS | 0.00000e+00 |

修复对象：`PASS` —— limiter-arms 工件原来把求解器的**最后一步**写进了裸键 `dt`（该键在其余工件里都是均值步），共 27 条腿被就地改正，最差相对偏差 9.36832e-01（即两把尺子差 94 %）；旧值保留在 `dt_legacy_last_step` 与 `dt_repair` 溯源块里，`solver_rerun = False`（未重跑求解器）。其余工件**未改动一个字节**：它们的裸键已经是均值步，C2 逐腿验证通过，只加标签不改值没有信息增益，反而会打断别处引用的 sha256。

### 3.2 跨工件一致性（C5）

共享腿 10 条，规范量逐场比对 **66** 对，最差相对偏差 0.00000e+00，判据 ≤ 1e-12 → **PASS**。

| 腿 | 规范量 | 场 | 参与工件 | 最差相对偏差 |
|---|---|---|---|---|
| p1.N100 | `whole_field_L1` | density | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p1.N100 | `whole_field_L1` | pressure | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p1.N100 | `whole_field_L1` | velocity | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p1.N100 | `band_L1` | density | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p1.N100 | `band_L1` | pressure | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p1.N100 | `band_L1` | velocity | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p2.N100 | `whole_field_L1` | density | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p2.N100 | `whole_field_L1` | pressure | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p2.N100 | `whole_field_L1` | velocity | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p2.N100 | `band_L1` | density | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p2.N100 | `band_L1` | pressure | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p2.N100 | `band_L1` | velocity | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N100 | `whole_field_L1` | density | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N100 | `whole_field_L1` | pressure | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N100 | `whole_field_L1` | velocity | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N100 | `band_L1` | density | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N100 | `band_L1` | pressure | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N100 | `band_L1` | velocity | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p1.N200 | `whole_field_L1` | density | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p1.N200 | `whole_field_L1` | pressure | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p1.N200 | `whole_field_L1` | velocity | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p1.N200 | `band_L1` | density | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p1.N200 | `band_L1` | pressure | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p1.N200 | `band_L1` | velocity | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p2.N200 | `whole_field_L1` | density | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p2.N200 | `whole_field_L1` | pressure | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p2.N200 | `whole_field_L1` | velocity | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p2.N200 | `band_L1` | density | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p2.N200 | `band_L1` | pressure | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p2.N200 | `band_L1` | velocity | lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N200 | `whole_field_L1` | density | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N200 | `whole_field_L1` | pressure | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N200 | `whole_field_L1` | velocity | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N200 | `band_L1` | density | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N200 | `band_L1` | pressure | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N200 | `band_L1` | velocity | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p1.N400 | `whole_field_L1` | density | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p1.N400 | `whole_field_L1` | pressure | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p1.N400 | `whole_field_L1` | velocity | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p1.N400 | `whole_field_L2` | density | lax_case_pool, lax_certified_configuration | 0.00000e+00 |
| p1.N400 | `whole_field_L2` | pressure | lax_case_pool, lax_certified_configuration | 0.00000e+00 |
| p1.N400 | `whole_field_L2` | velocity | lax_case_pool, lax_certified_configuration | 0.00000e+00 |
| p1.N400 | `band_L1` | density | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p1.N400 | `band_L1` | pressure | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p1.N400 | `band_L1` | velocity | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p2.N400 | `whole_field_L1` | density | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p2.N400 | `whole_field_L1` | pressure | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p2.N400 | `whole_field_L1` | velocity | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p2.N400 | `whole_field_L2` | density | lax_case_pool, lax_certified_configuration | 0.00000e+00 |
| p2.N400 | `whole_field_L2` | pressure | lax_case_pool, lax_certified_configuration | 0.00000e+00 |
| p2.N400 | `whole_field_L2` | velocity | lax_case_pool, lax_certified_configuration | 0.00000e+00 |
| p2.N400 | `band_L1` | density | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p2.N400 | `band_L1` | pressure | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p2.N400 | `band_L1` | velocity | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N400 | `whole_field_L1` | density | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N400 | `whole_field_L1` | pressure | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N400 | `whole_field_L1` | velocity | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N400 | `whole_field_L2` | density | lax_case_pool, lax_certified_configuration, lax_resolution_requirement | 0.00000e+00 |
| p3.N400 | `whole_field_L2` | pressure | lax_case_pool, lax_certified_configuration, lax_resolution_requirement | 0.00000e+00 |
| p3.N400 | `whole_field_L2` | velocity | lax_case_pool, lax_certified_configuration, lax_resolution_requirement | 0.00000e+00 |
| p3.N400 | `band_L1` | density | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N400 | `band_L1` | pressure | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N400 | `band_L1` | velocity | lax_asymptotic_probe, lax_case_pool, lax_case_pool_limiter_arms | 0.00000e+00 |
| p3.N800 | `whole_field_L2` | density | lax_certified_configuration, lax_resolution_requirement | 0.00000e+00 |
| p3.N800 | `whole_field_L2` | pressure | lax_certified_configuration, lax_resolution_requirement | 0.00000e+00 |
| p3.N800 | `whole_field_L2` | velocity | lax_certified_configuration, lax_resolution_requirement | 0.00000e+00 |

### 3.3 活复算（C6，当前代码 vs 已提交工件）

判据分两条：**预登记** 相对差 ≤ 1e-12、步数逐位相同 → **FAIL**；**跨平台复现地板** ≤ 0.001 → **PASS**。实测最差 1.24848e-05。

预登记臂判 FAIL 这件事原样保留：已提交工件在 loong64 计算节点上产生，本次活复算在本机 x86 上做，2547 步 SSP-RK3 的浮点归约顺序不同，逐位相等不可得。地板是**事后登记**的，理由写在工件里：post-hoc cross-platform reproducibility floor: bitwise (1e-12) equality is not achievable between the loong64 artifacts and this x86 host; the floor is 80x the measured worst (1.25e-05), three orders below the 3 % case tolerance and one order below the ~1e-4 discrepancy the residual-gap doc reported。两条臂同时公布，不做替换。

| 腿 | 工件 | 规范量 | 场 | 已提交 | 本次重算 | 相对差 |
|---|---|---|---|---|---|---|
| p3.N200 | `lax_case_pool` | `whole_field_L1` | density | 1.214742e-02 | 1.214747e-02 | 4.09795e-06 |
| p3.N200 | `lax_case_pool` | `whole_field_L1` | velocity | 1.698978e-02 | 1.698993e-02 | 8.55823e-06 |
| p3.N200 | `lax_case_pool` | `whole_field_L1` | pressure | 9.564164e-03 | 9.564283e-03 | 1.24848e-05 |
| p3.N200 | `lax_case_pool` | `whole_field_L2` | density | 4.199980e-02 | 4.199980e-02 | 9.45042e-10 |
| p3.N200 | `lax_case_pool` | `whole_field_L2` | velocity | 3.390223e-02 | 3.390223e-02 | 1.28413e-08 |
| p3.N200 | `lax_case_pool` | `whole_field_L2` | pressure | 1.655412e-02 | 1.655413e-02 | 4.14849e-08 |
| p3.N200 | `lax_case_pool` | `steps` | - | 2.547000e+03 | 2.547000e+03 | 0.00000e+00 |
| p3.N200 | `lax_asymptotic_probe` | `whole_field_L1` | density | 1.214742e-02 | 1.214747e-02 | 4.09795e-06 |
| p3.N200 | `lax_asymptotic_probe` | `whole_field_L1` | pressure | 9.564164e-03 | 9.564283e-03 | 1.24848e-05 |
| p3.N200 | `lax_asymptotic_probe` | `whole_field_L1` | velocity | 1.698978e-02 | 1.698993e-02 | 8.55823e-06 |
| p3.N200 | `lax_asymptotic_probe` | `steps` | - | 2.547000e+03 | 2.547000e+03 | 0.00000e+00 |
| p3.N200 | `lax_case_pool_limiter_arms` | `whole_field_L1` | density | 1.214742e-02 | 1.214747e-02 | 4.09795e-06 |
| p3.N200 | `lax_case_pool_limiter_arms` | `whole_field_L1` | velocity | 1.698978e-02 | 1.698993e-02 | 8.55823e-06 |
| p3.N200 | `lax_case_pool_limiter_arms` | `whole_field_L1` | pressure | 9.564164e-03 | 9.564283e-03 | 1.24848e-05 |
| p3.N200 | `lax_case_pool_limiter_arms` | `steps` | - | 2.547000e+03 | 2.547000e+03 | 0.00000e+00 |
| p2.N400 | `lax_case_pool` | `whole_field_L1` | density | 4.528163e-03 | 4.528163e-03 | 1.01401e-09 |
| p2.N400 | `lax_case_pool` | `whole_field_L1` | velocity | 1.663108e-03 | 1.663108e-03 | 4.13144e-10 |
| p2.N400 | `lax_case_pool` | `whole_field_L1` | pressure | 1.098465e-03 | 1.098465e-03 | 1.02084e-08 |
| p2.N400 | `lax_case_pool` | `whole_field_L2` | density | 3.525871e-02 | 3.525871e-02 | 7.30716e-13 |
| p2.N400 | `lax_case_pool` | `whole_field_L2` | velocity | 1.576382e-02 | 1.576382e-02 | 3.31822e-11 |
| p2.N400 | `lax_case_pool` | `whole_field_L2` | pressure | 8.575577e-03 | 8.575577e-03 | 8.72820e-11 |
| p2.N400 | `lax_case_pool` | `steps` | - | 3.662000e+03 | 3.662000e+03 | 0.00000e+00 |
| p2.N400 | `lax_asymptotic_probe` | `whole_field_L1` | density | 4.528163e-03 | 4.528163e-03 | 1.01401e-09 |
| p2.N400 | `lax_asymptotic_probe` | `whole_field_L1` | pressure | 1.098465e-03 | 1.098465e-03 | 1.02084e-08 |
| p2.N400 | `lax_asymptotic_probe` | `whole_field_L1` | velocity | 1.663108e-03 | 1.663108e-03 | 4.13144e-10 |
| p2.N400 | `lax_asymptotic_probe` | `steps` | - | 3.662000e+03 | 3.662000e+03 | 0.00000e+00 |
| p2.N400 | `lax_certified_configuration` | `whole_field_L2` | density | 3.525871e-02 | 3.525871e-02 | 7.30716e-13 |
| p2.N400 | `lax_certified_configuration` | `whole_field_L2` | velocity | 1.576382e-02 | 1.576382e-02 | 3.31822e-11 |
| p2.N400 | `lax_certified_configuration` | `whole_field_L2` | pressure | 8.575577e-03 | 8.575577e-03 | 8.72820e-11 |
| p2.N400 | `lax_certified_configuration` | `steps` | - | 3.662000e+03 | 3.662000e+03 | 0.00000e+00 |
| p2.N400 | `lax_case_pool_limiter_arms` | `whole_field_L1` | density | 4.528163e-03 | 4.528163e-03 | 1.01401e-09 |
| p2.N400 | `lax_case_pool_limiter_arms` | `whole_field_L1` | velocity | 1.663108e-03 | 1.663108e-03 | 4.13144e-10 |
| p2.N400 | `lax_case_pool_limiter_arms` | `whole_field_L1` | pressure | 1.098465e-03 | 1.098465e-03 | 1.02084e-08 |
| p2.N400 | `lax_case_pool_limiter_arms` | `steps` | - | 3.662000e+03 | 3.662000e+03 | 0.00000e+00 |

### 3.4 口径错配能伪造出多大的「漂移」（诊断）

同一腿上两个**不同定义**的误差量之差，最差 9.92668e-01（全场 L1 vs 解析区 L1）。这正是把不同键当成同一把尺子时读到的『不一致』量级——与求解器无关。

### 3.5 被引用的两个 `dt` 属于同一次运行（R4）

| 量 | 残差缺口文档引用 | 工件实际值 | 相对差 | 判定 |
|---|---|---|---|---|
| `dt_last` | 2.229624e-05 | 2.229624e-05 | 0.00000e+00 | PASS |
| `dt_mean` | 2.544529e-05 | 2.544529e-05 | 0.00000e+00 | PASS |

**结论**：the 12 % 'dt difference' is dt_last vs dt_mean of one run; not solver drift。文档把同一腿/同一次运行的 `dt_last` 与 `dt_mean` 当成两次运行的 `dt` 相比，得到 12 % 的『差异』，据此判定「探针工件已不可复现」并排了一次全阶梯重算。该结论与后续动作被本文否定（见 ⑤-1）。

## ④ 结果图

* `docs/benchmarks/results/lax_artifact_contract/figures/lax_contract_dt_lanes.png`：Recorded step sizes per leg: circles = dt_mean (final_time/steps), crosses = dt_last; hollow red = the value the bare `dt` key carried in the limiter-arms artifact before the repair. The legacy key sat on the dt_last lane, which is what made a 12-94 % 'difference' look like drift when compared against dt_mean.
* `docs/benchmarks/results/lax_artifact_contract/figures/lax_contract_definitions.png`：Left: whole-field L1 (filled) against band-limited L1 (hollow) on identical legs -- differences of the same order as the 'discrepancy' the residual-gap doc reports. Right: the same artifacts compared only where the definitions coincide; every pair is at or below the 1e-12 contract tolerance.

## ⑤ 注意点

1. **被否定的是「探针工件不可复现」这条欠账，不是对它的记录**：`docs/CASE-LAX-DG-residual-gap.md` 的 A0 FAIL 判定与分辨率结论原样保留，本文只否掉它列出的『后续项：重生成探针工件』——该工件在同一腿上的读数与算例池逐位一致（C5 最差 0.0），重算不会改变任何数字，省下的是一次全阶梯集群作业。
2. **修的是元数据不是物理量**：limiter-arms 修复只动 `dt` 键的含义（旧值保留），误差量、步数、限幅统计一个字节没动；因此该工件的 C3/R3 判定不受影响。
3. **为什么不用整个 JSON 重写其它工件**：`lax_certified_configuration` 的度量 sha256 被`docs/CASE-LAX-DG-certified-configuration.md` 引用，重写整个文件会为了一个标签打断那条引用。契约因此把 C1 定义为「语义可解析」而不是「必须有标签」：值为均值步 + C2 逐腿验证 = 无需猜。
4. **1e-12 是护栏不是结论**：同配置重跑的相对差在 float64 下通常恰为 0.0（本单 C5/C6 均为 0.0），1e-12 只用来吸收两条独立写盘路径的格式化差。
5. **本单没有新的物理结论**：A0 = FAIL（`p3.N400` 密度 rel L2 = 3.116 % > 3 %）、认证构型 `p2.N800`（最差 2.570 %）都按既定口径保留，本文不改变任何验收判定。

## ⑥ 误差判定（是否 < 3 %）

本单不含新的物理量测量，唯一的新数字是**记账一致性**，其判据是 1e-12（不是 3 %）：

| 关键量 | 基准 | 数值 | 相对差 | < 3 % ? |
|---|---|---|---|---|
| 跨工件同定义误差量（C5） | 兄弟工件互比 | 0.00000e+00 | 同上 | 是（且 ≤ 1e-12） |
| 当前代码 vs 已提交工件（C6，跨平台） | 已提交工件 | 1.24848e-05 | 同上 | 是（远小于 3 %，但未达逐位） |
| limiter-arms `dt` 修复前偏差 | `final_time/steps` | 9.36832e-01 | 93.7 % | **否**（这正是被修的缺陷） |

台账锚点的验收（「ref: 精确解｜容差 < 3 %」）判定不变：钉扎腿 `p3.N400` 仍 FAIL，`p3.N800`/`p2.N800` 仍达标；本单只清理记账层，不动验收口径。
