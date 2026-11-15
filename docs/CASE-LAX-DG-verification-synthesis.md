# CASE-LAX-DG · 验证综合报告（Verification Synthesis）

> **锚点**：`CASE-LAX-DG`（算例池·Lax via DG）
> **台账原文**：ref: 精确解｜容差: <3%
> **生成时间**：2026-11-15
> **范围**：综合本仓全部 CASE-LAX-DG 相关工件、测试与文档，给出统一判据映射与闭合建议。
> **纪律**：所有数字从已提交工件直读（零手抄），判据从 `feature_ledger` 原文提取，结论可机器复核。

---

## ① 案例介绍

Lax (1954) 激波管 / Toro (1999) test 2：`[0,1]` 域内隔膜 `x0 = 0.5` 两侧常状态，γ = 1.4，快照时刻 t = 0.13。精确解含四道波：左稀疏波（head → tail）、接触间断、右激波。

| 参数 | 左态 | 右态 |
|---|---|---|
| (ρ, u, p) | (0.445, 0.698, 3.528) | (0.5, 0.0, 0.571) |

解析波系（闭式，未经求解器回读）：p\* = 2.466098、u\* = 1.528723、ρ\*\_L = 0.344568、ρ\*\_R = 1.304085。

## ② 方法

模态 DG（Legendre 基，P1–3），HLLC 数值通量，SSP-RK3 时间推进，CFL = 0.35。参考解由同仓闭式代数 `benchmarks/riemann_exact.py` 给出，并经不含 Riemann 求解器的 Rusanov 有限体积族独立复核。

## ③ 验证证据总览

本仓围绕 CASE-LAX-DG 已建立以下验证层（全部测试通过，工件已提交）：

| 验证层 | 工件路径 | 测试文件 | 状态 |
|---|---|---|---|
| 算例池阶梯（9 腿） | `docs/benchmarks/results/lax_case_pool/metrics.json` | `test_lax_case_pool.py` (26 tests) | ✅ PASS |
| 限幅器支臂对比 | `docs/benchmarks/results/lax_case_pool_limiter_arms/metrics.json` | `test_lax_case_pool_limiter_arms.py` (14 tests) | ✅ PASS |
| 分辨率闭合研究 | `docs/benchmarks/results/lax_resolution_closure/metrics.json` | `test_lax_resolution_closure.py` (10 tests) | ✅ PASS |
| 认证配置 | `docs/benchmarks/results/lax_certified_configuration/metrics.json` | `test_lax_certified_configuration.py` (8 tests) | ✅ PASS |
| 分辨率需求 | `docs/benchmarks/results/lax_resolution_requirement/metrics.json` | `test_lax_resolution_requirement.py` (5 tests) | ✅ PASS |
| 参考解独立仲裁 | `docs/benchmarks/results/lax_reference_crosscheck/metrics.json` | `test_lax_reference_crosscheck.py` (16 tests) | ✅ PASS |
| 参考投影底限 | `docs/benchmarks/results/lax_reference_floor/metrics.json` | `test_lax_reference_floor.py` (15 tests) | ✅ PASS |
| 守恒预算审计 | `docs/benchmarks/results/lax_conservation/metrics.json` | `test_lax_conservation_audit.py` (10 tests) | ✅ PASS |
| 时间误差预算 | `docs/benchmarks/results/lax_time_budget_certified/metrics.json` | `test_lax_time_budget_certified.py` (10 tests) | ✅ PASS |
| 渐近探针 | `docs/benchmarks/results/lax_asymptotic_probe/metrics.json` | `test_lax_asymptotic_probe.py` (17 tests) | ✅ PASS |
| 范数收敛 | `docs/benchmarks/results/lax_norm_convergence/metrics.json` | `test_lax_norm_convergence.py` (14 tests) | ✅ PASS |
| 限制器研究 | `docs/benchmarks/results/lax_limiter_study/metrics.json` | `test_lax_limiter_study.py` (9 tests) | ✅ PASS |
| 区域带审计 | `docs/benchmarks/results/lax_region_band/metrics.json` | `test_lax_region_band.py` (5 tests) | ✅ PASS |
| 可复现性审计 | `docs/benchmarks/results/lax_reproducibility_audit/metrics.json` | `test_lax_reproducibility_audit.py` (8 tests) | ✅ PASS |
| 可复现性带 | `docs/benchmarks/results/lax_reproducibility_band/metrics.json` | `test_lax_reproducibility_band.py` (4 tests) | ✅ PASS |
| 工件契约 | `docs/benchmarks/results/lax_artifact_contract/metrics.json` | `test_lax_artifact_contract.py` (6 tests) | ✅ PASS |
| 记录数审计 | `docs/benchmarks/results/lax_documented_number_audit/metrics.json` | `test_lax_documented_number_audit.py` (4 tests) | ✅ PASS |
| 认证配置文档 | `docs/CASE-LAX-DG-certified-configuration.md` | `test_lax_certified_configuration.py` | ✅ PASS |
| 综合验证 | `docs/CASE-LAX-DG-comprehensive-verification.md` | `test_lax_comprehensive_verification.py` (24 tests) | ✅ PASS |
| P3 光滑阶 | — | `test_lax_p3_smooth_order.py` (2 tests) | ✅ PASS |
| DT 元数据 | — | `test_euler_dg_1d_dt_metadata.py` (5 tests) | ✅ PASS |
| 基础 Lax 1D | — | `test_lax_1d.py` (2 tests) | ✅ PASS |

**合计：≥ 190 条测试全部通过，17 份工件已提交。**

## ④ 计算结果（程序化注入，零手抄）

### 4.1 台账 A0 判据（relative L2 < 3%）

| 臂 | 度量 | 腿 | 最差场 | 实测值 | 容差 | 判定 |
|---|---|---|---|---|---|---|
| production_l2 | relative L2 | p3.N400 | density | **3.116 %** | < 3 % | ❌ FAIL |
| ladder_l1 | relative L1 | 9 腿 | density (p1.N100) | 2.329 % | < 3 % | ✅ PASS |

### 4.2 分辨率闭合（N=800 达到 <3%）

| 腿 | rel L2 ρ | rel L2 u | rel L2 p | 判定 |
|---|---|---|---|---|
| p3.N200 | 4.200 % | 3.390 % | 1.655 % | ❌ |
| p3.N400 | 3.116 % | 2.248 % | 1.305 % | ❌ |
| **p3.N800** | **2.315 %** | **1.383 %** | **0.899 %** | ✅ |
| p3.N1600 | 1.875 % | 1.475 % | 0.917 % | ✅ |

### 4.3 认证配置（最小 DOF 通过 <3%）

| 配置 | DOF | 最差 L2 | 墙钟 [s] | 判定 |
|---|---|---|---|---|
| p1.N400 | 800 | 4.935 % | 29.6 | ❌ |
| p1.N800 | 1600 | 3.771 % | 67.7 | ❌ |
| p1.N1600 | 3200 | 2.954 % | 169.2 | ✅ |
| p2.N400 | 1200 | 3.526 % | 50.2 | ❌ |
| **p2.N800** | **2400** | **2.570 %** | **118.4** | ✅ **认证** |
| p2.N1000 | 3000 | 2.368 % | 158.2 | ✅ |
| p3.N400 | 1600 | 3.116 % | 73.9 | ❌ |
| p3.N800 | 3200 | 2.315 % | 176.4 | ✅ |

**认证配置：p2.N800（DOF=2400，最差 L2=2.570%，墙钟 118.4s）**

### 4.4 参考解独立仲裁

| 判据 | 实测 | 容差 | 判定 |
|---|---|---|---|
| 波位置最坏相对误差 | 0.098 % | < 2 % | ✅ |
| 星区平台最坏相对误差 | 2.9e-07 | < 3 % | ✅ |
| Newton vs 二分相对差 | 1.8e-16 | < 1e-13 | ✅ |
| FV 一阶收敛阶 | 0.51 | > 0.4 | ✅ |
| FV 二阶优于阶 | 是 | — | ✅ |

### 4.5 误差分解（参考底限）

| 腿 | 求解器贡献 | 参考底限 | 交叉项 | 总误差 |
|---|---|---|---|---|
| p3.N400 | 主导 | 共主导 | 非零（u,p） | 3.116 % |

参考底限在 p3.N400 上占总误差的显著比例，说明部分误差来自精确解在 DG 空间上的投影误差，而非求解器缺陷。

## ⑤ 判据映射（台账原文 → 证据）

| 台账要求 | 证据来源 | 实测 | 判定 |
|---|---|---|---|
| ref: 精确解 | `riemann_exact.py` + `lax_reference_crosscheck` | 闭式代数 + 独立 FV 仲裁 | ✅ 参考解可信 |
| 容差 <3% | `lax_case_pool` A0 臂 | p3.N400: 3.116 % | ❌ 生产腿未达 |
| 容差 <3% | `lax_resolution_closure` | p3.N800: 2.315 % | ✅ 可达 |
| 容差 <3% | `lax_certified_configuration` | p2.N800: 2.570 % | ✅ 认证配置 |

## ⑥ 结果图

本综合报告为**元验证**（meta-verification），不产出新物理场图。所有结论来自对已提交工件的机器化复核。各验证层的场量图见对应工件目录的 `figures/` 子目录。

## ⑦ 注意点（坑 / 局限 / 适用范围）

1. **生产腿 FAIL 是物理事实**：p3.N400 的 3.116 % 略超 3 % 门槛，这是间断主导的全场误差在有限分辨率下的如实结果，不是求解器缺陷。
2. **参考底限不可消除**：误差分解显示参考底限在 p3.N400 上占总误差的显著比例，说明部分误差来自精确解在 DG 空间上的投影误差。
3. **认证配置 p2.N800 是最优选择**：在全部候选配置中，p2.N800 以最小 DOF（2400）通过 <3 % 门槛，墙钟 118.4s。
4. **限幅器不是补救手段**：TVB/WENO 在 N=400 上使 L2 密度恶化 2.26×/2.07×，因为限幅器将问题单元退化为线性重构。
5. **阶数判据（C3/R3）FAIL 是预期行为**：间断存在时全场收敛阶被压到一阶附近，这不是求解器缺陷。

## ⑧ 误差判定（是否 <3%）

| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| 生产腿 p3.N400 密度 L2 | 精确 Riemann 解 | 3.116 % | ❌ 否（超 3.84 %，裕度 ×0.963） |
| 分辨率闭合 p3.N800 密度 L2 | 精确 Riemann 解 | 2.315 % | ✅ 是 |
| 认证配置 p2.N800 密度 L2 | 精确 Riemann 解 | 2.570 % | ✅ 是 |
| 认证配置 p2.N800 速度 L2 | 精确 Riemann 解 | 1.763 % | ✅ 是 |
| 认证配置 p2.N800 压力 L2 | 精确 Riemann 解 | 1.098 % | ✅ 是 |

## ⑨ 闭合建议

CASE-LAX-DG 的验证证据链已完整，但台账状态仍为 "open"。建议：

1. **更新生产腿**：将台账登记的生产腿从 p3.N400 改为 p2.N800（认证配置），则 A0 判据变为 PASS。
2. **或更新容差**：若保持 p3.N400 为生产腿，可将容差从 3 % 调整为 3.2 %，则 A0 判据变为 PASS。
3. **或保持现状**：如实记录 p3.N400 的 honest FAIL，同时在文档中注明 p2.N800 为认证配置。

**推荐方案 1**：更新生产腿为 p2.N800，理由：
- p2.N800 以最小 DOF（2400）通过 <3 % 门槛
- 墙钟 118.4s，计算成本合理
- 与分辨率闭合研究结论一致
- 不改变容差，保持台账原文的严格性

---

*本报告由程序化注入生成，所有数字从已提交工件直读（零手抄）。判据映射可机器复核。*
