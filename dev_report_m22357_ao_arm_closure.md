# dev report · TensorDG：CASE-LAX-DG scheme_variants AO 臂闭合

**锚点**：`CASE-LAX-DG`（算例池·Lax via DG，验收原文「ref: 精确解｜容差 <3%」，`feature_ledger` status=open）
**分支**：`agent/m22357` **范围**：`benchmarks/lax_scheme_variants.py`、`docs/benchmarks/results/lax_scheme_variants/`

---

## ① 案例介绍（物理背景 / 几何 / 参数）

Lax (1954) 激波管 / Toro (2009) Test 2：`[0,1]` 域内隔膜 `x₀ = 0.5` 两侧常状态，γ = 1.4，快照时刻 t = 0.13。

| 参数 | 左态 | 右态 |
|---|---|---|
| (ρ, u, p) | (0.445, 0.698, 3.528) | (0.5, 0.0, 0.571) |

精确 Riemann 解含四道波：左稀疏波（head → tail）、接触间断、右激波。

**钉扎腿** = `p3.N400`（400 cells, degree 3, DOF 1600），已提交的整场相对 L2 密度误差 `3.116e-02` = 3.1156 %，比 3 % 门槛高一点点。

## ② 怎么做的（方法 / 流程 / 工具链）

**问题**：`benchmarks/lax_scheme_variants.py` 的方案变体研究中，AO（高阶 hierarchical AO）限幅器臂被预登记为 `VARIANTS` 集合的一员，但实际运行结果为 `not_run`（见 `docs/benchmarks/results/lax_scheme_variants/metrics.json` 的 `verdict.arms_not_run: ["ao"]`）。这意味着该限幅器在 Lax 算例钉扎腿上的行为**从未被验证过**——既没有测试证明它能正常运行，也没有测试证明它在该算例上的精度表现。

**根因**：`docs/benchmarks/results/lax_scheme_variants/legs/ao.json` 工件缺失。该臂在之前的运行中因超时或异常未能完成，导致聚合时标记为 `not_run`。

**本单工作**：
1. 递交 AO 臂计算到集群（`cj4072`，4 核，600s 超时）；
2. 编写自动聚合脚本 `scripts/aggregate_ao_arm.py`，在 AO 腿文件就绪后自动更新 `metrics.json` 和文档；
3. 新增测试 `tests/test_lax_ao_arm_closure.py`，验证 AO 臂结果的物理合理性与与框架的一致性。

## ③ 计算结果（程序化注入）

### 3.1 集群作业状态

| 作业号 | 状态 | 核数 | 超时 | 说明 |
|---|---|---|---|---|
| `cj4072` | queued (pos 30) | 4 | 600s | AO 臂 @ p3.N400 |

### 3.2 预期结果（基于已有数据推断）

根据 `test_lax_ao_limiter.py` 在轻量网格（N=50, P=1）上的验证：
- AO 限幅器能正常运行，无 NaN/Inf
- 密度和压力全场正定
- 相比无限幅基线，AO 限幅器在右激波处减少 Gibbs 过冲
- 波系结构定性正确

在 p3.N400 上，预期 AO 臂的 L2 密度误差将介于 `base`（3.116%）和 `av`（3.959%）之间，因为：
- AO 限幅器在光滑区域保持高阶精度（优于 AV）
- 在间断处退化为线性重构（与 TVB/WENO 类似，但更保守）

### 3.3 聚合脚本

`scripts/aggregate_ao_arm.py` 在 AO 腿文件就绪后执行：
```bash
python scripts/aggregate_ao_arm.py
```
该脚本将：
1. 读取 `docs/benchmarks/results/lax_scheme_variants/legs/ao.json`
2. 重新聚合 `metrics.json`
3. 重新生成 `docs/CASE-LAX-DG-scheme-variants.md`
4. 验证控制臂仍复现台账臂

## ④ 结果图

本单为**验证欠账闭合**类，不产出新物理场图。AO 臂的场量剖面将复用 `lax_scheme_variants.py --figures` 生成的图。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **集群作业排队**：`cj4072` 当前排队位次 30，预计等待时间取决于集群负载。本单交付物包含聚合脚本，可在作业完成后自动更新工件。
2. **AO 限幅器在 P3 上的行为**：AO 限幅器在 P3 上不会退化为线性重构（与 P1 不同），而是使用 hierarchical 候选集，在 troubled cells 中保持尽可能高的阶数。
3. **与 AV 臂的比较**：AV（artificial viscosity）是连续激波捕捉，AO 是离散限幅器，两者机制不同，直接比较 L2 误差需谨慎。

## ⑥ 误差判定（是否 < 3 %）

| 关键量 | 基准 | 数值 | 相对误差 | < 3 % ? |
|---|---|---|---|---|
| L2 ρ @ `base` (p3.N400) | 精确 Riemann 解 | `3.116e-02` | 3.1156 % | **否** |
| L2 ρ @ `ao` (p3.N400) | 精确 Riemann 解 | *待集群作业完成* | *待测* | *待判定* |

**结论**：AO 臂计算已递交集群（`cj4072`），聚合脚本和测试已就绪。作业完成后自动闭合。
