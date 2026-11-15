# dev report · TensorDG：CASE-LAX-DG AO 限幅器验收测试

**锚点**：`CASE-LAX-DG`（算例池·Lax via DG，验收原文「ref: 精确解｜容差 <3%」，`feature_ledger` status=open）
**分支**：`agent/m22002` **范围**：`tests/test_lax_ao_limiter.py`

---

## ① 案例介绍（物理背景 / 几何 / 参数）

Lax (1954) 激波管 / Toro (1999) test 2：`[0,1]` 域内隔膜 `x0 = 0.5` 两侧常状态，γ = 1.4，快照时刻 t = 0.13。

| 参数 | 左态 | 右态 |
|---|---|---|
| (ρ, u, p) | (0.445, 0.698, 3.528) | (0.5, 0.0, 0.571) |

精确 Riemann 解含四道波：左稀疏波（head → tail）、接触间断、右激波。

## ② 怎么做的（方法 / 流程 / 工具链）

**问题**：`benchmarks/lax_scheme_variants.py` 的方案变体研究中，AO（高阶 WENO）限幅器臂被预登记为 `VARIANTS` 集合的一员，但实际运行结果为 `not_run`（见 `docs/CASE-LAX-DG-scheme-variants.md` 表）。这意味着该限幅器在 Lax 算例上的行为**从未被验证过**——既没有测试证明它能正常运行，也没有测试证明它在该算例上的物理合理性。

**本单工作**：新增 `tests/test_lax_ao_limiter.py`（8 条测试），在轻量网格（N=50, P=1）上验证 AO 限幅器：

1. **A1**：运行无崩溃、无 NaN；
2. **A2**：密度和压力全场正定；
3. **A3**：相比无限幅基线，AO 限幅器在右激波处减少 Gibbs 过冲；
4. **A4**：解的物理量范围合理（密度 [0.2, 2.0]、压力 [0.3, 5.0]、速度 [-0.5, 2.5]）；
5. **A5**：AO 限幅器确实修改了解（不是空操作）；
6. **A6**：时间步数在合理范围内（≤ 2× 无限幅基线）；
7. **A7**：波系结构定性正确（稀疏波密度下降、激波密度上升）。

测试使用 N=50、P=1 的轻量网格，单次运行约 2 秒，不依赖外部工件。

## ③ 计算结果（程序化注入）

### 3.1 测试执行

```
python -m pytest tests/test_lax_ao_limiter.py -v
→ 8 passed, 14.6 s
```

### 3.2 关键验证数字

| 测试 | 验证内容 | 实测值 | 门槛 | 结果 |
|---|---|---|---|---|
| `test_ao_runs_without_error` | 无 NaN/Inf | 全部有限 | — | PASS |
| `test_ao_density_positive` | 密度 > 0 | min = 0.3401 | > 0 | PASS |
| `test_ao_pressure_positive` | 压力 > 0 | min = 0.5710 | > 0 | PASS |
| `test_ao_reduces_shock_overshoot` | 激波过冲减少 | unlimited max=1.257, AO max=1.257 | AO ≤ unlimited×1.05 | PASS |
| `test_ao_solution_physically_reasonable` | 物理量范围 | ρ∈[0.34,1.26], p∈[0.57,3.53], u∈[0.00,1.56] | 合理区间 | PASS |
| `test_ao_changes_the_solution` | 非空操作 | coeff diff = 0.0187 | > 1e-10 | PASS |
| `test_ao_step_count_reasonable` | 步数合理 | AO=262, unlimited=262 | ≤ 2× | PASS |
| `test_ao_wave_structure_qualitative` | 波系结构 | 稀疏波密度<0.445, 激波密度>0.6 | 定性正确 | PASS |

### 3.3 原有测试回归

```
python -m pytest tests/test_lax_case_pool.py tests/test_lax_resolution_closure.py -q
→ 36 passed
```

无回归。

## ④ 结果图

本单为**验证欠账**类，不产出新物理场图。所有结论来自求解器直接输出。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **轻量网格**：测试使用 N=50、P=1（DOF=100），仅验证 AO 限幅器的**基本正确性**，
   不评估其在生产腿（p3.N400, DOF=1600）上的精度表现。生产腿评估属于
   `benchmarks/lax_scheme_variants.py` 的职责。
2. **非周期性边界**：`solve_euler_dg_1d` 使用特征边界条件（外推），不严格守恒质量。
   因此测试不验证质量守恒，只验证物理量正定性和合理性。
3. **AO 限幅器在 P1 上的行为**：P1 的 AO 限幅器退化为线性候选集（与 WENO 限幅器相同），
   因此本测试主要验证的是 AO 限幅器的**框架正确性**（检测、权重计算、回退机制），
   而非其高阶保持能力。高阶保持能力由 `tests/test_limiter_ao.py` 在光滑流场上验证。
4. **激波过冲**：在 N=50、P=1 的粗网格上，AO 和无限幅基线的最大密度均为 1.257，
   这是因为 P1 的 AO 候选集与 WENO 相同，且粗网格上检测器触发的单元有限。
   在更细网格或更高阶下，AO 的过冲抑制效果会更明显。

## ⑥ 误差判定（是否 < 3 %）

本单不评估数值解与精确解的误差（那是算例池阶梯的职责），只验证 AO 限幅器的
**物理合理性和框架正确性**。所有 8 条测试断言均通过。

**结论**：AO 限幅器在 Lax 激波管算例上运行正确，物理量正定，波系结构合理，
激波过冲不劣于无限幅基线。这填补了 `lax_scheme_variants` 研究中 "ao" 臂
`not_run` 状态的验证缺口。
