# TensorDG 首波扩编审计报告（m240）

> 上游：TensorDG@c485c623（main，2026-10-23 一天建成）
> 工作区：/nfs/wangxi/tensor_eco/agent_work/m240/TensorDG
> 分支：agent/m240（基于 main，无本地改动）
> 审计时间：2026-10-24
> Python：/nfs/wangxi/hermes-agent/.venv/bin/python（3.12.13，pytest 9.1.1，torch 2.14）
> 纪律：PYTHONPATH=. 压位；数字全部从工件直读，零手抄；上游 main 只读。

---

## 1. 环境与测试基线

命令：`PYTHONPATH=. /nfs/wangxi/hermes-agent/.venv/bin/python -m pytest -v`

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.7.0
rootdir: /nfs/wangxi/tensor_eco/agent_work/m240/TensorDG
configfile: pyproject.toml
testpaths: tests
collecting ... collected 21 items

tests/test_advection_1d.py::test_solution_is_finite_and_mass_conservative PASSED
tests/test_advection_1d.py::test_degree_one_converges_at_expected_rate PASSED
tests/test_advection_1d.py::test_zero_final_time_returns_initial_projection PASSED
tests/test_advection_2d.py::test_mass_conservation_and_negative_velocity PASSED
tests/test_advection_2d.py::test_degree_one_converges_at_second_order PASSED
tests/test_advection_2d.py::test_documented_two_dimensional_case_is_below_three_percent PASSED
tests/test_burgers_1d.py::test_burgers_conserves_mass_and_matches_characteristics PASSED
tests/test_burgers_1d.py::test_burgers_p2_has_near_third_order_convergence PASSED
tests/test_euler_sod_1d.py::test_sod_state_remains_physical_and_finite PASSED
tests/test_euler_sod_1d.py::test_sod_early_time_conservation_and_wave_direction PASSED
tests/test_euler_sod_1d.py::test_sod_respects_requested_tensor_dtype PASSED
tests/test_euler_sod_1d.py::test_exact_sod_star_state PASSED
tests/test_euler_sod_1d.py::test_documented_sod_case_matches_exact_solution_within_three_percent PASSED
tests/test_heat_1d.py::test_heat_sipg_p1_converges_at_second_order PASSED
tests/test_heat_1d.py::test_documented_heat_case_is_below_three_percent PASSED
tests/test_lax_1d.py::test_lax_star_state_and_short_time_positivity PASSED
tests/test_lax_1d.py::test_documented_lax_case_matches_exact_riemann_within_three_percent PASSED
tests/test_poisson_2d.py::test_sipg_degree_one_converges_and_meets_three_percent_tolerance PASSED
tests/test_poisson_2d.py::test_sipg_degree_two_has_third_order_accuracy PASSED
tests/test_vortex_2d.py::test_vortex_initial_projection_and_conservation PASSED
tests/test_vortex_2d.py::test_documented_vortex_case_is_below_three_percent_for_every_field PASSED

============================= 21 passed in 37.70s ==============================
```

**结论：21/21 PASS**。回归覆盖：
- **conservation**：test_solution_is_finite_and_mass_conservative / test_mass_conservation_and_negative_velocity / test_burgers_conserves_mass_and_matches_characteristics / test_sod_early_time_conservation_and_wave_direction / test_vortex_initial_projection_and_conservation
- **convergence**：test_degree_one_converges_at_expected_rate / test_degree_one_converges_at_second_order / test_burgers_p2_has_near_third_order_convergence / test_heat_sipg_p1_converges_at_second_order / test_sipg_degree_one_converges_and_meets_three_percent_tolerance / test_sipg_degree_two_has_third_order_accuracy
- **admissibility**：test_sod_state_remains_physical_and_finite / test_lax_star_state_and_short_time_positivity / test_exact_sod_star_state
- **3% 文档门**：test_documented_two_dimensional_case_is_below_three_percent / test_documented_sod_case_matches_exact_solution_within_three_percent / test_documented_heat_case_is_below_three_percent / test_documented_lax_case_matches_exact_riemann_within_three_percent / test_documented_vortex_case_is_below_three_percent_for_every_field

---

## 2. 基准抽检（数字直读工件）

### 2a. 1D 周期线性对流（README 声明 0.000236%）

命令：`python -m benchmarks.linear_advection_1d --degree 2 --cells 10 20 40 80`

```
1D periodic linear advection | DG degree=2 | PyTorch
cells       L2 error   relative L2       EOC       steps
   10   8.56563e-04   1.21136e-03      nan      625
   20   1.06970e-04   1.51279e-04    3.001     1250
   40   1.33721e-05   1.89110e-05    3.000     2500
   80   1.67157e-06   2.36396e-06    3.000     5000
```

| cells | relative L2 | EOC |
|------:|------------:|----:|
| 10 | 1.211360e-03 | nan |
| 20 | 1.512790e-04 | 3.001 |
| 40 | 1.891100e-05 | 3.000 |
| 80 | 2.363960e-06 | 3.000 |

- 最细网格 80 cells → relative L2 = **2.36396e-06 = 0.000236%**（与 README 声明 0.000236% 完全一致 ✅）
- EOC = 3.000，符合 P2 DG 理论最优阶（degree+1 = 3）

### 2b. Sod 激波管（README 声明 max 2.517658%，P0/FV）

命令：`python -m benchmarks.compare_euler_sod_1d --cells 3200 --time 0.2`

```
Sod exact comparison | cells=3200 steps=3499
star state: p*=0.303130178051, u*=0.927452620049
density : relative L2 = 1.422348%
velocity: relative L2 = 2.517658%
pressure: relative L2 = 0.816946%
results: sod_comparison_results
PASS: all field errors < 3%
```

| 场 | 实测 relative L2 | README 声明 max | 判定 |
|---|---:|---:|---|
| density | 1.422348% | 2.517658% | ✅ ≤ 声明 |
| velocity | **2.517658%** | 2.517658% | ✅ 精确匹配 |
| pressure | 0.816946% | 2.517658% | ✅ ≤ 声明 |

max 场 = velocity = 2.517658%，与 README 声明 max 2.517658% **逐字一致**。

### 2c. 其他 6 个基准（全部复跑）

| 基准 | 命令 | 实测 | README 声明 | 判定 |
|---|---|---:|---:|---|
| 2D 线性对流 | `--cells 24 --degree 1` | 0.281233% | 0.281233% | ✅ 精确 |
| 2D Poisson | `--cells 16 --degree 1` | 0.380026% | 0.380026% | ✅ 精确 |
| Smooth Burgers | `--cells 64 --degree 2` | 0.001820% | 0.001820% | ✅ 精确 |
| 1D Heat | `--cells 16 --degree 2` | 0.003604% | 0.003604% | ✅ 精确 |
| Lax 激波管 | `--cells 400 --degree 3` | max 2.769420% (ρ) | max 2.769420% | ✅ 精确 |
| 2D 等熵涡 | `--cells 192` | max 1.972990% (v_y) | max 1.972990% | ✅ 精确 |

**结论：8/8 基准全部 PASS，数字与 README 声明逐字一致，无一条需要 honest FAIL。**

---

## 3. 能力-声明对照（README Verified benchmarks / Current scope）

### 3.1 Verified benchmarks（8 条）

| # | 基准 | 方法声明 | 代码在场 | 测试在场 | 实测误差 | 3%门 |
|---|---|---|---|---|---:|---|
| 1 | 1D 周期线性对流 | P2 Legendre DG, upwind, RK4 | `tensordg/advection_1d.py` | `tests/test_advection_1d.py` | 0.000236% | ✅ |
| 2 | 2D 周期线性对流 | P1 tensor-product DG, upwind, RK4 | `tensordg/advection_2d.py` | `tests/test_advection_2d.py` | 0.281233% | ✅ |
| 3 | Euler Sod 激波管 | P0 DG/FV, Rusanov, SSP-RK3 | `tensordg/euler_1d.py` + `benchmarks/compare_euler_sod_1d.py` | `tests/test_euler_sod_1d.py` | max 2.517658% | ✅ |
| 4 | 2D Poisson | P1 SIPG | `tensordg/poisson_2d.py` | `tests/test_poisson_2d.py` | 0.380026% | ✅ |
| 5 | Smooth Burgers | P2 DG, Rusanov, SSP-RK3 | `tensordg/burgers_1d.py` | `tests/test_burgers_1d.py` | 0.001820% | ✅ |
| 6 | 1D Heat | P2 SIPG, Crank-Nicolson | `tensordg/heat_1d.py` | `tests/test_heat_1d.py` | 0.003604% | ✅ |
| 7 | Euler Lax 激波管 | P3 DG, HLLC, positivity fallback | `tensordg/euler_dg_1d.py` | `tests/test_lax_1d.py` | max 2.769420% | ✅ |
| 8 | 2D 等熵涡 | P0 DG/FV, Rusanov | `tensordg/euler_2d.py` | `tests/test_vortex_2d.py` | max 1.972990% | ✅ |

### 3.2 Current scope 逐条对照

| README 声明 | 代码在场性 | 备注 |
|---|---|---|
| Tensor-based DG operators on uniform 1D/2D Cartesian meshes | ✅ `advection_1d/2d`, `burgers_1d`, `euler_1d/2d`, `heat_1d`, `poisson_2d` | 全部 Cartesian |
| Modal Legendre bases and Gauss-Legendre projection/error quadrature | ✅ `benchmarks/_modal_1d.py`, `_modal_2d.py`, `euler_dg_1d.py` 用 `np.polynomial.legendre.leggauss` | |
| High-order smooth advection and Burgers, plus SIPG Poisson/heat verification | ✅ P2 advection/Burgers/heat, P1/P2 SIPG Poisson | |
| General P0 Euler Riemann solvers and high-order HLLC DG with local positivity fallback | ✅ `euler_1d.py` Rusanov+HLLC, `euler_dg_1d.py` 高阶 HLLC + `_positivity_filter` | |
| Two-dimensional Euler isentropic-vortex verification | ✅ `euler_2d.py` + `benchmarks/isentropic_vortex_2d.py` | |
| Regression tests for conservation, convergence, physical admissibility, exact-reference comparisons, and the documented 3% error threshold | ✅ 21 测试覆盖全部 5 类 | 见 §1 |

### 3.3 README 明确声明的缺口（Current scope 末尾）

> Robust TVB/WENO shock limiting, sparse/matrix-free elliptic solvers, unstructured meshes, high-order multidimensional Euler/NS, and CAD-coupled boundary handling are **out of scope** for the current release.

代码中 grep `TODO/FIXME/XXX/HACK`：**零命中**。代码整洁，无遗留标记。

**能力-声明对照结论：README 所有声明均有对应代码与测试在场；8 个基准数字全部可复现；缺口声明诚实。**

---

## 4. 代码规模与结构

```
tensordg/   1173 行（9 模块 + __init__）
benchmarks/ 1140 行（17 模块）
tests/       229 行（8 模块，21 测试）
总计        2542 行（不含 __pycache__）
```

模块分工清晰：
- `advection_1d/2d.py`：线性对流（upwind 数值通量 + RK4）
- `burgers_1d.py`：Burgers（Rusanov + SSP-RK3）
- `euler_1d.py`：P0 Euler（Rusanov + HLLC 两种通量）
- `euler_dg_1d.py`：高阶 Euler DG（HLLC + positivity fallback）
- `euler_2d.py`：2D Euler 等熵涡（P0/Rusanov/SSP-RK3）
- `heat_1d.py`：热方程（SIPG + Crank-Nicolson）
- `poisson_2d.py`：2D Poisson（SIPG，支持 P1/P2）

---

## 5. 与 TensorFVM 互补位分析

| 能力 | TensorDG | TensorFVM（推测） | 互补性 |
|---|---|---|---|
| 空间离散 | 模态 Legendre DG（高阶） | 有限体积（P0 为主） | 高阶 vs 鲁棒 |
| 网格 | Cartesian 1D/2D | 可能支持非结构 | 互补 |
| 通量 | Upwind/Rusanov/HLLC | HLLC/ AUSM 等 | 重叠 |
| 椭圆求解 | SIPG（密集矩阵） | 可能 multigrid | DG 缺 matrix-free |
| 激波处理 | 仅 positivity fallback | 可能有 TVD/WENO | DG 缺限制器 |
| 可微性 | PyTorch 原生 | 可能 | DG 优势 |

**DG 独特价值**：高阶精度（EOC=3 for P2）、天然可微（PyTorch 全链路）、模态基易于 p-adaptivity。
**DG 当前短板**：无限制器、无 matrix-free、无非结构网格、仅 1D/2D Cartesian。

---

## 6. 对 eco 有用的 backlog 登记

详见 `/nfs/wangxi/tensor_eco/agent_work/expand_follow/backlog.md` TensorDG 节。

优先级排序（对 eco 价值 × 可实现性）：

| # | 项目 | eco 价值 | 与 TensorFVM 互补 | 可实现性 | 优先级 |
|---|---|---|---|---|---|
| 1 | TVB/WENO 限制器 | 高阶激波捕捉 → 可压缩流 | TensorFVM 有 TVD，DG 缺 | 中（需 test suite） | **P0** |
| 2 | Matrix-free 椭圆求解 | SIPG 当前 O(N^3) 密集矩阵不可扩展 | FVM 可能有 multigrid | 中 | **P0** |
| 3 | 2D Euler 高阶（非 P0） | 当前 2D Euler 仅 P0/FV | 高阶输送问题 | 中 | **P1** |
| 4 | 非结构网格支持 | 真实几何 | CAD 库耦合 | 高（大工程） | P2 |
| 5 | Navier-Stokes 粘性通量 | 完整 CFD | FVM 互补 | 中 | P2 |
| 6 | 3D 扩展 | 真实 3D 问题 | 全生态需要 | 高 | P3 |

---

## 7. 开发候选提案

详见 `/nfs/wangxi/tensor_eco/agent_work/expand_follow/proposals/tensordg-tvb-weno-limiter.txt` 和 `tensordg-matrix-free-sipg.txt`。

---

## 8. 总结

| 维度 | 结论 |
|---|---|
| 测试套件 | 21/21 PASS（37.70s） |
| 8 基准复现 | 8/8 数字与 README 逐字一致 |
| 3% 文档门 | 全部满足 |
| 能力-声明对照 | 全部在场，缺口声明诚实 |
| 代码质量 | 无 TODO/FIXME，整洁 |
| 对 eco 价值 | 高阶可微 DG 独特位；限制器+matrix-free 是首要补强方向 |
| 建议 | 优先推进 TVB/WENO 限制器 + matrix-free SIPG |

**判决：TensorDG 首波审计 PASS，可进入 backlog 登记与开发候选阶段。**
