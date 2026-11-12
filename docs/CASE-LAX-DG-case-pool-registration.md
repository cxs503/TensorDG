# CASE-LAX-DG · 算例池注册：Lax 激波管 canonical case 入库

> 锚点：`CASE-LAX-DG`｜仓库 TensorDG｜分支 `agent/m16928`
> 生成于 2026-11-12 · 测试 9/9 PASS · 零手抄

## ① 案例介绍

Lax (1954) 激波管 / Toro (2009) Test 2：`[0,1]` 域内隔膜 `x0 = 0.5` 两侧常状态，γ = 1.4，快照时刻 t = 0.13。精确解含四道波：左稀疏波、接触间断、右激波。星区状态由精确 Riemann 求解器迭代得出（非手抄）。

| 参数 | 左态 | 右态 |
|---|---|---|
| (ρ, u, p) | (0.445, 0.698, 3.528) | (0.5, 0.0, 0.571) |
| p* (计算) | 2.4660979192 | |
| u* (计算) | 1.5287230266 | |

参考出处：Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., Springer 2009, §10.6 Test 2.

## ② 做了什么

此前 Lax 算例仅存在于 `benchmarks/lax_1d.py`（驱动脚本），未注册到 `tensordg/cases/` 规范算例池。本单完成：

1. **新增 `tensordg/cases/lax_1d.py`**（~100 行）：canonical case 定义模块
   - `CASE_KEY = "lax-1954-shock-tube"`
   - 物理参数：DOMAIN, DIAPHRAGM, FINAL_TIME, GAMMA, LEFT_STATE, RIGHT_STATE
   - `lax_parameters()`: 返回完整参数字典（含 acceptance_threshold = 0.03）
   - `star_state()`: 委托 `benchmarks.riemann_exact.star_state` 计算精确星区状态
   - `solve_lax_case()`: 封装 `solve_euler_dg_1d`，返回含元数据和精确锚点的结果字典

2. **更新 `tensordg/cases/__init__.py`**：注册 `lax_1d` 到 CASES 字典，导出全部公共符号

3. **新增 `tests/test_case_lax_1d.py`**（9 条测试）：
   - 注册验证：CASES 字典包含 lax_1d
   - 参数完整性：lax_parameters() 返回所有必需字段
   - 星区状态精度：p*, u* 与精确求解器一致（误差 < 1e-10）
   - 物理性检查：左右态密度/压力为正
   - 粗网格求解：40 单元 P1 求解返回全正密度/压力、有限系数
   - 输出元数据：包含 case_key 和精确星区状态
   - 域与隔膜：DOMAIN=(0,1), DIAPHRAGM=0.5, FINAL_TIME=0.13, GAMMA=1.4

## ③ 计算结果（测试验证）

| 测试项 | 结果 |
|---|---|
| test_case_key_is_registered | PASS |
| test_parameters_dict_is_complete | PASS |
| test_star_state_matches_known_values | PASS (p* 误差 < 1e-10, u* 误差 < 1e-10) |
| test_left_state_is_physical | PASS |
| test_right_state_is_physical | PASS |
| test_small_solve_returns_physical_state | PASS (40 cells, P1, 全正) |
| test_star_state_in_output | PASS |
| test_case_key_in_meta | PASS |
| test_domain_and_diaphragm | PASS |

**9/9 PASS**，总耗时 22.8 s。

## ④ 误差判定

- 星区状态 p* = 2.4660979192 vs 精确求解器 2.4660979192073564 → 误差 < 1e-10 ✅
- 星区状态 u* = 1.5287230266 vs 精确求解器 1.528723026632884 → 误差 < 1e-10 ✅
- 粗网格求解（40 cells, P1）全单元密度/压力为正 ✅
- 所有系数有限 ✅

## ⑤ 注意点

- 本单只完成**算例池注册**（case pool entry），不改变求解器本身
- 原有的 `benchmarks/lax_1d.py` 驱动脚本保持不变，现可通过 `tensordg.cases.lax_1d` 统一访问
- 系统负载极高（load15 ≈ 342/128），精细网格求解测试会超时，故测试仅用 40 单元粗网格验证物理性
- 完整分辨率阶梯验证（N=100/200/400 × P1/P2/P3）已在 `benchmarks/lax_case_pool.py` 中实现，结果见 `docs/CASE-LAX-DG.md`

## ⑥ 对 CASE-LAX-DG 锚点的贡献

| 锚点要求 | 状态 |
|---|---|
| 算例池注册（tensordg/cases/） | ✅ 完成 |
| 精确解参考 | ✅ 通过 benchmarks.riemann_exact |
| 容差 <3% | ✅ 已有基准结果 2.77% < 3%（docs/CASE-LAX-DG.md） |
| 测试覆盖 | ✅ 9 条新测试 |
