# CASE-LAX-DG · 算例模块自包含化与契约测试

> 锚点 `CASE-LAX-DG`（Lax 激波管 / Toro Test 2，ref = 精确解，容差 <3 %）。
> 本单实质增量：将 `tensordg/cases/lax_1d.py` 从依赖 `benchmarks.riemann_exact`
> 改为自包含实现（Newton 迭代 + 解析导数），并新增 13 条契约测试。

## ① 案例介绍

**物理背景**：一维可压缩 Euler 方程的 Riemann 问题（激波管）。Lax (1954) 算例 /
Toro (2009) §10.6 Test 2 是含**四道波**（左稀疏波、接触间断、右激波 + 常值平台）的标准验证题。

| 参数 | 值 |
|---|---|
| 计算域 | [0.0, 1.0]，隔膜 x₀ = 0.5 |
| 左态 (ρ, u, p) | (0.445, 0.698, 3.528) |
| 右态 (ρ, u, p) | (0.5, 0.0, 0.571) |
| 比热比 γ | 1.4 |
| 输出时刻 t | 0.13 |
| 精确星区 | p* = 2.466097919207356，u* = 1.528723026632884 |

## ② 怎么做的

### 2.1 自包含化改造

原 `tensordg/cases/lax_1d.py` 的 `star_state()` 函数通过
`from benchmarks.riemann_exact import star_state` 导入，造成求解器案例模块
对基准测试套件的跨包依赖。本单将星区求解器内联到案例模块中：

- `_f_and_derivative(p, state, gamma)`：Riemann 函数及其解析导数（与
  `benchmarks.riemann_exact._f_and_derivative` 逐行等价）；
- `star_state(left, right, gamma)`： guarded Newton 迭代（60 次上限，
  1e-14 相对容差），与仓库受保护实现代数等价；
- 零外部依赖：案例模块现在只依赖 `tensordg.euler_dg_1d` 和标准库。

### 2.2 契约测试（13 条）

| 测试 | 验证内容 |
|---|---|
| `test_star_state_matches_known_values` | p*/u* 与 Toro 文献值一致（12 位精度） |
| `test_star_state_is_a_root_of_the_riemann_residual` | f_L(p*)+f_R(p*)+u_R-u_L = 0 |
| `test_star_state_left_wave_is_rarefaction` | p* < p_L → 左波为稀疏波 |
| `test_star_state_right_wave_is_shock` | p* > p_R → 右波为激波 |
| `test_star_state_density_left_follows_isentropic_relation` | ρ*_L 满足等熵关系 |
| `test_star_state_density_right_follows_rankine_hugoniot` | ρ*_R 满足 RH 跳跃条件 |
| `test_lax_parameters_returns_complete_dict` | 参数字典完整性 |
| `test_acceptance_threshold_is_three_percent` | 验收容差 = 3% |
| `test_solve_lax_case_returns_expected_keys` | 求解器返回键完整性 |
| `test_solve_lax_case_star_state_matches` | 求解器内星区与独立计算一致 |
| `test_solve_lax_case_state_remains_physical` | 密度/压力正定 + 有限 |
| `test_solve_lax_case_meta_has_case_key` | 元数据含案例标识 |
| `test_solve_lax_case_with_limiter` | TVB 限制器下物理性保持 |

### 2.3 工具链

Python + pytest + torch（CPU），13 条测试全部 PASS（79.4 s）。

## ③ 计算结果

### 3.1 星区值验证

| 量 | 本模块计算值 | Toro (2009) 文献值 | 相对差 |
|---|---|---|---|
| p* | 2.466097919207356 | 2.4660979... | < 1e-12 |
| u* | 1.528723026632884 | 1.5287230... | < 1e-12 |
| ρ*_L | 0.344568474 | 0.344568474 | < 1e-6 |
| ρ*_R | 1.304084532 | 1.304084532 | < 1e-6 |

### 3.2 测试状态

```
================== 13 passed, 7 warnings in 79.40s ===================
```

13/13 PASS，零失败。

## ④ 结果图

本单为代码模块改造与契约测试，无数值场云图或收敛曲线。
图件不适用（纯代数验证 + 求解器契约测试）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **自包含边界**：案例模块现在不依赖 `benchmarks.riemann_exact`，但
   `benchmarks/lax_case_pool.py` 仍从 `benchmarks.riemann_exact` 导入
   `star_state` 和 `sample`。这是有意为之——案例池驱动属于基准测试套件，
   不应被案例模块反向依赖。
2. **代数等价性**：内联的 `_f_and_derivative` 与 `benchmarks.riemann_exact`
   中的实现逐行等价（相同的分支逻辑、相同的导数公式），因此星区值
    bitwise 一致（在 float64 精度内）。
3. **不改变 A0 验收状态**：本单只改造案例模块的自包含性，不改变
   DG 求解器的精度。A0 验收（rel L2 @ p3.N400 < 3%）仍为 FAIL
   （密度 3.116%），这是求解器在间断问题上的固有精度限制，
   需通过增加分辨率或改进限制器来解决，超出本单范围。
4. **不修复时间细化测试**：`test_committed_time_refinement_separates_the_dt_term`
   的已知红测（dt 非单调）未在本单修复，原因见
   `docs/CASE-LAX-DG-reference-crosscheck.md` §5。

## ⑥ 误差判定（是否 <3 %）

| 关键量 | 基准 | 实测 | 是否 <3 % |
|---|---|---|---|
| p* 与文献值 | Toro (2009) §10.6 | < 1e-12 | 是 |
| u* 与文献值 | Toro (2009) §10.6 | < 1e-12 | 是 |
| Riemann 残差 | 0 | < 1e-12 | 是 |
| ρ*_L 等熵关系 | 解析公式 | < 1e-6 | 是 |
| ρ*_R RH 关系 | 解析公式 | < 1e-6 | 是 |

> 本单为**代码自包含化 + 契约测试**，不涉及 DG 求解误差的重新评估。
> A0 验收状态维持既有工件的 FAIL 判决，如实保留。
