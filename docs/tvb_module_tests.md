# TensorDG TVB 模块直接单元测试验证报告

> 任务：m3666（自找活·tp-qa-2 verifier 席）
> 日期：2026-11-04
> 分支：`agent/m3666`
> 环境：Python 3.11.6 / torch 2.7.1 / pytest 9.2.0

## 1. 案例介绍

**验证对象**：`tensordg/tvb.py` 模块中的 `troubled_from_differences` 函数。

**为什么需要这个测试**：该函数是 1D 和 2D 限制器共用的 TVB 问题单元检测核心实现
（m2549 从 `limiters.py` 中提取）。此前所有测试都通过 `limiters.py` 的
`_troubled_cell_indicator` 间接调用它，没有直接测试。作为限制器系统的基石，
直接测试可以：
1. 隔离验证尺度一致性准则（scale-consistent criterion）
2. 覆盖边界情况（仿射数据、光滑极值、间断）
3. 为未来开发者提供明确的行为文档

**物理背景**：TVB（Total Variation Bounded）准则由 Cockburn & Shu (1989) 提出，
用于检测 DG 解中需要限制的高阶模态。尺度一致性修复（m2500）将比较量从
`max(|d_left|, |d_right|)`（O(dx·u')）改为 `|d_left - minmod(d_left, d_right)|`
（O(dx²·u'') 光滑区 / O(dx·u') 间断处），使 TVB 常数 M 成为网格无关参数。

## 2. 怎么做的

新增 `tests/test_tvb.py`，包含 18 条单元测试，覆盖：

| 测试类别 | 测试数 | 验证内容 |
|---|---:|---|
| minmod 辅助函数 | 3 | 同号选小、异号归零、零值处理 |
| 仿射数据 | 2 | 线性剖面永不标记为问题单元 |
| 光滑极值 | 3 | M=0 时标记，大 M 时释放，阈值边界 |
| 间断 | 2 | 任何 M 下都标记 |
| 光滑减速 | 2 | M=0 时标记，适当 M 时释放 |
| 多变量 | 2 | 跨变量归约、形状保持 |
| 舍入底 | 1 | 近仿射数据不因舍入误判 |
| 数据类型 | 2 | float32/float64 保持 |
| 尺度一致性 | 1 | 光滑波上 M 的网格无关性 |

## 3. 计算结果

```
tests/test_tvb.py::test_minmod_same_sign_selects_smaller PASSED
tests/test_tvb.py::test_minmod_opposite_sign_returns_zero PASSED
tests/test_tvb.py::test_minmod_one_zero_returns_zero PASSED
tests/test_tvb.py::test_affine_data_never_troubled PASSED
tests/test_tvb.py::test_affine_data_with_nonzero_M_still_not_troubled PASSED
tests/test_tvb.py::test_smooth_extremum_troubled_at_strict_tvd PASSED
tests/test_tvb.py::test_smooth_extremum_released_at_large_M PASSED
tests/test_tvb.py::test_smooth_extremum_threshold_boundary PASSED
tests/test_tvb.py::test_discontinuity_always_troubled PASSED
tests/test_tvb.py::test_discontinuity_with_small_M PASSED
tests/test_tvb.py::test_smooth_decelerating_troubled_at_M_zero PASSED
tests/test_tvb.py::test_smooth_decelerating_released_at_appropriate_M PASSED
tests/test_tvb.py::test_multi_variable_troubled_reduces_across_vars PASSED
tests/test_tvb.py::test_multi_variable_slope_shape PASSED
tests/test_tvb.py::test_roundoff_floor_does_not_trigger_on_near_affine PASSED
tests/test_tvb.py::test_dtype_preserved PASSED
tests/test_tvb.py::test_dtype_float64 PASSED
tests/test_tvb.py::test_scale_consistency_smooth_wave PASSED

======================== 18 passed, 2 warnings in 8.34s ========================
```

**回归验证**：现有限制器测试套件（27 条）全部通过：
```
tests/test_limiter_units.py (3 passed)
tests/test_limiter_validation.py (7 passed)
tests/test_limiters.py (17 passed)
```

## 4. 误差判定

本测试为纯单元逻辑测试，不涉及数值误差比较。所有断言均为精确匹配或容差匹配
（`torch.allclose` / `pytest.approx`），验证函数行为与数学定义一致。

关键验证点：
- **尺度一致性**：光滑波 `u = sin(2πx)` 在 M=100 时，40/80/160 网格的问题单元比例
  均 < 10%，证明 M 的网格无关性 ✅
- **阈值边界**：deviation = 0.2, dx = 0.1 时，M=20（threshold = 0.2）不标记，
  M=19（threshold = 0.19）标记，验证严格大于比较 ✅
- **仿射数据**：偏差为零，永不标记 ✅
- **间断**：偏差 O(dx·u') 主导 M·dx²，始终标记 ✅

## 5. 注意点 / 局限

1. **仅测试核心函数**：`troubled_from_differences` 是纯张量运算，不涉及求解器循环。
   完整的限制器行为（含 SSP-RK 子步应用、正性过滤等）由现有测试套件覆盖。
2. **未测试 2D 方向分裂**：`tvb.py` 的函数是 1D 原语，2D 限制器在 `euler_dg_2d.py`
   中逐轴调用。2D 行为由 `test_limiter_2d.py` 覆盖。
3. **轻量级**：所有测试在 < 10 秒内完成，适合高负载环境。

## 6. 交付物

| 文件 | 作用 |
|---|---|
| `tests/test_tvb.py` | 18 条直接单元测试 |

## 7. 复现命令

```bash
PYTHONPATH=. python -m pytest tests/test_tvb.py -v
PYTHONPATH=. python -m pytest tests/test_limiter_units.py tests/test_limiter_validation.py tests/test_limiters.py -v
```
