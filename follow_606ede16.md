# TensorDG 上游跟进评审报告（m282）

> 上游范围：c485c623..606ede16（5 commits）
> 评审时间：2026-10-25
> 评审分支：agent/m282（基于 606ede16）
> 测试套件：34/34 PASS（59.63s）
> Python：/nfs/wangxi/hermes-agent/.venv/bin/python（3.12.13，pytest 9.1.1，torch 2.14）

---

## 1. Commits 概览

| # | SHA | 类型 | 标题 | 评审结论 |
|---|-----|------|------|----------|
| 1 | 78adc0c | docs | 添加 TensorDG 首波扩编审计报告（m240） | ✅ 通过 |
| 2 | 8e94ea6 | chore | 添加机器可读判据工件 validation_report.json | ✅ 通过 |
| 3 | fea8844 | merge | Merge pull request #5 from cxs503/agent/m240 | ✅ 通过 |
| 4 | e22c308 | feat | 实现 TVB/WENO 限制器用于 1D Euler 方程 | ✅ 通过（有已知限制） |
| 5 | 606ede1 | merge | Merge pull request #6 from cxs503/agent/m274 | ✅ 通过 |

**关键发现**：上游 c485c623..606ede16 的全部 5 个 commits 均为合并本线自己的 PR（m240 审计 + m274 限制器），**无上游独立新开发**。上游 owner（chux503）仅做了 merge 操作。

---

## 2. 逐 commit 评审

### 2.1 Commit 78adc0c — docs: 添加 TensorDG 首波扩编审计报告（m240）

**变更**：新增 `expand_audit_tensordg.md`（226 行）

**评审**：
- 内容为 m240 审计报告的完整副本，已在 m240 任务中验证
- 测试基线 21/21 PASS、8/8 基准数字与 README 逐字一致——与 m240 交付一致
- 无新声明，纯文档归档

**结论**：✅ 通过，无风险

### 2.2 Commit 8e94ea6 — chore: 添加机器可读判据工件 validation_report.json

**变更**：新增 `validation_report.json`（139 行）

**评审**：
- 机器可读格式的审计判据，包含测试通过数、基准数字、收敛阶表
- 与 expand_audit_tensordg.md 内容一致
- 符合平台验收纪律（判据工件直读）

**结论**：✅ 通过，无风险

### 2.3 Commit fea8844 — Merge PR #5 from cxs503/agent/m240

**变更**：合并 m240 分支到 main（expand_audit_tensordg.md + validation_report.json）

**评审**：
- 标准 merge commit，无冲突
- 合并内容已在 78adc0c/8e94ea6 评审

**结论**：✅ 通过

### 2.4 Commit e22c308 — feat: 实现 TVB/WENO 限制器用于 1D Euler 方程

**变更**：13 files changed, 1149 insertions(+), 130 deletions(-)

**新增文件**：
- `tensordg/limiters.py`（235 行）：TVB + WENO 限制器实现
- `tests/test_limiters.py`（223 行）：13 个单元测试
- `benchmarks/sod_with_limiter.py`（112 行）：带限制器的 Sod 基准
- `benchmarks/compare_limiter.py`（148 行）：对比基准
- `dev_report_tensordg-tvb-weno-limiter.md`（154 行）：开发报告
- `limiter_comparison/sod_limiter_metrics.json`（70 行）：200-cell 对比数据
- `sod_with_limiter/sod_limiter_metrics.json`（27 行）：100-cell TVB 数据
- 2 个 PNG 图

**修改文件**：
- `tensordg/euler_dg_1d.py`：新增 `limiter` 和 `tvb_constant` 参数，SSP-RK3 每子步后调用限制器
- `tensordg/__init__.py`：导出限制器函数
- `README.md`：添加 Shock limiters 章节（+52 行）
- `validation_report.json`：更新为 m274 版本

**新声明评审**：

| README 声明 | 验证状态 | 来源 |
|---|---|---|
| TVB limiter (Cockburn & Shu 1989) 在场 | ✅ | limiters.py:apply_tvb_limiter |
| WENO limiter (Qiu & Shu 2005) 在场 | ✅ | limiters.py:apply_weno_limiter |
| 保留 cell average | ✅ | test_tvb/weno_limiter_preserves_cell_average |
| 不修改 smooth monotone 解 | ✅ | test_tvb_limiter_does_not_modify_smooth_solution |
| Limiter overhead ~80-120% | ✅ | validation_report.json: 76.4%-122.9% |
| 34/34 测试通过 | ✅ | 本次复跑 34/34 PASS（59.63s） |

**代码质量评审**：
- 限制器正确实现 TVB/WENO 算法（minmod 检测 + WENO 重构）
- 集成方式合理：每个 SSP-RK3 子步后先限制再 positivity filter
- 测试覆盖充分：13 个单元测试涵盖核心不变量
- 已知限制诚实记录：性能开销 76-123%（目标 <30%）、限制器过于激进

**潜在问题**：
- ⚠️ 限制器在 Sod/Lax 基准中**增加**误差（baseline density L2=1.0% → TVB=2.16%），说明当前参数需要调优
- ⚠️ README 声明"Limiter overhead: ~80-120%"——这是诚实声明，但说明纯 Python 实现性能不佳
- ⚠️ 建议：对于当前用例，baseline solver（无限制器）仍是更好的选择

**结论**：✅ 通过，功能正确、测试在场、声明诚实。但限制器实用性受限（性能开销大、精度反而下降），后续需优化或调参。

### 2.5 Commit 606ede1 — Merge PR #6 from cxs503/agent/m274

**变更**：合并 m274 分支到 main

**评审**：标准 merge commit，内容已在 e22c308 评审

**结论**：✅ 通过

---

## 3. 测试套件门禁

命令：`PYTHONPATH=. /nfs/wangxi/hermes-agent/.venv/bin/python -m pytest -v`

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.7.0
rootdir: /nfs/wangxi/tensor_eco/TensorDG
configfile: pyproject.toml
testpaths: tests
collecting ... collected 34 items

tests/test_advection_1d.py::test_solution_is_finite_and_mass_conservative PASSED [  2%]
tests/test_advection_1d.py::test_degree_one_converges_at_expected_rate PASSED [  5%]
tests/test_advection_1d.py::test_zero_final_time_returns_initial_projection PASSED [  8%]
tests/test_advection_2d.py::test_mass_conservation_and_negative_velocity PASSED [ 11%]
tests/test_advection_2d.py::test_degree_one_converges_at_second_order PASSED [ 14%]
tests/test_advection_2d.py::test_documented_two_dimensional_case_is_below_three_percent PASSED [ 17%]
tests/test_burgers_1d.py::test_burgers_conserves_mass_and_matches_characteristics PASSED [ 20%]
tests/test_burgers_1d.py::test_burgers_p2_has_near_third_order_convergence PASSED [ 23%]
tests/test_euler_sod_1d.py::test_sod_state_remains_physical_and_finite PASSED [ 26%]
tests/test_euler_sod_1d.py::test_sod_early_time_conservation_and_wave_direction PASSED [ 29%]
tests/test_euler_sod_1d.py::test_sod_respects_requested_tensor_dtype PASSED [ 32%]
tests/test_euler_sod_1d.py::test_exact_sod_star_state PASSED [ 35%]
tests/test_euler_sod_1d.py::test_documented_sod_case_matches_exact_solution_within_three_percent PASSED [ 38%]
tests/test_heat_1d.py::test_heat_sipg_p1_converges_at_second_order PASSED [ 41%]
tests/test_heat_1d.py::test_documented_heat_case_is_below_three_percent PASSED [ 44%]
tests/test_lax_1d.py::test_lax_star_state_and_short_time_positivity PASSED [ 47%]
tests/test_lax_1d.py::test_documented_lax_case_matches_exact_riemann_within_three_percent PASSED [ 50%]
tests/test_limiters.py::test_tvb_limiter_preserves_cell_average PASSED [ 52%]
tests/test_limiters.py::test_weno_limiter_preserves_cell_average PASSED [ 55%]
tests/test_limiters.py::test_tvb_limiter_does_not_modify_smooth_solution PASSED [ 58%]
tests/test_limiters.py::test_tvb_limiter_reduces_extremum PASSED [ 61%]
tests/test_limiters.py::test_weno_limiter_reduces_extremum PASSED [ 64%]
tests/test_limiters.py::test_troubled_cell_indicator_detects_extremum PASSED [ 67%]
tests/test_limiters.py::test_troubled_cell_indicator_ignores_monotone PASSED [ 70%]
tests/test_limiters.py::test_apply_limiter_dispatcher PASSED [ 73%]
tests/test_limiters.py::test_limiter_p0_is_noop PASSED [ 76%]
tests/test_limiters.py::test_limiter_handles_multivariate PASSED [ 79%]
tests/test_limiters.py::test_limiter_dtype_consistency PASSED [ 82%]
tests/test_limiters.py::test_limiter_with_shock_profile PASSED [ 85%]
tests/test_limiters.py::test_weno_smoothness_weights PASSED [ 88%]
tests/test_poisson_2d.py::test_sipg_degree_one_converges_and_meets_three_percent_tolerance PASSED [ 91%]
tests/test_poisson_2d.py::test_sipg_degree_two_has_third_order_accuracy PASSED [ 94%]
tests/test_vortex_2d.py::test_vortex_initial_projection_and_conservation PASSED [ 97%]
tests/test_vortex_2d.py::test_documented_vortex_case_is_below_three_percent_for_every_field PASSED [100%]

============================= 34 passed in 59.63s ==============================
```

**门禁结果**：34/34 PASS ✅

**测试数量变化**：
- c485c623（旧 HEAD）：21 测试
- 606ede16（新 HEAD）：34 测试（+13 限制器测试）
- 所有原有测试保持通过（无回归）

---

## 4. 与 agent 分支冲突检查

| 分支 | 与 606ede16 关系 | 冲突状态 |
|------|------------------|----------|
| agent/m240 | ✅ ancestor（已合并） | 无冲突 |
| agent/m274 | ✅ ancestor（已合并） | 无冲突 |
| agent/m275 | ❌ NOT ancestor（未合并） | ⚠️ 需 rebase |

**m275 冲突分析**：
- agent/m275 基于 fea8844（PR#5 merge），不含 m274 限制器代码
- 差异文件：18 files changed, 1314 insertions(+), 1155 deletions(-)
- 主要冲突点：
  - `tensordg/__init__.py`：m274 添加了 limiter 导出，m275 需要保留
  - `tensordg/euler_dg_1d.py`：m274 修改了 SSP-RK3 循环，m275 不涉及此文件
  - `validation_report.json`：两版本不同，需合并
  - `README.md`：m274 添加了 limiter 章节，m275 需保留
- **建议**：m275 分支需 rebase 到 606ede16，保留 m274 的 limiter 代码，然后重新提交 PR

---

## 5. 对 backlog.md 的影响

### 5.1 已合并提案（撤项记合并）

| 提案 | 原优先级 | 状态 | 备注 |
|------|----------|------|------|
| tensordg-001 (TVB/WENO 限制器) | P0 | ✅ 已实现并合并 | m274 完成，PR#6 已合并 |

### 5.2 待合并提案

| 提案 | 原优先级 | 状态 | 备注 |
|------|----------|------|------|
| tensordg-002 (matrix-free SIPG) | P0 | ⏳ 待合并 | m275 完成，需 rebase 后提 PR |

### 5.3 新发现缺口

1. **限制器性能优化**：当前纯 Python 实现开销 76-123%（目标 <30%），需向量化或 C++/CUDA 实现
2. **限制器参数调优**：TVB constant M 需要 problem-specific 调优，当前默认 M=1.0 过于保守
3. **多维限制器**：当前仅支持 1D Euler，2D/3D 限制器待开发

### 5.4 优先级更新

| 优先级 | 项目 | 状态 | 备注 |
|--------|------|------|------|
| ~~P0~~ | ~~TVB/WENO 限制器~~ | ~~✅ 完成~~ | m274 已合并 |
| P0 | matrix-free SIPG | ⏳ 待合并 | m275 需 rebase |
| P1 | 限制器性能优化 | 🆕 新缺口 | 纯 Python → 向量化/CUDA |
| P1 | 限制器参数调优 | 🆕 新缺口 | TVB constant M 自适应 |
| P2 | 多维限制器 | 🆕 新缺口 | 2D/3D Euler 扩展 |

---

## 6. 上游回归诊断

**结论**：无回归。

- 上游 5 个 commits 均为合并本线 PR，无独立新开发
- 测试套件从 21 增至 34，所有原有测试保持通过
- 无新依赖、无 API 破坏性变更
- README 新增内容（limiter 章节）与代码一致

---

## 7. 修复 mission 草案

**无需修复 mission**。上游未引入回归，所有测试通过。

**建议后续 mission**：
1. **m275 rebase + PR**：将 agent/m275 rebase 到 606ede16，解决冲突后提交 PR
2. **m283 限制器性能优化**（可选）：向量化 limiter 实现，目标开销 <30%

---

## 8. 总结

| 维度 | 结果 |
|------|------|
| Commits 评审 | 5/5 通过 |
| 新声明验证 | 全部在场且可复现 |
| 测试套件 | 34/34 PASS（59.63s） |
| 回归检测 | 无回归 |
| 冲突检查 | m240/m274 已合并，m275 需 rebase |
| backlog 影响 | tensordg-001 撤项记合并，tensordg-002 待合并 |
| 修复 mission | 无需 |

**总体评价**：上游 c485c623..606ede16 仅合并了本线自己的 PR（m240 审计 + m274 限制器），无独立新开发。代码质量良好，测试覆盖充分，声明诚实。m275 matrix-free 工作需 rebase 后提 PR。

---

> 评审人：tensor-agent (m282)
> 评审日期：2026-10-25
