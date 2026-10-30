# 对标差异表：deal.II 收敛表纪律 → TensorDG

## 1. deal.II 收敛表机制提炼

| 特性 | deal.II 实现 | TensorDG 实现 | 状态 |
|------|-------------|--------------|------|
| **h 细化序列** | `GridGenerator::hyper_cube` + `GridTools::globally_refine` 循环 | `ncells_seq = [base * 2^i for i in range(n_levels)]` | ✅ 等效 |
| **p 细化序列** | `FE_Q(degree)` 循环 | `degree` 参数化，外部循环 | ✅ 等效 |
| **误差范数** | `VectorTools::integrate_difference` (L2, H1, L∞) | 高斯求积 L2 + 逐点 L∞ | ✅ 等效 |
| **观测阶** | `ConvergenceTable::evaluate_all_convergence_rates()` | `log2(err_coarse / err_fine)` | ✅ 等效 |
| **Richardson 外推** | `ConvergenceTable::add_extra_column("error(h^p)_extrapolated")` | `|e_fine - e_medium| / (r^p - 1)` | ✅ 等效 |
| **收敛表输出** | `ConvergenceTable::write_text()` / LaTeX / gnuplot | `print_table()` + JSON + matplotlib | ✅ 等效 |
| **自动断言** | 无内置断言（需手动检查） | `assert_order(tolerance=0.3)` | ✅ 超越 |
| **GCI 计算器** | 无内置 | `compute_gci()` + `gci_three_grid()` | ✅ 新增 |

## 2. 关键差异与改进

### 2.1 自动断言机制（超越 deal.II）
- deal.II 的 `ConvergenceTable` 仅计算和输出，不自动验证
- TensorDG 的 `ConvergenceTable.assert_order()` 在观测阶 < 理论阶 - 0.3 时自动失败
- 支持 `assert_all_orders()` 批量验证所有细化级别

### 2.2 GCI 网格收敛指数（新增）
- 实现 Roache (1994) / Celik et al. (2008) 标准
- 三网格分析：表观阶、GCI_fine、GCI_coarse、外推值、渐近范围检验
- 单元测试覆盖 p=1,2,3 精确幂律 + 非渐近情况

### 2.3 格式覆盖
| 格式 | deal.II 教程 | TensorDG |
|------|-------------|----------|
| SIPG Poisson | step-4, step-12 | `converge_poisson.py` |
| DG Advection | step-12 (hyperbolic) | `converge_advection.py` |
| TVB Limiter | 无 | `converge_advection.py --limiter tvb` |
| WENO Limiter | 无 | `converge_advection.py --limiter weno` |

## 3. 收敛表结果摘要

### 3.1 SIPG Poisson

| Degree | 理论阶 | 观测阶 (L2) | Richardson 外推 | 状态 |
|--------|--------|------------|----------------|------|
| p=1 | 2 | 1.9999 | 1.19e-04 | ✅ PASS |
| p=2 | 3 | 2.9971 | 3.76e-06 | ✅ PASS |
| p=3 | 4 | 3.9997 | 1.36e-09 | ✅ PASS |

### 3.2 Advection DG (p=2)

| Limiter | 理论阶 | 观测阶 (L2) | 观测阶 (L∞) | 状态 |
|---------|--------|------------|------------|------|
| None | 3 | 3.0000 | 2.9995 | ✅ PASS |
| TVB | 3 | 3.0000 | 2.9995 | ✅ PASS |
| WENO | 3 | 3.0000 | 2.9995 | ✅ PASS |

### 3.3 GCI 分析 (Poisson p=2)

| 指标 | 值 |
|------|-----|
| 表观阶 | 2.9971 |
| GCI_fine (21) | 0.000125% |
| GCI_fine (32) | 0.000125% |
| 外推值 | 3.76e-06 |
| 渐近范围 | ✅ Yes |

## 4. 验收判据对照

| 判据 | 要求 | 实际 | 状态 |
|------|------|------|------|
| h 细化 ≥4 档 | ≥4 | 5 档 | ✅ |
| 观测阶 vs 理论阶断言 | 容差 0.3 | 全部通过 | ✅ |
| SIPG Poisson 收敛表 | 1 张 | 3 张 (p=1,2,3) | ✅ |
| TVD/WENO 收敛表 | 各 1 张 | 各 1 张 (p=2) | ✅ |
| GCI 计算器 | 入库+单测 | `converge.py` + 22 单测 | ✅ |
| 零手抄 | 程序化生成 | 全自动 | ✅ |
| 收敛表脚本 exit 可失败 | 观测阶 < 理论-0.3 即红 | `assert_order()` 实现 | ✅ |

## 5. 文件清单

| 文件 | 用途 |
|------|------|
| `tensordg/converge.py` | 核心收敛表模块 + GCI 计算器 |
| `tensordg/converge_poisson.py` | SIPG Poisson 收敛表生成器 |
| `tensordg/converge_advection.py` | TVD/WENO 收敛表生成器 |
| `tensordg/run_all_convergence.py` | 主运行器（生成所有表+图） |
| `tests/test_converge.py` | GCI + ConvergenceTable 单元测试 (22 项) |
| `artifacts/converge_poisson_p{1,2,3}.json` | Poisson 收敛表数据 |
| `artifacts/converge_advection_{none,tvb,weno}_p2.json` | Advection 收敛表数据 |
| `artifacts/gci_poisson_p2.json` | GCI 分析结果 |
| `artifacts/convergence_plots.png` | 收敛曲线图 |
| `docs/benchmark_convergence_dealii_tensordg.md` | 本对标差异表 |
