# Burgers 方程正弦初值 Benchmark — TensorDG 验证报告

**算例编号**: CASE-BURGERS-SIN  
**仓库**: TensorDG (branch: agent/m15303)  
**日期**: 2026-11-12  
**复现命令**: `bash artifacts/burgers_sine_benchmark/repro_cmd.sh`

---

## ① 问题描述与参考出处

### 物理背景
一维无粘 Burgers 方程是非线性双曲守恒律的标准验证算例，兼具非线性波 steepening 与解析可解性：

$$u_t + \left(\frac{u^2}{2}\right)_x = 0, \quad x \in [0, 1], \quad \text{periodic BC}$$

### 初值条件
$$u(x, 0) = \bar{u} + A \sin(2\pi x) = 0.2 + 0.8 \sin(2\pi x)$$

### 精确参考解（Hopf-Cole / 特征线法）
在激波形成前（光滑区），精确解由隐式特征线方程给出：

$$u(x, t) = 0.2 + 0.8 \sin\bigl(2\pi(x - u \cdot t)\bigr)$$

通过 Newton 迭代求解（收敛阈值 1e-14，最多 40 次迭代）。

**文献出处**:
- Hopf, E. (1950). "The partial differential equation ut + uux = (μ/2π)uxx". *Communications on Pure and Applied Mathematics*, 3(3), 201-230.
- Cole, J. D. (1951). "On a quasi-linear parabolic equation occurring in aerodynamics". *Quarterly of Applied Mathematics*, 9(1), 72-80.
- 亦可参见 Whitham, G. B. (1974). *Linear and Nonlinear Waves*. Wiley, Chapter 2.

### 激波形成时间
$$t_c = \frac{1}{\max|u_0'(x)|} = \frac{1}{A \cdot 2\pi} = \frac{1}{0.8 \times 2\pi} \approx 0.1989$$

本算例取 t = 0.1 < t_c，处于光滑区，无激波，可用连续精确解验证。

---

## ② 方法与实现

### 数值方法
| 组件 | 选取 |
|------|------|
| 空间离散 | Modal DG（Legendre 基），Rusanov 通量 |
| 时间积分 | SSP-RK3（Shu-Osher 三阶保单调格式） |
| 多项式阶数 | P1 / P2 / P3 扫描 |
| 网格 | 均匀周期网格 N = 8, 16, 32, 64, 128 |
| CFL | 0.1（自适应时间步） |
| 精度 | float64 |

### 实现位置
- 求解器: `tensordg/burgers_1d.py` — `solve_burgers()`, `exact_profile()`, `burgers_error()`
- Benchmark 驱动: `benchmarks/burgers_sine_benchmark.py`
- 可视化: `benchmarks/plot_burgers_sine.py`
- 测试: `tests/test_burgers_1d.py`

### 关键算法细节
1. **Rusanov 通量**: F* = ½(f(u⁻) + f(u⁺)) - ½α(u⁺ - u⁻)，α = max(|u⁻|, |u⁺|)
2. **SSP-RK3**: 三阶保单调时间推进，保证非线性稳定性
3. **精确解 Newton 迭代**: 求解 u = u₀(x - ut) 的隐式方程，利用 ∂/∂u 的解析导数

---

## ③ 参数表（无量纲数完整）

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| 均值 | ū | 0.2 | 背景流速度 |
| 振幅 | A | 0.8 | 正弦扰动幅度 |
| 域长 | L | 1.0 | 周期域 |
| 最终时间 | T | 0.1 | 光滑区（T/T_c = 0.503） |
| 激波时间 | T_c | 0.1989 | 1/(A·2π) |
| 雷诺数 | Re | ∞ | 无粘（Euler 极限） |
| 最大波速 | max|u| | 1.0 | ū + A |
| CFL 数 | CFL | 0.1 | 时间步控制 |
| 多项式阶 | p | 1, 2, 3 | 收敛性扫描 |
| 网格单元 | N | 8–128 | h-收敛扫描 |

---

## ④ 对照表（ref vs got + 误差判定）

### 4.1 h-收敛（t = 0.1）

| 阶数 | N | h=1/N | 相对 L2 误差 (%) | EOC | <5% |
|------|---|-------|------------------|-----|-----|
| P1 | 8 | 0.125 | 4.5919 | — | ✅ |
| P1 | 16 | 0.0625 | 1.2200 | 1.91 | ✅ |
| P1 | 32 | 0.03125 | 0.3312 | 1.88 | ✅ |
| P1 | 64 | 0.015625 | 0.0873 | 1.92 | ✅ |
| P1 | 128 | 0.0078125 | 0.0226 | 1.95 | ✅ |
| **P2** | **8** | **0.125** | **0.4339** | **—** | **✅** |
| **P2** | **16** | **0.0625** | **0.0992** | **2.13** | **✅** |
| **P2** | **32** | **0.03125** | **0.0133** | **2.90** | **✅** |
| **P2** | **64** | **0.015625** | **0.0018** | **2.87** | **✅** |
| **P2** | **128** | **0.0078125** | **0.0002** | **2.92** | **✅** |
| P3 | 8 | 0.125 | 0.1658 | — | ✅ |
| P3 | 16 | 0.0625 | 0.0104 | 3.99 | ✅ |
| P3 | 32 | 0.03125 | 0.0008 | 3.68 | ✅ |
| P3 | 64 | 0.015625 | 0.00006 | 3.87 | ✅ |
| P3 | 128 | 0.0078125 | 0.000004 | 3.92 | ✅ |

**判定**: 所有配置相对 L2 误差 < 5% ✅  
**最优配置 P2 N=128**: 相对 L2 = 0.000241% ≪ 5% ✅

### 4.2 时间敏感性（P2, N=64）

| t | t/t_c | 相对 L2 (%) | 步数 | <5% |
|---|-------|-------------|------|-----|
| 0.01 | 0.050 | 0.0004 | 32 | ✅ |
| 0.05 | 0.251 | 0.0006 | 160 | ✅ |
| 0.10 | 0.503 | 0.0018 | 320 | ✅ |
| 0.15 | 0.754 | 0.0139 | 480 | ✅ |
| 0.19 | 0.955 | 0.8757 | 608 | ✅ |

### 4.3 收敛阶验证

| 阶数 p | 理论阶 p+1 | 实测 EOC | 匹配 |
|--------|-----------|----------|------|
| P1 | 2 | 1.88–1.95 | ✅ |
| P2 | 3 | 2.87–2.92 | ✅ |
| P3 | 4 | 3.68–3.99 | ✅ |

---

## ⑤ 收敛性与参数敏感性

### 5.1 h-收敛阶
- P1: EOC ≈ 1.9 → 接近理论 2 阶
- P2: EOC ≈ 2.9 → 接近理论 3 阶
- P3: EOC ≈ 3.9 → 接近理论 4 阶

**结论**: 空间离散达到设计精度，非线性通量和时间积分未引入额外耗散主导项。

### 5.2 时间推进影响
- t/t_c < 0.5 时误差极低（<0.002%）
- t/t_c → 1（接近激波形成）时误差增长但仍 <1%
- 原因：波形 steepening 导致梯度增大，DG 需要更高分辨率

### 5.3 多项式阶 vs 网格密度
- P3 N=32 (96 DOF) 精度 ≈ P2 N=64 (192 DOF)
- 高阶方法在光滑区效率优势明显

---

## ⑥ 失败模式与工件

### 6.1 已知限制
1. **仅限光滑区**: t ≥ t_c ≈ 0.199 时形成激波，需 TVB limiter（见 `burgers_shock_1d.py`）
2. **一维周期域**: 不验证多维/非周期边界
3. **无粘**: 不验证粘性项（见 `heat_1d.py` / `advection_diffusion_1d.py`）

### 6.2 潜在失败模式
- **CFL 过大**: 非线性问题 CFL > 0.3 可能导致 SSP-RK3 失稳
- **后激波时间**: t > t_c 时 Newton 迭代不收敛（特征线交叉）
- **低精度浮点**: float32 在 P3 N=128 时可能引入舍入误差主导

### 6.3 工件清单

| 文件 | 说明 |
|------|------|
| `artifacts/burgers_sine_benchmark/metrics.json` | 全量数值结果（JSON） |
| `artifacts/burgers_sine_benchmark/profile_p2_n128_t01.csv` | P2 N=128 剖面对比数据 |
| `artifacts/burgers_sine_benchmark/fig1_profile_comparison.png` | 数值 vs 精确剖面图 |
| `artifacts/burgers_sine_benchmark/fig2_convergence.png` | h-收敛曲线 |
| `artifacts/burgers_sine_benchmark/fig3_time_evolution.png` | 波形 steepening 时间演化 |
| `artifacts/burgers_sine_benchmark/fig4_error_vs_time.png` | 误差随时间增长 |
| `artifacts/burgers_sine_benchmark/repro_cmd.sh` | 一键复现脚本 |

### 6.4 生态摩擦（无）
本算例完全复用 TensorDG 现有能力，无额外开发需求。基础设施成熟。

---

## 误差判定总结

| 关键量 | 基准值 | 计算值 | 相对误差 | <5% |
|--------|--------|--------|----------|-----|
| 相对 L2 误差 (P2 N=128) | 0 (exact) | 0.000241% | — | ✅ |
| 收敛阶 P2 | 3 | 2.92 | 2.7% | ✅ |
| 质量守恒 | 0.2 | 0.200000000000 | <1e-12 | ✅ |

**最终判定**: ✅ PASS — 所有指标满足 <5% 容差要求。

---

## 复现命令

```bash
cd /root/tp-agent/work/m15303
bash artifacts/burgers_sine_benchmark/repro_cmd.sh
```

或分步执行：
```bash
python benchmarks/burgers_sine_benchmark.py
python benchmarks/plot_burgers_sine.py
```
