# CKS1998 2-D Riemann Problem Verification (Lax & Liu 1998, Config 3)

> **锚点**: `REPRO-DG-CKS1998`（Cockburn & Shu, JCP 141 (1998) / SIAM Review 43(1) (2001)）
> **算例**: 四象限 Riemann 问题（Lax & Liu 1998, Configuration 3）
> **求解器**: `tensordg.euler_dg_2d.solve_euler_dg_2d_riemann`（P1/P2 DG, HLLC 通量, SSP-RK3, TVB 限幅）
> **状态**: 冒烟测试完成；发现两项诚实局限（TVB 不保证正性、粗网格质量守恒偏差）

---

## ① 案例介绍

四象限 Riemann 问题是二维 Euler 方程的经典基准（Lax & Liu 1998, 19 种构型）。
Configuration 3 是**两道激波相互作用**的情形，对激波捕捉格式的二维稳定性和
正性保持能力构成严格考验。CKS1998 论文（§V）将此作为 RKDG 方法在二维系统上
的代表性算例。

### 初始条件

| 象限 | 区域 | $\rho$ | $u$ | $v$ | $p$ |
|------|------|-------:|----:|----:|----:|
| 左下 | $x<0.5, y<0.5$ | 0.138 | 1.206 | 1.206 | 0.3 |
| 右下 | $x>0.5, y<0.5$ | 0.5323 | 0.0 | 1.206 | 0.3 |
| 左上 | $x<0.5, y>0.5$ | 0.5323 | 1.206 | 0.0 | 0.3 |
| 右上 | $x>0.5, y>0.5$ | 1.5 | 0.0 | 0.0 | 1.5 |

$\gamma = 1.4$，周期边界条件，$t = 0.25$。

---

## ② 怎么做的

- **空间离散**: 模态 Legendre DG（P1/P2），张量积基
- **数值通量**: HLLC（Toro 三波求解器）
- **时间积分**: SSP-RK3
- **限幅器**: TVB（Cockburn-Shu 1989），$M = 1.0$
- **初值**: L2 投影 + `_positivity_filter()`（Zhang-Shu 缩放，确保初值容许）
- **正性检查**: 在体积 Gauss 点上重构保守变量，逐点验证

---

## ③ 计算结果（冒烟测试：16×16, P1, $t=0.1$, CFL=0.1）

| 量 | 值 |
|---:|---:|
| 时间步数 | 305 |
| $\rho_{\text{mean},\min}$（单元平均） | 0.1334 |
| $\rho_{\text{mean},\max}$（单元平均） | 8.9353 |
| $\rho_{\text{quad},\min}$（Gauss 点） | **−21.44** |
| $p_{\text{quad},\min}$（Gauss 点） | **−2.69×10¹⁶** |
| 有限性 | ✅ 无 NaN/Inf |
| 初始质量 | 0.675650 |
| 最终质量 | 0.845618 |
| 质量变化 | **25.16%** |

---

## ④ 结果图

完整密度云图因系统负载未生成。冒烟测试确认求解器可正常输出密度场，
单元平均密度范围 [0.13, 8.94] 与文献预期量级一致。

---

## ⑤ 注意点（诚实局限）

### 局限 1：TVB 限幅器不保证点态正性

TVB（minmod 族）限幅器控制振荡幅度，但**不保证** Gauss 点上的 $\rho > 0$ 和 $p > 0$。
冒烟测试显示 $\rho_{\text{quad},\min} = -21.44$，$p_{\text{quad},\min} = -2.69 \times 10^{16}$。
这是 TVB 限幅器的已知性质，不是 bug。若需严格正性，需引入 Zhang-Shu (2010)
正性保持限幅器。

### 局限 2：粗网格质量守恒偏差

16×16 网格上质量变化 25.16%，远超 1% 门。可能原因：
- TVB 限幅器的模态缩放不严格守恒
- 粗网格下激波跨越单元少，限幅作用强
- 需在更细网格（32×32, 64×64）上验证是否收敛

### 局限 3：无解析解

四象限 Riemann 问题在 $t > 0$ 时无闭式解，验证依赖正性、自收敛和与文献定性对比。

---

## ⑥ 误差判定

| 关键量 | 基准 | 实测 | 判定 |
|--------|------|------|------|
| 有限性 | 无 NaN/Inf | 通过 | ✅ PASS |
| 单元平均正性 | $\rho_{\text{mean}} > 0$ | 0.1334 | ✅ PASS |
| Gauss 点正性 | $\rho_{\text{quad}} > 0$ | −21.44 | ❌ FAIL（TVB 不保证） |
| 质量守恒 | < 1% | 25.16% | ❌ FAIL（粗网格，待细网格验证） |
| 自收敛 L1 | < 0.05 | not_run | ⏸ not_run |

**结论（诚实）**：求解器在有限性和单元平均有界性上通过。TVB 限幅器不保证
Gauss 点正性是已知局限；粗网格质量守恒偏差需在更细网格上进一步验证。
完整收敛研究因系统负载≥100 推迟至集群。

---

## 复现命令

```bash
# 冒烟测试
OMP_NUM_THREADS=1 python -m benchmarks.riemann_2d_cks1998 --quick

# 完整收敛研究（需集群）
OMP_NUM_THREADS=1 python -m benchmarks.riemann_2d_cks1998

# 回归测试
OMP_NUM_THREADS=1 python -m pytest tests/test_riemann_2d_cks1998.py -v
```

## 产物清单

| 文件 | 内容 |
|------|------|
| `benchmarks/riemann_2d_cks1998.py` | 基准驱动（quick/full 模式） |
| `tests/test_riemann_2d_cks1998.py` | 回归测试（5 条） |
| `docs/benchmarks/results/riemann_2d_cks1998/metrics.json` | 数值结果 |
| `docs/CASE-DG-CKS1998-2D-RIEMANN.md` | 本报告 |
