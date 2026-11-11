# TensorDG: 等温壁能量闭合缺陷的定量刻画（残差 ∝ κ·ρ_wall）

> 类型：**验证欠账的定量刻画 + 契约测试**（`tensordg/ns_dg_2d.py` 壁面 `kappa dT/dn` 通量）
> 触发：`tests/test_ns_heat_conduction.py` 的两条 `xfail(strict=True)`——「精确稳态在壁胞的能量残差不随加密衰减」。
> 该断言只说"不衰减"，没说缺陷长什么样；本单把缺陷的**形状**测出来，给后续修复一个可判据的靶子。
> 工件：`validation/ns_heat_wall_closure/wall_energy_scaling.json`（程序化生成）、
> `validation/ns_heat_wall_closure/probe_wall_energy_scaling.py`（可重跑探针）、
> `tests/test_ns_wall_energy_scaling.py`（6 条，全 PASS）、
> 图：`docs/figures/ns_wall_energy_scaling/residual_vs_h.png`、`kappa_linearity.png`
> 刻意**不改** `tensordg/`：两条 xfail 保持原样（honest FAIL 不得覆盖）；本单交付的是缺陷刻画与判据，不是修复。

## 1. 案例介绍（物理背景/几何/参数表）

算子：`tensordg.ns_dg_2d._rhs_ns`（二维张量积模态 DG、HLLC 无粘通量、BR1/SIPG 粘性通量、
等温无滑移壁 Dirichlet 闭合 `wall_gradient="taylor"`）。

算例：**精确稳态热传导**，在可压缩 Navier–Stokes 方程下恒等成立：

$$p=\text{const},\quad u=v=0,\quad T(y)=T_{bot}+(T_{top}-T_{bot})y,\quad \rho=p/T,$$

于是 $\nabla\cdot(\kappa\nabla T)=0$（$T$ 线性）、$\nabla p=0$、$u=0$，壁面 Dirichlet 值
$T_{bot}=1$、$T_{top}=0.5$ 与内部剖面**逐点一致**：精确解应当就是离散稳态。

| 参数 | 值 |
|---|---|
| 域 | $[0,1]\times[0,1]$，$n_x=2$（x 周期），$n_y$ 加密 |
| 多项式阶 | $p=1$（Gauss–Legendre $p+2$ 点求积） |
| $\mu$ | 0.1 |
| Prandtl | 0.71 |
| $\kappa=\mu c_p/Pr$ | 0.4929577465 |
| 压力 $p$ | 1.0 |
| $T_{bot}/T_{top}$ | 1.0 / 0.5 |
| $\rho_{bot}/\rho_{top}$ | 1.0 / 2.0 |
| 壁面闭合 | `wall_gradient="taylor"`，penalty $=2(p+1)^2=8$ |

## 2. 怎么做的（方法/流程）

1. 把精确解投影到 DG 模态系数（`_initial_coefficients`），**不跑时间推进**：直接取一步残差 `_rhs_ns`。
   精确解应当满足离散方程，残差的任何非零成分都是离散缺陷（且随加密必须收敛到 0）。
2. 分通道取残差：质量行、动量两行、能量行；能量行再按壁胞（`ymin`/`ymax` 单元）与内部单元分开，
   并按模态拆（mode 0 = 单元平均，mode ≥ 1 = 形状）。
3. 三次标度实验定位缺陷来源：$\kappa\times\{0.5,1,2\}$、$p\times\{0.5,1,2\}$，
   其余参数不动——若缺陷来自粘性应力或压力功项，这两种标度会分道扬镳。
4. 加密序列 $n_y\in\{16,32,64,128\}$ 测渐近值，与 $\kappa\rho_{wall}$ 比较。
5. 把上述事实写成 6 条契约测试（含 2 条正控：质量/动量行必须机器零；内部残差必须仍是一阶）。

## 3. 计算结果（程序化注入，零手抄）

### 3.1 加密序列（`wall_energy_scaling.json` 直出）

| $n_y$ | mass row | momentum rows | wall energy (bottom, mode 0) | wall energy (top, mode 0) | top wall modes $\ge 1$ | interior energy | bottom$/\kappa$ | top$/\kappa$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1.0e-14 | 0.0e+00 | 0.542124 | 0.814997 | 4.6e-13 | 0.170932 | 1.0997 | 1.6533 |
| 32 | 3.8e-14 | 0.0e+00 | 0.516931 | 0.896597 | 2.7e-15 | 0.092912 | 1.0486 | 1.8188 |
| 64 | 7.3e-14 | 0.0e+00 | 0.504798 | 0.940217 | 2.5e-13 | 0.048554 | 1.0240 | 1.9073 |
| 128 | 8.1e-14 | 0.0e+00 | 0.498842 | 0.962796 | 5.9e-13 | 0.024835 | 1.0119 | 1.9531 |

**读法**：质量行与动量行在**所有** $n_y$ 上都是 $0$（机器零）；能量行的壁胞残差**不衰减**，
反而单调趋于非零常数（下壁 $\to\kappa\rho_{bot}=0.4930$，上壁 $\to\kappa\rho_{top}=0.9859$），
而**内部**单元能量残差仍以一阶（$h^1$）衰减（$n_y$: 64→128 比值 1.955）。
上壁能量残差的 mode ≥ 1 分量恒为 $0$：缺陷是**纯单元平均的通量偏置**，不是形状误差。

### 3.2 标度实验：缺陷只住在 $\kappa\,dT/dn$ 里

$\kappa$ 标度（$n_y=32$，$p$ 固定）：

| $\kappa$ 倍数 | 下壁残差 | 上壁残差 |
|---:|---:|---:|
| 0.5 | 0.258466 | 0.448298 |
| 1 | 0.516931 | 0.896597 |
| 2 | 1.033862 | 1.793194 |

线性度相对散布 = `0.0`（浮点 round-off 级）。
压力标度（$n_y=32$，$\kappa$ 固定）：

| $p$ 倍数 | 下壁残差 | 上壁残差 |
|---:|---:|---:|
| 0.5 | 0.516931 | 0.896597 |
| 1 | 0.516931 | 0.896597 |
| 2 | 0.516931 | 0.896597 |

压力独立性相对散布 = `0.0`。

**结论**：缺陷**精确正比于 $\kappa$**、**完全与 $p$ 无关**⇒ 不是粘性应力项、不是压力功/焓通量项，
只能是热传导通量在壁面与内部界面之间的**系统性 $O(h)$ 偏置**；且该偏置带一阶$\rho$量纲
（渐近值恰为 $\kappa\rho_{wall}$）。`_gas_gradients` 的 `conversion_state` 文档已指出同类机制：
商法则 $dT=(dp-T\,d\rho)/\rho$ 的**分母取法**决定了同一单元在两个面上的 $\nabla T$ 是否一致，
内部面两侧误差抵消、壁面只剩一侧，于是偏置不被抵消。这是**待验证的机制假设**，不是本单已证结论。

### 3.3 图

- `docs/figures/ns_wall_energy_scaling/residual_vs_h.png`：三种残差 vs $h$（log–log）+ 两条 $\kappa\rho_{wall}$ 渐近线。
  壁胞残差贴着水平渐近线，内部残差是 $h^1$ 直线。
- `docs/figures/ns_wall_energy_scaling/kappa_linearity.png`：$\kappa$ 加倍、残差加倍。

## 4. 注意点（坑/局限/适用范围）

1. **只对 $p=1$ 测过**；$p\ge2$ 时一侧面导数的渐近阶不同（`_wall_normal_derivative` 文档已记录 P3 变差），
   本规律是否保持需另测——本单不声明。
2. 用 `wall_gradient="taylor"`（探针默认）；`face`/`one_sided`/`fd2`/`recovered` 路径未测。
   两条既有 xfail 说明两种重构都中招，但"渐近值都是 $\kappa\rho_{wall}$"未验证。
3. 残差是**一步 RHS**，不是时间推进误差；`test_ns_heat_conduction.py` 已证明 $t=0.05$ 内精确态被保持，
   说明该偏置在短时内被其它项吸收——不要在短时保持性上覆盖本结论。
4. $n_y=128$ 时仍有 1.19%（下壁）/-2.34%（上壁）的有限分辨率偏差，
   故契约测试用 3% 容差并固定在 $n_y=128$；这是本单最紧的可复现判据。
5. 本单**没有修复缺陷**：`tensordg/` 一行未动。仓库里两条 `xfail(strict=True)` 原样保留。

## 5. 误差判定（关键量 vs 基准，逐项）

基准：解析事实「精确稳态的离散残差 = 0」。

| 关键量 | 基准 | 实测（$n_y=128$） | 相对偏差 | <3%? |
|---|---|---|---|---|
| 质量行残差 | 0 | 8.1e-14 | 机器零 | 是 |
| 动量行残差 | 0 | 0.0e+00 | 机器零 | 是 |
| 壁胞能量残差 mode ≥ 1 | 0 | 5.9e-13 | 机器零 | 是 |
| 下壁能量残差 / $\kappa\rho_{bot}$ | 0 | 1.0119 | +1.19% | **否** |
| 上壁能量残差 / $\kappa\rho_{top}$ | 0 | 0.9766 | -2.34% | **否** |
| 内部能量残差阶 | $h^1$ | 比值 1.955 | 一阶成立 | 是 |

**显式回答**：等温壁能量闭合**未达 <3%**（壁胞残差相对基准为 100%，即完全不衰减）。
同算子的质量/动量行与能量模态形状达机器精度。这是已知缺陷的定量确认，不是新缺陷。

## 6. 下一步（给修复单的判据）

1. 让 `conversion_state` 在壁面与内部界面保持一致（同一单元的 $\nabla T$ 在两侧相同），
   然后重跑本探针：壁胞残差应从"趋于 $\kappa\rho_{wall}$"翻转成"$\to 0$ 且按 $h^1$ 衰减"。
2. 修复后按仓库棘轮惯例翻转 `tests/test_ns_heat_conduction.py` 的两条 xfail **和**
   `tests/test_ns_wall_energy_scaling.py::test_wall_energy_residual_carries_the_density_offset`
   （后者是刻画现状的正向棘轮，修复会让它失败——这是预期的再裁定信号）。
3. 复测 $p=2,3$ 与其余 `wall_gradient` 分支，确认规律与修复方向在全部路径一致。
