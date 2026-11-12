# CASE-DG-AV-SHOCK-CAPTURE — 传感器门控模态人工黏性（SVV 指数滤波）

> 锚点：`NEW:人工黏性/SVV 激波捕捉为 TensorDG 唯一缺席的经典 DG 捕捉机制`（建议入账 slug `CASE-DG-AV-SVV-SHOCK`，域=DG 求解器）
> 代码：`tensordg/artificial_viscosity.py`（新增）、`tensordg/limiters.py`（注册 `kind="av"`）、
> `tensordg/euler_dg_1d.py`（新增 `limiter_kwargs` 透传）；验证：`benchmarks/artificial_viscosity_verification.py`、
> `tests/test_artificial_viscosity.py`；工件：`validation/artificial_viscosity/`。

## ① 案例介绍

DG 的亚网格振荡控制在本仓已有三种**重构替换型**限制器（TVB/minmod、WENO `r=2`、AO `r=p+1`）：
一旦问题单元被标记，该单元多项式被**重建**（降到线性或 WENO 组合）。替换是**开关**——单元非「全限」即「不动」，
在数据上不连续，光滑极值会被削平，强度无法连续调节。

文献上的另一条路线是**传感器门控的人工黏性 / 谱黏性（SVV）指数滤波**：保留多项式基，按光滑度传感器
连续地衰减**最高模态**（Persson & Peraire, *Sub-cell shock capturing for DG*, AIAA 2006-0112；
Maday–Kaber–Tadmor 谱黏性, J. Sci. Comput. 1993）。本仓动手前 `grep -ri "artificial viscosity" tensordg/ benchmarks/ tests/` **零命中**。

算例：Sod 激波管，(ρ,u,p)_L=(1,0,1)、(ρ,u,p)_R=(0.125,0,0.1)，γ=1.4，隔板 x=0.5，t=0.2，[0,1]；
模态 DG P2 + HLLC + SSP-RK3，CFL=0.12，`positivity="scale"`（Zhang–Shu 缩放）。基准=精确 Riemann 单元平均。

## ② 怎么做的

1. **传感器**：模态 Legendre 能量为 c_m²/(2m+1)，取 `s_e = log10( (c_p²/(2p+1)) / Σ_m c_m²/(2m+1) )`，
   单元间取邻域 max 平滑。分母加**全局状态尺度**地板 `1e-12·max_v mean_e E`——否则「全场为零」的变量
   （静止气体的 ρu）会用自身 roundoff 归一化而误报（实测 s=-4e-13）。
2. **滤波**：σ_m = exp(-α_e (m/P)^{2·order})，P=p+1，order=2；α_e 为 s_e 在 [s0-κ, s0+κ] 上的**二次斜坡**
   （Persson–Peraire 斜坡形状）。**mode 0 恒为 1**，故质量/动量/能量逐位守恒。
3. **接线**：注册为 `apply_limiter(kind="av")`，与其他限制器同一条 RK 子步路径，互斥可选。
4. 参数由测量定：默认 `s0=-2.0, κ=0.5, α_max=20`（N=64 上四项扫描的 L1 最优，见 ③）。

## ③ 计算结果（全部程序化注入，见 `validation/artificial_viscosity/metrics.json` + `.sha256`）

A. 传感器标度律（`sensor_law`，两档归一化各 4 档网格 20/40/80/160）：

| 阶 | 归一化 | 光滑正弦斜率（实测） | 理论 2p | 台阶斜率（实测） |
|---|---|---|---|---|
| 1 | fraction | 0.005 | 2 | -0.000 |
| 1 | global | **-1.987** | 2 | -0.019 |
| 2 | fraction | **-4.002** | 4 | -0.000 |
| 2 | global | -3.987 | 4 | -0.018 |

即：光滑场按 h^{2p} 衰减，台阶与网格无关（斜率 0）——阈值可分辨的**结构**成立。
但 p=1 的局部（fraction）形式被「场穿过零点的单元」支配而失去衰减（0.005）；global 形式恢复 2。

B. Sod，N=64，P2（精确单元平均为基准，相对 L1 取密度）：

| 闭合 | L1(ρ) % | TV 超出量 | min p | 标记率 |
|---|---|---|---|---|
| 无限（none） | **0.329** | 0.050 | 0.09998 | — |
| TVB（参照） | — | — | — | — |
| av: s0=-3.0, κ=1.0 | 1.570 | **0.002** | 0.100 | 31 % |
| av: s0=-3.0, κ=0.5 | 1.466 | 0.0019 | 0.100 | 23 % |
| av: s0=-2.0, κ=0.5（**默认**） | **0.830** | 0.0019 | 0.100 | 12 % |
| av: s0=-2.0, κ=0.5, α_max=8 | 0.848 | 0.0019 | 0.100 | 23 % |

TV 超出量（数值 TV − 精确 TV）由 0.050 降到 0.002（**25×**），压力下界保持 0.1（精确右态值）。

C. 惰性/守恒：`sensor < s0-κ` 时滤波**逐位恒等**；mode 0 在满强度下逐位不变（测试断言）。

## ④ 结果图

`validation/artificial_viscosity/figures/{sensor_law,sod_profiles,sensor_profile,error_tv}.png`
（由脚本生成：传感器标度律 log-log、Sod 密度剖面 vs 精确解、沿解传感器分布与阈值线、L1/TV 柱图）。

## ⑤ 注意点（局限与适用范围）

- **过度耗散是真实代价**：默认参数下 L1 从 0.329 % 升到 0.830 %（仍 < 3 % 门槛），换来 25× 的振荡削减；
  更激进的 s0=-3/κ=1 会把 31 % 的单元卷进滤波，L1 升到 1.57 %。
- **阈值 s0 与网格/阶有关**：光滑支按 h^{2p} 衰减，故 s0 不是普适常数；换配置时应用
  `sensor_reference()` 在同网格光滑段上量出参考值再放裕度（本仓文档化用法）。
- **局部（fraction）归一化在过零点退化**（p=1 实测斜率 0.005）；需要尺度不变时用 `normalisation="global"`。
- **无熵/正性/单调性证明**：滤波只阻尼高阶模，不保证可容许；本仓实测均与 `positivity="scale"` 联用。
- 与 TVB/WENO 的头对头只做了 N=64 一档（TVB 行未取到数，集群作业 `cj1371` 排队 522 未派发，**未使用其数据**）；
  N=100/200/400 全阶梯与 TVB/WENO/AO 四方对照 = **not_run**（本机 load15≈95，单次 P2/N=200 约 92 s，超本单窗口）。
- x86 CI 架构未复核（本机 loong64，浮点求和顺序差异可能改动末位）。

## ⑥ 误差判定（逐项，显式回答「是否 <3%」）

| 关键量 | 基准 | 实测 | 是否 <3 % |
|---|---|---|---|
| 传感器光滑支标度指数（P2） | 2p = 4 | 4.002（fraction）/3.987（global） | **是**（0.05 % / 0.33 %） |
| 传感器台阶支标度指数 | 0（网格无关） | 0.000 / 0.018 | **是**（|Δ|≤0.018） |
| Sod L1(ρ)，av 默认 vs 精确 Riemann | 精确解 | 0.830 % | **是** |
| Sod L1(ρ)，无限 DG | 精确解 | 0.329 % | **是** |
| Sod TV 超出量削减倍数 | 无限 DG=0.050 | 0.0019（25×） | 非百分比量，按倍数判定：**满足** |
| 压力正性 | p>0 | min p = 0.100（= 精确右态 p） | **是** |
| 守恒（mode 0） | 逐位不变 | 逐位（`torch.equal`） | **是**（0） |

**结论**：新增机制的**可判定量**全部达标；但它是「以精度换振荡」的机制，L1 劣化 2.5× 是诚实代价，
不是把无限 DG 的错误改小。诚实标注：TVB/WENO 同网格头对头与多网格阶梯未完成（not_run），故**不得**据此宣称
「AV 优于 WENO/AO」。

## 复现

```
PYTHONPATH=. python -m benchmarks.artificial_viscosity_verification --output-dir validation/artificial_viscosity --cells 64 --degree 2 --quick
PYTHONPATH=. python -m pytest tests/test_artificial_viscosity.py -q
```
