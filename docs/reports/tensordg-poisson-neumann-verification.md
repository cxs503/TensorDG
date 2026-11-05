# TensorDG：SIPG Poisson 的 Neumann（自然）边界条件 —— 实现与解析基准验证

> 分支 `agent/m3478`（自找活 m3478，TensorDG）。工件：`validation/poisson_neumann.json`、
> `validation/poisson_neumann_convergence.png`；复现：`python -m benchmarks.verify_poisson_2d_neumann`
> （默认网格，100 s 量级，单机 CPU）。

## 1. 案例介绍：为什么需要弹这件活

`tensordg/poisson_2d.py` 与它的 sparse / matrix-free 后端把 **四边齐次 Dirichlet** 和单一制造解
`u = sin(pi x) sin(pi y)` 硬编码在装配里：边界类型不是参数，只有一种可能。全仓 `grep -i neumann`
只命中若干篇报告里的散文（讲壁面罚项调小会退化成"近 Neumann"），没有一行装配代码。也就是说：

* 平台能算的扩散问题只有"四边给值"一种；绝热壁、对称面、给定热流、纯 Neumann 相容性问题
  一律无法表达，也不存在任何验证证据；
* 一旦有人绕过 API 手改矩阵，唯一的判据（3 % 文档门 + 收敛阶）也没有被搬到新路径上。

**物理与几何**：单位正方形 $\Omega=(0,1)^2$ 上的一致 Cartesian 网格，模态 Legendre 基、张量积
局部空间 $Q_p$，SIPG 对称内罚格式（Arnold, Brezzi, Cockburn & Marini, *SIAM J. Numer. Anal.*
**39** (2002) 1749–1779, eq. (2.9)）。弱形式

$$a(u,v)=\int_\Omega\nabla u\cdot\nabla v
-\sum_{\Gamma_D}\big(\{\nabla u\cdot n\}[v]+\{\nabla v\cdot n\}[u]\big)
+\sum_{\Gamma_D}\frac{\sigma}{h}[u][v]
=\int_\Omega f v+\sum_{\Gamma_N}\int_{\Gamma_N} g\,v ,$$

即 **Neumann 面只改右端项**（一致性载荷 $\int g v$），不进入矩阵；$g=\partial u/\partial n$ 取外法向。

三个制造解（都能闭式求导，$f=-\Delta u$ 与 $g$ 都是程序生成的）：

| 代号 | 精确解 | 边界 | $f=-\Delta u$ | Neumann 数据 |
|---|---|---|---|---|
| A | $\cos(\pi x)\cos(\pi y)$ | 四边全 Neumann | $2\pi^2\cos(\pi x)\cos(\pi y)$ | $g\equiv 0$ |
| B | $\cos(\frac{\pi x}{2})\cos(\frac{\pi y}{2})$ | 左/下 Neumann，右/上 Dirichlet 0 | $\frac{\pi^2}{2}\cos(\frac{\pi x}{2})\cos(\frac{\pi y}{2})$ | $g\equiv 0$ |
| C | $\sin(\pi x)\sin(\pi y)$ | 左边 Neumann，其余 Dirichlet 0 | $2\pi^2\sin(\pi x)\sin(\pi y)$ | $g=-\pi\sin(\pi y)$ |

A 的相容性 $\int_\Omega f=0$ 且 $\int_{\partial\Omega}g=0$ 自动成立，但它 **只把解定到相差一个常数**
（离散算子的零空间就是常数），必须处理奇异系统；C 则专门检查非齐次 $g$ 的载荷项，A 和 B 的
$g$ 都为零，只看它们无法证明"载荷接对了"。

## 2. 怎么做（方法 / 流程 / 工具链）

1. **先把面算子抽出来**（`tensordg/poisson_2d.py`）：原实现把内面 SIPG 块和 Dirichlet 边界块
   内联在 200 行的 `solve_poisson_2d` 里。抽出 `interior_face_operator()` 与
   `dirichlet_face_operator()` 两个模块级函数，原求解器改为调用它们，**装配公式一个字没改**——
   这是让新路径可以复用同一份面算子、而不是复制一份公式（复制会漂移）的前提。
2. **新模块 `tensordg/poisson_2d_bc.py`**：`solve_poisson_2d_bc(bc=..., forcing=..., neumann_flux=...)`，
   `bc` 是逐边 `{side: "dirichlet"|"neumann"}` 的部分字典（缺省 Dirichlet，向后兼容），
   `forcing` 是任意 $f(x,y)$，`neumann_flux` 是逐边 $g(x,y)$（`None` 表示齐次）。
   面装配顺序：体积项 → 内面（与旧路径相同）→ 逐边（Dirichlet 加块，Neumann 只加右端载荷）→ 源项。
3. **奇异系统处理**：四边全 Neumann 时检测零空间（断言 $\|A\mathbf{1}\|_\infty/\|A\|_\infty<10^{-10}$，
   实测 $10^{-16}$ 量级，说明装配确实把常数吃掉了），再用秩一规范
   $A+\frac{1}{n}11^T$ 求解，把剩下的可加常数钉住。若零空间比常数更大（解在某个方向上是
   常数），该规范会失效，模块**直接报错而不是静默给一个错代表元**——这是明确的适用范围边界。
4. **误差度量 `l2_error_2d(...)`**：对任意可调用精确解在 2p+4 阶 Gauss–Legendre 求积下算
   相对 $L_2$；`remove_constant=True` 先减掉逐点差的均值，这是纯 Neumann 解唯一正确的比较方式。
5. **判据**：$p=1\Rightarrow$ 观测阶 2、$p=2\Rightarrow$ 阶 3（SIPG 最优阶 $p+1$）；最细网格相对
   $L_2<3\,\%$；全 Dirichlet 路径与旧求解器 **逐位一致**；去掉 Neumann 载荷必须造成显著误差（灵敏度负控）。

## 3. 计算结果（数字由 `validation/poisson_neumann.json` 程序化注入）

| p | case | cells | DOFs | relative $L_2$ | max abs error | observed order | matrix sym. | nullspace res. |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `A_pure_neumann` | 8 | 256 | 1.519e-02 | 2.36e-02 | -- | 1.1e-14 | 8.9e-17 |
| 1 | `A_pure_neumann` | 16 | 1024 | 3.800e-03 | 6.05e-03 | 1.999 | 1.1e-14 | 8.9e-17 |
| 1 | `A_pure_neumann` | 32 | 4096 | 9.503e-04 | 1.52e-03 | 2.000 | 1.1e-14 | 8.9e-17 |
| 1 | `B_mixed_zero_flux` | 8 | 256 | 3.800e-03 | 6.05e-03 | -- | 1.1e-14 | 0.0e+00 |
| 1 | `B_mixed_zero_flux` | 16 | 1024 | 9.503e-04 | 1.52e-03 | 2.000 | 1.1e-14 | 0.0e+00 |
| 1 | `B_mixed_zero_flux` | 32 | 4096 | 2.376e-04 | 3.81e-04 | 2.000 | 1.1e-14 | 0.0e+00 |
| 1 | `C_mixed_inhomogeneous_flux` | 8 | 256 | 1.515e-02 | 2.35e-02 | -- | 1.1e-14 | 0.0e+00 |
| 1 | `C_mixed_inhomogeneous_flux` | 16 | 1024 | 3.798e-03 | 6.05e-03 | 1.996 | 1.1e-14 | 0.0e+00 |
| 1 | `C_mixed_inhomogeneous_flux` | 32 | 4096 | 9.501e-04 | 1.52e-03 | 1.999 | 1.1e-14 | 0.0e+00 |
| 2 | `A_pure_neumann` | 8 | 576 | 4.809e-04 | 4.92e-04 | -- | 1.1e-13 | 7.9e-17 |
| 2 | `A_pure_neumann` | 16 | 2304 | 6.030e-05 | 6.06e-05 | 2.996 | 1.1e-13 | 7.9e-17 |
| 2 | `A_pure_neumann` | 32 | 9216 | 7.544e-06 | 7.49e-06 | 2.999 | 1.1e-13 | 7.9e-17 |
| 2 | `B_mixed_zero_flux` | 8 | 576 | 6.015e-05 | 6.06e-05 | -- | 1.1e-13 | 0.0e+00 |
| 2 | `B_mixed_zero_flux` | 16 | 2304 | 7.534e-06 | 7.49e-06 | 2.997 | 1.1e-13 | 0.0e+00 |
| 2 | `B_mixed_zero_flux` | 32 | 9216 | 9.425e-07 | 9.30e-07 | 2.999 | 1.1e-13 | 0.0e+00 |
| 2 | `C_mixed_inhomogeneous_flux` | 8 | 576 | 4.794e-04 | 5.06e-04 | -- | 1.1e-13 | 0.0e+00 |
| 2 | `C_mixed_inhomogeneous_flux` | 16 | 2304 | 6.022e-05 | 6.14e-05 | 2.993 | 1.1e-13 | 0.0e+00 |
| 2 | `C_mixed_inhomogeneous_flux` | 32 | 9216 | 7.539e-06 | 7.57e-06 | 2.998 | 1.1e-13 | 0.0e+00 |

* 负控（同一问题 C，故意丢掉 Neumann 载荷）：n=8 时 54.6 %，
  n=16 时 55.4 % —— 不收敛、且比"接对了"的 0.38 %
  大 146 倍，说明本组测试对"载荷接错"是灵敏的，
  不是靠零通量的巧合过关。
* 全 Dirichlet 回归（新路径 vs 旧 `solve_poisson_2d` 的模态系数）：

| degree | cells | max abs difference | 旧路径相对 $L_2$ |
|---:|---:|---:|---:|
| p=1 | 8 | 0.0e+00 | 1.5186e-02 |
| p=1 | 16 | 0.0e+00 | 3.8003e-03 |
| p=2 | 8 | 0.0e+00 | 4.7862e-04 |
| p=2 | 16 | 0.0e+00 | 6.0154e-05 |

  差值恒为 0.0 —— 重构没有动到数值路径。

图：`validation/poisson_neumann_convergence.png`（四联图：p=1 与 p=2 的误差-网格曲线并叠 $h^2$/$h^3$
参考斜率与 3 % 门；A 的解场云图；C 在首列单元中 $x=h/2$ 处的 $u_h(y)$ 与精确曲线）。

## 4. 注意点（踩过的坑 / 局限 / 适用范围）

* **纯 Neumann 的"真值"不是唯一的**：不做 `remove_constant` 直接比 $L_2$，误差会被任意常数污染。
  仓库里已有的 Dirichlet 只需比场本身，这个差异是新路径特有的。
* **秩一规范的语义**：它钉住的是 $1^Tu=1^Tb$（模态系数和的约束），不是"逐点均值为零"。实测
  逐点均值与零均值精确解的偏差 ~1e-12，但要写清楚这是推论而不是定义；因此测试断言的是
  `constant_offset`（逐点差的均值）为 0，而不是 `coeff.sum()` 等于 0。
* **第一次装配错在边界单元的索引**：Dirichlet 边界的 `(ix, iy)` 我写反了一次，症状是
  "纯 Neumann 全部正常（2.00 / 3.00 阶）、全 Dirichlet 却差 52 %"。纯 Neumann 这条路径不碰
  Dirichlet 块，所以它的绿不能替代 Dirichlet 的回归——这就是为什么本文把"与旧求解器逐位一致"
  作为第一判据。
* **零空间更大时不给结果**：解若沿某方向只能是常数（例如全 Neumann 且解与 y 无关），
  一维零空间变成"任意 y 的函数"族，秩一规范不再唯一确定代表元。模块用 `nullspace_residual`
  与 `linear_residual` 暴露诊断，超出适用范围会抛错。
* **后端范围**：本次只做 dense 后端。`sipg_sparse` / `sipg_matrixfree` 仍是 Dirichlet-only，
  它们的装配是独立实现，接 Neumann 需要各自改，不在本单范围（已如实写在 README 口径之外）。
* 精度受限于稠密求解：$p=2,32^2$ 已 9216 DOF、矩阵 680 MB，再往上必须走 sparse/matrix-free。

## 5. 误差判定（逐项对照基准，显式回答"是否 < 3 %"）

| 关键量 | 基准 | 数值 | 相对误差 | < 3 % ? |
|---|---|---|---|---|
| A 纯 Neumann，p=1，n=16 | 解析 $\cos\cos$ | 3.800e-03 | 0.38 % | **是** |
| A 纯 Neumann，p=1，n=32 | 同上 | 9.503e-04 | 0.095 % | **是** |
| B 混合零通量，p=1，n=32 | 解析 $\cos(\pi x/2)\cos(\pi y/2)$ | 2.376e-04 | 0.024 % | **是** |
| C 非齐次 Neumann，p=1，n=32 | 解析 $\sin\sin$ | 9.501e-04 | 0.095 % | **是** |
| C 非齐次 Neumann，p=2，n=32 | 同上 | 7.539e-06 | 0.0008 % | **是** |
| 收敛阶 p=1（A/B/C） | 理论 2 | 1.999, 2.000 / 2.000, 2.000 / 1.996, 1.999 | 偏差 < 0.4 % | **是** |
| 收敛阶 p=2（A/B/C） | 理论 3 | 2.996, 2.999 / 2.997, 2.999 / 2.993, 2.998 | 偏差 < 0.7 % | **是** |
| 全 Dirichlet 与旧求解器 | 旧 `solve_poisson_2d` | max diff 0.0（p=1,2；n=8,16） | 0 % | **是（逐位一致）** |
| 负控：丢掉 Neumann 载荷 | 解析 $\sin\sin$，p=1，n=16 | 55.4 % | +55.4 % | **否（预期）** |

结论：**所有正例判据 < 3 % 全部通过**，观测收敛阶与 SIPG 理论最优阶 $p+1$ 一致（$p=1$: 2.00，
$p=2$: 3.00，偏差 < 0.7 %）；负控如预期失败，证明验证面是灵敏的。未覆盖面（sparse / matrix-free
的 Neumann、非 Cartesian 网格、Robin 边界、$g$ 为间断常数的情形）如实保留为缺口。
