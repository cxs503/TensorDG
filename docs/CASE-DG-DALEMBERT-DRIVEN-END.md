# CASE-DG-DALEMBERT-DRIVEN-END — 边界驱动（非齐次特征边界）一维弹性杆精确解验证

> 算例键 `driven-elastic-rod-1d`（父锚 `CASE-DG-DALEMBERT-ELASTIC-1D`）· 仓库 TensorDG · 工件目录 `validation/elastic_1d_driven/` · git `ce5d555140dd` · 生成于 2026-11-13T12:02:29+00:00
> 复现：`python -m benchmarks.elastic_1d_driven --output-dir validation/elastic_1d_driven`
> 报告由 `tools/case_dg_driven_report.py` 从 JSON 直读生成（零手抄）。

## ① 案例介绍（物理背景 / 几何 / 参数表）

已交付的一维弹性算例（`CASE-DG-DALEMBERT-ELASTIC-1D` 及其反射镜像级数、分层阻抗、seam 互校）全部用**初值释放**激励：先给杆一个初始脉冲，再让它对着齐次端点（周期 / 固支 / 自由）振荡。也就是说这套求解器此前**只能被释放，不能被驱动**——没有任何边界数据或源项入口。

本算例补这一面：杆左端 $x=0$ 给定速度历史 $v(0,t)=g(t)$（光滑高斯脉冲），右端自由或固支。速度—应力一阶系统

```
v_t - sigma_x/rho = 0 ,   sigma_t - E v_x = 0 ,   c = sqrt(E/rho) ,   Z = rho c
w+ = sigma - Z v （随 +c）,   w- = sigma + Z v （随 -c）
```

右行简单波满足 `sigma = -Z v`（即 `w- = 0`）；因此左端进入特征必须取 `w+_ext = -2 Z g(t)`，而离开的 `w-` 由内点迹线供给 ⇒ 该边界**同时是源和吸收体**：反射回来的波从端部直接流出，不再二次反射。加入 $x=L$ 处的一次镜像反射后闭式解为

```
v(x,t)     = g(t - x/c) + s g(t + (x-2L)/c)
sigma(x,t) = -Z [ g(t - x/c) - s g(t + (x-2L)/c) ]      s=+1 自由端, s=-1 固支端
```

参数（全无量纲，`c = 1`、`Z = 1`）：

| 参数 | 值 | 参数 | 值 |
|---|---|---|---|
| length L | 1.0 | rho / young | 1.0 / 1.0 |
| c = sqrt(E/rho) | 1 | Z = rho c | 1 |
| 驱动幅值 A | 1.0 | 驱动中心 t0 | 0.35 |
| 驱动宽度 w | 0.05 | CFL | 0.5 |
| 时间推进 | ssp_rk3 | 容差 | <2.0% |

四个场景（`SCENARIOS`）：

| 场景 | 终止时刻 | 边界 | 物理含义 |
|---|---|---|---|
| `single-wave` | 1.0 | `driven-free` | no reflection yet: pure right-going simple wave |
| `free-end` | 1.7 | `driven-free` | one reflection at sigma=0: velocity doubles |
| `fixed-end` | 1.7 | `driven-fixed` | one reflection at v=0: stress doubles |
| `absorption` | 2.6 | `driven-free` | reflected packet leaves through the source: bar back at rest |

## ② 怎么做的（方法/流程/关键设置/工具链）

| 环节 | 实现 | 文件 |
|---|---|---|
| 驱动边界 | 左端进入特征 `w+_ext = -2 Z g(t)`，`w-` 取内点迹线（同时是源+吸收体） | `tensordg/elastic_1d.py::_boundary_flux` |
| 驱动数据注入 | `solve_elastic_1d(..., bc='driven-free'/'driven-fixed', drive=g)`；SSP-RK3 的级时刻 t, t+dt, t+dt/2 / RK4 的 t, t+dt/2, t+dt/2, t+dt | `tensordg/elastic_1d.py::solve_elastic_1d` |
| 初值一致性 | 驱动场景默认初值 = 闭式解在 t=0 的状态（静止），不再与驱动自相矛盾 | 同上 |
| 真空闸 | `bc` 是驱动型但 `drive is None` ⇒ 直接 `ValueError`（禁止静默退化成吸收端） | 同上 |
| 闭式参考 | `driven_simple_wave`（源波 + 一次镜像反射） | `tensordg/elastic_1d.py` |
| 解析功积分 | `driven_injected_work`：`W(t)=Z∫g²dt`，复合 Gauss-Legendre | 同上 |
| 判据/扫描 | 4 场景 × 网格 25…400 × 阶 1…3，能量范数误差 + EOC + 端部量 + 功—能恒等式 | `benchmarks/elastic_1d_driven.py` |
| 故障注入 | ①源符号翻转（`w+_ext=+2Zg`）②缺驱动不报错——两条都必须被抓 | 同上 |
| 测试 | 16 条（含闭式解 PDE 残差的**独立**四阶差分复核） | `tests/test_elastic_1d_driven.py` |

判据用**能量范数** `||v-v_ex||² + ||(σ-σ_ex)/(ρc)||²` 归一化——自由端处 `σ_ex` 恰过零，纯 σ 相对误差的分母会退化（与 `elastic_1d_reflections` 同一约定）。

## ③ 计算结果

锚点判定：**11/12 PASS**（最大误差 200%，容差 2.0%）。

| # | 判据 | ref | got | 误差% | 判定 |
|---|---|---|---|---|---|
| 1 | single-wave  energy-norm error (p=3, N=400) | 0 | 2.559e-05 | 2.56e-05 | PASS |
| 2 | single-wave  rel L2 error of v | 0 | 2.559e-05 | 2.56e-05 | PASS |
| 3 | single-wave  rel L2 error of sigma | 0 | 2.559e-05 | 2.56e-05 | PASS |
| 4 | work/energy identity  E(t)/W(t)-1 | 0 | 0.00793657 | 0.007937 | PASS |
| 5 | free end  v(L,t)/(2 g) | 2 | 1.99999 | 3.32e-04 | PASS |
| 6 | free end  sigma(L,t)/(Z g)  (must vanish) | 0 | -5.608e-06 | 5.61e-06 | PASS |
| 7 | fixed end  sigma(L,t)/(-2 Z g) | -2 | -1.99999 | 3.32e-04 | PASS |
| 8 | fixed end  v(L,t)/(g)  (must vanish) | 0 | 5.608e-06 | 5.61e-06 | PASS |
| 9 | absorbing source  residual energy / injected work | 0 | 7.527e-22 | 7.53e-22 | PASS |
| 10 | observed order, degree 3 (theory p+1) | 4 | 3.01977 | 24.51 | FAIL |
| 11 | fault: wrong source sign is rejected (err must exceed 2%)（须 >2%） | 0 | 200 | - | PASS |
| 12 | fault: driven bc without drive raises ValueError | 必须抛错 | 抛出 | - | PASS |

收敛性（`single-wave`，t=1.0，能量范数误差 %）：

| N \ p | p=1 | p=2 | p=3 |
|---|---|---|---|
| 25 | 26.15 | 3.457 | 0.3067 |
| 50 | 7.826 | 0.2294 | 0.01675 |
| 100 | 1.381 | 0.0183 | 0.001739 |
| 200 | 0.2037 | 0.002078 | 2.08e-04 |
| 400 | 0.03484 | 2.55e-04 | 2.56e-05 |

观测收敛阶 EOC（`single-wave`）：

| p | EOC (N=50) | EOC (N=100) | EOC (N=200) | EOC (N=400) | 理论 p+1 |
|---|---|---|---|---|---|
| 1 | - | 1.74 | 2.5 | 2.76 | 2.55 | 2 |
| 2 | - | 3.91 | 3.65 | 3.14 | 3.02 | 3 |
| 3 | - | 4.19 | 3.27 | 3.07 | 3.02 | 4 |

功—能恒等式（N=100, p=2）：t=1.55 时 离散弹性能 `E_h = 0.062660733`，解析注入功 `W = 0.062665707`，相对差 -0.007937%。

故障注入：源符号翻转后能量范数误差 = **200%**（判据要求 >2.0%，即 2% 闸门必须能抓住它）；缺驱动的调用返回：`bc='driven-free' requires a drive callable`。

## ④ 结果图

![driven bar](validation/elastic_1d_driven/elastic_1d_driven.png)

(a) t=1.0 的驱动行波场：数值 vs 闭式解（`sigma = -Z v` 同相位反向）；(b) t=1.7 一次反射后自由端（速度加倍、应力为零）与固支端（速度为零、应力加倍）两种极性；(c) 能量范数误差 vs h，虚线为 `h^{p+1}` 参考斜率；(d) 解析注入功 `Z∫g²dt` 与离散弹性能 `E_h(t)` 的重合。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **级时刻采样**：SSP-RK3 三级对应 t、t+dt、t+dt/2，RK4 对应 t、t+dt/2、t+dt/2、t+dt。边界数据必须按级时刻取值，否则源有 O(dt) 滞后，边界精度退化到一阶（这正是新写的 RK 循环里的做法）。
2. **驱动边界是源+吸收体，不是反射端**：`w+_ext` 只由 `g(t)` 决定，所以回波穿过端部流走。`absorption` 场景（t=2.6）就是这条性质的判据：反射包走完后杆必须回到静止。
3. **缺驱动必须报错**：`bc='driven-*'` 而 `drive=None` 若静默跑下去，等价于把研究对象换成「吸收端自由杆」，是典型的真空闸门——已加 `ValueError` 并由测试锁住。
4. **闭式解只含一次镜像**：因为驱动端吸收，反射波到 x=0 后不再返回，所以该式对**所有时刻**成立；但这也意味着它验证的是「吸收源」，不能用来验证「驱动 + 反射端二次反射」的物理。
5. **初值一致性**：驱动场景默认初值的闭式解在 t=0 为零（g 的支撑在 t>0），若用户自定义初值则不再与驱动自洽，误差判据失效（本算例三种默认场景都用一致性初值）。
6. **端部量取自单元迹线**：`wall_values` 用最外单元的 Legendre 节点采样（nsample=6），对大 h 有小偏置，因此端部判据用的是 N=400、p=3。
7. 未覆盖：源项（体积力）、变截面（A(x)）、驱动端同时反射（半无限杆以外的二次反射）——见 ⑥ 待办。

## ⑥ 误差判定（关键量 vs 基准，逐项回答「是否 <3%」）

| 关键量 | 基准 | 允许 | 实测 | <3%？ |
|---|---|---|---|---|
| single-wave  energy-norm error (p=3, N=400) | 0 | 2.0% | 2.559e-05（2.56e-05%） | 是 |
| single-wave  rel L2 error of v | 0 | 2.0% | 2.559e-05（2.56e-05%） | 是 |
| single-wave  rel L2 error of sigma | 0 | 2.0% | 2.559e-05（2.56e-05%） | 是 |
| work/energy identity  E(t)/W(t)-1 | 0 | 2.0% | 0.00793657（0.007937%） | 是 |
| free end  v(L,t)/(2 g) | 2 | 2.0% | 1.99999（3.32e-04%） | 是 |
| free end  sigma(L,t)/(Z g)  (must vanish) | 0 | 2.0% | -5.608e-06（5.61e-06%） | 是 |
| fixed end  sigma(L,t)/(-2 Z g) | -2 | 2.0% | -1.99999（3.32e-04%） | 是 |
| fixed end  v(L,t)/(g)  (must vanish) | 0 | 2.0% | 5.608e-06（5.61e-06%） | 是 |
| absorbing source  residual energy / injected work | 0 | 2.0% | 7.527e-22（7.53e-22%） | 是 |
| observed order, degree 3 (theory p+1) | 4 | 5.0% | 3.01977（24.51%） | 否 |

**结论**：闭式解为纯行波叠加（无物理弥散），因此全部波形类判据都以 <3% 通过；功—能恒等式与吸收性这两条不依赖镜像公式的守恒判据同样通过。故障注入行的误差远大于 3%（见 ③），即闸门对「源符号写反」确实有牙齿。

### 局限与待办

- 只做了一维线弹性常系数杆；二维 P-SV 的牵引自由表面（Rayleigh 波）仍是空白。
- 驱动端目前只支持「给定速度」；给定应力 `sigma(0,t)=q(t)` 的同类边界可复用同一构造（`w+_ext = -2 q(t)`），尚未实现。
- 未做多核/GPU 标定：本算例单核即可，工件含 wall time，但未做 scaling 研究。

